"""The PD14 spike-management benchmark setup shared by the tools: PD14's
populations as FusedPops (naer2/greenfield/fused.py) with their
SimulationTopology connections (0.1ms delay buckets); one round = one fused
kernel per population, every population on its own stream.

model="standin" (the benchmarks): the 8 cortical populations, firing on
command at PD14-like rates (fire_mask / fire_at), their connections among
each other. model="pd14" (the matching target): the real network -- the 8
cortical populations as PD14_neuron (each population's own constants,
V_rest from its K_cortical, initial potentials as build() draws them), the
thalamic population T as PD14_poisson, and every connection, T's included;
nothing is commanded. t0: the time of round 0 (T fires in [700, 710] ms)."""

import math
import time

import torch

from naer2.greenfield import FusedPop
from naer2.greenfield.fused import pd14_neuron_params, pd14_poisson_params
from naer2.topology import SimulationTopology
from test.networks.PD14.net import build, description

RATE = dict(E_23=0.0, I_23=2.2, E_4=5.0, I_4=6.0, E_5=8.5, I_5=8.5, E_6=1.7, I_6=7.6)  # Hz
RES = 0.1  # ms: timestep, delay buckets and ring slots
# model "pd14": ring run capacity per population (links per (group, slot) cell, G=256) -- the peak demand
# through the thalamic volley at 700 ms (3.4k-9.3k, 30-45 rounds after it; quiet activity peaks at 160-350)
# + 25%, in multiples of 256: ~2.2 GB of rings (2048 for all overflows ~7 rounds into the volley)
PD14_MAIN_CAP = dict(E_23=7168, I_23=7168, E_4=8192, I_4=11776, E_5=9216, I_5=5760, E_6=7168, I_6=4352)


class PD14:
    def __init__(self, G: int, unroll=None, stream=None, warps=4, main_cap: int | dict | None = None,
                 count_received: bool = True, model: str = "standin", t0: float = 10.0):
        """unroll / stream: None = the walk default (naer2.greenfield.WALK);
        main_cap: ring run capacity, one for all or per population (default:
        8 * G for the stand-in, PD14_MAIN_CAP for "pd14")."""
        if model not in ("standin", "pd14"):
            raise ValueError("model must be 'standin' or 'pd14'")
        torch.cuda.set_device(0)
        torch.manual_seed(0)
        self.G, self.warps, self.model, self.t0 = G, warps, model, t0
        names = dict(zip(description["populations"]["size"], description["populations"]["name"]))
        if model == "pd14":
            names[description["T"]["size"]] = "T"
        net = build()
        members = {p: names[len(p.shape)] for p in net.populations if len(p.shape) in names}
        if model == "standin":
            self.pops = {name: FusedPop(len(p.shape), group_size=G) for p, name in members.items()}
        else:
            self.pops = {}
            for p, name in members.items():
                if name == "T":
                    self.pops[name] = FusedPop(len(p.shape), group_size=G, model="pd14_poisson",
                                               params=pd14_poisson_params(p.instance))
                else:
                    self.pops[name] = FusedPop(len(p.shape), group_size=G, model="pd14_neuron",
                                               params=pd14_neuron_params(p.instance))
                    self.pops[name].potential = p.potential.to(device="cuda", dtype=torch.float32).clone()
        for p in self.pops.values():
            p.unroll = unroll or p.unroll
            p.stream = stream or p.stream
            p.count_received = count_received  # False: PD14's receive only (no stand-in counters)
        t0 = time.time()
        self.links = self.segments = 0
        for p, name in members.items():
            conns = [c for c in p.incoming if c._source in members]
            if not conns:  # T: no incoming connections
                continue
            topos = [SimulationTopology(c.fabric.topology, len(p.shape), G, "cuda", delay=c.fabric.delay,
                                        resolution=RES, attrs={"weight": c.fabric.weight}) for c in conns]
            self.links += sum(t.n_links for t in topos)
            self.segments += sum(t.n_segments for t in topos)
            n_slots = int((max(float(t.bucket_mean.max()) for t in topos) + 0.2) / RES) + 2
            cap = main_cap.get(name) if isinstance(main_cap, dict) else main_cap
            cap = cap or (PD14_MAIN_CAP[name] if model == "pd14" else 8 * G)
            self.pops[name].connect([self.pops[members[c._source]] for c in conns], topos, n_slots,
                                    main_cap=cap, side_cap=16)
        torch.cuda.synchronize()
        self.build_s = time.time() - t0
        del net
        self.zeros = {n: torch.zeros(p.n, device="cuda") for n, p in self.pops.items()}
        self.streams = {n: torch.cuda.Stream() for n in self.pops}
        # a round's dependencies through a hub (the main stream): every population after the previous
        # round, the next round after all of them; events made once (torch's wait_stream makes one per
        # call and costs ~5 us against ~0.6 us for a record or wait, tools/experiments/stream_sync_cost.py)
        self.done = {n: torch.cuda.Event() for n in self.pops}
        self.ready = torch.cuda.Event()

    def fire_mask(self, r: int) -> dict:
        """Which neurons fire in round r (deterministic per round)."""
        g = torch.Generator(device="cuda").manual_seed(1000 + r)
        return {n: torch.rand(p.n, device="cuda", generator=g) < RATE[n] * 1e-4 for n, p in self.pops.items()
                if n in RATE}

    def fire_at(self, mask: dict, r: int) -> dict:
        """Round r's stand-in fire times from a fire mask (fire times are
        absolute: a neuron fires iff its time lies in the run's window)."""
        t = self.t0 + RES * r
        return {n: torch.where(m, t, math.inf) for n, m in mask.items()}

    def fire(self, r: int) -> dict:
        """Round r's stand-in fire times (deterministic per round)."""
        g = torch.Generator(device="cuda").manual_seed(1000 + r)
        t = self.t0 + RES * r
        return {n: torch.where(torch.rand(p.n, device="cuda", generator=g) < RATE[n] * 1e-4,
                               torch.full((p.n,), t, device="cuda"), torch.full((p.n,), math.inf, device="cuda"))
                for n, p in self.pops.items() if n in RATE}

    def round(self, r: int, fire: dict | None = None, concurrent=True) -> None:
        """Round r: every population's run, each on its own stream after the
        previous round (concurrent), or one after another on the current one.
        fire: the stand-in's fire times (model "standin"); None for "pd14"."""
        t = self.t0 + RES * r
        fire = fire or {}
        if not concurrent:
            for n, p in self.pops.items():
                p.run(t, t + RES, fire.get(n), self.zeros[n], uniform=True, num_warps=self.warps, round_id=r)
            return
        main = torch.cuda.current_stream()
        self.ready.record(main)
        for n, p in self.pops.items():
            s = self.streams[n]
            s.wait_event(self.ready)
            p.run(t, t + RES, fire.get(n), self.zeros[n], uniform=True, num_warps=self.warps, round_id=r,
                  stream=s)
            self.done[n].record(s)
        for n in self.pops:
            main.wait_event(self.done[n])

    def check(self) -> None:
        for p in self.pops.values():
            p.check()

    def delivered(self) -> int:
        return sum(int(p.n_received.sum()) for p in self.pops.values())
