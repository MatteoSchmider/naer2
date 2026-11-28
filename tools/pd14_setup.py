"""The PD14 spike-management benchmark setup shared by the tools: PD14's
populations as PopulationRuntimes (naer2/sim/runtime.py) running kernels
generated from their hooks (naer2/transpiler/templates/fused_template.py),
with their SimulationTopology connections (0.1ms delay buckets); one round =
one fused kernel per population, every population on its own stream.

model="standin" (the benchmarks): the 8 cortical populations as
PD14_commanded (PD14's update and receive, firing on command at PD14-like
rates: fire_mask / fire_at), their connections among each other.
model="pd14" (the matching target): the real network -- the 8 cortical
populations as PD14_neuron (each population's own constants, V_rest from
its K_cortical, initial potentials as build() draws them), the thalamic
population T as PD14_poisson, and every connection, T's included; nothing
is commanded. t0: the time of round 0 (T fires in [700, 710] ms). Each
population's kernel is its own (its static attrs are literals in it): the
first run compiles one per population."""

import time

import torch

from naer2.ring import RESERVE
from naer2.sim.runtime import PopulationRuntime
from naer2.transpiler.templates.fused_template import FusedKernelTemplate
from test.networks.PD14.commanded import PD14_commanded
from test.networks.PD14.net import build, description

# Hz: PD14's settled rates through Network.step (4000-5000 ms, the L4e -> L2/3e projection doubled, 2026-11-21)
RATE = dict(E_23=0.81, I_23=2.74, E_4=4.21, I_4=5.60, E_5=7.18, I_5=8.30, E_6=1.07, I_6=7.50)
RES = 0.1  # ms: timestep, delay buckets and ring slots
# model "pd14": ring run capacity per population (links per (group, slot) cell, G=256; the arenas sized for it in
# every slot at once: no resize while measured) -- the peak demand
# through the thalamic volley at 700 ms (9.0k-31.8k, 32-42 rounds after its onset; measured 2026-11-22 on the 4090,
# the L4e -> L2/3e projection doubled: L2/3 joins the volley now) + 25%, in multiples of 256 -- several GB of
# rings (fits a 24 GB card, not the 780M)
PD14_MAIN_CAP = dict(E_23=34048, I_23=39936, E_4=11520, I_4=24320, E_5=33024, I_5=18944, E_6=14592, I_6=13568)


class PD14:
    def __init__(self, G: int, unroll=None, stream=None, warps=4, run_cap: int | dict | None = None,
                 count_received: bool = True, model: str = "standin", t0: float = 10.0):
        """unroll / stream: None = the walk default (naer2.transpiler.templates.fused.WALK);
        run_cap: ring run capacity, one for all or per population (default:
        8 * G for the stand-in, PD14_MAIN_CAP for "pd14") -- every group's arena that many records a slot."""
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
        K = dict(zip(description["populations"]["name"], description["populations"]["K_cortical"]))
        self.pops = {}
        for p, name in members.items():
            if model == "standin":
                pop = PD14_commanded(K_cortical=K[name]) * len(p.shape)
                state = {a: v for a, v in pop.instance.__dict__.items() if a in pop.dynamic_attrs}
            else:  # the real populations, with their connections
                pop = p
                state = {a: v for a, v in p.instance.__dict__.items() if a in p.dynamic_attrs}
                if name != "T":
                    state["potential"] = p.potential.to(dtype=torch.float32)
            self.pops[name] = PopulationRuntime(FusedKernelTemplate(pop).build(), len(p.shape), group_size=G,
                                                state=state)
        for p in self.pops.values():
            p.unroll = unroll or p.unroll
            p.stream_block = stream or p.stream_block
            p.count_received = count_received  # False: PD14's receive only (no stand-in counters)
        t0 = time.time()
        self.links = self.segments = 0
        for p, name in members.items():
            conns = [c for c in p.incoming if c._source in members]
            if not conns:  # T: no incoming connections
                continue
            topos = [c.fabric.simulation(len(p.shape), G, "cuda", resolution=RES, attrs=["weight"]) for c in conns]
            self.links += sum(t.n_links for t in topos)
            self.segments += sum(t.n_segments for t in topos)
            n_slots = int((max(float(t.bucket_mean.max()) for t in topos) + 0.2) / RES) + 2
            cap = run_cap.get(name) if isinstance(run_cap, dict) else run_cap
            cap = cap or (PD14_MAIN_CAP[name] if model == "pd14" else 8 * G)
            self.pops[name].connect([self.pops[members[c._source]] for c in conns], topos, n_slots,
                                    arena=(256, n_slots * (-(-cap // 256) + RESERVE) + 1))
        torch.cuda.synchronize()
        self.build_s = time.time() - t0
        del net
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
        """Round r's commands from a fire mask: a fresh fire_now tensor per
        population (the send consumes it, so each is used once)."""
        return {n: m.float() for n, m in mask.items()}

    def fire(self, r: int) -> dict:
        """Round r's commands (deterministic per round)."""
        return self.fire_at(self.fire_mask(r), r)

    def round(self, r: int, fire: dict | None = None, concurrent=True) -> None:
        """Round r: every population's run, each on its own stream after the
        previous round (concurrent), or one after another on the current one.
        fire: the stand-in's commands (model "standin", see fire_at); None for "pd14"."""
        t = self.t0 + RES * r
        for n, f in (fire or {}).items():
            self.pops[n].state["fire_now"] = f
        if not concurrent:
            for p in self.pops.values():
                p.run(t, t + RES, num_warps=self.warps, round_id=r)
            return
        main = torch.cuda.current_stream()
        self.ready.record(main)
        for n, p in self.pops.items():
            s = self.streams[n]
            s.wait_event(self.ready)
            p.run(t, t + RES, num_warps=self.warps, round_id=r, stream=s)
            self.done[n].record(s)
        for n in self.pops:
            main.wait_event(self.done[n])

    def check(self) -> None:
        for p in self.pops.values():
            p.check()

    def delivered(self) -> int:
        return sum(int(p.n_received.sum()) for p in self.pops.values())
