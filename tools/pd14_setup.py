"""The PD14 spike-management benchmark setup shared by pd14_roundtrip.py and
trace_round.py: the 8 cortical populations as green-field Pops with their
TopologyNew connections (0.1ms delay buckets), firing at PD14-like rates,
one round = every update kernel, then every intake kernel."""

import math
import time

import torch

import test.test_update_emission_gpu as T
from naer2.topology_new import TopologyNew
from test.networks.PD14.net import build, description

RATE = dict(E_23=0.0, I_23=2.2, E_4=5.0, I_4=6.0, E_5=8.5, I_5=8.5, E_6=1.7, I_6=7.6)  # Hz
RES = 0.1  # ms: timestep, delay buckets and ring slots


class PD14:
    def __init__(self, G: int, backend="triton", unroll=None, stream=None, warps_update=4, warps_intake=4,
                 main_cap: int | None = None, entry_block: int | None = None, count_received: bool = True):
        """unroll / stream / entry_block: None = the backend's walk default (T.WALK)."""
        T.BACKEND = backend
        torch.cuda.set_device(0)
        torch.manual_seed(0)
        self.G, self.warps_update, self.warps_intake = G, warps_update, warps_intake
        names = dict(zip(description["populations"]["size"], description["populations"]["name"]))
        net = build()
        cortical = {p: names[len(p.shape)] for p in net.populations if len(p.shape) in names}
        self.pops = {name: T.Pop(len(p.shape), group_size=G) for p, name in cortical.items()}
        for p in self.pops.values():
            p.unroll = unroll or p.unroll
            p.stream = stream or p.stream
            p.entry_block = entry_block or p.entry_block
            p.count_received = count_received  # False: PD14's receive only (no stand-in counters)
        t0 = time.time()
        self.links = self.segments = 0
        for p, name in cortical.items():
            conns = [c for c in p.incoming if c._source in cortical]
            topos = [TopologyNew(c.fabric.topology, len(p.shape), G, "cuda", delay=c.fabric.delay, resolution=RES,
                                 attrs={"weight": c.fabric.weight}) for c in conns]
            self.links += sum(t.n_links for t in topos)
            self.segments += sum(t.n_segments for t in topos)
            n_slots = int((max(float(t.bucket_mean.max()) for t in topos) + 0.2) / RES) + 2
            self.pops[name].connect([self.pops[cortical[c._source]] for c in conns], topos, n_slots,
                                    main_cap=main_cap or 8 * G, side_cap=16)
        torch.cuda.synchronize()
        self.build_s = time.time() - t0
        del net
        self.zeros = {n: torch.zeros(p.n, device="cuda") for n, p in self.pops.items()}
        self.streams = {n: torch.cuda.Stream() for n in self.pops}

    def fire_mask(self, r: int) -> dict:
        """Which neurons fire in round r (deterministic per round)."""
        g = torch.Generator(device="cuda").manual_seed(1000 + r)
        return {n: torch.rand(p.n, device="cuda", generator=g) < RATE[n] * 1e-4 for n, p in self.pops.items()}

    def fire_at(self, mask: dict, r: int) -> dict:
        """Round r's stand-in fire times from a fire mask (fire times are
        absolute: a neuron fires iff its time lies in the run's window)."""
        t = 10.0 + RES * r
        return {n: torch.where(m, t, math.inf) for n, m in mask.items()}

    def fire(self, r: int) -> dict:
        """Round r's stand-in fire times (deterministic per round)."""
        g = torch.Generator(device="cuda").manual_seed(1000 + r)
        t = 10.0 + RES * r
        return {n: torch.where(torch.rand(p.n, device="cuda", generator=g) < RATE[n] * 1e-4,
                               torch.full((p.n,), t, device="cuda"), torch.full((p.n,), math.inf, device="cuda"))
                for n, p in self.pops.items()}

    def phase(self, fns: dict, concurrent: bool) -> None:
        if not concurrent:
            for fn in fns.values():
                fn()
            return
        main = torch.cuda.current_stream()
        for name, fn in fns.items():
            self.streams[name].wait_stream(main)
            with torch.cuda.stream(self.streams[name]):
                fn()
        for name in fns:
            main.wait_stream(self.streams[name])

    def update(self, r: int, fire: dict, concurrent=True) -> None:
        t = 10.0 + RES * r
        self.phase({n: (lambda p=p, n=n: p.run(t, t + RES, fire[n], self.zeros[n], uniform=True,
                                               num_warps=self.warps_update)) for n, p in self.pops.items()},
                   concurrent)

    def intake(self, concurrent=True, traces=None) -> None:
        self.phase({n: (lambda p=p, n=n: p.plan.run(True, num_warps=self.warps_intake,
                                                    trace=traces[n] if traces else None))
                    for n, p in self.pops.items()}, concurrent)

    def round(self, r: int, fire: dict, concurrent=True, intake_traces=None) -> None:
        self.update(r, fire, concurrent)
        self.intake(concurrent, intake_traces)

    def check(self) -> None:
        for p in self.pops.values():
            p.ring.check()

    def delivered(self) -> int:
        return sum(int(p.n_received.sum()) for p in self.pops.values())
