"""The PD14 spike-management benchmark setup shared by pd14_roundtrip.py and
trace_round.py: the 8 cortical populations as green-field Pops with their
TopologyNew connections (0.1ms delay buckets), firing at PD14-like rates,
one round = every update kernel, then every intake kernel; fused=True: as
FusedPops (naer2/greenfield/fused.py), one round = one fused kernel per
population, intake included."""

import math
import time

import torch

from naer2.greenfield import Pop
from naer2.greenfield.fused import FusedPop
from naer2.topology_new import TopologyNew
from test.networks.PD14.net import build, description

RATE = dict(E_23=0.0, I_23=2.2, E_4=5.0, I_4=6.0, E_5=8.5, I_5=8.5, E_6=1.7, I_6=7.6)  # Hz
RES = 0.1  # ms: timestep, delay buckets and ring slots


class PD14:
    def __init__(self, G: int, unroll=None, stream=None, warps_update=4, warps_intake=4,
                 main_cap: int | None = None, entry_block: int | None = None, count_received: bool = True,
                 fused: bool = False):
        """unroll / stream / entry_block: None = the walk default (naer2.greenfield.WALK)."""
        self.fused = fused
        torch.cuda.set_device(0)
        torch.manual_seed(0)
        self.G, self.warps_update, self.warps_intake = G, warps_update, warps_intake
        names = dict(zip(description["populations"]["size"], description["populations"]["name"]))
        net = build()
        cortical = {p: names[len(p.shape)] for p in net.populations if len(p.shape) in names}
        make = FusedPop if fused else Pop
        self.pops = {name: make(len(p.shape), group_size=G) for p, name in cortical.items()}
        for p in self.pops.values():
            p.unroll = unroll or p.unroll
            p.stream = stream or p.stream
            if hasattr(p, "entry_block"):  # split kernels only
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
        # a round's dependencies through a hub (the main stream): every population after the previous
        # phase, the next phase after all of them; events made once (torch's wait_stream makes one per
        # call and costs ~5 us against ~0.6 us for a record or wait, tools/experiments/stream_sync_cost.py)
        self.done = {n: torch.cuda.Event() for n in self.pops}
        self.ready = torch.cuda.Event()

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
        """fns: per population, fn(stream) launching its kernel on stream
        (None: the current one)."""
        if not concurrent:
            for fn in fns.values():
                fn(None)
            return
        main = torch.cuda.current_stream()
        self.ready.record(main)
        for name, fn in fns.items():
            s = self.streams[name]
            s.wait_event(self.ready)
            fn(s)
            self.done[name].record(s)
        for name in fns:
            main.wait_event(self.done[name])

    @staticmethod
    def _on(stream, fn):
        if stream is None:
            return fn()
        with torch.cuda.stream(stream):
            return fn()

    def update(self, r: int, fire: dict, concurrent=True) -> None:
        t = 10.0 + RES * r
        if self.fused:
            fns = {n: (lambda s, p=p, n=n: p.run(t, t + RES, fire[n], self.zeros[n], uniform=True,
                                                 num_warps=self.warps_update, round_id=r, stream=s))
                   for n, p in self.pops.items()}
        else:
            fns = {n: (lambda s, p=p, n=n: self._on(s, lambda: p.run(t, t + RES, fire[n], self.zeros[n],
                                                                    uniform=True, num_warps=self.warps_update)))
                   for n, p in self.pops.items()}
        self.phase(fns, concurrent)

    def intake(self, concurrent=True, traces=None) -> None:
        if self.fused:  # inside the fused kernel
            return
        self.phase({n: (lambda s, p=p, n=n: self._on(s, lambda: p.plan.run(
                        True, num_warps=self.warps_intake, trace=traces[n] if traces else None)))
                    for n, p in self.pops.items()}, concurrent)

    def round(self, r: int, fire: dict, concurrent=True, intake_traces=None) -> None:
        self.update(r, fire, concurrent)
        self.intake(concurrent, intake_traces)

    def check(self) -> None:
        for p in self.pops.values():
            p.check() if self.fused else p.ring.check()

    def delivered(self) -> int:
        return sum(int(p.n_received.sum()) for p in self.pops.values())
