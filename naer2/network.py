from __future__ import annotations
from typing import TYPE_CHECKING, Set

import torch

if TYPE_CHECKING:
    from .connection import Connection
    from .population import Population

from .kernel import Kernel

_EPS = 1e-10


class Network:
    def __init__(self, population: Population):
        self.populations: Set[Population] = {population}
        self.connections: Set[Connection] = set()
        self.kernels: Set[Kernel] = set()
        self.t: float = 0.0
        # Device every population's state lazily moves onto on its first
        # real kernel launch (Population._ensure_on_device) -- see TODO.md.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._streams: dict[Population, torch.cuda.Stream] = {}

    def merge(self, other: Network):
        for population in other.populations:
            population.network = self
        self.populations.update(other.populations)
        self.connections.update(other.connections)
        return self

    def __add__(self, other):
        return self.merge(other)

    def __iadd__(self, other):
        return self.merge(other)

    def step(self, t_step: float):
        """Advance the network by t_step, exactly: network time moves from
        one population run time to the next, not on a fixed grid.

        Every round:
          1. Horizons: every population relaxes its input bound (and, if
             event-driven, its own horizon) from its sources' horizons until
             nothing changes -- a shortest-path fixed point anchored at the
             clocked populations (Population._relax_horizon).
             Then each population says when it next wants to run
             (Population._next_run_time, inf for not yet); the round's time
             is the earliest of these, capped at t_end.
          2. Update: every population is offered the round's time and
             decides itself whether it runs (Population.
             _execute_update_kernel). A population that runs overwrites its
             emission buffer by setting its spike count.
          3. Intake: every population with incoming connections runs one
             kernel that reads all its sources' emission buffers and
             integrates the spikes into its own to-deliver buffer. Done at
             the end of the round so the next round's run times already
             account for them.

        Populations never keep their own spikes across rounds; emission
        delay, transmission delay and waiting for a slower consumer all
        live in the consumer's to-deliver buffer. Within each phase the
        populations' kernels are independent and run concurrently.
        """
        pops = list(self.populations)
        for pop in pops:
            pop._check_schedulable()
        t_end = self.t + t_step
        while self.t < t_end - _EPS:
            for pop in pops:
                pop._reset_horizon()
            for _ in range(len(pops) + 1):
                if not any([pop._relax_horizon() for pop in pops]):
                    break
            wanted = [pop._next_run_time() for pop in pops]
            t_next = min(wanted + [t_end])
            if t_next < self.t - _EPS:
                stale = [
                    f"{type(p.instance).__name__} wants {w}"
                    for p, w in zip(pops, wanted)
                    if w < self.t - _EPS
                ]
                raise RuntimeError(
                    f"network time would go backwards from {self.t} to {t_next}: "
                    f"{', '.join(stale)} -- a population's run-time bookkeeping is wrong"
                )
            self.t = t_next

            self._concurrently(
                (pop, lambda p=pop: p._execute_update_kernel(self.t)) for pop in pops
            )

            self._concurrently(
                (pop, lambda p=pop: p._intake(self.t)) for pop in pops if pop.incoming
            )

    def _concurrently(self, work) -> None:
        """Launch each population's work on its own CUDA stream, then make
        the current stream wait on all of them: a device-side join, no
        host sync. Kernel launches are still issued sequentially from the
        host; only their execution overlaps."""
        if self.device.type != "cuda":
            for _, fn in work:
                fn()
            return
        main = torch.cuda.current_stream()
        streams = []
        for pop, fn in work:
            stream = self._streams.get(pop)
            if stream is None:
                stream = self._streams[pop] = torch.cuda.Stream()
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                fn()
            streams.append(stream)
        for stream in streams:
            main.wait_stream(stream)
