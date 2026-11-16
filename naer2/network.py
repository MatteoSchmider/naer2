from __future__ import annotations

import math
from typing import TYPE_CHECKING, Set

import torch

if TYPE_CHECKING:
    from .connection import Connection
    from .population import Population

_EPS = 1e-10
EVENT_DRIVEN_DEPTH = 8  # the history depth of a network with an event-driven consumer (see _plan_histories)


class Network:
    def __init__(self, population: Population):
        self.populations: Set[Population] = {population}
        self.connections: Set[Connection] = set()
        self.t: float = 0.0
        # Default delay quantization of connections without their own
        # resolution (Connection.resolution); a population's ring slot width
        # is the finest among its incoming connections' (TODO.md 1e, D-12).
        self.resolution: float = 0.1
        # The round counter: every population run in one round gets its id
        # (a run reads its sources' runs of earlier rounds only).
        self.round_id: int = 0
        self._planned = False
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
        if not self._planned:
            self._plan_histories(pops)
            self._planned = True
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
            self.round_id += 1  # every run of this round reads its sources' earlier rounds only

            self._concurrently(
                (pop, lambda stream, p=pop: p._execute_update_kernel(self.t, stream=stream)) for pop in pops
            )

            # host bookkeeping only, after the round's join (reads on the current stream see every run):
            # an event-driven population that ran notes its ring's next pending arrival; every
            # consumer notes its sources' new runs (the spikes themselves its next run pulls)
            for pop in pops:
                if pop._event_driven and pop.incoming and pop.last_update == self.t:
                    pop._note_pending_arrivals()
            for pop in pops:
                if pop.incoming:
                    pop._intake(self.t)

    def _plan_histories(self, pops) -> None:
        """The emission histories' depth K, once, before the first round: a
        source's history must keep every run its slowest consumer hasn't
        read yet -- the most runs it makes between two of that consumer's
        reads, + 2 (naer2/history.py). From the timesteps where both are
        known: 2 + ceil(consumer timestep / source timestep); else 3 (every
        population running every round) -- at least EVENT_DRIVEN_DEPTH with
        an event-driven consumer anywhere (it reads whenever an arrival is
        due: no cadence to derive from). One K for the whole network (a
        consumer's sources must share theirs); a population's own
        history_depth overrides, an overrun still raises (check()). TODO:
        grow a history when a consumer falls behind (test_network_step's
        gap test)."""
        depth = 3
        if any(pop._event_driven and pop.incoming for pop in pops):
            depth = EVENT_DRIVEN_DEPTH
        for pop in pops:
            for conn in pop.incoming:
                src = conn._source
                if pop.timestep and src.timestep:
                    depth = max(depth, 2 + math.ceil(pop.timestep / src.timestep - 1e-9))
        for pop in pops:
            pop._derived_depth = depth

    def _concurrently(self, work) -> None:
        """Launch each population's work on its own CUDA stream: every
        stream after what came before (one pre-made event recorded on the
        current stream), the current stream after all of them (one pre-made
        event per stream) -- a device-side join, no host sync. work: (pop,
        fn(stream)); fn(None) runs on the current stream (CPU: in order)."""
        if self.device.type != "cuda":
            for _, fn in work:
                fn(None)
            return
        main = torch.cuda.current_stream()
        if not hasattr(self, "_ready"):
            self._ready = torch.cuda.Event()
            self._done: dict = {}
        self._ready.record(main)
        launched = []
        for pop, fn in work:
            stream = self._streams.get(pop)
            if stream is None:
                stream = self._streams[pop] = torch.cuda.Stream()
                self._done[pop] = torch.cuda.Event()
            stream.wait_event(self._ready)
            fn(stream)
            self._done[pop].record(stream)
            launched.append(pop)
        for pop in launched:
            main.wait_event(self._done[pop])
