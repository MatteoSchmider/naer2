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
        # is the finest among its incoming connections' (TODO.md, D-12).
        self.resolution: float = 0.1
        # The round counter: every population run in one round gets its id
        # (a run reads its sources' runs of earlier rounds only).
        self.round_id: int = 0
        self._planned = False
        # Device every population's state lazily moves onto on its first
        # real kernel launch (Population._ensure_on_device).
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._streams: dict[Population, torch.cuda.Stream] = {}
        # Fused rounds (naer2/fusion.py): a round in which every population runs as ONE kernel launch, in
        # planned and captured windows -- off by default (one kernel: the most registers any population
        # needs, longer compiles); PD14 turns it on: net = pop.network; net.fuse_rounds = True
        self._fuse_rounds = False
        self._fused = None

    @property
    def fuse_rounds(self) -> bool:
        return self._fuse_rounds

    @fuse_rounds.setter
    def fuse_rounds(self, on: bool) -> None:
        if bool(on) != self._fuse_rounds:
            self._fuse_rounds = bool(on)
            self._graphs = {}  # (captured either way: captured anew)

    def _fused_round(self):
        """The fused kernel of every population (fuse_rounds; None if off, or a population without its
        runtime): built once, again when a runtime or its kernel is replaced."""
        if not self._fuse_rounds or self.device.type != "cuda":
            return None
        runtimes = [pop._runtime for pop in sorted(self.populations, key=id)]  # (a fixed order: the key)
        if any(rt is None for rt in runtimes):
            return None
        key = tuple((id(rt), id(rt.kernel)) for rt in runtimes)
        if self._fused is None or self._fused.key != key:
            from .fusion import FusedRound
            self._fused = FusedRound(runtimes)
        return self._fused

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
          2. Run: every population is offered the round's time and decides
             itself whether it runs (Population._execute_update_kernel) --
             one fused kernel, with the round's id, on its own stream: it
             pulls what its sources emitted in earlier rounds (their
             emission histories) into its ring, delivers the due slots,
             updates, sends, and emits into its own history. The runs of a
             round are independent and run concurrently; the round joins
             before the next.
          3. Host bookkeeping, after the join: an event-driven population
             that ran notes its ring's next pending arrival, and every
             consumer notes its sources' new runs (Population._intake: the
             emission summary, read only for an event-driven consumer), so
             the next round's run times already account for them.

        Emission delay, transmission delay and waiting for a slower
        consumer all live in the consumer's ring; a source's history keeps
        its runs until every consumer has pulled them (its depth:
        _plan_histories).
        """
        pops = list(self.populations)
        for pop in pops:
            pop._check_schedulable()
            pop._prepare_step()
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
                if pop._is_event_driven and pop._ins and pop.last_update == self.t:
                    pop._note_pending_arrivals()
            for pop in pops:
                if pop._ins:
                    pop._intake(self.t)

    def plan(self, max_launches: int = 1000, until: float | None = None, commit: bool = False) -> list[list[tuple]]:
        """A dry run of step(): the next launches without running any kernel -- each launch one round, the
        list of (population, t_start, t_end, gated) its runs; at most max_launches of them, or up to network
        time `until`. The network's and its populations' scheduling state are as before afterwards.

        Exact for clocked and update() populations: their run times and windows follow from timesteps and
        minimum delays alone. An event-driven population's runs depend on its arrivals -- data: it gets a
        GATED launch at every time an arrival could make a run due (Population._possible_run_times: per
        source run, its start + the minimum delay, and the end of every slot its arrivals can land in), its
        window ending at its input bound as step() computes it; its kernel is to skip when nothing is due
        then (not implemented yet). Every run step() makes is among them, at the same time; the window ends
        of a population whose sources are event-driven too are conservative (their pending arrivals unknown,
        every possible one assumed). commit: keep the scheduling state where the plan ends (run_planned
        then makes the planned runs), else restore it."""
        pops = list(self.populations)
        for pop in pops:
            pop._check_schedulable()
            pop._prepare_step()
        if not self._planned:
            self._plan_histories(pops)
            self._planned = True
        saved = [(pop, pop._schedule_state()) for pop in pops]
        t0, round0 = self.t, self.round_id
        launches = []
        try:
            while len(launches) < max_launches:
                for pop in pops:
                    pop._reset_horizon()
                for _ in range(len(pops) + 1):
                    if not any([pop._relax_horizon() for pop in pops]):
                        break
                t_next = min([pop._next_run_time() for pop in pops] + ([until] if until is not None else []))
                if t_next == math.inf or (until is not None and t_next >= until - _EPS):
                    break
                self.t = t_next
                self.round_id += 1
                launch = []
                for pop in pops:
                    window = pop._execute_update_kernel(self.t, launch=False)
                    if window is not None:
                        launch.append((pop, window[0], window[1], pop._is_event_driven))
                launches.append(launch)
                for pop in pops:
                    if pop._ins:
                        pop._intake(self.t, planning=True)
        finally:
            if not commit:
                for pop, state in saved:
                    pop._restore_schedule(state)
                self.t, self.round_id = t0, round0
        return launches

    def run_planned(self, max_launches: int = 1000, until: float | None = None) -> int:
        """Up to max_launches rounds (or to network time `until`) as ONE planned window: plan them
        (plan(commit=True)), their runs' entries as a Schedule (naer2/schedule.py: in device memory, one
        upload), then every round's runs launched straight from it -- no per-round host scheduling, and every
        launch the same but its entry's index. The window's statuses are judged before the next window's runs
        reuse their slots. An event-driven population's runs are gated (plan): its kernel decides whether it
        runs (fused.py: _gate_skip, from its sources' headers and its event state), a skipped run an empty
        one; its ring's delivery position comes back from the device after the window. Its host-side
        scheduling then holds the plan's view (every possible arrival assumed): further windows plan from
        it, step() would schedule it more often than needed. Every population needs its runtime: step() a
        round first. Returns the rounds made."""
        from .schedule import Schedule

        pops = list(self.populations)
        if any(pop._runtime is None for pop in pops):
            raise RuntimeError("run_planned: every population needs its runtime -- step() a round first")
        round0 = self.round_id
        launches = self.plan(max_launches, until, commit=True)
        saved_round = self.round_id
        self.round_id = round0
        schedule = Schedule(self, launches, self.device)
        self.round_id = saved_round
        for pop in pops:  # two windows' runs of status slots: one window in flight while the next launches
            rt = pop._runtime
            need = 2 * schedule.n_runs.get(id(rt), 0)
            if rt.status_slots < need:
                rt._alloc_slots(1 << math.ceil(math.log2(need)))
        fused = self._fused_round()
        for runs in schedule.launches:
            if fused is not None and fused.covers([pop._runtime for pop, _, _ in runs]):
                self._launch_fused(fused, schedule, runs)
                continue
            self._concurrently(
                (pop, lambda stream, pop=pop, i=i, run_id=run_id: pop._runtime.launch_entry(
                    schedule.entries(pop._runtime), i, run_id, stream))
                for pop, i, run_id in runs)
        schedule.commit()
        for pop in pops:
            if pop._is_event_driven and pop._ins:
                pop._runtime.sync_ring_pos()
        self._schedule = schedule  # (its entries stay alive until the next window's)
        return len(launches)

    @staticmethod
    def _launch_fused(fused, schedule, runs) -> None:
        """A planned round's runs as one fused launch (naer2/fusion.py), each run's bookkeeping its runtime's
        (launch_entry's): its status slot free first, then pending, journaled for a rerun."""
        by_rt = {id(pop._runtime): (pop._runtime, i, run_id) for pop, i, run_id in runs}
        params = []
        for rt in fused.runtimes:
            _, i, run_id = by_rt[id(rt)]
            rt.free_slot()
            params.append(rt.entry_params(schedule.entries(rt), i, run_id))
        fused.launch(params, torch.cuda.current_stream())
        for rt, p in zip(fused.runtimes, params):
            rt._launches.journal[(id(rt), p[2])] = (rt, p)
            rt._pending.append(p[2])

    def run_graph(self, max_launches: int = 1000, until: float | None = None) -> int:
        """As run_planned, but the window replayed from a captured device graph (naer2/graph.py): captured
        once per window shape (which populations run in which round, gated or not), every window of that
        shape then a replay -- its entries written first, no launch from the host. One window may be in
        flight while the next is planned and filled: status slots for two windows' runs, the older one
        judged (waited for) when its slots are needed. Returns the rounds made."""
        from .graph import WindowGraph
        from .schedule import Schedule

        pops = list(self.populations)
        if any(pop._runtime is None for pop in pops):
            raise RuntimeError("run_graph: every population needs its runtime -- step() a round first")
        round0 = self.round_id
        launches = self.plan(max_launches, until, commit=True)
        if not launches:
            return 0
        saved_round, self.round_id = self.round_id, round0
        schedule = Schedule(self, launches, self.device, upload=False)
        self.round_id = saved_round
        for pop in pops:
            rt = pop._runtime
            n = schedule.n_runs.get(id(rt), 0)
            if rt.status_slots < 2 * n:
                rt._alloc_slots(1 << math.ceil(math.log2(2 * n)))
            rt.make_room(n)
        if not hasattr(self, "_graphs"):
            self._graphs = {}
        sig = schedule.signature()
        graph = self._graphs.get(sig)
        if graph is None or not graph.valid(schedule):
            graph = self._graphs[sig] = WindowGraph(self, schedule, barrier=getattr(self, "graph_barrier", False),
                                                    fused=self._fused_round())
            self._captures = getattr(self, "_captures", 0) + 1
        graph.replay(schedule)
        schedule.commit()
        if any(pop._is_event_driven and pop._ins for pop in pops):
            torch.cuda.synchronize()  # (the gated runs' delivery positions: only their kernels know)
            for pop in pops:
                if pop._is_event_driven and pop._ins:
                    pop._runtime.sync_ring_pos()
        return len(launches)

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
