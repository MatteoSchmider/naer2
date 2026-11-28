"""
A network: its populations and connections, its time and round, and the four ways to advance it -- step(),
plan() (a dry run), run_planned() and run_graph() (planned and captured windows). What runs when is the
scheduler's (naer2/sim/scheduler.py), running it the executor's (naer2/sim/executor.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Set

import torch

from .sim.executor import Executor
from .sim.scheduler import Scheduler

if TYPE_CHECKING:
    from .connection import Connection
    from .population import Population



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
        # When each population runs, and how far (naer2/sim/scheduler.py)
        self.scheduler = Scheduler()
        self._planned = False
        # Device every population's state lazily moves onto on its first
        # real kernel launch (Population._ensure_on_device).
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Running what the scheduler decides: stepped rounds, planned and captured windows (naer2/sim/executor.py)
        self.executor = Executor(self)
        # Periodic windows (naer2/periodic.py): a clock-only network's captured windows extrapolated once
        # learned, not planned -- off by default; PD14 turns it on (its tune)
        self.periodic_windows = False

    @property
    def fuse_rounds(self) -> bool:
        """Fused rounds (naer2/fusion.py): a round in which every population runs as ONE kernel launch, in
        planned and captured windows -- on by default (PD14, Brunel: not slower; its cost: the most registers
        any population needs, longer compiles); off: a kernel per population, e.g. to look at one population's
        own kernel."""
        return self.executor.fuse

    @fuse_rounds.setter
    def fuse_rounds(self, on: bool) -> None:
        if bool(on) != self.executor.fuse:
            self.executor.fuse = bool(on)
            self.executor.graphs = {}  # (captured either way: captured anew)

    def merge(self, other: Network):
        for population in other.populations:
            population.network = self
        self.populations.update(other.populations)
        self.connections.update(other.connections)
        self.scheduler.merge(other.scheduler)
        return self

    def __add__(self, other):
        return self.merge(other)

    def __iadd__(self, other):
        return self.merge(other)

    def step(self, t_step: float):
        """Advance the network by t_step, exactly: network time moves from
        one population run time to the next, not on a fixed grid.

        Every round (naer2/sim/scheduler.py):
          1. Horizons: every population relaxes its input bound (and, if
             event-driven, its own horizon) from its sources' horizons until
             nothing changes -- a shortest-path fixed point anchored at the
             clocked populations. Then each population says when it next
             wants to run (inf for not yet); the round's time is the
             earliest of these, capped at t_end (Scheduler.next_time).
          2. Run: every population whose time it is takes its run
             (Scheduler.take; Executor.step) -- its kernel, with the round's
             id, on its own stream: it pulls what its sources emitted in earlier rounds
             (their emission histories) into its ring, delivers the due
             slots, updates, sends, and emits into its own history. The runs
             of a round are independent and run concurrently; the round
             joins before the next.
          3. Host bookkeeping, after the join: an event-driven population
             that ran notes its ring's next pending arrival
             (Scheduler.arrived), and every consumer notes its sources' new
             runs (Scheduler.intake: the emission summary, read only for an
             event-driven consumer), so the next round's run times already
             account for them.

        Emission delay, transmission delay and waiting for a slower
        consumer all live in the consumer's ring; a source's history keeps
        its runs until every consumer has pulled them (its depth:
        Scheduler.history_depth).
        """
        states = self._prepare()
        for self.t, runs in self.scheduler.rounds(states, self.t, until=self.t + t_step):
            self.round_id += 1  # every run of this round reads its sources' earlier rounds only
            self.executor.step(runs, self.round_id)

    def _prepare(self) -> list:
        """Before a step or plan: the scheduler's states for the populations (Scheduler.prepare); the histories'
        depth, once (Scheduler.history_depth)."""
        pops = list(self.populations)
        states = self.scheduler.prepare(pops)
        if not self._planned:
            depth = self.scheduler.history_depth(pops, self.resolution)
            for pop in pops:
                pop._derived_depth = depth
            self._planned = True
        return states

    def plan(self, max_launches: int = 1000, until: float | None = None, commit: bool = False) -> list[list[tuple]]:
        """A dry run of step(): the next launches without running any kernel -- each launch one round, the
        list of (population, t_start, t_end, gated) its runs; at most max_launches of them, or up to network
        time `until`. The network's and its populations' scheduling state are as before afterwards.

        Exact for clocked and update() populations: their run times and windows follow from timesteps and
        minimum delays alone. An event-driven population's runs depend on its arrivals -- data: it gets a
        GATED launch at every time an arrival could make a run due (Scheduler.possible_run_times: per
        source run, its start + the minimum delay, and the end of every slot its arrivals can land in), its
        window ending at its input bound as step() computes it; its kernel is to skip when nothing is due
        then (not implemented yet). Every run step() makes is among them, at the same time; the window ends
        of a population whose sources are event-driven too are conservative (their pending arrivals unknown,
        every possible one assumed). commit: the plan on the scheduling state itself, left where the plan
        ends (run_planned then makes the planned runs), else on a copy."""
        self._prepare()
        states = self.scheduler.states(copy=not commit)
        t, launches = self.t, []
        for t, runs in self.scheduler.rounds(states, t, until, planning=True, max_rounds=max_launches):
            launches.append(runs)
        if commit:
            self.t, self.round_id = t, self.round_id + len(launches)
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
        self._need_runtimes("run_planned")
        round0 = self.round_id
        return self.executor.planned(self.plan(max_launches, until, commit=True), round0)

    def run_graph(self, max_launches: int = 1000, until: float | None = None) -> int:
        """As run_planned, but the window replayed from a captured device graph (naer2/graph.py): captured
        once per window shape (which populations run in which round, gated or not), every window of that
        shape then a replay -- its entries written first, no launch from the host. One window may be in
        flight while the next is planned and filled: status slots for two windows' runs, the older one
        judged (waited for) when its slots are needed. A clock-only network's windows, with periodic_windows,
        extrapolated once learned (naer2/periodic.py). Returns the rounds made."""
        self._need_runtimes("run_graph")
        return self.executor.captured(max_launches, until, periodic=self.periodic_windows)

    def _need_runtimes(self, what: str) -> None:
        if any(pop._runtime is None for pop in self.populations):
            raise RuntimeError(f"{what}: every population needs its runtime -- step() a round first")
