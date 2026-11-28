"""
The network's scheduling: when each population runs, and how far -- every round's horizons relaxed to a fixed
point, the round's time the earliest any population wants, each population that wants it taking its run, then
the sources' new runs taken in (Network.step's docstring). One round function for both its uses: step() runs
it on the live state and launches the runs; plan() runs it on a copy (or the live state, committing), with
every possible arrival of an event-driven population assumed instead of read from the device.

  PopSchedule      one population's scheduling: what is fixed for a step (its incoming connections, clock,
                   kind) and what the rounds change (its last run, horizon, input bound, pending run times, ...)
  Scheduler        the network's: prepare() at a step's start, then rounds() -- per round next_time(), take(),
                   and, after the round, arrived() and intake() -- the device's feedback passed in (a
                   population's emission summary, its ring's next pending arrival: Population._emission_summary,
                   _pending_arrival -- host syncs, TODO.md D-13); states() a copy for a dry run; history_depth()
                   the emission histories' depth from the populations' cadences

The model's delays and clocks it schedules from are the populations' (Population._own_timestep, _min_delay,
_max_delay, _slot_width).
"""

from __future__ import annotations

import dataclasses
import math

_EPS = 1e-10
EVENT_DRIVEN_DEPTH = 8  # the history depth of a network with an event-driven consumer (Scheduler.history_depth)


def same_time(a: float, b: float) -> bool:
    """Equality of two run times up to float rounding: 0.2 * 3 and 0.3 + 0.3
    must count as the same round. inf only equals inf."""
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


@dataclasses.dataclass(eq=False)
class PopSchedule:
    """One population's scheduling (see Scheduler).

    horizon: no spike of this population before it is still unknown (clocked / update() populations: where
      their last run computed to; event-driven ones: derived every round from their inputs).
    bound: no input arrival before it is still unknown.
    run_times: pending run times of an event-driven population -- intake appends a source run's earliest
      emission + the connection's minimum delay; later per-link arrivals come from its ring after each run
      (arrived).
    floors: a run time's earliest spike where it is earlier (a slot-end run: its slot's records arrive, and
      fire, anywhere in the slot -- D-1's exact times; the horizon must be no later); an exact run time is its
      own.
    read_run: per incoming connection, the source run intake last read (a source read once per run, even when
      rounds repeat at the same time).
    """

    pop: object
    # fixed for a step (Scheduler.prepare): (connection, source index, minimum delay) per incoming connection
    ins: list = dataclasses.field(default_factory=list)
    clock: float | None = None
    event_driven: bool = False
    autonomous: bool = False
    # what the rounds change
    last_update: float = -math.inf
    run_id: int = 0
    window: tuple | None = None
    horizon: float = 0.0
    bound: float = math.inf
    run_times: list = dataclasses.field(default_factory=list)
    floors: dict = dataclasses.field(default_factory=dict)
    read_run: dict = dataclasses.field(default_factory=dict)

    def copy(self) -> PopSchedule:
        return dataclasses.replace(self, run_times=list(self.run_times), floors=dict(self.floors),
                                   read_run=dict(self.read_run))

    def state(self) -> tuple:
        """What the rounds change, as plain values (naer2/periodic.py compares and shifts them)."""
        return (self.last_update, self.run_id, self.window, self.horizon, self.bound, list(self.run_times),
                dict(self.read_run), dict(self.floors))

    def set_state(self, state: tuple) -> None:
        (self.last_update, self.run_id, self.window, self.horizon, self.bound, run_times, read_run,
         floors) = state
        self.run_times, self.read_run, self.floors = list(run_times), dict(read_run), dict(floors)

    def next_run_time(self, t_now: float) -> float:
        """When this population next wants to run, given this round's relaxed horizons; math.inf for "not
        until something changes". Never earlier than the network's current time t_now.

          - never run yet: 0.0, for every population
          - a clock (explicit timestep, or a source's derived one): its horizon (last run + timestep)
          - update() dynamics, no timestep: as soon as its input bound is past its horizon, i.e. it can
            integrate further
          - event-driven: its earliest pending arrival, once the input bound has reached it"""
        if self.last_update == -math.inf:
            return 0.0
        if self.clock is not None:
            return self.horizon
        if self.autonomous:
            if self.bound > self.horizon and not same_time(self.bound, self.horizon):
                return max(self.horizon, t_now)
            return math.inf
        arrival = min(self.run_times, default=math.inf)
        if arrival <= self.bound or same_time(arrival, self.bound):
            return arrival
        return math.inf

    def note_run_time(self, r: float, floor: float) -> None:
        """A pending run time (one already there within rounding: that one), its earliest spike floor."""
        q = next((q for q in self.run_times if same_time(r, q)), None)
        if q is None:
            self.run_times.append(q := r)
        if floor < self.floors.get(q, q):
            self.floors[q] = floor


class Scheduler:
    """See the module docstring. The states are per population (kept across steps and networks merged);
    a round's functions take the list of them prepare() returned, or a copy of it (states())."""

    def __init__(self):
        self._by_pop: dict = {}
        self.pops: list = []
        self._states: list[PopSchedule] = []

    def state(self, pop) -> PopSchedule:
        s = self._by_pop.get(pop)
        if s is None:
            s = self._by_pop[pop] = PopSchedule(pop)
        return s

    def merge(self, other: Scheduler) -> None:
        self._by_pop.update(other._by_pop)

    def prepare(self, pops: list) -> list[PopSchedule]:
        """Start of a step or plan: every population schedulable, and what the rounds look up constantly
        fixed for it (incoming connections with their sources' indices and minimum delays, clock, kind) --
        the scheduler's Python is a good part of a round's host time. Returns the live states, in pops' order."""
        for pop in pops:
            _check_schedulable(pop)
        index = {pop: i for i, pop in enumerate(pops)}
        states = []
        for pop in pops:
            s = self.state(pop)
            s.clock = pop._own_timestep()
            s.ins = [(c, index[c._source], pop._min_delay(c)) for c in pop.incoming]
            s.event_driven = pop._event_driven
            s.autonomous = pop.has_autonomous_dynamics
            states.append(s)
        self.pops, self._states = list(pops), states
        return states

    def states(self, copy: bool = False) -> list[PopSchedule]:
        """The live states (prepare's order), or a copy of them to plan on."""
        return [s.copy() for s in self._states] if copy else self._states

    def rounds(self, states: list[PopSchedule], t: float, until: float | None, planning: bool = False,
               max_rounds: int | None = None):
        """The rounds from network time t on, one at a time: (t, its runs [(population, t_start, t_end,
        gated)]) -- the caller launches them (or not: planning), and the round's bookkeeping follows when the
        next is asked for (arrived, unless planning; intake). Stepping (planning False): up to until, a round
        at until itself included (the times capped there). Planning: the rounds before until (or with no end,
        while a population wants one), at most max_rounds."""
        n = 0
        while max_rounds is None or n < max_rounds:
            if not planning and t >= until - _EPS:
                return
            t_next = self.next_time(states, t, cap=until)
            if t_next == math.inf or (planning and until is not None and t_next >= until - _EPS):
                return
            t, n = t_next, n + 1
            yield t, self.take(states, t)
            # (after the round's join: reads on the current stream see every run)
            if not planning:
                self.arrived(states, t)
            self.intake(states, planning=planning)

    @staticmethod
    def history_depth(pops: list, resolution: float) -> int:
        """The emission histories' depth K: a source's history must keep every run its slowest consumer
        hasn't read yet -- the most runs it makes between two of that consumer's reads, + 2 (naer2/history.py).
        From the timesteps: 2 + ceil(consumer timestep / source timestep), an unset one taken as the network's
        resolution (the finest a population runs at); at least 3 (every population running every round) -- at
        least EVENT_DRIVEN_DEPTH with an event-driven consumer anywhere (it reads whenever an arrival is due:
        no cadence to derive from). One K for the whole network (a consumer's sources must share theirs); a
        population's own history_depth overrides, an overrun still raises (check()). TODO: grow a history when
        a consumer falls behind (test_network_step's gap test)."""
        depth = 3
        if any(pop._event_driven and pop.incoming for pop in pops):
            depth = EVENT_DRIVEN_DEPTH
        for pop in pops:
            for conn in pop.incoming:
                reads, runs = pop._own_timestep() or resolution, conn._source._own_timestep() or resolution
                depth = max(depth, 2 + math.ceil(reads / runs - 1e-9))
        return depth

    # ---- a round

    def next_time(self, states: list[PopSchedule], t_now: float, cap: float | None = None) -> float:
        """The next round's time: every horizon relaxed to a fixed point (input bound = min over the incoming
        connections of source horizon + minimum delay; an event-driven population emits nothing before its
        earliest pending run: its horizon min(bound, that run's floor)), then the earliest time any
        population wants (next_run_time), at most cap."""
        for s in states:
            if s.ins:
                s.bound = math.inf
            if s.event_driven and s.last_update != -math.inf:
                s.horizon = math.inf
        for _ in range(len(states) + 1):
            changed = False
            for s in states:
                bound = math.inf
                for _conn, j, min_delay in s.ins:
                    b = states[j].horizon + min_delay
                    if b < bound:
                        bound = b
                horizon = s.horizon
                if s.event_driven and s.last_update != -math.inf:
                    horizon = min(bound, min((s.floors.get(r, r) for r in s.run_times), default=math.inf))
                if bound != s.bound or horizon != s.horizon:
                    s.bound, s.horizon, changed = bound, horizon, True
            if not changed:
                break
        wanted = [s.next_run_time(t_now) for s in states]
        t_next = min(wanted + ([cap] if cap is not None else []))
        if t_next < t_now - _EPS:
            stale = [f"{type(s.pop.instance).__name__} wants {w}" for s, w in zip(states, wanted)
                     if w < t_now - _EPS]
            raise RuntimeError(f"network time would go backwards from {t_now} to {t_next}: {', '.join(stale)} "
                               f"-- a population's run-time bookkeeping is wrong")
        return t_next

    @staticmethod
    def take(states: list[PopSchedule], t: float) -> list[tuple]:
        """The round at time t: every population whose next run time it is takes its run -- its window ends at
        t + its clock, else at its input bound. Returns the runs, [(population, t_start, t_end, gated)] (gated:
        event-driven -- a planned run its kernel skips when nothing is due)."""
        runs = []
        for s in states:
            if not same_time(s.next_run_time(t), t):
                continue
            s.last_update = t
            s.run_id += 1
            end = t + s.clock if s.clock is not None else s.bound
            s.window = (t, end)
            if not s.event_driven:
                s.horizon = end
            s.run_times = [r for r in s.run_times if r > end and not same_time(r, end)]
            if s.floors:
                s.floors = {r: f for r, f in s.floors.items() if r in s.run_times}
            runs.append((s.pop, t, end, s.event_driven))
        return runs

    @staticmethod
    def arrived(states: list[PopSchedule], t: float) -> None:
        """After the round at t ran (on the device): every event-driven population that ran notes its ring's
        next pending arrival (Population._pending_arrival -- a readback)."""
        for s in states:
            if s.event_driven and s.ins and s.last_update == t:
                arrival = s.pop._pending_arrival()
                if arrival is not None:
                    s.note_run_time(*arrival)

    def intake(self, states: list[PopSchedule], planning: bool = False) -> None:
        """End of every round: every consumer takes in what every source that ran since it last read it
        emitted -- each source run read once; an event-driven consumer schedules a run at the run's earliest
        emission + the connection's minimum delay (its emission summary: a readback, Population.
        _emission_summary; later per-link arrivals are found in its ring: arrived). planning: every time an
        arrival of the source run could make a run due instead (possible_run_times), nothing read."""
        for s in states:
            if not s.ins:
                continue
            read = s.read_run
            for conn, j, min_delay in s.ins:
                src = states[j]
                if src.run_id == 0 or read.get(conn) == src.run_id:
                    continue
                read[conn] = src.run_id
                if not s.event_driven:
                    continue  # its own clock or its inputs' horizon runs it: no summary needed
                if planning:
                    for r, floor in self.possible_run_times(s, conn, src, min_delay):
                        s.note_run_time(r, floor)
                    continue
                n, earliest = src.pop._emission_summary()  # a host sync (TODO.md, D-13)
                if n:
                    s.run_times.append(earliest + min_delay)
        # (the spikes themselves: the fused kernel's pull takes them in at the consumer's next run, straight
        # from the sources' histories)

    @staticmethod
    def possible_run_times(s: PopSchedule, conn, src: PopSchedule, min_delay: float) -> list[tuple[float, float]]:
        """(Planning) Every time a spike of the source's last run [ts, te) through conn could make this
        event-driven population's run due, without knowing its spikes: ts + the minimum delay (intake's,
        exact for a source emitting at its window's start), and the end of every slot an arrival in
        [ts + min delay, te + max delay] can land in (Population._pending_arrival's) -- each with its earliest
        spike (a slot's: its start, or ts + the minimum delay)."""
        ts, te = src.window
        w = s.pop._slot_width()
        k0 = math.floor((ts + min_delay) / w + 1e-9)
        k1 = math.floor((te + s.pop._max_delay(conn)) / w + 1e-9)
        return [(ts + min_delay, ts + min_delay)] + [(round((k + 1) * w, 12), max(round(k * w, 12), ts + min_delay))
                                                     for k in range(k0, k1 + 1)]


def _check_schedulable(pop) -> None:
    """An update() population needs a clock: its own timestep, its outgoing connections' smallest delay (a
    source), or inputs bounding how far it may integrate."""
    if pop.has_autonomous_dynamics and not pop.incoming and pop._own_timestep() is None:
        raise ValueError(
            f"{type(pop.instance).__name__} population has update() dynamics but "
            f"neither an explicit timestep, nor incoming connections, nor outgoing ones with a nonzero "
            f"delay, so nothing determines how far each update may integrate: set population.timestep."
        )
