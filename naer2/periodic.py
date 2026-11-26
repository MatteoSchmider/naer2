"""
Periodic windows: a clock-only network's planned windows (Network.plan -> naer2/schedule.py) repeat -- the same
launches, every entry and every piece of scheduling state shifted by the same amount from one window to the
next (run ids and rounds by the window's runs and rounds, ring slots by the slots it delivers, times by its
length). Once three planned windows in a row show it, the next ones are extrapolated instead of planned: no
dry run of the populations' scheduling, no entries written one by one -- window k after the last planned one
is that one + k x the shift, in one numpy step per runtime (from the last planned window, not accumulated:
no drift), the network's and every population's scheduling state set the same way. Optional, for networks
whose populations are all clocked (no event-driven one, no gated launch); Network.run_graph uses it where it
applies and plans otherwise.

    Periodic()                   learns from planned windows (learn), extrapolates trusted ones (next)
    .learn(network, schedule)    a window just planned and committed: one more sample
    .next(network, n)            the next window of n rounds, extrapolated (None: not trusted / not this n)
    .forget()                    back to planning (the network changed, a run stopped, ...)

What is compared and shifted: the entries (integer columns exactly, times up to float rounding), the network's
time and round, each population's scheduling state (Population._schedule_state: numbers shifted, everything
else equal), each runtime's run id and ring position.
"""

import math

import numpy as np

from .schedule import Schedule

_TOL = 1e-6  # (float leaves: the shift the same up to rounding -- the plan accumulates its times)


def _diff(a, b):
    """b - a over a state's leaves (tuples, lists, dicts by key), None where a leaf must stay as it is;
    raises ValueError where the two don't have the same shape."""
    if isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError("shape")
        return type(a)(_diff(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("keys")
        return {k: _diff(a[k], b[k]) for k in a}
    if isinstance(a, bool) or a is None or not isinstance(a, (int, float)):
        if a is not b and a != b:
            raise ValueError("value")
        return None
    if isinstance(a, float) and (math.isinf(a) or math.isinf(b)):
        if a != b:
            raise ValueError("inf")
        return None
    return b - a


def _same(d1, d2) -> bool:
    """Whether two shifts are the same (integers exactly, floats up to rounding)."""
    if isinstance(d1, (tuple, list)):
        return len(d1) == len(d2) and all(_same(x, y) for x, y in zip(d1, d2))
    if isinstance(d1, dict):
        return d1.keys() == d2.keys() and all(_same(d1[k], d2[k]) for k in d1)
    if d1 is None or d2 is None:
        return d1 is None and d2 is None
    if isinstance(d1, int) and isinstance(d2, int):
        return d1 == d2
    return abs(d1 - d2) <= _TOL * max(1.0, abs(d1))


def _shifted(x, d, k: int):
    """x + k d over a state's leaves (d from _diff: None leaves as they are)."""
    if isinstance(x, (tuple, list)):
        return type(x)(_shifted(a, b, k) for a, b in zip(x, d))
    if isinstance(x, dict):
        return {key: _shifted(x[key], d[key], k) for key in x}
    return x if d is None else x + k * d


class _Snapshot:
    """A planned and committed window: its schedule, and the state it left (network, populations, runtimes)."""

    def __init__(self, network, schedule: Schedule):
        self.schedule = schedule
        self.n = len(schedule.launches)
        self.signature = schedule.signature()
        self.network = (network.t, network.round_id)
        self.pops = {id(p): (p, p._schedule_state()) for p in network.populations}
        self.runtimes = {key: (rt.run_id, rt.ring.ring_pos) for key, rt in schedule.runtimes.items()}
        self.rows = schedule.rows


class _Shift:
    """The shift from one snapshot to the next (raises ValueError if they don't have the same shape)."""

    def __init__(self, a: _Snapshot, b: _Snapshot):
        if a.signature != b.signature or a.n != b.n or a.pops.keys() != b.pops.keys() \
                or a.rows.keys() != b.rows.keys():
            raise ValueError("shape")
        self.network = _diff(a.network, b.network)
        self.pops = {key: _diff(a.pops[key][1], b.pops[key][1]) for key in a.pops}
        self.runtimes = {key: _diff(a.runtimes[key], b.runtimes[key]) for key in a.runtimes}
        self.rows = {}
        for key in a.rows:
            d = b.rows[key].astype(np.int64) - a.rows[key].astype(np.int64)
            ints = d[:, [0, 1, 2, 3]]
            if (ints != ints[0]).any() or (d[:, [6, 7]] != 0).any():
                raise ValueError("entries")
            t = b.rows[key][:, 4:6].view(np.float32).astype(np.float64) - \
                a.rows[key][:, 4:6].view(np.float32).astype(np.float64)
            if np.abs(t - t[0, 0]).max() > _TOL * max(1.0, abs(t[0, 0])):
                raise ValueError("times")
            self.rows[key] = (ints[0], float(t[0, 0]))

    def same(self, other: "_Shift") -> bool:
        return (_same(self.network, other.network) and self.pops.keys() == other.pops.keys()
                and all(_same(self.pops[k], other.pops[k]) for k in self.pops)
                and all(_same(self.runtimes[k], other.runtimes[k]) for k in self.runtimes)
                and all((self.rows[k][0] == other.rows[k][0]).all() and _same(self.rows[k][1], other.rows[k][1])
                        for k in self.rows))


class ShiftedSchedule:
    """Window k after a planned one (base), its shift applied k times: what a Schedule offers run_graph and
    naer2/graph.py -- the entries (rows), the runtimes, the runs per runtime, the signature, commit() -- its
    launches (the runs' list) only built if asked for (a capture, a recovery)."""

    def __init__(self, base: _Snapshot, shift: _Shift, k: int):
        self.base, self.shift, self.k = base, shift, k
        self.runtimes = base.schedule.runtimes
        self.n_runs = base.schedule.n_runs
        self._signature = base.signature
        self.rows = {}
        for key, rows in base.rows.items():
            ints, dt = shift.rows[key]
            r = rows.copy()
            r[:, 0:4] += (k * ints).astype(np.int32)
            t = rows[:, 4:6].view(np.float32).astype(np.float64) + k * dt
            r[:, 4:6] = t.astype(np.float32).view(np.int32)
            self.rows[key] = r

    def signature(self) -> tuple:
        return self._signature

    @property
    def launches(self) -> list:
        base, d = self.base.schedule.launches, self.shift.runtimes
        k = self.k
        return [[(pop, i, run_id + k * d[id(pop._runtime)][0]) for pop, i, run_id in runs] for runs in base]

    def commit(self) -> None:
        for key, rt in self.runtimes.items():
            rt.run_id, rt.ring.ring_pos = _shifted(self.base.runtimes[key], self.shift.runtimes[key], self.k)


class Periodic:
    """See the module docstring."""

    def __init__(self):
        self.forget()

    def forget(self) -> None:
        self._samples: list[_Snapshot] = []
        self._shift: _Shift | None = None
        self._k = 0
        self._end = None  # (the network's time and round where the last window, planned or not, ended)

    @staticmethod
    def applies(network, launches) -> bool:
        """Whether a planned window qualifies: every population clocked, no gated launch."""
        return (not any(p._is_event_driven for p in network.populations)
                and not any(g for runs in launches for *_, g in runs))

    def learn(self, network, schedule: Schedule, start: tuple) -> None:
        """A window just planned and committed; start: the network's (time, round) before it -- the samples
        are consecutive windows, or they start over."""
        if start != self._end:
            self._samples = []
        snap = _Snapshot(network, schedule)
        self._end = snap.network
        self._samples = (self._samples + [snap])[-3:]
        self._shift, self._k = None, 0
        if len(self._samples) == 3:
            a, b, c = self._samples
            try:
                s1, s2 = _Shift(a, b), _Shift(b, c)
            except ValueError:
                return
            if s1.same(s2):
                self._shift = s2

    def next(self, network, n: int) -> ShiftedSchedule | None:
        """The next window (n rounds) extrapolated -- its schedule, the network's and populations' scheduling
        state set to where it ends (as plan(commit=True) leaves them) -- or None (not trusted yet, another
        length, or the network not the one learned from)."""
        if self._shift is None:
            return None
        base = self._samples[-1]
        if (network.t, network.round_id) != self._end:  # (something else ran in between: plan again)
            self.forget()
            return None
        if n != base.n or {id(p) for p in network.populations} != base.pops.keys() \
                or any(base.schedule.runtimes.get(id(p._runtime)) is not p._runtime for p in network.populations):
            return None
        self._k += 1
        k = self._k
        schedule = ShiftedSchedule(base, self._shift, k)
        network.t, network.round_id = _shifted(base.network, self._shift.network, k)
        for key, (pop, state) in base.pops.items():
            pop._restore_schedule(_shifted(state, self._shift.pops[key], k))
        self._end = (network.t, network.round_id)
        return schedule
