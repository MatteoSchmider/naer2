"""
Network.step()'s horizon scheduling (Population._relax_horizon /
_next_run_time / the run check in _execute_update_kernel), on one network
covering every kind of population:

  A   timestep 0.2, update()          advances strictly by its timestep
  B   timestep 0.3, update()          advances strictly by its timestep
  C   event-driven, B -> C delay 0.5  runs when an arrival is due, then
                                      processes up to B's horizon + 0.5
  D   event-driven, B -> D per-link   same, but a spike's 0.9 link arrives
      delays {0.5, 0.9}               past the window its 0.5 link ran in:
                                      the run that finds it records it
  U   update(), no timestep,          integrates to A's horizon + 0.1, the
      A -> U delay 0.1                window its input allows
  Z   event-driven, C -> Z delay 0    zero delay: lags its source by one
                                      round at the same time (two runs at 0)
  E1, E2  event-driven cycle,         bounded only through each other
      delay 0.3 both ways

Every population runs once at t = 0. Intake is the real host side
(Population._intake: read each source once per run, schedule event-driven
consumers from the source's emission summary). The update kernel is a
stand-in (the real one is being redesigned): it records the run's window,
leaves one spike stamped at the window's start in the emission buffer, and,
like the real one will, records a later per-link arrival it finds.
"""

import pytest
import torch

from naer2.population import Population
from naer2.spike_gather import BlockedSpikes
from test.nodes import counter, delay_probe
from test.patterns import all_to_all

EXPECTED = {
    0.0: ["A->0.2", "B->0.3", "C->0.5", "D->0.5", "E1->0.3", "E2->0.3", "U->0.1", "Z->0.0", "Z->0.5"],
    0.1: ["U->0.3"],
    0.2: ["A->0.4"],
    0.3: ["B->0.6", "E1->0.6", "E2->0.6", "U->0.5"],
    0.4: ["A->0.6"],
    0.5: ["C->1.1", "D->1.1", "U->0.7", "Z->1.1"],
    0.6: ["A->0.8", "B->0.9", "E1->0.9", "E2->0.9"],
    0.7: ["U->0.9"],
    0.8: ["A->1.0"],
    0.9: ["B->1.2", "E1->1.2", "E2->1.2", "U->1.1"],
    1.0: ["A->1.2"],
    1.1: ["C->1.7", "D->1.7", "U->1.3", "Z->1.7"],
    1.2: ["A->1.4", "B->1.5", "E1->1.5", "E2->1.5"],
    1.3: ["U->1.5"],
}


class _Ran(Exception):
    pass


def _delays(conn) -> list[float]:
    d = getattr(conn.fabric, "delay", None)
    return sorted(set(d.tolist())) if isinstance(d, torch.Tensor) else [d or 0.0]


@pytest.fixture
def stand_ins(monkeypatch):
    runs: list[tuple[float, Population, float]] = []
    taken: dict = {}  # conn -> emission times still to deliver

    def kernel(self, device):  # reached only once the run check has passed
        start, end = self._window
        runs.append((start, self, end))
        self._fresh = BlockedSpikes(
            torch.zeros(1, dtype=torch.int32), torch.tensor([start]),
            torch.ones(1, dtype=torch.int32), 1,
        )
        for conn in self.incoming:
            for e in taken.get(conn, []):
                later = [e + d for d in _delays(conn) if e + d > end + 1e-9]
                if later and min(later) not in self._run_times:
                    self._run_times.append(min(later))
            taken[conn] = [e for e in taken.get(conn, []) if e + max(_delays(conn)) > end + 1e-9]
        raise _Ran

    original_update = Population._execute_update_kernel

    def update(self, t):
        try:
            original_update(self, t)
        except _Ran:
            pass

    original_intake = Population._intake

    def intake(self, t):  # real intake, plus noting what was taken for the kernel stand-in
        before = dict(self._read_run)
        original_intake(self, t)
        for conn in self.incoming:
            if self._read_run.get(conn) != before.get(conn):
                taken.setdefault(conn, []).append(conn._source._window[0])

    monkeypatch.setattr(Population, "_ensure_on_device", kernel)
    monkeypatch.setattr(Population, "_execute_update_kernel", update)
    monkeypatch.setattr(Population, "_intake", intake)
    return runs


def test_every_kind_of_population_runs_exactly_when_and_as_far_as_it_may(stand_ins):
    pops = {}
    for name, node, timestep in [
        ("A", counter, 0.2), ("B", counter, 0.3), ("C", delay_probe, None),
        ("D", delay_probe, None), ("U", counter, None), ("Z", delay_probe, None),
        ("E1", delay_probe, None), ("E2", delay_probe, None),
    ]:
        pops[name] = node() * 2
        pops[name].timestep = timestep
    p = pops
    (p["B"] >> all_to_all >> p["C"]).fabric.delay = 0.5
    (p["B"] >> all_to_all >> p["D"]).fabric.delay = torch.tensor([0.5, 0.9, 0.5, 0.9])
    (p["A"] >> all_to_all >> p["U"]).fabric.delay = 0.1
    p["C"] >> all_to_all >> p["Z"]
    (p["E1"] >> all_to_all >> p["E2"]).fabric.delay = 0.3
    (p["E2"] >> all_to_all >> p["E1"]).fabric.delay = 0.3
    network = p["A"].network + p["B"].network + p["E1"].network

    network.step(1.3)

    names = {pop: name for name, pop in pops.items()}
    got: dict[float, list[str]] = {}
    for start, pop, end in stand_ins:
        got.setdefault(round(start, 6), []).append(f"{names[pop]}->{round(end, 6)}")
    assert {t: sorted(r) for t, r in got.items()} == EXPECTED


def test_a_source_run_without_spikes_schedules_nothing(monkeypatch):
    """Intake reads the source's emission summary: B runs every 0.3 but
    never emits, so event-driven C only ever runs at t = 0."""
    runs = []

    def kernel(self, device):
        runs.append((round(self._window[0], 6), self))
        self._fresh = BlockedSpikes(
            torch.zeros(1, dtype=torch.int32), torch.zeros(1),
            torch.zeros(1, dtype=torch.int32), 1,
        )
        raise _Ran

    original_update = Population._execute_update_kernel

    def update(self, t):
        try:
            original_update(self, t)
        except _Ran:
            pass

    monkeypatch.setattr(Population, "_ensure_on_device", kernel)
    monkeypatch.setattr(Population, "_execute_update_kernel", update)

    B = counter() * 2
    B.timestep = 0.3
    C = delay_probe() * 2
    (B >> all_to_all >> C).fabric.delay = 0.5
    B.network.step(2.0)

    assert [t for t, pop in runs if pop is C] == [0.0]
    assert len([t for t, pop in runs if pop is B]) == 7  # 0.0, 0.3, ..., 1.8


def test_update_population_without_timestep_or_inputs_is_rejected():
    lonely = counter() * 2
    with pytest.raises(ValueError, match="set population.timestep"):
        lonely.network.step(1.0)
