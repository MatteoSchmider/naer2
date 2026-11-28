"""
Periodic windows (naer2/periodic.py, Network.periodic_windows): a clock-only network's captured windows, once
three planned ones in a row repeat, extrapolated instead of planned -- the same entries as planning gives (the
integer columns exactly, the times as float32), so the same results; something else running in between
(step()) makes it plan again; an event-driven network never uses them.
"""

import numpy as np
import pytest
import torch

import naer2.graph as graph
from test.networks.mini_ei import build
from test.test_fusion import _same_state

_REPLAY = graph.WindowGraph.replay  # (the real one: each run records through its own wrapper)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernels"
)


def _mini(periodic: bool):
    torch.manual_seed(0)
    network, pops = build()
    for pop in pops.values():
        pop.count_received = True
        pop.potential = torch.normal(-52.0, 4.0, pop.potential.shape)
        for conn in pop.incoming:
            conn.delay = 0.5
    network.step(0.1)
    network.periodic_windows = periodic
    return network, pops


def _windows(network, pops, n, between=None, monkeypatch=None):
    """n windows of 10 rounds through run_graph, every one's entries recorded (by population: E's, I's)."""
    rows = []

    def recording(self, schedule):
        rows.append({name: schedule.rows[id(p.runtime)].copy() for name, p in pops.items()})
        return _REPLAY(self, schedule)

    monkeypatch.setattr(graph.WindowGraph, "replay", recording)
    for w in range(n):
        network.run_graph(max_launches=10)
        if between is not None and w == between:
            network.step(0.3)
    for pop in pops.values():
        pop.check()
    state = {(n_, a): t.detach().cpu().clone() for n_, p in pops.items() for a, t in p.runtime.state.items()}
    received = {n_: p.runtime.n_received.detach().cpu().clone() for n_, p in pops.items()}
    return rows, state, received


def _same_rows(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key][:, [0, 1, 2, 3, 6, 7]], b[key][:, [0, 1, 2, 3, 6, 7]]), "integer columns"
        assert np.array_equal(a[key][:, 4:6], b[key][:, 4:6]), "times (float32)"


@pytest.mark.parametrize("between", [None, 4], ids=["windows only", "a step in between"])
def test_periodic_windows_are_the_planned_ones(between, monkeypatch):
    planned_net, planned_pops = _mini(periodic=False)
    rows_p, state_p, received_p = _windows(planned_net, planned_pops, 9, between, monkeypatch)
    net, pops = _mini(periodic=True)
    rows, state, received = _windows(net, pops, 9, between, monkeypatch)
    assert net.executor.periodic._k > 0, "some windows were extrapolated"
    for a, b in zip(rows_p, rows):
        _same_rows(a, b)
    for n in received:
        assert torch.equal(received[n], received_p[n]), f"{n}: received counts differ"
    _same_state(state_p, state)


def test_an_event_driven_network_plans_every_window():
    from test.test_network_step import _net
    network, A, B, conn, fires = _net(0.5)
    network.step(0.05)
    network.periodic_windows = True
    for _ in range(4):
        network.run_graph(max_launches=2)
    assert network.executor.periodic._shift is None and network.executor.periodic._k == 0
