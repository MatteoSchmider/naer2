"""
Brunel (2000) model A end to end (test/networks/Brunel, BRUNEL.md): event-driven LIF neurons with delta
synapses firing inside delivery at exact arrival times (D-1, D-3), driven by update()-only Poisson sources
running every D (their derived timestep), one homogeneous delay everywhere.

At N = 5000 with a fixed in-degree K = C_E = 1000 (the coupling of Brunel's N = 12500: its balance, at less
cost): the asynchronous irregular state's rates -- ~38 Hz, as an independent time-driven reference gives
(tools/brunel_reference.py: 37.6 Hz, CV(ISI) 0.42 at N = 12500) -- exact spike times off any grid, every
neuron's inputs in time order, no links lost.
"""

import math
import warnings

import pytest
import torch

import naer2 as n2
from test.networks.Brunel import build, theory

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm)")


@n2.node
class _checked_neuron:
    """Brunel_neuron (test/networks/Brunel/neuron.py), counting its inputs, and the ones stepping back in time
    (an arrival before its last exact time: never, D-1)."""

    def __init__(self, tau: float = 20.0, theta: float = 20.0, V_r: float = 10.0, t_ref: float = 2.0):
        self.tau, self.theta, self.V_r, self.t_ref = tau, theta, V_r, t_ref
        self.V: float = 0.0
        self.last_time: float = 0.0
        self.t_ref_end: float = 0.0
        self.back: float = 0.0
        self.inputs: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        if self.V >= self.theta:
            n2.spike(time)
            self.V = self.V_r
            self.t_ref_end = time + self.t_ref
            self.last_time = time + self.t_ref

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.inputs += 1.0
        if time >= self.t_ref_end:
            if time < self.last_time:
                self.back += 1.0
            self.V = self.V * math.exp((self.last_time - time) / self.tau) + edge.weight
            self.last_time = time


def _run(network, pops, until, D=1.5):
    """Steps of D, every population's spikes collected from its history after each: {name: [(e, node)]}."""
    spikes, read = {k: [] for k in pops}, {k: 0 for k in pops}
    t = 0.0
    while t < until - 1e-9:
        network.step(D)
        t += D
        for k, p in pops.items():
            rt = p.runtime
            rt.settle()
            for run in range(read[k] + 1, rt.run_id + 1):
                spikes[k] += rt.history.spikes(run)
            read[k] = rt.run_id
    return spikes, t


def test_brunel_A_asynchronous_irregular():
    N, K, T0 = 5000, 1000, 50.0
    network, pops = build(N, K=K, neuron=_checked_neuron)
    assert pops["P"].timestep is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # (a ring that ran out: links lost)
        spikes, t = _run(network, pops, 200.0)
        for p in pops.values():
            p.check()
    assert pops["P"]._own_timestep() == 1.5
    ref = theory(N, K=K)
    for name in ("E", "I", "P"):
        n = len(pops[name].shape)
        times = torch.tensor([e for e, _ in spikes[name]])
        rate = float((times >= T0).sum()) / n / ((t - T0) / 1000.0)
        if name == "P":
            assert abs(rate - ref["rate_ext_Hz"]) < 0.05 * ref["rate_ext_Hz"], rate
            continue
        assert 30.0 < rate < 46.0, (name, rate)
        # exact times, off the 0.1 ms grid (a grid-based simulation's would all sit on it)
        on_grid = torch.isclose(times * 10, torch.round(times * 10), atol=1e-3).float().mean()
        assert float(on_grid) < 0.05, float(on_grid)
        # irregular: the ISIs' CV well above a regular, mean-driven firing's (~0.1)
        nodes = torch.tensor([j for _, j in spikes[name]])
        cvs = []
        for j in range(min(n, 500)):
            tj = torch.sort(times[(nodes == j) & (times >= T0)]).values
            if len(tj) >= 4:
                isi = tj.diff()
                cvs.append(float(isi.std() / isi.mean()))
        assert len(cvs) > 200 and sum(cvs) / len(cvs) > 0.3, sum(cvs) / max(len(cvs), 1)
        # every neuron's inputs in time order (D-1) -- ~67 a ms each (C_E 38 Hz + C_I 38 Hz + C_ext 20 Hz)
        assert int(pops[name].back.sum()) == 0
        assert float(pops[name].inputs.sum()) / n / t > 40.0
