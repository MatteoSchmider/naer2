"""
The fused kernel's hand-written PD14 hooks (naer2/greenfield/fused.py,
MODEL) against the hooks themselves (test/networks/PD14/neuron.py,
poisson.py) written out in torch, run by run: the state every hook assigns,
which neurons spike and their emission times (bare n2.spike(): the window's
start, D-4). The references follow the hooks' expressions in their order, in
float32; exp/log may differ from the GPU's by an ulp, hence the tolerances.
"""

import math

import pytest
import torch
import triton
import triton.language as tl

from naer2.greenfield.fused import FusedPop, pd14_neuron_params, pd14_poisson_params
from test.networks.PD14.neuron import PD14_neuron
from test.networks.PD14.poisson import PD14_poisson
from test.topologies import random_topology

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernel"
)

DEVICE = "cuda:0"
G = 256


def _neuron_reference(neuron, state, time, received=None):
    """One run of PD14_neuron: update(time), then send(time) (the final
    pass; receive only adds to current, which the kernel's counters give)."""
    potential, current, t_ref_end, last_time = (state[k].clone() for k in ("potential", "current", "t_ref_end",
                                                                           "last_time"))
    if received is not None:
        current = current + received
    dt = time - last_time
    last_time = torch.full_like(last_time, time)
    P_V = torch.exp(-dt / neuron.t_mem)
    P_I = torch.exp(-dt / neuron.t_syn)
    P_VI = neuron.R_mem * neuron.t_syn / (neuron.t_mem - neuron.t_syn) * (P_V - P_I)
    refractory = time < t_ref_end
    potential = torch.where(refractory, neuron.V_reset,
                            neuron.V_rest + P_V * (potential - neuron.V_rest) + P_VI * current)
    current = P_I * current
    fires = potential > neuron.V_thresh
    potential = torch.where(fires, neuron.V_reset, potential)
    t_ref_end = torch.where(fires, time + neuron.t_ref, t_ref_end)
    return dict(potential=potential, current=current, t_ref_end=t_ref_end, last_time=last_time), fires


def _state(pop):
    return dict(potential=pop.potential.clone(), current=pop.current.clone(), t_ref_end=pop.t_ref_end.clone(),
                last_time=pop.last_time.clone())


def _check_run(pop, ref, fires, t_start):
    for k in ("potential", "current", "t_ref_end", "last_time"):
        assert torch.allclose(getattr(pop, k), ref[k], rtol=1e-5, atol=1e-5), k
    got = sorted(node for _, node in pop.emitted())
    assert got == fires.nonzero().flatten().tolist()
    assert all(e == pytest.approx(t_start) for e, _ in pop.emitted())


@pytest.mark.parametrize("K_cortical", [1500, 2900])
def test_pd14_neuron_matches_its_hooks(K_cortical):
    """No connections: update + send only, from a random state with neurons
    above threshold and refractory ones; V_rest is the population's own."""
    torch.manual_seed(0)
    neuron = PD14_neuron(K_cortical=K_cortical)
    n = 700
    pop = FusedPop(n, group_size=G, device=DEVICE, model="pd14_neuron", params=pd14_neuron_params(neuron))
    pop.potential = torch.normal(-55.0, 6.0, (n,), device=DEVICE)
    pop.current = torch.rand(n, device=DEVICE) * 0.5
    pop.t_ref_end = 1.0 + torch.rand(n, device=DEVICE) * 0.4
    pop.last_time = torch.full((n,), 0.9, device=DEVICE)
    spiked = 0
    for r in range(8):
        t = round(1.0 + 0.1 * r, 6)
        before = _state(pop)
        pop.run(t, t + 0.1, None, round_id=r)
        ref, fires = _neuron_reference(neuron, before, torch.tensor(t + 0.1, dtype=torch.float32).item())
        _check_run(pop, ref, fires, t)
        spiked += int(fires.sum())
    assert spiked > 0


def test_pd14_neuron_integrates_this_runs_deliveries():
    """A source firing every run into a PD14_neuron consumer: the update sees
    the current with this run's deliveries added (receive before update)."""
    torch.manual_seed(1)
    neuron = PD14_neuron(K_cortical=2000)
    src = FusedPop(300, group_size=G, device=DEVICE)  # the stand-in, firing on command
    dst = FusedPop(500, group_size=G, device=DEVICE, model="pd14_neuron", params=pd14_neuron_params(neuron))
    dst.connect([src], [random_topology(300, 500, G, delay=0.1, k=40, seed=3, weight=True)], n_slots=4)
    dst.potential = torch.normal(-60.0, 4.0, (500,), device=DEVICE)
    delivered = 0
    for r in range(6):
        t = round(1.0 + 0.1 * r, 6)
        before, received = _state(dst), dst.received.clone()
        dst.run(t, t + 0.1, None, round_id=r)
        ref, fires = _neuron_reference(neuron, before, torch.tensor(t + 0.1, dtype=torch.float32).item(),
                                       dst.received - received)
        _check_run(dst, ref, fires, t)
        delivered += int(dst.n_received.sum())
        src.run(t, t + 0.1, torch.where(torch.rand(300) < 0.3, t, math.inf), round_id=r)
    dst.check()
    assert delivered > 0


@triton.jit
def _rand_reference(out, t, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + offs, tl.rand(t.to(tl.int32, bitcast=True), offs), mask=offs < n)


def test_pd14_poisson_matches_its_hooks():
    """The thalamic population across its start and stop: random.random()
    as the transpiler lowers it, tl.rand(t bitcast, offs) at the window end."""
    poisson = PD14_poisson(rate=0.12, t_start=700.0, t_stop=701.0)
    n = 902
    pop = FusedPop(n, group_size=G, device=DEVICE, model="pd14_poisson", params=pd14_poisson_params(poisson))
    next_spike = torch.full((n,), poisson.t_start, device=DEVICE)
    u = torch.empty(n, device=DEVICE)
    spiked = 0
    for r in range(20):
        t = round(699.5 + 0.1 * r, 6)
        pop.run(t, t + 0.1, None, round_id=r)
        time = torch.tensor(t + 0.1, dtype=torch.float32).item()
        _rand_reference[(triton.cdiv(n, 256),)](u, time, n, 256)
        fires = (poisson.t_start <= time) & (time <= poisson.t_stop) & (next_spike <= time)
        fires = torch.as_tensor(fires, device=DEVICE) & torch.ones(n, dtype=torch.bool, device=DEVICE)
        next_spike = torch.where(fires, next_spike + -torch.log(u) / poisson.rate, next_spike)
        assert torch.allclose(pop.next_spike, next_spike, rtol=1e-6, atol=1e-4)
        got = sorted(node for _, node in pop.emitted())
        assert got == fires.nonzero().flatten().tolist(), t
        assert all(e == pytest.approx(t) for e, _ in pop.emitted())
        spiked += len(got)
    assert spiked > n  # every neuron spikes at the start, then at ~0.12 per ms
