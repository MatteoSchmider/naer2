"""
Fused rounds (naer2/fusion.py, Network.fuse_rounds): a round in which every population runs as one kernel
launch, each population's share its own generated body -- the same runs as unfused, so the same results,
exactly: the mini E/I network (test/networks/mini_ei: two PD14-neuron populations, all four connections, both
clocked) through planned and captured windows, fused and not, every node attr and received count compared.
"""

import pytest
import torch

from test.networks.mini_ei import build

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernels"
)


def _run(mode: str, fuse: bool):
    torch.manual_seed(0)
    network, pops = build()
    for pop in pops.values():
        pop.count_received = True
        pop.potential = torch.normal(-52.0, 4.0, pop.potential.shape)  # (some fire from the start)
        for conn in pop.incoming:
            conn.delay = 0.5
    network.step(0.1)  # the first round: every runtime built
    network.fuse_rounds = fuse
    run = network.run_planned if mode == "planned" else network.run_graph
    for _ in range(3):
        run(max_launches=10)
    for pop in pops.values():
        pop.check()
    launches = network._fused.launches if network._fused is not None else 0
    state = {(n, a): t.detach().cpu().clone() for n, p in pops.items() for a, t in p.runtime.state.items()}
    received = {n: p.runtime.n_received.detach().cpu().clone() for n, p in pops.items()}
    return state, received, launches


@pytest.mark.parametrize("mode", ["planned", "graph"])
def test_fused_rounds_run_what_unfused_ones_do(mode):
    state, received, none = _run(mode, fuse=False)
    state_f, received_f, launches = _run(mode, fuse=True)
    # every round one fused launch: the planned windows' 30; a graph's at its captures (two, then replays)
    assert none == 0 and launches == (30 if mode == "planned" else 20)
    assert sum(int(r.sum()) for r in received.values()) > 0, "links were delivered"
    for n in received:
        assert torch.equal(received[n], received_f[n]), f"{n}: received counts differ"
    for key in state:
        assert torch.equal(state[key], state_f[key]), f"{key}: differs"


def test_rounds_not_every_population_runs_launch_per_population():
    """A population with a coarser timestep: the rounds it sits out launch the other alone (its own kernel),
    the rounds both run fused -- and the results are the unfused run's."""
    def run(fuse):
        torch.manual_seed(0)
        network, pops = build()
        pops["E"].timestep, pops["I"].timestep = 0.1, 0.2  # (both given: the history depth from them)
        for pop in pops.values():
            pop.count_received = True
            pop.potential = torch.normal(-52.0, 4.0, pop.potential.shape)
            for conn in pop.incoming:
                conn.delay = 0.5
        network.step(0.2)
        network.fuse_rounds = fuse
        rounds = network.plan(max_launches=10)
        mixed = sum(1 for r in rounds if len(r) == 1)
        network.run_planned(max_launches=10)
        for pop in pops.values():
            pop.check()
        launches = network._fused.launches if network._fused is not None else 0
        return ({(n, a): t.detach().cpu().clone() for n, p in pops.items() for a, t in p.runtime.state.items()},
                launches, mixed, len(rounds))
    state, _, _, _ = run(False)
    state_f, launches, mixed, n_rounds = run(True)
    assert 0 < mixed < n_rounds and launches == n_rounds - mixed
    for key in state:
        assert torch.equal(state[key], state_f[key]), f"{key}: differs"
