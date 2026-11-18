"""
Populations on the fused kernel's runtime (naer2/population.py: run(),
fused_kernel(), runtime; TODO.md, runtime integration step 3), built
through the DSL and run directly -- no Network: the kernel generated from
their hooks, their dynamic attrs as the runtime's state, their incoming
connections' simulation topologies from Fabric.simulation.
"""

import math

import pytest
import torch

from test.networks.PD14.connection import PD14_connection
from test.networks.PD14.neuron import PD14_neuron
from test.networks.PD14.poisson import PD14_poisson
from test.nodes import commanded

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernel"
)


def _pair(n_src=300, n_dst=500, seed=0):
    torch.manual_seed(seed)
    src, dst = commanded() * n_src, PD14_neuron(K_cortical=2000) * n_dst
    conn = src >> PD14_connection(0.1) >> dst
    conn.weight = torch.rand(conn.shape)
    conn.delay = 0.2
    dst.delay_resolution = 0.1
    dst.count_received = True
    return src, dst, conn


def _expected(conn, fires, n):
    """Per target: links of the firing sources, and the sum of their weights."""
    offs, tgt = conn.fabric.topology.source_offsets, conn.fabric.topology.target_indices.long()
    weight = conn.fabric.weight.cpu().double()
    count, total = torch.zeros(n, dtype=torch.int64), torch.zeros(n, dtype=torch.float64)
    for s in fires.nonzero().flatten().tolist():
        count.index_add_(0, tgt[offs[s]:offs[s + 1]], torch.ones(int(offs[s + 1] - offs[s]), dtype=torch.int64))
        total.index_add_(0, tgt[offs[s]:offs[s + 1]], weight[offs[s]:offs[s + 1]])
    return count, total


def test_a_population_runs_its_generated_kernel():
    """The source fires once (round 0); its links reach the consumer in round
    2 (0.2 ms delay) -- every one, with its weight, nothing earlier."""
    src, dst, conn = _pair()
    fires = torch.rand(300, generator=torch.Generator().manual_seed(1)) < 0.3
    for r in range(4):
        t = round(1.0 + 0.1 * r, 6)
        src.fire_now = fires.float() if r == 0 else torch.zeros(300)
        src.run(t, t + 0.1, round_id=r)
        dst.run(t, t + 0.1, round_id=r)
        src.check()
        dst.check()
        if r == 0:
            assert sorted(node for _, node in src.emitted()) == fires.nonzero().flatten().tolist()
            assert src._emission_summary() == (int(fires.sum()), pytest.approx(1.0))
        if r < 2:
            assert int(dst.runtime.n_received.sum()) == 0, r
    count, total = _expected(conn, fires, 500)
    assert count.sum() > 0
    assert torch.equal(dst.runtime.n_received.cpu().long(), count)
    assert torch.allclose(dst.runtime.received.cpu().double(), total, atol=1e-4)


def test_the_state_is_the_population_s_own():
    """A dynamic attr is the runtime's state tensor -- what the kernel writes,
    the attr shows -- and assigning one afterwards swaps the runtime's too."""
    src, dst, _ = _pair()
    dst.potential = torch.full((500,), -60.0)
    dst.run(1.0, 1.1, round_id=0)
    assert dst.potential is dst.runtime.state["potential"]
    assert not torch.all(dst.potential == -60.0)  # the update ran on it
    dst.potential = torch.full((500,), -70.0)
    assert dst.runtime.state["potential"] is dst.potential and dst.potential.is_cuda
    dst.run(1.1, 1.2, round_id=1)
    assert not torch.all(dst.potential == -70.0)


def test_a_population_without_connections():
    """The thalamic Poisson population: no incoming connections, no receive."""
    T = PD14_poisson(rate=0.12, t_start=1.0, t_stop=2.0) * 902
    total = 0
    for r in range(10):
        t = round(1.0 + 0.1 * r, 6)
        T.run(t, t + 0.1, round_id=r)
        total += len(T.emitted())
    T.check()
    assert total > 902  # all at the start, then ~0.12 per ms
    assert torch.all(T.next_spike >= 1.0) and T.next_spike is T.runtime.state["next_spike"]


@pytest.mark.parametrize("ring", ["cells", "arena"])
def test_a_full_ring_run_doubles_its_capacity(ring):
    """cells: main capacity 64 per (group, slot); arena: every group's
    arena at its minimum (4-record blocks: per slot its reserved ones and a
    partly used one). The first
    burst overflows it -- a warning, its excess links lost, the capacity
    doubled before the next run (no host sync: the flags' host copy) -- and
    a later, smaller burst is delivered exactly."""
    src, dst, conn = _pair()
    if ring == "cells":
        dst.ring_block, dst.main_cap = None, 64
    else:
        dst.ring_block, dst.ring_in_flight = 4, 0.0
    big = torch.rand(300, generator=torch.Generator().manual_seed(2)) < 0.3
    small = torch.rand(300, generator=torch.Generator().manual_seed(3)) < 0.01
    with pytest.warns(RuntimeWarning, match="main ring run overflowed" if ring == "cells" else "ran out"):
        for r in range(8):
            t = round(1.0 + 0.1 * r, 6)
            src.fire_now = (big if r == 0 else small if r == 4 else torch.zeros(300, dtype=torch.bool)).float()
            src.run(t, t + 0.1, round_id=r)
            dst.run(t, t + 0.1, round_id=r)
            if r == 5:  # the big burst delivered (round 2), the small one not yet (round 6)
                torch.cuda.synchronize()
                before = dst.runtime.n_received.cpu().long().clone()
    dst.check()  # nothing left flagged
    if ring == "cells":
        assert dst.runtime.ring.main_cap == 128 and dst.runtime.growths[0][1:] == ("main", 128)
    else:
        from naer2.ring import POOL_RESERVE

        n_slots = dst.runtime.ring.n_slots  # the minimum arena: per slot its reserved blocks and a partly used one
        assert dst.runtime.growths[0][1:] == ("arena", 4 * 2 * (n_slots * (POOL_RESERVE + 1) + 1))
    count_big, _ = _expected(conn, big, 500)
    assert int(before.sum()) < int(count_big.sum())  # the overflowing run lost links
    count_small, _ = _expected(conn, small, 500)
    assert int(count_small.sum()) > 0
    assert torch.equal(dst.runtime.n_received.cpu().long() - before, count_small)


# ------------------------------------------------------------ gaps (fail until closed)


def test_gap_populations_draw_their_own_random_numbers():
    """TODO.md (RNG seed): random.random() lowers to a draw seeded from the
    run's time and the lane's neuron index only -- two populations draw the
    same numbers for the same lane at the same time. Each population owning
    a public seed gives them their own streams (different by default, the
    same when set equal)."""
    A = PD14_poisson(rate=0.5, t_start=1.0, t_stop=2.0) * 500
    B = PD14_poisson(rate=0.5, t_start=1.0, t_stop=2.0) * 500
    for pop in (A, B):
        pop.run(1.0, 1.1, round_id=0)  # every neuron fires at the start, then draws its next spike
    assert not torch.equal(A.next_spike, B.next_spike)  # equal today: the same draws
    C = PD14_poisson(rate=0.5, t_start=1.0, t_stop=2.0) * 500
    C.seed = A.seed
    C.run(1.0, 1.1, round_id=0)
    assert torch.equal(A.next_spike, C.next_spike)
