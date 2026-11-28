"""
Populations on the fused kernel's runtime (naer2/population.py: run(),
fused_kernel(), runtime; TODO.md, runtime integration step 3), built
through the DSL and run directly -- no Network: the kernel generated from
their hooks, their dynamic attrs as the runtime's state, their incoming
connections' simulation topologies from Fabric.simulation.
"""

import math
import warnings

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


def test_a_run_too_large_for_its_ring_stops_and_reruns():
    """Every group's arena at its minimum (4-record blocks: per slot its
    reserved ones and a partly used one). The burst's intake doesn't fit: that run stops with
    nothing committed, so does every run after it (the source's, the
    consumer's), the ring grows to the run's need x grow_to once the device
    is idle, and every stopped run runs again -- every link delivered,
    exactly, nothing lost, no warning; the rounds after it too (each rerun
    with its own commands)."""
    src, dst, conn = _pair()
    dst.ring_block, dst.ring_in_flight = 4, 0.0
    # the burst, then a different pattern every round right after it: the runs stopped behind the burst's
    # intake are rerun with the commands they were launched with, not the later ones
    pattern = {r: torch.rand(300, generator=torch.Generator().manual_seed(2 + r)) < (0.3 if r == 0 else 0.01)
               for r in (0, 1, 2, 4)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # (a lost link warns)
        for r in range(8):
            t = round(1.0 + 0.1 * r, 6)
            src.fire_now = pattern.get(r, torch.zeros(300, dtype=torch.bool)).float()
            src.run(t, t + 0.1, round_id=r)
            dst.run(t, t + 0.1, round_id=r)
            if r == 0:
                start = dst.runtime.ring.n_blocks
                for rt in (src.runtime, dst.runtime):  # a stop noticed late: only once the status slots run out
                    judge = rt._judge_pending
                    rt._judge_pending = lambda stream, wait, judge=judge: judge(stream, wait) if wait else None
        dst.check()
    rt = dst.runtime
    assert rt.stops and rt.growths
    assert rt.ring.n_blocks > start
    count = sum(_expected(conn, p, 500)[0] for p in pattern.values())
    assert int(count.sum()) > 0
    assert torch.equal(rt.n_received.cpu().long(), count)


def test_a_ring_arena_shrinks_back_and_keeps_what_it_holds():
    """An arena larger than the runs need (here: started at every incoming
    link) shrinks once their need stays below a third of it (need x grow_at
    < half the size) for shrink_window runs -- to grow_to x the need -- and
    keeps what it holds: a burst taken in before the shrink (2 ms delay) and
    delivered after it arrives exactly."""
    src, dst, conn = _pair()
    conn.delay = 2.0
    dst.ring_block, dst.ring_in_flight = 4, 1.0
    burst = torch.rand(300, generator=torch.Generator().manual_seed(3)) < 0.05
    for r in range(30):
        t = round(1.0 + 0.1 * r, 6)
        src.fire_now = (burst if r == 1 else torch.zeros(300, dtype=torch.bool)).float()
        src.run(t, t + 0.1, round_id=r)
        dst.run(t, t + 0.1, round_id=r)
        if r == 0:
            dst.runtime.shrink_window = 5
            start = dst.runtime.ring.n_blocks
    dst.check()
    rt = dst.runtime
    shrinks = [r for r in rt.resizes if r[1] < start]
    assert shrinks and 3 <= shrinks[0][0] < 21  # between the burst's intake and its delivery
    assert not [g for g in rt.growths if g[1] == "arena"]  # (the burst may outgrow the log: a stop, a rerun)
    count, _ = _expected(conn, burst, 500)
    assert int(count.sum()) > 0
    assert torch.equal(rt.n_received.cpu().long(), count)


# ------------------------------------------------------------ gaps (fail until closed)


def test_populations_draw_their_own_random_numbers():
    """random.random() draws from the population's own stream (Population.seed:
    RandomLoweringTransformer) -- two populations different by default, the
    same when their seeds are set equal."""
    A = PD14_poisson(rate=0.5, t_start=1.0, t_stop=2.0) * 500
    B = PD14_poisson(rate=0.5, t_start=1.0, t_stop=2.0) * 500
    for pop in (A, B):
        pop.run(1.0, 1.1, round_id=0)  # every neuron fires at the start, then draws its next spike
    assert not torch.equal(A.next_spike, B.next_spike)
    C = PD14_poisson(rate=0.5, t_start=1.0, t_stop=2.0) * 500
    C.seed = A.seed
    C.run(1.0, 1.1, round_id=0)
    assert torch.equal(A.next_spike, C.next_spike)
