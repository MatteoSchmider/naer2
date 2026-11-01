"""
Integration tests for the spike-management kernels (naer2/spike_gather.py)
wired into real Population/Network objects, built the normal way through
the DSL (test/networks/gather_net: A, B -> C via static_synapse/all_to_all).

Written before the integration itself (see TODO.md): spikes are controlled
with static data via Population.set_fresh_output() rather than by running
the update kernel, so this is a test of the *wiring* -- each population
getting a real, persistent CarryBuffer; conn ids in a merge matching a
population's real `incoming` connection order; Network sequencing every
population's compact() before any merge_incoming() reads it -- not a retest
of compact()/merge()'s own correctness, which test_spike_gather_gpu.py
already covers directly.

Skipped entirely when no CUDA/ROCm device is visible to torch.
"""

import pytest
import torch

from naer2.spike_gather import BlockedSpikes
from test.networks.gather_net import build

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a real GPU (CUDA/ROCm); spike_gather's kernels have no CPU reference",
)

DEVICE = "cuda:0"


@pytest.fixture(autouse=True, scope="module")
def _device():
    torch.cuda.set_device(0)


def _blocked(groups: list[list[tuple[float, int]]], cap: int) -> BlockedSpikes:
    n = len(groups)
    time = torch.full((n * cap,), float("nan"), device=DEVICE, dtype=torch.float32)
    node = torch.full((n * cap,), -1, device=DEVICE, dtype=torch.int32)
    count = torch.zeros(n, device=DEVICE, dtype=torch.int32)
    for g, entries in enumerate(groups):
        for i, (t, s) in enumerate(entries):
            time[g * cap + i] = t
            node[g * cap + i] = s
        count[g] = len(entries)
    return BlockedSpikes(node, time, count, cap)


def _pairs(time, node, count):
    n = int(count[0].item())
    return list(zip(time[:n].tolist(), node[:n].tolist()))


def _triples(time, node, conn, count):
    n = int(count.item()) if count.dim() == 0 else int(count[0].item())
    return list(zip(time[:n].tolist(), node[:n].tolist(), conn[:n].tolist()))


@pytest.fixture
def net():
    return build()


class TestPopulationCompact:
    """Population.compact() against a real Population (no incoming
    connections needed here -- compact() is purely a source-side concern)."""

    def test_runs_against_a_real_population_and_stores_due(self, net):
        _, pops = net
        A = pops["A"]

        A.set_fresh_output(_blocked([[(5.0, 0), (1.0, 1), (3.0, 2)]], cap=8))
        due = A.compact(t_hi=10.0)

        assert _pairs(due.time, due.node, due.count) == [(1.0, 1), (3.0, 2), (5.0, 0)]
        assert A._due is due

    def test_carry_persists_on_the_population_across_calls(self, net):
        _, pops = net
        A = pops["A"]

        A.set_fresh_output(_blocked([[(1.0, 0), (30.0, 1)]], cap=8))
        due1 = A.compact(t_hi=5.0)
        assert _pairs(due1.time, due1.node, due1.count) == [(1.0, 0)]

        A.set_fresh_output(_blocked([[(7.0, 2)]], cap=8))
        due2 = A.compact(t_hi=10.0)
        assert _pairs(due2.time, due2.node, due2.count) == [(7.0, 2)], \
            "the delayed spike (30.0, 1) from the first call must not leak into this round"

        A.set_fresh_output(_blocked([[]], cap=8))
        due3 = A.compact(t_hi=35.0)
        assert _pairs(due3.time, due3.node, due3.count) == [(30.0, 1)], \
            "the carried-over spike must survive to whichever later call it's finally due in"

    def test_each_population_owns_its_own_carry(self, net):
        _, pops = net
        A, B = pops["A"], pops["B"]

        A.set_fresh_output(_blocked([[(100.0, 0)]], cap=8))
        A.compact(t_hi=5.0)
        B.set_fresh_output(_blocked([[(1.0, 1)]], cap=8))
        due_b = B.compact(t_hi=5.0)

        assert _pairs(due_b.time, due_b.node, due_b.count) == [(1.0, 1)], \
            "A's not-yet-due carry must not bleed into B's compaction"


class TestPopulationMergeIncoming:
    """Population.merge_incoming() against C's two real connections from
    gather_net (conn0 = A->C, conn1 = B->C, matching C.incoming's order --
    the same order compile() numbers conn0_/conn1_ pointers by)."""

    def test_tags_each_source_with_its_real_connection_index(self, net):
        _, pops = net
        A, B, C = pops["A"], pops["B"], pops["C"]
        assert C.incoming[0]._source is A and C.incoming[1]._source is B

        A.set_fresh_output(_blocked([[(5.0, 1)]], cap=8))
        A.compact(t_hi=10.0)
        B.set_fresh_output(_blocked([[(2.0, 7)]], cap=8))
        B.compact(t_hi=10.0)

        time, node, conn, count = C.merge_incoming(per_src_cap=8)

        assert _triples(time, node, conn, count) == [(2.0, 7, 1), (5.0, 1, 0)]

    def test_raises_if_a_source_has_not_compacted_this_round(self, net):
        _, pops = net
        C = pops["C"]
        with pytest.raises(ValueError):
            C.merge_incoming(per_src_cap=8)


class TestNetworkGather:
    """Network.gather(): the two-phase coordination -- every population's
    compact() first (a barrier other populations' merge_incoming() can rely
    on), then merge_incoming() for populations that have incoming
    connections."""

    def test_compacts_every_population_then_merges_targets(self, net):
        network, pops = net
        A, B, C = pops["A"], pops["B"], pops["C"]

        A.set_fresh_output(_blocked([[(5.0, 1)]], cap=8))
        B.set_fresh_output(_blocked([[(2.0, 7)]], cap=8))
        C.set_fresh_output(_blocked([[]], cap=8))

        results = network.gather(t_hi=10.0)

        assert A._due is not None and B._due is not None and C._due is not None, \
            "compact() must run for every population, not just the ones feeding a merge"
        assert C in results
        assert A not in results and B not in results, \
            "a population with no incoming connections has nothing to merge"
        time, node, conn, count = results[C]
        assert _triples(time, node, conn, count) == [(2.0, 7, 1), (5.0, 1, 0)]

    def test_lifecycle_survives_across_two_gather_calls(self, net):
        network, pops = net
        A, B, C = pops["A"], pops["B"], pops["C"]

        A.set_fresh_output(_blocked([[(1.0, 0), (30.0, 9)]], cap=8))
        B.set_fresh_output(_blocked([[]], cap=8))
        C.set_fresh_output(_blocked([[]], cap=8))
        results1 = network.gather(t_hi=5.0)
        time1, node1, conn1, count1 = results1[C]
        assert _triples(time1, node1, conn1, count1) == [(1.0, 0, 0)]

        A.set_fresh_output(_blocked([[]], cap=8))
        B.set_fresh_output(_blocked([[(2.0, 1)]], cap=8))
        C.set_fresh_output(_blocked([[]], cap=8))
        results2 = network.gather(t_hi=35.0)
        time2, node2, conn2, count2 = results2[C]
        assert _triples(time2, node2, conn2, count2) == [(2.0, 1, 1), (30.0, 9, 0)], \
            "A's delayed spike from the first gather() call must surface in the second"
