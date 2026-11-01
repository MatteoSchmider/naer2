"""
Real-hardware tests for naer2/spike_gather.py's two Triton kernels
(compact(), merge()). No CPU reference here on purpose: a GPU-resident sort
measurably beats a CPU round trip at every size tried (see TODO.md's "Stay
on GPU" entry), so there's nothing worth checking a GPU implementation
against on the CPU this time -- correctness is checked by actually running
these kernels on a GPU.

Skipped entirely when no CUDA/ROCm device is visible to torch.
"""

import pytest
import torch

from naer2.spike_gather import BlockedSpikes, CarryBuffer, CompactedSpikes, compact, merge

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a real GPU (CUDA/ROCm); these kernels have no CPU reference to fall back to",
)

DEVICE = "cuda:0"


@pytest.fixture(autouse=True, scope="module")
def _device():
    torch.cuda.set_device(0)


def _blocked(groups: list[list[tuple[float, int]]], cap: int) -> BlockedSpikes:
    """groups[g] = [(time, node), ...] for group g; padded to `cap` with
    unwritten scratch (nan/-1), matching what a real kernel's local spike
    buffer looks like past its count -- never read, but worth not making
    accidentally plausible."""
    n = len(groups)
    time = torch.full((n * cap,), float("nan"), device=DEVICE, dtype=torch.float32)
    node = torch.full((n * cap,), -1, device=DEVICE, dtype=torch.int32)
    count = torch.zeros(n, device=DEVICE, dtype=torch.int32)
    for g, entries in enumerate(groups):
        assert len(entries) <= cap
        for i, (t, s) in enumerate(entries):
            time[g * cap + i] = t
            node[g * cap + i] = s
        count[g] = len(entries)
    return BlockedSpikes(node, time, count, cap)


def _compacted(pairs: list[tuple[float, int]], cap: int = 8) -> CompactedSpikes:
    time = torch.zeros(cap, device=DEVICE, dtype=torch.float32)
    node = torch.zeros(cap, device=DEVICE, dtype=torch.int32)
    for i, (t, s) in enumerate(pairs):
        time[i] = t
        node[i] = s
    count = torch.tensor([len(pairs)], device=DEVICE, dtype=torch.int32)
    return CompactedSpikes(time, node, count)


def _pairs(time: torch.Tensor, node: torch.Tensor, count: torch.Tensor) -> list[tuple[float, int]]:
    n = int(count[0].item())
    return list(zip(time[:n].tolist(), node[:n].tolist()))


class TestCompact:
    def test_no_carry_everything_due_gets_sorted(self):
        carry = CarryBuffer(cap=16, device=DEVICE)
        fresh = _blocked([[(5.0, 0), (1.0, 1), (3.0, 2)]], cap=8)

        due = compact(fresh, carry, t_hi=10.0)

        assert _pairs(due.time, due.node, due.count) == [(1.0, 1), (3.0, 2), (5.0, 0)]
        assert int(carry.count[0].item()) == 0

    def test_splits_due_from_not_yet_due(self):
        carry = CarryBuffer(cap=16, device=DEVICE)
        fresh = _blocked([[(5.0, 0), (15.0, 1), (3.0, 2)]], cap=8)

        due = compact(fresh, carry, t_hi=10.0)

        assert _pairs(due.time, due.node, due.count) == [(3.0, 2), (5.0, 0)]
        assert _pairs(carry.time, carry.node, carry.count) == [(15.0, 1)]

    def test_boundary_is_inclusive_of_t_hi(self):
        carry = CarryBuffer(cap=16, device=DEVICE)
        fresh = _blocked([[(10.0, 0), (10.5, 1)]], cap=8)

        due = compact(fresh, carry, t_hi=10.0)

        assert _pairs(due.time, due.node, due.count) == [(10.0, 0)]
        assert _pairs(carry.time, carry.node, carry.count) == [(10.5, 1)]

    def test_folds_carry_from_a_previous_round_in_with_fresh_multi_group(self):
        # 3 groups on purpose -- not a power of two, unlike the sort size
        # itself, to make sure group count isn't accidentally assumed to be.
        carry = CarryBuffer(cap=16, device=DEVICE)
        carry.time[0] = 20.0
        carry.node[0] = 9
        carry.count[0] = 1
        fresh = _blocked([[(1.0, 0), (2.0, 1)], [(25.0, 5)], []], cap=4)

        due = compact(fresh, carry, t_hi=10.0)

        assert _pairs(due.time, due.node, due.count) == [(1.0, 0), (2.0, 1)]
        assert _pairs(carry.time, carry.node, carry.count) == [(20.0, 9), (25.0, 5)]

    def test_lifecycle_across_three_rounds(self):
        """The actual use pattern: one CarryBuffer reused call after call,
        each call folding in that round's fresh output and advancing t_hi."""
        carry = CarryBuffer(cap=16, device=DEVICE)

        due1 = compact(_blocked([[(1.0, 0), (30.0, 1)]], cap=8), carry, t_hi=5.0)
        assert _pairs(due1.time, due1.node, due1.count) == [(1.0, 0)]
        assert _pairs(carry.time, carry.node, carry.count) == [(30.0, 1)]

        due2 = compact(_blocked([[(7.0, 2)]], cap=8), carry, t_hi=10.0)
        assert _pairs(due2.time, due2.node, due2.count) == [(7.0, 2)]
        assert _pairs(carry.time, carry.node, carry.count) == [(30.0, 1)], \
            "the delayed spike from round 1 must survive untouched through round 2"

        due3 = compact(_blocked([[]], cap=8), carry, t_hi=35.0)
        assert _pairs(due3.time, due3.node, due3.count) == [(30.0, 1)]
        assert int(carry.count[0].item()) == 0

    def test_empty_fresh_still_flushes_pure_carry(self):
        carry = CarryBuffer(cap=16, device=DEVICE)
        carry.time[0] = 1.0
        carry.node[0] = 7
        carry.count[0] = 1

        due = compact(_blocked([[]], cap=8), carry, t_hi=10.0)

        assert _pairs(due.time, due.node, due.count) == [(1.0, 7)]
        assert int(carry.count[0].item()) == 0

    def test_nothing_due_is_empty_not_an_error(self):
        carry = CarryBuffer(cap=16, device=DEVICE)
        fresh = _blocked([[(100.0, 0)]], cap=8)

        due = compact(fresh, carry, t_hi=10.0)

        assert int(due.count[0].item()) == 0
        assert _pairs(carry.time, carry.node, carry.count) == [(100.0, 0)]


class TestMerge:
    def test_combines_and_tags_sources_by_connection(self):
        s0 = _compacted([(5.0, 1)])
        s1 = _compacted([(2.0, 7)])

        time, node, conn, count = merge([s0, s1], conns=[0, 1], per_src_cap=8)

        n = int(count[0].item())
        result = list(zip(time[:n].tolist(), node[:n].tolist(), conn[:n].tolist()))
        assert result == [(2.0, 7, 1), (5.0, 1, 0)]

    def test_ties_break_ascending_by_connection_then_node(self):
        # Both the key's packing and the tie-break are a documented
        # contract (see spike_gather.py), not incidental: lower (conn, node)
        # sorts first among equal times.
        s0 = _compacted([(1.0, 0)])
        s1 = _compacted([(2.0, 7)])
        s2 = _compacted([(1.0, 3), (3.0, 4)])

        time, node, conn, count = merge([s0, s1, s2], conns=[0, 1, 2], per_src_cap=8)

        n = int(count[0].item())
        result = list(zip(time[:n].tolist(), node[:n].tolist(), conn[:n].tolist()))
        assert result == [(1.0, 0, 0), (1.0, 3, 2), (2.0, 7, 1), (3.0, 4, 2)]

    def test_single_source_is_a_pure_passthrough(self):
        s0 = _compacted([(3.0, 1), (1.0, 2), (2.0, 3)])

        time, node, conn, count = merge([s0], conns=[5], per_src_cap=8)

        n = int(count[0].item())
        result = list(zip(time[:n].tolist(), node[:n].tolist(), conn[:n].tolist()))
        assert result == [(1.0, 2, 5), (2.0, 3, 5), (3.0, 1, 5)]

    def test_a_source_with_nothing_due_contributes_nothing(self):
        s0 = _compacted([])
        s1 = _compacted([(1.0, 1)])

        time, node, conn, count = merge([s0, s1], conns=[0, 1], per_src_cap=8)

        n = int(count[0].item())
        result = list(zip(time[:n].tolist(), node[:n].tolist(), conn[:n].tolist()))
        assert result == [(1.0, 1, 1)]
