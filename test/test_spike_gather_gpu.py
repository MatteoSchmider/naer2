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

from naer2.spike_gather import BlockedSpikes, CarryBuffer, CompactedSpikes, combine, compact, merge

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


def _random_round(n_groups: int, cap: int, uniform_time: float | None, gen: torch.Generator) -> BlockedSpikes:
    """Real-kernel-shaped fresh output: random per-group counts, scratch
    past each count, and either one shared timestamp (what bare n2.spike()
    emits) or random, unsorted times."""
    count = torch.randint(0, cap + 1, (n_groups,), generator=gen, dtype=torch.int32)
    if uniform_time is None:
        time = torch.rand(n_groups * cap, generator=gen) * 20.0
    else:
        time = torch.full((n_groups * cap,), uniform_time)
    local = torch.arange(n_groups * cap) % cap
    valid = local < count.repeat_interleave(cap)
    time[~valid] = float("nan")
    node = torch.where(valid, torch.arange(n_groups * cap, dtype=torch.int32), -1)
    return BlockedSpikes(node.to(DEVICE), time.to(DEVICE), count.to(DEVICE), cap)


def _valid_entries(fresh: BlockedSpikes) -> list[tuple[float, int]]:
    local = torch.arange(fresh.time.shape[0], device=DEVICE) % fresh.cap
    valid = local < fresh.count.repeat_interleave(fresh.cap)
    return list(zip(fresh.time[valid].tolist(), fresh.node[valid].tolist()))


def _check_sorted_split(due, carry, expected: list[tuple[float, int]], t_hi: float):
    """Ties have no defined order, so compare as sets of (time, node) plus
    an explicit time-ascending check on each output."""
    got_due = _pairs(due.time, due.node, due.count)
    got_carry = _pairs(carry.time, carry.node, carry.count)
    for got in (got_due, got_carry):
        times = [t for t, _ in got]
        assert times == sorted(times)
    assert sorted(got_due) == sorted(e for e in expected if e[0] <= t_hi)
    assert sorted(got_carry) == sorted(e for e in expected if e[0] > t_hi)


class TestCompactAtScale:
    """compact() at real-network width (PD14's largest population has 685
    groups) -- the size that used to take tens of minutes to compile under
    a single tl.sort -- across several rounds, for both the generic sorted
    path and the fresh_sorted fast path."""

    N_GROUPS = 685
    CAP = 64

    @pytest.mark.parametrize("fresh_sorted", [False, True])
    def test_multi_round_lifecycle_matches_reference(self, fresh_sorted):
        gen = torch.Generator().manual_seed(1)
        carry = CarryBuffer(cap=self.N_GROUPS * self.CAP, device=DEVICE)
        pending: list[tuple[float, int]] = []
        for rnd, t_hi in enumerate([5.0, 10.0, 15.0, 25.0]):
            uniform = float(3 + 5 * rnd) if fresh_sorted else None
            fresh = _random_round(self.N_GROUPS, self.CAP, uniform, gen)
            expected = pending + _valid_entries(fresh)

            due = compact(fresh, carry, t_hi=t_hi, fresh_sorted=fresh_sorted)

            _check_sorted_split(due, carry, expected, t_hi)
            pending = [e for e in expected if e[0] > t_hi]

    def test_fast_path_agrees_with_generic_path_on_uniform_fresh(self):
        # carry straddling the shared fresh time on both sides, including
        # an exact tie, so the merge's tie rule is actually exercised
        results = []
        for fresh_sorted in (False, True):
            gen = torch.Generator().manual_seed(2)
            fresh = _random_round(self.N_GROUPS, self.CAP, 7.0, gen)
            carry = CarryBuffer(cap=16, device=DEVICE)
            carry.time[:5] = torch.tensor([1.0, 7.0, 7.0, 9.0, 12.0])
            carry.node[:5] = torch.tensor([-1, -2, -3, -4, -5], dtype=torch.int32)
            carry.count[0] = 5
            expected = _pairs(carry.time, carry.node, carry.count) + _valid_entries(fresh)

            due = compact(fresh, carry, t_hi=8.0, fresh_sorted=fresh_sorted)

            _check_sorted_split(due, carry, expected, 8.0)
            results.append((sorted(_pairs(due.time, due.node, due.count)),
                            _pairs(carry.time, carry.node, carry.count)))
        assert results[0] == results[1]


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


class TestCombine:
    """combine() -- the scheduling step's building block (see Network.step() and
    TODO.md): unlike merge(), neither input gets a connection id assigned
    here, since both inputs are themselves already merge()/combine()
    outputs (already (time, node, conn)-tagged)."""

    def test_combines_two_already_tagged_streams(self):
        a = merge([_compacted([(5.0, 1)])], conns=[0], per_src_cap=8)
        b = merge([_compacted([(2.0, 7)])], conns=[1], per_src_cap=8)

        time, node, conn, count = combine(a, b)

        n = int(count[0].item())
        result = list(zip(time[:n].tolist(), node[:n].tolist(), conn[:n].tolist()))
        assert result == [(2.0, 7, 1), (5.0, 1, 0)]

    def test_preserves_each_input_s_own_conn_tags_unchanged(self):
        # a already combines conns 0 and 1; b is conn 2 alone -- combine()
        # must not reassign or collapse any of these.
        a = merge([_compacted([(1.0, 0)]), _compacted([(4.0, 9)])], conns=[0, 1], per_src_cap=8)
        b = merge([_compacted([(2.0, 5)])], conns=[2], per_src_cap=8)

        time, node, conn, count = combine(a, b)

        n = int(count[0].item())
        result = list(zip(time[:n].tolist(), node[:n].tolist(), conn[:n].tolist()))
        assert result == [(1.0, 0, 0), (2.0, 5, 2), (4.0, 9, 1)]

    def test_empty_a_is_a_pure_passthrough_of_b(self):
        a = merge([_compacted([])], conns=[0], per_src_cap=8)
        b = merge([_compacted([(3.0, 2)])], conns=[4], per_src_cap=8)

        time, node, conn, count = combine(a, b)

        n = int(count[0].item())
        result = list(zip(time[:n].tolist(), node[:n].tolist(), conn[:n].tolist()))
        assert result == [(3.0, 2, 4)]

    def test_empty_b_is_a_pure_passthrough_of_a(self):
        a = merge([_compacted([(3.0, 2)])], conns=[4], per_src_cap=8)
        b = merge([_compacted([])], conns=[0], per_src_cap=8)

        time, node, conn, count = combine(a, b)

        n = int(count[0].item())
        result = list(zip(time[:n].tolist(), node[:n].tolist(), conn[:n].tolist()))
        assert result == [(3.0, 2, 4)]
