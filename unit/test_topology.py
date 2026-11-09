"""
Unit tests for naer2.topology: Topology.blocked / BlockedTopology.

Pure torch, no Triton, no network. Checks that the block-major layout is a
faithful re-arrangement of the canonical CSR: every link lands in exactly the
block that owns its target, each (block, source) range holds exactly the
canonical links from that source into that block, and per-link tensors
round-trip through to_blocked / from_blocked.
"""

import torch
import pytest

from naer2.topology import Topology, delay_buckets


def _topology(n_src: int, n_tgt: int, density: float, seed: int) -> Topology:
    g = torch.Generator().manual_seed(seed)
    rows = [torch.nonzero(torch.rand(n_tgt, generator=g) < density).flatten() for _ in range(n_src)]
    offsets = torch.tensor([0] + [len(r) for r in rows]).cumsum(0)
    targets = torch.cat(rows) if offsets[-1] > 0 else torch.zeros(0, dtype=torch.int64)
    return Topology(offsets, targets)


CASES = [
    pytest.param(_topology(6, 10, 1.0, 0), 10, 4, id="dense_6x10_G4_ragged_last_block"),
    pytest.param(_topology(20, 37, 0.15, 1), 37, 8, id="sparse_20x37_G8"),
    pytest.param(_topology(9, 25, 0.05, 2), 25, 5, id="sparse_with_empty_rows_G5"),
    pytest.param(_topology(5, 7, 0.5, 3), 7, 16, id="single_block_G_ge_targets"),
    pytest.param(_topology(4, 6, 0.0, 4), 6, 4, id="no_links_at_all"),
]


@pytest.mark.parametrize("topo,n_tgt,group_size", CASES)
def test_blocked_layout_faithful_to_canonical_csr(topo: Topology, n_tgt: int, group_size: int):
    bt = topo.blocked(group_size, n_tgt)
    n_src = len(topo.source_offsets) - 1
    m = len(topo)

    assert bt.offsets.shape == (-(-n_tgt // group_size), n_src + 1)
    assert int(bt.block_base[-1]) == m
    assert sorted(bt.perm.tolist()) == list(range(m)), "perm must be a permutation of every link"
    assert m == 0 or int(bt.target_local.max()) < group_size

    canon_src = torch.repeat_interleave(torch.arange(n_src), topo.source_offsets.diff())
    for b in range(bt.n_blocks):
        sl = bt.block_slice(b)
        # the block's links are exactly the canonical links whose target it owns
        owned = (topo.target_indices // group_size) == b
        assert sorted(bt.perm[sl].tolist()) == sorted(torch.nonzero(owned).flatten().tolist())
        assert (b * group_size + bt.target_local[sl].to(torch.int64) == topo.target_indices[bt.perm[sl]]).all()
        for s in range(n_src):
            lo, hi = int(bt.offsets[b, s]), int(bt.offsets[b, s + 1])
            expect = sorted(torch.nonzero(owned & (canon_src == s)).flatten().tolist())
            assert sorted(bt.perm[lo:hi].tolist()) == expect
            assert sl.start <= lo <= hi <= sl.stop


@pytest.mark.parametrize("topo,n_tgt,group_size", CASES)
def test_to_blocked_from_blocked_roundtrip(topo: Topology, n_tgt: int, group_size: int):
    bt = topo.blocked(group_size, n_tgt)
    w = torch.rand(len(topo))
    assert torch.equal(bt.from_blocked(bt.to_blocked(w)), w)


# ------------------------------------------------------------------ delay bucketing
#
# blocked(bucket=...) sorts each (block, source) row by a given per-link
# bucket id; delay_buckets() computes those ids from per-link delays.


def test_no_buckets_is_the_plain_layout():
    topo = _topology(6, 10, 1.0, 0)
    bt = topo.blocked(4, 10)
    assert bt.offsets.ndim == 2
    assert bt.n_buckets == 1
    assert bt.bucket is None


def test_bucket_ids_must_cover_every_link():
    topo = _topology(6, 10, 1.0, 0)
    with pytest.raises(ValueError, match="one entry per link"):
        topo.blocked(4, 10, bucket=torch.zeros(len(topo) - 1, dtype=torch.int64))


def _bucket_links(bt, b, s, bucket):
    """Blocked positions of block b's links out of source s in `bucket`."""
    lo, hi = int(bt.offsets[b, s]), int(bt.offsets[b, s + 1])
    return [k for k in range(lo, hi) if int(bt.bucket[k]) == bucket]


def test_one_source_four_links_land_in_four_distinct_buckets():
    # 1 block (group_size=4, 4 targets), one source neuron with 4 outgoing
    # links to targets 0..3 in buckets [0, 1, 2, 3].
    topo = Topology(torch.tensor([0, 4]), torch.tensor([0, 1, 2, 3]))

    bt = topo.blocked(4, 4, bucket=torch.tensor([0, 1, 2, 3]))

    assert bt.offsets.shape == (1, 2)  # same table as unbucketed: n_blocks=1, n_src+1=2
    assert bt.n_buckets == 4
    assert bt.bucket.dtype == torch.int16
    for bucket, expected_target in enumerate([0, 1, 2, 3]):
        links = _bucket_links(bt, 0, 0, bucket)
        assert len(links) == 1, f"bucket {bucket} should hold exactly one link"
        assert int(bt.target_local[links[0]]) == expected_target


def test_multiple_links_share_a_bucket_and_rows_are_sorted_by_bucket():
    topo = Topology(torch.tensor([0, 4]), torch.tensor([0, 1, 2, 3]))

    bt = topo.blocked(4, 4, bucket=torch.tensor([0, 2, 0, 2]))

    def bucket_targets(bucket):
        return sorted(int(bt.target_local[k]) for k in _bucket_links(bt, 0, 0, bucket))

    assert bucket_targets(0) == [0, 2]
    assert bucket_targets(1) == []
    assert bucket_targets(2) == [1, 3]
    assert bt.bucket.tolist() == [0, 0, 2, 2]
    assert int(bt.block_base[0]) == 0
    assert int(bt.block_base[1]) == 4


@pytest.mark.parametrize("topo,n_tgt,group_size", CASES)
def test_bucketed_perm_is_still_a_full_permutation_faithful_to_canonical_csr(
    topo: Topology, n_tgt: int, group_size: int
):
    n_buckets = 4
    torch.manual_seed(0)
    bucket = torch.randint(0, n_buckets, (len(topo),))

    bt = topo.blocked(group_size, n_tgt, bucket=bucket)
    n_src = len(topo.source_offsets) - 1
    m = len(topo)

    assert bt.offsets.shape == (-(-n_tgt // group_size), n_src + 1)
    assert torch.equal(bt.offsets, topo.blocked(group_size, n_tgt).offsets)
    assert int(bt.block_base[-1]) == m
    assert sorted(bt.perm.tolist()) == list(range(m))
    assert torch.equal(bt.bucket.to(torch.int64), bucket[bt.perm.to(torch.int64)])

    canon_src = torch.repeat_interleave(torch.arange(n_src), topo.source_offsets.diff())
    for b in range(bt.n_blocks):
        owned = (topo.target_indices // group_size) == b
        for s in range(n_src):
            lo, hi = int(bt.offsets[b, s]), int(bt.offsets[b, s + 1])
            row_buckets = bt.bucket[lo:hi].tolist()
            assert row_buckets == sorted(row_buckets)
            for bk in range(n_buckets):
                expect = sorted(torch.nonzero(owned & (canon_src == s) & (bucket == bk)).flatten().tolist())
                got = sorted(int(bt.perm[k]) for k in _bucket_links(bt, b, s, bk))
                assert got == expect


# ------------------------------------------------------------------ delay_buckets


def test_resolution_grid_is_anchored_at_the_minimum_delay():
    # residuals above d_min = 0.15: 0, 0.05, 0.12, 0.31 -> grid of 0.1
    db = delay_buckets(torch.tensor([0.15, 0.2, 0.27, 0.46]), resolution=0.1)
    assert db.d_min == pytest.approx(0.15)
    assert db.bucket.tolist() == [0, 0, 1, 2]  # 0.0x -> 0, 0.12 -> 1, 0.31 -> 3, renumbered to 2
    assert db.mean.tolist() == pytest.approx([0.175, 0.27, 0.46])
    assert db.resolution == 0.1


def test_empty_grid_buckets_are_dropped_and_renumbered():
    db = delay_buckets(torch.tensor([1.0, 1.05, 1.9]), resolution=0.1)
    assert db.num_buckets == 2
    assert db.bucket.tolist() == [0, 0, 1]


def test_num_buckets_gives_equal_count_buckets():
    delay = torch.linspace(0.1, 5.0, 1000)
    db = delay_buckets(delay, num_buckets=8)
    assert db.num_buckets == 8
    counts = torch.bincount(db.bucket)
    assert counts.min() >= 120 and counts.max() <= 130
    # buckets are ordered by delay and their means increase
    assert torch.all(db.mean[1:] > db.mean[:-1])
    assert db.resolution == pytest.approx((5.0 - 0.1) / 8)


def test_equal_delays_always_share_a_bucket():
    delay = torch.tensor([0.5] * 50 + [1.0] * 10 + [2.0] * 40)
    db = delay_buckets(delay, num_buckets=10)
    for value in (0.5, 1.0, 2.0):
        assert len(set(db.bucket[delay == value].tolist())) == 1
    assert db.num_buckets == 3


def test_homogeneous_delay_is_one_bucket():
    db = delay_buckets(torch.full((20,), 1.5), resolution=0.1)
    assert db.num_buckets == 1
    assert db.mean.tolist() == pytest.approx([1.5])


def test_exactly_one_of_resolution_or_num_buckets():
    with pytest.raises(ValueError, match="exactly one"):
        delay_buckets(torch.rand(5))
    with pytest.raises(ValueError, match="exactly one"):
        delay_buckets(torch.rand(5), resolution=0.1, num_buckets=4)
