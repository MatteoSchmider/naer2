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

from naer2.topology import Topology


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
# Scope: a connection whose whole delay spread fits within one population
# timestep (TODO.md's "intra-timestep transmission delay" -- the inter-timestep case,
# delay >= timestep, is deliberately out of scope here and just clamps into
# the last bucket rather than erroring).


def test_default_n_buckets_is_unchanged_from_plain_blocked():
    topo = _topology(6, 10, 1.0, 0)
    bt_plain = topo.blocked(4, 10)
    bt_default = topo.blocked(4, 10, n_buckets=1)
    assert bt_default.offsets.ndim == 2
    assert torch.equal(bt_default.offsets, bt_plain.offsets)
    assert torch.equal(bt_default.perm, bt_plain.perm)
    assert bt_default.n_buckets == 1
    assert bt_default.bucket_width is None


def test_n_buckets_without_delay_or_timestep_raises():
    topo = _topology(6, 10, 1.0, 0)
    with pytest.raises(ValueError, match="n_buckets"):
        topo.blocked(4, 10, n_buckets=4)
    with pytest.raises(ValueError, match="n_buckets"):
        topo.blocked(4, 10, delay=torch.zeros(len(topo)), n_buckets=4)


def test_one_source_four_links_land_in_four_distinct_buckets():
    # 1 block (group_size=4, 4 targets), one source neuron with 4 outgoing
    # links to targets 0..3, delays [0.1, 0.4, 0.6, 0.9], timestep=1.0,
    # n_buckets=4 -> bucket_width=0.25 -> buckets [0, 1, 2, 3].
    topo = Topology(torch.tensor([0, 4]), torch.tensor([0, 1, 2, 3]))
    delay = torch.tensor([0.1, 0.4, 0.6, 0.9])

    bt = topo.blocked(4, 4, delay=delay, n_buckets=4, timestep=1.0)

    assert bt.offsets.shape == (1, 4, 2)  # n_blocks=1, n_buckets=4, n_src+1=2
    assert bt.bucket_width == 0.25
    for bucket, expected_target in enumerate([0, 1, 2, 3]):
        lo, hi = int(bt.offsets[0, bucket, 0]), int(bt.offsets[0, bucket, 1])
        assert hi - lo == 1, f"bucket {bucket} should hold exactly one link"
        assert int(bt.target_local[lo]) == expected_target


def test_multiple_links_share_a_bucket_and_bucket_order_is_ascending():
    # Same single block, one source, 4 links, delays grouped two-and-two
    # into buckets 0 and 2 (bucket_width=0.25): [0.05, 0.1] -> bucket 0,
    # [0.55, 0.6] -> bucket 2. Buckets 1 and 3 stay empty.
    topo = Topology(torch.tensor([0, 4]), torch.tensor([0, 1, 2, 3]))
    delay = torch.tensor([0.05, 0.55, 0.1, 0.6])

    bt = topo.blocked(4, 4, delay=delay, n_buckets=4, timestep=1.0)

    def bucket_targets(bucket):
        lo, hi = int(bt.offsets[0, bucket, 0]), int(bt.offsets[0, bucket, 1])
        return sorted(bt.target_local[lo:hi].tolist())

    assert bucket_targets(0) == [0, 2]  # targets of delay 0.05 and 0.1
    assert bucket_targets(1) == []
    assert bucket_targets(2) == [1, 3]  # targets of delay 0.55 and 0.6
    assert bucket_targets(3) == []
    # block_base still spans the whole block across all its buckets
    assert int(bt.block_base[0]) == 0
    assert int(bt.block_base[1]) == 4


def test_delay_at_or_beyond_timestep_clamps_into_the_last_bucket():
    topo = Topology(torch.tensor([0, 2]), torch.tensor([0, 1]))
    delay = torch.tensor([1.0, 5.0])  # both >= timestep=1.0

    bt = topo.blocked(4, 4, delay=delay, n_buckets=4, timestep=1.0)

    lo, hi = int(bt.offsets[0, 3, 0]), int(bt.offsets[0, 3, 1])
    assert sorted(bt.target_local[lo:hi].tolist()) == [0, 1]
    for bucket in range(3):
        lo, hi = int(bt.offsets[0, bucket, 0]), int(bt.offsets[0, bucket, 1])
        assert hi == lo


@pytest.mark.parametrize("topo,n_tgt,group_size", CASES)
def test_bucketed_perm_is_still_a_full_permutation_faithful_to_canonical_csr(
    topo: Topology, n_tgt: int, group_size: int
):
    n_buckets = 4
    timestep = 1.0
    torch.manual_seed(0)
    delay = torch.rand(len(topo)) * timestep  # stay within scope: all < timestep

    bt = topo.blocked(group_size, n_tgt, delay=delay, n_buckets=n_buckets, timestep=timestep)
    n_src = len(topo.source_offsets) - 1
    m = len(topo)

    assert bt.offsets.shape == (-(-n_tgt // group_size), n_buckets, n_src + 1)
    assert int(bt.block_base[-1]) == m
    assert sorted(bt.perm.tolist()) == list(range(m))

    canon_src = torch.repeat_interleave(torch.arange(n_src), topo.source_offsets.diff())
    bucket = (delay / (timestep / n_buckets)).floor().clamp(0, n_buckets - 1).to(torch.int64)
    for b in range(bt.n_blocks):
        owned = (topo.target_indices // group_size) == b
        for bk in range(n_buckets):
            for s in range(n_src):
                lo, hi = int(bt.offsets[b, bk, s]), int(bt.offsets[b, bk, s + 1])
                expect = sorted(torch.nonzero(owned & (canon_src == s) & (bucket == bk)).flatten().tolist())
                assert sorted(bt.perm[lo:hi].tolist()) == expect
