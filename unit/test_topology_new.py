"""
naer2.topology_new.TopologyNew: canonical topology -> simulation layout
(source, group, bucket segments). Pure torch on the CPU: every segment must
be exactly one non-empty (source, group, bucket) piece of the canonical
links, in that order, with target order kept inside; attributes must round
trip through the simulation order.
"""

import torch
import pytest

from naer2.topology import Topology
from naer2.topology_new import TopologyNew


def _random_topology(n_src, n_tgt, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = [torch.sort(torch.randperm(n_tgt, generator=g)[: torch.randint(0, k + 1, (1,), generator=g).item()]).values
            for _ in range(n_src)]
    offsets = torch.tensor([0] + [len(r) for r in rows]).cumsum(0)
    return Topology(offsets, torch.cat(rows) if rows else torch.zeros(0, dtype=torch.int64))


def _expected_segments(topo, n_groups_size, bucket):
    """Canonical links grouped into (source, group, bucket) pieces."""
    src = torch.repeat_interleave(torch.arange(len(topo.source_offsets) - 1), topo.source_offsets.diff())
    pieces = {}
    for i, (s, t, b) in enumerate(zip(src.tolist(), topo.target_indices.tolist(), bucket.tolist())):
        pieces.setdefault((s, t // n_groups_size, b), []).append(i)
    return pieces


@pytest.mark.parametrize("group_size", [4, 16, 64])
def test_segments_are_exactly_the_non_empty_source_group_bucket_pieces(group_size):
    topo = _random_topology(30, 100, 20)
    delay = torch.rand(len(topo), generator=torch.Generator().manual_seed(1)) * 1.0 + 0.1
    t = TopologyNew(topo, 100, group_size, "cpu", delay=delay, resolution=0.2)
    bucket = torch.floor((delay.double() - delay.double().min()) / 0.2 + 1e-9).long()
    _, bucket = torch.unique(bucket, return_inverse=True)
    want = _expected_segments(topo, group_size, bucket)

    segs = t.segments()
    assert len(segs) == t.n_segments == len(want)
    assert [s[:3] for s in segs] == sorted(want)  # ordered by (source, group, bucket)
    perm = t.perm.tolist()
    for s, g, b, lo, hi in segs:
        assert perm[lo:hi] == want[(s, g, b)]  # exactly its links, target order kept
        assert t.link_target[lo:hi].tolist() == [topo.target_indices[i].item() % group_size for i in perm[lo:hi]]
    assert t.seg_offsets[-1].item() == t.n_segments
    assert t.seg_start[-1].item() == len(topo)


def test_bucket_means_and_minimum_delay():
    topo = Topology(torch.tensor([0, 4]), torch.tensor([0, 1, 2, 3]))
    t = TopologyNew(topo, 4, 4, "cpu", delay=torch.tensor([0.15, 0.2, 0.27, 0.46]), resolution=0.1)
    assert t.d_min == pytest.approx(0.15)
    assert t.bucket_mean.tolist() == pytest.approx([0.175, 0.27, 0.46])
    assert [s[2] for s in t.segments()] == [0, 1, 2]


@pytest.mark.parametrize("delay", [None, 1.5])
def test_no_or_homogeneous_delay_is_one_segment_per_source_and_group(delay):
    topo = _random_topology(20, 64, 30, seed=2)
    t = TopologyNew(topo, 64, 16, "cpu", delay=delay)
    assert t.n_buckets == 1
    assert t.bucket_mean.tolist() == [delay or 0.0]
    assert t.d_min == (delay or 0.0)
    want = _expected_segments(topo, 16, torch.zeros(len(topo), dtype=torch.long))
    assert [s[:3] for s in t.segments()] == sorted(want)
    assert t.perm.tolist() == list(range(len(topo)))  # already in (source, group) order


def test_attributes_round_trip_through_simulation_order():
    topo = _random_topology(25, 80, 15, seed=3)
    delay = torch.rand(len(topo)) + 0.1
    weight = torch.randn(len(topo))
    t = TopologyNew(topo, 80, 8, "cpu", delay=delay, num_buckets=4, attrs={"weight": weight})
    assert torch.equal(t.read("weight"), weight)
    assert torch.equal(t.attrs["weight"], weight[t.perm.long()])  # simulation order
    t.write("weight", weight * 2)
    assert torch.equal(t.read("weight"), weight * 2)


def test_large_groups_and_targets_survive_the_packing():
    """group ids >= 32768 set bit 31 of seg_gb; targets >= 32768 wrap in int16."""
    n_tgt = 40000 * 2
    topo = Topology(torch.tensor([0, 3]), torch.tensor([1, 39999, 79999]))
    t = TopologyNew(topo, n_tgt, 2, "cpu")  # 40000 groups
    assert [s[1] for s in t.segments()] == [0, 19999, 39999]
    t2 = TopologyNew(topo, n_tgt, 65536, "cpu")
    assert (t2.link_target.to(torch.int32) & 0xFFFF).tolist() == [1, 39999, 79999 % 65536]


def test_locality_is_links_per_segment():
    topo = Topology(torch.tensor([0, 8]), torch.arange(8))
    assert TopologyNew(topo, 8, 8, "cpu").locality() == 8.0
    assert TopologyNew(topo, 8, 1, "cpu").locality() == 1.0
