"""
Unit tests for naer2.topology: delay_buckets (a connection's static delay
quantization; the simulation form is in unit/test_simulation_topology.py).

Pure torch, no Triton, no network.
"""

import pytest
import torch

from naer2.topology import delay_buckets


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
