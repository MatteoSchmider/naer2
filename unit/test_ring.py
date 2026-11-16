"""naer2/ring.py's host side: the layout constraints a Ring enforces."""

import pytest
import torch

from naer2.ring import Ring


def test_run_capacity_must_be_a_multiple_of_4():
    with pytest.raises(ValueError, match="multiples of 4"):
        Ring(1, 8, 0.1, "cpu", main_cap=6)


def test_cells_and_flags_start_empty():
    ring = Ring(3, 8, 0.1, "cpu", main_cap=8, side_cap=4)
    assert ring.cell(2, 9) == 2 * 8 + 1
    assert ring.main_count.numel() == ring.side_count.numel() == 24
    assert ring.flags.tolist() == [0, 0, 0, 0] and float(ring.watermark) == float("-inf")
    ring.check()


def test_growing_keeps_every_record_and_clamps_overflowing_counts():
    torch.manual_seed(0)
    ring = Ring(3, 8, 0.1, "cpu", main_cap=8, side_cap=4)
    ring.buf[:4 * 3 * 8 * (8 + 4)] = torch.randint(0, 1000, (4 * 3 * 8 * (8 + 4),), dtype=torch.int32)
    ring.main_count.copy_(torch.randint(0, 12, (24,), dtype=torch.int32))  # some beyond the capacity: lost claims
    ring.side_count.copy_(torch.randint(0, 5, (24,), dtype=torch.int32))
    ring.watermark.fill_(1.5)
    ring.flags.copy_(torch.tensor([1, 0, 0, 0], dtype=torch.int32))
    ring.ring_pos = 17
    main_before, side_before = [f.clone() for f in ring.main], [f.clone() for f in ring.side]
    counts = ring.main_count.clone()
    ring.grow(main_cap=16, side_cap=8)
    assert (ring.main_cap, ring.side_cap, ring.ring_pos) == (16, 8, 17)
    assert torch.equal(ring.main_count, counts.clamp(max=8))
    assert float(ring.watermark) == 1.5 and ring.flags.tolist() == [1, 0, 0, 0]
    for before, after, old_cap, new_cap in ((main_before, ring.main, 8, 16), (side_before, ring.side, 4, 8)):
        for f_before, f_after in zip(before, after):  # per cell: its first old_cap records unchanged
            assert torch.equal(f_after.view(24, new_cap)[:, :old_cap], f_before.view(24, old_cap))
    with pytest.raises(ValueError, match="only grow"):
        ring.grow(main_cap=12)
