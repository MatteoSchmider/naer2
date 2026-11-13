"""naer2/greenfield/ring.py's host side: the layout constraints a Ring enforces."""

import pytest

from naer2.greenfield.ring import Ring


def test_run_capacity_must_be_a_multiple_of_4():
    with pytest.raises(ValueError, match="multiples of 4"):
        Ring(1, 8, 0.1, "cpu", main_cap=6)


def test_cells_and_flags_start_empty():
    ring = Ring(3, 8, 0.1, "cpu", main_cap=8, side_cap=4)
    assert ring.cell(2, 9) == 2 * 8 + 1
    assert ring.main_count.numel() == ring.side_count.numel() == 24
    assert ring.flags.tolist() == [0, 0, 0] and float(ring.watermark) == float("-inf")
    ring.check()
