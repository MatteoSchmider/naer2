"""naer2/ring.py's host side: the layout constraints a Ring enforces."""

import pytest
import torch

from naer2.ring import POOL_RESERVE, Ring, pool_log_cap


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
    for rec in ring._records:  # every main and side record
        rec.copy_(torch.randint(0, 1000, rec.shape, dtype=torch.int32))
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


def test_a_pooled_ring_starts_with_its_reserve_per_main_run():
    """Every main run holds POOL_RESERVE blocks (slot s of a group: blocks
    s * R .. (s + 1) * R), the rest of every group's arena is free; side
    runs hold none."""
    R = POOL_RESERVE
    ring = Ring(3, 8, 0.1, "cpu", pool=(4, 8 * R + 2))
    assert (ring.main_cap, ring.side_cap, ring.arena) == ((8 * R + 2) * 4,) * 3  # a run can take the whole arena
    assert ring.table[:, :R].flatten().tolist() == list(range(8 * R)) * 3 and bool((ring.table[:, R:] == -1).all())
    assert ring.nblk.tolist() == [R] * 24 and bool((ring.side_table == -1).all())
    assert ring.top.tolist() == [2, 2, 2] and ring.stack[1, :2].tolist() == [8 * R, 8 * R + 1]
    with pytest.raises(ValueError, match="more blocks than slots"):
        Ring(3, 8, 0.1, "cpu", pool=(4, 8 * R))


def test_growing_a_pooled_ring_keeps_records_tables_and_free_blocks():
    """Two groups: some blocks in use (in main and side tables, one missing:
    -1), the rest free; doubling the arena keeps every run's records where
    its table points and adds the new blocks to the free stacks."""
    torch.manual_seed(1)
    ring = Ring(2, 3, 0.1, "cpu", pool=(4, 8))
    ring._records[0].copy_(torch.randint(0, 1000, ring._records[0].shape, dtype=torch.int32))
    ring.table[0, :2] = torch.tensor([3, 1], dtype=torch.int32)  # group 0, slot 0: blocks 3, 1
    ring.side_table[1, 0] = 0  # group 0, slot 1: side block 0
    ring.table[4, :2] = torch.tensor([2, -1], dtype=torch.int32)  # group 1, slot 1: block 2, one lost
    ring.main_count[0], ring.side_count[1], ring.main_count[4] = 7, 2, 6
    ring.stack[0, :1] = torch.tensor([2], dtype=torch.int32)
    ring.top[0] = 1  # group 0: block 2 free
    ring.stack[1, :3] = torch.tensor([0, 1, 3], dtype=torch.int32)
    ring.top[1] = 3  # group 1: blocks 0, 1, 3 free
    nblk = ring.nblk.clone()
    main_before, side_before = [f.clone() for f in ring.main], [f.clone() for f in ring.side]
    ring.grow(pool_nb=16)
    assert (ring.pool_nb, ring.arena) == (16, 64)
    for before, after in ((main_before, ring.main), (side_before, ring.side)):
        for f_before, f_after in zip(before, after):  # per cell: its first 32 records' views unchanged
            assert torch.equal(f_after.view(6, 64)[:, :32], f_before.view(6, 32))
    assert ring.table[0, :2].tolist() == [3, 1] and ring.table[4, :2].tolist() == [2, -1]
    assert torch.equal(ring.nblk, nblk) and ring.top.tolist() == [9, 11]
    assert sorted(ring.stack[0, :9].tolist()) == [2] + list(range(8, 16))
    assert sorted(ring.stack[1, :11].tolist()) == [0, 1, 3] + list(range(8, 16))
    assert (ring.main_count[0], ring.side_count[1], ring.main_count[4]) == (7, 2, 6)
    with pytest.raises(ValueError, match="arena"):
        ring.grow(pool_nb=8)


@pytest.mark.parametrize("pool", [None, (4, 40)], ids=["cells", "pooled"])
def test_the_header_holds_every_region_s_offset(pool):
    """_ring reads the layout from the buffer's header: it must name exactly where Ring's own views are."""
    ring = Ring(3, 8, 0.1, "cpu", main_cap=8, side_cap=4, pool=pool)
    h = ring.buf[:16].tolist()
    at = lambda t: t.storage_offset()
    assert h[0] == at(ring._records[0]) and h[1] == at(ring._records[1])
    assert (h[2], h[3], h[4], h[5]) == (at(ring.main_count), at(ring.side_count), at(ring.watermark), at(ring.flags))
    assert all(x % 4 == 0 for x in (h[0], h[1]))  # records on 16-byte boundaries
    if pool:
        assert (h[6], h[7], h[8], h[9], h[10], h[11]) == (at(ring.table), at(ring.side_table), at(ring.stack),
                                                          at(ring.top), at(ring.nblk), at(ring.lost))
        assert h[12] == (at(ring.log_rec) - h[0]) // 4 and at(ring.log_rec) % 4 == 0
        assert h[13] == pool_log_cap(4, 40, 8) and len(ring.log_rec) == 24 * h[13]
