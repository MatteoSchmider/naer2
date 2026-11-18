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
    assert ring.flags.tolist() == [0] * 5 and float(ring.watermark) == float("-inf")
    ring.check()


def test_growing_keeps_every_record_and_clamps_overflowing_counts():
    torch.manual_seed(0)
    ring = Ring(3, 8, 0.1, "cpu", main_cap=8, side_cap=4)
    for rec in ring._records:  # every main and side record
        rec.copy_(torch.randint(0, 1000, rec.shape, dtype=torch.int32))
    ring.main_count.copy_(torch.randint(0, 12, (24,), dtype=torch.int32))  # some beyond the capacity: lost claims
    ring.side_count.copy_(torch.randint(0, 5, (24,), dtype=torch.int32))
    ring.watermark.fill_(1.5)
    ring.flags.copy_(torch.tensor([1, 0, 0, 0, 0], dtype=torch.int32))
    ring.ring_pos = 17
    main_before, side_before = [f.clone() for f in ring.main], [f.clone() for f in ring.side]
    counts = ring.main_count.clone()
    ring.grow(main_cap=16, side_cap=8)
    assert (ring.main_cap, ring.side_cap, ring.ring_pos) == (16, 8, 17)
    assert torch.equal(ring.main_count, counts.clamp(max=8))
    assert float(ring.watermark) == 1.5 and ring.flags.tolist() == [1, 0, 0, 0, 0]
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


def _pooled_ring_in_use():
    """Two groups of 3 slots, 4-record blocks, 8 per group (reserve: 6): cells 0 and 4 hold a block past
    their reserve (ids 7 and 6), cell 1 a side block; records random, counts within what they hold."""
    torch.manual_seed(1)
    ring = Ring(2, 3, 0.1, "cpu", pool=(4, 8))
    ring._records[0].copy_(torch.randint(0, 1000, ring._records[0].shape, dtype=torch.int32))
    ring.table[0, 2], ring.table[4, 2] = 7, 6
    ring.nblk[0], ring.nblk[4] = 3, 3
    ring.side_table[1, 0] = 6
    ring.main_count.copy_(torch.tensor([10, 5, 8, 0, 12, 3], dtype=torch.int32))
    ring.side_count[1] = 2
    ring.top[0] = 0  # group 0: every block in use (7 held, 6 in a side run)
    ring.stack[1, :1], ring.top[1] = torch.tensor([7], dtype=torch.int32), 1  # group 1: 6 held
    return ring


def _runs(ring):
    """Every run's records, through the views (main: first count per cell; side: first side count)."""
    main = [f.view(6, -1) for f in ring.main]
    side = [f.view(6, -1) for f in ring.side]
    return ([[f[c, :int(ring.main_count[c])].tolist() for f in main] for c in range(6)],
            [[f[c, :int(ring.side_count[c])].tolist() for f in side] for c in range(6)])


def _free_stacks_are_the_unused_ids(ring):
    for grp in range(ring.n_groups):
        cells = range(grp * ring.n_slots, (grp + 1) * ring.n_slots)
        used = {int(ring.table[c, i]) for c in cells for i in range(int(ring.nblk[c]))}
        used |= {int(ring.side_table[c, i]) for c in cells for i in range(-(-int(ring.side_count[c]) // 4))}
        free = ring.stack[grp, :int(ring.top[grp])].tolist()
        assert sorted(free) == sorted(set(range(ring.pool_nb)) - used) and len(set(free)) == len(free)


@pytest.mark.parametrize("pool_nb", [16, 9], ids=["grow", "shrink"])
def test_resizing_a_pooled_ring_keeps_every_run(pool_nb):
    """Growing or shrinking the arenas (and the logs) keeps every run's records where its table points, the
    reserve in place, the free stacks exactly the unused ids."""
    ring = _pooled_ring_in_use()
    before = _runs(ring)
    nblk, top_used = ring.nblk.clone(), [8 - int(t) for t in ring.top]
    tables = ring.table.clone(), ring.side_table.clone()
    ring.resize(pool_nb=pool_nb, log_cap=6)
    if pool_nb > 8:  # growing: every block keeps its id
        assert torch.equal(ring.table[:, :8], tables[0]) and torch.equal(ring.side_table[:, :8], tables[1])
        assert bool((ring.table[:, 8:] == -1).all())
    assert (ring.pool_nb, ring.log_cap, ring.arena) == (pool_nb, 6, 4 * pool_nb)
    assert _runs(ring) == before
    assert torch.equal(ring.nblk, nblk) and int(ring.lost[0]) == 0 and int(ring.flags[0]) == 0
    assert ring.table[:, :2].flatten().tolist() == list(range(6)) * 2  # the reserve keeps its ids
    assert [pool_nb - int(t) for t in ring.top] == top_used
    _free_stacks_are_the_unused_ids(ring)
    assert len(ring.log_rec) == 6 * 6


def test_shrinking_below_the_blocks_in_use_is_flagged():
    ring = _pooled_ring_in_use()
    ring.resize(pool_nb=7)  # group 0 uses 8 blocks
    assert int(ring.lost[0]) == 1 and int(ring.flags[0]) == 1
    assert int(ring.table[0, 2]) == -1 or int(ring.side_table[1, 0]) == -1  # one of its past-reserve blocks


@pytest.mark.parametrize("pool", [None, (4, 40)], ids=["cells", "pooled"])
def test_the_header_holds_every_region_s_offset(pool):
    """_ring reads the layout from the buffer's header: it must name exactly where Ring's own views are."""
    ring = Ring(3, 8, 0.1, "cpu", main_cap=8, side_cap=4, pool=pool)
    h = ring.buf[:32].tolist()
    at = lambda t: t.storage_offset()
    assert h[0] == at(ring._records[0]) and h[1] == at(ring._records[1])
    assert (h[2], h[3], h[4], h[5]) == (at(ring.main_count), at(ring.side_count), at(ring.watermark), at(ring.flags))
    assert all(x % 4 == 0 for x in (h[0], h[1]))  # records on 16-byte boundaries
    assert h[6] == at(ring.lost) and h[7] == at(ring.stats) and h[8] == at(ring.done)
    assert (h[16], h[17]) == (ring.main_cap, ring.side_cap)
    assert at(ring.status) == h[5] and len(ring.status) == 5 + 1 + 2 * 3 + 3  # flags, lost, stats, groups done
    if pool:
        assert (h[9], h[10], h[11], h[12], h[13]) == (at(ring.table), at(ring.side_table), at(ring.stack),
                                                      at(ring.top), at(ring.nblk))
        assert h[14] == (at(ring.log_rec) - h[0]) // 4 and at(ring.log_rec) % 4 == 0
        assert h[15] == pool_log_cap(4, 40, 8) and len(ring.log_rec) == 24 * h[15] and h[18] == 40
