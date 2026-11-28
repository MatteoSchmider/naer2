"""naer2/ring.py's host side: the layout constraints a Ring enforces."""

import pytest
import torch

from naer2.ring import RESERVE, Ring, default_log_cap


def test_blocks_are_multiples_of_4_and_an_arena_more_than_its_reserves():
    with pytest.raises(ValueError, match="multiple of 4"):
        Ring(1, 8, 0.1, "cpu", arena=(6, 40))
    with pytest.raises(ValueError, match="more blocks than its slots' reserves"):
        Ring(3, 8, 0.1, "cpu", arena=(4, 8 * RESERVE))


def test_cells_and_flags_start_empty():
    ring = Ring(3, 8, 0.1, "cpu", arena=(4, 8 * RESERVE + 2))
    assert ring.cell(2, 9) == 2 * 8 + 1
    assert ring.run_count.numel() == ring.head.numel() == 24
    assert not ring.run_count.any() and not ring.head.any()
    assert ring.flags.tolist() == [0] * 4
    ring.check()


def test_every_slot_s_run_starts_with_its_reserve():
    """Every run holds RESERVE blocks (slot s of a group: blocks s * R .. (s + 1) * R), the rest of every
    group's arena is free; a run can take the whole arena."""
    R = RESERVE
    ring = Ring(3, 8, 0.1, "cpu", arena=(4, 8 * R + 2))
    assert ring.run_cap == ring.arena == (8 * R + 2) * 4
    assert ring.table[:, :R].flatten().tolist() == list(range(8 * R)) * 3 and bool((ring.table[:, R:] == -1).all())
    assert ring.nblk.tolist() == [R] * 24
    assert ring.top.tolist() == [2, 2, 2] and ring.stack[1, :2].tolist() == [8 * R, 8 * R + 1]


def _ring_in_use():
    """Two groups of 3 slots, 4-record blocks, 8 per group (reserve: 6): cells 0 and 1 hold a block past their
    reserve (ids 7 and 6: group 0 uses every block), cell 4 one (id 6); records random, counts within what they
    hold."""
    torch.manual_seed(1)
    ring = Ring(2, 3, 0.1, "cpu", arena=(4, 8))
    ring._records.copy_(torch.randint(0, 1000, ring._records.shape, dtype=torch.int32))
    ring.table[0, 2], ring.table[1, 2], ring.table[4, 2] = 7, 6, 6
    ring.nblk[0], ring.nblk[1], ring.nblk[4] = 3, 3, 3
    ring.run_count.copy_(torch.tensor([10, 9, 8, 0, 12, 3], dtype=torch.int32))
    ring.top[0] = 0  # group 0: every block in use
    ring.stack[1, :1], ring.top[1] = torch.tensor([7], dtype=torch.int32), 1  # group 1: 7 free
    return ring


def _runs(ring):
    """Every run's records, through the view (its first count per cell)."""
    runs = [f.view(6, -1) for f in ring.runs]
    return [[f[c, :int(ring.run_count[c])].tolist() for f in runs] for c in range(6)]


def _free_stacks_are_the_unused_ids(ring):
    for grp in range(ring.n_groups):
        cells = range(grp * ring.n_slots, (grp + 1) * ring.n_slots)
        used = {int(ring.table[c, i]) for c in cells for i in range(int(ring.nblk[c]))}
        free = ring.stack[grp, :int(ring.top[grp])].tolist()
        assert sorted(free) == sorted(set(range(ring.n_blocks)) - used) and len(set(free)) == len(free)


@pytest.mark.parametrize("n_blocks", [16, 9], ids=["grow", "shrink"])
def test_resizing_keeps_every_run(n_blocks):
    """Growing or shrinking the arenas (and the logs) keeps every run's records where its table points, the
    reserve in place, the free stacks exactly the unused ids."""
    ring = _ring_in_use()
    before = _runs(ring)
    nblk, top_used = ring.nblk.clone(), [8 - int(t) for t in ring.top]
    table = ring.table.clone()
    ring.resize(n_blocks=n_blocks, log_cap=6)
    if n_blocks > 8:  # growing: every block keeps its id
        assert torch.equal(ring.table[:, :8], table) and bool((ring.table[:, 8:] == -1).all())
    assert (ring.n_blocks, ring.log_cap, ring.arena) == (n_blocks, 6, 4 * n_blocks)
    assert _runs(ring) == before
    assert torch.equal(ring.nblk, nblk) and int(ring.lost[0]) == 0 and int(ring.flags[0]) == 0
    assert ring.table[:, :2].flatten().tolist() == list(range(6)) * 2  # the reserve keeps its ids
    assert [n_blocks - int(t) for t in ring.top] == top_used
    _free_stacks_are_the_unused_ids(ring)
    assert len(ring.log_rec) == 6 * 6


def test_shrinking_below_the_blocks_in_use_is_flagged():
    ring = _ring_in_use()
    ring.resize(n_blocks=7)  # group 0 uses 8 blocks
    assert int(ring.lost[0]) == 1 and int(ring.flags[0]) == 1
    assert int(ring.table[0, 2]) == -1 or int(ring.table[1, 2]) == -1  # one of its past-reserve blocks


def test_the_header_holds_every_region_s_offset():
    """_ring reads the layout from the buffer's header: it must name exactly where Ring's own views are."""
    ring = Ring(3, 8, 0.1, "cpu", arena=(4, 40))
    h = ring.buf[:32].tolist()
    at = lambda t: t.storage_offset()
    assert h[0] == at(ring._records) and h[0] % 4 == 0  # records on 16-byte boundaries
    assert (h[1], h[2], h[3], h[4]) == (at(ring.run_count), at(ring.head), at(ring.flags), at(ring.lost))
    assert (h[5], h[6]) == (at(ring.stats), at(ring.done))
    assert (h[7], h[8], h[9], h[10]) == (at(ring.table), at(ring.stack), at(ring.top), at(ring.nblk))
    assert at(ring.status) == h[3] and len(ring.status) == 4 + 1 + 2 * 3 + 3  # flags, lost, stats, groups done
    assert h[11] == (at(ring.log_rec) - h[0]) // 4 and at(ring.log_rec) % 4 == 0
    assert h[12] == default_log_cap(4, 40, 8) and len(ring.log_rec) == 24 * h[12]
    assert (h[13], h[14]) == (ring.run_cap, 40)
