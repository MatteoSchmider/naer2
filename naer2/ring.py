"""
A consumer's rings of arrival slots: where its fused kernel's pull intake
puts every link it takes in, and where delivery reads them from (naer2/
transpiler/templates/fused.py).

One ring per target GROUP (the neurons one program owns). Slot k of a
group's ring covers arrivals in [k * slot_width, (k + 1) * slot_width) and
lives at index k % N_SLOTS. Each slot holds
  - a main run: records appended in emission order, and
  - a side run: out-of-order records (emission-delayed, or below the
    watermark), kept sorted by emission time by insertion.
A record is one LINK: emission time e, (connection << 24) | source neuron,
the link's index in its connection (its attributes, e.g. the weight) and
its target within the group -- delivery goes from the record straight to
the receive. A record goes to the main runs if e >= the watermark (largest
e ever appended to a group's main runs) and it isn't emission-delayed (e
before its source's run window end); otherwise to the side runs.

An arrival before the ring position (already delivered) goes into the first
undelivered slot (late, the timestep quantization); one at or past ring
position + N_SLOTS is far future (flag 1, not implemented yet); a full main
run sets flag 0, a full side run flag 3 (its links are lost: the runtime
doubles that capacity, grow()); a uniform-emission run that would need a
side run sets flag 2.
"""

import torch
import triton
import triton.language as tl

# One ring per consumer population; n_slots covers its longest incoming delay (+ emission
#   delay) at one slot width -- the finest resolution any of its incoming connections requires (D-12; the
#   coarser connections could be optimized later); capacities per population, below
# GAP (D-1): a record's time word carries the link's arrival, e + its bucket's mean delay, once
#   the pull writes it
# Run capacities: these defaults, per population (Population.main_cap / side_cap), each doubling after an
# overflow (PopulationRuntime._grow_if_full; a burst like PD14's 902-spike thalamic volley lands in every slot of
# its delay range at once).
DEFAULT_MAIN_CAP = 8192  # links per (group, slot): PD14 at G=256 needs ~350 quiet, up to ~9.3k in its thalamic volley
DEFAULT_SIDE_CAP = 2048
_N_FLAGS = 4  # main run full, far future, uniform contract violated, side run full
# pooled rings: the blocks every main run keeps -- it starts with them, and keeps them when delivered. PD14 (256-record
# blocks): a quiet slot ends its life with ~220-360 records, so 2 blocks cover it and a quiet run hardly ever takes
# a new one (each a barrier in the pull); could follow the network's arrival rate x delay (Little's law) instead
POOL_RESERVE = 2
# pooled rings, the pull's LDS counter per slot: count | blocks held << POOL_CNT_BITS (both come back with every
# claim); so a run holds at most POOL_MAX_BLOCKS blocks and 2**POOL_CNT_BITS - 1 records
POOL_CNT_BITS = 20
POOL_MAX_BLOCKS = 2047


def pool_log_cap(pool_b: int, pool_nb: int, n_slots: int) -> int:
    """Entries of a cell's overflow log -- the records one run can put past the blocks it holds: half its
    group's arena over its slots (PD14: one cell grows by up to ~1400 records in one run of the thalamic volley,
    by ~80 in quiet ones; the log is scratch for bursts, its size a candidate for tightening)."""
    return pool_b * pool_nb // 2 // n_slots


@triton.jit
def _f32(p):
    return p.to(tl.pointer_type(tl.float32))


@triton.jit
def _ring(ring, n_groups, N_SLOTS: tl.constexpr, MAIN_CAP: tl.constexpr, SIDE_CAP: tl.constexpr,
          POOL_B: tl.constexpr, POOL_NB: tl.constexpr):
    """Typed pointers into a Ring buffer, see Ring for the layout: the four
    fields of a record -- time, source word, link index, target -- for the
    main and the side runs (POOL_B > 0: both in the groups' arenas); the
    per-cell counts; watermark and flags; then (POOL_B > 0) the cells' main
    and side block tables, the groups' free stacks and their tops. Field f
    of the record with index i (see _rec, _run_rec) is at field_ptr + 4 * i."""
    cells = n_groups * N_SLOTS
    main_time = ring
    if POOL_B > 0:
        side_time = ring  # one arena per group: main and side blocks alike
        main_count = ring + 4 * n_groups * POOL_NB * POOL_B
    else:
        side_time = ring + 4 * cells * MAIN_CAP
        main_count = side_time + 4 * cells * SIDE_CAP
    side_count = main_count + cells
    watermark = side_count + cells
    flags = watermark + 1
    table = flags + 4  # _N_FLAGS
    if POOL_B > 0:
        side_table = table + cells * POOL_NB
        stack = side_table + cells * POOL_NB
    else:
        side_table = table
        stack = table
    top = stack + n_groups * POOL_NB
    nblk = top + n_groups  # per cell: the blocks its main run holds (pooled)
    lost = nblk + cells  # set once an arena ran out: tables (and so free stacks) may hold missing blocks (-1)
    # per group and slot, the overflow log's records (fused.py, _pool_fixup) -- their index from the buffer's
    # start (a record's place is one index either way), at a word divisible by 4
    if POOL_B > 0:
        log_rec = (4 * n_groups * POOL_NB * POOL_B + 2 * cells + 5 + 2 * cells * POOL_NB + n_groups * POOL_NB
                   + n_groups + cells + 1 + 3) // 4
    else:
        log_rec = 0
    return (
        _f32(main_time), main_time + 1, main_time + 2, main_time + 3, main_count,
        _f32(side_time), side_time + 1, side_time + 2, side_time + 3, side_count,
        _f32(watermark), flags, table, side_table, stack, top, nblk, lost, log_rec,
    )


@triton.jit
def _rec(cell, pos, n_cells):
    """Record index of entry pos of a cell's run. Runs are cut into cache
    lines of 4 records and line b of every cell sits next to line b of all
    the others: the live front of all cells is one dense region (the pull's
    appends and delivery's reads stay in few lines and pages), bursts reach
    further back. Capacities are multiples of 4."""
    return ((pos >> 2) * n_cells + cell) * 4 + (pos & 3)


class Ring:
    """One consumer's rings of arrival slots, one ring per target group
    (see the module docstring), as one int32 buffer: per (group, slot) cell
    a main run of MAIN_CAP records and a side run of SIDE_CAP, each one
    16-byte record (time, (conn << 24) | source, link index, target),
    interleaved by cache line across cells (see _rec); then main counts,
    side counts, the watermark (float32) and the flags. Cell of (group g,
    slot k): g * n_slots + k % n_slots. main / side: the four fields as
    copies in run order, record pos of cell c at c * cap + pos (for
    inspection and tests)."""

    def __init__(self, n_groups: int, n_slots: int, slot_width: float, device,
                 main_cap: int = DEFAULT_MAIN_CAP, side_cap: int = DEFAULT_SIDE_CAP, pool: tuple[int, int] | None = None):
        """pool: (block, blocks) -- every group's main and side runs in one arena of `blocks` blocks of
        `block` records, a run taking blocks as it fills (through its block table) and giving them back once
        delivered; a run can grow to the whole arena (main_cap, side_cap ignored). Every main run keeps
        POOL_RESERVE blocks (it starts with them, and keeps them when delivered), so an arena needs more than
        n_slots x POOL_RESERVE blocks. None: every cell its own main_cap + side_cap records (see _rec)."""
        self.n_groups, self.n_slots, self.slot_width = n_groups, n_slots, slot_width
        self.pool_b, self.pool_nb = pool or (0, 1)
        if self.pool_b:
            if self.pool_b % 4 or self.pool_nb <= n_slots * POOL_RESERVE:
                raise ValueError("a pool block is a positive multiple of 4 records, and an arena more blocks than "
                                 "slots (each main run keeps one in reserve)")
            main_cap = side_cap = self._run_cap()
        if main_cap % 4 or side_cap % 4:
            raise ValueError("run capacities must be multiples of 4 (entries are interleaved by cache line)")
        self._allocate(main_cap, side_cap, device)
        self.watermark.fill_(float("-inf"))
        self.ring_pos = 0  # first slot not yet delivered

    def _run_cap(self) -> int:
        """Pooled: the records one run can hold -- the whole arena, within the claim counter's fields."""
        cap = min(self.pool_nb, POOL_MAX_BLOCKS) * self.pool_b
        return min(cap, ((1 << POOL_CNT_BITS) - 1) // self.pool_b * self.pool_b)

    @property
    def arena(self) -> int:
        """Records per group (pooled), else per cell (main + side)."""
        return self.pool_b * self.pool_nb if self.pool_b else self.main_cap + self.side_cap

    def _allocate(self, main_cap: int, side_cap: int, device) -> None:
        if self.pool_b:
            main_cap = side_cap = self._run_cap()
        self.main_cap, self.side_cap = main_cap, side_cap
        cells = self.n_groups * self.n_slots
        if self.pool_b:
            m, s = self.n_groups * self.pool_nb * self.pool_b, 0
            n_table, n_stack = cells * self.pool_nb, self.n_groups * self.pool_nb
            log_cap = pool_log_cap(self.pool_b, self.pool_nb, self.n_slots)
            n_top = self.n_groups + cells + 1
            # (then the log's records: 4 words each, from a word index divisible by 4 -- see _ring)
            pre = 4 * m + 2 * cells + 1 + _N_FLAGS + 2 * n_table + n_stack + n_top
            n_top += (-pre) % 4 + 4 * cells * log_cap
        else:
            m, s = cells * main_cap, cells * side_cap
            n_table = n_stack = n_top = 0
        self.buf = torch.zeros(4 * m + 4 * s + 2 * cells + 1 + _N_FLAGS + 2 * n_table + n_stack + n_top,
                               device=device, dtype=torch.int32)
        self._records = (self.buf[:4 * m].view(m, 4), self.buf[4 * m:4 * m + 4 * s].view(s, 4))
        o = 4 * m + 4 * s
        self.main_count = self.buf[o:o + cells]
        self.side_count = self.buf[o + cells:o + 2 * cells]
        self.watermark = self.buf[o + 2 * cells:o + 2 * cells + 1].view(torch.float32)
        o += 2 * cells + 1
        self.flags = self.buf[o:o + _N_FLAGS]
        o += _N_FLAGS
        if self.pool_b:
            self.table = self.buf[o:o + n_table].view(cells, self.pool_nb)
            self.side_table = self.buf[o + n_table:o + 2 * n_table].view(cells, self.pool_nb)
            o += 2 * n_table
            self.stack = self.buf[o:o + n_stack].view(self.n_groups, self.pool_nb)
            self.top = self.buf[o + n_stack:o + n_stack + self.n_groups]
            o2 = o + n_stack + self.n_groups
            self.nblk = self.buf[o2:o2 + cells]
            self.lost = self.buf[o2 + cells:o2 + cells + 1]  # see _ring
            lr = o2 + cells + 1
            lr += (-lr) % 4
            self.log_rec = self.buf[lr:lr + 4 * cells * log_cap].view(-1, 4)
            self.table.fill_(-1)
            self.side_table.fill_(-1)
            # every main run its reserved blocks (slot s of a group: blocks s * R .. (s + 1) * R), the rest free
            R = POOL_RESERVE
            self.table[:, :R] = torch.arange(self.n_slots * R, device=device, dtype=torch.int32).view(
                self.n_slots, R).repeat(self.n_groups, 1)
            self.nblk.fill_(R)
            self.stack.zero_()
            free = self.pool_nb - self.n_slots * R
            self.stack[:, :free] = torch.arange(self.n_slots * R, self.pool_nb, device=device, dtype=torch.int32)
            self.top.fill_(free)

    def grow(self, main_cap: int | None = None, side_cap: int | None = None, pool_nb: int | None = None) -> None:
        """Larger run capacities (pooled: a larger arena, pool_nb blocks per group), keeping what the ring
        holds: a record's place (_rec; pooled: its block, a group-local id) doesn't depend on the capacity,
        so the records, tables and free stacks copy over; the counts are clamped to what was written (an
        overflowing run's claims beyond its capacity were lost), the watermark and flags carried along. buf
        is a new buffer: whoever holds its address (a kernel launch's arguments) takes it again."""
        old_buf, (main_rec, side_rec) = self.buf, self._records
        main_count, side_count = self.main_count.clone(), self.side_count.clone()
        watermark, flags = self.watermark.clone(), self.flags.clone()
        old_main, old_side = self.main_cap, self.side_cap
        if self.pool_b:
            old_nb = self.pool_nb
            if pool_nb is None or pool_nb < old_nb:
                raise ValueError("a pooled ring grows by its arena: pool_nb >= the current blocks per group")
            table, side_table, stack, top = self.table.clone(), self.side_table.clone(), self.stack.clone(), \
                self.top.clone()
            nblk, lost = self.nblk.clone(), self.lost.clone()
            self.pool_nb = pool_nb
            self._allocate(0, 0, old_buf.device)
            g, b = self.n_groups, self.pool_b
            self._records[0].view(g, pool_nb * b, 4)[:, :old_nb * b] = main_rec.view(g, old_nb * b, 4)
            self.table[:, :old_nb] = table
            self.side_table[:, :old_nb] = side_table
            # every group's free blocks: the old ones still free, then the new ids
            free = torch.arange(pool_nb, device=old_buf.device, dtype=torch.int32)
            keep = torch.arange(old_nb, device=old_buf.device)[None, :] < top[:, None]
            self.stack.zero_()
            self.stack[:, :old_nb] = torch.where(keep, stack, 0)
            idx = top[:, None].long() + torch.arange(pool_nb - old_nb, device=old_buf.device)[None, :]
            self.stack.scatter_(1, idx, free[old_nb:].expand(g, -1).contiguous())
            self.top.copy_(top + (pool_nb - old_nb))
            self.nblk.copy_(nblk)
            self.lost.copy_(lost)
            self.main_count.copy_(main_count.clamp(max=old_main))
            self.side_count.copy_(side_count.clamp(max=old_side))
        else:
            main_cap, side_cap = main_cap or old_main, side_cap or old_side
            if main_cap % 4 or side_cap % 4 or main_cap < old_main or side_cap < old_side:
                raise ValueError("capacities only grow, in multiples of 4")
            self._allocate(main_cap, side_cap, old_buf.device)
            self._records[0][:len(main_rec)] = main_rec
            self._records[1][:len(side_rec)] = side_rec
            self.main_count.copy_(main_count.clamp(max=old_main))
            self.side_count.copy_(side_count.clamp(max=old_side))
        self.watermark.copy_(watermark)
        self.flags.copy_(flags)
        if old_buf.is_cuda:
            # the old buffer is released here, but the runs queued before the growth (and the copies above) on
            # this stream may still be using it: keep the allocator from reusing it until they are done (it was
            # allocated on another stream, whose pool would otherwise take it back at once)
            old_buf.record_stream(torch.cuda.current_stream(old_buf.device))

    def cell(self, group: int, slot: int) -> int:
        return group * self.n_slots + slot % self.n_slots

    def _fields(self, rec: torch.Tensor, cap: int, table=None):
        cells = self.n_groups * self.n_slots
        cell = torch.arange(cells, device=rec.device).repeat_interleave(cap)
        pos = torch.arange(cap, device=rec.device).repeat(cells)
        if self.pool_b:  # see _run_rec: through the run's block table, a missing block's records zero
            blk = table[cell, pos // self.pool_b].long()
            r = rec[((cell // self.n_slots) * self.pool_nb + blk.clamp(min=0)) * self.pool_b + pos % self.pool_b]
            r = torch.where((blk >= 0)[:, None], r, 0)
        else:
            r = rec[((pos >> 2) * cells + cell) * 4 + (pos & 3)]  # see _rec
        return r.view(torch.float32)[:, 0], r[:, 1], r[:, 2], r[:, 3]

    @property
    def main(self):
        """(time, source word, link index, target) of the main runs, run order (record pos of cell c at
        c * main_cap + pos)."""
        return self._fields(self._records[0], self.main_cap, self.table if self.pool_b else None)

    @property
    def side(self):
        return self._fields(self._records[0] if self.pool_b else self._records[1], self.side_cap,
                            self.side_table if self.pool_b else None)

    def check(self) -> None:
        main_full, far, uniform, side_full = self.flags.tolist()
        if main_full or side_full:
            if self.pool_b:
                raise RuntimeError(f"a group's ring arena ran out ({self.pool_nb} blocks of {self.pool_b} records)")
            raise RuntimeError(
                f"a ring run exceeded its capacity ({'main' if main_full else 'side'}: main {self.main_cap}, "
                f"side {self.side_cap} entries)"
            )
        if far:
            raise NotImplementedError(
                f"an arrival lies at or beyond {self.n_slots} slots ahead of the ring "
                f"position (e.g. a long emission delay): the far-future list isn't implemented"
            )
        if uniform:
            raise RuntimeError(
                "a uniform-emission run took in a spike below the watermark or past its "
                "source's run window: the uniform contract (bare n2.spike() only) was violated"
            )
