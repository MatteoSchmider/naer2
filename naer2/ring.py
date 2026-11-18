"""
A consumer's rings of arrival slots: where its fused kernel's pull intake
puts every link it takes in, and where delivery reads them from (naer2/
transpiler/templates/fused.py).

One ring per target GROUP (the neurons one program owns). Slot k of a
group's ring covers arrivals in [k * slot_width, (k + 1) * slot_width) and
lives at index k % N_SLOTS: its CELL. Each cell holds
  - a main run: records appended in emission order, and
  - a side run: out-of-order records (emission-delayed, or below the
    watermark), kept sorted by emission time by insertion.
A record is one LINK: emission time e, (connection << 24) | source neuron,
the link's index in its connection (its attributes, e.g. the weight) and
its target within the group -- delivery goes from the record straight to
the receive. A record goes to the main runs if e >= the watermark (largest
e ever appended to a group's main runs) and it isn't emission-delayed (e
before its source's run window end); otherwise to the side runs.

Where the runs live, two ways:
  - per-cell runs: every cell its own main_cap + side_cap records,
    interleaved by cache line across cells (_rec)
  - pooled (Population's default): every group's runs in one ARENA of
    fixed-size blocks, a run taking blocks as it fills (its block table)
    and giving them back when delivered (a free stack per group). A main
    run keeps POOL_RESERVE blocks for good, fixed by its slot -- in quiet
    activity a run hardly ever outgrows them, so records go where plain
    arithmetic says. Past them, the pull writes into the cell's overflow
    log and moves the records into new blocks once, after its last barrier
    (fused.py: _append_links, _pool_fixup).

An arrival before the ring position (already delivered) goes into the first
undelivered slot (late, the timestep quantization); one at or past ring
position + N_SLOTS is far future (flag 1, not implemented yet); a full main
run (pooled: an arena or a cell's log out of room) sets flag 0, a full side
run flag 3 -- their links are lost, and the runtime grows the ring, grow();
a uniform-emission run that would need a side run sets flag 2.
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


# The buffer's layout lives in one place, Ring._allocate: the header (its first _HEADER words) holds every
# region's word offset, which _ring reads -- the regions, in this order (pooled: the side records are the
# arena too, and the tables, free stacks, block counts, lost flag and overflow logs exist):
_REGIONS = ("main records", "side records", "main counts", "side counts", "watermark", "flags", "main tables",
            "side tables", "free stacks", "stack tops", "block counts", "lost", "log records")
_HEADER = 16


@triton.jit
def _ring(ring, n_groups, N_SLOTS: tl.constexpr, MAIN_CAP: tl.constexpr, SIDE_CAP: tl.constexpr,
          POOL_B: tl.constexpr, POOL_NB: tl.constexpr):
    """Typed pointers into a Ring buffer, from its header (Ring._allocate): the four fields of a record -- time,
    source word, link index, target -- for the main and the side runs (field f of record i at field_ptr + 4 * i,
    see _rec, _run_rec), the per-cell counts, watermark and flags, and (pooled) the cells' main and side block
    tables, the groups' free stacks and their tops, the main runs' block counts, the lost flag, the
    overflow logs' first record (an index from the main records, like any record's) and a cell's log's
    capacity (pool_log_cap)."""
    main_time = ring + tl.load(ring + 0)
    side_time = ring + tl.load(ring + 1)
    flags = ring + tl.load(ring + 5)
    return (
        _f32(main_time), main_time + 1, main_time + 2, main_time + 3, ring + tl.load(ring + 2),
        _f32(side_time), side_time + 1, side_time + 2, side_time + 3, ring + tl.load(ring + 3),
        _f32(ring + tl.load(ring + 4)), flags, ring + tl.load(ring + 6), ring + tl.load(ring + 7),
        ring + tl.load(ring + 8), ring + tl.load(ring + 9), ring + tl.load(ring + 10), ring + tl.load(ring + 11),
        tl.load(ring + 12), tl.load(ring + 13),
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
    (see the module docstring), as one int32 buffer (layout: _allocate,
    _REGIONS): the records, each 16 bytes (time, (conn << 24) | source, link
    index, target) -- per-cell runs or the groups' arenas --, the main and
    side counts, the watermark (float32) and the flags, and (pooled) the
    block tables, free stacks, block counts and overflow logs. Cell of
    (group g, slot k): g * n_slots + k % n_slots. main / side: the four
    fields as copies in run order, record pos of cell c at c * cap + pos
    (for inspection and tests)."""

    def __init__(self, n_groups: int, n_slots: int, slot_width: float, device,
                 main_cap: int = DEFAULT_MAIN_CAP, side_cap: int = DEFAULT_SIDE_CAP,
                 pool: tuple[int, int] | None = None):
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
        """The buffer: the header, then the regions (_REGIONS) -- record regions on 16-byte boundaries."""
        if self.pool_b:
            main_cap = side_cap = self._run_cap()
        self.main_cap, self.side_cap = main_cap, side_cap
        cells, g = self.n_groups * self.n_slots, self.n_groups
        pooled = bool(self.pool_b)
        m = g * self.pool_nb * self.pool_b if pooled else cells * main_cap
        log_cap = pool_log_cap(self.pool_b, self.pool_nb, self.n_slots) if pooled else 0
        words = {"main records": 4 * m, "side records": 0 if pooled else 4 * cells * side_cap,
                 "main counts": cells, "side counts": cells, "watermark": 1, "flags": _N_FLAGS,
                 "main tables": cells * self.pool_nb if pooled else 0,
                 "side tables": cells * self.pool_nb if pooled else 0,
                 "free stacks": g * self.pool_nb if pooled else 0, "stack tops": g if pooled else 0,
                 "block counts": cells if pooled else 0, "lost": 1 if pooled else 0,
                 "log records": 4 * cells * log_cap}
        off, o = {}, _HEADER
        for name in _REGIONS:
            if name.endswith("records"):
                o += (-o) % 4
            off[name] = o
            o += words[name]
        if pooled:
            off["side records"] = off["main records"]  # one arena per group: main and side blocks alike
        self.buf = torch.zeros(o, device=device, dtype=torch.int32)
        header = [off[n] for n in _REGIONS[:-1]] + [(off["log records"] - off["main records"]) // 4, log_cap]
        self.buf[:len(header)] = torch.tensor(header, dtype=torch.int32)
        view = lambda name, *shape: self.buf[off[name]:off[name] + words[name]].view(*shape)
        self._records = (view("main records", -1, 4), view("main records" if pooled else "side records", -1, 4))
        self.main_count, self.side_count = view("main counts", -1), view("side counts", -1)
        self.watermark = view("watermark", -1).view(torch.float32)
        self.flags = view("flags", -1)
        if pooled:
            self.table, self.side_table = view("main tables", cells, -1), view("side tables", cells, -1)
            self.stack, self.top = view("free stacks", g, -1), view("stack tops", -1)
            self.nblk, self.lost = view("block counts", -1), view("lost", -1)
            self.log_rec = view("log records", -1, 4)
            self.table.fill_(-1)
            self.side_table.fill_(-1)
            # every main run its reserved blocks (slot s of a group: blocks s * R .. (s + 1) * R), the rest free
            R = POOL_RESERVE
            self.table[:, :R] = torch.arange(self.n_slots * R, device=device, dtype=torch.int32).view(
                self.n_slots, R).repeat(g, 1)
            self.nblk.fill_(R)
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
