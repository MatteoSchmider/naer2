"""
A consumer's rings of arrival slots: where its fused kernel's pull intake
puts every link it takes in, and where delivery reads them from (naer2/
transpiler/templates/fused.py).

One ring per target GROUP (the neurons one program owns). Slot k of a
group's ring covers arrivals in [k * slot_width, (k + 1) * slot_width) and
lives at index k % N_SLOTS: its CELL, holding a run of records appended in
emission order (uniform sources only: fused.py _pull). A record is one LINK:
emission time e, (connection << 24) | source neuron, the link's index in its
connection (its attributes, e.g. the weight) and its target within the
group -- delivery goes from the record straight to the receive. A queue
population (fused.py _pull_queues) has a cell per incoming connection
instead, its records one per SEGMENT (e, source word, first link, links),
consumed from its head (the queue heads).

Where the runs live, two ways:
  - per-cell runs: every cell its own main_cap records,
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
position + N_SLOTS is far future (flag 1, not implemented yet); a run the
ring is too small for stops without committing anything (flag 3) and the
runtime resizes the ring and reruns it -- nothing is lost (flag 0: links
lost, only a shrink below the blocks in use could, which the runtime never
asks for); a uniform-emission run with a spike below the watermark (the
largest e a group's runs took in) or past its window sets flag 2.
"""

import torch
import triton
import triton.language as tl

# One ring per consumer population; n_slots covers its longest incoming delay (+ emission
#   delay) at one slot width -- the finest resolution any of its incoming connections requires (D-12; the
#   coarser connections could be optimized later); capacities per population, below
# A record's time word is its emission time e; delivery computes receive's time from it and the slot
#   (D-1, BRUNEL.md: e + D for one homogeneous delay, else the slot's center or end)
# Run capacity (per-cell runs): this default, per population (Population.main_cap), grown after a run that
# didn't fit (PopulationRuntime._grow_stopped; a burst like PD14's 902-spike thalamic volley lands in every slot of
# its delay range at once).
DEFAULT_MAIN_CAP = 8192  # links per (group, slot): PD14 at G=256 needs ~350 quiet, up to ~9.3k in its thalamic volley
# links lost, far future, uniform contract violated, a run stopped (the ring too small for it: nothing lost, the
# host resizes and reruns it)
_N_FLAGS = 4
FLAG_STOPPED = 3
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


def _hold_until_done(buf: torch.Tensor) -> None:
    """A ring's old buffer, released after a grow / resize: the runs queued before it (and the copies) on this
    stream may still use it -- keep the allocator from reusing it until they are done (it was allocated on
    another stream, whose pool would otherwise take it back at once)."""
    if buf.is_cuda:
        buf.record_stream(torch.cuda.current_stream(buf.device))


@triton.jit
def _f32(p):
    return p.to(tl.pointer_type(tl.float32))


# The buffer's layout lives in one place, Ring._allocate: the header (its first _HEADER words) holds every
# region's word offset, which _ring reads -- the regions, in this order (pooled: the tables, free stacks, block
# counts and overflow logs exist; the queue heads: a queue population's):
_REGIONS = ("main records", "main counts", "queue heads", "flags", "lost", "group stats", "groups done",
            "main tables", "free stacks", "stack tops", "block counts", "log records")
# header words: the regions' offsets (but the log records'), then these sizes
_H_FLAGS, _H_STATS, _H_DONE = 3, 5, 6
_H_SIZES = len(_REGIONS) - 1  # log records (an index from the main records), log cap, main cap, blocks per group
# (flags, lost, the group stats and the groups done are contiguous: the run's status, Ring.status, one copy to the
# host per run)
_HEADER = 32


@triton.jit
def _ring(ring, POOL_B: tl.constexpr):
    """Typed pointers into a Ring buffer, from its header (Ring._allocate): the four fields of a record -- time,
    source word, link index, target (a segment record: first link, links) -- (field f of record i at field_ptr +
    4 * i, see _rec, _run_rec), the per-cell counts, the queue heads, the flags, and (pooled) the cells' block
    tables, the groups' free stacks and their tops, the cells' block counts, the lost flag, the overflow logs'
    first record (an index from the main records, like any record's); then the sizes, all runtime values (a ring
    grows or shrinks without a recompile): a cell's overflow log capacity, the runs' capacity, the arena's blocks
    per group; and the group stats, two words per group the pull writes every run: pooled, the blocks the group
    holds after the pull (or would need: a stopped run) and the most any of its cells put into its overflow log;
    per-cell runs, the longest run. (The groups done: _run_gate, _run_done.) Returns (main_time, main_src,
    main_lo, main_n, main_count, head, flags, table, stack, top, nblk, lost, log_rec, ovf_cap, main_cap, nb,
    stats)."""
    main_time = ring + tl.load(ring + 0)
    return (
        _f32(main_time), main_time + 1, main_time + 2, main_time + 3, ring + tl.load(ring + 1),
        ring + tl.load(ring + 2), ring + tl.load(ring + 3), ring + tl.load(ring + 7), ring + tl.load(ring + 8),
        ring + tl.load(ring + 9), ring + tl.load(ring + 10), ring + tl.load(ring + 4),
        tl.load(ring + 11), tl.load(ring + 12), tl.load(ring + 13), tl.load(ring + 14),
        ring + tl.load(ring + 5),
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
    index, target) -- per-cell runs or the groups' arenas --, the counts, the
    queue heads and the flags, and (pooled) the block tables, free stacks,
    block counts and overflow logs. Cell of (group g, slot k): g * n_slots +
    k % n_slots. main: the four fields as copies in run order, record pos of
    cell c at c * cap + pos (for inspection and tests)."""

    def __init__(self, n_groups: int, n_slots: int, slot_width: float, device,
                 main_cap: int = DEFAULT_MAIN_CAP, pool: tuple[int, int] | None = None, log_cap: int | None = None):
        """pool: (block, blocks) -- every group's runs in one arena of `blocks` blocks of `block` records, a run
        taking blocks as it fills (through its block table) and giving them back once delivered; a run can grow
        to the whole arena (main_cap ignored). Every run keeps POOL_RESERVE blocks (it starts with them, and
        keeps them when delivered), so an arena needs more than n_slots x POOL_RESERVE blocks. log_cap: records
        a cell's overflow log takes per run (default: pool_log_cap). None: every cell its own main_cap records
        (see _rec)."""
        self.n_groups, self.n_slots, self.slot_width = n_groups, n_slots, slot_width
        self.pool_b, self.pool_nb = pool or (0, 1)
        self.log_cap = (log_cap if log_cap is not None else pool_log_cap(self.pool_b, self.pool_nb, n_slots)) \
            if self.pool_b else 0
        if self.pool_b:
            if self.pool_b % 4 or self.pool_nb <= n_slots * POOL_RESERVE:
                raise ValueError("a pool block is a positive multiple of 4 records, and an arena more blocks than "
                                 "slots (each run keeps one in reserve)")
            main_cap = self._run_cap()
        if main_cap % 4:
            raise ValueError("run capacities must be multiples of 4 (entries are interleaved by cache line)")
        self._allocate(main_cap, device)
        self.ring_pos = 0  # first slot not yet delivered
        # the first slot still holding records, where the host knows it (an event-driven population's ring, read
        # back after its runs: Population._note_pending_arrivals; inf: none) -- its next stepped run starts there
        # at the latest (PopulationRuntime.run; a gated run: fused.py _run_from); None: unknown
        self.first_pending: float | None = None

    def _run_cap(self) -> int:
        """Pooled: the records one run can hold -- the whole arena, within the claim counter's fields."""
        cap = min(self.pool_nb, POOL_MAX_BLOCKS) * self.pool_b
        return min(cap, ((1 << POOL_CNT_BITS) - 1) // self.pool_b * self.pool_b)

    @property
    def arena(self) -> int:
        """Records per group (pooled), else per cell."""
        return self.pool_b * self.pool_nb if self.pool_b else self.main_cap

    def _allocate(self, main_cap: int, device, zero_records: bool = True) -> None:
        """The buffer: the header, then the regions (_REGIONS) -- record regions on 16-byte boundaries.
        zero_records False: the records left as they come (a grow / resize copies its own over them; past the
        counts, nothing reads a record), every other region zeroed."""
        if self.pool_b:
            main_cap = self._run_cap()
        self.main_cap = main_cap
        cells, g = self.n_groups * self.n_slots, self.n_groups
        pooled = bool(self.pool_b)
        m = g * self.pool_nb * self.pool_b if pooled else cells * main_cap
        log_cap = self.log_cap if pooled else 0
        words = {"main records": 4 * m, "main counts": cells, "queue heads": cells, "flags": _N_FLAGS,
                 "main tables": cells * self.pool_nb if pooled else 0,
                 "free stacks": g * self.pool_nb if pooled else 0, "stack tops": g if pooled else 0,
                 "block counts": cells if pooled else 0, "lost": 1, "group stats": 2 * g, "groups done": g,
                 "log records": 4 * cells * log_cap}
        off, o = {}, _HEADER
        for name in _REGIONS:
            if name.endswith("records"):
                o += (-o) % 4
            off[name] = o
            o += words[name]
        if zero_records:
            self.buf = torch.zeros(o, device=device, dtype=torch.int32)
        else:
            self.buf = torch.empty(o, device=device, dtype=torch.int32)
            self.buf[:off["main records"]].zero_()
            self.buf[off["main records"] + words["main records"]:off["log records"]].zero_()
        header = [off[n] for n in _REGIONS[:-1]] + [(off["log records"] - off["main records"]) // 4, log_cap,
                                                     main_cap, self.pool_nb]
        self.buf[:len(header)] = torch.tensor(header, dtype=torch.int32)
        view = lambda name, *shape: self.buf[off[name]:off[name] + words[name]].view(*shape)
        self._records = view("main records", -1, 4)
        self.main_count, self.head = view("main counts", -1), view("queue heads", -1)
        self.flags = view("flags", -1)
        self.lost = view("lost", -1)
        # the run's status: flags, lost, group stats, groups done -- contiguous (see _REGIONS)
        self.status = self.buf[off["flags"]:off["groups done"] + words["groups done"]]
        self.stats = view("group stats", g, 2)
        self.done = view("groups done", -1)
        if pooled:
            self.table = view("main tables", cells, -1)
            self.stack, self.top = view("free stacks", g, -1), view("stack tops", -1)
            self.nblk = view("block counts", -1)
            self.log_rec = view("log records", -1, 4)
            self.table.fill_(-1)
            # every run its reserved blocks (slot s of a group: blocks s * R .. (s + 1) * R), the rest free
            R = POOL_RESERVE
            self.table[:, :R] = torch.arange(self.n_slots * R, device=device, dtype=torch.int32).view(
                self.n_slots, R).repeat(g, 1)
            self.nblk.fill_(R)
            free = self.pool_nb - self.n_slots * R
            self.stack[:, :free] = torch.arange(self.n_slots * R, self.pool_nb, device=device, dtype=torch.int32)
            self.top.fill_(free)

    def grow(self, main_cap: int) -> None:
        """Per-cell runs: a larger run capacity, keeping what the ring holds -- a record's place (_rec) doesn't
        depend on the capacity, so the records copy over as one block; the counts are clamped to what was
        written, the flags carried along. buf is a new buffer: whoever holds its address (a launch's arguments)
        takes it again. (Pooled rings: resize.)"""
        if self.pool_b:
            raise ValueError("a pooled ring changes size through resize()")
        old_buf, rec = self.buf, self._records
        keep = dict(main_count=self.main_count.clone(), head=self.head.clone(), flags=self.flags.clone(),
                    lost=self.lost.clone(), done=self.done.clone())
        old = self.main_cap
        if main_cap % 4 or main_cap < old:
            raise ValueError("capacities only grow, in multiples of 4")
        self._allocate(main_cap, old_buf.device, zero_records=False)
        self._records[:len(rec)] = rec
        keep["main_count"].clamp_(max=old)
        for name, v in keep.items():
            getattr(self, name).copy_(v)
        _hold_until_done(old_buf)

    def resize(self, pool_nb: int | None = None, log_cap: int | None = None) -> None:
        """Pooled: the arenas to pool_nb blocks per group and the overflow logs to log_cap records per cell
        (between runs: the logs are empty), keeping what the ring holds. Growing (or keeping) the arena:
        every block keeps its id -- the records copy over as they are (one copy of the old arena), the tables
        too, the new ids join the free stacks. Shrinking: the blocks in use -- every run's first nblk -- are
        renumbered: the reserve keeps its fixed ids, the rest follow it in order; their records copy over, the
        tables are remapped, the free stacks hold every other id. All on the device, in stream order (no host
        sync). A shrink below the blocks in use loses the ones that don't fit: marked missing, the lost flag
        and flag 0 set (the runtime never asks for one). buf is a new buffer (see grow)."""
        if not self.pool_b:
            raise ValueError("resize() is for pooled rings (per-cell runs: grow())")
        pool_nb, log_cap = pool_nb or self.pool_nb, log_cap or self.log_cap
        g, S, B, R = self.n_groups, self.n_slots, self.pool_b, POOL_RESERVE
        if pool_nb <= S * R:
            raise ValueError("an arena needs more blocks than its slots' reserves")
        dev, cells, nb = self.buf.device, g * S, self.pool_nb
        if pool_nb >= nb:
            self._grow_arena(pool_nb, log_cap)
            return
        old_buf, rec = self.buf, self._records.view(g * nb, B, 4)
        keep = dict(main_count=self.main_count.clone(), head=self.head.clone(), flags=self.flags.clone(),
                    lost=self.lost.clone(), nblk=self.nblk.clone(), done=self.done.clone())
        table = self.table.long()
        col = torch.arange(nb, device=dev)
        group = torch.arange(cells, device=dev) // S
        main_used = (col[None, :] < keep["nblk"].long()[:, None]) & (table >= 0)
        used = torch.zeros(g * nb + 1, dtype=torch.bool, device=dev)  # (+1: where the unused entries go)
        dump = g * nb
        used.index_fill_(0, torch.where(main_used, group[:, None] * nb + table.clamp(min=0), dump).flatten(), True)
        used = used[:dump].view(g, nb)
        reserve = col[None, :] < S * R
        movable = used & ~reserve
        new_id = torch.where(reserve, col[None, :], S * R + torch.cumsum(movable.long(), 1) - 1)  # [g, nb]
        fits = new_id < pool_nb
        dropped = (used & ~fits).any()
        self.pool_nb, self.log_cap = pool_nb, log_cap
        self._allocate(0, dev)
        # the blocks' records, through a scratch row for the ones that don't move
        dst = torch.where(used & fits, torch.arange(g, device=dev)[:, None] * pool_nb + new_id, g * pool_nb)
        moved = torch.zeros(g * pool_nb + 1, B, 4, dtype=rec.dtype, device=dev)
        moved.index_copy_(0, dst.flatten(), rec)
        self._records.view(g * pool_nb, B, 4).copy_(moved[:g * pool_nb])
        del moved
        # the tables through the new ids (entries past what a run holds: none)
        remap = torch.where(fits, new_id, -1)  # [g, nb]
        n = min(nb, pool_nb)
        ids = torch.gather(remap[group], 1, table.clamp(min=0))
        self.table[:, :n] = torch.where(main_used, ids, -1)[:, :n].to(torch.int32)
        # the free stacks: every id not in use, in order
        in_use = torch.zeros(g, pool_nb + 1, dtype=torch.bool, device=dev)
        in_use.scatter_(1, torch.where(used & fits, new_id, pool_nb), True)
        in_use = in_use[:, :pool_nb]
        self.stack.copy_(torch.argsort(in_use.to(torch.int8), dim=1, stable=True).to(torch.int32))
        self.top.copy_((pool_nb - in_use.sum(1)).to(torch.int32))
        # (a run can't hold more blocks than the arena has: past them, its records are lost)
        keep["nblk"].clamp_(max=min(pool_nb, POOL_MAX_BLOCKS))
        for k in ("main_count", "head"):
            keep[k].clamp_(max=self.main_cap)
        for name, v in keep.items():
            getattr(self, name).copy_(v)
        self.lost.copy_(torch.maximum(self.lost, dropped.to(torch.int32)))
        self.flags[0] = torch.maximum(self.flags[0], dropped.to(torch.int32))
        _hold_until_done(old_buf)

    def _grow_arena(self, pool_nb: int, log_cap: int) -> None:
        """resize() to at least as many blocks: every block keeps its id (see resize)."""
        g, B, nb, dev = self.n_groups, self.pool_b, self.pool_nb, self.buf.device
        old_buf = self.buf
        old = dict(main_count=self.main_count, head=self.head, flags=self.flags, lost=self.lost, nblk=self.nblk,
                   done=self.done)  # views of the old buffer
        old_rec, old_top = self._records.view(g, nb * B * 4), self.top
        old_tables = (self.table, self.stack)
        self.pool_nb, self.log_cap = pool_nb, log_cap
        self._allocate(0, dev, zero_records=False)
        self._records.view(g, pool_nb * B * 4)[:, :nb * B * 4].copy_(old_rec)
        for old_t, new_t in zip(old_tables, (self.table, self.stack)):
            new_t[:, :nb].copy_(old_t)  # (the tables' entries past nb: -1, none held)
        extra = pool_nb - nb
        if extra:  # the new ids onto every free stack, above what it holds
            at = old_top.long()[:, None] + torch.arange(extra, device=dev)
            self.stack.scatter_(1, at, torch.arange(nb, pool_nb, device=dev, dtype=torch.int32).expand(g, -1))
        self.top.copy_(old_top + extra)
        for name, v in old.items():
            getattr(self, name).copy_(v)
        _hold_until_done(old_buf)

    def cell(self, group: int, slot: int) -> int:
        return group * self.n_slots + slot % self.n_slots

    @property
    def main(self):
        """(time, source word, link index, target) of the runs, run order (record pos of cell c at
        c * main_cap + pos)."""
        rec, cap = self._records, self.main_cap
        cells = self.n_groups * self.n_slots
        cell = torch.arange(cells, device=rec.device).repeat_interleave(cap)
        pos = torch.arange(cap, device=rec.device).repeat(cells)
        if self.pool_b:  # see _run_rec: through the run's block table, a missing block's records zero
            blk = self.table[cell, pos // self.pool_b].long()
            r = rec[((cell // self.n_slots) * self.pool_nb + blk.clamp(min=0)) * self.pool_b + pos % self.pool_b]
            r = torch.where((blk >= 0)[:, None], r, 0)
        else:
            r = rec[((pos >> 2) * cells + cell) * 4 + (pos & 3)]  # see _rec
        return r.view(torch.float32)[:, 0], r[:, 1], r[:, 2], r[:, 3]

    def check(self) -> None:
        lost, far, uniform, _stopped = self.flags.tolist()
        if lost:
            raise RuntimeError(f"a ring lost links (an arena shrunk below its blocks in use: {self.pool_nb} blocks "
                               f"of {self.pool_b} records)")
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
