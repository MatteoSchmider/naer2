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

Where the runs live: every group's runs in one ARENA of fixed-size blocks,
a run taking blocks as it fills (its block table) and giving them back when
delivered (a free stack per group). A slot's run keeps RESERVE blocks for
good, fixed by its slot -- in quiet activity a run hardly ever outgrows
them, so records go where plain arithmetic says. Past them, the pull writes
into the cell's overflow log and moves the records into new blocks once,
after its last barrier (fused.py: _append_links, _arena_fixup). The host
sizes the arenas and logs from every run's need (PopulationRuntime: grown
or shrunk with Ring.resize; a run too large for them stops and reruns).

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
# links lost, far future, uniform contract violated, a run stopped (the ring too small for it: nothing lost, the
# host resizes and reruns it)
_N_FLAGS = 4
FLAG_STOPPED = 3
# the blocks every slot's run keeps -- it starts with them, and keeps them when delivered. PD14 (256-record blocks):
# a quiet slot ends its life with ~220-360 records, so 2 blocks cover it and a quiet run hardly ever takes a new one
# (each a barrier in the pull); could follow the network's arrival rate x delay (Little's law) instead
RESERVE = 2
# the pull's LDS counter per slot: count | blocks held << CNT_BITS (both come back with every claim); so a run
# holds at most MAX_BLOCKS blocks and 2**CNT_BITS - 1 records
CNT_BITS = 20
MAX_BLOCKS = 2047


def default_log_cap(block: int, n_blocks: int, n_slots: int) -> int:
    """Entries of a cell's overflow log -- the records one run can put past the blocks it holds: half its
    group's arena over its slots (PD14: one cell grows by up to ~1400 records in one run of the thalamic volley,
    by ~80 in quiet ones; the log is scratch for bursts, its size a candidate for tightening)."""
    return block * n_blocks // 2 // n_slots


def _hold_until_done(buf: torch.Tensor) -> None:
    """A ring's old buffer, released after a resize: the runs queued before it (and the copies) on this
    stream may still use it -- keep the allocator from reusing it until they are done (it was allocated on
    another stream, whose pool would otherwise take it back at once)."""
    if buf.is_cuda:
        buf.record_stream(torch.cuda.current_stream(buf.device))


@triton.jit
def _f32(p):
    return p.to(tl.pointer_type(tl.float32))


# The buffer's layout lives in one place, Ring._allocate: the header (its first _HEADER words) holds every
# region's word offset, which _ring reads -- the regions, in this order (the queue heads: a queue population's):
_REGIONS = ("records", "run counts", "queue heads", "flags", "lost", "group stats", "groups done",
            "block tables", "free stacks", "stack tops", "block counts", "log records")
# header words: the regions' offsets (but the log records'), then these sizes
_H_FLAGS, _H_STATS, _H_DONE = 3, 5, 6
_H_SIZES = len(_REGIONS) - 1  # log records (an index from the records), log cap, run cap, blocks per group
# (flags, lost, the group stats and the groups done are contiguous: the run's status, Ring.status, one copy to the
# host per run)
_HEADER = 32


@triton.jit
def _ring(ring, BLOCK: tl.constexpr):
    """Typed pointers into a Ring buffer, from its header (Ring._allocate): the four fields of a record -- time,
    source word, link index, target (a segment record: first link, links) -- (field f of record i at field_ptr +
    4 * i, see _run_rec), the per-cell counts, the queue heads, the flags, the cells' block tables, the groups'
    free stacks and their tops, the cells' block counts, the lost flag, the overflow logs' first record (an index
    from the records, like any record's); then the sizes, all runtime values (a ring grows or shrinks
    without a recompile): a cell's overflow log capacity, the runs' capacity, the arena's blocks per group; and
    the group stats, two words per group the pull writes every run: the blocks the group holds after the pull (or
    would need: a stopped run) and the most any of its cells put into its overflow log. (The groups done:
    _run_gate, _run_done.) Returns (rec_time, rec_src,
    rec_lo, rec_n, run_count, head, flags, table, stack, top, nblk, lost, log_rec, ovf_cap, run_cap, nb,
    stats)."""
    rec_time = ring + tl.load(ring + 0)
    return (
        _f32(rec_time), rec_time + 1, rec_time + 2, rec_time + 3, ring + tl.load(ring + 1),
        ring + tl.load(ring + 2), ring + tl.load(ring + 3), ring + tl.load(ring + 7), ring + tl.load(ring + 8),
        ring + tl.load(ring + 9), ring + tl.load(ring + 10), ring + tl.load(ring + 4),
        tl.load(ring + 11), tl.load(ring + 12), tl.load(ring + 13), tl.load(ring + 14),
        ring + tl.load(ring + 5),
    )


class Ring:
    """One consumer's rings of arrival slots, one ring per target group
    (see the module docstring), as one int32 buffer (layout: _allocate,
    _REGIONS): the groups' arenas of records, each 16 bytes (time, (conn <<
    24) | source, link index, target), the counts, the queue heads, the
    flags, the block tables, free stacks, block counts and overflow logs.
    Cell of (group g, slot k): g * n_slots + k % n_slots. runs: the four
    fields as copies in run order, record pos of cell c at c * run_cap + pos
    (for inspection and tests)."""

    def __init__(self, n_groups: int, n_slots: int, slot_width: float, device, arena: tuple[int, int],
                 log_cap: int | None = None):
        """arena: (block, n_blocks) -- every group's runs in one arena of n_blocks blocks of `block` records, a
        run taking blocks as it fills (through its block table) and giving them back once delivered; a run can
        grow to the whole arena. Every run keeps RESERVE blocks (it starts with them, and keeps them when
        delivered), so an arena needs more than n_slots x RESERVE blocks. log_cap: records a cell's overflow log
        takes per run (default: default_log_cap)."""
        self.n_groups, self.n_slots, self.slot_width = n_groups, n_slots, slot_width
        self.block, self.n_blocks = arena
        if self.block <= 0 or self.block % 4 or self.n_blocks <= n_slots * RESERVE:
            raise ValueError("a block is a positive multiple of 4 records, and an arena more blocks than its "
                             "slots' reserves")
        self.log_cap = log_cap if log_cap is not None else default_log_cap(self.block, self.n_blocks, n_slots)
        self._allocate(device)
        self.ring_pos = 0  # first slot not yet delivered
        # the first slot still holding records, where the host knows it (an event-driven population's ring, read
        # back after its runs: Population._pending_arrival; inf: none) -- its next stepped run starts there
        # at the latest (PopulationRuntime.run; a gated run: fused.py _run_from); None: unknown
        self.first_pending: float | None = None

    @property
    def run_cap(self) -> int:
        """The records one run can hold -- the whole arena, within the claim counter's fields."""
        cap = min(self.n_blocks, MAX_BLOCKS) * self.block
        return min(cap, ((1 << CNT_BITS) - 1) // self.block * self.block)

    @property
    def arena(self) -> int:
        """Records per group."""
        return self.block * self.n_blocks

    def _allocate(self, device, zero_records: bool = True) -> None:
        """The buffer: the header, then the regions (_REGIONS) -- record regions on 16-byte boundaries.
        zero_records False: the records left as they come (a resize copies its own over them; past the counts,
        nothing reads a record), every other region zeroed."""
        cells, g, nb = self.n_groups * self.n_slots, self.n_groups, self.n_blocks
        words = {"records": 4 * g * nb * self.block, "run counts": cells, "queue heads": cells,
                 "flags": _N_FLAGS, "block tables": cells * nb, "free stacks": g * nb, "stack tops": g,
                 "block counts": cells, "lost": 1, "group stats": 2 * g, "groups done": g,
                 "log records": 4 * cells * self.log_cap}
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
            self.buf[:off["records"]].zero_()
            self.buf[off["records"] + words["records"]:off["log records"]].zero_()
        header = [off[n] for n in _REGIONS[:-1]] + [(off["log records"] - off["records"]) // 4, self.log_cap,
                                                     self.run_cap, nb]
        self.buf[:len(header)] = torch.tensor(header, dtype=torch.int32)
        view = lambda name, *shape: self.buf[off[name]:off[name] + words[name]].view(*shape)
        self._records = view("records", -1, 4)
        self.run_count, self.head = view("run counts", -1), view("queue heads", -1)
        self.flags = view("flags", -1)
        self.lost = view("lost", -1)
        # the run's status: flags, lost, group stats, groups done -- contiguous (see _REGIONS)
        self.status = self.buf[off["flags"]:off["groups done"] + words["groups done"]]
        self.stats = view("group stats", g, 2)
        self.done = view("groups done", -1)
        self.table = view("block tables", cells, -1)
        self.stack, self.top = view("free stacks", g, -1), view("stack tops", -1)
        self.nblk = view("block counts", -1)
        self.log_rec = view("log records", -1, 4)
        self.table.fill_(-1)
        # every run its reserved blocks (slot s of a group: blocks s * R .. (s + 1) * R), the rest free
        R = RESERVE
        self.table[:, :R] = torch.arange(self.n_slots * R, device=device, dtype=torch.int32).view(
            self.n_slots, R).repeat(g, 1)
        self.nblk.fill_(R)
        free = nb - self.n_slots * R
        self.stack[:, :free] = torch.arange(self.n_slots * R, nb, device=device, dtype=torch.int32)
        self.top.fill_(free)

    def resize(self, n_blocks: int | None = None, log_cap: int | None = None) -> None:
        """The arenas to n_blocks blocks per group and the overflow logs to log_cap records per cell
        (between runs: the logs are empty), keeping what the ring holds. Growing (or keeping) the arena:
        every block keeps its id -- the records copy over as they are (one copy of the old arena), the tables
        too, the new ids join the free stacks. Shrinking: the blocks in use -- every run's first nblk -- are
        renumbered: the reserve keeps its fixed ids, the rest follow it in order; their records copy over, the
        tables are remapped, the free stacks hold every other id. All on the device, in stream order (no host
        sync). A shrink below the blocks in use loses the ones that don't fit: marked missing, the lost flag
        and flag 0 set (the runtime never asks for one). buf is a new buffer: whoever holds its address (a
        launch's arguments, a captured graph) takes it again."""
        n_blocks, log_cap = n_blocks or self.n_blocks, log_cap or self.log_cap
        g, S, B, R = self.n_groups, self.n_slots, self.block, RESERVE
        if n_blocks <= S * R:
            raise ValueError("an arena needs more blocks than its slots' reserves")
        dev, cells, nb = self.buf.device, g * S, self.n_blocks
        if n_blocks >= nb:
            self._grow_arena(n_blocks, log_cap)
            return
        old_buf, rec = self.buf, self._records.view(g * nb, B, 4)
        keep = dict(run_count=self.run_count.clone(), head=self.head.clone(), flags=self.flags.clone(),
                    lost=self.lost.clone(), nblk=self.nblk.clone(), done=self.done.clone())
        table = self.table.long()
        col = torch.arange(nb, device=dev)
        group = torch.arange(cells, device=dev) // S
        in_table = (col[None, :] < keep["nblk"].long()[:, None]) & (table >= 0)
        used = torch.zeros(g * nb + 1, dtype=torch.bool, device=dev)  # (+1: where the unused entries go)
        dump = g * nb
        used.index_fill_(0, torch.where(in_table, group[:, None] * nb + table.clamp(min=0), dump).flatten(), True)
        used = used[:dump].view(g, nb)
        reserve = col[None, :] < S * R
        movable = used & ~reserve
        new_id = torch.where(reserve, col[None, :], S * R + torch.cumsum(movable.long(), 1) - 1)  # [g, nb]
        fits = new_id < n_blocks
        dropped = (used & ~fits).any()
        self.n_blocks, self.log_cap = n_blocks, log_cap
        self._allocate(dev)
        # the blocks' records, through a scratch row for the ones that don't move
        dst = torch.where(used & fits, torch.arange(g, device=dev)[:, None] * n_blocks + new_id, g * n_blocks)
        moved = torch.zeros(g * n_blocks + 1, B, 4, dtype=rec.dtype, device=dev)
        moved.index_copy_(0, dst.flatten(), rec)
        self._records.view(g * n_blocks, B, 4).copy_(moved[:g * n_blocks])
        del moved
        # the tables through the new ids (entries past what a run holds: none)
        remap = torch.where(fits, new_id, -1)  # [g, nb]
        n = min(nb, n_blocks)
        ids = torch.gather(remap[group], 1, table.clamp(min=0))
        self.table[:, :n] = torch.where(in_table, ids, -1)[:, :n].to(torch.int32)
        # the free stacks: every id not in use, in order
        in_use = torch.zeros(g, n_blocks + 1, dtype=torch.bool, device=dev)
        in_use.scatter_(1, torch.where(used & fits, new_id, n_blocks), True)
        in_use = in_use[:, :n_blocks]
        self.stack.copy_(torch.argsort(in_use.to(torch.int8), dim=1, stable=True).to(torch.int32))
        self.top.copy_((n_blocks - in_use.sum(1)).to(torch.int32))
        # (a run can't hold more blocks than the arena has: past them, its records are lost)
        keep["nblk"].clamp_(max=min(n_blocks, MAX_BLOCKS))
        for k in ("run_count", "head"):
            keep[k].clamp_(max=self.run_cap)
        for name, v in keep.items():
            getattr(self, name).copy_(v)
        self.lost.copy_(torch.maximum(self.lost, dropped.to(torch.int32)))
        self.flags[0] = torch.maximum(self.flags[0], dropped.to(torch.int32))
        _hold_until_done(old_buf)

    def _grow_arena(self, n_blocks: int, log_cap: int) -> None:
        """resize() to at least as many blocks: every block keeps its id (see resize)."""
        g, B, nb, dev = self.n_groups, self.block, self.n_blocks, self.buf.device
        old_buf = self.buf
        old = dict(run_count=self.run_count, head=self.head, flags=self.flags, lost=self.lost, nblk=self.nblk,
                   done=self.done)  # views of the old buffer
        old_rec, old_top = self._records.view(g, nb * B * 4), self.top
        old_tables = (self.table, self.stack)
        self.n_blocks, self.log_cap = n_blocks, log_cap
        self._allocate(dev, zero_records=False)
        self._records.view(g, n_blocks * B * 4)[:, :nb * B * 4].copy_(old_rec)
        for old_t, new_t in zip(old_tables, (self.table, self.stack)):
            new_t[:, :nb].copy_(old_t)  # (the tables' entries past nb: -1, none held)
        extra = n_blocks - nb
        if extra:  # the new ids onto every free stack, above what it holds
            at = old_top.long()[:, None] + torch.arange(extra, device=dev)
            self.stack.scatter_(1, at, torch.arange(nb, n_blocks, device=dev, dtype=torch.int32).expand(g, -1))
        self.top.copy_(old_top + extra)
        for name, v in old.items():
            getattr(self, name).copy_(v)
        _hold_until_done(old_buf)

    def cell(self, group: int, slot: int) -> int:
        return group * self.n_slots + slot % self.n_slots

    @property
    def runs(self):
        """(time, source word, link index, target) of the runs, run order (record pos of cell c at
        c * run_cap + pos)."""
        rec, cap = self._records, self.run_cap
        cells = self.n_groups * self.n_slots
        cell = torch.arange(cells, device=rec.device).repeat_interleave(cap)
        pos = torch.arange(cap, device=rec.device).repeat(cells)
        # see _run_rec: through the run's block table, a missing block's records zero
        blk = self.table[cell, pos // self.block].long()
        r = rec[((cell // self.n_slots) * self.n_blocks + blk.clamp(min=0)) * self.block + pos % self.block]
        r = torch.where((blk >= 0)[:, None], r, 0)
        return r.view(torch.float32)[:, 0], r[:, 1], r[:, 2], r[:, 3]

    def check(self) -> None:
        lost, far, uniform, _stopped = self.flags.tolist()
        if lost:
            raise RuntimeError(f"a ring lost links (an arena shrunk below its blocks in use: {self.n_blocks} blocks "
                               f"of {self.block} records)")
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
