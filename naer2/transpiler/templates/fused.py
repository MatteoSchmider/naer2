"""
The fused kernel's template: one launch per population and run -- intake,
delivery, update, send and emission in one Gluon kernel, one program per
group. This module is the FIXED part, written by hand and optimized (what
limits optimizing it further: limitations.md, next to this file); the hooks
come in from outside: a kernel entry (generated from a population's hooks,
naer2/transpiler/templates/fused_template.py, or written by hand --
test/fused_reference.py, the reference) calls

  _pull(...)                  phase 1, no user code at all
  _deliver(state, RECEIVE)    phase 2, RECEIVE(state, hit, time, w, srcword,
                              ectx, nodes) -> state a constexpr gluon.jit
                              function, state the tuple of node attrs
                              receive touches; ectx, nodes: for its
                              n2.spike() (_emit_delivered, D-3)
  (update, send: the entry's own code, then)
  _emit_spikes(fires, e)      one n2.spike() site's spikes into the history,
                              then _emit_close() once (_emit: both, one site)

 Intake is PULLED: every program takes in its own group's entries
itself, so no program depends on another within the launch (a push intake
appending into any group's ring would need a grid-wide barrier before
delivery).

Per program (group g):
  1. pull intake: for every incoming connection, the runs of its source
     since this consumer last read it (naer2/history.py: slots
     tagged with an earlier round, run id past the last-read id); per spike
     g's segments among the source's (sorted by (group, bucket)) through
     the topology's group directory, every link of them one 16-byte record
     (e, source word, link index, target) in its slot's run of g's ring
     (naer2/ring.py: past the blocks a run holds, into its cell's overflow
     log, moved into new blocks after the pull's last barrier -- see "the
     ring's arenas" below). The slotted ring takes uniform sources
     only: all fresh runs of one emission time as one stream of spike
     blocks (each spike's run and its connection's pointers from read-only
     LDS tables), a block of spikes x a block of their segments at a time,
     the link loop four links per step. Queues (every incoming connection
     one delay value: _pull_queues) take any sources, a record per segment.
  2. delivery: the due slots of g's ring in slot order, straight from the
     link records: STREAM records at a time, each one 128-bit load, its
     fields staged flat in LDS, walked U links per broadcast LDS read, every
     link received in order by its target lane (the scalar receive: lanes
     compare against the link's target, no cross-lane traffic); a
     delivered slot's blocks go back to the free stack, but for
     the run's reserve. Queues: merged by arrival, their segments walked in
     batches (_walk_batch).
  3. update and send (the entry's), emission into the history (flat list
     per run, no per-group sort: consumers order what they need themselves).

Contracts: a run reads source runs from earlier rounds only (round tags);
the last-read run ids are double-buffered by the consumer's run parity (all
programs read the same values); entries arriving before the ring position
go to its first undelivered slot (late), at or past N_SLOTS ahead are far
future (flag 1), lost links flag 0, uniform violations flag 2, a run the
ring is too small for stops (flag 3), a consumer falling behind its
sources' history raises the overrun flag.

`# GAP (...)` / `# LIMITATION (...)` comments mark open work and known
limitations, D-n the decisions in TODO.md; what limits further
optimization here: limitations.md, next to this file.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from naer2.history import EmissionHistory, history_ptrs, history_tag
from naer2.ring import CNT_BITS as _CNT_BITS, MAX_BLOCKS as _MAX_BLOCKS
from naer2.ring import FLAG_STOPPED as _FLAG_STOPPED, RESERVE as _RESERVE, _ring
from naer2.topology import SimulationTopology, directory_ptrs, topology_ptrs
from naer2.trace import mark, pin, pin_s, trace_end

FUSED_TRACE_TAGS = ("start", "state", "pulled", "slot", "headers", "streamed", "walked", "delivered", "sent", "end",
                    "runs", "spikes", "found", "appended",  # TRACE >= 2: inside the pull
                    "a:segments", "a:mean", "a:claimed", "a:targets", "a:stored",
                    # TRACE >= 3: inside the append, per tile
                    "f:bar", "f:fixed", "f:stored", "d:bar", "d:freed")  # (TRACE >= 3: temporary dissection)
# the pull: fresh runs batched by emission time (~30% over one by one: the former tools/fused_sweep.py, PD14,
# G=256, 780M); the group directory always on (as fast again over binary search)
# delivery walk: links staged in LDS per stream block, links per broadcast LDS read (tools/walk_sweep.py on
# the split kernels, PD14, G=256, 780M: larger U costs more in registers than it saves; an explicit
# occupancy target (waves_per_eu) gains <= 3% and spills beyond 12 waves)
WALK = dict(stream=256, unroll=8)
SPK = gl.constexpr(32)   # spikes per pull block (uniform sources)
MSEG = gl.constexpr(4)   # the pull's spike layout (LP2: MSEG // 4 per thread); the append packs its own tiles
RESERVE = gl.constexpr(_RESERVE)  # blocks every slot's run keeps (naer2/ring.py)
MAX_BLOCKS = gl.constexpr(_MAX_BLOCKS)  # blocks one run can hold (the claim counter's field)
CNT_BITS = gl.constexpr(_CNT_BITS)  # the claim counter: count | blocks held << CNT_BITS
FLAG_STOPPED = gl.constexpr(_FLAG_STOPPED)  # the ring's flag: a run stopped (too small for it), rerun by the host


# ------------------------------------------------- the plan


@triton.jit
def _fused_plan(plan, n_conns):
    """Typed pointers into a FusedPlan buffer."""
    addr = plan.to(tl.pointer_type(tl.int64))
    base = plan + 10 * n_conns
    return (
        addr, addr + n_conns, addr + 2 * n_conns, addr + 3 * n_conns,  # history, topology, link_target, attrs
        addr + 4 * n_conns,                                            # group directory
        base, base + n_conns, base + 2 * n_conns,                      # history cap, topology n_sources, n_segments
        base + 3 * n_conns, base + 4 * n_conns,                        # topology n_buckets, n_links
        base + 6 * n_conns, base + 7 * n_conns,                        # directory n_pairs, words per source
        base + 8 * n_conns,                                            # last-read run ids [2, n_conns]
    )


# Built by Population from its incoming Connections: per connection its source Population's
#   EmissionHistory and its Fabric's SimulationTopology (Fabric.topology + delay at the consumer's slot
#   resolution, the finest among its incoming connections, D-12). Connections numbered grouped by receive
#   VARIANT (GAP, D-6): the connection id in every record's source word then selects the variant by compile-time
#   comparisons
# GAP (D-7): attr_addr points at each connection's per-link attribute BLOCK -- every edge
#   tensor attr its receive (edge.x) or forward (self.x) references, contiguous per link (link l's attrs at
#   block + l * n_attrs; SimulationTopology's [n_attrs, n_links] becomes [n_links, n_attrs]), D-7
# its rows (n_conns words each): 0-9 the addresses (int64), 10-17 the counts, 18-19 the last-read ids, 20 the
# minimum delays (float32), 21 the histories' programs
PLAN_ROWS, _PLAN_G = 22, 21
PLAN_G = gl.constexpr(_PLAN_G)


class FusedPlan:
    """A consumer's incoming connections, built once into one int32 buffer:
    per connection its source's EmissionHistory and its SimulationTopology
    addresses (+ the topology's link_target and attribute arrays, which the
    delivery walk reads, and its group directory), the history's cap, the
    topology's source, segment, bucket and link counts, its most segments of
    one source, the directory's pair count and words per source, and the
    last-read run ids (double-buffered by the consumer's run parity); per
    connection its minimum delay and its source history's programs."""

    def __init__(self, histories: list[EmissionHistory], topologies: list[SimulationTopology], n_groups: int, device,
                 min_delays: list[float] | None = None):
        n = len(histories)
        if n > 256:
            raise ValueError("a consumer can take at most 256 incoming connections (8-bit connection id)")
        if len({h.K for h in histories}) > 1:
            raise ValueError("all of a consumer's sources must keep histories of the same depth")
        for t in topologies:
            if t.n_sources >= 1 << 24:
                raise ValueError("a source population can have at most 2**24 neurons (24-bit source id)")
            if t.n_groups != n_groups:
                raise ValueError("every incoming topology must use the consumer's group layout")
        self.n_conns, self.histories, self.topologies = n, histories, topologies
        self.K = histories[0].K if histories else 3
        plan = torch.zeros(PLAN_ROWS * n, dtype=torch.int32)
        links = [t.link_addresses() for t in topologies]
        dirs = [t.directory() for t in topologies]
        plan[: 10 * n].view(torch.int64)[:] = torch.tensor(
            [h.buf.data_ptr() for h in histories] + [t.buf.data_ptr() for t in topologies]
            + [a[1] for a in links] + [a[2] for a in links] + [d[0].data_ptr() for d in dirs], dtype=torch.int64)
        max_segs = [int(t.seg_offsets.diff().max()) if t.n_sources else 0 for t in topologies]
        for i, values in enumerate((
            [h.R for h in histories], [t.n_sources for t in topologies], [t.n_segments for t in topologies],
            [t.n_buckets for t in topologies], [t.n_links for t in topologies], max_segs,
            [d[1] for d in dirs], [d[2] for d in dirs],
        )):
            plan[(10 + i) * n:(11 + i) * n] = torch.tensor(values, dtype=torch.int32)
        # per connection its minimum delay as the host schedules with it (a gated run's due time: _gate_skip)
        plan[20 * n:21 * n] = torch.tensor(min_delays if min_delays is not None else [0.0] * n,
                                           dtype=torch.float32).view(torch.int32)
        # per connection its source history's programs (G: a run's spikes in G regions of R, history.py; 0: one
        # flat list, a uniform producer's)
        plan[_PLAN_G * n:(_PLAN_G + 1) * n] = torch.tensor([h.G if h.regions else 0 for h in histories],
                                                            dtype=torch.int32)
        self.plan = plan.to(device)


# ------------------------------------------------- pull helpers


@triton.jit
def _rand(seed, offs):
    """random.random() as the transpiler lowers it (RandomLoweringTransformer)."""
    return tl.rand(seed, offs)


@triton.jit
def _randn(seed, offs):
    """random.gauss / normalvariate's standard normal, as the transpiler lowers it."""
    return tl.randn(seed, offs)


@triton.jit
def _popcount(x):
    v = x.to(tl.uint32)
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) >> 24).to(tl.int32)


@gluon.jit
def _segments_of(s, tv, g, dir_offs, dir_first, dir_words, W):
    """Per spike (source s), its segments into group g as [a, b), through
    the topology's group directory: three dependent steps (bitmap word +
    directory offset -> the pair's first segment and the next pair's -> the
    segments); a clear bit stops after the first. (Two interleaved binary
    searches over the source's segments instead were ~30% slower, PD14.)"""
    bit = g % 32
    at = 2 * (s * W + g // 32)
    wb = gl.load(dir_words + at, mask=tv, other=0)
    pre = gl.load(dir_words + at + 1, mask=tv, other=0)
    po = gl.load(dir_offs + s, mask=tv, other=0)
    has = tv & (((wb >> bit) & 1) != 0)
    pair = po + pre + _popcount(wb & ((1 << bit) - 1))
    a = gl.load(dir_first + pair, mask=has, other=0)
    b = gl.load(dir_first + pair + 1, mask=has, other=0)
    return a, b


@gluon.jit
def _table(x, LAYOUT: gl.constexpr):
    """x (1-D) as a read-only LDS table, looked up with .gather (a barrier after the stores)."""
    t = gl.allocate_shared_memory(x.dtype, x.shape, LAYOUT)
    t.store(x)
    return t


@gluon.jit
def _store_records(rec0, at, e, srcw, l, tgt, m, NUM_WARPS: gl.constexpr):
    """Ring records (e, source word, link index, target) as ONE 128-bit
    store each: the four words joined into a [records, 4] tensor whose rows
    sit whole in one thread. rec0: the ring's word 0; at: each record's
    first word (a multiple of 4); all 1-D, one per record."""
    N: gl.constexpr = at.shape[0]
    LR: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [NUM_WARPS, 1], [1, 0])
    # join adds a last dimension, so the nesting interleaves: join(join(a, c), join(b, d)) -> (a, b, c, d)
    words = gl.reshape(gl.join(gl.join(e.to(gl.int32, bitcast=True), l), gl.join(srcw, tgt)), [N, 4])
    words = gl.convert_layout(words, LR)
    col = gl.arange(0, 4, layout=gl.SliceLayout(0, LR))
    at2 = gl.multiple_of(gl.convert_layout(at, gl.SliceLayout(1, LR)), 4)
    m2 = gl.convert_layout(m, gl.SliceLayout(1, LR))
    gl.store(rec0 + at2[:, None] + col[None, :], words, mask=m2[:, None] & (col[None, :] < 4))


@gluon.jit
def _target_ptr(p, GROUP_SIZE: gl.constexpr):
    """A topology's link targets: uint8 up to a group size of 256, else uint16 (held as int16, read & 0xFFFF)
    -- SimulationTopology.target_bits."""
    if GROUP_SIZE <= 256:
        return p.to(gl.pointer_type(gl.uint8))
    else:
        return p.to(gl.pointer_type(gl.int16))


@gluon.jit
def _target_word(l, GROUP_SIZE: gl.constexpr):
    """The 32-bit word holding link l's target."""
    if GROUP_SIZE <= 256:
        return l >> 2
    else:
        return l >> 1


@gluon.jit
def _target_in(word, l, GROUP_SIZE: gl.constexpr):
    """Link l's target out of its 32-bit word."""
    if GROUP_SIZE <= 256:
        return (word >> ((l & 3) * 8)) & 0xFF
    else:
        return (word >> ((l & 1) * 16)) & 0xFFFF


# ------------------------------------------------- the ring's arenas (naer2/ring.py)
#
# A record's place: its run's block (_table_blk: fixed for a slot's run's first RESERVE blocks, else the
# run's block table), then its index in the group's arena (_blk_rec) -- _run_rec, both at once. A run outgrows
# its blocks only in the pull: there, _append_links writes past them into the cell's overflow log and
# _arena_fixup gives it blocks after the pull's last barrier; delivery frees them (_free_blocks).


@gluon.jit
def _run_rec(cell, pos, table, N_SLOTS: gl.constexpr, BLOCK: gl.constexpr, nb, RES: gl.constexpr):
    """(record index, valid) of entry pos of a cell's run in its group's arena, not valid where a block is
    missing (-1: the arena ran out when it was claimed; its links are lost). RES: the run's fixed blocks (a
    slot's RESERVE, a queue's none)."""
    return _blk_rec(cell, pos, _table_blk(table, cell, pos, N_SLOTS, BLOCK, nb, RES), N_SLOTS, BLOCK, nb)


@gluon.jit
def _table_blk(table, cell, pos, N_SLOTS: gl.constexpr, BLOCK: gl.constexpr, nb,
               RES: gl.constexpr):
    """The block id of entry pos of a cell's run: within its first RES blocks fixed (slot s's blocks s * RES
    ...: no table read), past them from its block table (a masked read: in quiet runs hardly any lane has
    one)."""
    fixed = pos < RES * BLOCK
    blk = gl.load(table + cell * nb + gl.minimum(pos // BLOCK, nb - 1), mask=~fixed, other=0)
    return gl.where(fixed, (cell % N_SLOTS) * RES + pos // BLOCK, blk)


@gluon.jit
def _blk_rec(cell, pos, blk, N_SLOTS: gl.constexpr, BLOCK: gl.constexpr, nb):
    """(record index, valid) of entry pos of a cell's run from its block id (_table_blk)."""
    return ((cell // N_SLOTS) * nb + gl.maximum(blk, 0)) * BLOCK + pos % BLOCK, blk >= 0


# ARENA_CHECK (a development aid, PopulationRuntime.arena_check: compiled out by default): every block a pull's
# fix-up pops and every block delivery frees, against a held count per (group, block) in device memory (chk:
# words 16.. -- rebuilt from the tables after every allocation and resize, runtime._arena_check_buf): a pop finds
# 0, a free 1, the id within the arena -- the first violation recorded (words 1..6: kind 1 pop / 2 free, group,
# cell, block id, the count found, the round (pop) or slot (free); word 0 counts them), nothing synchronized
ARENA_CHK_HDR = gl.constexpr(16)


@gluon.jit
def _arena_note(chk, bad, kind: gl.constexpr, g, cell, bid, old, when):
    """The first arena violation among the lanes `bad`, recorded (see ARENA_CHECK)."""
    z = gl.zeros_like(bid)
    n = gl.atomic_add(chk + z, 1, mask=bad)
    first = bad & (n == 0)
    gl.store(chk + 1 + z, z + kind, mask=first)
    gl.store(chk + 2 + z, z + g, mask=first)
    gl.store(chk + 3 + z, z + cell, mask=first)
    gl.store(chk + 4 + z, bid, mask=first)
    gl.store(chk + 5 + z, old, mask=first)
    gl.store(chk + 6 + z, z + when, mask=first)


@gluon.jit
def _arena_take(chk, v, g, cell, bid, nb, when, kind: gl.constexpr, delta: gl.constexpr, expect: gl.constexpr):
    """A block popped (delta +1, the count expected 0) or freed (-1, expected 1) on lanes v (ARENA_CHECK)."""
    ok_id = (bid >= 0) & (bid < nb)
    old = gl.atomic_add(chk + ARENA_CHK_HDR + g * nb + gl.where(ok_id, bid, 0), gl.zeros_like(bid) + delta,
                        mask=v & ok_id)
    old = gl.where(v & ok_id, old, expect)
    _arena_note(chk, v & (~ok_id | (old != expect)), kind, g, cell, bid, old, when)


@gluon.jit
def _arena_fixup(s_cnt, cnt0, rec0, table, stack, log_rec, ovf_cap, top0, g, sl, LV: gl.constexpr,
                N_SLOTS: gl.constexpr, run_cap, BLOCK: gl.constexpr, nb,
                NUM_WARPS: gl.constexpr, NSP: gl.constexpr, chk, when, ARENA_CHECK: gl.constexpr):
    """After the pull's last barrier, only if some slot grew past its blocks (from the counts in LDS: the same
    answer in every warp). First the check: everything fits -- every slot's log within ovf_cap, the blocks
    the slots lack within the free stack, every run within run_cap. If not, nothing is changed (the pull
    wrote only past the committed counts and into the logs: scratch) and the run stops (fits False: the
    host resizes and runs it again). If it fits, every such slot takes what it needs -- + as much again as
    it grew this run where the stack has room for all of that (in a burst, the next run then mostly fits)
    -- and its log's records move into the new blocks. Returns (fits, the new top, the blocks the group
    holds -- would hold, if it didn't fit --, the most any slot put into its log): the run's status, the
    runtime sizes the arenas and logs from it."""
    MASK: gl.constexpr = (1 << CNT_BITS) - 1
    packed = s_cnt.load(LV)
    cnt = packed & MASK
    held = packed >> CNT_BITS
    live = sl < N_SLOTS
    need = (cnt + BLOCK - 1) // BLOCK
    new_top = top0
    most = gl.to_tensor(0)
    lack = gl.where(live & (need > held), need - held, 0)
    # (past what a run can hold at all: never fits; past the arena alone, lack > the free stack already)
    lack = gl.where(live & (cnt > MAX_BLOCKS * BLOCK), 1 << 24, lack)
    total = gl.sum(lack, axis=0)  # (the one reduction a run without growth pays)
    fits = gl.to_tensor(1) > 0
    if total > 0:
        over = gl.where(live, gl.maximum(cnt - held * BLOCK, 0), 0)  # records in each slot's log
        most = gl.max(over, axis=0)
        fits = fits & (most <= ovf_cap) & (total <= top0)
        if fits:
            grown = (gl.maximum(cnt - cnt0, 0) + BLOCK - 1) // BLOCK
            want = gl.where(lack > 0, gl.minimum(need + grown, MAX_BLOCKS) - held, 0)
            want = gl.minimum(want, nb - held)
            if gl.sum(want, axis=0) > top0:  # no room for the headroom: just what they lack
                want = lack
            before = gl.cumsum(want, axis=0) - want
            cell = g * N_SLOTS + sl
            for i in range(0, gl.max(want, axis=0)):
                bid = gl.load(stack + g * nb + gl.maximum(top0 - 1 - before - i, 0), mask=i < want, other=-1)
                gl.store(table + cell * nb + held + i, bid, mask=live & (i < want))
                if ARENA_CHECK:
                    _arena_take(chk, live & (i < want), g, cell, bid, nb, when, 1, 1, 0)
            s_cnt.store((packed & MASK) | ((held + want) << CNT_BITS))
            gl.barrier()
            # every slot's log into its new blocks, all slots' records in one pass over the lanes: record f of
            # the flat list belongs to the last slot whose first record is <= f (a binary search over the slots'
            # firsts in LDS; slots logging nothing share their first with the next one), its index in that
            # slot's log f - first (one scan for all slots, not a pass with two reductions per slot)
            SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
            first = gl.cumsum(over, axis=0) - over
            n_all = gl.sum(over, axis=0)
            t_first = _table(first, SM)
            t_h = _table(held * BLOCK, SM)
            gl.barrier()
            pq = gl.arange(0, 32 * NUM_WARPS, layout=LV)
            for f0 in range(0, n_all, 32 * NUM_WARPS):
                fl = f0 + pq
                m = fl < n_all
                s_f = gl.zeros_like(fl)
                for st in gl.static_range(10):  # (NSP <= 1024: a probe stride of 0 past log2(NSP) steps)
                    probe = s_f + (NSP >> (st + 1))
                    s_f = gl.where(t_first.gather(probe, 0) <= fl, probe, s_f)
                idx = fl - t_first.gather(s_f, 0)
                p = t_h.gather(s_f, 0) + idx
                cs = g * N_SLOTS + s_f
                blk = gl.load(table + cs * nb + gl.minimum(p // BLOCK, nb - 1), mask=m, other=0)
                dst = 4 * ((g * nb + blk) * BLOCK + p % BLOCK)
                src = 4 * (log_rec + cs * ovf_cap + idx)
                for fw in gl.static_range(4):
                    gl.store(rec0 + dst + fw, gl.load(rec0 + src + fw, mask=m), mask=m)
            gl.barrier()
            new_top = top0 - gl.sum(want, axis=0)
    used = gl.where(fits, nb - new_top, nb - top0 + total)
    return fits, new_top, used, most


@gluon.jit
def _free_blocks(table, stack, tp, g, cell, lo, hi, lost, LV: gl.constexpr, nb,
                 NUM_WARPS: gl.constexpr, chk, when, ARENA_CHECK: gl.constexpr):
    """A delivered run's blocks lo .. hi back on group g's free stack, above tp (delivery keeps the stack's
    top in a register and stores it once at its end: no barriers here). Once an arena ran out (lost: tables
    may hold missing blocks, -1), they are compacted on the way (a scan): the stack never holds a -1. Returns
    the new top."""
    bi = gl.arange(0, 32 * NUM_WARPS, layout=LV)
    if lost == 0:
        for i0 in range(lo, hi, 32 * NUM_WARPS):
            m = i0 + bi < hi
            bid = gl.load(table + cell * nb + i0 + bi, mask=m, other=-1)
            gl.store(stack + g * nb + tp + i0 - lo + bi, bid, mask=m)
            if ARENA_CHECK:
                _arena_take(chk, m, g, cell, bid, nb, when, 2, -1, 1)
        tp = tp + gl.maximum(hi - lo, 0)
    else:
        for i0 in range(lo, hi, 32 * NUM_WARPS):
            bid = gl.load(table + cell * nb + i0 + bi, mask=i0 + bi < hi, other=-1)
            v = (bid >= 0).to(gl.int32)
            gl.store(stack + g * nb + tp + gl.cumsum(v, axis=0) - v, bid, mask=v > 0)
            if ARENA_CHECK:
                _arena_take(chk, v > 0, g, cell, bid, nb, when, 2, -1, 1)
            tp = tp + gl.sum(v, axis=0)
    return tp


@gluon.jit
def _link_record(rec0, word, blk, j, w, n_l, lo_l, base, lim, cell, e2, srcw2, log_rec, ovf_cap,
                 T: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr, N_SLOTS: gl.constexpr,
                 BLOCK: gl.constexpr, nb):
    """Every tile segment's j-th link as its ring record; word: the 32-bit word holding its target; blk: its
    record's block id (_table_blk). A position past the blocks its cell holds (lim) goes into
    its cell's overflow log instead (entry position - lim, ovf_cap entries per cell; log_rec: the index of
    the logs' records; _arena_fixup moves it) -- either way one store of the record, its address arithmetic,
    nothing loaded."""
    wj = w & (j < n_l)
    l = lo_l + j
    tgt = _target_in(word, l, GROUP_SIZE)
    pos = base + j
    rec, ok = _blk_rec(cell, pos, blk, N_SLOTS, BLOCK, nb)
    direct = pos < lim
    in_log = (~direct) & (pos - lim < ovf_cap)  # (the cell's log full: not written, _arena_fixup stops the run)
    rec = gl.where(direct, rec, log_rec + cell * ovf_cap + (pos - lim))
    ok = gl.where(direct, ok, in_log)
    at = 4 * rec
    _store_records(rec0, gl.reshape(at, [T]), gl.reshape(e2, [T]), gl.reshape(srcw2, [T]), gl.reshape(l, [T]),
                   gl.reshape(tgt, [T]), gl.reshape(wj & ok, [T]), NUM_WARPS)


@gluon.jit
def _tile_segments(f0, n_pairs, lane, t_first, t_a, t_gb, t_st, SPK: gl.constexpr):
    """Tile f0's lanes: each its (spike, segment) pair -- its spike by binary search over the spikes' first
    pairs -- and the segment's group-bucket word and link range. Returns (fl, ok, sp, seg, gb, lo_l, n_l)."""
    fl = f0 + lane
    ok = fl < n_pairs
    sp = gl.zeros_like(fl)
    for st in gl.static_range(6):  # (SPK <= 64: a probe stride of 0 past log2(SPK) steps)
        probe = sp + (SPK >> (st + 1))
        sp = gl.where(t_first.gather(probe, 0) <= fl, probe, sp)
    seg = t_a.gather(sp, 0) + (fl - t_first.gather(sp, 0))
    gb = gl.load(t_gb.gather(sp, 0).to(gl.pointer_type(gl.int32)) + seg, mask=ok, other=0)
    stp = t_st.gather(sp, 0).to(gl.pointer_type(gl.int32))
    lo_l = gl.load(stp + seg, mask=ok, other=0)
    n_l = gl.load(stp + seg + 1, mask=ok, other=0) - lo_l
    return fl, ok, sp, seg, gb, lo_l, n_l


@gluon.jit
def _append_links(s, e, take, c, a, b, seg_start, seg_gb, bucket_mean, link_tgt, s_cnt, g, k_lo, slot_width,
                  rec_time, rec_src, rec_lo, rec_n, flags, trace, n_ev, table, log_rec, ovf_cap,
                  N_SLOTS: gl.constexpr, run_cap, LP2: gl.constexpr, LM: gl.constexpr,
                  SPK: gl.constexpr, MSEG: gl.constexpr,
                  TRACE: gl.constexpr, TRACE_CAP: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
                  BLOCK: gl.constexpr, nb):
    """A block of spikes (uniform sources): every segment [a, b) of the
    ones taken (take) claims a range of its slot's run in group
    g's ring -- the per-slot link counts live in LDS (s_cnt), the program
    being the run's only writer -- and writes one record per link there, in
    link order: (e, source word, link index, target). The block's segments
    packed into tiles of one lane each (32 x NUM_WARPS)."""
    # Packed tiles: the block's (spike, segment) pairs as one dense list, T = one lane each (no replicated
    # lanes) -- a tile's work follows the block's segments in all, not its spike with the most (a spike
    # reaches a group through a segment per delay bucket: up to ~20 in PD14, mostly 1-2 links each). A lane
    # finds its spike by binary search over the spikes' first pairs, and the spike's values in LDS tables.
    SMT: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    T: gl.constexpr = 32 * NUM_WARPS
    LT: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    m = gl.where(take, b - a, 0)
    srcw_s = (c << 24) | s
    zero = gl.zeros_like(s)  # per-spike pointers: vectors (batched) or the run's scalars, broadcast
    m1 = gl.convert_layout(m, LT)  # (a scan needs a blocked layout, not the tile's slice)
    first_m = gl.cumsum(m1, axis=0) - m1
    n_pairs = gl.sum(m1, axis=0)
    t_first = _table(first_m, SMT)
    t_a = _table(a + zero, SMT)
    t_e = _table(e + zero.to(gl.float32), SMT)
    t_srcw = _table(srcw_s, SMT)
    t_gb = _table((seg_gb + zero).to(gl.int64), SMT)
    t_st = _table((seg_start + zero).to(gl.int64), SMT)
    t_mean = _table((bucket_mean + zero).to(gl.int64), SMT)
    t_lt = _table((link_tgt + zero).to(gl.int64), SMT)
    gl.barrier()
    lane = gl.arange(0, T, layout=LT)
    for f0 in range(0, n_pairs, T):
        # (the next tile's segments loaded here, before this tile's claim and stores -- a pipeline over the
        # tiles -- was no faster on the 4090, with registers to spare or not: limitations.md)
        fl, ok, sp, seg, gb, lo_l, n_l = _tile_segments(f0, n_pairs, lane, t_first, t_a, t_gb, t_st, SPK)
        e1 = t_e.gather(sp, 0)
        srcw2 = t_srcw.gather(sp, 0)
        if TRACE >= 3:
            gb = pin(gb)
            n_l = pin(n_l)
            n_ev = mark(trace, n_ev, 14, gl.sum(ok.to(gl.int32), axis=0), TRACE_CAP)
        mean = gl.load(t_mean.gather(sp, 0).to(gl.pointer_type(gl.float32)) + (gb & 0xFFFF), mask=ok, other=0.0)
        if TRACE >= 3:
            mean = pin(mean)
            n_ev = mark(trace, n_ev, 15, 0, TRACE_CAP)
        k = gl.maximum(gl.floor((e1 + mean) / slot_width).to(gl.int32), k_lo)
        far = ok & (k - k_lo >= N_SLOTS)
        # the flags per lane (masked: normally no lane), not through reductions across warps
        gl.atomic_max(flags + 1 + gl.zeros_like(k), 1, mask=far)
        w = ok & ~far
        n_iter = gl.max(gl.where(w, n_l, 0), axis=0)  # before the full check: a bound
        slot = k % N_SLOTS
        n_rec = gl.where(w, n_l, 0)  # (a record per link)
        old = s_cnt.atomic_scatter_add(n_rec, slot, 0)  # the range's start | blocks held
        old = gl.convert_layout(old, LT)
        # the claim's start and its cell's blocks come back together: a lane knows alone which of its
        # positions its cell holds (written directly) and which it doesn't (into the cell's overflow log)
        # -- no lane waits on another, no barrier
        base = old & ((1 << CNT_BITS) - 1)
        lim = (old >> CNT_BITS) * BLOCK
        if TRACE >= 3:
            base = pin(base)
            n_ev = mark(trace, n_ev, 16, 0, TRACE_CAP)
        w = w & ~(base + n_rec > run_cap)  # past the capacity: not written (the pull's end stops the run)
        cell = g * N_SLOTS + slot
        # link by link over the tile: iteration j writes every segment's j-th link (work ~ the tile's longest
        # segment: in PD14 a few links); four links per iteration (_link_loads: their loads in flight together)
        e2 = e1
        lt32 = t_lt.gather(sp, 0).to(gl.pointer_type(gl.int32))  # the targets' words (4-byte aligned)
        rec0 = rec_src - 1
        for j in range(0, n_iter, 4):
            x0 = gl.load(lt32 + gl.where(w & (j < n_l), _target_word(lo_l + j, GROUP_SIZE), 0))
            x1 = gl.load(lt32 + gl.where(w & (j + 1 < n_l), _target_word(lo_l + j + 1, GROUP_SIZE), 0))
            x2 = gl.load(lt32 + gl.where(w & (j + 2 < n_l), _target_word(lo_l + j + 2, GROUP_SIZE), 0))
            x3 = gl.load(lt32 + gl.where(w & (j + 3 < n_l), _target_word(lo_l + j + 3, GROUP_SIZE), 0))
            # the records' block ids, in flight with the targets: one round trip per four links
            k0 = _table_blk(table, cell, base + j, N_SLOTS, BLOCK, nb, RESERVE)
            k1 = _table_blk(table, cell, base + j + 1, N_SLOTS, BLOCK, nb, RESERVE)
            k2 = _table_blk(table, cell, base + j + 2, N_SLOTS, BLOCK, nb, RESERVE)
            k3 = _table_blk(table, cell, base + j + 3, N_SLOTS, BLOCK, nb, RESERVE)
            if TRACE >= 3:
                x3 = pin(x3)
                n_ev = mark(trace, n_ev, 17, 0, TRACE_CAP)
            _link_record(rec0, x0, k0, j, w, n_l, lo_l, base, lim, cell, e2, srcw2, log_rec,
                         ovf_cap, T, NUM_WARPS, GROUP_SIZE, N_SLOTS, BLOCK, nb)
            _link_record(rec0, x1, k1, j + 1, w, n_l, lo_l, base, lim, cell, e2, srcw2, log_rec,
                         ovf_cap, T, NUM_WARPS, GROUP_SIZE, N_SLOTS, BLOCK, nb)
            _link_record(rec0, x2, k2, j + 2, w, n_l, lo_l, base, lim, cell, e2, srcw2, log_rec,
                         ovf_cap, T, NUM_WARPS, GROUP_SIZE, N_SLOTS, BLOCK, nb)
            _link_record(rec0, x3, k3, j + 3, w, n_l, lo_l, base, lim, cell, e2, srcw2, log_rec,
                         ovf_cap, T, NUM_WARPS, GROUP_SIZE, N_SLOTS, BLOCK, nb)
            if TRACE >= 3:
                n_ev = mark(trace, n_ev, 18, 0, TRACE_CAP)
    return n_ev


# ------------------------------------------------- stopping and rerunning a run


# one run's entry in a schedule (naer2/schedule.py: RUN_WORDS int32 words), what a launch would otherwise pass
# as arguments: run id, round id, the slots it delivers (k_lo .. k_hi), its window (t_start, t_end), gated,
# its status slot (-1: the run id mod n_status)
RUN_WORDS = gl.constexpr(8)


@gluon.jit
def _run_entry(sched, sched_i):
    """Entry sched_i of a run schedule: (run_id, round_id, k_lo, k_hi, t_start, t_end, gated) -- read by every
    program before anything else (the schedule: device memory for a planned window, pinned host memory for a
    direct run; naer2/schedule.py)."""
    e = sched + RUN_WORDS * sched_i
    run_id = gl.load(e)
    round_id = gl.load(e + 1)
    k_lo = gl.load(e + 2)
    k_hi = gl.load(e + 3)
    t_start = gl.load((e + 4).to(gl.pointer_type(gl.float32)))
    t_end = gl.load((e + 5).to(gl.pointer_type(gl.float32)))
    gated = gl.load(e + 6)
    return run_id, round_id, k_lo, k_hi, t_start, t_end, gated


@gluon.jit
def _status_slot(status, sched, sched_i, run_id, n_status, n_groups):
    """This run's status slot (4 n_groups words, _run_status): the entry's word 7 -- a captured window's runs
    each their own, in device memory (naer2/graph.py) -- or, -1, slot run_id mod n_status (a direct run's or
    a planned window's: the runtime's pinned host slots, written straight from the kernel)."""
    slot = gl.load(sched + RUN_WORDS * sched_i + 7)
    slot = gl.where(slot >= 0, slot, run_id % n_status)
    return status + slot * (4 * n_groups)


# A runtime's event state, [2, EV_HDR + n_groups] int32 double-buffered by run parity (run r reads row r % 2,
# every program writes row (r + 1) % 2: no program reads what another of the same run writes): the ring's
# delivery position (word 0), the runs made (word 1: a skipped gated run doesn't count), and per group its
# first slot still holding records (word EV_HDR + g; INT_MAX: none) -- what a gated run decides on, and where
# it delivers from (a skipped run delivers nothing: the host can't advance the position for it).
EV_HDR = gl.constexpr(2)
EV_NONE = gl.constexpr(2147483647)


@gluon.jit
def _gate_skip(gated, plan, n_conns, ev, run_id, round_id, t_start, t_end, slot_width, n_groups, SK: gl.constexpr,
               KP: gl.constexpr, MAX_CONNS: gl.constexpr, NUM_WARPS: gl.constexpr, QUEUES: gl.constexpr = False):
    """(skip, start) of a gated run (an event-driven population's planned launch: Network.plan): its due time,
    as the host would decide it (Population._next_run_time), is the earliest of (A) every fresh source run with
    spikes -- its earliest emission + its connection's minimum delay (intake's) -- and (B) the end of the first
    slot still holding records after the population's last run (any group's: _note_pending_arrivals'). Nothing
    due before its window's end: skip. Else the run starts at the due time, if later than its planned start --
    a launch is planned at the earliest time an arrival could be due (sources with precise spike times: a little
    later), and covers its window: the run exactly the stepped one (its start, its bare spikes, its first
    slot). Every program decides alike (from the same headers and event state). An ungated run never skips,
    starts as planned."""
    skip = gated != 0
    start = t_start
    if gated != 0:
        LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
        MAX_CJ: gl.constexpr = MAX_CONNS * KP
        (hist_addr, _ta, _lt, _aa, _da, _hc, _ts, _tg, _tb, _tl, _dp, _dw, last_read) = _fused_plan(plan, n_conns)
        cj = gl.arange(0, MAX_CJ, layout=LV)
        c_of = cj // KP
        j_of = cj % KP
        vj = (c_of < n_conns) & (j_of < SK)
        hbuf = gl.load(hist_addr + c_of, mask=vj, other=0)
        hdr = hbuf.to(gl.pointer_type(gl.int32)) + 8 * j_of
        tag = gl.load(hdr.to(gl.pointer_type(gl.int64)), mask=vj, other=0)
        rid = (tag & 0xFFFFFFFF).to(gl.int32)
        rnd = (tag >> 32).to(gl.int32)
        cnt = gl.load(hdr + 2, mask=vj, other=0)
        early = gl.load((hdr + 4).to(gl.pointer_type(gl.float32)), mask=vj, other=float("inf"))
        lastj = gl.load(last_read + (run_id % 2) * n_conns + c_of, mask=vj, other=0)
        dmin = gl.load((plan + 20 * n_conns + c_of).to(gl.pointer_type(gl.float32)), mask=vj, other=0.0)
        fresh = vj & (rid > lastj) & (rnd < round_id) & (cnt > 0)
        due = gl.min(gl.where(fresh, early + dmin, float("inf")), axis=0)
        evp = ev + (run_id % 2) * (EV_HDR + n_groups)
        gi = gl.arange(0, 32 * NUM_WARPS, layout=LV)
        if QUEUES:  # (each group's next arrival, exact: _ev_close; never written yet: NaN bits)
            nxt = gl.to_tensor(float("inf"))
            for g0 in range(0, n_groups, 32 * NUM_WARPS):
                v = gl.load(evp + EV_HDR + g0 + gi, mask=g0 + gi < n_groups, other=EV_NONE).to(gl.float32,
                                                                                                bitcast=True)
                nxt = gl.minimum(nxt, gl.min(gl.where(v == v, v, float("inf")), axis=0))
            due = gl.minimum(due, nxt)
        else:
            first = gl.to_tensor(EV_NONE)
            for g0 in range(0, n_groups, 32 * NUM_WARPS):
                v = gl.load(evp + EV_HDR + g0 + gi, mask=g0 + gi < n_groups, other=EV_NONE)
                first = gl.minimum(first, gl.min(v, axis=0))
            due = gl.minimum(due, gl.where(first < EV_NONE, (first + 1).to(gl.float32) * slot_width, float("inf")))
        skip = due >= t_end - 1e-6 * gl.maximum(gl.abs(t_end), 1.0)  # (due at its end: the next window's)
        start = gl.maximum(t_start, due)
    return skip, start


@gluon.jit
def _gate_carry(plan, n_conns, ev, pid, run_id, n_groups, MAX_CONNS: gl.constexpr, NUM_WARPS: gl.constexpr):
    """A skipped gated run: what the next run reads carried over unchanged -- the last-read run ids (the
    sources' runs stay unread) and the event state (this program's group's, and the delivery position)."""
    LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    (_ha, _ta, _lt, _aa, _da, _hc, _ts, _tg, _tb, _tl, _dp, _dw, last_read) = _fused_plan(plan, n_conns)
    ci = gl.arange(0, MAX_CONNS, layout=LV)
    cv = ci < n_conns
    p = run_id % 2
    gl.store(last_read + (1 - p) * n_conns + ci, gl.load(last_read + p * n_conns + ci, mask=cv), mask=cv)
    src = ev + p * (EV_HDR + n_groups)
    dst = ev + (1 - p) * (EV_HDR + n_groups)
    gl.store(dst, gl.load(src))
    gl.store(dst + 1, gl.load(src + 1))
    gl.store(dst + EV_HDR + pid, gl.load(src + EV_HDR + pid))


@gluon.jit
def _ev_close(ev, ring, pid, run_id, k_lo, k_hi, n_groups, N_SLOTS: gl.constexpr, NSP: gl.constexpr,
              NUM_WARPS: gl.constexpr, BLOCK: gl.constexpr, QUEUES: gl.constexpr = False, plan=0, n_conns=0):
    """A run done: the event state for the next run -- the delivery position past this run's slots, and this
    group's first slot still holding records (after delivery: later per-link arrivals of spikes taken in);
    QUEUES: instead its next arrival, the earliest queue head's e + D_c (float bits; inf: none)."""
    LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    (rec_time, _ms, _ml, _mn, run_count, head_p, _fl, table, _sk, _tp, _nbk, _lo, _lr, _oc, _mc, nb,
     _sts) = _ring(ring, BLOCK)
    rp = gl.maximum(k_lo, k_hi + 1)
    sl = gl.arange(0, NSP, layout=LV)
    live = sl < N_SLOTS
    if QUEUES:
        cell = pid * N_SLOTS + sl
        head = gl.load(head_p + cell, mask=live, other=0)
        has = live & (head < gl.load(run_count + cell, mask=live, other=0))
        blk = gl.load(table + cell * nb + head // BLOCK, mask=has, other=0)
        e = gl.load(rec_time + 4 * ((pid * nb + blk) * BLOCK + head % BLOCK), mask=has, other=float("inf"))
        d = gl.load((plan + 20 * n_conns + sl).to(gl.pointer_type(gl.float32)), mask=has, other=0.0)
        first = gl.min(gl.where(has, e + d, float("inf")), axis=0).to(gl.int32, bitcast=True)
    else:
        k = rp + sl
        c = gl.load(run_count + pid * N_SLOTS + k % N_SLOTS, mask=live, other=0)
        first = gl.min(gl.where(live & (c > 0), k, EV_NONE), axis=0)
    src = ev + (run_id % 2) * (EV_HDR + n_groups)
    dst = ev + (1 - run_id % 2) * (EV_HDR + n_groups)
    gl.store(dst, rp)
    gl.store(dst + 1, gl.load(src + 1) + 1)
    gl.store(dst + EV_HDR + pid, first)


@gluon.jit
def _run_from(ev, parity, pid, n_groups, k_lo, k_hi, t_start, slot_width):
    """The slot a gated run of program pid starts from -- its delivery's first and its pull's ring base
    (arrivals before it go into it: late; at or past the ring's length past it: far): k_lo (where the
    population's last run left off), or later where the group's ring holds nothing before -- its first slot
    still holding records (_ev_close), at most the run's planned start's slot (and k_hi). An event-driven
    population idle for a while starts at its run, not at its last run's end: the slots between are empty,
    and arrivals a ring's length past that end aren't far. (A stepped run: the host's, Ring.first_pending.)"""
    first = gl.load(ev + parity * (EV_HDR + n_groups) + EV_HDR + pid)
    ks = gl.floor(t_start / slot_width).to(gl.int32)
    return gl.maximum(k_lo, gl.minimum(gl.minimum(first, ks), k_hi))


@gluon.jit
def _run_gate(ring, ctl, pid, run_id):
    """Whether program pid takes part in run run_id: not once a run stopped (ctl, the launches' stop word:
    every run that starts after a stop waits for the host to rerun it), not if its group finished this run
    already (a rerun: the groups that finished before the stop skip). Read before anything is written."""
    done = ring + gl.load(ring + 6)
    return (gl.load(ctl) == 0) & (gl.load(done + pid) < run_id)


@gluon.jit
def _run_done(ring, pid, run_id):
    """Program pid's group finished run run_id (in the ring: what a rerun skips)."""
    gl.store(ring + gl.load(ring + 6) + pid, run_id)


@gluon.jit
def _run_status(status, ring, pid, run_id, finished, chk, ARENA_CHECK: gl.constexpr):
    """Program pid's last stores: its group's part of the run's status, straight into the run's pinned host
    slot (PopulationRuntime: no copy, no event) -- the blocks it holds and the most it logged (the pull's
    group stats), its ring's lost bit (1: links lost), and its done word: 2 run_id if it
    finished the run, 2 run_id + 1 if it stopped (or skipped: a run stopped before it) -- the host reads the
    whole run from two reductions (lost bits: + 4 if the arena check found a violation, ARENA_CHECK):
    landed once the least done word is >= 2 run_id (a slot's earlier runs
    wrote less), stopped if the largest is odd. Plain stores, done last: the host decides from the done
    words alone whether to rerun (and then reads the ring itself); the rest only sizes."""
    st = ring + gl.load(ring + 5) + 2 * pid
    fl = ring + gl.load(ring + 3)
    at = status + 4 * pid
    gl.store(at, gl.load(st))
    gl.store(at + 1, gl.load(st + 1))
    bits = gl.load(fl)
    if ARENA_CHECK:  # (4: the arena check recorded a violation -- the host raises when it judges this run)
        bits = bits | ((gl.load(chk) > 0).to(gl.int32) << 2)
    gl.store(at + 2, bits)
    gl.store(at + 3, 2 * run_id + gl.where(finished, 0, 1))


# ------------------------------------------------- delivery helpers


# receive's time (D-1, BRUNEL.md): T_A e + T_C k + T_B for a record emitted at e delivered from slot k -- by the
# population's incoming delays: one homogeneous delay D (1, 0, D: exact, e + D -- emission order is arrival order);
# any per-link delays (0, w, w / 2 or w: the slot's center or end -- every link of a slot at one time, so the
# times a target sees never step back); (1, 0, 0) the emission time itself (the reference kernels)
@gluon.jit
def _deliver_link(state, t, w, srcword, e, received, n_received, lane, mask, k, pid, log_ptr, log_e, log_cap,
                  RECEIVE: gl.constexpr, GROUP_SIZE: gl.constexpr, LOG: gl.constexpr, COUNT: gl.constexpr, rtime,
                  ectx):
    """ONE link: t its target within the group (uniform over the program,
    -1 for none), w its weight, e its time word. Every lane compares
    against t and the target lane runs the receive. A multapse's weights
    arrive one link at a time, in link order."""
    hit = (lane == t) & mask
    # RECEIVE(state, hit, time, w, srcword) -> state is the transpiled receive, rooted at
    #   `hit` (MaskAliasTransformer's is_target: the one target lane); state: the node attrs it touches, as a
    #   tuple; edge.x -> the link's staged value of x (here w); time = rtime (D-1: the arrival e + D, or the
    #   slot's time -- see above); the code chosen by the link's receive variant, a scalar (warp-uniform) branch on its
    #   connection id (D-6)
    # send runs here too, after every delivery, rooted at `hit` (D-2) -- and once more after
    #   update; only for connections whose receive writes a node attr send reads (fused_template's
    #   per-delivery send: PD14_neuron has none). Its n2.spike() (and receive's) fires any number of times per
    #   neuron and run (D-3): straight into the run's history (_emit_delivered; ectx: its slot, nodes: the lanes'
    #   neurons)
    # GAP (D-10): source.x is unsupported for now
    state = RECEIVE(state, hit, rtime, w, srcword, ectx, pid * GROUP_SIZE + lane)  # (rtime: the delivery time)
    if COUNT:  # dev: the stand-in's bookkeeping (tests)
        received = gl.where(hit, received + w, received)
        n_received = n_received + hit.to(gl.int32)
    if LOG:
        if t >= 0:
            at_log = gl.atomic_add(log_ptr, 1)
            if at_log < log_cap:
                rec = log_ptr + 1 + 4 * at_log
                gl.store(rec, k)
                gl.store(rec + 1, srcword >> 24)
                gl.store(rec + 2, srcword & 0xFFFFFF)
                gl.store(rec + 3, pid * GROUP_SIZE + t)
                gl.store(log_e + at_log, e)
    return state, received, n_received


@gluon.jit
def _pin_first(state):
    """pin() on the state's first member (trace stamps after the work that produced it)."""
    if len(state) > 0:
        state = (pin(state[0]),) + state[1:]
    return state


@gluon.jit
def _walk(r_tgt, r_w, r_src, r_e, n_w, n_max, state, received, n_received, lane2, mask2, k, pid, log_ptr, log_e,
          log_cap, LW: gl.constexpr, RECEIVE: gl.constexpr, GROUP_SIZE: gl.constexpr, U: gl.constexpr,
          NUM_WARPS: gl.constexpr, LOG: gl.constexpr, COUNT: gl.constexpr, STAGE_E: gl.constexpr, tk,
          T_A: gl.constexpr, ectx, node2):
    """The staged stream block (_stage_records): every warp its own list, the links to its own neurons in
    link order, U of them a broadcast LDS read ([NUM_WARPS, U]: each warp its row), received one by one by
    their target lanes. state, received, n_received: [NUM_WARPS, GROUP_SIZE // NUM_WARPS], a row per warp
    (_to_rows); n_w: each warp's links [NUM_WARPS] (LW), n_max the most; a list past its links: masked."""
    # GAP (D-7): one broadcast LDS read and one register select per staged attr slot (here: w);
    #   up to the most attrs any variant's receive reads (D-7)
    # GAP (D-6): per link, its variant from srcword >> 24 is selected alongside -- a variant is the same for all links of one source spike through one connection,
    #   but links of different spikes and connections interleave here, so the choice is made per link
    L3: gl.constexpr = gl.BlockedLayout([1, U], [1, 32], [NUM_WARPS, 1], [1, 0])  # a warp's row whole per thread
    SM2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    u3 = gl.arange(0, U, layout=gl.SliceLayout(0, L3))
    for g in range(0, (n_max + U - 1) // U):
        tg = r_tgt.index(g).reinterpret(gl.int32, [NUM_WARPS, U], SM2).load(L3)
        wg = r_w.index(g).reinterpret(gl.float32, [NUM_WARPS, U], SM2).load(L3)
        if LOG:
            sg = r_src.index(g).reinterpret(gl.int32, [NUM_WARPS, U], SM2).load(L3)
        if STAGE_E:
            eg = r_e.index(g).reinterpret(gl.float32, [NUM_WARPS, U], SM2).load(L3)
        for u in gl.static_range(U):
            sel = u3[None, :] == u  # in-thread: a register select from each warp's row
            t = gl.convert_layout(gl.sum(gl.where(sel, tg, 0), axis=1), LW)
            w = gl.convert_layout(gl.sum(gl.where(sel, wg, 0.0), axis=1), LW)
            live = (g * U + u) < n_w
            if LOG:
                srcword = gl.convert_layout(gl.sum(gl.where(sel, sg, 0), axis=1), LW)
            else:
                srcword = 0
            if STAGE_E:
                e = gl.convert_layout(gl.sum(gl.where(sel, eg, 0.0), axis=1), LW)
            else:
                e = gl.zeros_like(w)
            hit = (lane2 == t[:, None]) & mask2 & live[:, None]
            # RECEIVE(state, hit, time, w, srcword) -> state: the transpiled receive (_deliver_link), rooted at
            #   `hit`, each warp's one target lane -- elementwise, every warp's link's values along its row
            state = RECEIVE(state, hit, (e * T_A + tk)[:, None], w[:, None], srcword, ectx, node2)  # (the delivery time)
            if COUNT:  # dev: the stand-in's bookkeeping (tests)
                received = gl.where(hit, received + w[:, None], received)
                n_received = n_received + hit.to(gl.int32)
            if LOG:  # each warp's link: its record in the delivery log (in delivery order per target)
                lv = live & (t >= 0)
                at_log = gl.atomic_add(log_ptr + gl.zeros_like(t), 1, mask=lv)
                lw = lv & (at_log < log_cap)
                rec = log_ptr + 1 + 4 * at_log
                gl.store(rec, gl.zeros_like(t) + k, mask=lw)
                gl.store(rec + 1, srcword >> 24, mask=lw)
                gl.store(rec + 2, srcword & 0xFFFFFF, mask=lw)
                gl.store(rec + 3, pid * GROUP_SIZE + t, mask=lw)
                gl.store(log_e + at_log, e, mask=lw)
    return state, received, n_received


@gluon.jit
def _to_rows(x, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr, LWN: gl.constexpr):
    """A [GROUP_SIZE] neuron tensor (the program's layout) as delivery's rows, one per warp (_walk)."""
    return gl.convert_layout(gl.reshape(x, [NUM_WARPS, GROUP_SIZE // NUM_WARPS]), LWN)


@gluon.jit
def _from_rows(x, GROUP_SIZE: gl.constexpr, LN: gl.constexpr):
    """Delivery's rows back to the program's [GROUP_SIZE] layout."""
    return gl.convert_layout(gl.reshape(x, [GROUP_SIZE]), LN)


@gluon.jit
def _state_to_rows(state, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr, LWN: gl.constexpr):
    out = ()
    for i in gl.static_range(len(state)):
        out = out + (_to_rows(state[i], NUM_WARPS, GROUP_SIZE, LWN),)
    return out


@gluon.jit
def _state_from_rows(state, GROUP_SIZE: gl.constexpr, LN: gl.constexpr):
    out = ()
    for i in gl.static_range(len(state)):
        out = out + (_from_rows(state[i], GROUP_SIZE, LN),)
    return out


@gluon.jit
def _stage_records(rec0, cell, b0, nm, s_atp, f_tgt, f_w, f_src, f_e, table, LW: gl.constexpr,
                   STREAM: gl.constexpr, U: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
                   LOG: gl.constexpr, STAGE_E: gl.constexpr, N_SLOTS: gl.constexpr, run_cap, BLOCK: gl.constexpr,
                   nb, STAGE_W: gl.constexpr):
    """Links b0 .. b0 + STREAM of a slot's run staged in LDS for the walk, each into its target warp's
    list: each record ONE 128-bit load, split into its fields in registers, the weight by link index through
    the connection's attribute pointer (a read-only LDS table); then a stable partition by target warp (one
    scan over the block: a link's place in its warp's list -- per target, link order kept), its fields
    scattered there. f_*: flat [STREAM // U, NUM_WARPS, U] buffers, link r of warp q's list at
    (r // U) NUM_WARPS U + q U + r % U (_walk reads a warp's U links at once). The block's padding (past
    nm: the block's end, last in every warp's scan) is spread over the lists after their real links -- no
    list holds more than STREAM -- past their counts. Returns (the links in the block, each warp's count
    [NUM_WARPS] in LW, the most)."""
    LR: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [NUM_WARPS, 1], [1, 0])
    LRQ: gl.constexpr = gl.SliceLayout(1, LR)
    q = b0 + gl.arange(0, STREAM, layout=LRQ)
    qv = q < nm
    rec, ok = _run_rec(cell, q, table, N_SLOTS, BLOCK, nb, RESERVE)
    qv = qv & ok
    at = gl.multiple_of(4 * rec, 4)
    col = gl.arange(0, 4, layout=gl.SliceLayout(0, LR))
    words = gl.load(rec0 + at[:, None] + col[None, :], mask=qv[:, None] & (col[None, :] < 4), other=0)
    el, st = gl.split(gl.reshape(words, [STREAM, 2, 2]))  # (e, link index), (source word, target)
    e_i, l = gl.split(el)
    srcw, t = gl.split(st)
    l = gl.convert_layout(l, LRQ)
    srcw = gl.convert_layout(srcw, LRQ)
    t = gl.convert_layout(t, LRQ)
    # GAP (D-7): the link's attribute block, loaded here by link index l through the
    #   connection's block pointer (D-7): its variant's attrs into the stage's attr slots; here: the weight
    # The load is unconditional -- a connection whose receive reads no per-link attr still needs a
    #   valid attr block (or a constexpr that skips the load), else this reads past its topology
    if STAGE_W:
        atp = s_atp.gather(srcw >> 24, 0).to(gl.pointer_type(gl.float32))
        w = gl.load(atp + l, mask=qv, other=0.0)
    else:  # (the receive reads no per-link attr: nothing to load)
        w = gl.zeros_like(l).to(gl.float32)
    # GAP (forward): the transpiled edge forward goes here, on link lanes (one record per lane, rooted
    #   at qv -- the old template's lane_mask="_lk_valid"), its assigned attrs stored back at l before staging,
    #   so receive sees the updated values as in the no-delay template. The variant differs per LANE here
    #   (unlike in the walk): each variant present in the block runs masked to its lanes (D-6); time = the
    #   delivery time (D-1, as in _walk)
    # LIMITATION (D-8): one link twice in one stream block (its
    #   source spiking in two runs taken in together) is read by both lanes before either stores: the later
    #   lane works on a stale value, and the last store wins, overwriting the changes of every forward call on
    #   that link in the block but the last
    # GAP (D-9): forward's destination.x (the target neuron's state) isn't wired up -- the target is
    #   known here (t) and its state is in scope in the walk; connecting the two is left for later
    # GAP (D-10): source.x unsupported for now
    NPW: gl.constexpr = GROUP_SIZE // NUM_WARPS
    pos = q - b0
    if NUM_WARPS == 1:  # one list: the block as it is
        wq = gl.zeros_like(q)
        rank = pos
    else:
        wq = gl.where(qv, t // NPW, pos % NUM_WARPS)  # its target's warp (padding: spread)
        wi = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(0, LR))
        oh = (wq[:, None] == wi[None, :]).to(gl.int32)
        rank = gl.sum(oh * (gl.cumsum(oh, axis=0) - 1), axis=1)  # its place in its warp's list
    wl = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(0, LR))
    n_w = gl.convert_layout(gl.sum(((wq[:, None] == wl[None, :]) & qv[:, None]).to(gl.int32), axis=0), LW)
    dst = (rank // U) * (NUM_WARPS * U) + wq * U + rank % U
    f_tgt.scatter(gl.where(qv, t, -1), dst, 0)
    f_w.scatter(w, dst, 0)
    if LOG:
        f_src.scatter(srcw, dst, 0)
    if STAGE_E:
        f_e.scatter(gl.convert_layout(e_i, LRQ).to(gl.float32, bitcast=True), dst, 0)
    gl.barrier()
    return gl.minimum(nm - b0, STREAM), n_w, gl.max(n_w, axis=0)


QLC = gl.constexpr(2048)  # links a batch unpacks into its inboxes at once (_walk_batch)


@gluon.jit
def _merge_segments(rec_time, rec0, cell0, cur, nmv, dcls, cv, s_arr, m_t, m_src, m_lo, m_n, table,
                    STREAM: gl.constexpr, NUM_WARPS: gl.constexpr, N_SLOTS: gl.constexpr, BLOCK: gl.constexpr, nb,
                    head, t_cut):
    """The next STREAM segment records of a population's queues (one a connection: cells cell0 + c) in arrival
    order, each from its head + cursor, only what arrives before t_cut -- an exact merge a block at a time: each
    queue's next STREAM records the candidates (the merged next STREAM are among them), a candidate's rank its place
    in its queue + a binary search into each other queue's (their arrivals in LDS, s_arr: sorted, padded with inf;
    ties: the lower queue first); the records placed whole at their rank -- m_t (arrival), m_src, m_lo (first
    link), m_n (links); past STREAM the untaken's dump. Returns (the records taken, each queue's taken count in
    cv's layout). (A record whose block is missing -- an arena that lost blocks -- counts as arriving last.)"""
    LR: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [NUM_WARPS, 1], [1, 0])
    LRQ: gl.constexpr = gl.SliceLayout(1, LR)
    i = gl.arange(0, STREAM, layout=LRQ)
    col = gl.arange(0, 4, layout=gl.SliceLayout(0, LR))
    for c in gl.static_range(N_SLOTS):
        cur_c = gl.sum(gl.where(cv == c, cur + head, 0), axis=0)
        nm_c = gl.sum(gl.where(cv == c, nmv, 0), axis=0)
        d_c = gl.sum(gl.where(cv == c, dcls, 0.0), axis=0)
        q = cur_c + i
        rec, ok = _run_rec(cell0 + c, q, table, N_SLOTS, BLOCK, nb, 0)
        a = gl.load(rec_time + 4 * rec, mask=(q < nm_c) & ok, other=float("inf")) + d_c
        s_arr.scatter(gl.where(a < t_cut, a, float("inf")), c * STREAM + i, 0)
    gl.barrier()
    taken = gl.zeros_like(cur)
    for c in gl.static_range(N_SLOTS):
        cur_c = gl.sum(gl.where(cv == c, cur + head, 0), axis=0)
        nm_c = gl.sum(gl.where(cv == c, nmv, 0), axis=0)
        d_c = gl.sum(gl.where(cv == c, dcls, 0.0), axis=0)
        q = cur_c + i
        rec, ok = _run_rec(cell0 + c, q, table, N_SLOTS, BLOCK, nb, 0)
        qv = (q < nm_c) & ok
        at = gl.multiple_of(4 * rec, 4)
        words = gl.load(rec0 + at[:, None] + col[None, :], mask=qv[:, None] & (col[None, :] < 4), other=0)
        el, st = gl.split(gl.reshape(words, [STREAM, 2, 2]))  # (e, first link), (source word, links)
        e_i, lo = gl.split(el)
        srcw, nl = gl.split(st)
        a = gl.where(qv, gl.convert_layout(e_i, LRQ).to(gl.float32, bitcast=True) + d_c, float("inf"))
        qv = qv & (a < t_cut)
        a = gl.where(qv, a, float("inf"))
        rank = i
        for c2 in gl.static_range(N_SLOTS):
            if c2 != c:
                pos = gl.zeros_like(i)
                for stp in gl.static_range(12):
                    if (STREAM >> (stp + 1)) > 0:
                        probe = pos + (STREAM >> (stp + 1))
                        v = s_arr.gather(c2 * STREAM + probe - 1, 0)
                        if c2 < c:
                            pos = gl.where(v <= a, probe, pos)
                        else:
                            pos = gl.where(v < a, probe, pos)
                v = s_arr.gather(c2 * STREAM + pos, 0)
                if c2 < c:
                    pos = gl.where(v <= a, pos + 1, pos)
                else:
                    pos = gl.where(v < a, pos + 1, pos)
                rank = rank + pos
        take = qv & (rank < STREAM)
        dst = gl.where(take, rank, STREAM + i)
        m_t.scatter(a, dst, 0)
        m_src.scatter(gl.convert_layout(srcw, LRQ), dst, 0)
        m_lo.scatter(gl.convert_layout(lo, LRQ), dst, 0)
        m_n.scatter(gl.convert_layout(nl, LRQ), dst, 0)
        taken = gl.where(cv == c, gl.sum(take.to(gl.int32), axis=0), taken)
    gl.barrier()
    return gl.sum(taken, axis=0), taken


@gluon.jit
def _unpack(f, m, base, n_take, s_off, m_t, m_lo, m_src, s_ltp, s_atp, STREAM: gl.constexpr,
            GROUP_SIZE: gl.constexpr, STAGE_W: gl.constexpr):
    """Link f of a batch (its records' links side by side, from base): its target, weight and arrival -- its
    record by a search over the records' first links (s_off), its link the record's first + its place."""
    fa = base + f
    r = gl.zeros_like(f)
    for st in gl.static_range(12):
        if (STREAM >> (st + 1)) > 0:
            probe = r + (STREAM >> (st + 1))
            r = gl.where((probe < n_take) & (s_off.gather(gl.minimum(probe, STREAM - 1), 0) <= fa), probe, r)
    link = m_lo.gather(r, 0) + (fa - s_off.gather(r, 0))
    conn = m_src.gather(r, 0) >> 24
    lt = s_ltp.gather(conn, 0).to(gl.pointer_type(gl.int32))
    t = _target_in(gl.load(lt + _target_word(link, GROUP_SIZE), mask=m, other=0), link, GROUP_SIZE)
    if STAGE_W:
        w = gl.load(s_atp.gather(conn, 0).to(gl.pointer_type(gl.float32)) + link, mask=m, other=0.0)
    else:
        w = gl.zeros_like(f).to(gl.float32)
    return t, w, m_t.gather(r, 0), m_src.gather(r, 0)


@gluon.jit
def _walk_batch(n_take, m_t, m_src, m_lo, m_n, s_off, ib_w, ib_a, b_cnt, b_start, b_run, b_tmp, s_atp, s_ltp, state,
                received, n_received, lane, mask, pid, ectx, RECEIVE: gl.constexpr, GROUP_SIZE: gl.constexpr,
                STREAM: gl.constexpr, NUM_WARPS: gl.constexpr, COUNT: gl.constexpr, STAGE_W: gl.constexpr,
                LOG: gl.constexpr = False, ib_s=0, log_ptr=0, log_e=0, log_cap=0, k_log=0,
                SEND1: gl.constexpr = None, QL: gl.constexpr = QLC):
    """A block's n_take segment records (in arrival order: m_t their delivery times) delivered, a BATCH at a
    time -- as many records as QL links hold: their links unpacked (each segment's contiguous in the topology),
    sorted stably into one INBOX per neuron (arrival order, link order: a count per neuron, the inboxes' starts by
    a scan, each link's place its inbox's start + its neuron's earlier links in the batch), then BEATS: every
    neuron its inbox's next entry, the receive across the whole group at once, as many beats as the longest
    inbox (the batch's depth) -- no barrier between beats (a neuron's inbox and state its lane's alone).
    state, received, n_received: the program's neuron layout (lane)."""
    LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    L: gl.constexpr = 32 * NUM_WARPS
    L2: gl.constexpr = gl.BlockedLayout([1, 32], [32, 1], [NUM_WARPS, 1], [1, 0])
    LIr: gl.constexpr = gl.SliceLayout(1, L2)
    LJr: gl.constexpr = gl.SliceLayout(0, L2)
    rq = gl.arange(0, STREAM, layout=LV)
    nv = gl.where(rq < n_take, m_n.gather(rq, 0), 0)
    offv = gl.cumsum(nv, axis=0) - nv
    s_off.store(offv)
    gv = gl.arange(0, GROUP_SIZE, layout=LV)
    fl = gl.arange(0, L, layout=LV)
    ii = gl.arange(0, L, layout=LIr)
    gl.barrier()
    r0 = gl.to_tensor(0)
    while r0 < n_take:
        base = gl.sum(gl.where(rq == r0, offv, 0), axis=0)
        fits = (rq >= r0) & (rq < n_take) & (offv + nv - base <= QL)
        r1 = gl.maximum(gl.max(gl.where(fits, rq + 1, 0), axis=0), r0 + 1)
        n_links = gl.minimum(gl.sum(gl.where((rq >= r0) & (rq < r1), nv, 0), axis=0), QL)
        b_cnt.store(gl.zeros_like(gv))
        b_run.store(gl.zeros_like(gv))
        gl.barrier()
        # 1. each neuron's links in the batch
        for f0 in range(0, n_links, L):
            f = f0 + fl
            m = f < n_links
            t, _w, _a, _s = _unpack(f, m, base, n_take, s_off, m_t, m_lo, m_src, s_ltp, s_atp, STREAM, GROUP_SIZE,
                                    False)
            b_cnt.atomic_scatter_add(m.to(gl.int32), gl.where(m, t, 0), 0)
        gl.barrier()
        cnt = b_cnt.load(LV)
        b_start.store(gl.cumsum(cnt, axis=0) - cnt)
        depth = gl.max(cnt, axis=0)
        gl.barrier()
        # 2. every link to its place in its neuron's inbox, chunk by chunk in link order
        for f0 in range(0, n_links, L):
            f = f0 + fl
            m = f < n_links
            t, w, a, sw = _unpack(f, m, base, n_take, s_off, m_t, m_lo, m_src, s_ltp, s_atp, STREAM, GROUP_SIZE,
                                  STAGE_W)
            b_tmp.store(gl.where(m, t, -1 - fl))  # (an unused lane: a target of its own)
            gl.barrier()
            ti = b_tmp.gather(ii, 0)
            within = gl.zeros_like(ii)
            for j0 in gl.static_range(0, L, 32):
                jj = j0 + gl.arange(0, 32, layout=LJr)
                tj = b_tmp.gather(jj, 0)[None, :]
                within += gl.sum(((tj == ti[:, None]) & (jj[None, :] < ii[:, None])).to(gl.int32), axis=1)
            tq = gl.where(m, t, 0)
            slot = b_start.gather(tq, 0) + b_run.gather(tq, 0) + gl.convert_layout(within, LV)
            dst = gl.where(m, slot, QL + fl)
            ib_w.scatter(w, dst, 0)
            ib_a.scatter(a, dst, 0)
            if LOG:  # (each entry's source, for the delivery log)
                ib_s.scatter(sw, dst, 0)
            gl.barrier()
            b_run.atomic_scatter_add(m.to(gl.int32), tq, 0)
            gl.barrier()
        # 3. the beats
        cnt_l = b_cnt.gather(lane, 0)
        st_l = b_start.gather(lane, 0)
        nodes = pid * GROUP_SIZE + lane
        # (each neuron's next inbox entry; a beat: its inputs at the next time, received one by one -- dt 0 between
        # them -- then its per-delivery send once, SEND1)
        pk = gl.zeros_like(lane)
        for k in range(0, depth):
            first = (pk < cnt_l) & mask
            a = ib_a.gather(gl.where(first, st_l + pk, 0), 0)
            hit = first
            while gl.max(hit.to(gl.int32), axis=0) > 0:
                idx = gl.where(hit, st_l + pk, 0)
                w = ib_w.gather(idx, 0)
                state = RECEIVE(state, hit, a, w, 0, ectx, nodes)
                if COUNT:  # dev: the stand-in's bookkeeping (tests)
                    received = gl.where(hit, received + w, received)
                    n_received = n_received + hit.to(gl.int32)
                if LOG:  # every delivered link: its record in the delivery log (in delivery order per target)
                    srcword = ib_s.gather(idx, 0)
                    zl = gl.zeros_like(lane)
                    at_log = gl.atomic_add(log_ptr + zl, 1, mask=hit)
                    lw = hit & (at_log < log_cap)
                    rec = log_ptr + 1 + 4 * at_log
                    gl.store(rec, zl + k_log, mask=lw)
                    gl.store(rec + 1, srcword >> 24, mask=lw)
                    gl.store(rec + 2, srcword & 0xFFFFFF, mask=lw)
                    gl.store(rec + 3, nodes, mask=lw)
                    gl.store(log_e + at_log, a, mask=lw)
                pk = pk + hit.to(gl.int32)
                hit = hit & (pk < cnt_l) & (ib_a.gather(gl.where(hit & (pk < cnt_l), st_l + pk, 0), 0) == a)
            if SEND1 is not None:
                state = SEND1(state, first, a, gl.zeros_like(a), 0, ectx, nodes)
        gl.barrier()
        r0 = r1
    return state, received, n_received


@gluon.jit
def _shift_table(table, cell, nb, freed, held, LV: gl.constexpr, NUM_WARPS: gl.constexpr):
    """A queue's block table moved down by its freed blocks: entries freed .. held to 0 .. held - freed (the
    chunks in increasing order: a chunk's sources lie above what any earlier chunk wrote)."""
    bi = gl.arange(0, 32 * NUM_WARPS, layout=LV)
    for i0 in range(0, held - freed, 32 * NUM_WARPS):
        m = i0 + bi < held - freed
        v = gl.load(table + cell * nb + freed + i0 + bi, mask=m, other=-1)
        gl.barrier()
        gl.store(table + cell * nb + i0 + bi, v, mask=m)
        gl.barrier()


@gluon.jit
def _release_cell(table, stack, tp, pid, cell, held, nblk, lost_v, LV: gl.constexpr, nb,
                  NUM_WARPS: gl.constexpr, chk, k, ARENA_CHECK: gl.constexpr):
    """A delivered cell's blocks back on the group's free stack (the run keeps its first RESERVE
    blocks). Returns the stack's new top."""
    keep = gl.minimum(held, RESERVE)  # (fixed blocks: never missing, never given away)
    tp = _free_blocks(table, stack, tp, pid, cell, keep, held, lost_v, LV, nb, NUM_WARPS, chk, k, ARENA_CHECK)
    if held > keep:
        gl.store(nblk + cell, keep)
    return tp


# ------------------------------------------------- phase 1: the pull


@gluon.jit
def _spike_block(t0, n_tot, tab_first, tab_node, tab_time, LK: gl.constexpr, SPK: gl.constexpr,
                 MAX_CJ: gl.constexpr, LOG_CJ: gl.constexpr):
    """Spikes t0 .. t0 + SPK of an e-group's stream: each spike's (connection, slot) pair -- the last whose
    first spike is at or before it, a binary search over the LDS table (first[0] = 0; pairs without spikes
    here start where the next one does, so never the last one) -- and its node and time from the history.
    Returns (t, tv, cj_t, s, e)."""
    t = t0 + gl.arange(0, SPK, layout=LK)
    tv = t < n_tot
    cj_t = gl.zeros_like(t)
    f_t = gl.zeros_like(t)
    for st in gl.static_range(LOG_CJ):
        probe = cj_t + (MAX_CJ >> (st + 1))
        v = tab_first.gather(probe, 0)
        le = v <= t
        cj_t = gl.where(le, probe, cj_t)
        f_t = gl.where(le, v, f_t)
    i = (t - f_t).to(gl.int64)
    s = gl.load((tab_node.gather(cj_t, 0) + 4 * i).to(gl.pointer_type(gl.int32)), mask=tv, other=0)
    e = gl.load((tab_time.gather(cj_t, 0) + 4 * i).to(gl.pointer_type(gl.float32)), mask=tv, other=0.0)
    return t, tv, cj_t, s, e


@gluon.jit
def _pull_headers(plan, n_conns, SK: gl.constexpr, KP: gl.constexpr, MAX_CONNS: gl.constexpr,
                  NUM_WARPS: gl.constexpr, TABLES: gl.constexpr):
    """What the pull reads first -- per connection its last-read run ids (both parities: the run's isn't known
    yet), per (connection, history slot) its history's address and cap and its run's header -- loaded first
    thing in a program, before its schedule entry is even back (none of it depends on the entry; the run's
    own values only filter it, _pull): the chain to the headers is plan -> history -> header, issued with the
    entry's and the gate's loads, not after them. TABLES (the slotted ring's pull): per pair its topology's and
    directory's addresses and counts and its link targets' address too (the plan's, static: _pull's
    per-pair LDS tables). Read-only: a gated-off run just drops it."""
    MAX_CJ: gl.constexpr = MAX_CONNS * KP
    LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    (hist_addr, topo_addr, link_target_addr, _attr_addr, dir_addr, hcap_p, t_src_p, t_seg_p, _t_bkt_p,
     _t_lnk_p, dpairs_p, dwords_p, last_read) = _fused_plan(plan, n_conns)
    ci = gl.arange(0, MAX_CONNS, layout=LV)
    cv = ci < n_conns
    last0 = gl.load(last_read + ci, mask=cv, other=0)
    last1 = gl.load(last_read + n_conns + ci, mask=cv, other=0)
    cj = gl.arange(0, MAX_CJ, layout=LV)
    c_of = cj // KP
    j_of = cj % KP
    vj = (c_of < n_conns) & (j_of < SK)
    hbuf = gl.load(hist_addr + c_of, mask=vj, other=0)
    hcap = gl.load(hcap_p + c_of, mask=vj, other=0)  # (a source program's region: R spikes, history.py)
    hG = gl.load(plan + PLAN_G * n_conns + c_of, mask=vj, other=0)  # (its programs: G; 0: one flat list)
    lastj0 = gl.load(last_read + c_of, mask=vj, other=0)
    lastj1 = gl.load(last_read + n_conns + c_of, mask=vj, other=0)
    hdr = hbuf.to(gl.pointer_type(gl.int32)) + 8 * j_of
    tag = gl.load(hdr.to(gl.pointer_type(gl.int64)), mask=vj, other=0)  # run id and round, read whole
    cnt = gl.load(hdr + 2, mask=vj, other=0)
    wend = gl.load((hdr + 3).to(gl.pointer_type(gl.float32)), mask=vj, other=0.0)
    early = gl.load((hdr + 4).to(gl.pointer_type(gl.float32)), mask=vj, other=float("inf"))
    if TABLES:
        topo = (gl.load(topo_addr + c_of, mask=vj, other=0), gl.load(t_src_p + c_of, mask=vj, other=0),
                gl.load(t_seg_p + c_of, mask=vj, other=0), gl.load(dir_addr + c_of, mask=vj, other=0),
                gl.load(dpairs_p + c_of, mask=vj, other=0), gl.load(dwords_p + c_of, mask=vj, other=0),
                # every pair a real connection's link targets (the append reads targets unmasked)
                gl.load(link_target_addr + gl.minimum(c_of, n_conns - 1), mask=c_of * 0 < n_conns, other=0))
    else:
        topo = ()
    return last0, last1, hbuf, hcap, hG, lastj0, lastj1, tag, cnt, wend, early, topo


@gluon.jit
def _pull(pid, n_groups, plan, n_conns, ring, wmark_ptr, oflags, ctl, k_lo, slot_width, round_id, parity, hdrs,
          trace, n_ev,
          SK: gl.constexpr, KP: gl.constexpr, MAX_CONNS: gl.constexpr, LOG_CJ: gl.constexpr,
          N_SLOTS: gl.constexpr, NSP: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
          TRACE: gl.constexpr, TRACE_CAP: gl.constexpr, BLOCK: gl.constexpr, chk, ARENA_CHECK: gl.constexpr):
    """Phase 1 of program pid, the slotted ring (uniform sources only: every source run's spikes share one e):
    what the sources emitted since this consumer last read them, into the program's group's ring; returns
    (n_ev (tracing), ok). Not ok: the ring is too small for the run -- nothing committed (counts, tables, free
    stack, watermark as they were; only scratch past the counts written), its stop flag and the launches' stop
    word ctl set: the program stops here."""
    # (connection, source history slot) pairs, 2**LOG_CJ; SK: the sources' history depth
    MAX_CJ: gl.constexpr = MAX_CONNS * KP
    SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    # per (connection, slot) / per connection vectors, spikes x (connection, slot) for the spike decode,
    # spikes x segments for the expansion
    LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    # spikes x segments: one spike row per thread row (8 x warps rows), its MSEG segments over 4
    # threads -- per-spike values (three 64-bit pointers among them) held once per thread, not 4x
    LP2: gl.constexpr = gl.BlockedLayout([1, MSEG // 4], [8, 4], [NUM_WARPS, 1], [1, 0])
    LK: gl.constexpr = gl.SliceLayout(1, LP2)
    LM: gl.constexpr = gl.SliceLayout(0, LP2)
    (_hist_addr, _topo_addr, _link_target_addr, _attr_addr, _dir_addr, _hcap_p, _t_src_p, _t_seg_p, _t_bkt_p,
     _t_lnk_p, _dpairs_p, _dwords_p, last_read) = _fused_plan(plan, n_conns)
    (rec_time, rec_src, rec_lo, rec_n, run_count, _head, flags, table, stack, top, nblk, _lost, log_rec,
     ovf_cap, run_cap, nb, stats) = _ring(ring, BLOCK)

    # No user code in the pull -- it only moves (e, source, link, target) records into the ring;
    #   every hook runs at delivery, in delivery order (strict per-link order)
    ci = gl.arange(0, MAX_CONNS, layout=LV)
    cv = ci < n_conns
    # per (connection, slot): its run's header -- one chain (plan -> history -> header) for everything below,
    # loaded first thing in the program (_pull_headers); the run's parity picks its last-read ids
    last0, last1, hbuf, hcap, _hG, lastj0, lastj1, tag, cnt, wend, early, topo = hdrs
    last_c = gl.where(parity == 0, last0, last1)
    lastj = gl.where(parity == 0, lastj0, lastj1)
    cj = gl.arange(0, MAX_CJ, layout=LV)
    c_of = cj // KP
    j_of = cj % KP
    vj = (c_of < n_conns) & (j_of < SK)
    rid = (tag & 0xFFFFFFFF).to(gl.int32)
    rnd = (tag >> 32).to(gl.int32)
    # per connection: its newest and oldest run newer than last read (the last-read update, the overrun
    # check), reduced over its slots
    newer = vj & (rid > lastj) & (rnd < round_id)
    newest_c = gl.convert_layout(gl.max(gl.reshape(gl.where(newer, rid, 0), [MAX_CONNS, KP]), axis=1), LV)
    oldest_c = gl.convert_layout(gl.min(gl.reshape(gl.where(newer, rid, 2147483647), [MAX_CONNS, KP]), axis=1),
                                 LV)
    newest_c = gl.maximum(newest_c, last_c)
    gl.store(last_read + (1 - parity) * n_conns + ci, newest_c, mask=cv)  # idempotent across programs
    if gl.max((cv & (oldest_c < 2147483647) & (oldest_c > last_c + 1)).to(gl.int32), axis=0) > 0:
        gl.atomic_max(oflags, 1)  # a run fell out of the history before this consumer read it
    # the fresh runs, taken in in order of their emission time -- a run's spikes share one e but runs differ, and
    # a slot's run must grow in e
    fresh = newer & (cnt > 0)
    n_spk = gl.where(fresh, gl.minimum(cnt, hcap), 0)  # (a uniform source's history: one flat list)
    key = gl.where(fresh, early, float("inf"))
    n_runs = gl.sum(fresh.to(gl.int32), axis=0)
    node_addr = hbuf + 4 * (8 * SK + SK + j_of * hcap).to(gl.int64)  # (history.py: G = 1)
    time_addr = node_addr + 4 * (SK * hcap).to(gl.int64)
    wmark = gl.load(wmark_ptr + pid)
    # per (connection, slot) pair, everything a spike's decode needs, as read-only LDS tables: a
    # spike looks its pair's values up directly (no barriers, no layout conversions per block)
    # (the plan's per-pair values: loaded first thing, _pull_headers)
    tb_j, nsrc_j, nseg_j, dbuf_j, dpairs_j, dwords_j, ltp_j = topo
    _j_offs, j_start, j_gb, j_mean = topology_ptrs(tb_j.to(gl.pointer_type(gl.int32)), nsrc_j, nseg_j)
    j_doffs, j_dfirst, j_dwords = directory_ptrs(dbuf_j.to(gl.pointer_type(gl.int32)), nsrc_j, dpairs_j)
    tab_node = _table(node_addr, SM)
    tab_time = _table(time_addr, SM)
    tab_wend = _table(wend, SM)
    tab_start = _table(j_start.to(gl.int64), SM)
    tab_gb = _table(j_gb.to(gl.int64), SM)
    tab_mean = _table(j_mean.to(gl.int64), SM)
    tab_ltp = _table(ltp_j, SM)
    tab_doffs = _table(j_doffs.to(gl.int64), SM)
    tab_dfirst = _table(j_dfirst.to(gl.int64), SM)
    tab_dwords = _table(j_dwords.to(gl.int64), SM)
    tab_W = _table(dwords_j, SM)
    tab_first = gl.allocate_shared_memory(gl.int32, [MAX_CJ], SM)  # per e-group: each pair's first spike
    # this group's per-slot link counts, in LDS for the pull's appends (it is their only writer)
    sl = gl.arange(0, NSP, layout=LV)
    s_cnt = gl.allocate_shared_memory(gl.int32, [NSP], SM)
    cnt0 = gl.load(run_count + pid * N_SLOTS + sl, mask=sl < N_SLOTS, other=0)
    # count | blocks held << CNT_BITS, so every claim learns both (_append_links); the free stack top (the
    # counts at the start, cnt0: _arena_fixup's headroom, this run's growth)
    top0 = gl.load(top + pid)
    s_cnt.store(cnt0 | (gl.load(nblk + pid * N_SLOTS + sl, mask=sl < N_SLOTS, other=0) << CNT_BITS))
    gl.barrier()
    violated = gl.to_tensor(0)
    total = gl.to_tensor(0)
    if TRACE >= 2:
        n_runs = pin_s(n_runs)
        n_ev = mark(trace, n_ev, 10, n_runs, TRACE_CAP)
    # all fresh runs sharing an emission time as ONE stream of spikes
    # (normally every source's run of the previous round), in blocks
    # spanning runs and connections; e-groups in order of e -- the distinct emission times one after
    # another, each the least after the last (one reduction a group: quiet rounds have one)
    e0 = gl.min(key, axis=0)
    while e0 < float("inf"):
        n_g = gl.where(fresh & (key == e0), n_spk, 0)
        g_first = gl.cumsum(n_g, axis=0) - n_g
        n_tot = gl.sum(n_g, axis=0)
        total += n_tot
        e_max = gl.to_tensor(float("-inf"))
        tab_first.store(g_first)
        gl.barrier()
        for t0 in range(0, n_tot, SPK):
            # (loading the next block's spikes here, before this block's appends -- a pipeline over the
            # blocks -- was no faster on the 4090: limitations.md)
            t, tv, cj_t, s, e = _spike_block(t0, n_tot, tab_first, tab_node, tab_time, LK, SPK, MAX_CJ, LOG_CJ)
            if TRACE >= 2:
                s = pin(s)
                e = pin(e)
                n_ev = mark(trace, n_ev, 11, gl.sum(tv.to(gl.int32), axis=0), TRACE_CAP)
            # GAP (D-10): source.x unsupported for now; taken at emission, it would be read off
            #   the history here, next to the spike's node and e, and travel with its records
            wend_t = tab_wend.gather(cj_t, 0)
            c_t = cj_t // KP
            seg_start = tab_start.gather(cj_t, 0).to(gl.pointer_type(gl.int32))
            seg_gb = tab_gb.gather(cj_t, 0).to(gl.pointer_type(gl.int32))
            bucket_mean = tab_mean.gather(cj_t, 0).to(gl.pointer_type(gl.float32))
            ltp_t = _target_ptr(tab_ltp.gather(cj_t, 0), GROUP_SIZE)
            a, b = _segments_of(s, tv, pid, tab_doffs.gather(cj_t, 0).to(gl.pointer_type(gl.int32)),
                                tab_dfirst.gather(cj_t, 0).to(gl.pointer_type(gl.int32)),
                                tab_dwords.gather(cj_t, 0).to(gl.pointer_type(gl.int32)),
                                tab_W.gather(cj_t, 0))
            if TRACE >= 2:
                b = pin(b)
                n_ev = mark(trace, n_ev, 12, gl.sum((b > a).to(gl.int32), axis=0), TRACE_CAP)
            # (the uniform contract: no spike below the watermark or past its run's window -- else flag 2)
            take = tv & (e >= wmark) & (e < wend_t)
            has = b > a
            violated = gl.maximum(violated, gl.max((tv & has & ~take).to(gl.int32), axis=0))
            e_max = gl.maximum(e_max, gl.max(gl.where(take & has, e, float("-inf")), axis=0))
            n_ev = _append_links(s, e, take, c_t, a, b, seg_start, seg_gb, bucket_mean,
                                 ltp_t, s_cnt, pid, k_lo, slot_width,
                                 rec_time, rec_src, rec_lo, rec_n,
                                 flags, trace, n_ev, table, log_rec, ovf_cap, N_SLOTS, run_cap,
                                 LP2, LM, SPK, MSEG, TRACE, TRACE_CAP, NUM_WARPS, GROUP_SIZE, BLOCK, nb)
            if TRACE >= 2:
                n_ev = mark(trace, n_ev, 13, gl.sum(gl.where(take, b - a, 0), axis=0), TRACE_CAP)
        wmark = gl.maximum(wmark, e_max)  # later e-groups are compared against this one
        gl.barrier()  # the group's appends are in before the next group's (a slot's run grows in e)
        e0 = gl.min(gl.where(key > e0, key, float("inf")), axis=0)
    if violated > 0:
        gl.atomic_max(flags + 2, 1)
    gl.barrier()
    if TRACE >= 3:
        n_ev = mark(trace, n_ev, 19, 0, TRACE_CAP)
    ok, top0, used, most = _arena_fixup(s_cnt, cnt0, rec_src - 1, table, stack, log_rec, ovf_cap, top0,
                                       pid, sl, LV, N_SLOTS, run_cap, BLOCK, nb, NUM_WARPS, NSP, chk,
                                       round_id, ARENA_CHECK)
    if TRACE >= 3:
        n_ev = mark(trace, n_ev, 20, pin_s(most), TRACE_CAP)
    gl.store(stats + 2 * pid, used)  # the run's status: the blocks the group holds, the most logged
    gl.store(stats + 2 * pid + 1, most)
    if ok:
        packed = s_cnt.load(LV)  # the counts and block counts back to the ring
        gl.store(run_count + pid * N_SLOTS + sl, packed & ((1 << CNT_BITS) - 1), mask=sl < N_SLOTS)
        gl.store(nblk + pid * N_SLOTS + sl, packed >> CNT_BITS, mask=sl < N_SLOTS)
        gl.store(top + pid, top0)
    if TRACE >= 3:
        n_ev = mark(trace, n_ev, 21, 0, TRACE_CAP)
    if ok:
        gl.store(wmark_ptr + pid, wmark)
    else:
        # a ring too small for this run: nothing committed; the run stops here, and with it every run that
        # starts after (_run_gate) -- the host resizes and runs them again (PopulationRuntime, _Launches)
        gl.atomic_max(flags + FLAG_STOPPED, 1)
        gl.atomic_max(ctl, 1)
    gl.barrier()  # this program's ring entries are complete before it delivers
    if TRACE:
        n_ev = mark(trace, n_ev, 2, total, TRACE_CAP)
    return n_ev, ok


# ------------------------------------------------- phase 1, queues (SPIKES.md)
#
# A population whose incoming connections each have one delay value: a queue per (group, connection) -- a cell of
# the ring (N_SLOTS: the connections), its records in emission order, so in arrival order (e + D_c). The pull
# takes a connection's fresh runs in run order (a source's runs follow each other in time), each run's spikes
# merged into (e, node) order from its source programs' sorted regions (history.py) and appended in that order;
# delivery merges the queues by arrival (_deliver_queues). A queue's blocks: all through its block table (no
# fixed reserve); past the committed count, the pull writes scratch -- committed (counts, blocks, the free
# stack's top) only if the whole pull fits, else the run stops and reruns (_Launches).

QNB = gl.constexpr(256)  # time bins a source run's spikes are counted in (their chunks: whole bins)
QCH = gl.constexpr(512)  # the most spikes a chunk merges in LDS (bins grouped to QCH / 2 + one; past: merge path)


@gluon.jit
def _qbin(e, lo, inv):
    """A spike's time bin in its run (QNB over [earliest, window end)): monotone in e."""
    return gl.minimum(gl.maximum(gl.floor((e - lo) * inv).to(gl.int32), 0), QNB - 1)


@gluon.jit
def _qbound(gv, gvalid, bk, start, end, time_p, R_r, lo, inv):
    """Per region g (the lanes): the first index in [start, end) whose bin is bk or later -- a region is sorted by
    e (history.py), so its spikes of the bins before bk are [start, that)."""
    pos = start
    n_max = gl.max(gl.where(gvalid, end - start, 0), axis=0)
    step = gl.to_tensor(1)
    while step * 2 <= n_max:
        step = step * 2
    while step > 0:
        probe = pos + step
        ok = gvalid & (probe <= end)
        e = gl.load(time_p + (gv * R_r + probe - 1).to(gl.int64), mask=ok, other=0.0)
        pos = gl.where(ok & (_qbin(gl.where(ok, e, lo), lo, inv) < bk), probe, pos)
        step = step // 2
    return pos


@gluon.jit
def _qspike(t, s_cp, s_st, node_p, time_p, R_r, m, GP: gl.constexpr):
    """Spike t of a stream over the regions' ranges (s_cp: each range's first, s_st: its start in its region; a
    range without spikes starts where the next does): its node and e."""
    g = gl.zeros_like(t)
    for st in gl.static_range(10):  # (GP <= 1024: a probe stride of 0 past log2(GP) steps)
        if (GP >> (st + 1)) > 0:
            probe = g + (GP >> (st + 1))
            g = gl.where(s_cp.gather(probe, 0) <= t, probe, g)
    i = s_st.gather(g, 0) + (t - s_cp.gather(g, 0))
    at = (g * R_r + i).to(gl.int64)
    return gl.load(node_p + at, mask=m, other=0), gl.load(time_p + at, mask=m, other=0.0), g, i


@gluon.jit
def _qbefore(e, s, g2, start, end, node_p, time_p, R_r, m):
    """Per lane: how many spikes of region g2's range [start, end) come before (e, s) in (e, node) order (the
    region sorted; one neuron is one region's, so never a tie)."""
    pos = gl.zeros_like(s) + start
    step = gl.to_tensor(1)
    while step * 2 <= end - start:
        step = step * 2
    while step > 0:
        probe = pos + step
        ok = m & (probe <= end)
        at = (g2 * R_r + probe - 1).to(gl.int64)
        e2 = gl.load(time_p + at, mask=ok, other=0.0)
        s2 = gl.load(node_p + at, mask=ok, other=0)
        pos = gl.where(ok & ((e2 < e) | ((e2 == e) & (s2 < s))), probe, pos)
        step = step // 2
    return pos - start


@gluon.jit
def _qappend(n_w, s_se, s_sn, c, cell, tail, held, tp, lack, pid, rec0, table, stack, nb, seg_start,
             dir_offs, dir_first, dir_words, dwords_r, lt32, LV: gl.constexpr, NUM_WARPS: gl.constexpr,
             GROUP_SIZE: gl.constexpr, N_SLOTS: gl.constexpr, BLOCK: gl.constexpr, chk, when,
             ARENA_CHECK: gl.constexpr):
    """A merged window's n_w spikes (s_se, s_sn: in (e, node) order), a tile at a time: each spike's segment into
    the group (its links there: one delay, one segment) appended to the connection's queue (cell) in that
    order as one record -- positions by a scan, the queue's new blocks off the group's free stack (tp; none
    left: lack). Returns (tail, held, tp, lack)."""
    L: gl.constexpr = 32 * NUM_WARPS
    lane = gl.arange(0, L, layout=LV)
    for u0 in range(0, n_w, L):
        u = u0 + lane
        m = u < n_w
        s = s_sn.gather(gl.minimum(u, QCH - 1), 0)
        e = s_se.gather(gl.minimum(u, QCH - 1), 0)
        a, b = _segments_of(s, m, pid, dir_offs, dir_first, dir_words, dwords_r)
        has = m & (b > a)  # (one delay: one segment a spike and group)
        lo_l = gl.load(seg_start + a, mask=has, other=0)
        n_l = gl.where(has, gl.load(seg_start + a + 1, mask=has, other=0) - lo_l, 0)
        # one record per spike's segment into the group: (e, source word, its first link, its links) --
        # delivery expands it (_walk_batch)
        one = has.to(gl.int32)
        pos = tail + gl.cumsum(one, axis=0) - one
        tail = tail + gl.sum(one, axis=0)
        need = (tail + BLOCK - 1) // BLOCK
        while held < need:  # the queue's new blocks, off the group's free stack
            if (tp > 0) & (held < nb):
                bid = gl.load(stack + pid * nb + tp - 1)
                gl.store(table + cell * nb + held, bid)
                tp = tp - 1
                if ARENA_CHECK:  # (the block now the queue's: _arena_take -- a 1-lane tensor)
                    bid1 = bid + gl.arange(0, 1, layout=LV) * 0
                    _arena_take(chk, bid1 >= 0, pid, cell, bid1, nb, when, 1, 1, 0)
            else:
                lack = lack + 1
            held = held + 1
        gl.barrier()  # (the table's new entries before this tile's reads)
        blk = gl.load(table + cell * nb + pos // BLOCK, mask=has, other=0)
        rec, ok = _blk_rec(cell, pos, blk, N_SLOTS, BLOCK, nb)
        _store_records(rec0, 4 * rec, e, (c << 24) | s, lo_l, n_l, has & ok & (lack == 0), NUM_WARPS)
    return tail, held, tp, lack


@gluon.jit
def _pull_queues(pid, n_groups, plan, n_conns, ring, oflags, ctl, round_id, parity, hdrs, trace, n_ev,
                 SK: gl.constexpr, KP: gl.constexpr, MAX_CONNS: gl.constexpr, N_SLOTS: gl.constexpr,
                 NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr, TRACE: gl.constexpr, TRACE_CAP: gl.constexpr,
                 BLOCK: gl.constexpr, GP: gl.constexpr, chk=0, ARENA_CHECK: gl.constexpr = False):
    """Phase 1 of program pid, queues (see above): every fresh source run, in (connection, run) order, into its
    connection's queue of the group -- its spikes in (e, node) order (a time histogram, chunks of whole bins, each
    merged by rank in LDS), every link of a spike one record, positions by a scan (no atomics: the program is
    the queue's only writer). Returns (n_ev, ok) as _pull: not ok, nothing committed, the run stops."""
    MAX_CJ: gl.constexpr = MAX_CONNS * KP
    SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    LKC: gl.constexpr = gl.BlockedLayout([1, MAX_CJ], [32, 1], [NUM_WARPS, 1], [1, 0])
    L2: gl.constexpr = gl.BlockedLayout([1, 32], [32, 1], [NUM_WARPS, 1], [1, 0])
    LIr: gl.constexpr = gl.SliceLayout(1, L2)
    LJr: gl.constexpr = gl.SliceLayout(0, L2)
    L: gl.constexpr = 32 * NUM_WARPS
    (_hist_addr, topo_addr, link_target_addr, _attr_addr, dir_addr, _hcap_p, t_src_p, t_seg_p, _t_bkt_p, _t_lnk_p,
     dpairs_p, dwords_p, last_read) = _fused_plan(plan, n_conns)
    (rec_time, rec_src, _ml, _mn, run_count, _head, flags, table, stack, top, nblk, _lost, _lr, _oc, _mcap,
     nb, stats) = _ring(ring, BLOCK)
    rec0 = rec_src - 1
    ci = gl.arange(0, MAX_CONNS, layout=LV)
    cv = ci < n_conns
    last0, last1, hbuf, hcap, hG, lastj0, lastj1, tag, cnt, wend, early, _topo = hdrs
    last_c = gl.where(parity == 0, last0, last1)
    lastj = gl.where(parity == 0, lastj0, lastj1)
    cj = gl.arange(0, MAX_CJ, layout=LV)
    c_of = cj // KP
    j_of = cj % KP
    vj = (c_of < n_conns) & (j_of < SK)
    rid = (tag & 0xFFFFFFFF).to(gl.int32)
    rnd = (tag >> 32).to(gl.int32)
    newer = vj & (rid > lastj) & (rnd < round_id)
    newest_c = gl.convert_layout(gl.max(gl.reshape(gl.where(newer, rid, 0), [MAX_CONNS, KP]), axis=1), LV)
    oldest_c = gl.convert_layout(gl.min(gl.reshape(gl.where(newer, rid, 2147483647), [MAX_CONNS, KP]), axis=1),
                                 LV)
    newest_c = gl.maximum(newest_c, last_c)
    gl.store(last_read + (1 - parity) * n_conns + ci, newest_c, mask=cv)  # idempotent across programs
    if gl.max((cv & (oldest_c < 2147483647) & (oldest_c > last_c + 1)).to(gl.int32), axis=0) > 0:
        gl.atomic_max(oflags, 1)  # a run fell out of the history before this consumer read it
    fresh = newer & (cnt > 0)
    n_runs = gl.sum(fresh.to(gl.int32), axis=0)
    G1 = gl.maximum(hG, 1)
    cnt_addr = hbuf + 4 * (8 * SK + j_of * G1).to(gl.int64)
    node_addr = hbuf + 4 * (8 * SK + SK * G1 + j_of * G1 * hcap).to(gl.int64)
    time_addr = node_addr + 4 * (SK * G1 * hcap).to(gl.int64)
    n_flat = gl.where(fresh, gl.minimum(cnt, hcap), 0)
    # the fresh runs in (connection, run id) order: each pair's rank
    k_r = gl.convert_layout(gl.where(fresh, c_of * 1048576 + (rid & 1048575), 2147483647), gl.SliceLayout(0, LKC))
    k_l = gl.convert_layout(gl.where(fresh, c_of * 1048576 + (rid & 1048575), 2147483647), gl.SliceLayout(1, LKC))
    cj_r = gl.arange(0, MAX_CJ, layout=gl.SliceLayout(0, LKC))
    cj_l = gl.arange(0, MAX_CJ, layout=gl.SliceLayout(1, LKC))
    before = (k_r[None, :] < k_l[:, None]) | ((k_r[None, :] == k_l[:, None]) & (cj_r[None, :] < cj_l[:, None]))
    order = gl.convert_layout(gl.sum(before.to(gl.int32), axis=1), LV)
    # per connection: its topology, and its queue's count and blocks (committed at the end)
    tbuf_c = gl.load(topo_addr + ci, mask=cv, other=0)
    nsrc_c = gl.load(t_src_p + ci, mask=cv, other=0)
    nseg_c = gl.load(t_seg_p + ci, mask=cv, other=0)
    dbuf_c = gl.load(dir_addr + ci, mask=cv, other=0)
    dpairs_c = gl.load(dpairs_p + ci, mask=cv, other=0)
    dwords_c = gl.load(dwords_p + ci, mask=cv, other=0)
    ltp_c = gl.load(link_target_addr + ci, mask=cv, other=0)
    tails = gl.load(run_count + pid * N_SLOTS + ci, mask=cv, other=0)
    helds = gl.load(nblk + pid * N_SLOTS + ci, mask=cv, other=0)
    tp0 = gl.load(top + pid)
    tp = tp0
    lack = gl.to_tensor(0)  # blocks the free stack couldn't give
    total = gl.to_tensor(0)
    # LDS: the run's histogram, the regions' ranges, a chunk as loaded and as merged
    hist = gl.allocate_shared_memory(gl.int32, [QNB], SM)
    s_cp = gl.allocate_shared_memory(gl.int32, [GP], SM)
    s_st = gl.allocate_shared_memory(gl.int32, [GP], SM)
    s_ce = gl.allocate_shared_memory(gl.float32, [2 * QCH], SM)  # (past QCH: the unused lanes' dump; a lane count <= QCH)
    s_cn = gl.allocate_shared_memory(gl.int32, [2 * QCH], SM)
    s_se = gl.allocate_shared_memory(gl.float32, [2 * QCH], SM)
    s_sn = gl.allocate_shared_memory(gl.int32, [2 * QCH], SM)
    gv = gl.arange(0, GP, layout=LV)
    bq = gl.arange(0, QNB, layout=LV)
    lane = gl.arange(0, L, layout=LV)
    for ri in range(0, n_runs):
        sel = order == ri
        c = gl.sum(gl.where(sel, c_of, 0), axis=0)
        G_r = gl.sum(gl.where(sel, hG, 0), axis=0)  # (0: one flat list)
        R_r = gl.sum(gl.where(sel, hcap, 0), axis=0)
        node_p = gl.sum(gl.where(sel, node_addr, 0), axis=0).to(gl.pointer_type(gl.int32))
        time_p = gl.sum(gl.where(sel, time_addr, 0), axis=0).to(gl.pointer_type(gl.float32))
        cnt_p = gl.sum(gl.where(sel, cnt_addr, 0), axis=0).to(gl.pointer_type(gl.int32))
        lo = gl.sum(gl.where(sel, early, 0.0), axis=0)
        hi = gl.sum(gl.where(sel, wend, 0.0), axis=0)
        inv = QNB / gl.maximum(hi - lo, 1e-6)
        selc = ci == c
        nsrc_r = gl.sum(gl.where(selc, nsrc_c, 0), axis=0)
        _so, seg_start, _sg, _bm = topology_ptrs(gl.sum(gl.where(selc, tbuf_c, 0), axis=0).to(gl.pointer_type(gl.int32)),
                                                 nsrc_r, gl.sum(gl.where(selc, nseg_c, 0), axis=0))
        dir_offs, dir_first, dir_words = directory_ptrs(
            gl.sum(gl.where(selc, dbuf_c, 0), axis=0).to(gl.pointer_type(gl.int32)), nsrc_r,
            gl.sum(gl.where(selc, dpairs_c, 0), axis=0))
        dwords_r = gl.sum(gl.where(selc, dwords_c, 0), axis=0)
        lt32 = gl.sum(gl.where(selc, ltp_c, 0), axis=0).to(gl.pointer_type(gl.int32))
        cell = pid * N_SLOTS + c
        tail = gl.sum(gl.where(selc, tails, 0), axis=0)
        held = gl.sum(gl.where(selc, helds, 0), axis=0)
        # the run's regions (in program order) and their spikes
        gvalid = gv < gl.maximum(G_r, 1)
        cnt_g = gl.where(G_r == 0, gl.where(gv == 0, gl.sum(gl.where(sel, n_flat, 0), axis=0), 0),
                         gl.minimum(gl.load(cnt_p + gv, mask=gvalid & (G_r > 0), other=0), R_r))
        n_run = gl.sum(cnt_g, axis=0)
        total += n_run
        if TRACE >= 2:
            n_ev = mark(trace, n_ev, 10, pin_s(n_run), TRACE_CAP)  # ("runs": a source run's spikes)
        # 1. its time histogram
        hist.store(gl.zeros_like(bq))
        s_cp.store(gl.cumsum(cnt_g, axis=0) - cnt_g)
        s_st.store(gl.zeros_like(gv))
        gl.barrier()
        for t0 in range(0, n_run, L):
            t = t0 + lane
            m = t < n_run
            _s, e, _g, _i = _qspike(t, s_cp, s_st, node_p, time_p, R_r, m, GP)
            hist.atomic_scatter_add(m.to(gl.int32), _qbin(gl.where(m, e, lo), lo, inv), 0)
        gl.barrier()
        hv = hist.load(LV)
        if TRACE >= 2:
            n_ev = mark(trace, n_ev, 11, pin_s(gl.max(hv, axis=0)), TRACE_CAP)  # ("spikes": its histogram)
        cid = (gl.cumsum(hv, axis=0) - hv) // (QCH // 2)  # (each bin's chunk: a chunk at most QCH / 2 + a bin)
        n_ch = gl.max(gl.where(hv > 0, cid, -1), axis=0) + 1
        st_g = gl.zeros_like(gv)
        for k in range(0, n_ch):
            # 2. the chunk's bins, and per region its range (the regions sorted: one search a boundary)
            b_hi = gl.sum((cid <= k).to(gl.int32), axis=0)
            en_g = _qbound(gv, gvalid, b_hi, st_g, cnt_g, time_p, R_r, lo, inv)
            sz = en_g - st_g
            n_k = gl.sum(sz, axis=0)
            gl.barrier()  # (the last chunk's tables read)
            s_cp.store(gl.cumsum(sz, axis=0) - sz)
            s_st.store(st_g)
            gl.barrier()
            if n_k <= QCH:
                # 3. the chunk into LDS, then each spike's rank in (e, node, place) -- every place a distinct key
                for f0 in range(0, n_k, L):
                    f = f0 + lane
                    m = f < n_k
                    s, e, _g, _i = _qspike(f, s_cp, s_st, node_p, time_p, R_r, m, GP)
                    s_ce.scatter(gl.where(m, e, float("inf")), gl.where(m, f, QCH + (f - f0)), 0)
                    s_cn.scatter(s, gl.where(m, f, QCH + (f - f0)), 0)
                gl.barrier()
                for i0 in range(0, n_k, L):
                    i = i0 + gl.arange(0, L, layout=LIr)
                    vi = i < n_k
                    ei = s_ce.gather(gl.minimum(i, QCH - 1), 0)
                    ni = s_cn.gather(gl.minimum(i, QCH - 1), 0)
                    rank = gl.zeros([L], gl.int32, layout=LIr)
                    for j0 in range(0, n_k, 32):
                        jj = j0 + gl.arange(0, 32, layout=LJr)
                        ej = s_ce.gather(gl.minimum(jj, QCH - 1), 0)[None, :]
                        nj = s_cn.gather(gl.minimum(jj, QCH - 1), 0)[None, :]
                        e2 = ei[:, None]
                        n2 = ni[:, None]
                        bf = (ej < e2) | ((ej == e2) & ((nj < n2) | ((nj == n2) & (jj[None, :] < i[:, None]))))
                        rank += gl.sum((bf & (jj[None, :] < n_k)).to(gl.int32), axis=1)
                    dst = gl.where(vi, rank, QCH + (i - i0))
                    s_se.scatter(ei, dst, 0)
                    s_sn.scatter(ni, dst, 0)
                gl.barrier()
                if TRACE >= 2:
                    n_ev = mark(trace, n_ev, 12, n_k, TRACE_CAP)  # ("found": a chunk merged)
                # 4. in that order: appended
                tail, held, tp, lack = _qappend(n_k, s_se, s_sn, c, cell, tail, held, tp, lack, pid, rec0,
                                                table, stack, nb, seg_start, dir_offs, dir_first, dir_words, dwords_r,
                                                lt32, LV, NUM_WARPS, GROUP_SIZE, N_SLOTS, BLOCK, chk, round_id,
                                                ARENA_CHECK)
                gl.barrier()
                if TRACE >= 2:
                    n_ev = mark(trace, n_ev, 13, pin_s(tail), TRACE_CAP)  # ("appended": the chunk's links)
            else:
                # a chunk past QCH (a burst in one time bin, equal times): each spike's rank by merge path -- its
                # place in its region's range + the spikes of every other region's before it (each sorted) --,
                # merged and appended QCH at a time (slow, rare)
                for w0 in range(0, n_k, QCH):
                    for f0 in range(0, n_k, L):
                        f = f0 + lane
                        m = f < n_k
                        s, e, g, i = _qspike(f, s_cp, s_st, node_p, time_p, R_r, m, GP)
                        rank = f - s_cp.gather(g, 0)
                        for g2 in range(0, gl.maximum(G_r, 1)):
                            st2 = gl.sum(gl.where(gv == g2, st_g, 0), axis=0)
                            en2 = gl.sum(gl.where(gv == g2, en_g, 0), axis=0)
                            nb2 = _qbefore(e, s, g2, st2, en2, node_p, time_p, R_r, m & (g != g2))
                            rank += gl.where(g != g2, nb2, 0)
                        inw = m & (rank >= w0) & (rank < w0 + QCH)
                        dst = gl.where(inw, rank - w0, QCH + (f - f0))
                        s_se.scatter(e, dst, 0)
                        s_sn.scatter(s, dst, 0)
                    gl.barrier()
                    tail, held, tp, lack = _qappend(gl.minimum(n_k - w0, QCH), s_se, s_sn, c, cell, tail,
                                                    held, tp, lack, pid, rec0, table, stack, nb, seg_start, dir_offs,
                                                    dir_first, dir_words, dwords_r, lt32, LV, NUM_WARPS, GROUP_SIZE,
                                                    N_SLOTS, BLOCK, chk, round_id, ARENA_CHECK)
                    gl.barrier()
            st_g = en_g
        tails = gl.where(selc, tail, tails)
        helds = gl.where(selc, held, helds)
    fits = lack == 0
    gl.store(stats + 2 * pid, gl.where(fits, nb - tp, nb - tp0 + gl.sum(helds - gl.load(nblk + pid * N_SLOTS + ci,
                                                                                      mask=cv, other=0), axis=0)))
    gl.store(stats + 2 * pid + 1, 0)
    if fits:
        gl.store(run_count + pid * N_SLOTS + ci, tails, mask=cv)
        gl.store(nblk + pid * N_SLOTS + ci, helds, mask=cv)
        gl.store(top + pid, tp)
    else:
        gl.atomic_max(flags + FLAG_STOPPED, 1)
        gl.atomic_max(ctl, 1)
    gl.barrier()
    if TRACE:
        n_ev = mark(trace, n_ev, 2, total, TRACE_CAP)
    return n_ev, fits


# ------------------------------------------------- phase 2: delivery


@gluon.jit
def _deliver(state, pid, n_groups, plan, n_conns, ring, k_lo, k_hi, offs, lane, mask, received_ptr, n_received_ptr,
             log_ptr, log_cap, trace, n_ev, ectx, RECEIVE: gl.constexpr, GROUP_SIZE: gl.constexpr,
             N_SLOTS: gl.constexpr, STREAM: gl.constexpr, U: gl.constexpr,
             SPT: gl.constexpr, NUM_WARPS: gl.constexpr, MAX_CONNS: gl.constexpr, LOG: gl.constexpr,
             COUNT: gl.constexpr, STAGE_E: gl.constexpr, TRACE: gl.constexpr, TRACE_CAP: gl.constexpr,
             BLOCK: gl.constexpr, chk, ARENA_CHECK: gl.constexpr, STAGE_W: gl.constexpr = True,
             T_A: gl.constexpr = 1.0, T_C: gl.constexpr = 0.0, T_B: gl.constexpr = 0.0,
             QUEUES: gl.constexpr = False, t_cut=0.0, RECV1: gl.constexpr = None, SEND1: gl.constexpr = None):
    """Phase 2 of program pid: its group's due slots k_lo..k_hi, every link
    through RECEIVE (see the module docstring); state: the tuple of node
    attrs receive touches, [GROUP_SIZE] tensors in the program's neuron
    layout (lane's). STAGE_E: the walk stages each record's time word too
    (LOG, or a receive that reads time). ectx: the run's history slot for spikes inside delivery (_emit_ctx;
    () without any). QUEUES: the population's queues (_pull_queues) merged by arrival up to t_cut (the run's
    end), each from its head, their segment records walked in batches (_walk_batch: RECV1, SEND1 -- the log's
    time word the arrival, not the emission); then each head moved on, its passed blocks freed. Returns
    (state, n_ev)."""
    # neurons over lanes and warps (the program's layout, LN); during delivery as rows, one per warp (LWN:
    # [NUM_WARPS, GROUP_SIZE // NUM_WARPS]) -- each warp walks the links to its own neurons (_walk)
    LT: gl.constexpr = gl.BlockedLayout([SPT, U], [32, 1], [NUM_WARPS, 1], [1, 0])
    LN: gl.constexpr = gl.SliceLayout(1, LT)
    LWN: gl.constexpr = gl.BlockedLayout([1, SPT], [1, 32], [NUM_WARPS, 1], [1, 0])
    LW: gl.constexpr = gl.SliceLayout(1, LWN)
    LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    NPW: gl.constexpr = GROUP_SIZE // NUM_WARPS
    gl.static_assert(NPW == 32 * SPT, "a group: SPT neurons a lane, every lane of every warp")
    (_hist_addr, _topo_addr, _link_target_addr, attr_addr, _dir_addr, _hcap_p, _t_src_p, _t_seg_p, _t_bkt_p,
     _t_lnk_p, _dpairs_p, _dwords_p, _last_read) = _fused_plan(plan, n_conns)
    (rec_time, rec_src, rec_lo, rec_n, run_count, head_p, _flags, table, stack, top, nblk, lost, _log_rec,
     _ovf_cap, run_cap, nb, _stats) = _ring(ring, BLOCK)
    if COUNT:  # dev
        received = gl.load(received_ptr + offs, mask=mask, other=0.0)
        n_received = gl.load(n_received_ptr + offs, mask=mask, other=0)
    else:
        received = gl.zeros([GROUP_SIZE], gl.float32, layout=LN)
        n_received = gl.zeros_like(lane)
    log_e = (log_ptr + 4 * log_cap + 1).to(gl.pointer_type(gl.float32))
    # the stream block's fields, staged as each warp's list (_stage_records: [STREAM // U, NUM_WARPS, U], the
    # walk reads [NUM_WARPS, U] a step), and the connections' attribute pointers (weights by link index), a
    # read-only table
    # GAP (D-7): one block pointer per connection; one LDS stage buffer per attr slot -- the most
    #   attrs any variant's receive reads (here: the weight), D-7
    cq = gl.arange(0, MAX_CONNS, layout=LV)
    s_atp = gl.allocate_shared_memory(gl.int64, [MAX_CONNS], SM)
    if STAGE_W:  # (no per-link attr read: the table unused)
        s_atp.store(gl.load(attr_addr + cq, mask=cq < n_conns, other=0))
    if not QUEUES:  # (link records: the link walk's staging)
        f_tgt = gl.allocate_shared_memory(gl.int32, [STREAM * NUM_WARPS], SM)
        f_w = gl.allocate_shared_memory(gl.float32, [STREAM * NUM_WARPS], SM)
        if LOG:
            f_src = gl.allocate_shared_memory(gl.int32, [STREAM * NUM_WARPS], SM)
        else:  # placeholder, never touched
            f_src = f_tgt
        if STAGE_E:
            f_e = gl.allocate_shared_memory(gl.float32, [STREAM * NUM_WARPS], SM)
        else:  # placeholder, never touched
            f_e = f_w
    else:  # the queues' merged records (_merge_segments), each queue's candidates' arrivals, its delay
        s_arr = gl.allocate_shared_memory(gl.float32, [MAX_CONNS * STREAM], SM)
        m_n = gl.allocate_shared_memory(gl.int32, [2 * STREAM], SM)
        m_src = gl.allocate_shared_memory(gl.int32, [2 * STREAM], SM)
        m_t = gl.allocate_shared_memory(gl.float32, [2 * STREAM], SM)
        cv = gl.arange(0, MAX_CONNS, layout=LV)
        # (a connection's one delay: its minimum)
        dcls = gl.load((plan + 20 * n_conns + cv).to(gl.pointer_type(gl.float32)), mask=cv < n_conns, other=0.0)
    if not QUEUES:
        RW: gl.constexpr = [STREAM // U, NUM_WARPS * U]  # (a row: a walk step's [NUM_WARPS, U])
        r_tgt = f_tgt.reinterpret(gl.int32, RW, SM)
        r_w = f_w.reinterpret(gl.float32, RW, SM)
        r_src = f_src.reinterpret(gl.int32, RW, SM)
        r_e = f_e.reinterpret(gl.float32, RW, SM)
    wi = gl.arange(0, NUM_WARPS, layout=LW)
    ji = gl.arange(0, NPW, layout=gl.SliceLayout(0, LWN))
    lane2 = wi[:, None] * NPW + ji[None, :]
    node2 = pid * GROUP_SIZE + lane2  # (the rows' neurons: receive's n2.spike())
    mask2 = _to_rows(mask.to(gl.int32), NUM_WARPS, GROUP_SIZE, LWN) > 0
    state = _state_to_rows(state, NUM_WARPS, GROUP_SIZE, LWN)
    received = _to_rows(received, NUM_WARPS, GROUP_SIZE, LWN)
    n_received = _to_rows(n_received, NUM_WARPS, GROUP_SIZE, LWN)
    gl.barrier()
    # receive's time (D-1): T_A e + tk, tk the slot's term (see _deliver_link) -- a slot's records in emission
    #   order, so per target non-decreasing either way; queues: e + D_c, merged by arrival
    tp = gl.load(top + pid)  # the group's free-stack top through the delivery, stored once at its end
    lost_v = gl.load(lost)
    if QUEUES:  # segment records: the walk's buffers (_walk_batch)
        QL: gl.constexpr = QLC  # a batch's links (Brunel: ~25 a segment)
        m_lo = gl.allocate_shared_memory(gl.int32, [2 * STREAM], SM)
        s_off = gl.allocate_shared_memory(gl.int32, [STREAM], SM)
        ib_w = gl.allocate_shared_memory(gl.float32, [2 * QL], SM)  # (a batch's inboxes; past QL: a dump)
        ib_a = gl.allocate_shared_memory(gl.float32, [2 * QL], SM)
        b_cnt = gl.allocate_shared_memory(gl.int32, [GROUP_SIZE], SM)
        b_start = gl.allocate_shared_memory(gl.int32, [GROUP_SIZE], SM)
        b_run = gl.allocate_shared_memory(gl.int32, [GROUP_SIZE], SM)
        b_tmp = gl.allocate_shared_memory(gl.int32, [32 * NUM_WARPS], SM)
        if LOG:
            ib_s = gl.allocate_shared_memory(gl.int32, [2 * QL], SM)
        else:
            ib_s = b_tmp
        s_ltp = _table(gl.load(_link_target_addr + cq, mask=cq < n_conns, other=0), SM)
        gl.barrier()
    if QUEUES:
        # the queues' segment records merged by arrival, a block at a time, until nothing more arrives before
        # t_cut; each block's segments walked (_walk_batch) in the program's neuron layout
        cellv = pid * N_SLOTS + cv
        cvm = cv < N_SLOTS
        tails = gl.load(run_count + cellv, mask=cvm, other=0)
        heads = gl.load(head_p + cellv, mask=cvm, other=0)
        qstate = _state_from_rows(state, GROUP_SIZE, LN)
        qrec = _from_rows(received, GROUP_SIZE, LN)
        qnrec = _from_rows(n_received, GROUP_SIZE, LN)
        gl.barrier()
        cur = gl.zeros_like(cv)
        more = gl.to_tensor(1)
        while more > 0:
            nq, taken = _merge_segments(rec_time, rec_src - 1, pid * N_SLOTS, cur, tails, dcls, cv, s_arr, m_t,
                                        m_src, m_lo, m_n, table, STREAM, NUM_WARPS, N_SLOTS, BLOCK, nb,
                                        heads, t_cut)
            cur = cur + taken
            more = nq
            if TRACE:
                n_ev = mark(trace, n_ev, 5, nq, TRACE_CAP)
            qstate, qrec, qnrec = _walk_batch(nq, m_t, m_src, m_lo, m_n, s_off, ib_w, ib_a, b_cnt, b_start, b_run,
                                              b_tmp, s_atp, s_ltp, qstate, qrec, qnrec, lane, mask, pid, ectx, RECV1,
                                              GROUP_SIZE, STREAM, NUM_WARPS, COUNT, STAGE_W, LOG, ib_s, log_ptr, log_e,
                                              log_cap, k_lo, SEND1, QL)
            if TRACE:
                n_ev = mark(trace, n_ev, 6, nq, TRACE_CAP)
            gl.barrier()
        state = _state_to_rows(qstate, NUM_WARPS, GROUP_SIZE, LWN)
        received = _to_rows(qrec, NUM_WARPS, GROUP_SIZE, LWN)
        n_received = _to_rows(qnrec, NUM_WARPS, GROUP_SIZE, LWN)
        # each queue: its head on, its passed blocks back on the free stack, its table shifted down
        for c in gl.static_range(N_SLOTS):
            cell = pid * N_SLOTS + c
            h = gl.sum(gl.where(cv == c, heads + cur, 0), axis=0)
            n = gl.sum(gl.where(cv == c, tails, 0), axis=0)
            held = gl.load(nblk + cell)
            freed = gl.minimum(h // BLOCK, held)
            if freed > 0:
                tp = _free_blocks(table, stack, tp, pid, cell, 0, freed, lost_v, LV, nb, NUM_WARPS, chk, k_lo,
                                  ARENA_CHECK)
                _shift_table(table, cell, nb, freed, held, LV, NUM_WARPS)
                gl.store(nblk + cell, held - freed)
            gl.store(run_count + cell, n - freed * BLOCK)
            gl.store(head_p + cell, h - freed * BLOCK)
    else:  # the slotted ring: its due slots (not compiled for queues)
        for k in range(k_lo, k_hi + 1):
            cell = pid * N_SLOTS + k % N_SLOTS
            tk = k.to(gl.float32) * T_C + T_B  # (the delivery time's per-slot term: _deliver_link)
            nm = gl.minimum(gl.load(run_count + cell), run_cap)
            held = gl.load(nblk + cell)  # (for freeing below: read with the counts, its latency hidden)
            if TRACE:
                nm = pin_s(nm)
                n_ev = mark(trace, n_ev, 3, nm, TRACE_CAP)
            # the slot's link records are its stream: STREAM links at a time
            for b0 in range(0, nm, STREAM):
                nq, n_w, n_max = _stage_records(rec_src - 1, cell, b0, nm, s_atp, f_tgt, f_w, f_src, f_e,
                                                table, LW, STREAM, U, NUM_WARPS, GROUP_SIZE, LOG, STAGE_E, N_SLOTS,
                                                run_cap, BLOCK, nb, STAGE_W)
                if TRACE:
                    n_ev = mark(trace, n_ev, 5, nq, TRACE_CAP)
                state, received, n_received = _walk(
                    r_tgt, r_w, r_src, r_e, n_w, n_max, state, received, n_received, lane2, mask2, k, pid, log_ptr,
                    log_e, log_cap, LW, RECEIVE, GROUP_SIZE, U, NUM_WARPS, LOG, COUNT, STAGE_E, tk, T_A, ectx,
                    node2)
                if TRACE:
                    state = _pin_first(state)
                    n_ev = mark(trace, n_ev, 6, nq, TRACE_CAP)
                gl.barrier()
                if TRACE >= 3:
                    n_ev = mark(trace, n_ev, 22, 0, TRACE_CAP)
            # the delivered cell's blocks back on the group's free stack
            tp = _release_cell(table, stack, tp, pid, cell, held, nblk, lost_v, LV, nb, NUM_WARPS, chk, k,
                               ARENA_CHECK)
            if TRACE >= 3:
                n_ev = mark(trace, n_ev, 23, pin_s(tp), TRACE_CAP)
            gl.store(run_count + cell, 0)
    gl.store(top + pid, tp)
    state = _state_from_rows(state, GROUP_SIZE, LN)
    received = _from_rows(received, GROUP_SIZE, LN)
    n_received = _from_rows(n_received, GROUP_SIZE, LN)
    if COUNT:  # dev
        gl.store(received_ptr + offs, received, mask=mask)
        gl.store(n_received_ptr + offs, n_received, mask=mask)
    if TRACE:
        n_ev = mark(trace, n_ev, 7, 0, TRACE_CAP)
    return state, n_ev


# ------------------------------------------------- phase 3: emission


@gluon.jit
def _emit_spikes(fires, e, offs, pid, n_groups, history, hist_cap, run_id, K: gl.constexpr, SPT: gl.constexpr,
                 NUM_WARPS: gl.constexpr, REGIONS: gl.constexpr = False):
    """One n2.spike() site's spikes into the history: fires (which lanes
    spiked) and e (their emission times), [GROUP_SIZE] in the program's
    neuron layout -- REGIONS (a non-uniform producer): into program pid's
    own region of the run (hist_cap spikes, history.py), after what its
    earlier sites put there, in lane order: a fixed place, whatever the
    other programs do; else into the run's one flat list (hist_cap: its
    cap), where one count atomic places them. A run can call
    this any number of times (D-3: several sites, a neuron spiking more than
    once, a site in a loop)."""
    LS: gl.constexpr = gl.BlockedLayout([SPT], [32], [NUM_WARPS], [0])
    _n_fire = fires.to(gl.int32)
    _local_pos = gl.convert_layout(gl.cumsum(gl.convert_layout(_n_fire, LS), axis=0), fires.type.layout) - _n_fire
    n_emit = gl.sum(_n_fire, axis=0)
    # the region holds hist_cap (EmissionHistory: its run cap over the programs, default one spike per neuron);
    #   past it the overflow flag (the runtime raises: PopulationRuntime.check)
    if REGIONS:
        (h_tag, h_count, h_wend, h_min, h_counts, h_node, h_time, h_overflow) = history_ptrs(
            history, run_id % K, K, n_groups, hist_cap)
        base = gl.atomic_add(h_counts + pid, n_emit)  # (the program's own count: its sites one after another)
        gl.atomic_add(h_count, n_emit)  # (the run's: a sum, in any order)
        at = pid * hist_cap + base + _local_pos
    else:
        (h_tag, h_count, h_wend, h_min, h_counts, h_node, h_time, h_overflow) = history_ptrs(
            history, run_id % K, K, 1, hist_cap)
        base = gl.atomic_add(h_count, n_emit)  # this site's place in the run's flat list
        at = base + _local_pos
    put = fires & (base + _local_pos < hist_cap)
    if base + n_emit > hist_cap:
        gl.atomic_max(h_overflow, 1)
    gl.store(h_node + at, offs, mask=put)
    gl.store(h_time + at, e, mask=put)
    gl.atomic_min(h_min, gl.min(gl.where(put, e, float("inf")), axis=0))


@gluon.jit
def _any_lane(m):
    """Whether any lane of the mask m is set: a lowered while loop's test (IfTransformer) -- m [GROUP_SIZE] (the
    program's neurons) or [NUM_WARPS, GROUP_SIZE // NUM_WARPS] (delivery's rows, in receive)."""
    x = m.to(gl.int32)
    if len(x.shape) == 2:
        x = gl.max(x, axis=1)
    return gl.max(x, axis=0) > 0


@gluon.jit
def _emit_ctx(pid, n_groups, history, hist_cap, run_id, K: gl.constexpr):
    """Program pid's region of the run's history slot, for spikes inside delivery (_emit_delivered): (its count,
    the run's count, earliest, its nodes, its times, overflow, its cap)."""
    (_h_tag, h_count, _h_wend, h_min, h_counts, h_node, h_time, h_overflow) = history_ptrs(
        history, run_id % K, K, n_groups, hist_cap)
    return (h_counts + pid, h_count, h_min, h_node + pid * hist_cap, h_time + pid * hist_cap, h_overflow, hist_cap)


@gluon.jit
def _emit_delivered(fire, t, nodes, ectx):
    """An n2.spike() site inside delivery (D-3: in receive, or a send after every delivery): the lanes in fire
    spike at t, straight into the program's region of the run's history (ectx: _emit_ctx) -- a run's slot is
    read only after the run. One link a step: at most one lane fires per row of
    fire (the walk: a row per warp, each its own target lane; a merge: the one lane), so per row one masked
    atomic, none where nothing fires. nodes: the lanes' neurons; t: the spike time (broadcast to fire)."""
    p_count, h_count, h_min, h_node, h_time, h_overflow, hist_cap = ectx
    if len(fire.shape) == 1:  # (the program's neurons: any number of lanes -- a segment's targets, _walk_batch)
        # (each firing lane its place by an atomic -- the region is ordered at the run's end, _sort_region)
        f = fire.to(gl.int32)
        n1 = gl.sum(f, axis=0)
        if n1 > 0:
            zero1 = gl.zeros_like(f)
            pos1 = gl.atomic_add(p_count + zero1, 1, mask=fire)
            gl.atomic_add(h_count, n1)
            put1 = fire & (pos1 < hist_cap)
            tt1 = t + gl.zeros_like(f.to(gl.float32))
            gl.store(h_node + pos1, nodes, mask=put1)
            gl.store(h_time + pos1, tt1, mask=put1)
            gl.atomic_min(h_min, gl.min(gl.where(put1, tt1, float("inf")), axis=0))
            gl.atomic_max(h_overflow + zero1, 1, mask=fire & (pos1 >= hist_cap))
    else:
        AX: gl.constexpr = len(fire.shape) - 1
        tt = t + gl.zeros_like(fire.to(gl.float32))
        n = gl.sum(fire.to(gl.int32), axis=AX)
        node = gl.sum(gl.where(fire, nodes, 0), axis=AX)
        tm = gl.sum(gl.where(fire, tt, 0.0), axis=AX)
        m = n > 0
        zero = n * 0  # (n: a row per warp, or a scalar -- zeros_like takes no scalar)
        pos = gl.atomic_add(p_count + zero, 1, mask=m)  # (GAP, SPIKES.md: the warps' order within the region)
        gl.atomic_add(h_count + zero, 1, mask=m)
        put = m & (pos < hist_cap)
        gl.store(h_node + pos, node, mask=put)
        gl.store(h_time + pos, tm, mask=put)
        gl.atomic_min(h_min + zero, tm, mask=put)
        gl.atomic_max(h_overflow + zero, 1, mask=m & (pos >= hist_cap))


@gluon.jit
def _region_reset(pid, n_groups, history, hist_cap, run_id, K: gl.constexpr):
    """Program pid's count of the run's slot to 0, at its run's start, before it emits (REGIONS): the slot's old
    run stays whole until its producer's next run writes it -- a consumer still reading it (K - 1 runs behind)
    reads its region counts mid-pull, after its header (history.py). The program the count's only writer."""
    (_ht, _hc, _hw, _hm, h_counts, _hn, _hti, _ho) = history_ptrs(history, run_id % K, K, n_groups, hist_cap)
    gl.store(h_counts + pid, 0)


@gluon.jit
def _sort_region(pid, n_groups, history, hist_cap, run_id, K: gl.constexpr, GROUP_SIZE: gl.constexpr,
                 NUM_WARPS: gl.constexpr):
    """Program pid's region of the run's history (history.py) ordered by (e, node), in place -- a non-uniform
    producer's (SPIKES.md: a consumer's queues need a source's runs in e order; the total key makes it the same
    whatever order the warps emitted in). Up to 4 GROUP_SIZE spikes (four a neuron) through LDS: each one's rank among them
    (the ones before it in (e, node, position)), then each written to its rank. More: left as it is, the
    overflow flag set (EmissionHistory: R past 4 GROUP_SIZE raises)."""
    (_ht, _hc, _hw, _hm, h_counts, h_node, h_time, h_overflow) = history_ptrs(history, run_id % K, K, n_groups,
                                                                              hist_cap)
    RS: gl.constexpr = 4 * GROUP_SIZE
    LI: gl.constexpr = 32 * NUM_WARPS
    JB: gl.constexpr = 32
    node = h_node + pid * hist_cap
    time = h_time + pid * hist_cap
    gl.barrier()  # (the program's spikes all in its region: every warp's stores before the reads)
    n = gl.minimum(gl.load(h_counts + pid), hist_cap)
    if n > RS:
        gl.atomic_max(h_overflow, 1)
    elif n > 1:
        SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        L1: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
        L2: gl.constexpr = gl.BlockedLayout([1, JB], [32, 1], [NUM_WARPS, 1], [1, 0])
        LIr: gl.constexpr = gl.SliceLayout(1, L2)
        LJr: gl.constexpr = gl.SliceLayout(0, L2)
        q = gl.arange(0, RS, layout=L1)
        s_e = _table(gl.load(time + q, mask=q < n, other=float("inf")), SM)
        s_n = _table(gl.load(node + q, mask=q < n, other=0), SM)
        gl.barrier()
        for i0 in range(0, n, LI):
            i = i0 + gl.arange(0, LI, layout=LIr)
            vi = i < n
            ei = s_e.gather(gl.minimum(i, RS - 1), 0)
            ni = s_n.gather(gl.minimum(i, RS - 1), 0)
            rank = gl.zeros([LI], gl.int32, layout=LIr)
            for j0 in range(0, n, JB):
                j = j0 + gl.arange(0, JB, layout=LJr)
                ej = s_e.gather(gl.minimum(j, RS - 1), 0)[None, :]
                nj = s_n.gather(gl.minimum(j, RS - 1), 0)[None, :]
                e2 = ei[:, None]
                n2 = ni[:, None]
                before = (ej < e2) | ((ej == e2) & ((nj < n2) | ((nj == n2) & (j[None, :] < i[:, None]))))
                rank += gl.sum((before & (j[None, :] < n)).to(gl.int32), axis=1)
            gl.store(time + rank, ei, mask=vi)
            gl.store(node + rank, ni, mask=vi)


@gluon.jit
def _emit_close(pid, n_groups, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K: gl.constexpr,
                TRACE: gl.constexpr, TRACE_CAP: gl.constexpr, REGIONS: gl.constexpr = False):
    """The run's end: program 0 writes its header (window end, then the tag
    -- one store, never a new run id with an old round) and clears the next
    run's slot's header (its regions' counts: each program at its own run's
    start, _region_reset). Returns n_ev (tracing)."""
    if TRACE:
        n_ev = mark(trace, n_ev, 8, 0, TRACE_CAP)
    # GAP (D-11): edge backward out of scope -- needs this run's spikes at every incoming link of each firing
    #   target (a reverse topology), D-11
    G_h = n_groups if REGIONS else 1
    (n_tag, n_count, _nw, n_min, _ncs, _nn, _nt, _no) = history_ptrs(history, (run_id + 1) % K, K, G_h, hist_cap)
    if pid == 0:
        (h_tag, _hc, h_wend, _hm, _hcs, _hn, _ht, _ho) = history_ptrs(history, run_id % K, K, G_h, hist_cap)
        gl.store(h_wend, t_end)
        gl.store(h_tag, history_tag(run_id, round_id))  # one store: never a new run id with an old round
        gl.store(n_tag, history_tag(0, round_id))  # the next run's slot, cleared ahead (see history.py): a
        gl.store(n_count, 0)  # consumer that still needed it sees a gap in run ids and raises its overrun flag
        gl.store(n_min, float("inf"))
    if TRACE:
        n_ev = mark(trace, n_ev, 9, 0, TRACE_CAP)
        trace_end(trace, n_ev)
    return n_ev


@gluon.jit
def _emit(fires, e, offs, pid, n_groups, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K: gl.constexpr,
          SPT: gl.constexpr, NUM_WARPS: gl.constexpr, TRACE: gl.constexpr, TRACE_CAP: gl.constexpr):
    """A run with one n2.spike() site: _emit_spikes, then _emit_close."""
    _emit_spikes(fires, e, offs, pid, n_groups, history, hist_cap, run_id, K, SPT, NUM_WARPS)
    return _emit_close(pid, n_groups, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K, TRACE, TRACE_CAP)
