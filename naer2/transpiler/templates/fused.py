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
     (naer2/ring.py; pooled: past the blocks a run holds, into its cell's
     overflow log, moved into new blocks after the pull's last barrier --
     see "the pooled ring" below). Uniform
     sources: all fresh runs of one emission time as one stream of spike
     blocks (each spike's run and its connection's pointers from read-only
     LDS tables), a block of spikes x a block of their segments at a time,
     the link loop four links per step. Otherwise sequential per entry with
     a per-group watermark: main run if e >= the watermark and not
     emission-delayed, else sorted insertion into the side run -- correct
     in any spike order (the baseline; slow).
  2. delivery: the due slots of g's ring in slot order, straight from the
     link records: STREAM records at a time, each one 128-bit load, its
     fields staged flat in LDS, walked U links per broadcast LDS read, every
     link received in order by its target lane (the scalar receive: lanes
     compare against the link's target, no cross-lane traffic); a slot with
     side records merges main and side runs by e with two cursors; a
     delivered slot's blocks (pooled) go back to the free stack, but for
     the main run's reserve.
  3. update and send (the entry's), emission into the history (flat list
     per run, no per-group sort: consumers order what they need themselves).

Contracts: a run reads source runs from earlier rounds only (round tags);
the last-read run ids are double-buffered by the consumer's run parity (all
programs read the same values); entries arriving before the ring position
go to its first undelivered slot (late), at or past N_SLOTS ahead are far
future (flag 1), full runs flag 0, uniform violations flag 2, a consumer
falling behind its sources' history raises the overrun flag.

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
from naer2.ring import POOL_CNT_BITS as _POOL_CNT_BITS, POOL_MAX_BLOCKS as _POOL_MAX_BLOCKS
from naer2.ring import POOL_RESERVE as _POOL_RESERVE, _rec, _ring
from naer2.topology import SimulationTopology, directory_ptrs, topology_ptrs
from naer2.trace import mark, pin, pin_s, trace_end

FUSED_TRACE_TAGS = ("start", "state", "pulled", "slot", "headers", "streamed", "walked", "delivered", "sent", "end",
                    "runs", "spikes", "found", "appended",  # TRACE >= 2: inside the pull
                    "a:segments", "a:mean", "a:claimed", "a:targets", "a:stored",
                    # TRACE >= 3: inside the append, per tile
                    "f:bar", "f:fixed", "f:stored", "d:bar", "d:freed")  # (TRACE >= 3: temporary dissection)
PULL = dict(batch=True)  # PopulationRuntime default (naer2/runtime.py); tools/fused_sweep.py: batching ~30%
# (PD14, G=256, 780M); the group directory always on (as fast again over binary search)
# delivery walk: links staged in LDS per stream block, links per broadcast LDS read (tools/walk_sweep.py on
# the split kernels, PD14, G=256, 780M: larger U costs more in registers than it saves; an explicit
# occupancy target (waves_per_eu) gains <= 3% and spills beyond 12 waves)
WALK = dict(stream=256, unroll=8)
SPK = gl.constexpr(32)   # spikes per pull block (uniform sources)
MSEG = gl.constexpr(4)   # the pull's spike layout (LP2: MSEG // 4 per thread); the append packs its own tiles
POOL_RESERVE = gl.constexpr(_POOL_RESERVE)  # blocks every main run keeps (naer2/ring.py)
POOL_MAX_BLOCKS = gl.constexpr(_POOL_MAX_BLOCKS)  # blocks one main run can hold (the claim counter's field)
POOL_CNT_BITS = gl.constexpr(_POOL_CNT_BITS)  # the claim counter: count | blocks held << POOL_CNT_BITS


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
class FusedPlan:
    """A consumer's incoming connections, built once into one int32 buffer:
    per connection its source's EmissionHistory and its SimulationTopology
    addresses (+ the topology's link_target and attribute arrays, which the
    delivery walk reads, and its group directory), the history's cap, the
    topology's source, segment, bucket and link counts, its most segments of
    one source, the directory's pair count and words per source, and the
    last-read run ids (double-buffered by the consumer's run parity); per
    connection its minimum delay and its delay class, per class its delay."""

    def __init__(self, histories: list[EmissionHistory], topologies: list[SimulationTopology], n_groups: int, device,
                 min_delays: list[float] | None = None, classes: list[int] | None = None,
                 class_delays: list[float] | None = None):
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
        plan = torch.zeros(23 * n, dtype=torch.int32)
        links = [t.link_addresses() for t in topologies]
        dirs = [t.directory() for t in topologies]
        plan[: 10 * n].view(torch.int64)[:] = torch.tensor(
            [h.buf.data_ptr() for h in histories] + [t.buf.data_ptr() for t in topologies]
            + [a[1] for a in links] + [a[2] for a in links] + [d[0].data_ptr() for d in dirs], dtype=torch.int64)
        max_segs = [int(t.seg_offsets.diff().max()) if t.n_sources else 0 for t in topologies]
        for i, values in enumerate((
            [h.cap for h in histories], [t.n_sources for t in topologies], [t.n_segments for t in topologies],
            [t.n_buckets for t in topologies], [t.n_links for t in topologies], max_segs,
            [d[1] for d in dirs], [d[2] for d in dirs],
        )):
            plan[(10 + i) * n:(11 + i) * n] = torch.tensor(values, dtype=torch.int32)
        # per connection its minimum delay as the host schedules with it (a gated run's due time: _gate_skip)
        plan[20 * n:21 * n] = torch.tensor(min_delays if min_delays is not None else [0.0] * n,
                                           dtype=torch.float32).view(torch.int32)
        # the delay classes (several homogeneous delays, D-1: BRUNEL.md): per connection its class (its records'
        # cells: _append_links), per class its delay (the first n_classes words: _deliver_classes)
        if classes is not None:
            if len(classes) != n or len(class_delays) > n:
                raise ValueError("a class per connection, at most as many classes as connections")
            plan[21 * n:22 * n] = torch.tensor(classes, dtype=torch.int32)
            plan[22 * n:22 * n + len(class_delays)] = torch.tensor(class_delays, dtype=torch.float32).view(torch.int32)
        self.plan = plan.to(device)


# ------------------------------------------------- pull helpers


@gluon.jit
def _side_insert(side_time, side_src, side_lo, side_n, side_count, flags, cell, n_cells, e, srcword, lo, n,
                 side_table, stack, top, lost, g, side_cap, LB: gl.constexpr, BLK: gl.constexpr,
                 N_SLOTS: gl.constexpr, POOL_B: gl.constexpr, nb):
    """One entry into a side run, keeping it sorted by e (after every entry
    with time <= e). The program owns the run; barriers order the shift
    between its warps. Pooled (POOL_B > 0): a new block from the group's
    arena where the run reaches one; none left, the entry is lost (flag 3)."""
    lanes = gl.arange(0, BLK, layout=LB)
    cnt = gl.load(side_count + cell)
    ok = cnt < side_cap
    if POOL_B > 0:
        if ok & (cnt % POOL_B == 0):
            ok = _take_block(side_table, stack, top, g, cell, cnt // POOL_B, nb)
            gl.atomic_max(lost, 1, mask=~ok)
    if ok:
        p = cnt
        for s0 in range(0, cnt, BLK):
            q = s0 + lanes
            rq, _okq = _run_rec(cell, q, n_cells, side_table, N_SLOTS, POOL_B, nb, 0)
            v = gl.load(side_time + 4 * rq, mask=q < cnt, other=0.0)
            p = gl.minimum(p, gl.min(gl.where((q < cnt) & (v > e), q, cnt), axis=0))
        gl.barrier()
        for s0 in range(0, cnt - p, BLK):  # shift [p, cnt) up by one, top first
            q = cnt - 1 - s0 - lanes
            qm = q >= p
            r_src, _ok_src = _run_rec(cell, q, n_cells, side_table, N_SLOTS, POOL_B, nb, 0)
            r_dst, _ok_dst = _run_rec(cell, q + 1, n_cells, side_table, N_SLOTS, POOL_B, nb, 0)
            src_at = 4 * r_src
            dst_at = 4 * r_dst
            tv = gl.load(side_time + src_at, mask=qm, other=0.0)
            sv = gl.load(side_src + src_at, mask=qm, other=0)
            lv = gl.load(side_lo + src_at, mask=qm, other=0)
            nv = gl.load(side_n + src_at, mask=qm, other=0)
            gl.barrier()  # every warp has read before any overwrites
            gl.store(side_time + dst_at, tv, mask=qm)
            gl.store(side_src + dst_at, sv, mask=qm)
            gl.store(side_lo + dst_at, lv, mask=qm)
            gl.store(side_n + dst_at, nv, mask=qm)
            gl.barrier()
        r_p, _ok_p = _run_rec(cell, p, n_cells, side_table, N_SLOTS, POOL_B, nb, 0)
        at = 4 * r_p
        gl.store(side_time + at, e)
        gl.store(side_src + at, srcword)
        gl.store(side_lo + at, lo)
        gl.store(side_n + at, n)
        gl.store(side_count + cell, cnt + 1)
    else:
        gl.atomic_max(flags + 3, 1)  # side run full / the arena ran out (main: flag 0)
    gl.barrier()


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


# ------------------------------------------------- the pooled ring (naer2/ring.py)
#
# A record's place: its run's block (_table_blk: fixed for a main run's first POOL_RESERVE blocks, else the
# run's block table), then its index in the group's arena (_blk_rec) -- _run_rec, both at once. A run outgrows
# its blocks only in the pull: there, _append_links writes past them into the cell's overflow log and
# _pool_fixup gives it blocks after the pull's last barrier; delivery frees them (_free_blocks); the general
# pull and the side runs take blocks one at a time (_take_block).


@gluon.jit
def _run_rec(cell, pos, n_cells, table, N_SLOTS: gl.constexpr, POOL_B: gl.constexpr, nb,
             RES: gl.constexpr):
    """(record index, valid) of entry pos of a cell's run: interleaved across cells (_rec), or (POOL_B > 0)
    in its group's arena, not valid where a block is missing (-1: the arena ran out when it was claimed;
    its links are lost). RES: the run's fixed blocks (a main run's POOL_RESERVE, a side run's none)."""
    return _blk_rec(cell, pos, n_cells, _table_blk(table, cell, pos, N_SLOTS, POOL_B, nb, RES), N_SLOTS,
                    POOL_B, nb)


@gluon.jit
def _table_blk(table, cell, pos, N_SLOTS: gl.constexpr, POOL_B: gl.constexpr, nb,
               RES: gl.constexpr):
    """The block id of entry pos of a cell's run (pooled; 0 unpooled: unused): within its first RES blocks
    fixed (slot s's blocks s * RES ...: no table read), past them from its block table (a masked read: in
    quiet runs hardly any lane has one)."""
    if POOL_B > 0:
        fixed = pos < RES * POOL_B
        blk = gl.load(table + cell * nb + gl.minimum(pos // POOL_B, nb - 1), mask=~fixed, other=0)
        return gl.where(fixed, (cell % N_SLOTS) * RES + pos // POOL_B, blk)
    else:
        return pos * 0  # (pos may be a scalar)


@gluon.jit
def _blk_rec(cell, pos, n_cells, blk, N_SLOTS: gl.constexpr, POOL_B: gl.constexpr, nb):
    """(record index, valid) of entry pos of a cell's run from its block id (_table_blk); unpooled: _rec."""
    if POOL_B > 0:
        return ((cell // N_SLOTS) * nb + gl.maximum(blk, 0)) * POOL_B + pos % POOL_B, blk >= 0
    else:
        return _rec(cell, pos, n_cells), pos >= 0


@gluon.jit
def _take_block(table, stack, top, g, cell, b, nb):
    """Scalar, one block off group g's free stack into entry b of a run's block table (-1 if the arena ran
    out): for the general pull (sequential per record, the baseline) and the side runs (sorted insertion).
    Returns whether it got one."""
    tp = gl.load(top + g)
    bid = gl.load(stack + g * nb + gl.maximum(tp - 1, 0))
    ok = (tp > 0) & (bid >= 0)  # (a -1 on the stack: a block lost when the arena ran out)
    gl.barrier()  # every warp has read the top before any writes it
    gl.store(table + cell * nb + b, gl.where(ok, bid, -1))
    gl.store(top + g, gl.maximum(tp - 1, 0))
    gl.barrier()
    return ok


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
def _pool_fixup(s_cnt, cnt0, rec0, table, stack, log_rec, ovf_cap, top0, g, sl, LV: gl.constexpr,
                N_SLOTS: gl.constexpr, main_cap, POOL_B: gl.constexpr, nb,
                NUM_WARPS: gl.constexpr, NSP: gl.constexpr, chk, when, ARENA_CHECK: gl.constexpr):
    """After the pull's last barrier, only if some slot grew past its blocks (from the counts in LDS: the same
    answer in every warp). First the check: everything fits -- every slot's log within ovf_cap, the blocks
    the slots lack within the free stack, every run within main_cap. If not, nothing is changed (the pull
    wrote only past the committed counts and into the logs: scratch) and the run stops (fits False: the
    host resizes and runs it again). If it fits, every such slot takes what it needs -- + as much again as
    it grew this run where the stack has room for all of that (in a burst, the next run then mostly fits)
    -- and its log's records move into the new blocks. Returns (fits, the new top, the blocks the group
    holds -- would hold, if it didn't fit --, the most any slot put into its log): the run's status, the
    runtime sizes the arenas and logs from it."""
    MASK: gl.constexpr = (1 << POOL_CNT_BITS) - 1
    packed = s_cnt.load(LV)
    cnt = packed & MASK
    held = packed >> POOL_CNT_BITS
    live = sl < N_SLOTS
    need = (cnt + POOL_B - 1) // POOL_B
    new_top = top0
    most = gl.to_tensor(0)
    lack = gl.where(live & (need > held), need - held, 0)
    # (past what a run can hold at all: never fits; past the arena alone, lack > the free stack already)
    lack = gl.where(live & (cnt > POOL_MAX_BLOCKS * POOL_B), 1 << 24, lack)
    total = gl.sum(lack, axis=0)  # (the one reduction a run without growth pays)
    fits = gl.to_tensor(1) > 0
    if total > 0:
        over = gl.where(live, gl.maximum(cnt - held * POOL_B, 0), 0)  # records in each slot's log
        most = gl.max(over, axis=0)
        fits = fits & (most <= ovf_cap) & (total <= top0)
        if fits:
            grown = (gl.maximum(cnt - cnt0, 0) + POOL_B - 1) // POOL_B
            want = gl.where(lack > 0, gl.minimum(need + grown, POOL_MAX_BLOCKS) - held, 0)
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
            s_cnt.store((packed & MASK) | ((held + want) << POOL_CNT_BITS))
            gl.barrier()
            # every slot's log into its new blocks, all slots' records in one pass over the lanes: record f of
            # the flat list belongs to the last slot whose first record is <= f (a binary search over the slots'
            # firsts in LDS; slots logging nothing share their first with the next one), its index in that
            # slot's log f - first (one scan for all slots, not a pass with two reductions per slot)
            SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
            first = gl.cumsum(over, axis=0) - over
            n_all = gl.sum(over, axis=0)
            t_first = _table(first, SM)
            t_h = _table(held * POOL_B, SM)
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
                blk = gl.load(table + cs * nb + gl.minimum(p // POOL_B, nb - 1), mask=m, other=0)
                dst = 4 * ((g * nb + blk) * POOL_B + p % POOL_B)
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
def _link_record(rec0, word, blk, j, w, n_l, lo_l, base, lim, cell, n_cells, e2, srcw2, log_rec, ovf_cap,
                 T: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr, N_SLOTS: gl.constexpr,
                 POOL_B: gl.constexpr, nb):
    """Every tile segment's j-th link as its ring record; word: the 32-bit word holding its target; blk: its
    record's block id (pooled, _table_blk). Pooled, a position past the blocks its cell holds (lim) goes into
    its cell's overflow log instead (entry position - lim, ovf_cap entries per cell; log_rec: the index of
    the logs' records; _pool_fixup moves it) -- either way one store of the record, its address arithmetic,
    nothing loaded."""
    wj = w & (j < n_l)
    l = lo_l + j
    tgt = _target_in(word, l, GROUP_SIZE)
    pos = base + j
    rec, ok = _blk_rec(cell, pos, n_cells, blk, N_SLOTS, POOL_B, nb)
    if POOL_B > 0:
        direct = pos < lim
        in_log = (~direct) & (pos - lim < ovf_cap)  # (the cell's log full: not written, _pool_fixup stops the run)
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
def _append_links(s, e, main, c, a, b, seg_start, seg_gb, bucket_mean, link_tgt, s_cnt, g, k_lo, slot_width, n_cells,
                  main_time, main_src, main_lo, main_n, flags, trace, n_ev, table, log_rec, ovf_cap,
                  N_SLOTS: gl.constexpr, main_cap, LP2: gl.constexpr, LM: gl.constexpr,
                  SPK: gl.constexpr, MSEG: gl.constexpr,
                  TRACE: gl.constexpr, TRACE_CAP: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
                  POOL_B: gl.constexpr, nb, cls, NC: gl.constexpr):
    """A block of spikes (uniform sources): every segment [a, b) of the
    ones going to the main runs claims a range of its slot's run in group
    g's ring -- the per-slot link counts live in LDS (s_cnt), the program
    being the run's only writer -- and writes one record per link there, in
    link order: (e, source word, link index, target). The block's segments
    packed into tiles of one lane each (32 x NUM_WARPS). cls: each spike's
    connection's delay class (NC classes: N_SLOTS cells per group, NC rings of
    N_SLOTS // NC slots -- class q's slot k in cell q N_SLOTS / NC + k % (N_SLOTS / NC))."""
    NR: gl.constexpr = N_SLOTS // NC  # the ring's slots (a class's)
    # Packed tiles: the block's (spike, segment) pairs as one dense list, T = one lane each (no replicated
    # lanes) -- a tile's work follows the block's segments in all, not its spike with the most (a spike
    # reaches a group through a segment per delay bucket: up to ~20 in PD14, mostly 1-2 links each). A lane
    # finds its spike by binary search over the spikes' first pairs, and the spike's values in LDS tables.
    SMT: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    T: gl.constexpr = 32 * NUM_WARPS
    LT: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    m = gl.where(main, b - a, 0)
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
    if NC > 1:
        t_cls = _table(cls + zero, SMT)
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
        far = ok & (k - k_lo >= NR)
        # the flags per lane (masked: normally no lane), not through reductions across warps
        gl.atomic_max(flags + 1 + gl.zeros_like(k), 1, mask=far)
        w = ok & ~far
        n_iter = gl.max(gl.where(w, n_l, 0), axis=0)  # before the full check: a bound
        if NC > 1:  # (its class's ring)
            slot = t_cls.gather(sp, 0) * NR + k % NR
        else:
            slot = k % N_SLOTS
        old = s_cnt.atomic_scatter_add(gl.where(w, n_l, 0), slot, 0)  # the range's start (pooled: | blocks held)
        old = gl.convert_layout(old, LT)
        if POOL_B > 0:
            # the claim's start and its cell's blocks come back together: a lane knows alone which of its
            # positions its cell holds (written directly) and which it doesn't (into the cell's overflow log)
            # -- no lane waits on another, no barrier
            base = old & ((1 << POOL_CNT_BITS) - 1)
            lim = (old >> POOL_CNT_BITS) * POOL_B
        else:
            base = old
            lim = old
        if TRACE >= 3:
            base = pin(base)
            n_ev = mark(trace, n_ev, 16, 0, TRACE_CAP)
        w = w & ~(base + n_l > main_cap)  # past the capacity: not written (the pull's end stops the run)
        cell = g * N_SLOTS + slot
        # link by link over the tile: iteration j writes every segment's j-th link (work ~ the tile's longest
        # segment: in PD14 a few links); four links per iteration (_link_loads: their loads in flight together)
        e2 = e1
        lt32 = t_lt.gather(sp, 0).to(gl.pointer_type(gl.int32))  # the targets' words (4-byte aligned)
        rec0 = main_src - 1
        for j in range(0, n_iter, 4):
            x0 = gl.load(lt32 + gl.where(w & (j < n_l), _target_word(lo_l + j, GROUP_SIZE), 0))
            x1 = gl.load(lt32 + gl.where(w & (j + 1 < n_l), _target_word(lo_l + j + 1, GROUP_SIZE), 0))
            x2 = gl.load(lt32 + gl.where(w & (j + 2 < n_l), _target_word(lo_l + j + 2, GROUP_SIZE), 0))
            x3 = gl.load(lt32 + gl.where(w & (j + 3 < n_l), _target_word(lo_l + j + 3, GROUP_SIZE), 0))
            # the records' block ids (pooled), in flight with the targets: one round trip per four links
            k0 = _table_blk(table, cell, base + j, N_SLOTS, POOL_B, nb, POOL_RESERVE)
            k1 = _table_blk(table, cell, base + j + 1, N_SLOTS, POOL_B, nb, POOL_RESERVE)
            k2 = _table_blk(table, cell, base + j + 2, N_SLOTS, POOL_B, nb, POOL_RESERVE)
            k3 = _table_blk(table, cell, base + j + 3, N_SLOTS, POOL_B, nb, POOL_RESERVE)
            if TRACE >= 3:
                x3 = pin(x3)
                n_ev = mark(trace, n_ev, 17, 0, TRACE_CAP)
            _link_record(rec0, x0, k0, j, w, n_l, lo_l, base, lim, cell, n_cells, e2, srcw2, log_rec,
                         ovf_cap, T, NUM_WARPS, GROUP_SIZE, N_SLOTS, POOL_B, nb)
            _link_record(rec0, x1, k1, j + 1, w, n_l, lo_l, base, lim, cell, n_cells, e2, srcw2, log_rec,
                         ovf_cap, T, NUM_WARPS, GROUP_SIZE, N_SLOTS, POOL_B, nb)
            _link_record(rec0, x2, k2, j + 2, w, n_l, lo_l, base, lim, cell, n_cells, e2, srcw2, log_rec,
                         ovf_cap, T, NUM_WARPS, GROUP_SIZE, N_SLOTS, POOL_B, nb)
            _link_record(rec0, x3, k3, j + 3, w, n_l, lo_l, base, lim, cell, n_cells, e2, srcw2, log_rec,
                         ovf_cap, T, NUM_WARPS, GROUP_SIZE, N_SLOTS, POOL_B, nb)
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
def _gate_skip(gated, plan, n_conns, ev, run_id, round_id, t_start, slot_width, n_groups, SK: gl.constexpr,
               KP: gl.constexpr, MAX_CONNS: gl.constexpr, NUM_WARPS: gl.constexpr):
    """Whether a gated run (an event-driven population's planned launch: Network.plan) has nothing due at its
    time t_start, as the host would decide it (Population._next_run_time): due is the earliest of (A) every
    fresh source run with spikes -- its earliest emission + its connection's minimum delay (intake's) -- and
    (B) the end of the first slot still holding records after the population's last run (any group's:
    _note_pending_arrivals'). Every program decides alike (from the same headers and event state). An
    ungated run never skips."""
    skip = gated != 0
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
        first = gl.to_tensor(EV_NONE)
        gi = gl.arange(0, 32 * NUM_WARPS, layout=LV)
        for g0 in range(0, n_groups, 32 * NUM_WARPS):
            v = gl.load(evp + EV_HDR + g0 + gi, mask=g0 + gi < n_groups, other=EV_NONE)
            first = gl.minimum(first, gl.min(v, axis=0))
        due = gl.minimum(due, gl.where(first < EV_NONE, (first + 1).to(gl.float32) * slot_width, float("inf")))
        skip = due > t_start + 1e-6 * gl.maximum(gl.abs(t_start), 1.0)
    return skip


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
              NUM_WARPS: gl.constexpr, POOL_B: gl.constexpr, NC: gl.constexpr = 1):
    """A run done: the event state for the next run -- the delivery position past this run's slots, and this
    group's first slot still holding records (after delivery: later per-link arrivals of spikes taken in)."""
    LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    (_mt, _ms, _ml, _mn, main_count, _st, _ss, _sl, _sn, side_count, _wm, _fl, _tb, _stb, _sk, _tp, _nb, _lo,
     _lr, _oc, _mc, _sc, _nbk, _sts) = _ring(ring, POOL_B)
    rp = gl.maximum(k_lo, k_hi + 1)
    sl = gl.arange(0, NSP, layout=LV)
    live = sl < N_SLOTS
    NR: gl.constexpr = N_SLOTS // NC  # (cell sl: class sl // NR's slot rp + sl % NR)
    k = rp + sl % NR
    cell = pid * N_SLOTS + (sl // NR) * NR + k % NR
    c = gl.load(main_count + cell, mask=live, other=0) + gl.load(side_count + cell, mask=live, other=0)
    first = gl.min(gl.where(live & (c > 0), k, EV_NONE), axis=0)
    src = ev + (run_id % 2) * (EV_HDR + n_groups)
    dst = ev + (1 - run_id % 2) * (EV_HDR + n_groups)
    gl.store(dst, rp)
    gl.store(dst + 1, gl.load(src + 1) + 1)
    gl.store(dst + EV_HDR + pid, first)


@gluon.jit
def _run_gate(ring, ctl, pid, run_id):
    """Whether program pid takes part in run run_id: not once a run stopped (ctl, the launches' stop word:
    every run that starts after a stop waits for the host to rerun it), not if its group finished this run
    already (a rerun: the groups that finished before the stop skip). Read before anything is written."""
    done = ring + gl.load(ring + 8)
    return (gl.load(ctl) == 0) & (gl.load(done + pid) < run_id)


@gluon.jit
def _run_done(ring, pid, run_id):
    """Program pid's group finished run run_id (in the ring: what a rerun skips)."""
    gl.store(ring + gl.load(ring + 8) + pid, run_id)


@gluon.jit
def _run_status(status, ring, pid, run_id, finished, chk, ARENA_CHECK: gl.constexpr):
    """Program pid's last stores: its group's part of the run's status, straight into the run's pinned host
    slot (PopulationRuntime: no copy, no event) -- the blocks it holds and the most it logged (the pull's
    group stats), its ring's lost bits (main 1, side 2: the general pull), and its done word: 2 run_id if it
    finished the run, 2 run_id + 1 if it stopped (or skipped: a run stopped before it) -- the host reads the
    whole run from two reductions (lost bits: + 4 if the arena check found a violation, ARENA_CHECK):
    landed once the least done word is >= 2 run_id (a slot's earlier runs
    wrote less), stopped if the largest is odd. Plain stores, done last: the host decides from the done
    words alone whether to rerun (and then reads the ring itself); the rest only sizes."""
    st = ring + gl.load(ring + 7) + 2 * pid
    fl = ring + gl.load(ring + 5)
    at = status + 4 * pid
    gl.store(at, gl.load(st))
    gl.store(at + 1, gl.load(st + 1))
    bits = gl.load(fl) | (gl.load(fl + 3) << 1)
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
def _stage_records(rec0, cell, b0, nm, n_cells, s_atp, f_tgt, f_w, f_src, f_e, table, LW: gl.constexpr,
                   STREAM: gl.constexpr, U: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
                   LOG: gl.constexpr, STAGE_E: gl.constexpr, N_SLOTS: gl.constexpr, main_cap, POOL_B: gl.constexpr,
                   nb, STAGE_W: gl.constexpr):
    """Links b0 .. b0 + STREAM of a slot's main run staged in LDS for the walk, each into its target warp's
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
    rec, ok = _run_rec(cell, q, n_cells, table, N_SLOTS, POOL_B, nb, POOL_RESERVE)
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


@gluon.jit
def _merge_links(main_time, main_src, main_lo, main_n, side_time, side_src, side_lo, side_n, cell, nm, ns, n_cells,
                 attr_addr, state, received, n_received, lane, mask, k, pid, log_ptr, log_e, log_cap, table, side_table,
                 RECEIVE: gl.constexpr, GROUP_SIZE: gl.constexpr, LOG: gl.constexpr, COUNT: gl.constexpr,
                 N_SLOTS: gl.constexpr, main_cap, POOL_B: gl.constexpr, nb, tk, T_A: gl.constexpr, ectx):
    """A slot with side entries: its main and side runs of link records
    merged by emission time with two cursors (main first on ties), each
    link received in order."""
    # GAP (forward): as in _stage_records and _walk, one link at a
    #   time (forward strictly sequential here: no D-8 limitation); the main/side merge by the records' time
    #   word, the emission time (receive's time computed from it, D-1)
    im = gl.to_tensor(0)
    js = gl.to_tensor(0)
    for _ in range(nm + ns):
        rm, okm = _run_rec(cell, im, n_cells, table, N_SLOTS, POOL_B, nb, POOL_RESERVE)
        rs, oks = _run_rec(cell, js, n_cells, side_table, N_SLOTS, POOL_B, nb, 0)
        at_m = 4 * rm
        at_s = 4 * rs
        mv = im < nm
        sv = js < ns
        mr = mv & okm  # a record whose block is missing (lost links): taken in turn, delivered to no one
        sr = sv & oks
        em = gl.load(main_time + at_m, mask=mr, other=float("inf"))
        es = gl.load(side_time + at_s, mask=sr, other=float("inf"))
        take = mv & (em <= es)
        e = gl.where(take, em, es)
        srcword = gl.where(take, gl.load(main_src + at_m, mask=mr, other=0),
                           gl.load(side_src + at_s, mask=sr, other=0))
        l = gl.where(take, gl.load(main_lo + at_m, mask=mr, other=0), gl.load(side_lo + at_s, mask=sr, other=0))
        t = gl.where(take, gl.load(main_n + at_m, mask=mr, other=-1), gl.load(side_n + at_s, mask=sr, other=-1))
        im = im + take.to(gl.int32)
        js = js + (1 - take.to(gl.int32))
        w = gl.load(gl.load(attr_addr + (srcword >> 24)).to(gl.pointer_type(gl.float32)) + l)
        state, received, n_received = _deliver_link(
            state, t, w, srcword, e, received, n_received, lane, mask, k, pid, log_ptr, log_e, log_cap,
            RECEIVE, GROUP_SIZE, LOG, COUNT, e * T_A + tk, ectx)
    return state, received, n_received


# ------------------------------------------------- delivery: several delay classes (D-1, BRUNEL.md)
#
# A population whose incoming connections have several homogeneous delays: a ring per delay class (class q's slot
# k in cell q NR + k % NR of its group, NR = N_SLOTS // NC), each class's records in emission order -- its arrival
# order, e + D_q. A due slot's classes are merged by arrival (ties: the lower class first; within a class main
# before side, as _merge_links), every link received at its exact arrival.


@gluon.jit
def _stage_merged(main_time, rec0, cell0, cur, nmv, dcls, cv, s_atp, s_arr, m_tgt, m_w, m_src, m_t, f_tgt, f_w,
                  f_src, f_e, table, n_cells, LW: gl.constexpr, STREAM: gl.constexpr, U: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr, LOG: gl.constexpr,
                  STAGE_E: gl.constexpr, N_SLOTS: gl.constexpr, POOL_B: gl.constexpr, nb, STAGE_W: gl.constexpr,
                  NC: gl.constexpr):
    """The next STREAM links of a due slot's classes in arrival order, staged for the walk as _stage_records
    stages a run's (f_e: each link's arrival). An exact merge a block at a time: every class's next STREAM
    records (from its cursor, cur: [MAX_CONNS] in cv's layout) are the candidates -- the merged next STREAM are
    among them --, each candidate's rank among them its place in its class + a binary search into each other
    class's (their arrivals in LDS, s_arr: sorted, padded with inf); the ranks below STREAM are taken, scattered
    to their place (m_*: [2 STREAM], past STREAM the untaken's dump). Returns (the links taken, each warp's count,
    the most, each class's taken count in cv's layout: the cursors' advance). (A record whose block is missing --
    an arena that ran out, its links lost -- counts as arriving last: the order past it is undefined.)"""
    NR: gl.constexpr = N_SLOTS // NC
    LR: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [NUM_WARPS, 1], [1, 0])
    LRQ: gl.constexpr = gl.SliceLayout(1, LR)
    i = gl.arange(0, STREAM, layout=LRQ)
    col = gl.arange(0, 4, layout=gl.SliceLayout(0, LR))
    # 1. every class's candidates' arrivals into LDS
    for c in gl.static_range(NC):
        cur_c = gl.sum(gl.where(cv == c, cur, 0), axis=0)
        nm_c = gl.sum(gl.where(cv == c, nmv, 0), axis=0)
        d_c = gl.sum(gl.where(cv == c, dcls, 0.0), axis=0)
        q = cur_c + i
        rec, ok = _run_rec(cell0 + c * NR, q, n_cells, table, N_SLOTS, POOL_B, nb, POOL_RESERVE)
        e = gl.load(main_time + 4 * rec, mask=(q < nm_c) & ok, other=float("inf"))
        s_arr.scatter(e + d_c, c * STREAM + i, 0)
    gl.barrier()
    # 2. each candidate's rank; the taken ones to their place
    taken = gl.zeros_like(cur)
    for c in gl.static_range(NC):
        cur_c = gl.sum(gl.where(cv == c, cur, 0), axis=0)
        nm_c = gl.sum(gl.where(cv == c, nmv, 0), axis=0)
        d_c = gl.sum(gl.where(cv == c, dcls, 0.0), axis=0)
        q = cur_c + i
        rec, ok = _run_rec(cell0 + c * NR, q, n_cells, table, N_SLOTS, POOL_B, nb, POOL_RESERVE)
        qv = (q < nm_c) & ok
        at = gl.multiple_of(4 * rec, 4)
        words = gl.load(rec0 + at[:, None] + col[None, :], mask=qv[:, None] & (col[None, :] < 4), other=0)
        el, st = gl.split(gl.reshape(words, [STREAM, 2, 2]))  # (e, link index), (source word, target)
        e_i, l = gl.split(el)
        srcw, t = gl.split(st)
        a = gl.where(qv, gl.convert_layout(e_i, LRQ).to(gl.float32, bitcast=True) + d_c, float("inf"))
        l = gl.convert_layout(l, LRQ)
        srcw = gl.convert_layout(srcw, LRQ)
        t = gl.convert_layout(t, LRQ)
        rank = i
        for c2 in gl.static_range(NC):
            if c2 != c:  # the other class's candidates before this one: arriving earlier (a lower class: or as early)
                pos = gl.zeros_like(i)
                for stp in gl.static_range(12):  # (STREAM <= 4096: the steps past log2(STREAM) compiled out)
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
        if STAGE_W:
            atp = s_atp.gather(srcw >> 24, 0).to(gl.pointer_type(gl.float32))
            w = gl.load(atp + l, mask=take, other=0.0)
        else:
            w = gl.zeros_like(l).to(gl.float32)
        dst = gl.where(take, rank, STREAM + i)
        m_tgt.scatter(t, dst, 0)
        m_w.scatter(w, dst, 0)
        if LOG:
            m_src.scatter(srcw, dst, 0)
        if STAGE_E:
            m_t.scatter(a, dst, 0)
        taken = gl.where(cv == c, gl.sum(take.to(gl.int32), axis=0), taken)
    gl.barrier()
    # 3. the merged block, partitioned by target warp as _stage_records does
    n_take = gl.sum(taken, axis=0)
    qv = i < n_take
    t = m_tgt.gather(i, 0)
    w = m_w.gather(i, 0)
    NPW: gl.constexpr = GROUP_SIZE // NUM_WARPS
    if NUM_WARPS == 1:
        wq = gl.zeros_like(i)
        rank = i
    else:
        wq = gl.where(qv, t // NPW, i % NUM_WARPS)
        wi = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(0, LR))
        oh = (wq[:, None] == wi[None, :]).to(gl.int32)
        rank = gl.sum(oh * (gl.cumsum(oh, axis=0) - 1), axis=1)
    wl = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(0, LR))
    n_w = gl.convert_layout(gl.sum(((wq[:, None] == wl[None, :]) & qv[:, None]).to(gl.int32), axis=0), LW)
    dst = (rank // U) * (NUM_WARPS * U) + wq * U + rank % U
    f_tgt.scatter(gl.where(qv, t, -1), dst, 0)
    f_w.scatter(w, dst, 0)
    if LOG:
        f_src.scatter(m_src.gather(i, 0), dst, 0)
    if STAGE_E:
        f_e.scatter(m_t.gather(i, 0), dst, 0)
    gl.barrier()
    return n_take, n_w, gl.max(n_w, axis=0), taken


@gluon.jit
def _merge_classes(main_time, main_src, main_lo, main_n, side_time, side_src, side_lo, side_n, cell0, nmv, nsv, dcls,
                   cv, n_cells, attr_addr, state, received, n_received, lane, mask, k, pid, log_ptr, log_e, log_cap,
                   table, side_table, RECEIVE: gl.constexpr, GROUP_SIZE: gl.constexpr, LOG: gl.constexpr,
                   COUNT: gl.constexpr, N_SLOTS: gl.constexpr, POOL_B: gl.constexpr, nb, NC: gl.constexpr, ectx):
    """A due slot's classes, some with side entries: every class's main and side runs merged by arrival one
    link at a time (as _merge_links: the order (arrival, class, main first)), each received at its arrival."""
    NR: gl.constexpr = N_SLOTS // NC
    cellv = cell0 + cv * NR
    im = gl.zeros_like(cv)
    js = gl.zeros_like(cv)
    for _ in range(gl.sum(nmv + nsv, axis=0)):
        rm, okm = _run_rec(cellv, im, n_cells, table, N_SLOTS, POOL_B, nb, POOL_RESERVE)
        rs, oks = _run_rec(cellv, js, n_cells, side_table, N_SLOTS, POOL_B, nb, 0)
        mv = im < nmv
        sv = js < nsv
        mr = mv & okm  # a record whose block is missing (lost links): taken in turn, delivered to no one
        sr = sv & oks
        am = gl.load(main_time + 4 * rm, mask=mr, other=float("inf")) + dcls
        as_ = gl.load(side_time + 4 * rs, mask=sr, other=float("inf")) + dcls
        take_m = mv & (~sv | (am <= as_))
        a_c = gl.where(take_m, am, as_)
        has = mv | sv
        a = gl.min(gl.where(has, a_c, float("inf")), axis=0)
        csel = gl.min(gl.where(has & (a_c == a), cv, 1 << 30), axis=0)
        sel = cv == csel
        tm = gl.sum((sel & take_m).to(gl.int32), axis=0) > 0
        at = 4 * gl.sum(gl.where(sel, gl.where(take_m, rm, rs), 0), axis=0)
        okr = gl.sum((sel & gl.where(take_m, mr, sr)).to(gl.int32), axis=0) > 0
        srcword = gl.where(tm, gl.load(main_src + at, mask=okr, other=0), gl.load(side_src + at, mask=okr, other=0))
        l = gl.where(tm, gl.load(main_lo + at, mask=okr, other=0), gl.load(side_lo + at, mask=okr, other=0))
        t = gl.where(tm, gl.load(main_n + at, mask=okr, other=-1), gl.load(side_n + at, mask=okr, other=-1))
        im = im + (sel & take_m).to(gl.int32)
        js = js + (sel & ~take_m).to(gl.int32)
        w = gl.load(gl.load(attr_addr + (srcword >> 24)).to(gl.pointer_type(gl.float32)) + l)
        state, received, n_received = _deliver_link(  # (the log: the arrival, as _stage_merged stages it)
            state, t, w, srcword, a, received, n_received, lane, mask, k, pid, log_ptr, log_e, log_cap,
            RECEIVE, GROUP_SIZE, LOG, COUNT, a, ectx)
    return state, received, n_received


@gluon.jit
def _release_cell(table, side_table, stack, tp, pid, cell, held, ns, nblk, lost_v, LV: gl.constexpr, nb,
                  NUM_WARPS: gl.constexpr, chk, k, ARENA_CHECK: gl.constexpr, POOL_B: gl.constexpr):
    """A delivered cell's blocks back on the group's free stack (pooled: the main run keeps its first
    POOL_RESERVE blocks, the side run none). Returns the stack's new top."""
    keep = gl.minimum(held, POOL_RESERVE)  # (fixed blocks: never missing, never given away)
    tp = _free_blocks(table, stack, tp, pid, cell, keep, held, lost_v, LV, nb, NUM_WARPS, chk, k, ARENA_CHECK)
    if held > keep:
        gl.store(nblk + cell, keep)
    tp = _free_blocks(side_table, stack, tp, pid, cell, 0, (ns + POOL_B - 1) // POOL_B, lost_v, LV, nb, NUM_WARPS,
                      chk, k, ARENA_CHECK)
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
    entry's and the gate's loads, not after them. TABLES (the batched pull): per pair its topology's and
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
    hcap = gl.load(hcap_p + c_of, mask=vj, other=0)
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
    return last0, last1, hbuf, hcap, lastj0, lastj1, tag, cnt, wend, early, topo


@gluon.jit
def _pull(pid, n_groups, plan, n_conns, ring, wmark_ptr, oflags, ctl, k_lo, slot_width, round_id, parity, hdrs,
          trace, n_ev,
          SK: gl.constexpr, KP: gl.constexpr, MAX_CONNS: gl.constexpr, LOG_CJ: gl.constexpr,
          N_SLOTS: gl.constexpr, NSP: gl.constexpr,
          UNIFORM_SOURCES: gl.constexpr, BATCH: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
          TRACE: gl.constexpr, TRACE_CAP: gl.constexpr, POOL_B: gl.constexpr, chk, ARENA_CHECK: gl.constexpr,
          NC: gl.constexpr = 1):
    """Phase 1 of program pid: what the sources emitted since this consumer
    last read them, into the program's group's ring; returns (n_ev
    (tracing), ok). Not ok: the ring is too small for the run -- nothing
    committed (counts, tables, free stack, watermark as they were; only
    scratch past the counts written), its flag 4 and the launches' stop
    word ctl set: the program stops here."""
    # (connection, source history slot) pairs, 2**LOG_CJ; SK: the sources' history depth
    MAX_CJ: gl.constexpr = MAX_CONNS * KP
    SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    # per (connection, slot) / per connection vectors, spikes x (connection, slot) for the spike decode,
    # spikes x segments for the expansion
    LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    LKC: gl.constexpr = gl.BlockedLayout([1, MAX_CJ], [32, 1], [NUM_WARPS, 1], [1, 0])
    # spikes x segments: one spike row per thread row (8 x warps rows), its MSEG segments over 4
    # threads -- per-spike values (three 64-bit pointers among them) held once per thread, not 4x
    LP2: gl.constexpr = gl.BlockedLayout([1, MSEG // 4], [8, 4], [NUM_WARPS, 1], [1, 0])
    LK: gl.constexpr = gl.SliceLayout(1, LP2)
    LM: gl.constexpr = gl.SliceLayout(0, LP2)
    (_hist_addr, topo_addr, link_target_addr, attr_addr, dir_addr, _hcap_p, t_src_p, t_seg_p, t_bkt_p, t_lnk_p,
     dpairs_p, dwords_p, last_read) = _fused_plan(plan, n_conns)
    (main_time, main_src, main_lo, main_n, main_count, side_time, side_src, side_lo, side_n, side_count,
     _watermark, flags, table, side_table, stack, top, nblk, lost, log_rec, ovf_cap, main_cap, side_cap,
     nb, stats) = _ring(ring, POOL_B)
    n_cells = n_groups * N_SLOTS
    NR: gl.constexpr = N_SLOTS // NC  # the ring's slots; N_SLOTS: the cells per group (NC delay classes)
    cls_plan = plan + 21 * n_conns  # per connection its delay class (FusedPlan)

    # No user code in the pull -- it only moves (e, source, link, target) records into the ring;
    #   every hook runs at delivery, in delivery order (strict per-link order)
    ci = gl.arange(0, MAX_CONNS, layout=LV)
    cv = ci < n_conns
    # per (connection, slot): its run's header -- one chain (plan -> history -> header) for everything below,
    # loaded first thing in the program (_pull_headers); the run's parity picks its last-read ids
    last0, last1, hbuf, hcap, lastj0, lastj1, tag, cnt, wend, early, topo = hdrs
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
    # the fresh runs, taken in in order of their (earliest) emission time -- under uniform sources a run's
    # spikes share one e but runs differ, and main runs must grow in e
    fresh = newer & (cnt > 0)
    n_spk = gl.where(fresh, gl.minimum(cnt, hcap), 0)
    key = gl.where(fresh, early, float("inf"))
    n_runs = gl.sum(fresh.to(gl.int32), axis=0)
    node_addr = hbuf + 4 * (8 * SK + j_of * hcap).to(gl.int64)
    time_addr = node_addr + 4 * (SK * hcap).to(gl.int64)
    wmark = gl.load(wmark_ptr + pid)
    if UNIFORM_SOURCES and BATCH:
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
        if NC > 1:
            tab_cls = _table(gl.load(cls_plan + c_of, mask=vj, other=0), SM)
    if UNIFORM_SOURCES:
        # this group's per-slot link counts, in LDS for the pull's appends (it is their only writer)
        sl = gl.arange(0, NSP, layout=LV)
        s_cnt = gl.allocate_shared_memory(gl.int32, [NSP], SM)
        cnt0 = gl.load(main_count + pid * N_SLOTS + sl, mask=sl < N_SLOTS, other=0)
        if POOL_B > 0:
            # pooled: count | blocks held << POOL_CNT_BITS, so every claim learns both (_append_links); the free
            # stack top (the counts at the start, cnt0: _pool_fixup's headroom, this run's growth)
            top0 = gl.load(top + pid)
            s_cnt.store(cnt0 | (gl.load(nblk + pid * N_SLOTS + sl, mask=sl < N_SLOTS, other=0) << POOL_CNT_BITS))
        else:
            s_cnt.store(cnt0)
            top0 = gl.to_tensor(0)
        gl.barrier()
    violated = gl.to_tensor(0)
    total = gl.to_tensor(0)
    BLK: gl.constexpr = 32 * NUM_WARPS
    if TRACE >= 2:
        n_runs = pin_s(n_runs)
        n_ev = mark(trace, n_ev, 10, n_runs, TRACE_CAP)
    if UNIFORM_SOURCES and BATCH:
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
                main = tv & (e >= wmark) & (e < wend_t)
                has = b > a
                violated = gl.maximum(violated, gl.max((tv & has & ~main).to(gl.int32), axis=0))
                e_max = gl.maximum(e_max, gl.max(gl.where(main & has, e, float("-inf")), axis=0))
                if NC > 1:
                    cls_t = tab_cls.gather(cj_t, 0)
                else:
                    cls_t = 0
                n_ev = _append_links(s, e, main, c_t, a, b, seg_start, seg_gb, bucket_mean,
                                     ltp_t, s_cnt, pid, k_lo, slot_width, n_cells,
                                     main_time, main_src, main_lo, main_n,
                                     flags, trace, n_ev, table, log_rec, ovf_cap, N_SLOTS, main_cap,
                                     LP2, LM, SPK, MSEG, TRACE, TRACE_CAP, NUM_WARPS, GROUP_SIZE, POOL_B,
                                     nb, cls_t, NC)
                if TRACE >= 2:
                    n_ev = mark(trace, n_ev, 13, gl.sum(gl.where(main, b - a, 0), axis=0), TRACE_CAP)
            wmark = gl.maximum(wmark, e_max)  # later e-groups are compared against this one
            gl.barrier()  # the group's appends are in before the next group's (main runs grow in e)
            e0 = gl.min(gl.where(key > e0, key, float("inf")), axis=0)
    else:
        # the fresh runs one by one, by (e, pair): each pair's rank
        key_r = gl.convert_layout(key, gl.SliceLayout(0, LKC))
        cj_r = gl.arange(0, MAX_CJ, layout=gl.SliceLayout(0, LKC))
        key_l = gl.convert_layout(key, gl.SliceLayout(1, LKC))
        cj_l = gl.arange(0, MAX_CJ, layout=gl.SliceLayout(1, LKC))
        before = (key_r[None, :] < key_l[:, None]) | ((key_r[None, :] == key_l[:, None])
                                                      & (cj_r[None, :] < cj_l[:, None]))
        order = gl.convert_layout(gl.sum(before.to(gl.int32), axis=1), LV)  # rank by (e, cj); fresh runs first
        # per connection, its topology (picked per run below)
        tbuf_c = gl.load(topo_addr + ci, mask=cv, other=0)
        nsrc_c = gl.load(t_src_p + ci, mask=cv, other=0)
        nseg_c = gl.load(t_seg_p + ci, mask=cv, other=0)
        dbuf_c = gl.load(dir_addr + ci, mask=cv, other=0)
        dpairs_c = gl.load(dpairs_p + ci, mask=cv, other=0)
        dwords_c = gl.load(dwords_p + ci, mask=cv, other=0)
        ltp_c = gl.load(link_target_addr + ci, mask=cv, other=0)
        for ri in range(0, n_runs):
            # the run: its spikes, its connection's topology (scalars)
            sel = order == ri
            n_r = gl.sum(gl.where(sel, n_spk, 0), axis=0)
            naddr = gl.sum(gl.where(sel, node_addr, 0), axis=0)
            taddr = gl.sum(gl.where(sel, time_addr, 0), axis=0)
            wend_r = gl.sum(gl.where(sel, wend, 0.0), axis=0)
            c = gl.sum(gl.where(sel, c_of, 0), axis=0)
            selc = ci == c
            if NC > 1:
                cls_r = gl.load(cls_plan + c)
            else:
                cls_r = 0
            nsrc_r = gl.sum(gl.where(selc, nsrc_c, 0), axis=0)
            _seg_offsets, seg_start, seg_gb, bucket_mean = topology_ptrs(
                gl.sum(gl.where(selc, tbuf_c, 0), axis=0).to(gl.pointer_type(gl.int32)), nsrc_r,
                gl.sum(gl.where(selc, nseg_c, 0), axis=0))
            dir_offs, dir_first, dir_words = directory_ptrs(
                gl.sum(gl.where(selc, dbuf_c, 0), axis=0).to(gl.pointer_type(gl.int32)), nsrc_r,
                gl.sum(gl.where(selc, dpairs_c, 0), axis=0))
            dwords_r = gl.sum(gl.where(selc, dwords_c, 0), axis=0)
            ltp_r = _target_ptr(gl.sum(gl.where(selc, ltp_c, 0), axis=0), GROUP_SIZE)
            node_p = naddr.to(gl.pointer_type(gl.int32))
            time_p = taddr.to(gl.pointer_type(gl.float32))
            total += n_r
            if UNIFORM_SOURCES:
                e_max = gl.to_tensor(float("-inf"))
                for t0 in range(0, n_r, SPK):
                    # a block of the run's spikes: each one's segments into this group
                    i = t0 + gl.arange(0, SPK, layout=LK)
                    tv = i < n_r
                    s = gl.load(node_p + i, mask=tv, other=0)
                    e = gl.load(time_p + i, mask=tv, other=0.0)
                    a, b = _segments_of(s, tv, pid, dir_offs, dir_first, dir_words, dwords_r)
                    main = tv & (e >= wmark) & (e < wend_r)
                    has = b > a
                    violated = gl.maximum(violated, gl.max((tv & has & ~main).to(gl.int32), axis=0))
                    e_max = gl.maximum(e_max, gl.max(gl.where(main & has, e, float("-inf")), axis=0))
                    n_ev = _append_links(s, e, main, c, a, b, seg_start, seg_gb, bucket_mean, ltp_r, s_cnt, pid,
                                         k_lo, slot_width, n_cells, main_time, main_src, main_lo, main_n, flags,
                                         trace, n_ev, table, log_rec, ovf_cap, N_SLOTS, main_cap, LP2,
                                         LM, SPK, MSEG, TRACE, TRACE_CAP, NUM_WARPS, GROUP_SIZE, POOL_B, nb,
                                         cls_r, NC)
                wmark = gl.maximum(wmark, e_max)  # the next run (later e) is compared against this one
                gl.barrier()  # the run's appends are in before the next run's (main runs grow in e)
            else:
                # any emission times: sequential per entry (the baseline)
                for i in range(0, n_r):
                    s = gl.load(node_p + i)
                    e = gl.load(time_p + i)
                    a, b = _segments_of(s, True, pid, dir_offs, dir_first, dir_words, dwords_r)
                    srcw = (c << 24) | s
                    for sg in range(a, b):
                        gb = gl.load(seg_gb + sg)
                        lo_l = gl.load(seg_start + sg)
                        n_l = gl.load(seg_start + sg + 1) - lo_l
                        mean = gl.load(bucket_mean + (gb & 0xFFFF))
                        k = gl.maximum(gl.floor((e + mean) / slot_width).to(gl.int32), k_lo)
                        if k - k_lo >= NR:
                            gl.atomic_max(flags + 1, 1)
                        else:
                            cell = pid * N_SLOTS + cls_r * NR + k % NR
                            to_main = (e >= wmark) & (e < wend_r)
                            for jl in range(0, n_l):  # one record per link, in link order
                                l = lo_l + jl
                                tgt = gl.load(ltp_r + l).to(gl.int32) & 0xFFFF
                                if to_main:
                                    pos = gl.load(main_count + cell)
                                    if pos < main_cap:
                                        if POOL_B > 0:  # a new block where the cell doesn't hold one yet
                                            if pos // POOL_B >= gl.load(nblk + cell):
                                                got_b = _take_block(table, stack, top, pid, cell, pos // POOL_B,
                                                                    nb)
                                                gl.atomic_max(flags, 1, mask=~got_b)  # the arena ran out: links lost
                                                gl.atomic_max(lost, 1, mask=~got_b)
                                                gl.store(nblk + cell, pos // POOL_B + 1)
                                        rec, ok = _run_rec(cell, pos, n_cells, table, N_SLOTS, POOL_B, nb,
                                                           POOL_RESERVE)
                                        at = 4 * rec
                                        gl.store(main_time + at, e, mask=ok)
                                        gl.store(main_src + at, srcw, mask=ok)
                                        gl.store(main_lo + at, l, mask=ok)
                                        gl.store(main_n + at, tgt, mask=ok)
                                        gl.store(main_count + cell, pos + 1)
                                    else:
                                        gl.atomic_max(flags, 1)
                                    gl.barrier()
                                else:
                                    _side_insert(side_time, side_src, side_lo, side_n, side_count, flags, cell,
                                                 n_cells, e, srcw, l, tgt, side_table, stack, top, lost, pid, side_cap,
                                                 LV,
                                                 BLK, N_SLOTS, POOL_B, nb)
                            wmark = gl.where(to_main, gl.maximum(wmark, e), wmark)
    ok = gl.to_tensor(1) > 0
    if UNIFORM_SOURCES:
        if violated > 0:
            gl.atomic_max(flags + 2, 1)
        gl.barrier()
        if TRACE >= 3:
            n_ev = mark(trace, n_ev, 19, 0, TRACE_CAP)
        if POOL_B > 0:
            fits, top0, used, most = _pool_fixup(s_cnt, cnt0, main_src - 1, table, stack, log_rec, ovf_cap, top0,
                                                 pid, sl, LV, N_SLOTS, main_cap, POOL_B, nb, NUM_WARPS, NSP, chk,
                                                 round_id, ARENA_CHECK)
            if TRACE >= 3:
                n_ev = mark(trace, n_ev, 20, pin_s(most), TRACE_CAP)
            gl.store(stats + 2 * pid, used)  # the run's status: the blocks the group holds, the most logged
            gl.store(stats + 2 * pid + 1, most)
            if fits:
                packed = s_cnt.load(LV)  # the counts and block counts back to the ring
                gl.store(main_count + pid * N_SLOTS + sl, packed & ((1 << POOL_CNT_BITS) - 1), mask=sl < N_SLOTS)
                gl.store(nblk + pid * N_SLOTS + sl, packed >> POOL_CNT_BITS, mask=sl < N_SLOTS)
                gl.store(top + pid, top0)
            if TRACE >= 3:
                n_ev = mark(trace, n_ev, 21, 0, TRACE_CAP)
            ok = fits
        else:
            cnt = s_cnt.load(LV)
            longest = gl.max(gl.where(sl < N_SLOTS, cnt, 0), axis=0)
            gl.store(stats + 2 * pid, longest)  # the run's status: the longest run
            gl.store(stats + 2 * pid + 1, 0)
            ok = longest <= main_cap
            if ok:
                gl.store(main_count + pid * N_SLOTS + sl, cnt, mask=sl < N_SLOTS)  # the counts back
    if not UNIFORM_SOURCES:
        gl.barrier()  # the run's status (the general pull takes blocks one at a time: nothing logged)
        if POOL_B > 0:
            gl.store(stats + 2 * pid, nb - gl.load(top + pid))
        else:
            gl.store(stats + 2 * pid, 0)
        gl.store(stats + 2 * pid + 1, 0)
    if ok:
        gl.store(wmark_ptr + pid, wmark)
    else:
        # a ring too small for this run: nothing committed; the run stops here, and with it every run that
        # starts after (_run_gate) -- the host resizes and runs them again (PopulationRuntime, _Launches)
        gl.atomic_max(flags + 4, 1)
        gl.atomic_max(ctl, 1)
    gl.barrier()  # this program's ring entries are complete before it delivers
    if TRACE:
        n_ev = mark(trace, n_ev, 2, total, TRACE_CAP)
    return n_ev, ok


# ------------------------------------------------- phase 2: delivery


@gluon.jit
def _deliver(state, pid, n_groups, plan, n_conns, ring, k_lo, k_hi, offs, lane, mask, received_ptr, n_received_ptr,
             log_ptr, log_cap, trace, n_ev, ectx, RECEIVE: gl.constexpr, GROUP_SIZE: gl.constexpr,
             N_SLOTS: gl.constexpr, STREAM: gl.constexpr, U: gl.constexpr,
             SPT: gl.constexpr, NUM_WARPS: gl.constexpr, MAX_CONNS: gl.constexpr, LOG: gl.constexpr,
             COUNT: gl.constexpr, STAGE_E: gl.constexpr, TRACE: gl.constexpr, TRACE_CAP: gl.constexpr,
             POOL_B: gl.constexpr, chk, ARENA_CHECK: gl.constexpr, STAGE_W: gl.constexpr = True,
             T_A: gl.constexpr = 1.0, T_C: gl.constexpr = 0.0, T_B: gl.constexpr = 0.0, NC: gl.constexpr = 1):
    """Phase 2 of program pid: its group's due slots k_lo..k_hi, every link
    through RECEIVE (see the module docstring); state: the tuple of node
    attrs receive touches, [GROUP_SIZE] tensors in the program's neuron
    layout (lane's). STAGE_E: the walk stages each record's time word too
    (LOG, or a receive that reads time). ectx: the run's history slot for spikes inside delivery (_emit_ctx;
    () without any). NC > 1: several delay classes, each
    slot's classes merged by arrival (_stage_merged, _merge_classes; the log's time word then the arrival,
    not the emission). Returns (state, n_ev)."""
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
    (main_time, main_src, main_lo, main_n, main_count, side_time, side_src, side_lo, side_n, side_count,
     _watermark, _flags, table, side_table, stack, top, nblk, lost, _log_rec, _ovf_cap, main_cap, side_cap,
     nb, _stats) = _ring(ring, POOL_B)
    n_cells = n_groups * N_SLOTS
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
    if NC > 1:  # the class merge's buffers (_stage_merged), the classes' delays (FusedPlan)
        NR: gl.constexpr = N_SLOTS // NC
        # (each class's candidates' arrivals; MAX_CONNS: a power of two, at least NC)
        s_arr = gl.allocate_shared_memory(gl.float32, [MAX_CONNS * STREAM], SM)
        m_tgt = gl.allocate_shared_memory(gl.int32, [2 * STREAM], SM)
        m_w = gl.allocate_shared_memory(gl.float32, [2 * STREAM], SM)
        m_src = gl.allocate_shared_memory(gl.int32, [2 * STREAM], SM)
        m_t = gl.allocate_shared_memory(gl.float32, [2 * STREAM], SM)
        cv = gl.arange(0, MAX_CONNS, layout=LV)  # (NC <= the connections)
        dcls = gl.load((plan + 22 * n_conns + cv).to(gl.pointer_type(gl.float32)), mask=cv < NC, other=0.0)
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
    # receive's time (D-1): one class, T_A e + tk, tk the slot's term (see _deliver_link) -- a slot's records in
    #   emission order, so per target non-decreasing either way; several classes (NC > 1), e + D_q, the slot's
    #   classes merged by arrival (_stage_merged, _merge_classes)
    if POOL_B > 0:  # the group's free-stack top through the delivery, stored once at its end
        tp = gl.load(top + pid)
        lost_v = gl.load(lost)
    else:
        tp = gl.to_tensor(0)
        lost_v = gl.to_tensor(0)
    for k in range(k_lo, k_hi + 1):
        if NC == 1:
            cell = pid * N_SLOTS + k % N_SLOTS
            tk = k.to(gl.float32) * T_C + T_B  # (the delivery time's per-slot term: _deliver_link)
            nm = gl.minimum(gl.load(main_count + cell), main_cap)
            ns = gl.minimum(gl.load(side_count + cell), side_cap)
            if POOL_B > 0:
                held = gl.load(nblk + cell)  # (for freeing below: read with the counts, its latency hidden)
            else:
                held = gl.to_tensor(0)
            if TRACE:
                nm = pin_s(nm)
                n_ev = mark(trace, n_ev, 3, nm, TRACE_CAP)
            if ns == 0:
                # the slot's link records are its stream: STREAM links at a time
                for b0 in range(0, nm, STREAM):
                    nq, n_w, n_max = _stage_records(main_src - 1, cell, b0, nm, n_cells, s_atp, f_tgt, f_w, f_src, f_e,
                                                    table, LW, STREAM, U, NUM_WARPS, GROUP_SIZE, LOG, STAGE_E, N_SLOTS,
                                                    main_cap, POOL_B, nb, STAGE_W)
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
            else:  # side entries: merged one link at a time in the program's layout (rare: back and forth)
                st1, r1, n1 = _merge_links(
                    main_time, main_src, main_lo, main_n, side_time, side_src, side_lo, side_n, cell, nm, ns, n_cells,
                    attr_addr, _state_from_rows(state, GROUP_SIZE, LN), _from_rows(received, GROUP_SIZE, LN),
                    _from_rows(n_received, GROUP_SIZE, LN), lane, mask, k, pid, log_ptr, log_e, log_cap, table,
                    side_table, RECEIVE, GROUP_SIZE, LOG, COUNT, N_SLOTS, main_cap, POOL_B, nb, tk, T_A, ectx)
                state = _state_to_rows(st1, NUM_WARPS, GROUP_SIZE, LWN)
                received = _to_rows(r1, NUM_WARPS, GROUP_SIZE, LWN)
                n_received = _to_rows(n1, NUM_WARPS, GROUP_SIZE, LWN)
            if POOL_B > 0:  # the delivered cell's blocks back on the group's free stack
                tp = _release_cell(table, side_table, stack, tp, pid, cell, held, ns, nblk, lost_v, LV, nb, NUM_WARPS,
                                   chk, k, ARENA_CHECK, POOL_B)
                if TRACE >= 3:
                    n_ev = mark(trace, n_ev, 23, pin_s(tp), TRACE_CAP)
            gl.store(main_count + cell, 0)
            gl.store(side_count + cell, 0)
        else:  # several delay classes: the slot's cells merged by arrival
            cell0 = pid * N_SLOTS + k % NR
            cellv = cell0 + cv * NR
            cvm = cv < NC
            nmv = gl.minimum(gl.load(main_count + cellv, mask=cvm, other=0), main_cap)
            nsv = gl.minimum(gl.load(side_count + cellv, mask=cvm, other=0), side_cap)
            if POOL_B > 0:
                heldv = gl.load(nblk + cellv, mask=cvm, other=0)
            else:
                heldv = gl.zeros_like(cv)
            if TRACE:
                n_ev = mark(trace, n_ev, 3, pin_s(gl.sum(nmv, axis=0)), TRACE_CAP)
            if gl.sum(nsv, axis=0) == 0:
                cur = gl.zeros_like(cv)
                for b0 in range(0, gl.sum(nmv, axis=0), STREAM):
                    nq, n_w, n_max, taken = _stage_merged(
                        main_time, main_src - 1, cell0, cur, nmv, dcls, cv, s_atp, s_arr, m_tgt, m_w, m_src, m_t,
                        f_tgt, f_w, f_src, f_e, table, n_cells, LW, STREAM, U, NUM_WARPS, GROUP_SIZE,
                        LOG, STAGE_E, N_SLOTS, POOL_B, nb, STAGE_W, NC)
                    cur = cur + taken
                    state, received, n_received = _walk(
                        r_tgt, r_w, r_src, r_e, n_w, n_max, state, received, n_received, lane2, mask2, k, pid,
                        log_ptr, log_e, log_cap, LW, RECEIVE, GROUP_SIZE, U, NUM_WARPS, LOG, COUNT, STAGE_E, 0.0,
                        1.0, ectx, node2)  # (f_e: the arrival itself)
                    gl.barrier()
            else:
                st1, r1, n1 = _merge_classes(
                    main_time, main_src, main_lo, main_n, side_time, side_src, side_lo, side_n, cell0, nmv, nsv,
                    dcls, cv, n_cells, attr_addr, _state_from_rows(state, GROUP_SIZE, LN),
                    _from_rows(received, GROUP_SIZE, LN), _from_rows(n_received, GROUP_SIZE, LN), lane, mask, k,
                    pid, log_ptr, log_e, log_cap, table, side_table, RECEIVE, GROUP_SIZE, LOG, COUNT, N_SLOTS,
                    POOL_B, nb, NC, ectx)
                state = _state_to_rows(st1, NUM_WARPS, GROUP_SIZE, LWN)
                received = _to_rows(r1, NUM_WARPS, GROUP_SIZE, LWN)
                n_received = _to_rows(n1, NUM_WARPS, GROUP_SIZE, LWN)
            for c in gl.static_range(NC):
                cell = cell0 + c * NR
                if POOL_B > 0:
                    tp = _release_cell(table, side_table, stack, tp, pid, cell,
                                       gl.sum(gl.where(cv == c, heldv, 0), axis=0),
                                       gl.sum(gl.where(cv == c, nsv, 0), axis=0), nblk, lost_v, LV, nb, NUM_WARPS,
                                       chk, k, ARENA_CHECK, POOL_B)
                gl.store(main_count + cell, 0)
                gl.store(side_count + cell, 0)
    if POOL_B > 0:
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
def _emit_spikes(fires, e, offs, history, hist_cap, run_id, K: gl.constexpr, SPT: gl.constexpr,
                 NUM_WARPS: gl.constexpr):
    """One n2.spike() site's spikes into the history: fires (which lanes
    spiked) and e (their emission times), [GROUP_SIZE] in the program's
    neuron layout; their place in the run's flat list from one count
    atomic. A run can call this any number of times (D-3: several sites, a
    neuron spiking more than once)."""
    LS: gl.constexpr = gl.BlockedLayout([SPT], [32], [NUM_WARPS], [0])
    _n_fire = fires.to(gl.int32)
    _local_pos = gl.convert_layout(gl.cumsum(gl.convert_layout(_n_fire, LS), axis=0), fires.type.layout) - _n_fire
    n_emit = gl.sum(_n_fire, axis=0)
    # GAP (D-3): the history's cap is one spike per neuron and run (EmissionHistory's default); a run's spikes
    #   beyond it set the overflow flag (the runtime raises: PopulationRuntime.check)
    slot = run_id % K
    (h_tag, h_count, h_wend, h_min, h_node, h_time, h_overflow) = history_ptrs(history, slot, K, hist_cap)
    base = gl.atomic_add(h_count, n_emit)  # this site's place in the run's flat list
    put = fires & (base + _local_pos < hist_cap)
    if base + n_emit > hist_cap:
        gl.atomic_max(h_overflow, 1)
    gl.store(h_node + base + _local_pos, offs, mask=put)
    gl.store(h_time + base + _local_pos, e, mask=put)
    gl.atomic_min(h_min, gl.min(gl.where(put, e, float("inf")), axis=0))


@gluon.jit
def _emit_ctx(history, hist_cap, run_id, K: gl.constexpr):
    """The run's history slot, for spikes inside delivery (_emit_delivered): (count, earliest, nodes, times,
    overflow, cap)."""
    (_h_tag, h_count, _h_wend, h_min, h_node, h_time, h_overflow) = history_ptrs(history, run_id % K, K, hist_cap)
    return (h_count, h_min, h_node, h_time, h_overflow, hist_cap)


@gluon.jit
def _emit_delivered(fire, t, nodes, ectx):
    """An n2.spike() site inside delivery (D-3: in receive, or a send after every delivery): the lanes in fire
    spike at t, straight into the run's history (ectx: _emit_ctx) -- the history is a flat list per run in no
    order, and a run's slot is read only after the run. One link a step: at most one lane fires per row of
    fire (the walk: a row per warp, each its own target lane; a merge: the one lane), so per row one masked
    atomic, none where nothing fires. nodes: the lanes' neurons; t: the spike time (broadcast to fire)."""
    h_count, h_min, h_node, h_time, h_overflow, hist_cap = ectx
    AX: gl.constexpr = len(fire.shape) - 1
    tt = t + gl.zeros_like(fire.to(gl.float32))
    n = gl.sum(fire.to(gl.int32), axis=AX)
    node = gl.sum(gl.where(fire, nodes, 0), axis=AX)
    tm = gl.sum(gl.where(fire, tt, 0.0), axis=AX)
    m = n > 0
    zero = n * 0  # (n: a row per warp, or a scalar -- zeros_like takes no scalar)
    pos = gl.atomic_add(h_count + zero, 1, mask=m)
    put = m & (pos < hist_cap)
    gl.store(h_node + pos, node, mask=put)
    gl.store(h_time + pos, tm, mask=put)
    gl.atomic_min(h_min + zero, tm, mask=put)
    gl.atomic_max(h_overflow + zero, 1, mask=m & (pos >= hist_cap))


@gluon.jit
def _emit_close(pid, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K: gl.constexpr,
                TRACE: gl.constexpr, TRACE_CAP: gl.constexpr):
    """The run's end: program 0 writes its header (window end, then the tag
    -- one store, never a new run id with an old round) and clears the next
    run's slot. Returns n_ev (tracing)."""
    if TRACE:
        n_ev = mark(trace, n_ev, 8, 0, TRACE_CAP)
    # GAP (D-11): edge backward out of scope -- needs this run's spikes at every incoming link of each firing
    #   target (a reverse topology), D-11
    if pid == 0:
        (h_tag, _hc, h_wend, _hm, _hn, _ht, _ho) = history_ptrs(history, run_id % K, K, hist_cap)
        gl.store(h_wend, t_end)
        gl.store(h_tag, history_tag(run_id, round_id))  # one store: never a new run id with an old round
        (n_tag, n_count, _nw, n_min, _nn, _nt, _no) = history_ptrs(history, (run_id + 1) % K, K, hist_cap)
        gl.store(n_tag, history_tag(0, round_id))  # the next run's slot, cleared ahead (see history.py): a
        gl.store(n_count, 0)  # consumer that still needed it sees a gap in run ids and raises its overrun flag
        gl.store(n_min, float("inf"))
    if TRACE:
        n_ev = mark(trace, n_ev, 9, 0, TRACE_CAP)
        trace_end(trace, n_ev)
    return n_ev


@gluon.jit
def _emit(fires, e, offs, pid, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K: gl.constexpr,
          SPT: gl.constexpr, NUM_WARPS: gl.constexpr, TRACE: gl.constexpr, TRACE_CAP: gl.constexpr):
    """A run with one n2.spike() site: _emit_spikes, then _emit_close."""
    _emit_spikes(fires, e, offs, history, hist_cap, run_id, K, SPT, NUM_WARPS)
    return _emit_close(pid, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K, TRACE, TRACE_CAP)
