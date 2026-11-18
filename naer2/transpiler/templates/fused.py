"""
The fused kernel's template: one launch per population and run -- intake,
delivery, update, send and emission in one Gluon kernel, one program per
group. This module is the FIXED part, written by hand and optimized (what
limits optimizing it further: limitations.md, next to this file); the hooks
come in from outside: a kernel entry (generated from a population's hooks,
naer2/transpiler/templates/fused_template.py, or written by hand --
test/fused_reference.py, the reference) calls

  _pull(...)                  phase 1, no user code at all
  _deliver(state, RECEIVE)    phase 2, RECEIVE(state, hit, time, w, srcword)
                              -> state a constexpr gluon.jit function, state
                              the tuple of node attrs receive touches
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
                    "a:segments", "a:mean", "a:claimed", "a:targets", "a:stored")  # TRACE >= 3:
                    # inside the append, per tile
PULL = dict(batch=True)  # PopulationRuntime default (naer2/runtime.py); tools/fused_sweep.py: batching ~30%
# (PD14, G=256, 780M); the group directory always on (as fast again over binary search)
# delivery walk: links staged in LDS per stream block, links per broadcast LDS read (tools/walk_sweep.py on
# the split kernels, PD14, G=256, 780M: larger U costs more in registers than it saves; an explicit
# occupancy target (waves_per_eu) gains <= 3% and spills beyond 12 waves)
WALK = dict(stream=256, unroll=8)
SPK = gl.constexpr(32)   # spikes per pull block (uniform sources)
MSEG = gl.constexpr(4)   # segments per pull column block (uniform sources): a spike reaches a group with ~2-3 (PD14)
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
    last-read run ids (double-buffered by the consumer's run parity)."""

    def __init__(self, histories: list[EmissionHistory], topologies: list[SimulationTopology], n_groups: int, device):
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
        plan = torch.zeros(20 * n, dtype=torch.int32)
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


@gluon.jit
def _pool_fixup(s_cnt, cnt0, rec0, table, stack, log_rec, ovf_cap, top0, g, sl, LV: gl.constexpr,
                N_SLOTS: gl.constexpr, main_cap, POOL_B: gl.constexpr, nb,
                NUM_WARPS: gl.constexpr):
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
            s_cnt.store((packed & MASK) | ((held + want) << POOL_CNT_BITS))
            gl.barrier()
            pq = gl.arange(0, 32 * NUM_WARPS, layout=LV)
            for s_i in range(0, N_SLOTS):  # every slot's log into its new blocks
                h = gl.sum(gl.where(sl == s_i, held, 0), axis=0) * POOL_B
                c = gl.sum(gl.where(sl == s_i, cnt, 0), axis=0)
                cs = g * N_SLOTS + s_i
                for p0 in range(h, c, 32 * NUM_WARPS):
                    p = p0 + pq
                    m = p < c
                    blk = gl.load(table + cs * nb + gl.minimum(p // POOL_B, nb - 1), mask=m, other=0)
                    dst = 4 * ((g * nb + blk) * POOL_B + p % POOL_B)
                    src = 4 * (log_rec + cs * ovf_cap + (p - h))
                    for f in gl.static_range(4):
                        gl.store(rec0 + dst + f, gl.load(rec0 + src + f, mask=m), mask=m)
            gl.barrier()
            new_top = top0 - gl.sum(want, axis=0)
    used = gl.where(fits, nb - new_top, nb - top0 + total)
    return fits, new_top, used, most


@gluon.jit
def _free_blocks(table, stack, tp, g, cell, lo, hi, lost, LV: gl.constexpr, nb,
                 NUM_WARPS: gl.constexpr):
    """A delivered run's blocks lo .. hi back on group g's free stack, above tp (delivery keeps the stack's
    top in a register and stores it once at its end: no barriers here). Once an arena ran out (lost: tables
    may hold missing blocks, -1), they are compacted on the way (a scan): the stack never holds a -1. Returns
    the new top."""
    bi = gl.arange(0, 32 * NUM_WARPS, layout=LV)
    if lost == 0:
        for i0 in range(lo, hi, 32 * NUM_WARPS):
            m = i0 + bi < hi
            gl.store(stack + g * nb + tp + i0 - lo + bi, gl.load(table + cell * nb + i0 + bi, mask=m),
                     mask=m)
        tp = tp + gl.maximum(hi - lo, 0)
    else:
        for i0 in range(lo, hi, 32 * NUM_WARPS):
            bid = gl.load(table + cell * nb + i0 + bi, mask=i0 + bi < hi, other=-1)
            v = (bid >= 0).to(gl.int32)
            gl.store(stack + g * nb + tp + gl.cumsum(v, axis=0) - v, bid, mask=v > 0)
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
def _append_links(s, e, main, c, a, b, seg_start, seg_gb, bucket_mean, link_tgt, s_cnt, g, k_lo, slot_width, n_cells,
                  main_time, main_src, main_lo, main_n, flags, trace, n_ev, table, log_rec, ovf_cap,
                  N_SLOTS: gl.constexpr, main_cap, LP2: gl.constexpr, LM: gl.constexpr,
                  SPK: gl.constexpr, MSEG: gl.constexpr,
                  TRACE: gl.constexpr, TRACE_CAP: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
                  POOL_B: gl.constexpr, nb):
    """A block of spikes (uniform sources): every segment [a, b) of the
    ones going to the main runs claims a range of its slot's run in group
    g's ring -- the per-slot link counts live in LDS (s_cnt), the program
    being the run's only writer -- and writes one record per link there, in
    link order: (e, source word, link index, target). MSEG segments of the
    whole block at a time."""
    m = gl.where(main, b - a, 0)
    srcw = (c << 24) | s
    zero = gl.zeros_like(s)  # per-spike pointers: vectors (batched) or the run's scalars, broadcast
    seg_gb = seg_gb + zero
    seg_start = seg_start + zero
    bucket_mean = bucket_mean + zero
    link_tgt = link_tgt + zero
    for m0 in range(0, gl.max(m, axis=0), MSEG):
        col = m0 + gl.arange(0, MSEG, layout=LM)
        ok = col[None, :] < m[:, None]
        seg = a[:, None] + col[None, :]
        gb = gl.load(seg_gb[:, None] + seg, mask=ok, other=0)
        lo_l = gl.load(seg_start[:, None] + seg, mask=ok, other=0)
        n_l = gl.load(seg_start[:, None] + seg + 1, mask=ok, other=0) - lo_l
        if TRACE >= 3:
            gb = pin(gb)
            n_l = pin(n_l)
            n_ev = mark(trace, n_ev, 14, gl.sum(gl.sum(ok.to(gl.int32), axis=1), axis=0), TRACE_CAP)
        mean = gl.load(bucket_mean[:, None] + (gb & 0xFFFF), mask=ok, other=0.0)
        if TRACE >= 3:
            mean = pin(mean)
            n_ev = mark(trace, n_ev, 15, 0, TRACE_CAP)
        k = gl.maximum(gl.floor((e[:, None] + mean) / slot_width).to(gl.int32), k_lo)
        far = ok & (k - k_lo >= N_SLOTS)
        # the flags per lane (masked: normally no lane), not through reductions across warps
        gl.atomic_max(flags + 1 + gl.zeros_like(k), 1, mask=far)
        w = ok & ~far
        n_iter = gl.max(gl.max(gl.where(w, n_l, 0), axis=1), axis=0)  # before the full check: a bound
        slot = k % N_SLOTS
        claim = gl.reshape(gl.where(w, n_l, 0), [SPK * MSEG])
        slot_f = gl.reshape(slot, [SPK * MSEG])
        old = s_cnt.atomic_scatter_add(claim, slot_f, 0)  # old: the range's start (pooled: | blocks held)
        old = gl.convert_layout(gl.reshape(old, [SPK, MSEG]), LP2)
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
        # link by link over the tile: iteration j writes every segment's j-th link (work ~ the
        # tile's longest segment; a flat stream of links with a histogram + scan per link lost 8-11%);
        T: gl.constexpr = SPK * MSEG
        e2 = e[:, None] + gl.zeros_like(mean)
        srcw2 = srcw[:, None] + gl.zeros_like(seg)
        # four links per iteration, their targets' loads all in flight before the first record
        # store (one round trip per four links, not per link). Whole 32-bit words, unmasked: a
        # masked load is a branch and the waits fall back to vmcnt(0) across it; lanes without
        # that link re-read word 0 (every lane's link_tgt is a real connection's)
        lt32 = link_tgt.to(gl.pointer_type(gl.int32))  # the targets' words (4-byte aligned)
        for j in range(0, n_iter, 4):
            x0 = gl.load(lt32[:, None] + gl.where(w & (j < n_l), _target_word(lo_l + j, GROUP_SIZE), 0))
            x1 = gl.load(lt32[:, None] + gl.where(w & (j + 1 < n_l), _target_word(lo_l + j + 1, GROUP_SIZE), 0))
            x2 = gl.load(lt32[:, None] + gl.where(w & (j + 2 < n_l), _target_word(lo_l + j + 2, GROUP_SIZE), 0))
            x3 = gl.load(lt32[:, None] + gl.where(w & (j + 3 < n_l), _target_word(lo_l + j + 3, GROUP_SIZE), 0))
            # the records' block ids (pooled), in flight with the targets: one round trip per four links
            k0 = _table_blk(table, cell, base + j, N_SLOTS, POOL_B, nb, POOL_RESERVE)
            k1 = _table_blk(table, cell, base + j + 1, N_SLOTS, POOL_B, nb, POOL_RESERVE)
            k2 = _table_blk(table, cell, base + j + 2, N_SLOTS, POOL_B, nb, POOL_RESERVE)
            k3 = _table_blk(table, cell, base + j + 3, N_SLOTS, POOL_B, nb, POOL_RESERVE)
            if TRACE >= 3:
                x3 = pin(x3)
                n_ev = mark(trace, n_ev, 17, 0, TRACE_CAP)
            rec0 = main_src - 1
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


@gluon.jit
def _run_gate(ring, ctl, pid, run_id):
    """Whether program pid takes part in run run_id: not once a run stopped (ctl, the launches' stop word:
    every run that starts after a stop waits for the host to rerun it), not if its group finished this run
    already (a rerun: the groups that finished before the stop skip). Read before anything is written."""
    done = ring + gl.load(ring + 8)
    return (gl.load(ctl) == 0) & (gl.load(done + pid) < run_id)


@gluon.jit
def _run_done(ring, pid, run_id):
    """Program pid's group finished run run_id (its last store)."""
    gl.store(ring + gl.load(ring + 8) + pid, run_id)


# ------------------------------------------------- delivery helpers


@gluon.jit
def _deliver_link(state, t, w, srcword, e, received, n_received, lane, mask, k, pid, log_ptr, log_e, log_cap,
                  RECEIVE: gl.constexpr, GROUP_SIZE: gl.constexpr, LOG: gl.constexpr, COUNT: gl.constexpr):
    """ONE link: t its target within the group (uniform over the program,
    -1 for none), w its weight, e its time word. Every lane compares
    against t and the target lane runs the receive. A multapse's weights
    arrive one link at a time, in link order."""
    hit = (lane == t) & mask
    # RECEIVE(state, hit, time, w, srcword) -> state is the transpiled receive, rooted at
    #   `hit` (MaskAliasTransformer's is_target: the one target lane); state: the node attrs it touches, as a
    #   tuple; edge.x -> the link's staged value of x (here w); time = the link's arrival, e + its bucket's mean
    #   delay (D-1); the code chosen by the link's receive variant, a scalar (warp-uniform) branch on its
    #   connection id (D-6)
    # send runs here too, after every delivery, rooted at `hit` (D-2) -- and once more after
    #   update; only for connections whose receive writes a node attr send reads (fused_template's
    #   per-delivery send: PD14_neuron has none). GAP (D-3): its n2.spike() (and receive's) can fire any number of times
    #   per neuron and run (D-3): each spike appended to the program's emission buffer, its time the expression's
    # GAP (D-10): source.x is unsupported for now
    state = RECEIVE(state, hit, e, w, srcword)
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
def _walk(r_tgt, r_w, r_src, r_e, nq, state, received, n_received, lane, mask, ui, k, pid,
          log_ptr, log_e, log_cap, RECEIVE: gl.constexpr, GROUP_SIZE: gl.constexpr, U: gl.constexpr,
          LU: gl.constexpr, LOG: gl.constexpr, COUNT: gl.constexpr, STAGE_E: gl.constexpr):
    """The staged stream block's nq links, U per broadcast LDS read,
    received one by one by their target lanes."""
    # GAP (D-7): one broadcast LDS read and one register select per staged attr slot (here: w);
    #   up to the most attrs any variant's receive reads (D-7)
    # GAP (D-6, D-1): per link, its variant from srcword >> 24 (D-6) and its arrival time (D-1) are
    #   selected alongside -- a variant is the same for all links of one source spike through one connection,
    #   but links of different spikes and connections interleave here, so the choice is made per link
    for g in range(0, (nq + U - 1) // U):
        tg = r_tgt.index(g).load(LU)
        wg = r_w.index(g).load(LU)
        if LOG:
            sg = r_src.index(g).load(LU)
        if STAGE_E:
            eg = r_e.index(g).load(LU)
        for u in gl.static_range(U):
            t = gl.sum(gl.where(ui == u, tg, 0), axis=0)    # in-thread: a register select
            w = gl.sum(gl.where(ui == u, wg, 0.0), axis=0)
            if LOG:
                srcword = gl.sum(gl.where(ui == u, sg, 0), axis=0)
            else:
                srcword = 0
            if STAGE_E:
                e = gl.sum(gl.where(ui == u, eg, 0.0), axis=0)
            else:
                e = 0.0
            state, received, n_received = _deliver_link(
                state, t, w, srcword, e, received, n_received, lane, mask, k, pid, log_ptr, log_e, log_cap,
                RECEIVE, GROUP_SIZE, LOG, COUNT)
    return state, received, n_received


@gluon.jit
def _stage_records(rec0, cell, b0, nm, n_cells, s_atp, s_tgt, s_w, s_src, s_e, table, STREAM: gl.constexpr,
                   NUM_WARPS: gl.constexpr, LOG: gl.constexpr, STAGE_E: gl.constexpr, N_SLOTS: gl.constexpr,
                   main_cap, POOL_B: gl.constexpr, nb):
    """Links b0 .. b0 + STREAM of a slot's main run staged in LDS for the
    walk: each record ONE 128-bit load, split into its fields in registers,
    the weight by link index through the connection's attribute pointer (a
    read-only LDS table), every field stored straight from the load's layout
    into a flat LDS buffer -- no layout conversion. Returns the links in the
    block."""
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
    atp = s_atp.gather(srcw >> 24, 0).to(gl.pointer_type(gl.float32))
    w = gl.load(atp + l, mask=qv, other=0.0)
    # GAP (forward): the transpiled edge forward goes here, on link lanes (one record per lane, rooted
    #   at qv -- the old template's lane_mask="_lk_valid"), its assigned attrs stored back at l before staging,
    #   so receive sees the updated values as in the no-delay template. The variant differs per LANE here
    #   (unlike in the walk): each variant present in the block runs masked to its lanes (D-6); time = the
    #   link's arrival (D-1)
    # LIMITATION (D-8): one link twice in one stream block (its
    #   source spiking in two runs taken in together) is read by both lanes before either stores: the later
    #   lane works on a stale value, and the last store wins, overwriting the changes of every forward call on
    #   that link in the block but the last
    # GAP (D-9): forward's destination.x (the target neuron's state) isn't wired up -- the target is
    #   known here (t) and its state is in scope in the walk; connecting the two is left for later
    # GAP (D-10): source.x unsupported for now
    s_tgt.store(gl.where(qv, t, -1))
    s_w.store(w)
    if LOG:
        s_src.store(srcw)
    if STAGE_E:
        s_e.store(gl.convert_layout(e_i, LRQ).to(gl.float32, bitcast=True))
    gl.barrier()
    return gl.minimum(nm - b0, STREAM)


@gluon.jit
def _merge_links(main_time, main_src, main_lo, main_n, side_time, side_src, side_lo, side_n, cell, nm, ns, n_cells,
                 attr_addr, state, received, n_received, lane, mask, k, pid, log_ptr, log_e, log_cap, table, side_table,
                 RECEIVE: gl.constexpr, GROUP_SIZE: gl.constexpr, LOG: gl.constexpr, COUNT: gl.constexpr,
                 N_SLOTS: gl.constexpr, main_cap, POOL_B: gl.constexpr, nb):
    """A slot with side entries: its main and side runs of link records
    merged by emission time with two cursors (main first on ties), each
    link received in order."""
    # GAP (forward, D-1): as in _stage_records and _walk, one link at a
    #   time (forward strictly sequential here: no D-8 limitation); the main/side merge by the records' time
    #   word -- the arrival once the record carries it (D-1)
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
            RECEIVE, GROUP_SIZE, LOG, COUNT)
    return state, received, n_received


# ------------------------------------------------- phase 1: the pull


@gluon.jit
def _pull(pid, n_groups, plan, n_conns, ring, wmark_ptr, oflags, ctl, k_lo, slot_width, round_id, parity, trace, n_ev,
          SK: gl.constexpr, KP: gl.constexpr, MAX_CONNS: gl.constexpr, LOG_CJ: gl.constexpr,
          N_SLOTS: gl.constexpr, NSP: gl.constexpr,
          UNIFORM_SOURCES: gl.constexpr, BATCH: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
          TRACE: gl.constexpr, TRACE_CAP: gl.constexpr, POOL_B: gl.constexpr):
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
    (hist_addr, topo_addr, link_target_addr, attr_addr, dir_addr, hcap_p, t_src_p, t_seg_p, t_bkt_p, t_lnk_p,
     dpairs_p, dwords_p, last_read) = _fused_plan(plan, n_conns)
    (main_time, main_src, main_lo, main_n, main_count, side_time, side_src, side_lo, side_n, side_count,
     _watermark, flags, table, side_table, stack, top, nblk, lost, log_rec, ovf_cap, main_cap, side_cap,
     nb, stats) = _ring(ring, POOL_B)
    n_cells = n_groups * N_SLOTS

    # No user code in the pull -- it only moves (e, source, link, target) records into the ring;
    #   every hook runs at delivery, in delivery order (strict per-link order)
    ci = gl.arange(0, MAX_CONNS, layout=LV)
    cv = ci < n_conns
    hbuf_c = gl.load(hist_addr + ci, mask=cv, other=0)
    last_c = gl.load(last_read + parity * n_conns + ci, mask=cv, other=0)
    newest_c = last_c
    oldest_c = gl.zeros_like(last_c) + 2147483647
    for jj in gl.static_range(SK):  # per connection: its newest and oldest fresh run (last-read, overrun)
        tag = gl.load((hbuf_c.to(gl.pointer_type(gl.int32)) + 8 * jj).to(gl.pointer_type(gl.int64)),
                      mask=cv, other=0)  # run id and round, read whole
        rid = (tag & 0xFFFFFFFF).to(gl.int32)
        rnd = (tag >> 32).to(gl.int32)
        f = cv & (rid > last_c) & (rnd < round_id)
        newest_c = gl.where(f, gl.maximum(newest_c, rid), newest_c)
        oldest_c = gl.where(f, gl.minimum(oldest_c, rid), oldest_c)
    gl.store(last_read + (1 - parity) * n_conns + ci, newest_c, mask=cv)  # idempotent across programs
    if gl.max((cv & (oldest_c < 2147483647) & (oldest_c > last_c + 1)).to(gl.int32), axis=0) > 0:
        gl.atomic_max(oflags, 1)  # a run fell out of the history before this consumer read it
    # per (connection, slot): the fresh runs, taken in in order of their
    # (earliest) emission time -- under uniform sources a run's spikes
    # share one e but runs differ, and main runs must grow in e
    cj = gl.arange(0, MAX_CJ, layout=LV)
    c_of = cj // KP
    j_of = cj % KP
    vj = (c_of < n_conns) & (j_of < SK)
    hbuf = gl.load(hist_addr + c_of, mask=vj, other=0)
    hcap = gl.load(hcap_p + c_of, mask=vj, other=0)
    hdr = hbuf.to(gl.pointer_type(gl.int32)) + 8 * j_of
    tag = gl.load(hdr.to(gl.pointer_type(gl.int64)), mask=vj, other=0)  # run id and round, read whole
    rid = (tag & 0xFFFFFFFF).to(gl.int32)
    rnd = (tag >> 32).to(gl.int32)
    cnt = gl.load(hdr + 2, mask=vj, other=0)
    wend = gl.load((hdr + 3).to(gl.pointer_type(gl.float32)), mask=vj, other=0.0)
    early = gl.load((hdr + 4).to(gl.pointer_type(gl.float32)), mask=vj, other=float("inf"))
    lastj = gl.load(last_read + parity * n_conns + c_of, mask=vj, other=0)
    fresh = vj & (rid > lastj) & (rnd < round_id) & (cnt > 0)
    n_spk = gl.where(fresh, gl.minimum(cnt, hcap), 0)
    key = gl.where(fresh, early, float("inf"))
    key_r = gl.convert_layout(key, gl.SliceLayout(0, LKC))
    cj_r = gl.arange(0, MAX_CJ, layout=gl.SliceLayout(0, LKC))
    key_l = gl.convert_layout(key, gl.SliceLayout(1, LKC))
    cj_l = gl.arange(0, MAX_CJ, layout=gl.SliceLayout(1, LKC))
    before = (key_r[None, :] < key_l[:, None]) | ((key_r[None, :] == key_l[:, None]) & (cj_r[None, :] < cj_l[:, None]))
    order = gl.convert_layout(gl.sum(before.to(gl.int32), axis=1), LV)  # rank by (e, cj); fresh runs first
    n_runs = gl.sum(fresh.to(gl.int32), axis=0)
    node_addr = hbuf + 4 * (8 * SK + j_of * hcap).to(gl.int64)
    time_addr = node_addr + 4 * (SK * hcap).to(gl.int64)
    wmark = gl.load(wmark_ptr + pid)
    if UNIFORM_SOURCES and BATCH:
        # per (connection, slot) pair, everything a spike's decode needs, as read-only LDS tables: a
        # spike looks its pair's values up directly (no barriers, no layout conversions per block)
        tb_j = gl.load(topo_addr + c_of, mask=vj, other=0).to(gl.pointer_type(gl.int32))
        nsrc_j = gl.load(t_src_p + c_of, mask=vj, other=0)
        _j_offs, j_start, j_gb, j_mean = topology_ptrs(tb_j, nsrc_j, gl.load(t_seg_p + c_of, mask=vj, other=0))
        j_doffs, j_dfirst, j_dwords = directory_ptrs(
            gl.load(dir_addr + c_of, mask=vj, other=0).to(gl.pointer_type(gl.int32)), nsrc_j,
            gl.load(dpairs_p + c_of, mask=vj, other=0))
        tab_node = _table(node_addr, SM)
        tab_time = _table(time_addr, SM)
        tab_wend = _table(wend, SM)
        tab_start = _table(j_start.to(gl.int64), SM)
        tab_gb = _table(j_gb.to(gl.int64), SM)
        tab_mean = _table(j_mean.to(gl.int64), SM)
        # every pair a real connection's link targets (the append reads targets unmasked)
        tab_ltp = _table(gl.load(link_target_addr + gl.minimum(c_of, n_conns - 1), mask=c_of * 0 < n_conns,
                                 other=0), SM)
        tab_doffs = _table(j_doffs.to(gl.int64), SM)
        tab_dfirst = _table(j_dfirst.to(gl.int64), SM)
        tab_dwords = _table(j_dwords.to(gl.int64), SM)
        tab_W = _table(gl.load(dwords_p + c_of, mask=vj, other=0), SM)
        tab_first = gl.allocate_shared_memory(gl.int32, [MAX_CJ], SM)  # per e-group: each pair's first spike
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
        # spanning runs and connections; e-groups in order of e
        for ri in range(0, n_runs):
            e0 = gl.sum(gl.where(order == ri, early, 0.0), axis=0)
            if gl.sum((key < e0).to(gl.int32), axis=0) == ri:  # the first run of its e-group leads it
                n_g = gl.where(fresh & (key == e0), n_spk, 0)
                g_first = gl.cumsum(n_g, axis=0) - n_g
                n_tot = gl.sum(n_g, axis=0)
                total += n_tot
                e_max = gl.to_tensor(float("-inf"))
                tab_first.store(g_first)
                gl.barrier()
                for t0 in range(0, n_tot, SPK):
                    t = t0 + gl.arange(0, SPK, layout=LK)
                    tv = t < n_tot
                    # each spike's pair: the last whose first spike is at or before it (pairs
                    # without spikes here start where the next one does, so never the last one),
                    # a binary search over the LDS table (first[0] = 0)
                    cj_t = gl.zeros_like(t)
                    f_t = gl.zeros_like(t)
                    for st in gl.static_range(LOG_CJ):
                        probe = cj_t + (MAX_CJ >> (st + 1))
                        v = tab_first.gather(probe, 0)
                        le = v <= t
                        cj_t = gl.where(le, probe, cj_t)
                        f_t = gl.where(le, v, f_t)
                    i = (t - f_t).to(gl.int64)
                    s = gl.load((tab_node.gather(cj_t, 0) + 4 * i).to(gl.pointer_type(gl.int32)), mask=tv,
                                other=0)
                    e = gl.load((tab_time.gather(cj_t, 0) + 4 * i).to(gl.pointer_type(gl.float32)), mask=tv,
                                other=0.0)
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
                    n_ev = _append_links(s, e, main, c_t, a, b, seg_start, seg_gb, bucket_mean,
                                         ltp_t, s_cnt, pid, k_lo, slot_width, n_cells,
                                         main_time, main_src, main_lo, main_n,
                                         flags, trace, n_ev, table, log_rec, ovf_cap, N_SLOTS, main_cap,
                                         LP2, LM, SPK, MSEG, TRACE, TRACE_CAP, NUM_WARPS, GROUP_SIZE, POOL_B,
                                         nb)
                    if TRACE >= 2:
                        n_ev = mark(trace, n_ev, 13, gl.sum(gl.where(main, b - a, 0), axis=0), TRACE_CAP)
                wmark = gl.maximum(wmark, e_max)  # later e-groups are compared against this one
                gl.barrier()  # the group's appends are in before the next group's (main runs grow in e)
    else:
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
                                         LM, SPK, MSEG, TRACE, TRACE_CAP, NUM_WARPS, GROUP_SIZE, POOL_B, nb)
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
                        if k - k_lo >= N_SLOTS:
                            gl.atomic_max(flags + 1, 1)
                        else:
                            cell = pid * N_SLOTS + k % N_SLOTS
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
        if POOL_B > 0:
            fits, top0, used, most = _pool_fixup(s_cnt, cnt0, main_src - 1, table, stack, log_rec, ovf_cap, top0,
                                                 pid, sl, LV, N_SLOTS, main_cap, POOL_B, nb, NUM_WARPS)
            gl.store(stats + 2 * pid, used)  # the run's status: the blocks the group holds, the most logged
            gl.store(stats + 2 * pid + 1, most)
            if fits:
                packed = s_cnt.load(LV)  # the counts and block counts back to the ring
                gl.store(main_count + pid * N_SLOTS + sl, packed & ((1 << POOL_CNT_BITS) - 1), mask=sl < N_SLOTS)
                gl.store(nblk + pid * N_SLOTS + sl, packed >> POOL_CNT_BITS, mask=sl < N_SLOTS)
                gl.store(top + pid, top0)
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
             log_ptr, log_cap, trace, n_ev, RECEIVE: gl.constexpr, GROUP_SIZE: gl.constexpr, N_SLOTS: gl.constexpr,
             STREAM: gl.constexpr, U: gl.constexpr,
             SPT: gl.constexpr, NUM_WARPS: gl.constexpr, MAX_CONNS: gl.constexpr, LOG: gl.constexpr,
             COUNT: gl.constexpr, STAGE_E: gl.constexpr, TRACE: gl.constexpr, TRACE_CAP: gl.constexpr,
             POOL_B: gl.constexpr):
    """Phase 2 of program pid: its group's due slots k_lo..k_hi, every link
    through RECEIVE (see the module docstring); state: the tuple of node
    attrs receive touches, [GROUP_SIZE] tensors in the program's neuron
    layout (lane's). STAGE_E: the walk stages each record's time word too
    (LOG, or a receive that reads time). Returns (state, n_ev)."""
    # neurons over lanes and warps, the U links of a walk group whole in every thread
    LT: gl.constexpr = gl.BlockedLayout([SPT, U], [32, 1], [NUM_WARPS, 1], [1, 0])
    LN: gl.constexpr = gl.SliceLayout(1, LT)
    LU: gl.constexpr = gl.SliceLayout(0, LT)
    LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    ui = gl.arange(0, U, layout=LU)
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
    # the stream block's fields, staged flat (the walk reads them as [STREAM // U, U]), and the
    # connections' attribute pointers (weights by link index), a read-only table
    # GAP (D-7): one block pointer per connection; one LDS stage buffer per attr slot -- the most
    #   attrs any variant's receive reads (here: the weight), D-7
    cq = gl.arange(0, MAX_CONNS, layout=LV)
    s_atp = gl.allocate_shared_memory(gl.int64, [MAX_CONNS], SM)
    s_atp.store(gl.load(attr_addr + cq, mask=cq < n_conns, other=0))
    s_tgt = gl.allocate_shared_memory(gl.int32, [STREAM], SM)
    s_w = gl.allocate_shared_memory(gl.float32, [STREAM], SM)
    if LOG:
        s_src = gl.allocate_shared_memory(gl.int32, [STREAM], SM)
    else:  # placeholder, never touched
        s_src = s_tgt
    if STAGE_E:
        s_e = gl.allocate_shared_memory(gl.float32, [STREAM], SM)
    else:  # placeholder, never touched
        s_e = s_w
    r_tgt = s_tgt.reinterpret(gl.int32, [STREAM // U, U], SM)
    r_w = s_w.reinterpret(gl.float32, [STREAM // U, U], SM)
    r_src = s_src.reinterpret(gl.int32, [STREAM // U, U], SM)
    r_e = s_e.reinterpret(gl.float32, [STREAM // U, U], SM)
    gl.barrier()
    # GAP (D-1): every delivery's time is its link's arrival, e + its bucket's mean delay (D-1): the
    #   pull knows the mean when it appends, so the record's time word carries the arrival (the main/side
    #   decision on e is made by then)
    # LIMITATION (D-1): a slot's records are in emission order, not arrival order -- links of different
    #   buckets in one slot can reach a target out of arrival order by less than one slot width (a receive-side
    #   integrator sees time step back)
    if POOL_B > 0:  # the group's free-stack top through the delivery, stored once at its end
        tp = gl.load(top + pid)
        lost_v = gl.load(lost)
    else:
        tp = gl.to_tensor(0)
        lost_v = gl.to_tensor(0)
    for k in range(k_lo, k_hi + 1):
        cell = pid * N_SLOTS + k % N_SLOTS
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
                nq = _stage_records(main_src - 1, cell, b0, nm, n_cells, s_atp, s_tgt, s_w, s_src, s_e, table,
                                    STREAM, NUM_WARPS, LOG, STAGE_E, N_SLOTS, main_cap, POOL_B, nb)
                if TRACE:
                    n_ev = mark(trace, n_ev, 5, nq, TRACE_CAP)
                state, received, n_received = _walk(
                    r_tgt, r_w, r_src, r_e, nq, state, received, n_received, lane, mask, ui, k, pid,
                    log_ptr, log_e, log_cap, RECEIVE, GROUP_SIZE, U, LU, LOG, COUNT, STAGE_E)
                if TRACE:
                    state = _pin_first(state)
                    n_ev = mark(trace, n_ev, 6, nq, TRACE_CAP)
                gl.barrier()
        else:
            state, received, n_received = _merge_links(
                main_time, main_src, main_lo, main_n, side_time, side_src, side_lo, side_n, cell, nm, ns, n_cells,
                attr_addr, state, received, n_received, lane, mask, k, pid, log_ptr, log_e, log_cap, table,
                side_table, RECEIVE, GROUP_SIZE, LOG, COUNT, N_SLOTS, main_cap, POOL_B, nb)
        if POOL_B > 0:  # the delivered cell's blocks back on the group's free stack
            # the main run keeps its first POOL_RESERVE blocks, the side run none
            keep = gl.minimum(held, POOL_RESERVE)  # (fixed blocks: never missing, never given away)
            tp = _free_blocks(table, stack, tp, pid, cell, keep, held, lost_v, LV, nb, NUM_WARPS)
            if held > keep:
                gl.store(nblk + cell, keep)
            tp = _free_blocks(side_table, stack, tp, pid, cell, 0, (ns + POOL_B - 1) // POOL_B, lost_v, LV,
                              nb, NUM_WARPS)
        gl.store(main_count + cell, 0)
        gl.store(side_count + cell, 0)
    if POOL_B > 0:
        gl.store(top + pid, tp)
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
    # GAP (D-3): the history's cap is no longer one spike per neuron (D-3); a run's spikes beyond it raise
    #   the overflow flag
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
