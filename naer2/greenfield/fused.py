"""
The fused green-field kernel: one launch per population and run -- intake,
delivery, propagation and emission in one Gluon kernel, one program per
group. Intake is PULLED: every program takes in its own group's entries
itself, so no program depends on another within the launch (a push intake
appending into any group's ring would need a grid-wide barrier before
delivery).

Per program (group g):
  1. pull intake: for every incoming connection, the runs of its source
     since this consumer last read it (naer2/greenfield/history.py: slots
     tagged with an earlier round, run id past the last-read id); per spike
     g's segments among the source's (sorted by (group, bucket)) through
     the topology's group directory (or two interleaved binary searches),
     every link of them one 16-byte record (e, source word, link index,
     target) in its slot's run of g's ring (naer2/intake.py). Uniform
     emission: all fresh runs of one emission time as one stream of spike
     blocks (each spike's run and its connection's pointers from read-only
     LDS tables), a block of spikes x a block of their segments at a time,
     the link loop four links per step. Otherwise sequential per entry with
     a per-group watermark: main run if e >= the watermark and not
     emission-delayed, else sorted insertion into the side run -- correct
     in any spike order (the baseline; slow).
  2. delivery: the due slots of g's ring, as naer2/greenfield/update.py
     but straight from the link records: STREAM records at a time, each one
     128-bit load, its fields staged flat in LDS and walked (side-run merge
     where needed).
  3. propagation, stand-in send, emission into the history (flat list per
     run, no per-group sort: consumers order what they need themselves).

Contracts: a run reads source runs from earlier rounds only (round tags);
the last-read run ids are double-buffered by the consumer's run parity (all
programs read the same values); entries arriving before the ring position
go to its first undelivered slot (late), at or past N_SLOTS ahead are far
future (flag 1), full runs flag 0, uniform violations flag 2, a consumer
falling behind its sources' history raises the overrun flag.
"""

import contextlib
import math

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from naer2.intake import Ring, _rec, _ring
from naer2.topology_new import TopologyNew, directory_ptrs, topology_ptrs
from naer2.trace import Trace, mark, pin, pin_s, trace_begin, trace_end

from .history import EmissionHistory, history_ptrs, history_tag
from .pop import WALK
from .update import _receive, _walk

FUSED_TRACE_TAGS = ("start", "state", "pulled", "slot", "headers", "streamed", "walked", "delivered", "sent", "end",
                    "runs", "spikes", "found", "appended",  # TRACE >= 2: inside the pull
                    "a:segments", "a:mean", "a:claimed", "a:targets", "a:stored")  # TRACE >= 3:
                    # inside the append, per tile
PULL = dict(batch=True, directory=True)  # FusedPop defaults, see FusedPop.__init__; tools/fused_sweep.py: directory
# and batching ~30% (PD14, G=256, 780M)
SPK = gl.constexpr(32)   # spikes per pull block (uniform)
MSEG = gl.constexpr(4)   # segments per pull column block (uniform): a spike reaches a group with ~2-3 (PD14)


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


class FusedPlan:
    """A consumer's incoming connections, built once into one int32 buffer:
    per connection its source's EmissionHistory and its TopologyNew
    addresses (+ the topology's link_target and attribute arrays, which the
    delivery walk reads, and its group directory), the history's cap, the
    topology's source, segment, bucket and link counts, its most segments of
    one source, the directory's pair count and words per source, and the
    last-read run ids (double-buffered by the consumer's run parity)."""

    def __init__(self, histories: list[EmissionHistory], topologies: list[TopologyNew], n_groups: int, device):
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
        self.iters = max(max_segs + [1]).bit_length()  # binary-search steps: ceil(log2(most segments + 1))


# ------------------------------------------------- pull intake


@gluon.jit
def _group_range(seg_gb, lo, hi, g, ITERS: gl.constexpr):
    """Per spike, the sub-range of its segments [lo, hi) that go into group
    g: first with group >= g, first with group > g -- two binary searches,
    interleaved, ITERS dependent steps."""
    a_lo = lo
    a_hi = hi
    b_lo = lo
    b_hi = hi
    for _ in gl.static_range(ITERS):
        am = (a_lo + a_hi) // 2
        bm = (b_lo + b_hi) // 2
        av = a_lo < a_hi
        bv = b_lo < b_hi
        ga = (gl.load(seg_gb + am, mask=av, other=0) >> 16) & 0xFFFF
        gb = (gl.load(seg_gb + bm, mask=bv, other=0) >> 16) & 0xFFFF
        ra = av & (ga < g)
        rb = bv & (gb <= g)
        a_lo = gl.where(ra, am + 1, a_lo)
        a_hi = gl.where(av & ~ra, am, a_hi)
        b_lo = gl.where(rb, bm + 1, b_lo)
        b_hi = gl.where(bv & ~rb, bm, b_hi)
    return a_lo, b_lo


@gluon.jit
def _side_insert(side_time, side_src, side_lo, side_n, side_count, flags, cell, n_cells, e, srcword, lo, n,
                 SIDE_CAP: gl.constexpr, LB: gl.constexpr, BLK: gl.constexpr):
    """One entry into a side run, keeping it sorted by e (after every entry
    with time <= e). The program owns the run; barriers order the shift
    between its warps."""
    lanes = gl.arange(0, BLK, layout=LB)
    cnt = gl.load(side_count + cell)
    if cnt >= SIDE_CAP:
        gl.atomic_max(flags, 1)
    else:
        p = cnt
        for s0 in range(0, cnt, BLK):
            q = s0 + lanes
            v = gl.load(side_time + 4 * _rec(cell, q, n_cells), mask=q < cnt, other=0.0)
            p = gl.minimum(p, gl.min(gl.where((q < cnt) & (v > e), q, cnt), axis=0))
        gl.barrier()
        for s0 in range(0, cnt - p, BLK):  # shift [p, cnt) up by one, top first
            q = cnt - 1 - s0 - lanes
            qm = q >= p
            src_at = 4 * _rec(cell, q, n_cells)
            dst_at = 4 * _rec(cell, q + 1, n_cells)
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
        at = 4 * _rec(cell, p, n_cells)
        gl.store(side_time + at, e)
        gl.store(side_src + at, srcword)
        gl.store(side_lo + at, lo)
        gl.store(side_n + at, n)
        gl.store(side_count + cell, cnt + 1)
    gl.barrier()


@triton.jit
def _popcount(x):
    v = x.to(tl.uint32)
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) >> 24).to(tl.int32)


@gluon.jit
def _segments_of(s, tv, g, seg_offsets, seg_gb, dir_offs, dir_first, dir_words, W, ITERS: gl.constexpr,
                 DIR: gl.constexpr):
    """Per spike (source s), its segments into group g as [a, b).
    DIR: through the topology's group directory, three dependent steps
    (bitmap word + directory offset -> the pair's first segment and the
    next pair's -> the segments); a clear bit stops after the first.
    Otherwise two interleaved binary searches over the source's segments."""
    if DIR:
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
    else:
        lo = gl.load(seg_offsets + s, mask=tv, other=0)
        hi = gl.load(seg_offsets + s + 1, mask=tv, other=0)
        return _group_range(seg_gb, lo, hi, g, ITERS)


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
def _link_record(rec0, word, j, w, n_l, lo_l, base, cell, n_cells, e2, srcw2, T: gl.constexpr,
                 NUM_WARPS: gl.constexpr):
    """Every tile segment's j-th link as its ring record; word: the 32-bit word holding its target."""
    wj = w & (j < n_l)
    l = lo_l + j
    tgt = (word >> ((l & 1) * 16)) & 0xFFFF
    at = 4 * _rec(cell, base + j, n_cells)
    _store_records(rec0, gl.reshape(at, [T]), gl.reshape(e2, [T]), gl.reshape(srcw2, [T]), gl.reshape(l, [T]),
                   gl.reshape(tgt, [T]), gl.reshape(wj, [T]), NUM_WARPS)


@gluon.jit
def _append_links(s, e, main, c, a, b, seg_start, seg_gb, bucket_mean, link_tgt, s_cnt, g, k_lo, slot_width, n_cells,
                  main_time, main_src, main_lo, main_n, flags, trace, n_ev,
                  N_SLOTS: gl.constexpr, MAIN_CAP: gl.constexpr, LP2: gl.constexpr, LM: gl.constexpr,
                  SPK: gl.constexpr, MSEG: gl.constexpr,
                  TRACE: gl.constexpr, TRACE_CAP: gl.constexpr, NUM_WARPS: gl.constexpr):
    """A block of spikes (uniform emission): every segment [a, b) of the
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
        base = s_cnt.atomic_scatter_add(claim, gl.reshape(slot, [SPK * MSEG]), 0)  # old: the range's start
        base = gl.convert_layout(gl.reshape(base, [SPK, MSEG]), LP2)
        if TRACE >= 3:
            base = pin(base)
            n_ev = mark(trace, n_ev, 16, 0, TRACE_CAP)
        full = w & (base + n_l > MAIN_CAP)
        gl.atomic_max(flags + gl.zeros_like(k), 1, mask=full)
        w = w & ~full
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
            x0 = gl.load(lt32[:, None] + gl.where(w & (j < n_l), (lo_l + j) >> 1, 0))
            x1 = gl.load(lt32[:, None] + gl.where(w & (j + 1 < n_l), (lo_l + j + 1) >> 1, 0))
            x2 = gl.load(lt32[:, None] + gl.where(w & (j + 2 < n_l), (lo_l + j + 2) >> 1, 0))
            x3 = gl.load(lt32[:, None] + gl.where(w & (j + 3 < n_l), (lo_l + j + 3) >> 1, 0))
            if TRACE >= 3:
                x3 = pin(x3)
                n_ev = mark(trace, n_ev, 17, 0, TRACE_CAP)
            rec0 = main_src - 1
            _link_record(rec0, x0, j, w, n_l, lo_l, base, cell, n_cells, e2, srcw2, T, NUM_WARPS)
            _link_record(rec0, x1, j + 1, w, n_l, lo_l, base, cell, n_cells, e2, srcw2, T, NUM_WARPS)
            _link_record(rec0, x2, j + 2, w, n_l, lo_l, base, cell, n_cells, e2, srcw2, T, NUM_WARPS)
            _link_record(rec0, x3, j + 3, w, n_l, lo_l, base, cell, n_cells, e2, srcw2, T, NUM_WARPS)
            if TRACE >= 3:
                n_ev = mark(trace, n_ev, 18, 0, TRACE_CAP)
    return n_ev


@gluon.jit
def _stage_records(rec0, cell, b0, nm, n_cells, s_atp, s_tgt, s_w, s_src, s_e, STREAM: gl.constexpr,
                   NUM_WARPS: gl.constexpr, LOG: gl.constexpr):
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
    at = gl.multiple_of(4 * _rec(cell, q, n_cells), 4)
    col = gl.arange(0, 4, layout=gl.SliceLayout(0, LR))
    words = gl.load(rec0 + at[:, None] + col[None, :], mask=qv[:, None] & (col[None, :] < 4), other=0)
    el, st = gl.split(gl.reshape(words, [STREAM, 2, 2]))  # (e, link index), (source word, target)
    e_i, l = gl.split(el)
    srcw, t = gl.split(st)
    l = gl.convert_layout(l, LRQ)
    srcw = gl.convert_layout(srcw, LRQ)
    t = gl.convert_layout(t, LRQ)
    atp = s_atp.gather(srcw >> 24, 0).to(gl.pointer_type(gl.float32))
    w = gl.load(atp + l, mask=qv, other=0.0)
    s_tgt.store(gl.where(qv, t, -1))
    s_w.store(w)
    if LOG:
        s_src.store(srcw)
        s_e.store(gl.convert_layout(e_i, LRQ).to(gl.float32, bitcast=True))
    gl.barrier()
    return gl.minimum(nm - b0, STREAM)


@gluon.jit
def _merge_links(main_time, main_src, main_lo, main_n, side_time, side_src, side_lo, side_n, cell, nm, ns, n_cells,
                 attr_addr, current, received, n_received, lane, mask, k, pid,
                 log_ptr, log_e, log_cap, GROUP_SIZE: gl.constexpr, LOG: gl.constexpr, COUNT: gl.constexpr):
    """A slot with side entries: its main and side runs of link records
    merged by emission time with two cursors (main first on ties), each
    link received in order."""
    im = gl.to_tensor(0)
    js = gl.to_tensor(0)
    for _ in range(nm + ns):
        at_m = 4 * _rec(cell, im, n_cells)
        at_s = 4 * _rec(cell, js, n_cells)
        mv = im < nm
        sv = js < ns
        em = gl.load(main_time + at_m, mask=mv, other=float("inf"))
        es = gl.load(side_time + at_s, mask=sv, other=float("inf"))
        take = mv & (em <= es)
        e = gl.where(take, em, es)
        srcword = gl.where(take, gl.load(main_src + at_m, mask=mv, other=0), gl.load(side_src + at_s, mask=sv, other=0))
        l = gl.where(take, gl.load(main_lo + at_m, mask=mv, other=0), gl.load(side_lo + at_s, mask=sv, other=0))
        t = gl.where(take, gl.load(main_n + at_m, mask=mv, other=0), gl.load(side_n + at_s, mask=sv, other=0))
        im = im + take.to(gl.int32)
        js = js + (1 - take.to(gl.int32))
        w = gl.load(gl.load(attr_addr + (srcword >> 24)).to(gl.pointer_type(gl.float32)) + l)
        current, received, n_received = _receive(
            t, w, srcword, e, current, received, n_received, lane, mask, k, pid,
            log_ptr, log_e, log_cap, GROUP_SIZE, LOG, COUNT)
    return current, received, n_received


# ------------------------------------------------- the kernel


@gluon.jit(do_not_specialize=["n_neurons", "n_conns", "k_lo", "k_hi", "run_id", "round_id", "parity", "log_cap",
                              "hist_cap"])
def fused_kernel(
    t_ref_end_ptr, potential_ptr, current_ptr, last_time_ptr, n_neurons,
    plan, n_conns, ring, wmark_ptr, oflags, k_lo, k_hi, slot_width, round_id, parity,  # intake + delivery
    received_ptr, n_received_ptr, log_ptr, log_cap,          # stand-in receive, delivery log
    fire_time_ptr, emit_delay_ptr,                           # stand-in send
    history, hist_cap, run_id, t_start, t_end, trace_ptr,
    GROUP_SIZE: gl.constexpr, K: gl.constexpr, SK: gl.constexpr, KP: gl.constexpr, UNIFORM_TIME: gl.constexpr,
    N_SLOTS: gl.constexpr, MAIN_CAP: gl.constexpr, SIDE_CAP: gl.constexpr,
    LOG: gl.constexpr,
    STREAM: gl.constexpr, U: gl.constexpr, SPT: gl.constexpr, NUM_WARPS: gl.constexpr, MAX_CONNS: gl.constexpr,
    LOG_CJ: gl.constexpr,
    ITERS: gl.constexpr, COUNT: gl.constexpr = True,
    BATCH: gl.constexpr = True, DIR: gl.constexpr = True, NSP: gl.constexpr = 64,
    TRACE: gl.constexpr = 0, TRACE_CAP: gl.constexpr = 1,
):
    gl.static_assert(STREAM % U == 0, "STREAM must be a multiple of U")
    gl.static_assert(GROUP_SIZE % U == 0, "GROUP_SIZE must be a multiple of U")
    # (connection, source history slot) pairs, 2**LOG_CJ; K: own history depth, SK: the sources'
    MAX_CJ: gl.constexpr = MAX_CONNS * KP
    # delivery layouts (naer2/greenfield/update.py)
    LT: gl.constexpr = gl.BlockedLayout([SPT, U], [32, 1], [NUM_WARPS, 1], [1, 0])
    LN: gl.constexpr = gl.SliceLayout(1, LT)
    LU: gl.constexpr = gl.SliceLayout(0, LT)
    LS: gl.constexpr = gl.BlockedLayout([SPT], [32], [NUM_WARPS], [0])
    SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    # pull layouts: per (connection, slot) / per connection vectors, spikes x
    # (connection, slot) for the spike decode, spikes x segments for the expansion
    LV: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    LKC: gl.constexpr = gl.BlockedLayout([1, MAX_CJ], [32, 1], [NUM_WARPS, 1], [1, 0])
    # spikes x segments: one spike row per thread row (8 x warps rows), its MSEG segments over 4
    # threads -- per-spike values (three 64-bit pointers among them) held once per thread, not 4x
    LP2: gl.constexpr = gl.BlockedLayout([1, MSEG // 4], [8, 4], [NUM_WARPS, 1], [1, 0])
    LK: gl.constexpr = gl.SliceLayout(1, LP2)
    LM: gl.constexpr = gl.SliceLayout(0, LP2)

    pid = gl.program_id(0)
    n_groups = gl.num_programs(0)
    n_ev = gl.to_tensor(0)
    trace = trace_ptr  # placeholder when not tracing
    if TRACE:  # see naer2/trace.py; tags FUSED_TRACE_TAGS
        trace = trace_begin(trace_ptr, pid, TRACE_CAP)
        n_ev = mark(trace, n_ev, 0, 0, TRACE_CAP)
    lane = gl.arange(0, GROUP_SIZE, layout=LN)
    offs = pid * GROUP_SIZE + lane
    mask = offs < n_neurons
    ui = gl.arange(0, U, layout=LU)
    (hist_addr, topo_addr, link_target_addr, attr_addr, dir_addr, hcap_p, t_src_p, t_seg_p, t_bkt_p, t_lnk_p,
     dpairs_p, dwords_p, last_read) = _fused_plan(plan, n_conns)
    (main_time, main_src, main_lo, main_n, main_count, side_time, side_src, side_lo, side_n, side_count,
     _watermark, flags) = _ring(ring, n_groups, N_SLOTS, MAIN_CAP, SIDE_CAP)
    n_cells = n_groups * N_SLOTS

    # ---- 1. pull intake: what the sources emitted since this consumer last read them
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
    # (earliest) emission time -- under uniform emission a run's spikes
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
    if UNIFORM_TIME and BATCH:
        # per (connection, slot) pair, everything a spike's decode needs, as read-only LDS tables: a
        # spike looks its pair's values up directly (no barriers, no layout conversions per block)
        tb_j = gl.load(topo_addr + c_of, mask=vj, other=0).to(gl.pointer_type(gl.int32))
        nsrc_j = gl.load(t_src_p + c_of, mask=vj, other=0)
        j_offs, j_start, j_gb, j_mean, _tg, _at = topology_ptrs(
            tb_j, nsrc_j, gl.load(t_seg_p + c_of, mask=vj, other=0), gl.load(t_bkt_p + c_of, mask=vj, other=0),
            gl.load(t_lnk_p + c_of, mask=vj, other=0))
        j_doffs, j_dfirst, j_dwords = directory_ptrs(
            gl.load(dir_addr + c_of, mask=vj, other=0).to(gl.pointer_type(gl.int32)), nsrc_j,
            gl.load(dpairs_p + c_of, mask=vj, other=0))
        tab_node = _table(node_addr, SM)
        tab_time = _table(time_addr, SM)
        tab_wend = _table(wend, SM)
        tab_offs = _table(j_offs.to(gl.int64), SM)
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
    if UNIFORM_TIME:
        # this group's per-slot link counts, in LDS for the pull's appends (it is their only writer)
        sl = gl.arange(0, NSP, layout=LV)
        s_cnt = gl.allocate_shared_memory(gl.int32, [NSP], SM)
        s_cnt.store(gl.load(main_count + pid * N_SLOTS + sl, mask=sl < N_SLOTS, other=0))
        gl.barrier()
    violated = gl.to_tensor(0)
    total = gl.to_tensor(0)
    BLK: gl.constexpr = 32 * NUM_WARPS
    if TRACE >= 2:
        n_runs = pin_s(n_runs)
        n_ev = mark(trace, n_ev, 10, n_runs, TRACE_CAP)
    if UNIFORM_TIME and BATCH:
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
                    wend_t = tab_wend.gather(cj_t, 0)
                    c_t = cj_t // KP
                    seg_start = tab_start.gather(cj_t, 0).to(gl.pointer_type(gl.int32))
                    seg_gb = tab_gb.gather(cj_t, 0).to(gl.pointer_type(gl.int32))
                    bucket_mean = tab_mean.gather(cj_t, 0).to(gl.pointer_type(gl.float32))
                    ltp_t = tab_ltp.gather(cj_t, 0).to(gl.pointer_type(gl.int16))
                    a, b = _segments_of(s, tv, pid, tab_offs.gather(cj_t, 0).to(gl.pointer_type(gl.int32)),
                                        seg_gb, tab_doffs.gather(cj_t, 0).to(gl.pointer_type(gl.int32)),
                                        tab_dfirst.gather(cj_t, 0).to(gl.pointer_type(gl.int32)),
                                        tab_dwords.gather(cj_t, 0).to(gl.pointer_type(gl.int32)),
                                        tab_W.gather(cj_t, 0), ITERS, DIR)
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
                                         flags, trace, n_ev, N_SLOTS, MAIN_CAP, LP2, LM, SPK, MSEG,
                                         TRACE, TRACE_CAP, NUM_WARPS)
                    if TRACE >= 2:
                        n_ev = mark(trace, n_ev, 13, gl.sum(gl.where(main, b - a, 0), axis=0), TRACE_CAP)
                wmark = gl.maximum(wmark, e_max)  # later e-groups are compared against this one
                gl.barrier()  # the group's appends are in before the next group's (main runs grow in e)
    else:
        # per connection, its topology (picked per run below)
        tbuf_c = gl.load(topo_addr + ci, mask=cv, other=0)
        nsrc_c = gl.load(t_src_p + ci, mask=cv, other=0)
        nseg_c = gl.load(t_seg_p + ci, mask=cv, other=0)
        nbkt_c = gl.load(t_bkt_p + ci, mask=cv, other=0)
        nlnk_c = gl.load(t_lnk_p + ci, mask=cv, other=0)
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
            seg_offsets, seg_start, seg_gb, bucket_mean, _tg, _at = topology_ptrs(
                gl.sum(gl.where(selc, tbuf_c, 0), axis=0).to(gl.pointer_type(gl.int32)), nsrc_r,
                gl.sum(gl.where(selc, nseg_c, 0), axis=0), gl.sum(gl.where(selc, nbkt_c, 0), axis=0),
                gl.sum(gl.where(selc, nlnk_c, 0), axis=0))
            dir_offs, dir_first, dir_words = directory_ptrs(
                gl.sum(gl.where(selc, dbuf_c, 0), axis=0).to(gl.pointer_type(gl.int32)), nsrc_r,
                gl.sum(gl.where(selc, dpairs_c, 0), axis=0))
            dwords_r = gl.sum(gl.where(selc, dwords_c, 0), axis=0)
            ltp_r = gl.sum(gl.where(selc, ltp_c, 0), axis=0).to(gl.pointer_type(gl.int16))
            node_p = naddr.to(gl.pointer_type(gl.int32))
            time_p = taddr.to(gl.pointer_type(gl.float32))
            total += n_r
            if UNIFORM_TIME:
                e_max = gl.to_tensor(float("-inf"))
                for t0 in range(0, n_r, SPK):
                    # a block of the run's spikes: each one's segments into this group
                    i = t0 + gl.arange(0, SPK, layout=LK)
                    tv = i < n_r
                    s = gl.load(node_p + i, mask=tv, other=0)
                    e = gl.load(time_p + i, mask=tv, other=0.0)
                    a, b = _segments_of(s, tv, pid, seg_offsets, seg_gb, dir_offs, dir_first, dir_words, dwords_r,
                                        ITERS, DIR)
                    main = tv & (e >= wmark) & (e < wend_r)
                    has = b > a
                    violated = gl.maximum(violated, gl.max((tv & has & ~main).to(gl.int32), axis=0))
                    e_max = gl.maximum(e_max, gl.max(gl.where(main & has, e, float("-inf")), axis=0))
                    n_ev = _append_links(s, e, main, c, a, b, seg_start, seg_gb, bucket_mean, ltp_r, s_cnt, pid,
                                         k_lo, slot_width, n_cells, main_time, main_src, main_lo, main_n, flags,
                                         trace, n_ev, N_SLOTS, MAIN_CAP, LP2, LM, SPK, MSEG, TRACE, TRACE_CAP,
                                         NUM_WARPS)
                wmark = gl.maximum(wmark, e_max)  # the next run (later e) is compared against this one
                gl.barrier()  # the run's appends are in before the next run's (main runs grow in e)
            else:
                # any emission times: sequential per entry (the baseline)
                for i in range(0, n_r):
                    s = gl.load(node_p + i)
                    e = gl.load(time_p + i)
                    a, b = _segments_of(s, True, pid, seg_offsets, seg_gb, dir_offs, dir_first, dir_words, dwords_r,
                                    ITERS, DIR)
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
                                    if pos < MAIN_CAP:
                                        at = 4 * _rec(cell, pos, n_cells)
                                        gl.store(main_time + at, e)
                                        gl.store(main_src + at, srcw)
                                        gl.store(main_lo + at, l)
                                        gl.store(main_n + at, tgt)
                                        gl.store(main_count + cell, pos + 1)
                                    else:
                                        gl.atomic_max(flags, 1)
                                    gl.barrier()
                                else:
                                    _side_insert(side_time, side_src, side_lo, side_n, side_count, flags, cell,
                                                 n_cells, e, srcw, l, tgt, SIDE_CAP, LV, BLK)
                            wmark = gl.where(to_main, gl.maximum(wmark, e), wmark)
    if UNIFORM_TIME:
        if violated > 0:
            gl.atomic_max(flags + 2, 1)
        gl.barrier()
        gl.store(main_count + pid * N_SLOTS + sl, s_cnt.load(LV), mask=sl < N_SLOTS)  # the counts back to the ring
    gl.store(wmark_ptr + pid, wmark)
    gl.barrier()  # this program's ring entries are complete before it delivers
    if TRACE:
        n_ev = mark(trace, n_ev, 2, total, TRACE_CAP)

    # ---- 2. delivery (naer2/greenfield/update.py)
    current = gl.load(current_ptr + offs, mask=mask)
    if TRACE:
        current = pin(current)
        n_ev = mark(trace, n_ev, 1, 0, TRACE_CAP)
    if COUNT:
        received = gl.load(received_ptr + offs, mask=mask, other=0.0)
        n_received = gl.load(n_received_ptr + offs, mask=mask, other=0)
    else:
        received = gl.zeros_like(current)
        n_received = gl.zeros_like(lane)
    log_e = (log_ptr + 4 * log_cap + 1).to(gl.pointer_type(gl.float32))
    # the stream block's fields, staged flat (the walk reads them as [STREAM // U, U]), and the
    # connections' attribute pointers (weights by link index), a read-only table
    cq = gl.arange(0, MAX_CONNS, layout=LV)
    s_atp = gl.allocate_shared_memory(gl.int64, [MAX_CONNS], SM)
    s_atp.store(gl.load(attr_addr + cq, mask=cq < n_conns, other=0))
    s_tgt = gl.allocate_shared_memory(gl.int32, [STREAM], SM)
    s_w = gl.allocate_shared_memory(gl.float32, [STREAM], SM)
    if LOG:
        s_src = gl.allocate_shared_memory(gl.int32, [STREAM], SM)
        s_e = gl.allocate_shared_memory(gl.float32, [STREAM], SM)
    else:  # placeholders, never touched
        s_src = s_tgt
        s_e = s_w
    r_tgt = s_tgt.reinterpret(gl.int32, [STREAM // U, U], SM)
    r_w = s_w.reinterpret(gl.float32, [STREAM // U, U], SM)
    r_src = s_src.reinterpret(gl.int32, [STREAM // U, U], SM)
    r_e = s_e.reinterpret(gl.float32, [STREAM // U, U], SM)
    gl.barrier()
    for k in range(k_lo, k_hi + 1):
        cell = pid * N_SLOTS + k % N_SLOTS
        nm = gl.minimum(gl.load(main_count + cell), MAIN_CAP)
        ns = gl.minimum(gl.load(side_count + cell), SIDE_CAP)
        if TRACE:
            nm = pin_s(nm)
            n_ev = mark(trace, n_ev, 3, nm, TRACE_CAP)
        if ns == 0:
            # the slot's link records are its stream: STREAM links at a time
            for b0 in range(0, nm, STREAM):
                nq = _stage_records(main_src - 1, cell, b0, nm, n_cells, s_atp, s_tgt, s_w, s_src, s_e,
                                    STREAM, NUM_WARPS, LOG)
                if TRACE:
                    n_ev = mark(trace, n_ev, 5, nq, TRACE_CAP)
                current, received, n_received = _walk(
                    r_tgt, r_w, r_src, r_e, nq, current, received, n_received, lane, mask, ui, k, pid,
                    log_ptr, log_e, log_cap, GROUP_SIZE, U, LU, LOG, COUNT)
                if TRACE:
                    current = pin(current)
                    n_ev = mark(trace, n_ev, 6, nq, TRACE_CAP)
                gl.barrier()
        else:
            current, received, n_received = _merge_links(
                main_time, main_src, main_lo, main_n, side_time, side_src, side_lo, side_n, cell, nm, ns, n_cells,
                attr_addr, current, received, n_received, lane, mask, k, pid,
                log_ptr, log_e, log_cap, GROUP_SIZE, LOG, COUNT)
        gl.store(main_count + cell, 0)
        gl.store(side_count + cell, 0)
    if COUNT:
        gl.store(received_ptr + offs, received, mask=mask)
        gl.store(n_received_ptr + offs, n_received, mask=mask)
    if TRACE:
        n_ev = mark(trace, n_ev, 7, 0, TRACE_CAP)

    # ---- 3. propagation (PD14), stand-in send, emission into the history
    t_ref_end = gl.load(t_ref_end_ptr + offs, mask=mask)
    potential = gl.load(potential_ptr + offs, mask=mask)
    last_time = gl.load(last_time_ptr + offs, mask=mask)
    time = t_end
    dt = time - last_time
    last_time = gl.zeros_like(last_time) + time
    P_V = gl.exp(-dt / 10.0)
    P_I = gl.exp(-dt / 0.5)
    P_VI = 40.0 * 0.5 / (10.0 - 0.5) * (P_V - P_I)
    _m0 = mask & (time < t_ref_end)
    potential = gl.where(_m0, -65.0, potential)
    potential = gl.where(~_m0 & mask, -38.30576 + P_V * (potential - -38.30576) + P_VI * current, potential)
    current = P_I * current

    fire = gl.load(fire_time_ptr + offs, mask=mask, other=float("inf"))
    fires = mask & (fire >= t_start) & (fire < t_end)
    if UNIFORM_TIME:
        e = gl.zeros_like(fire) + t_start
    else:
        e = fire + gl.load(emit_delay_ptr + offs, mask=mask, other=0.0)
    _n_fire = fires.to(gl.int32)
    _local_pos = gl.convert_layout(gl.cumsum(gl.convert_layout(_n_fire, LS), axis=0), LN) - _n_fire
    n_emit = gl.sum(_n_fire, axis=0)
    potential = gl.where(fires, -65.0, potential)
    t_ref_end = gl.where(fires, time + 2.0, t_ref_end)
    gl.store(t_ref_end_ptr + offs, t_ref_end, mask=mask)
    gl.store(potential_ptr + offs, potential, mask=mask)
    gl.store(current_ptr + offs, current, mask=mask)
    gl.store(last_time_ptr + offs, last_time, mask=mask)
    if TRACE:
        n_ev = mark(trace, n_ev, 8, 0, TRACE_CAP)

    slot = run_id % K
    (h_tag, h_count, h_wend, h_min, h_node, h_time, h_overflow) = history_ptrs(history, slot, K, hist_cap)
    base = gl.atomic_add(h_count, n_emit)  # this program's place in the run's flat list
    put = fires & (base + _local_pos < hist_cap)
    if base + n_emit > hist_cap:
        gl.atomic_max(h_overflow, 1)
    gl.store(h_node + base + _local_pos, offs, mask=put)
    gl.store(h_time + base + _local_pos, e, mask=put)
    gl.atomic_min(h_min, gl.min(gl.where(put, e, float("inf")), axis=0))
    if pid == 0:
        gl.store(h_wend, t_end)
        gl.store(h_tag, history_tag(run_id, round_id))  # one store: never a new run id with an old round
        (n_tag, n_count, _nw, n_min, _nn, _nt, _no) = history_ptrs(history, (run_id + 1) % K, K, hist_cap)
        gl.store(n_tag, history_tag(0, round_id))  # the next run's slot, cleared ahead (see history.py): a
        gl.store(n_count, 0)  # consumer that still needed it sees a gap in run ids and raises its overrun flag
        gl.store(n_min, float("inf"))
    if TRACE:
        n_ev = mark(trace, n_ev, 9, 0, TRACE_CAP)
        trace_end(trace, n_ev)


# ------------------------------------------------- host side


def _direct_launch(kernel, n_programs, consts):
    """A compiled fused kernel's launcher, called directly: the JIT's
    dispatch -- binding, per-tensor specialization (alignment; on AMD whether
    the buffer lies within 2 GB), the cache key -- costs ~45 us per launch
    with this signature against ~5 us for the launch itself
    (tools/experiments/launch_cost.py). The caller keys the cache by
    everything that selects the compiled variant and drops it when its
    buffers change; args: the runtime arguments in signature order."""
    names = fused_kernel.arg_names
    n_rt = len(names) - len(consts)
    assert all(p.is_constexpr for p in fused_kernel.params[n_rt:]), "constexprs must come last"
    run, function, packed = kernel.run, kernel.function, kernel.packed_metadata
    get_stream = triton.runtime.driver.active.get_current_stream
    device = torch.cuda.current_device()

    def launch(args, stream=None):
        stream = stream if stream is not None else get_stream(device)
        enter, leave = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
        meta = kernel.launch_metadata((n_programs, 1, 1), stream, *args, *consts) if enter or leave else None
        run(n_programs, 1, 1, stream, function, packed, meta, enter, leave, *args, *consts)

    return launch


class FusedPop:
    """A green-field population on the fused kernel: PD14-like state, its
    EmissionHistory, and with connect() its rings, per-group watermarks and
    FusedPlan. run() is the whole run, intake included."""

    def __init__(self, n: int, group_size: int = 256, depth: int = 3, device="cuda"):
        self.n, self.group_size, self.device = n, group_size, torch.device(device)
        self.n_groups = -(-n // group_size)
        self.potential = torch.full((n,), -65.0, device=device)
        self.current = torch.zeros(n, device=device)
        self.t_ref_end = torch.zeros(n, device=device)
        self.last_time = torch.zeros(n, device=device)
        self.history = EmissionHistory(n, depth, device)
        self.run_id = 0
        self.received = torch.zeros(n, device=device)
        self.n_received = torch.zeros(n, device=device, dtype=torch.int32)
        self.stream, self.unroll = WALK["stream"], WALK["unroll"]
        self.count_received = True
        self.waves_per_eu = 0
        # pull intake (uniform emission): all fresh runs of one emission time as one spike stream
        # (else run by run), segments found through the group directory (else binary search)
        self.batch, self.use_directory = PULL["batch"], PULL["directory"]
        self.trace = None
        self.overrun = torch.zeros(1, device=device, dtype=torch.int32)
        self._no_delay = torch.zeros(n, device=device)
        self._launchers = {}  # compiled kernel per configuration, launched directly (see _direct_launch)
        self.connect([], [], n_slots=1)

    def connect(self, sources: list, topologies: list, n_slots: int, slot_width: float = 0.1, log_cap: int = 0,
                main_cap: int | None = None, side_cap: int | None = None):
        caps = {k: v for k, v in (("main_cap", main_cap), ("side_cap", side_cap)) if v}
        self.ring = Ring(self.n_groups, n_slots, slot_width, self.device, **caps)
        self.wmark = torch.full((self.n_groups,), -math.inf, device=self.device)
        self.plan = FusedPlan([p.history for p in sources], topologies, self.n_groups, self.device)
        self.log_cap = log_cap
        self.log = torch.zeros(1 + 5 * max(log_cap, 1), device=self.device, dtype=torch.int32)
        self._launchers = {}  # new buffers (their sizes are part of the AMD backend's specialization)

    def run(self, t_start, t_end, fire_time, emit_delay=None, uniform=True, deliver_until=None, round_id=None,
            num_warps=4, stream=None):
        """One run over [t_start, t_end) in round `round_id` (default: the
        run id): take in what the sources emitted in earlier rounds, deliver
        every ring slot that ends by `deliver_until` (default t_end), emit.
        stream: a torch.cuda.Stream to launch on (default: the current one;
        passing it saves switching the current stream, ~5 us)."""
        self.run_id += 1
        w = self.ring.slot_width
        k_lo = self.ring.ring_pos
        k_hi = int(((deliver_until if deliver_until is not None else t_end) + 1e-6) // w) - 1
        fire_time = fire_time.to(self.device)
        emit_delay = emit_delay.to(self.device) if emit_delay is not None else self._no_delay
        lanes = 32 * num_warps
        args = (
            self.t_ref_end, self.potential, self.current, self.last_time, self.n,
            self.plan.plan, self.plan.n_conns, self.ring.buf, self.wmark, self.overrun, k_lo, k_hi, w,
            round_id if round_id is not None else self.run_id, self.run_id % 2,
            self.received, self.n_received, self.log, self.log_cap,
            fire_time, emit_delay,
            self.history.buf, self.history.cap, self.run_id, t_start, t_end,
            self.trace.buf if self.trace is not None else self.log,
        )
        # what selects the compiled variant, besides the plan, ring and history (fixed between connect()s)
        key = (uniform, num_warps, self.waves_per_eu, self.count_received, self.batch, self.use_directory,
               self.stream, self.unroll, self.log_cap > 0, fire_time.dtype, emit_delay.dtype)
        launch = self._launchers.get(key)
        aligned = not (fire_time.data_ptr() | emit_delay.data_ptr()) & 15  # the specialization every call meets
        if launch is not None and aligned:
            launch(args, stream.cuda_stream if stream is not None else None)
            return self._advance(k_lo, k_hi)
        p2 = triton.next_power_of_2
        consts = dict(
            GROUP_SIZE=self.group_size, K=self.history.K, SK=self.plan.K, KP=p2(self.plan.K),
            UNIFORM_TIME=uniform,
            N_SLOTS=self.ring.n_slots, MAIN_CAP=self.ring.main_cap, SIDE_CAP=self.ring.side_cap,
            LOG=self.log_cap > 0, STREAM=self.stream, U=self.unroll,
            SPT=max(self.group_size // lanes, 1), NUM_WARPS=num_warps,
            MAX_CONNS=p2(max(self.plan.n_conns, 1)),
            LOG_CJ=(p2(max(self.plan.n_conns, 1)) * p2(self.plan.K)).bit_length() - 1,
            ITERS=self.plan.iters,
            COUNT=self.count_received, BATCH=self.batch, DIR=self.use_directory,
            NSP=p2(self.ring.n_slots),
            TRACE=self.trace.level if self.trace is not None else 0,
            TRACE_CAP=self.trace.cap if self.trace is not None else 1,
        )
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            kernel = fused_kernel[(self.n_groups,)](*args, **consts, num_warps=num_warps,
                                                    waves_per_eu=self.waves_per_eu)
        if aligned:
            self._launchers[key] = _direct_launch(kernel, self.n_groups, tuple(consts.values()))
        self._advance(k_lo, k_hi)

    def _advance(self, k_lo, k_hi):
        self.ring.ring_pos = max(k_lo, k_hi + 1)

    def check(self) -> None:
        self.ring.check()
        if int(self.overrun[0]):
            raise RuntimeError("a consumer fell behind a source's emission history: a run was overwritten "
                               "before it was read (raise the history depth K)")

    # ---- inspection

    def set_trace(self, cap: int | None = 1024, level: int = 1):
        """Stamp the following runs (naer2/trace.py); level 2 also inside the pull; None: stop."""
        self.trace = Trace(self.n_groups, cap, self.device, level=level) if cap else None
        self._launchers = {}

    def traced(self) -> list[dict]:
        return self.trace.decode(FUSED_TRACE_TAGS)

    def deliveries(self) -> list[tuple[int, float, int, int, int]]:
        """The delivery log, in log order: (slot, e, connection, source, target)."""
        n = min(int(self.log[0]), self.log_cap)
        rec = self.log[1:1 + 4 * self.log_cap].view(-1, 4)[:n].tolist()
        e = self.log[1 + 4 * self.log_cap:].view(torch.float32)[:n].tolist()
        return [(k, ei, c, s, t) for (k, c, s, t), ei in zip(rec, e)]

    def emitted(self) -> list[tuple[float, int]]:
        """The last run's spikes, [(e, node)] in flat-list order."""
        return self.history.spikes(self.run_id)
