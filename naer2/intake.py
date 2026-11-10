"""
Intake: once per round, every consumer population takes in what its sources
emitted into its rings of arrival slots (see Population._intake for the
host-side scheduling part, Network.step for when it runs, TODO.md for the
design and sizing).

Rings: one per target GROUP (the neurons one update-kernel program owns).
Slot k of a group's ring covers arrivals in [k * slot_width, (k + 1) *
slot_width) and lives at index k % N_SLOTS. Each slot holds
  - a main run: entries appended in emission order (no work), and
  - a side run: out-of-order entries (emission-delayed, or below the
    watermark), kept sorted by emission time by insertion.
An entry is one SEGMENT of one spike (see naer2/topology_new.py): emission
time e, (connection << 24) | source neuron, segment index. A spike from
source s expands into one entry per segment of s -- exactly the (group,
bucket) pieces s has links in -- each into its group's slot of arrival
e + mean delay of the segment's bucket. Delivery then walks the slot's
entries, each a contiguous range of the group's own links, with no lookup.
An entry goes to the main runs if e >= the watermark (largest e ever
appended to a main run, one per consumer) and it isn't emission-delayed (e
before its source's run window end); otherwise to the side runs.

Two kernels:
  intake_kernel (one program, any emission times): gather the active
    sources' emission buffers into one batch (each source group's entries
    are sorted -- emission contract, see naer2/emission.py -- so the batch
    is a set of sorted runs), merge the runs pairwise bottom-up (log2(runs)
    passes of a binary-search rank merge, never a sort from scratch), then
    insert in e order.
  intake_uniform_kernel (one program per chunk of source groups): the host
    knows every entry of the round has the same emission time (all active
    sources emit bare n2.spike()). Order is then free and every segment
    entry is inserted in parallel with atomics, straight from the emission
    buffers. Such entries never need a side run; one that would sets flag 2.

An arrival before the ring position (already delivered) goes into the first
undelivered slot (late, the timestep quantization); one at or past ring
position + N_SLOTS is far future (flag 1, not implemented yet); a full run
sets flag 0.

All arguments are packed into three int32 buffers (rings, plan, scratch);
the kernels compute typed pointers into them. Nothing is copied per round:
intake decides on the device which sources ran since it last read them
(their run ids vs its last-read ids, double-buffered by `parity` so every
program of a launch reads the same values). A launch only takes scalars.
"""

import torch
import triton
import triton.language as tl

from .emission import EmissionBuffer, emission_ptrs
from .spike_gather import _bit_length
from .topology_new import TopologyNew, topology_ptrs
from .trace import mark, pin_s, trace_begin, trace_end

# TODO: run capacities are fixed constants for now; a burst (e.g. PD14's
# 902-spike thalamic volley) lands in every slot of its delay range at once.
# They should become population-specific and adjust themselves on overflow.
DEFAULT_MAIN_CAP = 2048  # PD14 at group size 512: ~280 entries per (group, slot)
DEFAULT_SIDE_CAP = 512
_BLOCK = 128      # vector width of the general kernel
_BLOCK_G = 1024   # source groups per gather chunk (general kernel)
_BLOCK_B = 1      # source groups per program (uniform kernel): a group's few spikes per program
_BLOCK_S = 512    # segments per insertion tile: ~one tile per spike (PD14: ~450 segments)
_N_FLAGS = 3      # run overflow, far future, uniform contract violated


# ------------------------------------------------------------ packed layouts


@triton.jit
def _f32(p):
    return p.to(tl.pointer_type(tl.float32))


@triton.jit
def _ring(ring, n_groups, N_SLOTS: tl.constexpr, MAIN_CAP: tl.constexpr, SIDE_CAP: tl.constexpr):
    """Typed pointers into a Ring buffer, see Ring for the layout."""
    cells = n_groups * N_SLOTS
    main_time = ring
    main_src = main_time + cells * MAIN_CAP
    main_seg = main_src + cells * MAIN_CAP
    side_time = main_seg + cells * MAIN_CAP
    side_src = side_time + cells * SIDE_CAP
    side_seg = side_src + cells * SIDE_CAP
    main_count = side_seg + cells * SIDE_CAP
    side_count = main_count + cells
    watermark = side_count + cells
    return (
        _f32(main_time), main_src, main_seg, main_count,
        _f32(side_time), side_src, side_seg, side_count,
        _f32(watermark), watermark + 1,
    )


@triton.jit
def _plan(plan, n_conns):
    """Typed pointers into a consumer's plan buffer, see IntakePlan."""
    addr = plan.to(tl.pointer_type(tl.int64))
    base = plan + 10 * n_conns
    return (
        addr, addr + n_conns,                                  # emission buffer, topology addresses
        addr + 2 * n_conns, addr + 3 * n_conns, addr + 4 * n_conns,  # seg_start, link_target, attrs addresses
        base, base + n_conns,                                  # source n_groups, cap
        base + 2 * n_conns, base + 3 * n_conns,                # topology n_sources, n_segments
        base + 4 * n_conns, base + 5 * n_conns,                # topology n_buckets, n_links
        base + 6 * n_conns,                                    # last-read run ids [2, n_conns]
        base + 8 * n_conns,                                    # uniform programs: [conn | first group]
    )


@triton.jit
def _source(em_addr, em_n_groups, em_cap, last_read, c, n_conns, parity):
    """Connection c's source emission buffer (see naer2/emission.py): node,
    time and count pointers, its group count and cap, its last run's window
    end, and whether it ran since this consumer last read it. Reads
    last_read[parity], writes the new value to last_read[1 - parity]
    (idempotent across programs)."""
    n_groups = tl.load(em_n_groups + c)
    cap = tl.load(em_cap + c)
    buf = tl.load(em_addr + c).to(tl.pointer_type(tl.int32))
    node, time, count, run_id_p, window_end_p, _sc, _sm, _of = emission_ptrs(buf, n_groups, cap)
    run_id = tl.load(run_id_p)
    fresh = run_id != tl.load(last_read + parity * n_conns + c)
    tl.store(last_read + (1 - parity) * n_conns + c, run_id)
    return node, time, count, n_groups, cap, tl.load(window_end_p), fresh


@triton.jit
def _topology(topo_addr, t_n_sources, t_n_segments, t_n_buckets, t_n_links, c):
    """Connection c's segment offsets, segment (group, bucket) words and
    bucket mean delays (see naer2/topology_new.py)."""
    buf = tl.load(topo_addr + c).to(tl.pointer_type(tl.int32))
    seg_offsets, _start, seg_gb, bucket_mean, _target, _attrs = topology_ptrs(
        buf, tl.load(t_n_sources + c), tl.load(t_n_segments + c),
        tl.load(t_n_buckets + c), tl.load(t_n_links + c),
    )
    return seg_offsets, seg_gb, bucket_mean


# ------------------------------------------------------------ insertion


@triton.jit
def _insert_main(
    e, srcword, seg_lo, seg_hi, seg_gb, bucket_mean,
    main_time, main_src, main_seg, main_count, flags, ring_pos, slot_width,
    N_SLOTS: tl.constexpr, MAIN_CAP: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """Append one entry per segment [seg_lo, seg_hi) of a spike to its
    group's main run at its arrival slot, BLOCK_S segments at a time."""
    for t0 in range(seg_lo, seg_hi, BLOCK_S):
        seg = t0 + tl.arange(0, BLOCK_S)
        m = seg < seg_hi
        gb = tl.load(seg_gb + seg, mask=m, other=0)
        mean = tl.load(bucket_mean + (gb & 0xFFFF), mask=m, other=0.0)
        k = tl.maximum(tl.floor((e + mean) / slot_width).to(tl.int32), ring_pos)
        far = m & (k - ring_pos >= N_SLOTS)
        tl.atomic_max(flags + 1, tl.max(far.to(tl.int32), axis=0))
        w = m & ~far
        cell = ((gb >> 16) & 0xFFFF) * N_SLOTS + k % N_SLOTS
        pos = tl.atomic_add(main_count + cell, 1, mask=w)
        full = w & (pos >= MAIN_CAP)
        tl.atomic_max(flags, tl.max(full.to(tl.int32), axis=0))
        w = w & ~full
        at = cell * MAIN_CAP + pos
        tl.store(main_time + at, e + tl.zeros_like(mean), mask=w)
        tl.store(main_src + at, srcword + tl.zeros_like(seg), mask=w)
        tl.store(main_seg + at, seg, mask=w)


@triton.jit
def _insert_side(
    e, srcword, seg_lo, seg_hi, seg_gb, bucket_mean,
    side_time, side_src, side_seg, side_count, flags, ring_pos, slot_width,
    N_SLOTS: tl.constexpr, SIDE_CAP: tl.constexpr, BLOCK: tl.constexpr,
):
    """Insert one entry per segment of a spike into its group's side run at
    its arrival slot, keeping the run sorted by e (stable: after every
    entry with time <= e). Sequential; barriers make each insertion visible
    to the next."""
    lanes = tl.arange(0, BLOCK)
    for seg in range(seg_lo, seg_hi):
        gb = tl.load(seg_gb + seg)
        k = tl.maximum(tl.floor((e + tl.load(bucket_mean + (gb & 0xFFFF))) / slot_width).to(tl.int32), ring_pos)
        if k - ring_pos >= N_SLOTS:
            tl.atomic_max(flags + 1, 1)
        else:
            cell = ((gb >> 16) & 0xFFFF) * N_SLOTS + k % N_SLOTS
            base = cell * SIDE_CAP
            cnt = tl.load(side_count + cell)
            if cnt >= SIDE_CAP:
                tl.atomic_max(flags, 1)
            else:
                p = cnt
                for s0 in range(0, cnt, BLOCK):
                    q = s0 + lanes
                    v = tl.load(side_time + base + q, mask=q < cnt, other=0.0)
                    p = tl.minimum(p, tl.min(tl.where((q < cnt) & (v > e), q, cnt), axis=0))
                tl.debug_barrier()
                for s0 in range(0, cnt - p, BLOCK):  # shift [p, cnt) up by one, top first
                    q = cnt - 1 - s0 - lanes
                    qm = q >= p
                    tv = tl.load(side_time + base + q, mask=qm)
                    sv = tl.load(side_src + base + q, mask=qm)
                    gv = tl.load(side_seg + base + q, mask=qm)
                    tl.store(side_time + base + q + 1, tv, mask=qm)
                    tl.store(side_src + base + q + 1, sv, mask=qm)
                    tl.store(side_seg + base + q + 1, gv, mask=qm)
                    tl.debug_barrier()
                tl.store(side_time + base + p, e)
                tl.store(side_src + base + p, srcword)
                tl.store(side_seg + base + p, seg)
                tl.store(side_count + cell, cnt + 1)
        tl.debug_barrier()


# ------------------------------------------------------------ general kernel


@triton.jit
def _bsearch(time_ptr, lo, hi, x, iters, STRICT):
    """Per lane: number of entries in the sorted time_ptr[lo:hi] that are
    < x (STRICT, per lane) or <= x."""
    count = tl.zeros_like(lo)
    for _ in range(iters):
        active = lo < hi
        mid = (lo + hi) // 2
        v = tl.load(time_ptr + mid, mask=active, other=0.0)
        right = tl.where(STRICT, v < x, v <= x)
        count = tl.where(active & right, count + (mid + 1 - lo), count)
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    return count


@triton.jit
def _scratch(s, cap):
    return (
        _f32(s), s + cap, s + 2 * cap, s + 3 * cap,
        _f32(s + 4 * cap), s + 5 * cap, s + 6 * cap, s + 7 * cap,
        s + 8 * cap, s + 9 * cap + 1, _f32(s + 10 * cap + 2),
    )


@triton.jit(do_not_specialize=["n_conns", "n_groups", "batch_cap", "ring_pos", "parity"])
def intake_kernel(
    plan, scratch, ring, n_conns, n_groups, batch_cap, ring_pos, parity, slot_width,
    N_SLOTS: tl.constexpr, MAIN_CAP: tl.constexpr, SIDE_CAP: tl.constexpr,
    BLOCK: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_S: tl.constexpr,
):
    (em_addr, topo_addr, _ssa, _lta, _ata, em_n_groups, em_cap, t_n_sources, t_n_segments,
     t_n_buckets, t_n_links, last_read, _progs) = _plan(plan, n_conns)
    (a_time, a_node, a_conn, a_run, b_time, b_node, b_conn, b_run,
     run_start, run_tmp, window_end) = _scratch(scratch, batch_cap)
    (main_time, main_src, main_seg, main_count, side_time, side_src, side_seg, side_count,
     watermark, flags) = _ring(ring, n_groups, N_SLOTS, MAIN_CAP, SIDE_CAP)
    lanes = tl.arange(0, BLOCK)

    # 1. gather: every nonempty source group becomes one sorted run
    n = tl.cast(0, tl.int32)
    n_runs = tl.cast(0, tl.int32)
    for c in range(n_conns):
        node_p, time_p, count_p, src_groups, cap, wend, fresh = _source(
            em_addr, em_n_groups, em_cap, last_read, c, n_conns, parity
        )
        tl.store(window_end + c, wend)
        if fresh:
            for b0 in range(0, src_groups, BLOCK_G):
                grp = b0 + tl.arange(0, BLOCK_G)
                bm = grp < src_groups
                cnt = tl.minimum(tl.load(count_p + grp, mask=bm, other=0), cap)
                dst = n + tl.cumsum(cnt, axis=0) - cnt
                nonempty = cnt > 0
                rid = n_runs + tl.cumsum(nonempty.to(tl.int32), axis=0) - nonempty.to(tl.int32)
                tl.store(run_start + rid, dst, mask=nonempty)
                for l in range(tl.max(cnt, axis=0)):
                    lm = bm & (l < cnt)
                    tl.store(a_time + dst + l, tl.load(time_p + grp * cap + l, mask=lm), mask=lm)
                    tl.store(a_node + dst + l, tl.load(node_p + grp * cap + l, mask=lm), mask=lm)
                    tl.store(a_conn + dst + l, c + tl.zeros_like(grp), mask=lm)
                    tl.store(a_run + dst + l, rid, mask=lm)
                n += tl.sum(cnt, axis=0)
                n_runs += tl.sum(nonempty.to(tl.int32), axis=0)
    tl.store(run_start + n_runs, n)
    tl.debug_barrier()

    # 2. merge the sorted runs pairwise until one remains
    iters = _bit_length(n)
    while n_runs > 1:
        for s0 in range(0, n, BLOCK):
            i = s0 + lanes
            m = i < n
            rid = tl.load(a_run + i, mask=m, other=0)
            t = tl.load(a_time + i, mask=m, other=0.0)
            partner = rid ^ 1
            has_p = m & (partner < n_runs)
            own_start = tl.load(run_start + rid, mask=m, other=0)
            pair_start = tl.load(run_start + (rid & ~1), mask=m, other=0)
            p_lo = tl.load(run_start + partner, mask=has_p, other=0)
            p_hi = tl.load(run_start + partner + 1, mask=has_p, other=0)
            # stable: on equal e, the left run of a pair goes first
            rank = _bsearch(a_time, p_lo, p_hi, t, iters, (rid & 1) == 0)
            pos = pair_start + (i - own_start) + tl.where(has_p, rank, 0)
            tl.store(b_time + pos, t, mask=m)
            tl.store(b_node + pos, tl.load(a_node + i, mask=m), mask=m)
            tl.store(b_conn + pos, tl.load(a_conn + i, mask=m), mask=m)
            tl.store(b_run + pos, rid >> 1, mask=m)
        tl.debug_barrier()
        n_runs = (n_runs + 1) // 2
        for s0 in range(0, n_runs, BLOCK):
            k = s0 + lanes
            km = k < n_runs
            tl.store(run_tmp + k, tl.load(run_start + 2 * k, mask=km), mask=km)
        tl.debug_barrier()
        for s0 in range(0, n_runs, BLOCK):
            k = s0 + lanes
            km = k < n_runs
            tl.store(run_start + k, tl.load(run_tmp + k, mask=km), mask=km)
        tl.store(run_start + n_runs, n)
        for s0 in range(0, n, BLOCK):
            i = s0 + lanes
            m = i < n
            tl.store(a_time + i, tl.load(b_time + i, mask=m), mask=m)
            tl.store(a_node + i, tl.load(b_node + i, mask=m), mask=m)
            tl.store(a_conn + i, tl.load(b_conn + i, mask=m), mask=m)
            tl.store(a_run + i, tl.load(b_run + i, mask=m), mask=m)
        tl.debug_barrier()

    # 3. insert in e order, one entry per segment of each spike
    mark = tl.load(watermark)
    for i in range(n):
        e = tl.load(a_time + i)
        s = tl.load(a_node + i)
        c = tl.load(a_conn + i)
        to_main = (e >= mark) & (e < tl.load(window_end + c))
        mark = tl.where(to_main, e, mark)
        seg_offsets, seg_gb, bucket_mean = _topology(
            topo_addr, t_n_sources, t_n_segments, t_n_buckets, t_n_links, c
        )
        seg_lo = tl.load(seg_offsets + s)
        seg_hi = tl.load(seg_offsets + s + 1)
        if to_main:
            _insert_main(e, (c << 24) | s, seg_lo, seg_hi, seg_gb, bucket_mean,
                         main_time, main_src, main_seg, main_count, flags, ring_pos, slot_width,
                         N_SLOTS, MAIN_CAP, BLOCK_S)
        else:
            _insert_side(e, (c << 24) | s, seg_lo, seg_hi, seg_gb, bucket_mean,
                         side_time, side_src, side_seg, side_count, flags, ring_pos, slot_width,
                         N_SLOTS, SIDE_CAP, BLOCK)
    tl.store(watermark, mark)


# ------------------------------------------------------------ uniform kernel


INTAKE_TRACE_TAGS = ("start", "source", "spike", "inserted", "end")


@triton.jit(do_not_specialize=["n_conns", "n_groups", "ring_pos", "parity"])
def intake_uniform_kernel(
    plan, ring, n_conns, n_groups, ring_pos, parity, slot_width, trace_ptr,
    N_SLOTS: tl.constexpr, MAIN_CAP: tl.constexpr, SIDE_CAP: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_S: tl.constexpr,
    TRACE: tl.constexpr = 0, TRACE_CAP: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    n_ev = tl.cast(0, tl.int32)
    if TRACE:  # see naer2/trace.py; tags INTAKE_TRACE_TAGS
        trace = trace_begin(trace_ptr, pid, TRACE_CAP)
        n_ev = mark(trace, n_ev, 0, 0, TRACE_CAP)
    (em_addr, topo_addr, _ssa, _lta, _ata, em_n_groups, em_cap, t_n_sources, t_n_segments,
     t_n_buckets, t_n_links, last_read, progs) = _plan(plan, n_conns)
    (main_time, main_src, main_seg, main_count, _st, _ss, _sg, _sc,
     watermark, flags) = _ring(ring, n_groups, N_SLOTS, MAIN_CAP, SIDE_CAP)

    c = tl.load(progs + pid)
    node_p, time_p, count_p, src_groups, cap, wend, fresh = _source(
        em_addr, em_n_groups, em_cap, last_read, c, n_conns, parity
    )
    if fresh:
        lanes = tl.arange(0, BLOCK_B)
        grp = tl.load(progs + tl.num_programs(0) + pid) + lanes
        cnt = tl.minimum(tl.load(count_p + grp, mask=grp < src_groups, other=0), cap)
        if TRACE:
            n_ev = mark(trace, n_ev, 1, pin_s(tl.max(cnt, axis=0)), TRACE_CAP)
        if tl.max(cnt, axis=0) > 0:
            seg_offsets, seg_gb, bucket_mean = _topology(
                topo_addr, t_n_sources, t_n_segments, t_n_buckets, t_n_links, c
            )
            wmark = tl.load(watermark)
            for b in range(BLOCK_B):
                g_src = tl.sum(tl.where(lanes == b, grp, 0), axis=0)
                for l in range(tl.sum(tl.where(lanes == b, cnt, 0), axis=0)):
                    e = tl.load(time_p + g_src * cap + l)
                    s = tl.load(node_p + g_src * cap + l)
                    main = (e >= wmark) & (e < wend)
                    if main:
                        seg_lo = tl.load(seg_offsets + s)
                        seg_hi = tl.load(seg_offsets + s + 1)
                        if TRACE:
                            seg_hi = pin_s(seg_hi)
                            n_ev = mark(trace, n_ev, 2, seg_hi - seg_lo, TRACE_CAP)
                        _insert_main(e, (c << 24) | s, seg_lo, seg_hi,
                                     seg_gb, bucket_mean, main_time, main_src, main_seg, main_count,
                                     flags, ring_pos, slot_width, N_SLOTS, MAIN_CAP, BLOCK_S)
                        tl.atomic_max(watermark, e)
                        if TRACE:
                            n_ev = mark(trace, n_ev, 3, seg_hi - seg_lo, TRACE_CAP)
                    else:
                        tl.atomic_max(flags + 2, 1)
    if TRACE:
        n_ev = mark(trace, n_ev, 4, 0, TRACE_CAP)
        trace_end(trace, n_ev)


# ------------------------------------------------------------ host side


class Ring:
    """One consumer's rings of arrival slots, one ring per target group
    (see the module docstring), as one int32 buffer: per (group, slot) cell
    a main run (time, (conn << 24) | source, segment; MAIN_CAP each) and a
    side run (SIDE_CAP each), then main counts, side counts, the watermark
    (float32) and the flags. Cell of (group g, slot k): g * n_slots +
    k % n_slots."""

    def __init__(self, n_groups: int, n_slots: int, slot_width: float, device,
                 main_cap: int = DEFAULT_MAIN_CAP, side_cap: int = DEFAULT_SIDE_CAP):
        self.n_groups, self.n_slots, self.slot_width = n_groups, n_slots, slot_width
        self.main_cap, self.side_cap = main_cap, side_cap
        cells = n_groups * n_slots
        m, s = cells * main_cap, cells * side_cap
        self.buf = torch.zeros(3 * m + 3 * s + 2 * cells + 1 + _N_FLAGS, device=device, dtype=torch.int32)
        self.main = (self.buf[:m].view(torch.float32), self.buf[m:2 * m], self.buf[2 * m:3 * m])
        o = 3 * m
        self.side = (self.buf[o:o + s].view(torch.float32), self.buf[o + s:o + 2 * s], self.buf[o + 2 * s:o + 3 * s])
        o += 3 * s
        self.main_count = self.buf[o:o + cells]
        self.side_count = self.buf[o + cells:o + 2 * cells]
        self.watermark = self.buf[o + 2 * cells:o + 2 * cells + 1].view(torch.float32)
        self.watermark.fill_(float("-inf"))
        self.flags = self.buf[o + 2 * cells + 1:]
        self.ring_pos = 0  # first slot not yet delivered

    def cell(self, group: int, slot: int) -> int:
        return group * self.n_slots + slot % self.n_slots

    def check(self) -> None:
        overflow, far, uniform = self.flags.tolist()
        if overflow:
            raise RuntimeError(
                f"a ring run exceeded its capacity (main {self.main_cap}, side {self.side_cap} entries)"
            )
        if far:
            raise NotImplementedError(
                f"an arrival lies at or beyond {self.n_slots} slots ahead of the ring "
                f"position (e.g. a long emission delay): the far-future list isn't implemented"
            )
        if uniform:
            raise RuntimeError(
                "uniform_time intake got a spike below the watermark or past its "
                "source's run window: the uniform contract (bare n2.spike() only) was violated"
            )


class IntakePlan:
    """Everything about one consumer's intake, built once into one int32
    buffer: per incoming connection its source's EmissionBuffer address, its
    TopologyNew address and the addresses of that topology's seg_start,
    link_target and attribute arrays (int64 each; the update kernel's
    delivery walk reads links through these), the source's group count and cap,
    the topology's source, segment, bucket and link counts, the last-read
    run ids (double-buffered) and the uniform kernel's program table
    (connection, first source group). Emission buffers and topologies must
    stay allocated (they do: one per population / connection, for good).
    run() copies nothing: it launches with scalars only."""

    def __init__(self, ring: Ring, emissions: list[EmissionBuffer], topologies: list[TopologyNew]):
        device = ring.buf.device
        n = len(emissions)
        if n > 256:
            raise ValueError("a consumer can take at most 256 incoming connections (8-bit connection id)")
        for t in topologies:
            if t.n_sources >= 1 << 24:
                raise ValueError("a source population can have at most 2**24 neurons (24-bit source id)")
            if t.n_groups != ring.n_groups:
                raise ValueError("every incoming topology must use the consumer's group layout")
        self.ring, self.n_conns, self.parity = ring, n, 0
        self.emissions, self.topologies = emissions, topologies
        progs = [(c, g0) for c, em in enumerate(emissions) for g0 in range(0, em.n_groups, _BLOCK_B)]
        self.n_progs = max(len(progs), 1)

        plan = torch.zeros(18 * n + 2 * self.n_progs, dtype=torch.int32)
        links = [t.link_addresses() for t in topologies]
        plan[: 10 * n].view(torch.int64)[:] = torch.tensor(
            [em.buf.data_ptr() for em in emissions] + [t.buf.data_ptr() for t in topologies]
            + [a[0] for a in links] + [a[1] for a in links] + [a[2] for a in links],
            dtype=torch.int64,
        )
        for i, values in enumerate((
            [em.n_groups for em in emissions], [em.cap for em in emissions],
            [t.n_sources for t in topologies], [t.n_segments for t in topologies],
            [t.n_buckets for t in topologies], [t.n_links for t in topologies],
        )):
            plan[(10 + i) * n:(11 + i) * n] = torch.tensor(values, dtype=torch.int32)
        table = plan[18 * n:]
        table[: len(progs)] = torch.tensor([c for c, _ in progs], dtype=torch.int32)
        table[self.n_progs:self.n_progs + len(progs)] = torch.tensor([g for _, g in progs], dtype=torch.int32)
        self.plan = plan.to(device)

        self.batch_cap = sum(em.n_groups * em.cap for em in emissions) + 1
        self.scratch = torch.empty(10 * self.batch_cap + 2 + n, device=device, dtype=torch.int32)

    def run(self, uniform_time: bool = False, num_warps: int = 4, trace=None) -> None:
        """trace: a naer2.trace.Trace over n_progs programs (uniform kernel only)."""
        ring = self.ring
        consts = dict(N_SLOTS=ring.n_slots, MAIN_CAP=ring.main_cap, SIDE_CAP=ring.side_cap, BLOCK_S=_BLOCK_S,
                      num_warps=num_warps)
        if uniform_time:
            intake_uniform_kernel[(self.n_progs,)](
                self.plan, ring.buf, self.n_conns, ring.n_groups, ring.ring_pos, self.parity,
                ring.slot_width, trace.buf if trace is not None else ring.buf, BLOCK_B=_BLOCK_B,
                TRACE=int(trace is not None), TRACE_CAP=trace.cap if trace is not None else 1, **consts,
            )
        else:
            intake_kernel[(1,)](
                self.plan, self.scratch, ring.buf, self.n_conns, ring.n_groups, self.batch_cap,
                ring.ring_pos, self.parity, ring.slot_width, BLOCK=_BLOCK, BLOCK_G=_BLOCK_G, **consts,
            )
        self.parity ^= 1
