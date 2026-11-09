"""
Intake: once per round, every consumer population takes in what its sources
emitted into its own ring of arrival slots (see Population._intake for the
host-side scheduling part, Network.step for when it runs, TODO.md item 1c
for the sizing behind this design).

Ring (one per consumer): slot k covers arrivals in [k * slot_width,
(k + 1) * slot_width) and lives at index k % N_SLOTS. Each slot holds
  - a main run: entries appended in emission order (no work), and
  - a side run: out-of-order entries (emission-delayed, or below the
    watermark), kept sorted by emission time by insertion.
An entry is (emission time e, source neuron, connection, delay bucket j);
it belongs to the slot of its arrival e + mean_delay[connection][j]. An
entry goes to the main runs if e >= the watermark (largest e ever appended
to a main run) and it isn't emission-delayed (e before its source's run
window end); otherwise to the side runs.

Two kernels:
  intake_kernel (one program, any emission times):
    1. gather  the active sources' emission buffers into one batch; every
               source block's entries are already sorted by e (emission
               contract, see Population), so the batch is a set of sorted
               runs, one per nonempty block
    2. merge   the runs pairwise, bottom-up (log2(runs) passes of a
               binary-search rank merge) -- a merge sort seeded with sorted
               runs, never a sort from scratch
    3. insert  in e order, one ring entry per delay bucket; since the batch
               is sorted, main runs stay sorted by appending
  intake_uniform_kernel (one program per chunk of source blocks): the host
    knows every entry of the round has the same emission time (all active
    sources emit bare n2.spike(); a round only ever holds one round's
    spikes). Order is then free and every (entry, bucket) pair is inserted
    in parallel with atomics, straight from the emission buffers. Such
    entries can never need a side run; one that would sets flag 2.

An arrival before the ring position (already delivered) goes into the first
undelivered slot (late, the timestep quantization); one at or past ring
position + N_SLOTS is far future (flag 1, not implemented yet); a full slot
sets flag 0. Delivery -- walking main and side of each due slot with two
cursors -- is the update kernel's job.

All arguments are packed into three int32 buffers (ring, plan, scratch);
the kernels compute typed pointers into them, which keeps Triton's launch
overhead (it grows with the argument count) small. Nothing is copied to the
device per round: every source population keeps a device-side state
[run id, window end of its last run] that its own update kernel writes, and
intake decides on the device which sources ran since it last read them
(last-read run ids, double-buffered by the `parity` scalar so that every
program of a launch reads the same values). A launch only takes scalars.
"""

import torch
import triton
import triton.language as tl

from .emission import EmissionBuffer, emission_ptrs
from .spike_gather import _bit_length

# TODO: the slot capacity is a fixed constant for now; a burst (e.g. PD14's
# 902-spike thalamic volley) lands in every slot of its delay range at once.
# It should become population-specific and adjust itself on overflow.
DEFAULT_SLOT_CAP = 32_768  # PD14 x10 with 44 buckets needs ~11k per slot
_BLOCK = 128      # vector width of the general kernel
_BLOCK_G = 1024   # source blocks per gather chunk (general kernel)
_BLOCK_B = 32     # source blocks per program (uniform kernel)
_N_FLAGS = 3      # slot overflow, far future, uniform contract violated


# ------------------------------------------------------------ packed layouts


@triton.jit
def _f32(p):
    return p.to(tl.pointer_type(tl.float32))


@triton.jit
def _ring(ring, N_SLOTS: tl.constexpr, SLOT_CAP: tl.constexpr):
    """Typed pointers into the ring buffer, see Ring for the layout."""
    C: tl.constexpr = N_SLOTS * SLOT_CAP
    main_count = ring + 8 * C
    side_count = main_count + N_SLOTS
    watermark = _f32(side_count + N_SLOTS)
    flags = side_count + N_SLOTS + 1
    return (
        _f32(ring), ring + C, ring + 2 * C, ring + 3 * C, main_count,
        _f32(ring + 4 * C), ring + 5 * C, ring + 6 * C, ring + 7 * C, side_count,
        watermark, flags,
    )


@triton.jit
def _plan(plan, n_conns, MAX_BUCKETS: tl.constexpr):
    """Typed pointers into a consumer's plan buffer, see IntakePlan."""
    base = plan + 2 * n_conns
    return (
        plan.to(tl.pointer_type(tl.int64)),            # emission buffer addresses
        base, base + n_conns, base + 2 * n_conns,      # n_blocks, cap, n_buckets
        base + 3 * n_conns,                            # last-read run ids [2, n_conns]
        _f32(base + 5 * n_conns),                      # bucket means [n_conns, MAX_BUCKETS]
        base + 5 * n_conns + n_conns * MAX_BUCKETS,    # uniform programs: [conn | first block]
    )


@triton.jit
def _source(src_addr, src_n_blocks, src_cap, last_read, c, n_conns, parity):
    """Connection c's source emission buffer (see naer2/emission.py): node,
    time and count pointers, its block count and cap, its last run's window
    end, and whether it ran since this consumer last read it. Reads
    last_read[parity], writes the new value to last_read[1 - parity]
    (idempotent across programs)."""
    n_blocks = tl.load(src_n_blocks + c)
    cap = tl.load(src_cap + c)
    buf = tl.load(src_addr + c).to(tl.pointer_type(tl.int32))
    node, time, count, run_id_p, window_end_p, _sc, _sm, _of = emission_ptrs(buf, n_blocks, cap)
    run_id = tl.load(run_id_p)
    fresh = run_id != tl.load(last_read + parity * n_conns + c)
    tl.store(last_read + (1 - parity) * n_conns + c, run_id)
    return node, time, count, n_blocks, cap, tl.load(window_end_p), fresh


@triton.jit
def _scratch(s, cap):
    return (
        _f32(s), s + cap, s + 2 * cap, s + 3 * cap,
        _f32(s + 4 * cap), s + 5 * cap, s + 6 * cap, s + 7 * cap,
        s + 8 * cap, s + 9 * cap + 1,
    )


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


@triton.jit(do_not_specialize=["n_conns", "batch_cap", "ring_pos", "parity"])
def intake_kernel(
    plan, scratch, ring, n_conns, batch_cap, ring_pos, parity, slot_width,
    N_SLOTS: tl.constexpr, SLOT_CAP: tl.constexpr, MAX_BUCKETS: tl.constexpr,
    BLOCK: tl.constexpr, BLOCK_G: tl.constexpr,
):
    (src_addr, src_n_blocks, src_cap, n_buckets, last_read, bucket_mean,
     _progs) = _plan(plan, n_conns, MAX_BUCKETS)
    window_end = _f32(_scratch(scratch, batch_cap)[9] + batch_cap + 1)  # [n_conns]
    (a_time, a_node, a_conn, a_run, b_time, b_node, b_conn, b_run,
     run_start, run_tmp) = _scratch(scratch, batch_cap)
    (main_time, main_node, main_conn, main_bucket, main_count,
     side_time, side_node, side_conn, side_bucket, side_count,
     watermark, flags) = _ring(ring, N_SLOTS, SLOT_CAP)
    lanes = tl.arange(0, BLOCK)

    # 1. gather: every nonempty source block becomes one sorted run
    n = tl.cast(0, tl.int32)
    n_runs = tl.cast(0, tl.int32)
    for c in range(n_conns):
        node_p, time_p, count_p, n_blocks, cap, wend, fresh = _source(
            src_addr, src_n_blocks, src_cap, last_read, c, n_conns, parity
        )
        tl.store(window_end + c, wend)
        if fresh:
            for b0 in range(0, n_blocks, BLOCK_G):
                blk = b0 + tl.arange(0, BLOCK_G)
                bm = blk < n_blocks
                cnt = tl.minimum(tl.load(count_p + blk, mask=bm, other=0), cap)
                dst = n + tl.cumsum(cnt, axis=0) - cnt
                nonempty = cnt > 0
                rid = n_runs + tl.cumsum(nonempty.to(tl.int32), axis=0) - nonempty.to(tl.int32)
                tl.store(run_start + rid, dst, mask=nonempty)
                for l in range(tl.max(cnt, axis=0)):
                    lm = bm & (l < cnt)
                    tl.store(a_time + dst + l, tl.load(time_p + blk * cap + l, mask=lm), mask=lm)
                    tl.store(a_node + dst + l, tl.load(node_p + blk * cap + l, mask=lm), mask=lm)
                    tl.store(a_conn + dst + l, c + tl.zeros_like(blk), mask=lm)
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

    # 3. insert in e order, one ring entry per delay bucket
    js = tl.arange(0, MAX_BUCKETS)
    mark = tl.load(watermark)
    for i in range(n):
        e = tl.load(a_time + i)
        node = tl.load(a_node + i)
        c = tl.load(a_conn + i)
        nb = tl.load(n_buckets + c)
        to_main = (e >= mark) & (e < tl.load(window_end + c))
        mark = tl.where(to_main, e, mark)

        jm = js < nb
        mean = tl.load(bucket_mean + c * MAX_BUCKETS + js, mask=jm, other=0.0)
        k = tl.maximum(tl.floor((e + mean) / slot_width).to(tl.int32), ring_pos)
        far = jm & (k - ring_pos >= N_SLOTS)
        ok = jm & ~far
        tl.atomic_max(flags + 1, tl.max(far.to(tl.int32), axis=0))
        idx = k % N_SLOTS

        if to_main:
            pos = tl.atomic_add(main_count + idx, 1, mask=ok)
            full = ok & (pos >= SLOT_CAP)
            tl.atomic_max(flags, tl.max(full.to(tl.int32), axis=0))
            w = ok & ~full
            at = idx * SLOT_CAP + pos
            tl.store(main_time + at, e + tl.zeros_like(mean), mask=w)
            tl.store(main_node + at, node + tl.zeros_like(js), mask=w)
            tl.store(main_conn + at, c + tl.zeros_like(js), mask=w)
            tl.store(main_bucket + at, js, mask=w)
        else:
            for j in range(nb):
                kj = tl.maximum(
                    tl.floor((e + tl.load(bucket_mean + c * MAX_BUCKETS + j)) / slot_width).to(tl.int32),
                    ring_pos,
                )
                if kj - ring_pos < N_SLOTS:
                    s = kj % N_SLOTS
                    base = s * SLOT_CAP
                    cnt = tl.load(side_count + s)
                    if cnt >= SLOT_CAP:
                        tl.atomic_max(flags, 1)
                    else:
                        # stable insert: after every entry with time <= e
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
                            nv = tl.load(side_node + base + q, mask=qm)
                            cv = tl.load(side_conn + base + q, mask=qm)
                            bv = tl.load(side_bucket + base + q, mask=qm)
                            tl.store(side_time + base + q + 1, tv, mask=qm)
                            tl.store(side_node + base + q + 1, nv, mask=qm)
                            tl.store(side_conn + base + q + 1, cv, mask=qm)
                            tl.store(side_bucket + base + q + 1, bv, mask=qm)
                            tl.debug_barrier()
                        tl.store(side_time + base + p, e)
                        tl.store(side_node + base + p, node)
                        tl.store(side_conn + base + p, c)
                        tl.store(side_bucket + base + p, j)
                        tl.store(side_count + s, cnt + 1)
                    tl.debug_barrier()
    tl.store(watermark, mark)


# ------------------------------------------------------------ uniform kernel


@triton.jit(do_not_specialize=["n_conns", "ring_pos", "parity"])
def intake_uniform_kernel(
    plan, ring, n_conns, ring_pos, parity, slot_width,
    N_SLOTS: tl.constexpr, SLOT_CAP: tl.constexpr, MAX_BUCKETS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    (src_addr, src_n_blocks, src_cap, n_buckets, last_read, bucket_mean,
     progs) = _plan(plan, n_conns, MAX_BUCKETS)
    (main_time, main_node, main_conn, main_bucket, main_count,
     _side_time, _side_node, _side_conn, _side_bucket, _side_count,
     watermark, flags) = _ring(ring, N_SLOTS, SLOT_CAP)

    pid = tl.program_id(0)
    c = tl.load(progs + pid)
    node_p, time_p, count_p, n_blocks, cap, wend, fresh = _source(
        src_addr, src_n_blocks, src_cap, last_read, c, n_conns, parity
    )
    if fresh:
        mark = tl.load(watermark)

        blk = tl.load(progs + tl.num_programs(0) + pid) + tl.arange(0, BLOCK_B)
        bm = blk < n_blocks
        cnt = tl.minimum(tl.load(count_p + blk, mask=bm, other=0), cap)
        js = tl.arange(0, MAX_BUCKETS)
        jm = js < tl.load(n_buckets + c)
        mean = tl.load(bucket_mean + c * MAX_BUCKETS + js, mask=jm, other=0.0)

        for l in range(tl.max(cnt, axis=0)):
            lm = bm & (l < cnt)
            e = tl.load(time_p + blk * cap + l, mask=lm, other=0.0)
            node = tl.load(node_p + blk * cap + l, mask=lm, other=0)
            main = lm & (e >= mark) & (e < wend)
            tl.atomic_max(flags + 2, tl.max((lm & ~main).to(tl.int32), axis=0))
            k = tl.maximum(tl.floor((e[:, None] + mean[None, :]) / slot_width).to(tl.int32), ring_pos)
            w = main[:, None] & jm[None, :]
            far = w & (k - ring_pos >= N_SLOTS)
            tl.atomic_max(flags + 1, tl.max(tl.max(far.to(tl.int32), axis=1), axis=0))
            w = w & ~far
            idx = k % N_SLOTS
            pos = tl.atomic_add(main_count + idx, 1, mask=w)
            full = w & (pos >= SLOT_CAP)
            tl.atomic_max(flags, tl.max(tl.max(full.to(tl.int32), axis=1), axis=0))
            w = w & ~full
            at = idx * SLOT_CAP + pos
            tl.store(main_time + at, e[:, None] + tl.zeros_like(mean)[None, :], mask=w)
            tl.store(main_node + at, node[:, None] + tl.zeros_like(k), mask=w)
            tl.store(main_conn + at, c + tl.zeros_like(k), mask=w)
            tl.store(main_bucket + at, js[None, :] + tl.zeros_like(k), mask=w)
            tl.atomic_max(watermark, tl.max(tl.where(main, e, float("-inf")), axis=0))


# ------------------------------------------------------------ host side


class Ring:
    """One consumer's ring of arrival slots (see the module docstring), as
    one int32 buffer: main time/node/conn/bucket, side time/node/conn/bucket
    (N_SLOTS * SLOT_CAP each, times stored as float32 bits), main counts,
    side counts (N_SLOTS each), watermark (float32), flags."""

    def __init__(self, n_slots: int, slot_width: float, device, slot_cap: int = DEFAULT_SLOT_CAP):
        self.n_slots, self.slot_cap, self.slot_width = n_slots, slot_cap, slot_width
        c = n_slots * slot_cap
        self.buf = torch.zeros(8 * c + 2 * n_slots + 1 + _N_FLAGS, device=device, dtype=torch.int32)
        parts = [self.buf[i * c:(i + 1) * c] for i in range(8)]
        parts[0], parts[4] = parts[0].view(torch.float32), parts[4].view(torch.float32)
        self.main, self.side = parts[:4], parts[4:]
        self.main_count = self.buf[8 * c:8 * c + n_slots]
        self.side_count = self.buf[8 * c + n_slots:8 * c + 2 * n_slots]
        self.watermark = self.buf[8 * c + 2 * n_slots:8 * c + 2 * n_slots + 1].view(torch.float32)
        self.watermark.fill_(float("-inf"))
        self.flags = self.buf[8 * c + 2 * n_slots + 1:]
        self.ring_pos = 0  # first slot not yet delivered

    def check(self) -> None:
        overflow, far, uniform = self.flags.tolist()
        if overflow:
            raise RuntimeError(f"an arrival slot exceeded its capacity of {self.slot_cap} entries")
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
    buffer: per incoming connection its source's EmissionBuffer address
    (int64), block count, cap and bucket count, the last-read run ids
    (double-buffered), the bucket mean delays (float32 [n_conns,
    MAX_BUCKETS]) and the uniform kernel's program table (connection, first
    block). The emission buffers must stay allocated (they do: one per
    population, for good). run() copies nothing: it launches with scalars
    only."""

    def __init__(self, ring: Ring, emissions: list[EmissionBuffer], bucket_mean: list[list[float]]):
        device = ring.buf.device
        n = len(emissions)
        self.ring, self.n_conns, self.parity = ring, n, 0
        self.emissions = emissions
        self.max_buckets = triton.next_power_of_2(max(max((len(m) for m in bucket_mean), default=1), 1))
        progs = [(c, b0) for c, em in enumerate(emissions) for b0 in range(0, em.n_blocks, _BLOCK_B)]
        self.n_progs = max(len(progs), 1)

        plan = torch.zeros(7 * n + n * self.max_buckets + 2 * self.n_progs, dtype=torch.int32)
        plan[: 2 * n].view(torch.int64)[:] = torch.tensor([em.buf.data_ptr() for em in emissions], dtype=torch.int64)
        plan[2 * n:3 * n] = torch.tensor([em.n_blocks for em in emissions], dtype=torch.int32)
        plan[3 * n:4 * n] = torch.tensor([em.cap for em in emissions], dtype=torch.int32)
        plan[4 * n:5 * n] = torch.tensor([len(m) for m in bucket_mean], dtype=torch.int32)
        means = plan[7 * n:7 * n + n * self.max_buckets].view(torch.float32).view(n, self.max_buckets)
        for c, m in enumerate(bucket_mean):
            means[c, : len(m)] = torch.tensor(m, dtype=torch.float32)
        table = plan[7 * n + n * self.max_buckets:]
        table[: len(progs)] = torch.tensor([c for c, _ in progs], dtype=torch.int32)
        table[self.n_progs:self.n_progs + len(progs)] = torch.tensor([b for _, b in progs], dtype=torch.int32)
        self.plan = plan.to(device)

        self.batch_cap = sum(em.n_blocks * em.cap for em in emissions) + 1
        self.scratch = torch.empty(10 * self.batch_cap + 2 + n, device=device, dtype=torch.int32)

    def run(self, uniform_time: bool = False) -> None:
        ring = self.ring
        if uniform_time:
            intake_uniform_kernel[(self.n_progs,)](
                self.plan, ring.buf, self.n_conns, ring.ring_pos, self.parity, ring.slot_width,
                N_SLOTS=ring.n_slots, SLOT_CAP=ring.slot_cap, MAX_BUCKETS=self.max_buckets,
                BLOCK_B=_BLOCK_B,
            )
        else:
            intake_kernel[(1,)](
                self.plan, self.scratch, ring.buf, self.n_conns, self.batch_cap, ring.ring_pos,
                self.parity, ring.slot_width,
                N_SLOTS=ring.n_slots, SLOT_CAP=ring.slot_cap, MAX_BUCKETS=self.max_buckets,
                BLOCK=_BLOCK, BLOCK_G=_BLOCK_G,
            )
        self.parity ^= 1
