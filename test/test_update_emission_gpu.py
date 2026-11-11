"""
Green-field update kernel (hand-written, outside population.py): the PD14
neuron kernel stripped down to what the spike-management redesign needs,
to pin down the update kernel's OUTPUT contract and its interface to the
bucketed topology and the intake ring before it goes back into the
transpiler templates.

  kept:        state loads/stores and the final propagation (PD14's exact
               propagator) up to the run's window end
  stand-in:    receive/send -- a neuron emits iff its given fire time lies
               in the run's window [t_start, t_end); its emission time is
               t_start (UNIFORM_TIME, i.e. bare n2.spike()) or its fire time
               + its emission delay
  contract:    naer2/emission.py's EmissionBuffer -- per group sorted by e
               (once at the source, so intake only merges), the flat spike
               list (placed by the summary-count atomic), the source state
               (run id, window end), the summary (count, earliest e) for
               scheduling
  delivery:    every program walks its own group's due slots k_lo..k_hi of
               the consumer's rings in slot order; each ring entry is one
               segment of a connection's TopologyNew, carrying its link range.
               Main run only (uniform emission): a chunk of entries becomes
               one link stream in delivery order (entry, then link), gathered
               one link per lane, then walked U links per round trip with the
               scalar receive (every lane compares against the link's target).
               Main and side runs (any emission times): merged by emission
               time with two cursors, each entry's links walked straight from
               memory. Delivered cells are freed. Stand-in receive: per target
               the summed weight and the number of links delivered, plus an
               optional log of every delivered link (slot, e, connection,
               source, target) for the order checks. test/update_gluon.py is
               the Gluon version of the uniform path (stream in LDS).

The emission output is fed into the real intake kernel and checked against
test_intake_gpu.py's reference end to end.
"""

import math
import random

import pytest
import torch
import triton
import triton.language as tl

from naer2.emission import EmissionBuffer, emission_flat_ptrs, emission_ptrs
from naer2.intake import IntakePlan, Ring, _plan, _rec, _ring
from naer2.spike_gather import _from_ordered_u32, _to_ordered_u32
from naer2.topology import Topology
from naer2.topology_new import TopologyNew, topology_ptrs
from naer2.trace import TRACE_HDR, TRACE_WORDS, mark, pin, pin_s, realtime
from test.test_intake_gpu import Reference, _check, _topology

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernel"
)

DEVICE = "cuda:0"
GROUP_SIZE = 32
BACKEND = "triton"  # Pop's update kernel; test_update_gluon_gpu.py reruns the uniform tests on "gluon"
# delivery walk defaults per backend: entries per chunk (the Triton
# kernel's entry search keeps 32), link stream length, links per memory /
# LDS round trip. Gluon's from tools/walk_sweep.py (PD14, G=256, 780M):
# chunk 128 / stream 256 / U 8, since the delivery was cut into helpers and
# the non-receive state is loaded after it (81 VGPRs at PD14's 8
# connections); larger U costs more in registers than it saves, and an
# explicit occupancy target (waves_per_eu) gains <= 3% (spills beyond 12).
WALK = {
    "triton": dict(entry_block=32, stream=64, unroll=8),
    "gluon": dict(entry_block=128, stream=256, unroll=8),
}


@pytest.fixture(autouse=True, scope="module")
def _device():
    torch.cuda.set_device(0)


# ------------------------------------------------------------ the kernel


@triton.jit
def _f32p(p):
    return p.to(tl.pointer_type(tl.float32))


# ------------------------------------------------ in-kernel tracing (AMD)
# TRACE=1: phases; TRACE=2: also the walk (see naer2/trace.py)

_TRACE_HDR = TRACE_WORDS
TRACE_TAGS = ("start", "state", "slot", "fetched", "staged", "entry", "tail", "delivered", "sent", "end")
(T_START, T_STATE, T_SLOT, T_FETCHED, T_STAGED, T_ENTRY, T_TAIL, T_DELIVERED, T_SENT,
 T_END) = (tl.constexpr(i) for i in range(len(TRACE_TAGS)))
_realtime, _pin, _pin_s, _mark = realtime, pin, pin_s, mark


@triton.jit
def _receive1(t, w, k, e, c, src, pid, current, received, n_received, mask,
              log_ptr, log_e, log_cap, GROUP_SIZE: tl.constexpr, LOG: tl.constexpr):
    """Deliver ONE link (t: its target within the group, uniform over the
    program, -1 for none; w: its weight): every lane compares against t and
    the target lane runs the receive -- the model's receive code, guarded,
    with no layout conversion or reduction. A multapse's weights arrive one
    link at a time, in link order. PD14: current += weight; plus the
    stand-in's received sum, link count and log."""
    lane = tl.arange(0, GROUP_SIZE)
    hit = (lane == t) & mask
    current = tl.where(hit, current + w, current)  # PD14 receive
    received = tl.where(hit, received + w, received)
    n_received = n_received + hit.to(tl.int32)
    if LOG:
        if t >= 0:
            at = tl.atomic_add(log_ptr, 1)
            if at < log_cap:
                rec = log_ptr + 1 + 4 * at
                tl.store(rec, k)
                tl.store(rec + 1, c)
                tl.store(rec + 2, src)
                tl.store(rec + 3, pid * GROUP_SIZE + t)
                tl.store(log_e + at, e)
    return current, received, n_received


@triton.jit
def _walk1(lo, hi, link_target_addr, attr_addr, k, e, c, src, pid, current, received, n_received, mask,
           log_ptr, log_e, log_cap, GROUP_SIZE: tl.constexpr, U: tl.constexpr, LOG: tl.constexpr):
    """Deliver links [lo, hi) of connection c straight from memory, U links'
    loads issued together per round trip, then received one by one."""
    ltp = tl.load(link_target_addr + c).to(tl.pointer_type(tl.int16))
    atp = tl.load(attr_addr + c).to(tl.pointer_type(tl.float32))
    for l0 in range(lo, hi, U):
        for u in tl.static_range(U):
            ok = l0 + u < hi
            t = tl.where(ok, tl.load(ltp + l0 + u, mask=ok, other=0).to(tl.int32) & 0xFFFF, -1)
            w = tl.load(atp + l0 + u, mask=ok, other=0.0)
            current, received, n_received = _receive1(
                t, w, k, e, c, src, pid, current, received, n_received, mask,
                log_ptr, log_e, log_cap, GROUP_SIZE, LOG)
    return current, received, n_received


@triton.jit(do_not_specialize=["n_neurons", "n_conns", "k_lo", "k_hi", "run_id", "log_cap"])
def update_kernel(
    t_ref_end_ptr, potential_ptr, current_ptr, last_time_ptr, n_neurons,
    plan, n_conns, ring, k_lo, k_hi, slot_width, scratch,   # delivery input
    received_ptr, n_received_ptr, log_ptr, log_cap,          # stand-in receive, delivery log
    fire_time_ptr, emit_delay_ptr,                           # stand-in send
    emission, run_id, t_start, t_end, trace_ptr,
    GROUP_SIZE: tl.constexpr, LOCAL_SPIKE_CAP: tl.constexpr, UNIFORM_TIME: tl.constexpr,
    N_SLOTS: tl.constexpr, MAIN_CAP: tl.constexpr, SIDE_CAP: tl.constexpr,
    ENTRY_BLOCK: tl.constexpr, LOG: tl.constexpr,
    TRACE: tl.constexpr = 0, TRACE_CAP: tl.constexpr = 1,
    STREAM: tl.constexpr = 64, U: tl.constexpr = 8,
):
    pid = tl.program_id(0)
    n_groups = tl.num_programs(0)
    trace = trace_ptr + pid * (TRACE_HDR + 2 * TRACE_CAP)
    n_ev = tl.cast(0, tl.int32)
    if TRACE:
        tl.store(trace.to(tl.pointer_type(tl.int64)), _realtime())
        n_ev = _mark(trace, n_ev, T_START, 0, TRACE_CAP)
    offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = offs < n_neurons
    (spike_node, spike_time, spike_count, state_run, state_end,
     sum_count, sum_min, overflow) = emission_ptrs(emission, n_groups, LOCAL_SPIKE_CAP)
    local_spike_count = tl.cast(0, tl.int32)

    # only what the receive touches is live during delivery; the rest of the
    # state is loaded after it (registers are occupancy)
    current = tl.load(current_ptr + offs, mask=mask)
    if TRACE:
        current = _pin(current)
        n_ev = _mark(trace, n_ev, T_STATE, 0, TRACE_CAP)

    # delivery: this group's due slots, in slot order; within a slot main
    # and side runs merged by emission time; each entry is one segment, i.e.
    # a contiguous range of this group's own links (its ring entry says
    # which), delivered in emission order, its links in order
    (_em_addr, _topo_addr, _seg_start_addr, link_target_addr, attr_addr, _eng, _ecap, _tns, _tnseg,
     _tnb, _tnl, _last_read, _progs) = _plan(plan, n_conns)
    (main_time, main_src, main_lo, main_n, main_count, side_time, side_src, side_lo, side_n, side_count,
     _watermark, _flags) = _ring(ring, n_groups, N_SLOTS, MAIN_CAP, SIDE_CAP)
    received = tl.load(received_ptr + offs, mask=mask, other=0.0)
    n_received = tl.load(n_received_ptr + offs, mask=mask, other=0)
    log_e = (log_ptr + 4 * log_cap + 1).to(tl.pointer_type(tl.float32))
    ei = tl.arange(0, ENTRY_BLOCK)
    buf_hdr = scratch + pid * (4 * ENTRY_BLOCK + 3 * STREAM)  # this program's delivery buffer:
    buf_stream = buf_hdr + 4 * ENTRY_BLOCK                     # chunk headers, then the link stream
    qi = tl.arange(0, STREAM)
    for k in range(k_lo, k_hi + 1):
        time = (k + 0.5) * slot_width  # every delivery of a slot at its centre
        cell = pid * N_SLOTS + k % N_SLOTS
        nm = tl.minimum(tl.load(main_count + cell), MAIN_CAP)
        ns = tl.minimum(tl.load(side_count + cell), SIDE_CAP)
        if TRACE:
            nm = _pin_s(nm)
            n_ev = _mark(trace, n_ev, T_SLOT, nm, TRACE_CAP)
        if ns == 0:
            # main run only (always, for uniform emissions), ENTRY_BLOCK
            # entries per chunk: the chunk's links become one stream in
            # delivery order (entry, then link -- long segments whole),
            # gathered one link per lane (bandwidth-bound), then walked U
            # links per memory round trip, each link a uniform scalar every
            # lane compares against (the scalar receive)
            for b0 in range(0, nm, ENTRY_BLOCK):
                em_ = b0 + ei < nm
                at = _rec(cell, b0 + ei, n_groups * N_SLOTS)
                ev = tl.load(main_time + 4 * at, mask=em_, other=0.0)  # 16-byte records
                srcw = tl.load(main_src + 4 * at, mask=em_, other=0)
                lo = tl.load(main_lo + 4 * at, mask=em_, other=0)
                n = tl.load(main_n + 4 * at, mask=em_, other=0)
                pos = tl.cumsum(n, axis=0) - n  # stream position of each entry's first link
                n_links = tl.sum(n, axis=0)
                tl.store(buf_hdr + ei, lo)
                tl.store(buf_hdr + ENTRY_BLOCK + ei, pos)
                tl.store(buf_hdr + 2 * ENTRY_BLOCK + ei, srcw)
                tl.store(_f32p(buf_hdr + 3 * ENTRY_BLOCK + ei), ev)
                tl.debug_barrier()
                if TRACE:
                    n_ev = _mark(trace, n_ev, T_FETCHED, n_links, TRACE_CAP)
                for s0 in range(0, n_links, STREAM):
                    q = s0 + qi
                    qv = q < n_links
                    # entry of stream position q: the last one starting at or before it
                    ent = tl.sum(((pos[None, :] <= q[:, None]) & em_[None, :]).to(tl.int32), axis=1) - 1
                    ent = tl.maximum(ent, 0)
                    lo_q = tl.load(buf_hdr + ent, mask=qv, other=0)
                    pos_q = tl.load(buf_hdr + ENTRY_BLOCK + ent, mask=qv, other=0)
                    c_q = tl.load(buf_hdr + 2 * ENTRY_BLOCK + ent, mask=qv, other=0) >> 24
                    ltp_q = tl.load(link_target_addr + c_q, mask=qv, other=0).to(tl.pointer_type(tl.int16))
                    atp_q = tl.load(attr_addr + c_q, mask=qv, other=0).to(tl.pointer_type(tl.float32))
                    l_q = lo_q + q - pos_q
                    t_q = tl.load(ltp_q + l_q, mask=qv, other=0).to(tl.int32) & 0xFFFF
                    w_q = tl.load(atp_q + l_q, mask=qv, other=0.0)
                    tl.store(buf_stream + qi, tl.where(qv, t_q, -1))
                    tl.store(_f32p(buf_stream + STREAM + qi), w_q)
                    if LOG:
                        tl.store(buf_stream + 2 * STREAM + qi, ent)
                    tl.debug_barrier()
                    if TRACE:
                        n_ev = _mark(trace, n_ev, T_STAGED, 0, TRACE_CAP)
                    nq = tl.minimum(n_links - s0, STREAM)
                    for u0 in range(0, nq, U):
                        for u in tl.static_range(U):
                            ok = u0 + u < nq
                            t = tl.load(buf_stream + u0 + u, mask=ok, other=-1)
                            w = tl.load(_f32p(buf_stream + STREAM + u0 + u), mask=ok, other=0.0)
                            c = tl.cast(0, tl.int32)
                            src = tl.cast(0, tl.int32)
                            e = tl.cast(0.0, tl.float32)
                            if LOG:
                                ent_j = tl.load(buf_stream + 2 * STREAM + u0 + u, mask=ok, other=0)
                                srcword = tl.load(buf_hdr + 2 * ENTRY_BLOCK + ent_j)
                                c = srcword >> 24
                                src = srcword & 0xFFFFFF
                                e = tl.load(_f32p(buf_hdr + 3 * ENTRY_BLOCK + ent_j))
                            current, received, n_received = _receive1(
                                t, w, k, e, c, src, pid, current, received, n_received, mask,
                                log_ptr, log_e, log_cap, GROUP_SIZE, LOG)
                    if TRACE >= 2:
                        current = _pin(current)
                        n_ev = _mark(trace, n_ev, T_ENTRY, nq, TRACE_CAP)
                    tl.debug_barrier()  # the stream is refilled next
        else:
            # main and side runs merged by emission time, two cursors
            im = tl.cast(0, tl.int32)
            js = tl.cast(0, tl.int32)
            for _ in range(nm + ns):
                at_m = 4 * _rec(cell, im, n_groups * N_SLOTS)
                at_s = 4 * _rec(cell, js, n_groups * N_SLOTS)
                em = tl.load(main_time + at_m, mask=im < nm, other=float("inf"))
                es = tl.load(side_time + at_s, mask=js < ns, other=float("inf"))
                if (im < nm) & (em <= es):
                    e = em
                    srcword = tl.load(main_src + at_m)
                    lo_i = tl.load(main_lo + at_m)
                    n_i = tl.load(main_n + at_m)
                    im += 1
                else:
                    e = es
                    srcword = tl.load(side_src + at_s)
                    lo_i = tl.load(side_lo + at_s)
                    n_i = tl.load(side_n + at_s)
                    js += 1
                current, received, n_received = _walk1(
                    lo_i, lo_i + n_i, link_target_addr, attr_addr,
                    k, e, srcword >> 24, srcword & 0xFFFFFF, pid, current, received, n_received, mask,
                    log_ptr, log_e, log_cap, GROUP_SIZE, U, LOG)
        tl.store(main_count + cell, 0)
        tl.store(side_count + cell, 0)
    tl.store(received_ptr + offs, received, mask=mask)
    tl.store(n_received_ptr + offs, n_received, mask=mask)
    if TRACE:
        current = _pin(current)
        n_ev = _mark(trace, n_ev, T_DELIVERED, 0, TRACE_CAP)

    # final propagation to the window end (PD14)
    t_ref_end = tl.load(t_ref_end_ptr + offs, mask=mask)
    potential = tl.load(potential_ptr + offs, mask=mask)
    last_time = tl.load(last_time_ptr + offs, mask=mask)
    time = t_end
    dt = time - last_time
    last_time = time
    P_V = tl.exp(-dt / 10.0)
    P_I = tl.exp(-dt / 0.5)
    P_VI = 40.0 * 0.5 / (10.0 - 0.5) * (P_V - P_I)
    _m0 = mask & (time < t_ref_end)
    potential = tl.where(_m0, -65.0, potential)
    potential = tl.where(~_m0 & mask, -38.30576 + P_V * (potential - -38.30576) + P_VI * current, potential)
    current = P_I * current

    # stand-in send: emit per the given fire times
    fire = tl.load(fire_time_ptr + offs, mask=mask, other=float("inf"))
    fires = mask & (fire >= t_start) & (fire < t_end)
    if UNIFORM_TIME:
        e = t_start + tl.zeros_like(fire)
    else:
        e = fire + tl.load(emit_delay_ptr + offs, mask=mask, other=0.0)
    _n_fire = fires.to(tl.int32)
    _local_pos = local_spike_count + tl.cumsum(_n_fire, axis=0) - _n_fire
    local_spike_count = local_spike_count + tl.sum(_n_fire, axis=0)
    _fits = fires & (_local_pos < LOCAL_SPIKE_CAP)
    tl.store(spike_node + pid * LOCAL_SPIKE_CAP + _local_pos, offs, mask=_fits)
    tl.store(spike_time + pid * LOCAL_SPIKE_CAP + _local_pos, e, mask=_fits)
    potential = tl.where(fires, -65.0, potential)
    t_ref_end = tl.where(fires, time + 2.0, t_ref_end)

    tl.store(t_ref_end_ptr + offs, t_ref_end, mask=mask)
    tl.store(potential_ptr + offs, potential, mask=mask)
    tl.store(current_ptr + offs, current, mask=mask)
    tl.store(last_time_ptr + offs, last_time, mask=mask)
    if TRACE:
        n_ev = _mark(trace, n_ev, T_SENT, 0, TRACE_CAP)

    # emission contract: sort the group's entries by e (once, at the source)
    if local_spike_count > LOCAL_SPIKE_CAP:
        tl.atomic_max(overflow, 1)
    n = tl.minimum(local_spike_count, LOCAL_SPIKE_CAP)
    slot = tl.arange(0, LOCAL_SPIKE_CAP)
    valid = slot < n
    if (not UNIFORM_TIME) and (n > 1):  # most groups emit 0 or 1 spikes per run
        tl.debug_barrier()
        t = tl.load(spike_time + pid * LOCAL_SPIKE_CAP + slot, mask=valid, other=float("inf"))
        node = tl.load(spike_node + pid * LOCAL_SPIKE_CAP + slot, mask=valid, other=0)
        key = (_to_ordered_u32(t).to(tl.uint64) << 32) | node.to(tl.uint32).to(tl.uint64)
        key = tl.sort(key)
        tl.debug_barrier()
        tl.store(spike_node + pid * LOCAL_SPIKE_CAP + slot, (key & 0xFFFFFFFF).to(tl.int32), mask=valid)
        tl.store(spike_time + pid * LOCAL_SPIKE_CAP + slot, _from_ordered_u32((key >> 32).to(tl.uint32)), mask=valid)
    tl.store(spike_count + pid, n)

    # summary for scheduling (per run parity; program 0 resets the other
    # parity for the next run) and the source state intake reads; the count
    # atomic hands this group its place in the flat spike list
    p = run_id % 2
    base = tl.atomic_add(sum_count + p, n)
    flat_node, flat_time = emission_flat_ptrs(emission, n_groups, LOCAL_SPIKE_CAP)
    tl.store(flat_node + base + _local_pos, offs, mask=_fits)
    tl.store(flat_time + base + _local_pos, e, mask=_fits)
    tl.atomic_min(sum_min + p, tl.min(tl.where(fires & (_local_pos < LOCAL_SPIKE_CAP), e, float("inf")), axis=0))
    if pid == 0:
        tl.store(sum_count + (1 - p), 0)
        tl.store(sum_min + (1 - p), float("inf"))
        tl.store(state_run, run_id)
        tl.store(state_end, t_end)
    if TRACE:
        n_ev = _mark(trace, n_ev, T_END, 0, TRACE_CAP)
        tl.store(trace + 4, n_ev)
        tl.store((trace + 2).to(tl.pointer_type(tl.int64)), _realtime())


# ------------------------------------------------------------ helpers


class Pop:
    """A PD14-like population's state plus its emission buffer; with
    connect(), also the consumer side: its rings, its IntakePlan over the
    incoming connections, and the stand-in receive's buffers."""

    def __init__(self, n: int, cap: int = 64, group_size: int = GROUP_SIZE):
        self.n, self.group_size = n, group_size
        self.n_groups = -(-n // group_size)
        self.potential = torch.full((n,), -65.0, device=DEVICE)
        self.current = torch.zeros(n, device=DEVICE)
        self.t_ref_end = torch.zeros(n, device=DEVICE)
        self.last_time = torch.zeros(n, device=DEVICE)
        self.emission = EmissionBuffer(self.n_groups, cap, DEVICE)
        self.run_id = 0
        self.received = torch.zeros(n, device=DEVICE)
        self.n_received = torch.zeros(n, device=DEVICE, dtype=torch.int32)
        self.log_cap = 0
        self.log = torch.zeros(1, device=DEVICE, dtype=torch.int32)
        self.scratch = torch.zeros(self.n_groups * 1024, device=DEVICE, dtype=torch.int32)  # >= 4*ENTRY_BLOCK + 3*STREAM
        self.trace_level, self.trace_cap = 0, 1
        # delivery walk: entries per chunk, link stream length, links per memory round trip
        walk = WALK[BACKEND]
        self.count_received = True  # Gluon: False compiles the stand-in's receive counters out (benchmarks)
        self.waves_per_eu = 0  # Gluon: occupancy the register allocator targets (0: its own choice)
        self.entry_block, self.stream, self.unroll = walk["entry_block"], walk["stream"], walk["unroll"]
        # "triton", or "gluon": test/update_gluon.py (LDS stream walk; uniform emission only)
        self.backend = BACKEND
        self.trace = torch.zeros(1, device=DEVICE, dtype=torch.int32)
        self.connect([], [], n_slots=1)

    def set_trace(self, level: int, cap: int = 1024):
        """Trace the following runs (see TRACE above; each run overwrites)."""
        self.trace_level, self.trace_cap = level, cap if level else 1
        self.trace = torch.zeros(self.n_groups * (_TRACE_HDR + 2 * self.trace_cap) if level else 1,
                                 device=DEVICE, dtype=torch.int32)

    def traced(self) -> list[dict]:
        """Per program: wall-clock start/end (10 ns ticks), and its events as
        (tag, aux, cycles since the program's first stamp, unwrapped)."""
        hdr = _TRACE_HDR
        if self.backend == "gluon":
            from test.update_gluon import GLUON_TRACE_TAGS as tags
        else:
            tags = TRACE_TAGS
        tr = self.trace.view(self.n_groups, hdr + 2 * self.trace_cap).cpu()
        rt = tr[:, :4].contiguous().view(torch.int64)
        out = []
        for g in range(self.n_groups):
            n = min(int(tr[g, 4]), self.trace_cap)
            ev = tr[g, hdr:hdr + 2 * n].view(n, 2).tolist()
            events, base, prev, acc = [], None, None, 0
            for word, cyc in ev:
                if prev is not None:
                    acc += (cyc - prev) % (1 << 20)
                prev = cyc
                events.append((tags[(word >> 24) & 0xFF], word & 0xFFFFFF, acc))
            out.append(dict(start=int(rt[g, 0]), end=int(rt[g, 1]), events=events, dropped=int(tr[g, 4]) - n))
        return out

    def connect(self, sources: list, topologies: list, n_slots: int, slot_width: float = 0.1, log_cap: int = 0,
                main_cap: int | None = None, side_cap: int | None = None):
        caps = {k: v for k, v in (("main_cap", main_cap), ("side_cap", side_cap)) if v}
        self.ring = Ring(self.n_groups, n_slots, slot_width, DEVICE, **caps)
        self.plan = IntakePlan(self.ring, [p.emission for p in sources], topologies)
        self.log_cap = log_cap
        self.log = torch.zeros(1 + 5 * max(log_cap, 1), device=DEVICE, dtype=torch.int32)

    def run(self, t_start, t_end, fire_time, emit_delay=None, uniform=True, deliver_until=None, num_warps=4):
        """One run of the update kernel over [t_start, t_end): deliver every
        ring slot that ends by `deliver_until` (default t_end), then emit."""
        self.run_id += 1
        w = self.ring.slot_width
        k_lo = self.ring.ring_pos
        k_hi = int(((deliver_until if deliver_until is not None else t_end) + 1e-6) // w) - 1
        if self.backend == "gluon" and uniform:
            from test.update_gluon import update_kernel_gluon
            update_kernel_gluon[(self.n_groups,)](
                self.t_ref_end, self.potential, self.current, self.last_time, self.n,
                self.plan.plan, self.plan.n_conns, self.ring.buf, k_lo, k_hi, w, self.scratch,
                self.received, self.n_received, self.log, self.log_cap,
                fire_time.to(DEVICE), (emit_delay if emit_delay is not None else torch.zeros(self.n)).to(DEVICE),
                self.emission.buf, self.run_id, t_start, t_end, self.trace,
                GROUP_SIZE=self.group_size, LOCAL_SPIKE_CAP=self.emission.cap, UNIFORM_TIME=True,
                N_SLOTS=self.ring.n_slots, MAIN_CAP=self.ring.main_cap, SIDE_CAP=self.ring.side_cap,
                ENTRY_BLOCK=self.entry_block, LOG=self.log_cap > 0, STREAM=self.stream, U=self.unroll,
                SPT=max(self.group_size // (32 * num_warps), 1), NUM_WARPS=num_warps,
                SPQ=max(self.stream // (32 * num_warps), 1), SPE=max(self.entry_block // (32 * num_warps), 1),
                MAX_CONNS=triton.next_power_of_2(max(self.plan.n_conns, 1)), COUNT=self.count_received,
                TRACE=int(self.trace_level > 0), TRACE_CAP=self.trace_cap, num_warps=num_warps,
                waves_per_eu=self.waves_per_eu,
            )
            self.ring.ring_pos = max(k_lo, k_hi + 1)
            return
        update_kernel[(self.n_groups,)](
            self.t_ref_end, self.potential, self.current, self.last_time, self.n,
            self.plan.plan, self.plan.n_conns, self.ring.buf, k_lo, k_hi, w, self.scratch,
            self.received, self.n_received, self.log, self.log_cap,
            fire_time.to(DEVICE), (emit_delay if emit_delay is not None else torch.zeros(self.n)).to(DEVICE),
            self.emission.buf, self.run_id, t_start, t_end, self.trace,
            GROUP_SIZE=self.group_size, LOCAL_SPIKE_CAP=self.emission.cap, UNIFORM_TIME=uniform,
            N_SLOTS=self.ring.n_slots, MAIN_CAP=self.ring.main_cap, SIDE_CAP=self.ring.side_cap,
            ENTRY_BLOCK=32, LOG=self.log_cap > 0,
            TRACE=self.trace_level, TRACE_CAP=self.trace_cap, STREAM=self.stream, U=self.unroll,
            num_warps=num_warps,
        )
        self.ring.ring_pos = max(k_lo, k_hi + 1)

    def deliveries(self) -> list[tuple[int, float, int, int, int]]:
        """The delivery log, in log order: (slot, e, connection, source, target)."""
        n = min(int(self.log[0]), self.log_cap)
        rec = self.log[1:1 + 4 * self.log_cap].view(-1, 4)[:n].tolist()
        e = self.log[1 + 4 * self.log_cap:].view(torch.float32)[:n].tolist()
        return [(k, ei, c, s, t) for (k, c, s, t), ei in zip(rec, e)]

    def emitted(self) -> dict[int, list[tuple[float, int]]]:
        """group -> [(time, node)] in buffer order."""
        cap = self.emission.cap
        time, node, count = self.emission.time.tolist(), self.emission.node.tolist(), self.emission.count.tolist()
        return {b: [(time[b * cap + i], node[b * cap + i]) for i in range(c)] for b, c in enumerate(count)}

    def flat(self) -> list[tuple[float, int]]:
        """The last run's flat spike list: [(time, node)], length = its summary count."""
        n, _ = self.emission.summary(self.run_id)
        return list(zip(self.emission.flat_time[:n].tolist(), self.emission.flat_node[:n].tolist()))


def _fire_pattern(n, t_start, t_end, p, rng):
    """Per neuron: a fire time inside the window with probability p, else inf."""
    return torch.tensor([t_start + rng.random() * (t_end - t_start) if rng.random() < p else math.inf for _ in range(n)])


# ------------------------------------------------------------ tests


def test_uniform_emission_buffer_state_and_summary():
    rng = random.Random(0)
    pop = Pop(200)
    fire = _fire_pattern(200, 1.0, 1.1, 0.3, rng)
    pop.run(1.0, 1.1, fire, uniform=True)

    want = sorted((1.0, i) for i in range(200) if fire[i] < math.inf)
    got = sorted(entry for entries in pop.emitted().values() for entry in entries)
    assert [(pytest.approx(t), i) for t, i in want] == got
    for g, entries in pop.emitted().items():
        assert all(g * GROUP_SIZE <= i < (g + 1) * GROUP_SIZE for _, i in entries)
    assert pop.emission.state[0].item() == 1
    assert pop.emission.state[1:].view(torch.float32).item() == pytest.approx(1.1)
    assert pop.emission.summary(1) == (len(want), pytest.approx(1.0))
    assert sorted(pop.flat()) == got  # the flat list: the same spikes, no gaps
    assert torch.allclose(pop.last_time, torch.full((200,), 1.1, device=DEVICE))


def test_exact_times_with_emission_delays_are_sorted_per_group():
    rng = random.Random(1)
    pop = Pop(200)
    fire = _fire_pattern(200, 1.0, 1.1, 0.5, rng)
    delay = torch.tensor([rng.choice([0.0, 0.0, 0.03, 0.4]) for _ in range(200)])
    pop.run(1.0, 1.1, fire, delay, uniform=False)

    e = (fire.to(torch.float32) + delay.to(torch.float32)).tolist()  # the kernel's float32 sum
    for g, entries in pop.emitted().items():
        times = [t for t, _ in entries]
        assert times == sorted(times), f"group {g} not sorted: {times}"
        want = sorted((e[i], i) for i in range(g * GROUP_SIZE, min((g + 1) * GROUP_SIZE, 200)) if fire[i] < math.inf)
        assert sorted(entries) == want
    fired = [i for i in range(200) if fire[i] < math.inf]
    assert pop.emission.summary(1) == (len(fired), min(e[i] for i in fired))
    assert sorted(pop.flat()) == sorted((e[i], i) for i in fired)


def test_summary_is_per_run():
    rng = random.Random(2)
    pop = Pop(100)
    pop.run(1.0, 1.1, _fire_pattern(100, 1.0, 1.1, 0.5, rng))
    fire2 = _fire_pattern(100, 1.1, 1.2, 0.1, rng)
    pop.run(1.1, 1.2, fire2)
    assert pop.emission.summary(2) == (int((fire2 < math.inf).sum()), pytest.approx(1.1))
    assert sorted(n for _, n in pop.flat()) == [i for i in range(100) if fire2[i] < math.inf]  # run 2's only
    pop.run(1.2, 1.3, torch.full((100,), math.inf))
    assert pop.emission.summary(3) == (0, math.inf)


def test_group_overflow_is_flagged_not_written_past():
    pop = Pop(64, cap=8)
    pop.run(1.0, 1.1, torch.full((64,), 1.05))  # 32 spikes per group, cap 8
    assert pop.emission.overflow.item() == 1
    assert pop.emission.count.tolist() == [8, 8]


@pytest.mark.parametrize("uniform", [True, False])
def test_emission_feeds_intake_end_to_end(uniform):
    """Update kernel -> emission buffer -> real intake kernel -> ring, over
    three rounds (the second population doesn't run in round 2), against
    the intake reference."""
    rng = random.Random(3)
    a, b = Pop(150), Pop(90)
    # into a consumer of 64 neurons in 4 groups: per-link and homogeneous delay
    topologies = [_topology(150, 64, 16, seed=0), _topology(90, 64, 16, delay=0.2, seed=1)]
    ring, ref = Ring(4, 32, 0.1, DEVICE), Reference(4, 32, 0.1)
    plan = IntakePlan(ring, [a.emission, b.emission], topologies)
    for r, t in enumerate((1.0, 1.1, 1.2)):
        ran = [a] if r == 1 else [a, b]
        for pop in ran:
            pop.run(t, t + 0.1, _fire_pattern(pop.n, t, t + 0.1, 0.3, rng),
                    torch.tensor([rng.choice([0.0, 0.0, 0.2]) for _ in range(pop.n)]), uniform=uniform)
        plan.run(uniform)
        ref.intake(0, [a.emission, b.emission], [p in ran for p in (a, b)], [t + 0.1] * 2, topologies)
    ring.check()
    _check(ring, ref)
    assert ring.main_count.sum().item() > 0


def test_kernel_takes_a_segment_topology():
    """The delivery interface: a real incoming connection's TopologyNew is
    accepted (walking it is the next step)."""
    pop = Pop(96)
    pop.topo = _topology(50, 96, GROUP_SIZE, seed=4)
    pop.run(1.0, 1.1, torch.full((96,), math.inf))
    assert pop.emission.summary(1) == (0, math.inf)


# ------------------------------------------------------------ delivery


def _expected_deliveries(topo: TopologyNew, c: int, spikes, ring_pos: int, slot_width: float):
    """Per spike (e, source) of connection c taken in at ring position
    ring_pos: every link it delivers, as (slot, e, connection, source,
    target, weight) -- through its segments, like intake and the walk."""
    w = torch.tensor(slot_width, dtype=torch.float32)
    offsets, start, gb = topo.seg_offsets.tolist(), topo.seg_start.tolist(), topo.seg_gb.tolist()
    mean, target, weight = topo.bucket_mean.tolist(), topo.link_target.tolist(), topo.attrs["weight"].tolist()
    out = []
    for e, s in spikes:
        for seg in range(offsets[s], offsets[s + 1]):
            group, j = (gb[seg] >> 16) & 0xFFFF, gb[seg] & 0xFFFF
            arrival = torch.tensor(e, dtype=torch.float32) + torch.tensor(mean[j], dtype=torch.float32)
            k = max(int(torch.floor(arrival / w)), ring_pos)
            for link in range(start[seg], start[seg + 1]):
                out.append((k, e, c, s, group * topo.group_size + (target[link] & 0xFFFF), weight[link]))
    return out


@pytest.mark.parametrize("long_segments", [False, True])
@pytest.mark.parametrize("uniform", [True, False])
def test_deliveries_arrive_in_their_run_in_slot_then_emission_order(uniform, long_segments):
    """Sources A, B -> consumer C (3 groups) over six rounds, in network
    order: every population's update kernel (C delivers its due slots),
    then intake. Every delivered link must come out in the run that covers
    its slot, per group in slot order and within a slot in emission order,
    and C's stand-in receive must sum exactly those weights."""
    rng = random.Random(5)
    a, b, consumer = Pop(150), Pop(90), Pop(96)
    topologies = [  # per-link delays (A -> C) and one homogeneous delay (B -> C)
        _topology(a.n, 96, GROUP_SIZE, delay="random", k=24, seed=10, weight=True),
        # long_segments: ~30 links per (source, group), several U-link walk steps per entry
        _topology(b.n, 96, GROUP_SIZE, delay=0.2, k=90 if long_segments else 24, seed=11, weight=True),
    ]
    consumer.connect([a, b], topologies, n_slots=16, log_cap=200_000)

    expected_by_run, delivered_by_run = [], []
    for r in range(6):
        t = round(1.0 + 0.1 * r, 6)
        before = len(consumer.deliveries())
        consumer.run(t, t + 0.1, torch.full((96,), math.inf), uniform=uniform)
        delivered_by_run.append(consumer.deliveries()[before:])
        for src in (a, b):
            src.run(t, t + 0.1, _fire_pattern(src.n, t, t + 0.1, 0.3, rng),
                    torch.tensor([rng.choice([0.0, 0.0, 0.15]) for _ in range(src.n)]), uniform=uniform)
        ring_pos = consumer.ring.ring_pos
        consumer.plan.run(uniform)
        consumer.ring.check()
        spikes = [[(e, s) for entries in src.emitted().values() for e, s in entries] for src in (a, b)]
        expected_by_run.append([d for c in range(2) for d in
                                _expected_deliveries(topologies[c], c, spikes[c], ring_pos, 0.1)])

    # which run delivers which slot: run r delivers slots up to (t_r + 0.1) / 0.1 - 1
    due_until = [int(round((1.0 + 0.1 * r + 0.1) / 0.1)) - 1 for r in range(6)]
    pending = [d for run in expected_by_run for d in run]
    for r, delivered in enumerate(delivered_by_run):
        lo = due_until[r - 1] + 1 if r else -10**9
        want = sorted((k, e, c, s, tg) for k, e, c, s, tg, _ in pending if lo <= k <= due_until[r])
        got = sorted(delivered)
        assert got == want, f"run {r}"
        for grp in range(consumer.n_groups):
            order = [(k, e) for k, e, c, s, tg in delivered if tg // GROUP_SIZE == grp]
            assert order == sorted(order), f"run {r}, group {grp}: not in slot then emission order"

    delivered_all = [d for run in delivered_by_run for d in run]
    assert delivered_all, "nothing was delivered"
    sums = torch.zeros(96, dtype=torch.float64)
    counts = torch.zeros(96, dtype=torch.int64)
    for k, e, c, s, tg, wt in pending:
        if k <= due_until[-1]:
            sums[tg] += wt
            counts[tg] += 1
    assert torch.equal(consumer.n_received.cpu().long(), counts)
    assert torch.allclose(consumer.received.cpu().double(), sums, atol=1e-4)
    # delivered cells are freed: every cell not holding a still-pending slot is empty
    still_pending = {consumer.ring.cell(tg // GROUP_SIZE, k) for k, e, c, s, tg, _ in pending if k > due_until[-1]}
    counts_now = (consumer.ring.main_count + consumer.ring.side_count).tolist()
    assert all(n == 0 for cell, n in enumerate(counts_now) if cell not in still_pending)
    assert all(counts_now[cell] > 0 for cell in still_pending)


# ------------------------------------------------------------ tracing


@pytest.mark.parametrize("level", [1, 2])
def test_tracing_changes_nothing_and_stamps_every_entry(level):
    """The traced kernel delivers exactly what the untraced one does; every
    program stamps its phases, and with TRACE=2 the walk accounts for every
    link gathered into the stream."""
    results = []
    for traced in (False, True):
        rng = random.Random(5)
        a, consumer = Pop(150), Pop(96)
        topo = _topology(a.n, 96, GROUP_SIZE, delay="random", k=40, seed=10, weight=True)
        consumer.connect([a], [topo], n_slots=16)
        if traced:
            consumer.set_trace(level, 4096)
        programs = []
        for r in range(6):
            t = round(1.0 + 0.1 * r, 6)
            consumer.run(t, t + 0.1, torch.full((96,), math.inf))
            if traced:
                programs.append(consumer.traced())
            a.run(t, t + 0.1, _fire_pattern(a.n, t, t + 0.1, 0.3, rng), uniform=True)
            consumer.plan.run(True)
        results.append((consumer.n_received.cpu(), consumer.received.cpu(), programs))
    assert torch.equal(results[0][0], results[1][0])
    # uniform intake appends equal-e entries by parallel atomics: their order,
    # hence the float summation order, varies run to run
    assert torch.allclose(results[0][1], results[1][1], atol=1e-4)
    for run in results[1][2]:
        for prog in run:
            tags = [tag for tag, _, _ in prog["events"]]
            assert prog["dropped"] == 0 and tags[0] == "start" and tags[-1] == "end"
            assert {"state", "delivered", "sent"} <= set(tags)
            assert prog["end"] >= prog["start"] > 0
            cycles = [c for _, _, c in prog["events"]]
            assert cycles == sorted(cycles)
            if level == 2:  # stream walk: every link gathered into the stream is walked
                fetched = sum(aux for tag, aux, _ in prog["events"] if tag == "fetched")
                walked = sum(aux for tag, aux, _ in prog["events"] if tag == "entry")
                assert walked == fetched
