"""
Gluon twin of test_update_emission_gpu.py's green-field update kernel, for
the explicit-LDS delivery walk. Same arguments, same outputs, same
contracts; scope: uniform emission times (bare n2.spike(), what PD14 runs),
so the ring has main runs only and the per-group sort is skipped.

What Gluon buys here, over the Triton kernel's scalar receive:
  - the chunk's link stream (target, weight, in delivery order) lives in
    explicitly allocated LDS instead of global scratch
  - explicit layouts: LT = [GROUP_SIZE, U] puts the neurons over lanes and
    warps and the U links of a walk group entirely inside every thread. A
    walk group is one broadcast LDS read (all lanes, same address), and
    picking link u out of it is a register select -- the receive needs no
    cross-lane traffic at all. Plain Triton chooses layouts itself and pays
    barriers / LDS round trips for the same thing.
Node state stays in registers (the G axis of LT); the receive is the
model's code on the target lane, applied link by link in order.

The Triton helpers (pointer layouts of emission, ring and plan buffers) are
called as they are: Gluon takes triton.jit functions, their tensors pick up
the layouts of their inputs.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from naer2.emission import emission_flat_ptrs, emission_ptrs
from naer2.intake import _plan, _rec, _ring
from naer2.trace import mark, pin, pin_s, trace_begin, trace_end

GLUON_TRACE_TAGS = ("start", "state", "slot", "headers", "streamed", "walked", "delivered", "sent", "end")


@gluon.jit
def _f32p(p):
    return p.to(gl.pointer_type(gl.float32))


# ------------------------------------------------- delivery pieces


@gluon.jit
def _headers(main_lo, main_n, main_src, main_time, cell, b0, nm, n_cells, ei, LOG: gl.constexpr):
    """The ring records of one chunk: entries b0 .. b0 + ENTRY_BLOCK of the
    cell's main run."""
    em_ = b0 + ei < nm
    at = _rec(cell, b0 + ei, n_cells)
    lo = gl.load(main_lo + 4 * at, mask=em_, other=0)
    n = gl.load(main_n + 4 * at, mask=em_, other=0)
    srcw = gl.load(main_src + 4 * at, mask=em_, other=0)
    if LOG:
        ev = gl.load(main_time + 4 * at, mask=em_, other=0.0)
    else:
        ev = gl.zeros_like(lo).to(gl.float32)
    return lo, n, srcw, ev


@gluon.jit
def _links(lo, n, srcw, ev, s0, qi, ltp_tab, atp_tab, STREAM: gl.constexpr, LQ: gl.constexpr, LOG: gl.constexpr):
    """Stream block s0 of a chunk: per stream position its entry (histogram
    of the entries' starts + scan), then its link's target and weight.
    Each entry's two link addresses are formed once, with the stream offset
    folded in (base + lo - pos), so a stream position needs just two
    gathers by entry and its own position. Returns (target or -1, weight,
    source word, e, links in the block, links in the chunk)."""
    pos = gl.cumsum(n, axis=0) - n
    n_links = gl.sum(n, axis=0)
    live = n > 0
    c = srcw >> 24
    delta = (lo - pos).to(gl.int64)
    t_base = gl.gather(ltp_tab, c, 0) + 2 * delta   # byte addresses: int16 targets
    w_base = gl.gather(atp_tab, c, 0) + 4 * delta   # float32 weights
    q = s0 + qi
    qv = q < n_links
    rel = pos - s0
    before = gl.sum((live & (rel < 0)).to(gl.int32), axis=0)
    starts = gl.histogram(rel, STREAM, mask=live & (rel >= 0) & (rel < STREAM), layout=LQ)
    ent = gl.maximum(before + gl.cumsum(starts, axis=0) - 1, 0)
    q64 = q.to(gl.int64)
    ltp = (gl.gather(gl.convert_layout(t_base, LQ), ent, 0) + 2 * q64).to(gl.pointer_type(gl.int16))
    atp = (gl.gather(gl.convert_layout(w_base, LQ), ent, 0) + 4 * q64).to(gl.pointer_type(gl.float32))
    t_q = gl.where(qv, gl.load(ltp, mask=qv, other=0).to(gl.int32) & 0xFFFF, -1)
    w_q = gl.load(atp, mask=qv, other=0.0)
    if LOG:
        srcw_q = gl.gather(gl.convert_layout(srcw, LQ), ent, 0)
        e_q = gl.gather(gl.convert_layout(ev, LQ), ent, 0)
    else:
        srcw_q = gl.zeros_like(t_q)
        e_q = gl.zeros_like(w_q)
    return t_q, w_q, srcw_q, e_q, gl.minimum(n_links - s0, STREAM), n_links


@gluon.jit
def _stage(s_tgt, s_w, s_src, s_e, t_q, w_q, srcw_q, e_q, STREAM: gl.constexpr, U: gl.constexpr,
           LG: gl.constexpr, LOG: gl.constexpr):
    s_tgt.store(gl.convert_layout(gl.reshape(t_q, [STREAM // U, U]), LG))
    s_w.store(gl.convert_layout(gl.reshape(w_q, [STREAM // U, U]), LG))
    if LOG:
        s_src.store(gl.convert_layout(gl.reshape(srcw_q, [STREAM // U, U]), LG))
        s_e.store(gl.convert_layout(gl.reshape(e_q, [STREAM // U, U]), LG))
    gl.barrier()


@gluon.jit
def _walk(r_tgt, r_w, r_src, r_e, nq, current, received, n_received, lane, mask, ui, k, pid,
          log_ptr, log_e, log_cap, GROUP_SIZE: gl.constexpr, U: gl.constexpr, LU: gl.constexpr,
          LOG: gl.constexpr, COUNT: gl.constexpr):
    """The staged stream block's nq links, U per broadcast LDS read,
    received one by one by their target lanes."""
    for g in range(0, (nq + U - 1) // U):
        tg = r_tgt.index(g).load(LU)
        wg = r_w.index(g).load(LU)
        if LOG:
            sg = r_src.index(g).load(LU)
            eg = r_e.index(g).load(LU)
        for u in gl.static_range(U):
            t = gl.sum(gl.where(ui == u, tg, 0), axis=0)    # in-thread: a register select
            w = gl.sum(gl.where(ui == u, wg, 0.0), axis=0)
            hit = (lane == t) & mask
            current = gl.where(hit, current + w, current)  # PD14 receive
            if COUNT:  # the stand-in's bookkeeping (tests), not PD14
                received = gl.where(hit, received + w, received)
                n_received = n_received + hit.to(gl.int32)
            if LOG:
                if t >= 0:
                    srcword = gl.sum(gl.where(ui == u, sg, 0), axis=0)
                    at_log = gl.atomic_add(log_ptr, 1)
                    if at_log < log_cap:
                        rec = log_ptr + 1 + 4 * at_log
                        gl.store(rec, k)
                        gl.store(rec + 1, srcword >> 24)
                        gl.store(rec + 2, srcword & 0xFFFFFF)
                        gl.store(rec + 3, pid * GROUP_SIZE + t)
                        gl.store(log_e + at_log, gl.sum(gl.where(ui == u, eg, 0.0), axis=0))
    return current, received, n_received


@gluon.jit(do_not_specialize=["n_neurons", "n_conns", "k_lo", "k_hi", "run_id", "log_cap"])
def update_kernel_gluon(
    t_ref_end_ptr, potential_ptr, current_ptr, last_time_ptr, n_neurons,
    plan, n_conns, ring, k_lo, k_hi, slot_width, scratch,   # delivery input
    received_ptr, n_received_ptr, log_ptr, log_cap,          # stand-in receive, delivery log
    fire_time_ptr, emit_delay_ptr,                           # stand-in send
    emission, run_id, t_start, t_end, trace_ptr,
    GROUP_SIZE: gl.constexpr, LOCAL_SPIKE_CAP: gl.constexpr, UNIFORM_TIME: gl.constexpr,
    N_SLOTS: gl.constexpr, MAIN_CAP: gl.constexpr, SIDE_CAP: gl.constexpr,
    ENTRY_BLOCK: gl.constexpr, LOG: gl.constexpr,
    STREAM: gl.constexpr, U: gl.constexpr, SPT: gl.constexpr, NUM_WARPS: gl.constexpr, MAX_CONNS: gl.constexpr,
    SPQ: gl.constexpr, SPE: gl.constexpr, COUNT: gl.constexpr = True,
    TRACE: gl.constexpr = 0, TRACE_CAP: gl.constexpr = 1,
):
    gl.static_assert(UNIFORM_TIME, "the Gluon kernel covers uniform emission times only")
    gl.static_assert(STREAM % U == 0, "STREAM must be a multiple of U")
    # layouts (see the module docstring); SPT = neurons per thread
    LT: gl.constexpr = gl.BlockedLayout([SPT, U], [32, 1], [NUM_WARPS, 1], [1, 0])
    LN: gl.constexpr = gl.SliceLayout(1, LT)     # [GROUP_SIZE]: one neuron per lane
    LU: gl.constexpr = gl.SliceLayout(0, LT)     # [U]: a walk group, whole in every thread
    LQ: gl.constexpr = gl.BlockedLayout([SPQ], [32], [NUM_WARPS], [0])  # [STREAM]: stream positions over lanes
    LE: gl.constexpr = gl.BlockedLayout([SPE], [32], [NUM_WARPS], [0])  # [ENTRY_BLOCK]: entries over lanes
    LS: gl.constexpr = gl.BlockedLayout([SPT], [32], [NUM_WARPS], [0])  # LN as a plain layout: scans
    SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    # the stream in LDS as [STREAM // U, U] rows (one walk group each): stored
    # whole through a 2-D view, read a row at a time with a dynamic index
    # (index() keeps the layout's rank, so the buffer's layout is a row's)
    SM2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    LG: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [NUM_WARPS, 1], [1, 0])  # [STREAM // U, U] to store

    pid = gl.program_id(0)
    n_groups = gl.num_programs(0)
    n_ev = gl.to_tensor(0)
    if TRACE:  # see naer2/trace.py; tags GLUON_TRACE_TAGS
        trace = trace_begin(trace_ptr, pid, TRACE_CAP)
        n_ev = mark(trace, n_ev, 0, 0, TRACE_CAP)
    lane = gl.arange(0, GROUP_SIZE, layout=LN)
    offs = pid * GROUP_SIZE + lane
    mask = offs < n_neurons
    (spike_node, spike_time, spike_count, state_run, state_end,
     sum_count, sum_min, overflow) = emission_ptrs(emission, n_groups, LOCAL_SPIKE_CAP)

    # only what the receive touches is live during delivery; the rest of the
    # state is loaded after it (registers are occupancy, occupancy is the
    # latency hiding of the walk's load chains)
    current = gl.load(current_ptr + offs, mask=mask)
    if TRACE:
        current = pin(current)
        n_ev = mark(trace, n_ev, 1, 0, TRACE_CAP)

    # delivery: this group's due slots in slot order; per slot the main run
    # in emission order, every entry's links in order -- as one link stream
    (_em_addr, _topo_addr, _seg_start_addr, link_target_addr, attr_addr, _eng, _ecap, _tns, _tnseg,
     _tnb, _tnl, _last_read, _progs) = _plan(plan, n_conns)
    (main_time, main_src, main_lo, main_n, main_count, _side_time, _side_src, _side_lo, _side_n, side_count,
     _watermark, _flags) = _ring(ring, n_groups, N_SLOTS, MAIN_CAP, SIDE_CAP)
    if COUNT:
        received = gl.load(received_ptr + offs, mask=mask, other=0.0)
        n_received = gl.load(n_received_ptr + offs, mask=mask, other=0)
    else:
        received = gl.zeros_like(current)
        n_received = gl.zeros_like(lane)
    log_e = (log_ptr + 4 * log_cap + 1).to(gl.pointer_type(gl.float32))
    ei = gl.arange(0, ENTRY_BLOCK, layout=LE)
    qi = gl.arange(0, STREAM, layout=LQ)
    ui = gl.arange(0, U, layout=LU)
    # the connections' link and attribute base pointers, on chip once: each
    # stream position picks its own by gather (no per-link global lookup)
    ci = gl.arange(0, MAX_CONNS, layout=LQ)
    s_tgt = gl.allocate_shared_memory(gl.int32, [STREAM // U, U], SM2)
    s_w = gl.allocate_shared_memory(gl.float32, [STREAM // U, U], SM2)
    r_tgt = s_tgt.reinterpret(gl.int32, [STREAM // U, U], SM)
    r_w = s_w.reinterpret(gl.float32, [STREAM // U, U], SM)
    if LOG:
        s_src = gl.allocate_shared_memory(gl.int32, [STREAM // U, U], SM2)
        s_e = gl.allocate_shared_memory(gl.float32, [STREAM // U, U], SM2)
    else:  # placeholders, never touched
        s_src = s_tgt
        s_e = s_w
    r_src = s_src.reinterpret(gl.int32, [STREAM // U, U], SM)
    r_e = s_e.reinterpret(gl.float32, [STREAM // U, U], SM)
    n_cells = n_groups * N_SLOTS
    for k in range(k_lo, k_hi + 1):
        cell = pid * N_SLOTS + k % N_SLOTS
        nm = gl.minimum(gl.load(main_count + cell), MAIN_CAP)
        if TRACE:
            nm = pin_s(nm)
            n_ev = mark(trace, n_ev, 2, nm, TRACE_CAP)
        for b0 in range(0, nm, ENTRY_BLOCK):
            # a chunk of the slot's entries, each a link range of its connection
            n_links = gl.sum(gl.load(main_n + 4 * _rec(cell, b0 + ei, n_cells), mask=b0 + ei < nm, other=0), axis=0)
            if TRACE:
                n_ev = mark(trace, n_ev, 3, n_links, TRACE_CAP)
            for s0 in range(0, n_links, STREAM):  # almost always one block
                # loaded where consumed, so nothing of it stays alive across
                # the walk (the walk is where register pressure peaks): the
                # chunk's records (L1-hot) and the connections' base pointers
                h_lo, h_n, h_srcw, h_ev = _headers(main_lo, main_n, main_src, main_time, cell, b0, nm, n_cells,
                                                   ei, LOG)
                ltp_tab = gl.load(link_target_addr + ci, mask=ci < n_conns, other=0)
                atp_tab = gl.load(attr_addr + ci, mask=ci < n_conns, other=0)
                t_q, w_q, srcw_q, e_q, nq, _n = _links(h_lo, h_n, h_srcw, h_ev, s0, qi, ltp_tab, atp_tab,
                                                       STREAM, LQ, LOG)
                _stage(s_tgt, s_w, s_src, s_e, t_q, w_q, srcw_q, e_q, STREAM, U, LG, LOG)
                if TRACE:
                    n_ev = mark(trace, n_ev, 4, nq, TRACE_CAP)
                current, received, n_received = _walk(
                    r_tgt, r_w, r_src, r_e, nq, current, received, n_received, lane, mask, ui, k, pid,
                    log_ptr, log_e, log_cap, GROUP_SIZE, U, LU, LOG, COUNT)
                if TRACE:
                    current = pin(current)
                    n_ev = mark(trace, n_ev, 5, nq, TRACE_CAP)
                gl.barrier()  # the stream is refilled next
        gl.store(main_count + cell, 0)
        gl.store(side_count + cell, 0)
    if COUNT:
        gl.store(received_ptr + offs, received, mask=mask)
        gl.store(n_received_ptr + offs, n_received, mask=mask)
    if TRACE:
        n_ev = mark(trace, n_ev, 6, 0, TRACE_CAP)

    # final propagation to the window end (PD14)
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

    # stand-in send: emit per the given fire times, uniform emission time
    fire = gl.load(fire_time_ptr + offs, mask=mask, other=float("inf"))
    fires = mask & (fire >= t_start) & (fire < t_end)
    _n_fire = fires.to(gl.int32)
    _local_pos = gl.convert_layout(gl.cumsum(gl.convert_layout(_n_fire, LS), axis=0), LN) - _n_fire
    local_spike_count = gl.sum(_n_fire, axis=0)
    _fits = fires & (_local_pos < LOCAL_SPIKE_CAP)
    gl.store(spike_node + pid * LOCAL_SPIKE_CAP + _local_pos, offs, mask=_fits)
    gl.store(spike_time + pid * LOCAL_SPIKE_CAP + _local_pos, gl.zeros_like(fire) + t_start, mask=_fits)
    potential = gl.where(fires, -65.0, potential)
    t_ref_end = gl.where(fires, time + 2.0, t_ref_end)

    gl.store(t_ref_end_ptr + offs, t_ref_end, mask=mask)
    gl.store(potential_ptr + offs, potential, mask=mask)
    gl.store(current_ptr + offs, current, mask=mask)
    gl.store(last_time_ptr + offs, last_time, mask=mask)
    if TRACE:
        n_ev = mark(trace, n_ev, 7, 0, TRACE_CAP)

    # emission contract (uniform: already sorted), summary, source state
    if local_spike_count > LOCAL_SPIKE_CAP:
        gl.atomic_max(overflow, 1)
    n_emit = gl.minimum(local_spike_count, LOCAL_SPIKE_CAP)
    gl.store(spike_count + pid, n_emit)
    p = run_id % 2
    base = gl.atomic_add(sum_count + p, n_emit)  # this group's place in the flat spike list
    flat_node, flat_time = emission_flat_ptrs(emission, n_groups, LOCAL_SPIKE_CAP)
    gl.store(flat_node + base + _local_pos, offs, mask=_fits)
    gl.store(flat_time + base + _local_pos, gl.zeros_like(fire) + t_start, mask=_fits)
    gl.atomic_min(sum_min + p, gl.min(gl.where(_fits, gl.zeros_like(fire) + t_start, float("inf")), axis=0))
    if pid == 0:
        gl.store(sum_count + (1 - p), 0)
        gl.store(sum_min + (1 - p), float("inf"))
        gl.store(state_run, run_id)
        gl.store(state_end, t_end)
    if TRACE:
        n_ev = mark(trace, n_ev, 8, 0, TRACE_CAP)
        trace_end(trace, n_ev)
