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

from naer2.emission import emission_ptrs
from naer2.intake import _plan, _ring


@gluon.jit
def _f32p(p):
    return p.to(gl.pointer_type(gl.float32))


@gluon.jit(do_not_specialize=["n_neurons", "n_conns", "k_lo", "k_hi", "run_id", "log_cap"])
def update_kernel_gluon(
    t_ref_end_ptr, potential_ptr, current_ptr, last_time_ptr, n_neurons,
    plan, n_conns, ring, k_lo, k_hi, slot_width, scratch,   # delivery input
    received_ptr, n_received_ptr, log_ptr, log_cap,          # stand-in receive, delivery log
    fire_time_ptr, emit_delay_ptr,                           # stand-in send
    emission, run_id, t_start, t_end,
    GROUP_SIZE: gl.constexpr, LOCAL_SPIKE_CAP: gl.constexpr, UNIFORM_TIME: gl.constexpr,
    N_SLOTS: gl.constexpr, MAIN_CAP: gl.constexpr, SIDE_CAP: gl.constexpr,
    ENTRY_BLOCK: gl.constexpr, LOG: gl.constexpr,
    STREAM: gl.constexpr, U: gl.constexpr, SPT: gl.constexpr, NUM_WARPS: gl.constexpr,
):
    gl.static_assert(UNIFORM_TIME, "the Gluon kernel covers uniform emission times only")
    gl.static_assert(STREAM % U == 0, "STREAM must be a multiple of U")
    # layouts (see the module docstring); SPT = neurons per thread
    LT: gl.constexpr = gl.BlockedLayout([SPT, U], [32, 1], [NUM_WARPS, 1], [1, 0])
    LN: gl.constexpr = gl.SliceLayout(1, LT)     # [GROUP_SIZE]: one neuron per lane
    LU: gl.constexpr = gl.SliceLayout(0, LT)     # [U]: a walk group, whole in every thread
    LQE: gl.constexpr = gl.BlockedLayout([1, ENTRY_BLOCK], [32, 1], [NUM_WARPS, 1], [1, 0])
    LQ: gl.constexpr = gl.SliceLayout(1, LQE)    # [STREAM]: one stream position per lane
    LP: gl.constexpr = gl.SliceLayout(0, LQE)    # [ENTRY_BLOCK]: whole in every thread
    LE: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])  # [ENTRY_BLOCK]: one entry per lane
    LS: gl.constexpr = gl.BlockedLayout([SPT], [32], [NUM_WARPS], [0])  # LN as a plain layout: scans
    SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    pid = gl.program_id(0)
    n_groups = gl.num_programs(0)
    lane = gl.arange(0, GROUP_SIZE, layout=LN)
    offs = pid * GROUP_SIZE + lane
    mask = offs < n_neurons
    (spike_node, spike_time, spike_count, state_run, state_end,
     sum_count, sum_min, overflow) = emission_ptrs(emission, n_groups, LOCAL_SPIKE_CAP)

    t_ref_end = gl.load(t_ref_end_ptr + offs, mask=mask)
    potential = gl.load(potential_ptr + offs, mask=mask)
    current = gl.load(current_ptr + offs, mask=mask)
    last_time = gl.load(last_time_ptr + offs, mask=mask)

    # delivery: this group's due slots in slot order; per slot the main run
    # in emission order, every entry's links in order -- as one link stream
    (_em_addr, _topo_addr, seg_start_addr, link_target_addr, attr_addr, _eng, _ecap, _tns, _tnseg,
     _tnb, _tnl, _last_read, _progs) = _plan(plan, n_conns)
    (main_time, main_src, main_seg, main_count, _side_time, _side_src, _side_seg, side_count,
     _watermark, _flags) = _ring(ring, n_groups, N_SLOTS, MAIN_CAP, SIDE_CAP)
    received = gl.load(received_ptr + offs, mask=mask, other=0.0)
    n_received = gl.load(n_received_ptr + offs, mask=mask, other=0)
    log_e = (log_ptr + 4 * log_cap + 1).to(gl.pointer_type(gl.float32))
    ei = gl.arange(0, ENTRY_BLOCK, layout=LE)
    qi = gl.arange(0, STREAM, layout=LQ)
    ui = gl.arange(0, U, layout=LU)
    hdr = scratch + pid * (4 * ENTRY_BLOCK)  # chunk headers: lo, stream position, source word, e
    s_tgt = gl.allocate_shared_memory(gl.int32, [STREAM], SM)
    s_w = gl.allocate_shared_memory(gl.float32, [STREAM], SM)
    s_ent = gl.allocate_shared_memory(gl.int32, [STREAM], SM)
    for k in range(k_lo, k_hi + 1):
        cell = pid * N_SLOTS + k % N_SLOTS
        nm = gl.minimum(gl.load(main_count + cell), MAIN_CAP)
        for b0 in range(0, nm, ENTRY_BLOCK):
            em_ = b0 + ei < nm
            at = cell * MAIN_CAP + b0 + ei
            ev = gl.load(main_time + at, mask=em_, other=0.0)
            srcw = gl.load(main_src + at, mask=em_, other=0)
            seg = gl.load(main_seg + at, mask=em_, other=0)
            ss = gl.load(seg_start_addr + (srcw >> 24), mask=em_, other=0).to(gl.pointer_type(gl.int32))
            lo = gl.load(ss + seg, mask=em_, other=0)
            n = gl.where(em_, gl.load(ss + seg + 1, mask=em_, other=0) - lo, 0)
            pos = gl.cumsum(n, axis=0) - n
            n_links = gl.sum(n, axis=0)
            gl.store(hdr + ei, lo)
            gl.store(hdr + ENTRY_BLOCK + ei, pos)
            gl.store(hdr + 2 * ENTRY_BLOCK + ei, srcw)
            gl.store(_f32p(hdr + 3 * ENTRY_BLOCK + ei), ev)
            pos_p = gl.convert_layout(pos, LP)
            live_p = gl.convert_layout(n, LP) > 0  # segments are never empty
            gl.barrier()
            for s0 in range(0, n_links, STREAM):
                # gather: one stream position per lane, in delivery order
                q = s0 + qi
                qv = q < n_links
                ent = gl.sum(((pos_p[None, :] <= q[:, None]) & live_p[None, :]).to(gl.int32), axis=1) - 1
                ent = gl.maximum(ent, 0)
                lo_q = gl.load(hdr + ent, mask=qv, other=0)
                pos_q = gl.load(hdr + ENTRY_BLOCK + ent, mask=qv, other=0)
                c_q = gl.load(hdr + 2 * ENTRY_BLOCK + ent, mask=qv, other=0) >> 24
                ltp = gl.load(link_target_addr + c_q, mask=qv, other=0).to(gl.pointer_type(gl.int16))
                atp = gl.load(attr_addr + c_q, mask=qv, other=0).to(gl.pointer_type(gl.float32))
                l_q = lo_q + q - pos_q
                t_q = gl.load(ltp + l_q, mask=qv, other=0).to(gl.int32) & 0xFFFF
                w_q = gl.load(atp + l_q, mask=qv, other=0.0)
                s_tgt.store(gl.where(qv, t_q, -1))
                s_w.store(w_q)
                if LOG:
                    s_ent.store(ent)
                gl.barrier()
                # walk: U links per broadcast LDS read, received one by one
                nq = gl.minimum(n_links - s0, STREAM)
                for g in gl.static_range(STREAM // U):
                    if g * U < nq:
                        tg = s_tgt.slice(g * U, U).load(LU)
                        wg = s_w.slice(g * U, U).load(LU)
                        if LOG:
                            eg = s_ent.slice(g * U, U).load(LU)
                        for u in gl.static_range(U):
                            t = gl.sum(gl.where(ui == u, tg, 0), axis=0)    # in-thread: a register select
                            w = gl.sum(gl.where(ui == u, wg, 0.0), axis=0)
                            hit = (lane == t) & mask
                            current = gl.where(hit, current + w, current)  # PD14 receive
                            received = gl.where(hit, received + w, received)
                            n_received = n_received + hit.to(gl.int32)
                            if LOG:
                                if t >= 0:
                                    ent_u = gl.sum(gl.where(ui == u, eg, 0), axis=0)
                                    srcword = gl.load(hdr + 2 * ENTRY_BLOCK + ent_u)
                                    at_log = gl.atomic_add(log_ptr, 1)
                                    if at_log < log_cap:
                                        rec = log_ptr + 1 + 4 * at_log
                                        gl.store(rec, k)
                                        gl.store(rec + 1, srcword >> 24)
                                        gl.store(rec + 2, srcword & 0xFFFFFF)
                                        gl.store(rec + 3, pid * GROUP_SIZE + t)
                                        gl.store(log_e + at_log, gl.load(_f32p(hdr + 3 * ENTRY_BLOCK + ent_u)))
                gl.barrier()  # the stream is refilled next
        gl.store(main_count + cell, 0)
        gl.store(side_count + cell, 0)
    gl.store(received_ptr + offs, received, mask=mask)
    gl.store(n_received_ptr + offs, n_received, mask=mask)

    # final propagation to the window end (PD14)
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

    # emission contract (uniform: already sorted), summary, source state
    if local_spike_count > LOCAL_SPIKE_CAP:
        gl.atomic_max(overflow, 1)
    n_emit = gl.minimum(local_spike_count, LOCAL_SPIKE_CAP)
    gl.store(spike_count + pid, n_emit)
    p = run_id % 2
    gl.atomic_add(sum_count + p, n_emit)
    gl.atomic_min(sum_min + p, gl.min(gl.where(_fits, gl.zeros_like(fire) + t_start, float("inf")), axis=0))
    if pid == 0:
        gl.store(sum_count + (1 - p), 0)
        gl.store(sum_min + (1 - p), float("inf"))
        gl.store(state_run, run_id)
        gl.store(state_end, t_end)
