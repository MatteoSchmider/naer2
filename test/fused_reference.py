"""
The fused kernel written by hand for PD14_neuron and PD14_poisson
(test/networks/PD14/): the reference the generated kernels are matched
against (test/test_fused_generated_gpu.py). The template's fixed phases
(naer2/transpiler/templates/fused.py) with hand-written hooks, in the exact
shape and signature of a generated kernel (naer2/transpiler/templates/
fused_template.py) -- one pointer per dynamic node attr, the template's
fixed arguments, its constexprs, and the node's static attrs as constexprs
of their own (the generator inlines them as literals) -- so a
PopulationRuntime runs either. Each hook is written exactly as it reads.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from naer2.trace import mark, trace_begin
from naer2.transpiler.templates.fused import (_deliver, _emit, _pin_first, _pull, _pull_headers, _rand, _run_done,
                                              _run_entry, _status_slot, _run_gate, _run_status, _gate_skip, _ev_close)
from naer2.transpiler.templates.fused_template import DO_NOT_SPECIALIZE, FusedKernel


@gluon.jit
def _receive_current(state, hit, time, w, srcword, ectx, nodes):
    """PD14_neuron.receive: current += edge.weight."""
    (current,) = state
    return (gl.where(hit, current + w, current),)


@gluon.jit
def _receive_none(state, hit, time, w, srcword, ectx, nodes):
    """PD14_poisson: no receive."""
    return state


@gluon.jit(do_not_specialize=list(DO_NOT_SPECIALIZE))
def pd14_neuron_kernel(
    t_ref_end_ptr, potential_ptr, current_ptr, last_time_ptr,
    n_neurons, plan, n_conns, ring, wmark_ptr, oflags, ctl, status, n_status, sched, sched_i, ev, slot_width,
    received_ptr, n_received_ptr, log_ptr, log_cap, history, hist_cap, trace_ptr, seed, check_ptr,
    GROUP_SIZE: gl.constexpr, K: gl.constexpr, SK: gl.constexpr, KP: gl.constexpr,
    N_SLOTS: gl.constexpr, LOG: gl.constexpr, STREAM: gl.constexpr,
    U: gl.constexpr, SPT: gl.constexpr, NUM_WARPS: gl.constexpr, MAX_CONNS: gl.constexpr, LOG_CJ: gl.constexpr,
    COUNT: gl.constexpr, NSP: gl.constexpr,
    TRACE: gl.constexpr, TRACE_CAP: gl.constexpr, POOL_B: gl.constexpr,
    T_MEM: gl.constexpr, T_SYN: gl.constexpr, R_MEM: gl.constexpr, T_REF: gl.constexpr, V_THRESH: gl.constexpr,
    V_RESET: gl.constexpr, V_REST: gl.constexpr,
):
    LT: gl.constexpr = gl.BlockedLayout([SPT, U], [32, 1], [NUM_WARPS, 1], [1, 0])
    LN: gl.constexpr = gl.SliceLayout(1, LT)
    pid = gl.program_id(0)
    n_groups = gl.num_programs(0)
    n_ev = gl.to_tensor(0)
    trace = trace_ptr
    if TRACE:
        trace = trace_begin(trace_ptr, pid, TRACE_CAP)
        n_ev = mark(trace, n_ev, 0, 0, TRACE_CAP)
    lane = gl.arange(0, GROUP_SIZE, layout=LN)
    offs = pid * GROUP_SIZE + lane
    mask = offs < n_neurons

    hdrs = _pull_headers(plan, n_conns, SK, KP, MAX_CONNS, NUM_WARPS, True)  # (first)
    run_id, round_id, k_lo, k_hi, t_start, t_end, gated = _run_entry(sched, sched_i)
    parity = run_id % 2
    status_run = _status_slot(status, sched, sched_i, run_id, n_status, n_groups)
    if gated != 0:
        k_lo = gl.load(ev + parity * (2 + n_groups))
    skip, t_start = _gate_skip(gated, plan, n_conns, ev, run_id, round_id, t_start, t_end, slot_width, n_groups,
                               SK, KP, MAX_CONNS, NUM_WARPS)
    fin = gl.to_tensor(0)
    if _run_gate(ring, ctl, pid, run_id) & ~skip:  # (see the generated kernels; gated skips: their branch)
        n_ev, ok = _pull(pid, n_groups, plan, n_conns, ring, wmark_ptr, oflags, ctl, k_lo, slot_width, round_id,
                         parity, hdrs, trace, n_ev, SK, KP, MAX_CONNS, LOG_CJ, N_SLOTS, NSP, NUM_WARPS,
                         GROUP_SIZE, TRACE, TRACE_CAP, POOL_B, check_ptr, 0)
        if ok:
            # receive touches current: the state through delivery
            state = (gl.load(current_ptr + offs, mask=mask),)
            if TRACE:
                state = _pin_first(state)
                n_ev = mark(trace, n_ev, 1, 0, TRACE_CAP)
            state, n_ev = _deliver(state, pid, n_groups, plan, n_conns, ring, k_lo, k_hi, offs, lane, mask, received_ptr,
                                   n_received_ptr, log_ptr, log_cap, trace, n_ev, (), _receive_current, GROUP_SIZE, N_SLOTS,
                                   STREAM, U, SPT, NUM_WARPS, MAX_CONNS, LOG, COUNT, LOG, TRACE,
                                   TRACE_CAP, POOL_B, check_ptr, 0)
            (current,) = state

            # update(): exact propagator integration over dt
            t_ref_end = gl.load(t_ref_end_ptr + offs, mask=mask)
            potential = gl.load(potential_ptr + offs, mask=mask)
            last_time = gl.load(last_time_ptr + offs, mask=mask)
            time = t_end
            dt = time - last_time
            last_time = gl.zeros_like(last_time) + time
            P_V = gl.exp(-dt / T_MEM)
            P_I = gl.exp(-dt / T_SYN)
            P_VI = R_MEM * T_SYN / (T_MEM - T_SYN) * (P_V - P_I)
            _m0 = mask & (time < t_ref_end)
            potential = gl.where(_m0, V_RESET, potential)
            potential = gl.where(~_m0 & mask, V_REST + P_V * (potential - V_REST) + P_VI * current, potential)
            current = P_I * current
            # send(): if potential > V_thresh: spike(); potential = V_reset; t_ref_end = time + t_ref
            fires = mask & (potential > V_THRESH)
            potential = gl.where(fires, V_RESET, potential)
            t_ref_end = gl.where(fires, time + T_REF, t_ref_end)
            gl.store(t_ref_end_ptr + offs, t_ref_end, mask=mask)
            gl.store(potential_ptr + offs, potential, mask=mask)
            gl.store(current_ptr + offs, current, mask=mask)
            gl.store(last_time_ptr + offs, last_time, mask=mask)
            e = gl.zeros_like(lane).to(gl.float32) + t_start  # bare n2.spike(): the window's start (D-4)
            n_ev = _emit(fires, e, offs, pid, n_groups, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K, SPT, NUM_WARPS,
                         TRACE, TRACE_CAP)
            _run_done(ring, pid, run_id)
            _ev_close(ev, ring, pid, run_id, k_lo, k_hi, n_groups, N_SLOTS, NSP, NUM_WARPS, POOL_B)
            fin = fin + 1
    _run_status(status_run, ring, pid, run_id, fin > 0, check_ptr, 0)


@gluon.jit(do_not_specialize=list(DO_NOT_SPECIALIZE))
def pd14_poisson_kernel(
    next_spike_ptr,
    n_neurons, plan, n_conns, ring, wmark_ptr, oflags, ctl, status, n_status, sched, sched_i, ev, slot_width,
    received_ptr, n_received_ptr, log_ptr, log_cap, history, hist_cap, trace_ptr, seed, check_ptr,
    GROUP_SIZE: gl.constexpr, K: gl.constexpr, SK: gl.constexpr, KP: gl.constexpr,
    N_SLOTS: gl.constexpr, LOG: gl.constexpr, STREAM: gl.constexpr,
    U: gl.constexpr, SPT: gl.constexpr, NUM_WARPS: gl.constexpr, MAX_CONNS: gl.constexpr, LOG_CJ: gl.constexpr,
    COUNT: gl.constexpr, NSP: gl.constexpr,
    TRACE: gl.constexpr, TRACE_CAP: gl.constexpr, POOL_B: gl.constexpr,
    P_RATE: gl.constexpr, P_START: gl.constexpr, P_STOP: gl.constexpr,
):
    LT: gl.constexpr = gl.BlockedLayout([SPT, U], [32, 1], [NUM_WARPS, 1], [1, 0])
    LN: gl.constexpr = gl.SliceLayout(1, LT)
    pid = gl.program_id(0)
    n_groups = gl.num_programs(0)
    n_ev = gl.to_tensor(0)
    trace = trace_ptr
    if TRACE:
        trace = trace_begin(trace_ptr, pid, TRACE_CAP)
        n_ev = mark(trace, n_ev, 0, 0, TRACE_CAP)
    lane = gl.arange(0, GROUP_SIZE, layout=LN)
    offs = pid * GROUP_SIZE + lane
    mask = offs < n_neurons

    hdrs = _pull_headers(plan, n_conns, SK, KP, MAX_CONNS, NUM_WARPS, True)  # (first)
    run_id, round_id, k_lo, k_hi, t_start, t_end, gated = _run_entry(sched, sched_i)
    parity = run_id % 2
    status_run = _status_slot(status, sched, sched_i, run_id, n_status, n_groups)
    if gated != 0:
        k_lo = gl.load(ev + parity * (2 + n_groups))
    skip, t_start = _gate_skip(gated, plan, n_conns, ev, run_id, round_id, t_start, t_end, slot_width, n_groups,
                               SK, KP, MAX_CONNS, NUM_WARPS)
    fin = gl.to_tensor(0)
    if _run_gate(ring, ctl, pid, run_id) & ~skip:  # (see the generated kernels; gated skips: their branch)
        n_ev, ok = _pull(pid, n_groups, plan, n_conns, ring, wmark_ptr, oflags, ctl, k_lo, slot_width, round_id,
                         parity, hdrs, trace, n_ev, SK, KP, MAX_CONNS, LOG_CJ, N_SLOTS, NSP, NUM_WARPS,
                         GROUP_SIZE, TRACE, TRACE_CAP, POOL_B, check_ptr, 0)
        if ok:
            state = ()  # no receive
            if TRACE:
                n_ev = mark(trace, n_ev, 1, 0, TRACE_CAP)
            state, n_ev = _deliver(state, pid, n_groups, plan, n_conns, ring, k_lo, k_hi, offs, lane, mask, received_ptr,
                                   n_received_ptr, log_ptr, log_cap, trace, n_ev, (), _receive_none, GROUP_SIZE, N_SLOTS,
                                   STREAM, U, SPT, NUM_WARPS, MAX_CONNS, LOG, COUNT, LOG, TRACE,
                                   TRACE_CAP, POOL_B, check_ptr, 0)

            # update() is a no-op; send():
            #   if t_start <= time <= t_stop: if next_spike <= time: spike(); next_spike += -log(random()) / rate
            # random.random() as RandomLoweringTransformer lowers it: tl.rand(t bitcast to int32, offs), t = the
            # kernel's time (the window end); drawn for every lane, used where the branch holds
            time = t_end
            next_spike = gl.load(next_spike_ptr + offs, mask=mask)
            _r = _rand(time.to(gl.int32, bitcast=True), offs)
            _m1 = mask & ((P_START <= time) & (time <= P_STOP))
            fires = _m1 & (next_spike <= time)
            next_spike = gl.where(fires, next_spike + -gl.log(_r) / P_RATE, next_spike)
            gl.store(next_spike_ptr + offs, next_spike, mask=mask)
            e = gl.zeros_like(lane).to(gl.float32) + t_start  # bare n2.spike(): the window's start (D-4)
            n_ev = _emit(fires, e, offs, pid, n_groups, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K, SPT, NUM_WARPS,
                         TRACE, TRACE_CAP)
            _run_done(ring, pid, run_id)
            _ev_close(ev, ring, pid, run_id, k_lo, k_hi, n_groups, N_SLOTS, NSP, NUM_WARPS, POOL_B)
            fin = fin + 1
    _run_status(status_run, ring, pid, run_id, fin > 0, check_ptr, 0)


def pd14_neuron_reference(neuron) -> FusedKernel:
    """The hand-written kernel for a PD14_neuron instance (its static attrs: V_rest depends on K_cortical)."""
    consts = dict(T_MEM=float(neuron.t_mem), T_SYN=float(neuron.t_syn), R_MEM=float(neuron.R_mem),
                  T_REF=float(neuron.t_ref), V_THRESH=float(neuron.V_thresh), V_RESET=float(neuron.V_reset),
                  V_REST=float(neuron.V_rest))
    return FusedKernel(None, pd14_neuron_kernel, attrs=["t_ref_end", "potential", "current", "last_time"],
                       state_attrs=["current"], edge_attr="weight", uniform_emission=True, stage_e=False,
                       consts=consts)


def pd14_poisson_reference(poisson) -> FusedKernel:
    """The hand-written kernel for a PD14_poisson instance."""
    consts = dict(P_RATE=float(poisson.rate), P_START=float(poisson.t_start), P_STOP=float(poisson.t_stop))
    return FusedKernel(None, pd14_poisson_kernel, attrs=["next_spike"], state_attrs=[], edge_attr=None,
                       uniform_emission=True, stage_e=False, consts=consts)
