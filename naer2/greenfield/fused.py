"""
The fused kernel written by hand: the reference the generated kernels are
matched against. The template's fixed phases (naer2/transpiler/templates/
fused.py: the pull, delivery, emission) with hand-written hooks -- the exact
shape a kernel generated from a population's hooks has: an entry that loads
the node attrs where their callbacks need them, passes the receive as a
constexpr function, and runs update and send itself.

MODEL picks the hooks: 0 the stand-in (PD14's update, a send firing at given
fire times -- controlled load for the benchmarks), 1 PD14_neuron, 2
PD14_poisson (test/networks/PD14/), each written exactly as its hooks read,
their static attrs as constexprs (the transpiler inlines them as literals).

FusedPop runs it as a population until the runtime moves into the library
(TODO.md 1e: the `# INTEGRATE` markers, here and in the template).
"""

import contextlib
import math

import torch
import triton
from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from naer2.history import EmissionHistory
from naer2.ring import Ring
from naer2.trace import Trace, mark, trace_begin
from naer2.transpiler.templates.fused import (FUSED_TRACE_TAGS, PULL, WALK, FusedPlan, _deliver, _emit, _pin_first,
                                              _pull, _rand)

__all__ = ["FUSED_TRACE_TAGS", "PULL", "WALK", "FusedPlan", "FusedPop", "fused_kernel", "MODELS",
           "pd14_neuron_params", "pd14_poisson_params"]


# the hand-written hooks of fused_kernel's section 3 (MODEL) and their static attrs, as the transpiler would
# inline them from a node instance (test/networks/PD14/)
MODELS = {"standin": 0, "pd14_neuron": 1, "pd14_poisson": 2}


def pd14_neuron_params(neuron) -> dict:
    """A PD14_neuron instance's static attrs (V_rest depends on K_cortical, per population)."""
    return dict(T_MEM=float(neuron.t_mem), T_SYN=float(neuron.t_syn), R_MEM=float(neuron.R_mem),
                T_REF=float(neuron.t_ref), V_THRESH=float(neuron.V_thresh), V_RESET=float(neuron.V_reset),
                V_REST=float(neuron.V_rest))


def pd14_poisson_params(poisson) -> dict:
    """A PD14_poisson instance's static attrs."""
    return dict(P_RATE=float(poisson.rate), P_START=float(poisson.t_start), P_STOP=float(poisson.t_stop))


# ------------------------------------------------- the hand-written hooks


@gluon.jit
def _receive_current(state, hit, time, w, srcword):
    """PD14_neuron.receive (and the stand-in's): current += edge.weight."""
    (current,) = state
    return (gl.where(hit, current + w, current),)


@gluon.jit
def _receive_none(state, hit, time, w, srcword):
    """PD14_poisson: no receive."""
    return state


# ------------------------------------------------- the kernel


@gluon.jit(do_not_specialize=["n_neurons", "n_conns", "k_lo", "k_hi", "run_id", "round_id", "parity", "log_cap",
                              "hist_cap"])
def fused_kernel(
    # INTEGRATE NODE: one <attr>_ptr per dynamic node attr (KernelBuilder's naming) in place of PD14's four,
    #   each loaded where its callbacks first need it, stored once at the end (the NODE markers below)
    # INTEGRATE HOST: k_lo, k_hi, slot_width, round_id, parity, run_id, t_start, t_end come from the scheduler
    #   (Population._execute_update_kernel's window, Network.step's round), see FusedPop.run
    # INTEGRATE DEV: received_ptr, n_received_ptr, log_ptr, log_cap (stand-in receive bookkeeping), trace_ptr
    # INTEGRATE HOOK(send): fire_time_ptr, emit_delay_ptr are the stand-in send's inputs, gone with the real send
    t_ref_end_ptr, potential_ptr, current_ptr, last_time_ptr, next_spike_ptr, n_neurons,
    plan, n_conns, ring, wmark_ptr, oflags, k_lo, k_hi, slot_width, round_id, parity,  # intake + delivery
    received_ptr, n_received_ptr, log_ptr, log_cap,          # stand-in receive, delivery log
    fire_time_ptr, emit_delay_ptr,                           # stand-in send
    history, hist_cap, run_id, t_start, t_end, trace_ptr,
    GROUP_SIZE: gl.constexpr, K: gl.constexpr, SK: gl.constexpr, KP: gl.constexpr,
    UNIFORM_SOURCES: gl.constexpr, UNIFORM_EMISSION: gl.constexpr,
    N_SLOTS: gl.constexpr, MAIN_CAP: gl.constexpr, SIDE_CAP: gl.constexpr,
    LOG: gl.constexpr,
    STREAM: gl.constexpr, U: gl.constexpr, SPT: gl.constexpr, NUM_WARPS: gl.constexpr, MAX_CONNS: gl.constexpr,
    LOG_CJ: gl.constexpr,
    ITERS: gl.constexpr, COUNT: gl.constexpr = True,
    BATCH: gl.constexpr = True, DIR: gl.constexpr = True, NSP: gl.constexpr = 64,
    TRACE: gl.constexpr = 0, TRACE_CAP: gl.constexpr = 1,
    # the hand-written hooks (section 3): 0 the stand-in, 1 PD14_neuron, 2 PD14_poisson; their static attrs
    MODEL: gl.constexpr = 0, T_MEM: gl.constexpr = 10.0, T_SYN: gl.constexpr = 0.5, R_MEM: gl.constexpr = 40.0,
    T_REF: gl.constexpr = 2.0, V_THRESH: gl.constexpr = -50.0, V_RESET: gl.constexpr = -65.0,
    V_REST: gl.constexpr = -38.30576, P_RATE: gl.constexpr = 0.12, P_START: gl.constexpr = 700.0,
    P_STOP: gl.constexpr = 710.0,
):
    gl.static_assert(STREAM % U == 0, "STREAM must be a multiple of U")
    gl.static_assert(GROUP_SIZE % U == 0, "GROUP_SIZE must be a multiple of U")
    gl.static_assert(MODEL == 0 or UNIFORM_EMISSION, "PD14_neuron and PD14_poisson only emit bare n2.spike()")
    # INTEGRATE CONFIG: GROUP_SIZE, N_SLOTS, MAIN_CAP, SIDE_CAP, STREAM, U, BATCH, DIR per population (ring.py,
    #   PULL, WALK); UNIFORM_EMISSION: every n2.spike() in this population's hooks is bare (from the AST);
    #   UNIFORM_SOURCES: every source's is (the batched pull) -- D-5
    # INTEGRATE DEV: COUNT, LOG, TRACE, TRACE_CAP
    # neurons over lanes and warps (the template's delivery layout)
    LT: gl.constexpr = gl.BlockedLayout([SPT, U], [32, 1], [NUM_WARPS, 1], [1, 0])
    LN: gl.constexpr = gl.SliceLayout(1, LT)
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

    # ---- 1. pull intake: what the sources emitted since this consumer last read them
    n_ev = _pull(pid, n_groups, plan, n_conns, ring, wmark_ptr, oflags, k_lo, slot_width, round_id, parity, trace,
                 n_ev, SK, KP, MAX_CONNS, LOG_CJ, ITERS, N_SLOTS, NSP, MAIN_CAP, SIDE_CAP, UNIFORM_SOURCES, BATCH,
                 DIR, NUM_WARPS, TRACE, TRACE_CAP)

    # ---- 2. delivery
    # INTEGRATE NODE(receive): load here exactly the node attrs the receive hooks read or write (PD14: current),
    #   held in registers through the walk as the state tuple; attrs only update/send touch are loaded after
    #   delivery (registers here are occupancy, TODO.md 1d)
    if MODEL == 2:
        state = ()  # PD14_poisson: no receive (no connections)
    else:
        state = (gl.load(current_ptr + offs, mask=mask),)
    if TRACE:
        state = _pin_first(state)
        n_ev = mark(trace, n_ev, 1, 0, TRACE_CAP)
    if MODEL == 2:
        state, n_ev = _deliver(state, pid, n_groups, plan, n_conns, ring, k_lo, k_hi, offs, lane, mask,
                               received_ptr, n_received_ptr, log_ptr, log_cap, trace, n_ev, _receive_none,
                               GROUP_SIZE, N_SLOTS, MAIN_CAP, SIDE_CAP, STREAM, U, SPT, NUM_WARPS, MAX_CONNS, LOG,
                               COUNT, LOG, TRACE, TRACE_CAP)
    else:
        state, n_ev = _deliver(state, pid, n_groups, plan, n_conns, ring, k_lo, k_hi, offs, lane, mask,
                               received_ptr, n_received_ptr, log_ptr, log_cap, trace, n_ev, _receive_current,
                               GROUP_SIZE, N_SLOTS, MAIN_CAP, SIDE_CAP, STREAM, U, SPT, NUM_WARPS, MAX_CONNS, LOG,
                               COUNT, LOG, TRACE, TRACE_CAP)
        (current,) = state

    # ---- 3. update, send, emission into the history -- MODEL picks the hand-written hooks: the stand-in
    # (PD14's update, a send firing at given fire times: controlled load for benchmarks), PD14_neuron or
    # PD14_poisson (test/networks/PD14/), each written exactly as its hooks read, constants as constexprs
    # where the transpiler inlines the instance's static attrs as literals
    # INTEGRATE NODE(update, send): load here the node attrs only update/send touch (PD14_neuron: t_ref_end,
    #   potential, last_time; PD14_poisson: next_spike); what receive touched is still live
    # INTEGRATE HOOK(update): the transpiled update goes here, time = t_end (the old templates' `time = t`);
    #   then send once more (after every delivery too, D-2)
    # INTEGRATE HOOK(send): the transpiled send goes here, in place of the hand-written ones below. n2.spike()
    #   lowers to an append to the program's emission buffer -- in delivery (receive, send after every
    #   delivery, D-2) and here alike, any number per neuron and run (D-3) -- which the emission below flushes
    #   into the history with one count atomic. Bare n2.spike() emits at the window's START, t_start (D-4; the
    #   old lowering used t, the end)
    time = t_end
    if MODEL == 2:
        # PD14_poisson: update() is a no-op; send():
        #   if t_start <= time <= t_stop: if next_spike <= time: spike(); next_spike += -log(random()) / rate
        # random.random() as RandomLoweringTransformer lowers it: tl.rand(t bitcast to int32, offs), t = the
        # kernel's time (the window end); drawn for every lane, used where the branch holds
        next_spike = gl.load(next_spike_ptr + offs, mask=mask)
        _r = _rand(time.to(gl.int32, bitcast=True), offs)
        _m1 = mask & ((P_START <= time) & (time <= P_STOP))
        fires = _m1 & (next_spike <= time)
        next_spike = gl.where(fires, next_spike + -gl.log(_r) / P_RATE, next_spike)
        # INTEGRATE NODE: every node attr any hook assigns, stored once, here
        gl.store(next_spike_ptr + offs, next_spike, mask=mask)
    else:
        # PD14_neuron.update(): exact propagator integration over dt
        t_ref_end = gl.load(t_ref_end_ptr + offs, mask=mask)
        potential = gl.load(potential_ptr + offs, mask=mask)
        last_time = gl.load(last_time_ptr + offs, mask=mask)
        dt = time - last_time
        last_time = gl.zeros_like(last_time) + time
        P_V = gl.exp(-dt / T_MEM)
        P_I = gl.exp(-dt / T_SYN)
        P_VI = R_MEM * T_SYN / (T_MEM - T_SYN) * (P_V - P_I)
        _m0 = mask & (time < t_ref_end)
        potential = gl.where(_m0, V_RESET, potential)
        potential = gl.where(~_m0 & mask, V_REST + P_V * (potential - V_REST) + P_VI * current, potential)
        current = P_I * current
        if MODEL == 1:
            # PD14_neuron.send(): if potential > V_thresh: spike(); potential = V_reset; t_ref_end = time + t_ref
            fires = mask & (potential > V_THRESH)
        else:
            # the stand-in send (INTEGRATE DEV): fire at the given fire times in the window
            fire = gl.load(fire_time_ptr + offs, mask=mask, other=float("inf"))
            fires = mask & (fire >= t_start) & (fire < t_end)
        potential = gl.where(fires, V_RESET, potential)
        t_ref_end = gl.where(fires, time + T_REF, t_ref_end)
        # INTEGRATE NODE: every node attr any hook assigns, stored once, here
        gl.store(t_ref_end_ptr + offs, t_ref_end, mask=mask)
        gl.store(potential_ptr + offs, potential, mask=mask)
        gl.store(current_ptr + offs, current, mask=mask)
        gl.store(last_time_ptr + offs, last_time, mask=mask)
    if UNIFORM_EMISSION:
        e = gl.zeros_like(lane).to(gl.float32) + t_start
    else:  # the stand-in send's non-uniform emission: fire time + emission delay
        e = (gl.load(fire_time_ptr + offs, mask=mask, other=float("inf"))
             + gl.load(emit_delay_ptr + offs, mask=mask, other=0.0))
    n_ev = _emit(fires, e, offs, pid, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K, SPT, NUM_WARPS,
                 TRACE, TRACE_CAP)


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


# INTEGRATE HOST: FusedPop's job becomes Population's -- its dynamic attrs as the state tensors, the
#   EmissionHistory, and ring + watermarks + FusedPlan built once the network's connections are fixed;
#   run() is called from Population._execute_update_kernel with that run's window (t_start, t_end,
#   deliver_until), on the stream Network keeps for the population, round_id = Network.step's round;
#   _direct_launch's cache keyed by everything that selects the compiled variant (the transpiled source too)
class FusedPop:
    """A green-field population on the fused kernel: PD14-like state, its
    EmissionHistory, and with connect() its rings, per-group watermarks and
    FusedPlan. run() is the whole run, intake included."""

    def __init__(self, n: int, group_size: int = 256, depth: int = 3, device="cuda", model: str = "standin",
                 params: dict | None = None, generated=None, defaults: dict | None = None):
        """model: the hand-written hooks (MODELS) -- "standin" (PD14's update,
        a send firing at the fire times run() is given), "pd14_neuron" or
        "pd14_poisson"; params: their static attrs (pd14_neuron_params /
        pd14_poisson_params; default: the kernel's, PD14's defaults). State
        starts at the node's defaults: potential at V_reset, next_spike at
        the Poisson's start. generated: a FusedKernel (naer2/transpiler/
        templates/fused_template.py) to run instead of the hand-written
        kernel -- its state is `state`, one tensor per dynamic attr, starting
        at `defaults` (the node instance's values)."""
        if model not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}")
        self.generated = generated
        self.state = {a: torch.full((n,), float((defaults or {}).get(a, 0.0)), device=device)
                      for a in (generated.attrs if generated is not None else ())}
        self.n, self.group_size, self.device = n, group_size, torch.device(device)
        self.n_groups = -(-n // group_size)
        self.model, self.params = model, dict(params or {})
        self.potential = torch.full((n,), self.params.get("V_RESET", -65.0), device=device)
        self.current = torch.zeros(n, device=device)
        self.t_ref_end = torch.zeros(n, device=device)
        self.last_time = torch.zeros(n, device=device)
        self.next_spike = torch.full((n,), self.params.get("P_START", 700.0), device=device)
        self.history = EmissionHistory(n, depth, device)  # INTEGRATE CONFIG: depth from the scheduler (history.py)
        self.run_id = 0
        self.received = torch.zeros(n, device=device)  # INTEGRATE DEV: received, n_received, log, trace
        self.n_received = torch.zeros(n, device=device, dtype=torch.int32)
        self.stream, self.unroll = WALK["stream"], WALK["unroll"]
        self.count_received = True
        self.waves_per_eu = 0
        # pull intake (uniform sources): all fresh runs of one emission time as one spike stream
        # (else run by run), segments found through the group directory (else binary search)
        self.batch, self.use_directory = PULL["batch"], PULL["directory"]
        self.trace = None
        self.overrun = torch.zeros(1, device=device, dtype=torch.int32)
        self._no_delay = torch.zeros(n, device=device)
        self._no_fire = torch.full((n,), math.inf, device=device)
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
            num_warps=4, stream=None, uniform_emission=None, uniform_sources=None):
        """One run over [t_start, t_end) in round `round_id` (default: the
        run id): take in what the sources emitted in earlier rounds, deliver
        every ring slot that ends by `deliver_until` (default t_end), emit.
        fire_time / emit_delay: the stand-in send's inputs (None for the
        PD14 models). uniform_emission: this run's spikes all at t_start (the
        stand-in send ignores fire times within the window); uniform_sources: every
        source's runs are (the batched pull); both default to `uniform`.
        stream: a torch.cuda.Stream to launch on (default: the current one;
        passing it saves switching the current stream, ~5 us)."""
        uniform_emission = uniform if uniform_emission is None else uniform_emission
        uniform_sources = uniform if uniform_sources is None else uniform_sources
        self.run_id += 1
        self._t_start, self._t_end = t_start, t_end
        w = self.ring.slot_width
        k_lo = self.ring.ring_pos
        k_hi = int(((deliver_until if deliver_until is not None else t_end) + 1e-6) // w) - 1
        fire_time = fire_time.to(self.device) if fire_time is not None else self._no_fire
        emit_delay = emit_delay.to(self.device) if emit_delay is not None else self._no_delay
        lanes = 32 * num_warps
        if self.generated is not None:
            return self._run_generated(k_lo, k_hi, w, round_id, uniform_sources, num_warps, stream)
        args = (
            self.t_ref_end, self.potential, self.current, self.last_time, self.next_spike, self.n,
            self.plan.plan, self.plan.n_conns, self.ring.buf, self.wmark, self.overrun, k_lo, k_hi, w,
            round_id if round_id is not None else self.run_id, self.run_id % 2,
            self.received, self.n_received, self.log, self.log_cap,
            fire_time, emit_delay,
            self.history.buf, self.history.cap, self.run_id, t_start, t_end,
            self.trace.buf if self.trace is not None else self.log,
        )
        # what selects the compiled variant, besides the plan, ring and history (fixed between connect()s)
        key = (uniform_emission, uniform_sources, num_warps, self.waves_per_eu, self.count_received, self.batch,
               self.use_directory,
               self.stream, self.unroll, self.log_cap > 0, fire_time.dtype, emit_delay.dtype)  # (+ model, params:
        # fixed per FusedPop)
        launch = self._launchers.get(key)
        aligned = not (fire_time.data_ptr() | emit_delay.data_ptr()) & 15  # the specialization every call meets
        if launch is not None and aligned:
            launch(args, stream.cuda_stream if stream is not None else None)
            return self._advance(k_lo, k_hi)
        p2 = triton.next_power_of_2
        consts = dict(
            GROUP_SIZE=self.group_size, K=self.history.K, SK=self.plan.K, KP=p2(self.plan.K),
            UNIFORM_SOURCES=uniform_sources, UNIFORM_EMISSION=uniform_emission,
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
            MODEL=MODELS[self.model], **self.params,
        )
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            kernel = fused_kernel[(self.n_groups,)](*args, **consts, num_warps=num_warps,
                                                    waves_per_eu=self.waves_per_eu)
        if aligned:
            self._launchers[key] = _direct_launch(kernel, self.n_groups, tuple(consts.values()))
        self._advance(k_lo, k_hi)

    def _run_generated(self, k_lo, k_hi, w, round_id, uniform_sources, num_warps, stream):
        """A generated kernel's launch: its dynamic attrs' tensors, then the template's fixed arguments
        (JIT dispatch every time: the cached direct launch is the runtime's, later)."""
        gk, lanes, p2 = self.generated, 32 * num_warps, triton.next_power_of_2
        args = [self.state[a] for a in gk.attrs] + [
            self.n, self.plan.plan, self.plan.n_conns, self.ring.buf, self.wmark, self.overrun, k_lo, k_hi, w,
            round_id if round_id is not None else self.run_id, self.run_id % 2,
            self.received, self.n_received, self.log, self.log_cap,
            self.history.buf, self.history.cap, self.run_id, self._t_start, self._t_end,
            self.trace.buf if self.trace is not None else self.log]
        consts = dict(
            GROUP_SIZE=self.group_size, K=self.history.K, SK=self.plan.K, KP=p2(self.plan.K),
            UNIFORM_SOURCES=uniform_sources, N_SLOTS=self.ring.n_slots, MAIN_CAP=self.ring.main_cap,
            SIDE_CAP=self.ring.side_cap, LOG=self.log_cap > 0, STREAM=self.stream, U=self.unroll,
            SPT=max(self.group_size // lanes, 1), NUM_WARPS=num_warps, MAX_CONNS=p2(max(self.plan.n_conns, 1)),
            LOG_CJ=(p2(max(self.plan.n_conns, 1)) * p2(self.plan.K)).bit_length() - 1, ITERS=self.plan.iters,
            COUNT=self.count_received, BATCH=self.batch, DIR=self.use_directory, NSP=p2(self.ring.n_slots),
            TRACE=self.trace.level if self.trace is not None else 0,
            TRACE_CAP=self.trace.cap if self.trace is not None else 1)
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            gk.kernel[(self.n_groups,)](*args, **consts, num_warps=num_warps, waves_per_eu=self.waves_per_eu)
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
