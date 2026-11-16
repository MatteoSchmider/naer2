"""
A population's runtime on the fused kernel: its state, its emission history,
its rings of arrival slots, its plan of incoming connections, and the
launch of its kernel (TODO.md, runtime integration step 1).

The kernel is a FusedKernel (naer2/transpiler/templates/fused_template.py):
generated from the population's hooks, or any kernel with the same entry
signature -- one pointer per dynamic node attr, then the template's fixed
arguments, then its constexprs (+ the kernel's own, FusedKernel.consts).

  state       one [n] tensor per dynamic attr (FusedKernel.attrs), starting
              at the given values -- a given device tensor is used as is
              (Population shares its attrs' tensors); a caller may replace a
              tensor between runs (same dtype and size)
  connect()   sources (their runtimes) and the SimulationTopology of each
              connection: builds the ring, the watermarks and the plan;
              uniform_sources when every source's kernel emits uniformly
              (else the general pull for all -- a mixed path is an open
              optimization, TODO.md D-5)
  run()       one run over [t_start, t_end) in round round_id: the sources'
              runs of earlier rounds taken in, every slot due by
              deliver_until delivered, update, send, emission

Ring capacity: a run kind (main, side) that overflows -- its claims beyond
the capacity lose their links -- doubles before the next run. No host sync:
every run copies its ring's flags into pinned host memory on its stream,
the next run looks at that copy once it has landed (check() waits for it).
The overflowing run's lost links are reported with a warning.

The launch: the first run per configuration goes through the JIT (and
compiles); later ones call the compiled kernel's launcher directly (~8 us
against ~50 us for the JIT's dispatch with this signature,
tools/experiments/launch_cost.py). The cache is keyed by everything that
selects the compiled variant and dropped when the buffers change.
"""

import contextlib
import math
import warnings

import torch
import triton
from triton import knobs

from .history import EmissionHistory
from .ring import _N_FLAGS, Ring
from .trace import Trace
from .transpiler.templates.fused import FUSED_TRACE_TAGS, PULL, WALK, FusedPlan


def _direct_launch(kernel, n_programs, consts):
    """A compiled kernel's launcher, called directly with its runtime
    arguments (consts: the constexprs' values, in signature order -- they
    come last)."""
    run, function, packed = kernel.run, kernel.function, kernel.packed_metadata
    get_stream = triton.runtime.driver.active.get_current_stream
    device = torch.cuda.current_device()

    def launch(args, stream=None):
        stream = stream if stream is not None else get_stream(device)
        enter, leave = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
        meta = kernel.launch_metadata((n_programs, 1, 1), stream, *args, *consts) if enter or leave else None
        run(n_programs, 1, 1, stream, function, packed, meta, enter, leave, *args, *consts)

    return launch


class PopulationRuntime:
    """See the module docstring. kernel: a FusedKernel; n: the population's
    size; state: initial values per dynamic attr (a scalar or an [n]
    tensor; default 0)."""

    def __init__(self, kernel, n: int, group_size: int = 256, depth: int = 3, device="cuda",
                 state: dict | None = None):
        self.kernel, self.n, self.group_size, self.device = kernel, n, group_size, torch.device(device)
        self.n_groups = -(-n // group_size)
        self.state = {}
        for a in kernel.attrs:
            v = (state or {}).get(a, 0.0)
            if isinstance(v, torch.Tensor):  # used as is (shared with its owner) once on the device, any dtype
                if v.shape != (n,):
                    raise ValueError(f"state '{a}' must have shape ({n},), not {tuple(v.shape)}")
                self.state[a] = v if v.device == self.device else v.to(self.device)
            else:
                self.state[a] = torch.full((n,), float(v), device=self.device)
        # depth from the scheduler (naer2/history.py)
        self.history = EmissionHistory(n, depth, self.device)
        self.run_id = 0
        # dev aids: the template's receive counters (count_received: off by default, a little work per
        # link), delivery log, trace
        self.received = torch.zeros(n, device=self.device)
        self.n_received = torch.zeros(n, device=self.device, dtype=torch.int32)
        self.count_received = False
        self.stream_block, self.unroll = WALK["stream"], WALK["unroll"]
        self.waves_per_eu = 0
        self.batch = PULL["batch"]
        self.trace = None
        self.overrun = torch.zeros(1, device=self.device, dtype=torch.int32)
        self._launchers = {}
        # the ring's flags after each run, copied to the host asynchronously (see the module docstring)
        self._flags_host = torch.zeros(_N_FLAGS, dtype=torch.int32, pin_memory=self.device.type == "cuda")
        self._flags_event = None
        self.growths: list[tuple[int, str, int]] = []  # (run id, "main" / "side", new capacity)
        signature = kernel.kernel.params
        self._constexpr_names = [p.name for p in signature if p.is_constexpr]
        self.connect([], [], n_slots=1)

    @property
    def uniform_emission(self) -> bool:
        return self.kernel.uniform_emission

    def connect(self, sources: list, topologies: list, n_slots: int, slot_width: float = 0.1, log_cap: int = 0,
                main_cap: int | None = None, side_cap: int | None = None):
        caps = {k: v for k, v in (("main_cap", main_cap), ("side_cap", side_cap)) if v}
        self.ring = Ring(self.n_groups, n_slots, slot_width, self.device, **caps)
        self.wmark = torch.full((self.n_groups,), -math.inf, device=self.device)
        self.plan = FusedPlan([s.history for s in sources], topologies, self.n_groups, self.device)
        self.uniform_sources = all(s.uniform_emission for s in sources)
        self.log_cap = log_cap
        self.log = torch.zeros(1 + 5 * max(log_cap, 1), device=self.device, dtype=torch.int32)
        self._launchers = {}  # new buffers (their sizes are part of the AMD backend's specialization)

    def run(self, t_start, t_end, deliver_until=None, round_id=None, num_warps=4, stream=None):
        """One run over [t_start, t_end) in round `round_id` (default: the
        run id): take in what the sources emitted in earlier rounds, deliver
        every ring slot that ends by `deliver_until` (default t_end), update,
        send, emit. stream: a torch.cuda.Stream to launch on (default: the
        current one)."""
        self._grow_if_full(stream, wait=False)
        self.run_id += 1
        w = self.ring.slot_width
        k_lo = self.ring.ring_pos
        k_hi = int(((deliver_until if deliver_until is not None else t_end) + 1e-6) // w) - 1
        args = [self.state[a] for a in self.kernel.attrs] + [
            self.n, self.plan.plan, self.plan.n_conns, self.ring.buf, self.wmark, self.overrun, k_lo, k_hi, w,
            round_id if round_id is not None else self.run_id, self.run_id % 2,
            self.received, self.n_received, self.log, self.log_cap,
            self.history.buf, self.history.cap, self.run_id, t_start, t_end,
            self.trace.buf if self.trace is not None else self.log]
        # what selects the compiled variant, besides the plan, ring and history (fixed between connect()s)
        key = (num_warps, self.waves_per_eu, self.count_received, self.batch, self.stream_block,
               self.unroll, self.log_cap > 0)
        launch = self._launchers.get(key)
        aligned = all(not (t.data_ptr() & 15) for t in self.state.values())  # the specialization every call meets
        if launch is not None and aligned:
            launch(args, stream.cuda_stream if stream is not None else None)
        else:
            consts = self._constexprs(num_warps)
            with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                compiled = self.kernel.kernel[(self.n_groups,)](*args, **consts, num_warps=num_warps,
                                                                waves_per_eu=self.waves_per_eu)
            if aligned:
                self._launchers[key] = _direct_launch(compiled, self.n_groups,
                                                      tuple(consts[c] for c in self._constexpr_names))
        self.ring.ring_pos = max(k_lo, k_hi + 1)
        if self.device.type == "cuda":
            with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                self._flags_host.copy_(self.ring.flags, non_blocking=True)
                self._flags_event = torch.cuda.Event()
                self._flags_event.record()

    def _grow_if_full(self, stream, wait: bool) -> None:
        """The last run's flags (their host copy, once landed -- or waited
        for): a full main or side run doubles that capacity (Ring.grow, on
        the run's stream), its flag cleared, the launch cache dropped (the
        capacities are constexprs)."""
        if self._flags_event is None or not (wait or self._flags_event.query()):
            return
        self._flags_event.synchronize()
        main_full, side_full = int(self._flags_host[0]), int(self._flags_host[3])
        self._flags_event = None
        if not (main_full or side_full):
            return
        ring = self.ring
        main_cap = 2 * ring.main_cap if main_full else None
        side_cap = 2 * ring.side_cap if side_full else None
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            ring.grow(main_cap=main_cap, side_cap=side_cap)
            ring.flags[0] = 0
            ring.flags[3] = 0
        self._launchers = {}
        for kind, cap in (("main", main_cap), ("side", side_cap)):
            if cap:
                self.growths.append((self.run_id, kind, cap))
                warnings.warn(f"a {kind} ring run overflowed by run {self.run_id}: its links beyond the capacity "
                              f"were lost; the capacity doubles to {cap} links", RuntimeWarning, stacklevel=3)

    def _constexprs(self, num_warps) -> dict:
        p2, lanes = triton.next_power_of_2, 32 * num_warps
        consts = dict(
            GROUP_SIZE=self.group_size, K=self.history.K, SK=self.plan.K, KP=p2(self.plan.K),
            UNIFORM_SOURCES=self.uniform_sources, N_SLOTS=self.ring.n_slots, MAIN_CAP=self.ring.main_cap,
            SIDE_CAP=self.ring.side_cap, LOG=self.log_cap > 0, STREAM=self.stream_block, U=self.unroll,
            SPT=max(self.group_size // lanes, 1), NUM_WARPS=num_warps, MAX_CONNS=p2(max(self.plan.n_conns, 1)),
            LOG_CJ=(p2(max(self.plan.n_conns, 1)) * p2(self.plan.K)).bit_length() - 1,
            COUNT=self.count_received, BATCH=self.batch, NSP=p2(self.ring.n_slots),
            TRACE=self.trace.level if self.trace is not None else 0,
            TRACE_CAP=self.trace.cap if self.trace is not None else 1,
            **self.kernel.consts)
        return {c: consts[c] for c in self._constexpr_names if c in consts}

    def check(self) -> None:
        """Wait for the last run's flags (growing a full ring), then raise on
        what remains flagged: far-future arrivals, the uniform contract,
        history overrun."""
        self._grow_if_full(None, wait=True)
        self.ring.check()
        if int(self.overrun[0]):
            raise RuntimeError("a consumer fell behind a source's emission history: a run was overwritten "
                               "before it was read (raise the history depth K)")

    # ---- inspection

    def set_trace(self, cap: int | None = 1024, level: int = 1):
        """Stamp the following runs (naer2/trace.py); level 2 also inside the pull, 3 inside the append; None: stop."""
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
