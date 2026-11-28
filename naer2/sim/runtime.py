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
              connection: builds the ring, the watermarks and the plan --
              the slotted ring for uniform sources only (every source's
              kernel emits uniformly), queues for any (SPIKES.md)
  run()       one run over [t_start, t_end) in round round_id: the sources'
              runs of earlier rounds taken in, every slot due by
              deliver_until delivered, update, send, emission
  launch()    one run's kernel launch, from its RunParams (a direct run, a
              planned one, a rerun)

Around it, in this package: its runs until known complete and their status
(ledger: naer2/sim/ledger.py), its ring's sizes following the runs' need
(sizer: sizing.py), and the device's stop word, journal and reruns
(recovery: recovery.py). Every size is a runtime value in the ring's
header: a resize keeps the compiled kernel and its launcher.

The launch: the first run per configuration goes through the JIT (and
compiles); later ones call the compiled kernel's launcher directly (~8 us
against ~50 us for the JIT's dispatch with this signature,
tools/experiments/launch_cost.py). The cache is keyed by everything that
selects the compiled variant and dropped when the buffers change.
"""

import contextlib
import math

import torch
import triton
from triton import knobs

from ..history import EmissionHistory
from ..ring import RESERVE, Ring
from ..schedule import write_entry
from ..trace import Trace
from ..transpiler.templates.fused import FUSED_TRACE_TAGS, WALK, FusedPlan
from .ledger import Ledger, RunParams
from .recovery import recovery
from .sizing import RingSizer


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


def _warp_cap(device) -> int:
    """The most warps a program gets by default: 4, on every device for now (a per-device choice -- an
    autotuning at startup -- is a later feature, TODO.md). Measured: PD14 at G=256 replayed from captured
    windows on the 4090 ~29 us a round at 4 warps against ~35-40 at 8 (4 warps: a round's 309 programs all
    resident at once, 4 a multiprocessor by registers; 8: 2, the rest a second wave -- tools/pd14_occupancy.py);
    the 780M ~425 us stepped at 4 against ~470 at 8. The RX 6600 (14 WGPs) was faster at 8 when stepped
    (~280 us a round against ~465 at 4, host-bound): its loss, until the warps are chosen per device."""
    return 4




class PopulationRuntime:
    """See the module docstring. kernel: a FusedKernel; n: the population's
    size; state: initial values per dynamic attr (a scalar or an [n]
    tensor; default 0)."""

    def __init__(self, kernel, n: int, group_size: int = 256, depth: int = 3, device="cuda",
                 state: dict | None = None, history_cap: int | None = None):
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
        # depth from the scheduler (naer2/history.py); history_cap: the spikes a run can emit (default: one per
        # neuron -- D-3: more where neurons fire inside delivery)
        # (a region per program where its spikes need ordering -- non-uniform emission; a uniform one's: one flat
        # list, history.py)
        self.history = EmissionHistory(n, depth, self.device, cap=history_cap, group_size=group_size,
                                       regions=not kernel.uniform_emission)
        self.run_id = 0
        # dev aids: the template's receive counters (count_received: off by default, a little work per
        # link), delivery log, trace
        self.received = torch.zeros(n, device=self.device)
        self.n_received = torch.zeros(n, device=self.device, dtype=torch.int32)
        self.count_received = False
        self.stream_block, self.unroll = WALK["stream"], WALK["unroll"]
        self.waves_per_eu = 0
        self.max_warps: int | None = None  # a program's warps at most (None: the device's cap, _warp_cap)
        self.maxnreg: int | None = None  # NVIDIA: registers per thread at most (None: the compiler's choice)
        self.seed = 0  # random.*'s stream (the population's own seed: Population.seed)
        # a development aid: every block the pull pops and delivery frees checked on the device (fused.py:
        # ARENA_CHECK; arena_errors() the first violation); off: compiled out
        self.arena_check = False
        self._chk, self._chk_key = None, None
        self.trace = None
        self.overrun = torch.zeros(1, device=self.device, dtype=torch.int32)
        self._launchers = {}
        self.recovery = recovery(self.device)
        self.ledger = Ledger(self, self.n_groups, self.device)
        self.sizer = RingSizer()
        signature = kernel.kernel.params
        self._constexpr_names = [p.name for p in signature if p.is_constexpr]
        self.connect([], [], n_slots=1)
        self.recovery.runtimes.add(self)

    @property
    def uniform_emission(self) -> bool:
        return self.kernel.uniform_emission

    def connect(self, sources: list, topologies: list, n_slots: int, slot_width: float = 0.1, log_cap: int = 0,
                arena: tuple[int, int] | None = None, min_delays: list[float] | None = None, queues: bool = False):
        """arena: (block, blocks per group) of the ring (Ring; default: 256-record blocks, the smallest arena --
        every slot's reserve and one more -- grown to the runs' need). min_delays: per connection its minimum
        delay as the host schedules with it (a gated run's due time; default 0). queues: a queue per connection
        (n_slots: the connections; fused.py _pull_queues); else the slotted ring, its sources' emission uniform
        (fused.py _pull). log_cap: the delivery log's records (a development aid; 0: none)."""
        if not queues and not all(s.uniform_emission for s in sources):
            raise NotImplementedError(
                "the slotted ring (per-link delays) takes uniform sources only (bare n2.spike() in send): a source "
                "emitting at precise times -- a loop, or inside delivery -- needs every incoming connection one "
                "delay value, its queue (SPIKES.md)")
        self.ledger.judge_pending(wait=True)  # (the old ring's)
        self.queues = queues
        self.ring = Ring(self.n_groups, n_slots, slot_width, self.device,
                         arena or (256, n_slots * (RESERVE + 1) + 1))
        self.wmark = torch.full((self.n_groups,), -math.inf, device=self.device)
        self.plan = FusedPlan([s.history for s in sources], topologies, self.n_groups, self.device,
                              min_delays=min_delays)
        # the event state (fused.py: _gate_skip, _ev_close): per run parity the delivery position, the runs
        # made, per group its first slot still holding records
        self.ev = torch.full((2, 2 + self.n_groups), 2147483647, device=self.device, dtype=torch.int32)
        self.ev[:, :2] = 0
        self.log_cap = log_cap
        self.log = torch.zeros(1 + 5 * max(log_cap, 1), device=self.device, dtype=torch.int32)
        self._launchers = {}  # new buffers (their sizes are part of the AMD backend's specialization)
        self.sizer.reset()

    # ---- running

    def run(self, t_start, t_end, deliver_until=None, round_id=None, num_warps=None, stream=None):
        """One run over [t_start, t_end) in round `round_id` (default: the
        run id): take in what the sources emitted in earlier rounds, deliver
        every ring slot that ends by `deliver_until` (default t_end), update,
        send, emit. stream: a torch.cuda.Stream to launch on (default: the
        current one). num_warps: default one lane per neuron, up to 8 warps;
        never more lanes than the group has neurons (see _warps)."""
        self.start(self.direct_params(t_start, t_end, deliver_until, round_id, num_warps), stream)

    def direct_params(self, t_start, t_end, deliver_until=None, round_id=None, num_warps=None) -> RunParams:
        """The next run's launch from the direct schedule (run()'s; a stepped round's, naer2/sim/executor.py):
        its entry written into its status slot's row of the ledger's pinned host schedule (the slot judged
        free first), the run id and the ring's delivery position moved on."""
        num_warps = self._warps(num_warps)
        self.ledger.judge_pending(wait=False)
        self.ledger.free_slot()  # (the slot this run's kernel writes)
        self.run_id += 1
        k_lo, k_hi = self.ring.ring_pos, self.delivers_until(deliver_until if deliver_until is not None else t_end)
        if self.ring.first_pending is not None:  # its ring empty before: from its first pending slot, at most
            # its own start's (an event-driven population idle past its ring's length; gated runs: _run_from)
            k_lo = max(k_lo, min(self.ring.first_pending, int(t_start // self.ring.slot_width), k_hi))
            self.ring.first_pending = None  # (this run's pull adds records)
        slot = self.ledger.slot(self.run_id)
        write_entry(self.ledger.sched_np[slot], self.run_id, round_id if round_id is not None else self.run_id,
                    k_lo, k_hi, t_start, t_end)
        self.ring.ring_pos = max(k_lo, k_hi + 1)
        return RunParams(self.ledger.sched_host, slot, self.run_id, num_warps, self._state())

    def entry_params(self, sched: torch.Tensor, i: int, run_id: int, num_warps=None, status=None) -> RunParams:
        """A planned run's launch (RunParams): entry i of schedule sched; status: where the kernel writes its
        status (None: the ledger's host slots)."""
        return RunParams(sched, i, run_id, self._warps(num_warps), self._state(), status)

    def _state(self) -> tuple:
        return tuple(self.state[a] for a in self.kernel.attrs)

    def start(self, params: RunParams, stream=None) -> None:
        """A run launched and kept until known complete: its status slot judged free, the launch, the run
        pending and journaled for a rerun (Ledger.launched)."""
        self.ledger.free_slot()
        self.launch(params, stream)
        self.ledger.launched(params)

    def launch(self, params: RunParams, stream) -> None:
        """One run's kernel launch (run(): the direct schedule; a planned window: naer2/schedule.py; a rerun:
        Recovery.recover)."""
        args, aligned = self.launch_args(params)
        num_warps = params.num_warps
        # what selects the compiled variant, besides the plan, ring and history (fixed between connect()s)
        key = (num_warps, self.waves_per_eu, self.maxnreg, self.arena_check, self.count_received, self.stream_block,
               self.unroll, self.log_cap > 0)
        launch = self._launchers.get(key)
        if launch is not None and aligned:
            launch(args, stream.cuda_stream if stream is not None else None)
        else:
            consts = self.constexprs(num_warps)
            # waves_per_eu: an AMD backend option (the CUDA backend rejects it)
            # maxnreg: the NVIDIA backend's (fewer registers: more programs resident, maybe spills)
            amd = {"waves_per_eu": self.waves_per_eu} if torch.version.hip else (
                {"maxnreg": self.maxnreg} if self.maxnreg else {})
            with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                compiled = self.kernel.kernel[(self.n_groups,)](*args, **consts, num_warps=num_warps, **amd)
            if aligned:
                self._launchers[key] = _direct_launch(compiled, self.n_groups,
                                                      tuple(consts[c] for c in self._constexpr_names))

    def launch_args(self, params: RunParams) -> tuple[list, bool]:
        """One run's kernel arguments (the generated kernel's, constexprs aside: constexprs()), and whether its
        state tensors are 16-byte aligned (the specialization every call meets) -- for its own kernel (launch)
        or its share of a fused round's (naer2/fusion.py)."""
        status = params.status
        if status is None:
            status = self.ledger.status_host if self.device.type == "cuda" else self.overrun
        args = list(params.state) + [
            self.n, self.plan.plan, self.plan.n_conns, self.ring.buf, self.wmark, self.overrun, self.recovery.ctl,
            status, self.ledger.slots, params.sched, params.index,
            self.ev, self.ring.slot_width, self.received, self.n_received, self.log, self.log_cap,
            self.history.buf, self.history.R, self.trace.buf if self.trace is not None else self.log, self.seed,
            self._arena_check_buf()]
        return args, all(not (t.data_ptr() & 15) for t in params.state)

    def constexprs(self, num_warps) -> dict:
        p2, lanes = triton.next_power_of_2, 32 * num_warps
        consts = dict(
            GROUP_SIZE=self.group_size, K=self.history.K, SK=self.plan.K, KP=p2(self.plan.K),
            N_SLOTS=self.ring.n_slots,
            LOG=self.log_cap > 0, STREAM=self.stream_block, U=self.unroll,
            SPT=max(self.group_size // lanes, 1), NUM_WARPS=num_warps, MAX_CONNS=p2(max(self.plan.n_conns, 1)),
            LOG_CJ=(p2(max(self.plan.n_conns, 1)) * p2(self.plan.K)).bit_length() - 1,
            COUNT=self.count_received, NSP=p2(self.ring.n_slots),
            TRACE=self.trace.level if self.trace is not None else 0,
            TRACE_CAP=self.trace.cap if self.trace is not None else 1,
            BLOCK=self.ring.block,
            ARENA_CHECK=bool(self.arena_check), QUEUES=self.queues,
            GP=p2(max([max(h.G if h.regions else 1, 1) for h in self.plan.histories], default=1)),
            **self.kernel.consts)
        return {c: consts[c] for c in self._constexpr_names if c in consts}

    def _warps(self, num_warps: int | None) -> int:
        """A program's warps: one lane per neuron of its group, up to max_warps (None: the device's cap,
        _warp_cap). More lanes than neurons put
        every neuron's values in several lanes (a replicated layout), and the kernel's emission then
        intermittently lost spikes (PD14 at G=64 with 4 warps: a spike's slot in the history left unwritten,
        found as the uniform contract's flag; limitations.md) -- so it isn't allowed. Fewer lanes than
        neurons serialize each lane's neurons and leave fewer waves to hide the memory round trips: PD14 on
        the RX 6600 in quiet activity, G=256 at 4 warps ~465 us a round (GPU-bound), at 8 ~280 (host-bound);
        G=512 at 16 warps ~620 (1024-thread programs: fewer resident per CU). Where more warps stop paying
        depends on the device (_warp_cap)."""
        cap = self.max_warps if self.max_warps is not None else _warp_cap(self.device)
        most = max(1, min(cap, self.group_size // 32))
        if num_warps is None:
            return most
        if 32 * num_warps > self.group_size and num_warps > 1:
            raise ValueError(f"{num_warps} warps are more lanes than a group of {self.group_size} neurons "
                             f"(at most {most}: a replicated layout loses spikes, see PopulationRuntime._warps)")
        return num_warps

    def ran(self, blocks: int, most: int, lost: int, run_id: int, n_runs: int) -> None:
        """(Ledger.judge) n_runs runs complete, run_id the last: the ring follows their need (RingSizer.follow;
        lost: their lost bits -- links lost, raised by check()); a resize wanted is applied now, the device
        drained (Recovery.resize)."""
        if self.sizer.follow(self.ring, blocks, most, run_id, n_runs):
            self.recovery.resize()

    # ---- the device's positions (readbacks)

    def runs_made(self) -> int:
        """The runs this runtime's kernels made (a skipped gated run doesn't count; a sync)."""
        return int(self.ev[(self.run_id + 1) % 2, 1]) if self.run_id else 0

    def sync_ring_pos(self) -> None:
        """The ring's delivery position from the device (after planned gated runs: only their kernels know
        which ran; a sync)."""
        self.ring.ring_pos = int(self.ev[(self.run_id + 1) % 2, 0])

    def next_arrival(self) -> float:
        """A queue population (queues): its queues' next arrival after its last run -- every group's earliest head's
        e + D_c, from the run's event state (fused.py _ev_close); inf: nothing queued. A readback."""
        if self.run_id == 0:
            return math.inf
        row = self.ev[1 - self.run_id % 2, 2:].view(torch.float32)
        row = torch.where(torch.isnan(row), torch.full_like(row, math.inf), row)
        return float(row.min()) if len(row) else math.inf

    def delivers_until(self, t: float) -> int:
        """The last ring slot a run delivering up to time t delivers (its slots end by t)."""
        return int((t + 1e-6) // self.ring.slot_width) - 1

    # ---- settling, checking

    def settle(self) -> None:
        """Before reading what runs wrote (on the host): every runtime's runs complete -- a stopped one rerun
        (Recovery.settle)."""
        if self.device.type == "cuda":
            self.recovery.settle()

    def check(self) -> None:
        """Judge every run's ring status (waiting for them: settle(); a
        stopped run rerun), then raise on what remains flagged: far-future
        arrivals, the uniform contract, history overrun."""
        self.settle()
        self.ring.check()
        if int(self.overrun[0]):
            raise RuntimeError("a consumer fell behind a source's emission history: a run was overwritten "
                               "before it was read (raise the history depth K)")
        if int(self.history.overflow[0]):  # (D-3: a neuron can fire more than once a run, inside delivery)
            raise RuntimeError(f"a run emitted more than its history's {self.history.cap} spikes: the rest were "
                               f"lost (raise Population.history_cap; TODO.md, D-3)")

    # ---- inspection

    def _arena_check_buf(self) -> torch.Tensor:
        """The arena check's buffer (fused.py: ARENA_CHECK): 16 words of the first violation, then a held count
        per (group, block) -- rebuilt from the ring's tables whenever the ring's buffer or arena size changed
        (an allocation, a resize: the device is idle then), the violation kept. Unchecked: a dummy word."""
        if not (self.arena_check and self.ring is not None):
            if self._chk is None or self._chk.numel() != 16:
                self._chk = torch.zeros(16, dtype=torch.int32, device=self.device)
            return self._chk
        ring = self.ring
        key = (ring.buf.data_ptr(), ring.n_blocks)
        if self._chk_key != key:
            g, S, nb = ring.n_groups, ring.n_slots, ring.n_blocks
            dev = self.device
            held = torch.zeros(g * nb, dtype=torch.int32, device=dev)
            group = torch.arange(g * S, device=dev) // S
            col = torch.arange(nb, device=dev)
            used = (col[None, :] < ring.nblk.long()[:, None]) & (ring.table[:, :nb] >= 0)
            idx = (group[:, None] * nb + ring.table[:, :nb].long().clamp(min=0))[used]
            held.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
            errors = self._chk[:16] if self._chk is not None else torch.zeros(16, dtype=torch.int32, device=dev)
            self._chk = torch.cat([errors, held])  # (on the current stream: the next launches after it)
            self._chk_key = key
        return self._chk

    def raise_on_arena_check(self, bits: int) -> None:
        """(Ledger.judge) A judged run's lost bits with the arena check's (4): raise with the violation it
        recorded -- before the device faults on what the broken arena does next (the host still reads it then)."""
        if bits & 4:
            raise RuntimeError(f"the arena check found a violation in a ring ({self.n} neurons, run "
                               f"{self.run_id}): {self.arena_errors()}")

    def arena_errors(self) -> dict | None:
        """The arena check's first violation (arena_check): its kind (pop: a popped block already held, or out
        of the arena; free: a freed block not held), group, cell, block id, the count found, the round (pop) or
        slot (free), and how many there were -- None if none."""
        if self._chk is None:
            return None
        v = self._chk[:7].tolist()
        if not v[0]:
            return None
        return dict(count=v[0], kind={1: "pop", 2: "free"}.get(v[1], v[1]), group=v[2], cell=v[3], block=v[4],
                    found=v[5], when=v[6])

    def set_trace(self, cap: int | None = 1024, level: int = 1):
        """Stamp the following runs (naer2/trace.py); level 2 also inside the pull, 3 inside the append; None: stop."""
        self.trace = Trace(self.n_groups, cap, self.device, level=level) if cap else None
        self._launchers = {}

    def traced(self) -> list[dict]:
        return self.trace.decode(FUSED_TRACE_TAGS)

    def deliveries(self) -> list[tuple[int, float, int, int, int]]:
        """The delivery log, in log order: (slot, e, connection, source, target)."""
        self.settle()
        n = min(int(self.log[0]), self.log_cap)
        rec = self.log[1:1 + 4 * self.log_cap].view(-1, 4)[:n].tolist()
        e = self.log[1 + 4 * self.log_cap:].view(torch.float32)[:n].tolist()
        return [(k, ei, c, s, t) for (k, c, s, t), ei in zip(rec, e)]

    def emitted(self) -> list[tuple[float, int]]:
        """The last run's spikes, [(e, node)] in flat-list order."""
        self.settle()
        return self.history.spikes(self.run_id)
