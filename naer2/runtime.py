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

Ring size, without a host sync: every run's kernel writes its status
straight into one of status_slots pinned host slots (per group: the
blocks it holds, the most it logged, lost bits, done -- fused.py:
_run_status; no copy, no event); later runs judge every slot that has
landed, in run order, and a run whose slot is still pending waits for it
-- so a size follows the runs at most status_slots behind (check() judges
them all).
  - per-cell runs: a run kind (main, side) past its capacity lost the
    links it couldn't take; its capacity doubles (Ring.grow)
  - pooled: the arenas (blocks per group) and overflow logs (records per
    cell) follow the runs' need -- the most blocks any group held, the
    most records any cell logged (Little's law at work: the arrival rate x
    how long a record stays) -- with headroom and hysteresis: grow when
    need x grow_at exceeds the size, shrink when need x grow_at stays below
    half of it for shrink_window runs, either to need x grow_to
    (Ring.resize). A run that ran out lost links (its need: at least twice
    what it had).
Lost links are reported with a warning. Every size is a runtime value in
the ring's header: a resize keeps the compiled kernel and its launcher.

The launch: the first run per configuration goes through the JIT (and
compiles); later ones call the compiled kernel's launcher directly (~8 us
against ~50 us for the JIT's dispatch with this signature,
tools/experiments/launch_cost.py). The cache is keyed by everything that
selects the compiled variant and dropped when the buffers change.
"""

import collections
import contextlib
import math
import time
import warnings
import weakref

import numpy as np
import torch
import triton
from triton import knobs

from .history import EmissionHistory
from .ring import POOL_MAX_BLOCKS, POOL_RESERVE, Ring
from .schedule import RUN_WORDS, write_entry
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


def _warp_cap(device) -> int:
    """The most warps a program gets by default: 4, on every device for now (a per-device choice -- an
    autotuning at startup -- is a later feature, TODO.md). Measured: PD14 at G=256 replayed from captured
    windows on the 4090 ~29 us a round at 4 warps against ~35-40 at 8 (4 warps: a round's 309 programs all
    resident at once, 4 a multiprocessor by registers; 8: 2, the rest a second wave -- tools/pd14_occupancy.py);
    the 780M ~425 us stepped at 4 against ~470 at 8. The RX 6600 (14 WGPs) was faster at 8 when stepped
    (~280 us a round against ~465 at 4, host-bound): its loss, until the warps are chosen per device."""
    return 4


class _WindowPart:
    """One runtime's runs of a replayed window (naer2/graph.py), pending as one: their run ids (in order), their
    entries' indices in the window's entry buffer, the status rows its kernels wrote (device) and the window's
    last node copied to the host -- judged together (PopulationRuntime._judge_part)."""

    __slots__ = ("window", "run_ids", "entries", "sched", "status", "status_host")

    def __init__(self, window, run_ids, entries, sched, status, status_host):
        self.window, self.run_ids, self.entries = window, run_ids, entries
        self.sched, self.status, self.status_host = sched, status, status_host

    def __len__(self) -> int:
        return len(self.run_ids)


class _Window:
    """A replayed window's runs as one journal entry: per runtime its part, and the order its runs were launched
    in (round by round) -- a recovery reruns the parts not yet judged complete, in that order."""

    def __init__(self, order):
        self._order = order  # () -> [(runtime, entry index, run id)], round by round (asked for on a rerun only)
        self.parts: dict = {}  # runtime id -> _WindowPart, until judged complete

    def runs(self):
        """(runtime, launch params) of every run of a part still pending, in launch order."""
        for rt, i, run_id in self._order():
            part = self.parts.get(id(rt))
            if part is not None:
                yield rt, rt.entry_params(part.sched, i, run_id, status=part.status)


class _Launches:
    """Every runtime's launches on one device, in launch order, each kept until its run is known complete
    (its status judged: every group done), and the stop word ctl the kernels share: a run whose ring is too
    small for it sets it and stops without committing anything (fused.py: _pull), and every run that starts
    after it stops at once (_run_gate) -- this stream's and every other's. recover() brings them all back:
    once no kernel runs any more, every ring that stopped a run grows to what it needs, and every launch not
    known complete runs again, in launch order (its groups that finished skip), the device idle again
    after. Nothing is lost, and the result is what it would have been with the larger ring from the start."""

    def __init__(self, device):
        self.ctl = torch.zeros(1, device=device, dtype=torch.int32)
        # (runtime id, run id) -> (runtime, launch params); ("window", id) -> a replayed window's runs (_Window)
        self.journal = collections.OrderedDict()
        self.runtimes = weakref.WeakSet()
        self.recoveries = 0
        self._resizing = False

    def resize(self) -> None:
        """Every resize a runtime asked for (PopulationRuntime._want), together: once no kernel runs on the
        device any more (a precaution: reallocating while other streams' kernels ran faulted the GPU, see
        TODO.md) -- one drain for all of them, every runtime's landed statuses judged first (any resize they
        want joins in)."""
        if self._resizing:
            return  # (a resize or recovery under way applies it)
        self._resizing = True
        try:
            torch.cuda.synchronize()
            for rt in list(self.runtimes):
                rt._judge_pending(None, wait=True)
            for rt in list(self.runtimes):
                rt._apply_wanted()
        finally:
            self._resizing = False

    def recover(self) -> None:
        resizing, self._resizing = self._resizing, True
        try:
            self._recover()
        finally:
            self._resizing = resizing

    def _recover(self) -> None:
        while True:
            torch.cuda.synchronize()  # nothing runs from here on: the rings can be reallocated
            for rt in list(self.runtimes):
                rt._pending.clear()  # (stale: their runs stopped, or are rerun below)
            for rt in list(self.runtimes):
                rt._grow_stopped()
                rt._apply_wanted()
            self.ctl.zero_()
            for entry in list(self.journal.values()):
                for rt, params in (entry.runs() if isinstance(entry, _Window) else (entry,)):
                    rt._launch(params, None)
            torch.cuda.synchronize()
            self.recoveries += 1
            if not int(self.ctl[0]):
                break
        self.journal.clear()

    def settle(self) -> None:
        """Every runtime's staged statuses judged (waiting for them): a stopped run is rerun before the host
        reads what the runs wrote (a history, a ring) -- every host readback of device state goes through here."""
        for rt in list(self.runtimes):
            rt._judge_pending(None, wait=True)


_LAUNCHES: dict = {}


def _launches(device) -> _Launches:
    """The launches of every runtime on this device (one stop word, one journal: their runs read each
    other's emission histories)."""
    key = (device.type, device.index if device.index is not None else torch.cuda.current_device()
           if device.type == "cuda" else 0)
    if key not in _LAUNCHES:
        _LAUNCHES[key] = _Launches(device)
    return _LAUNCHES[key]


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
        self.max_warps: int | None = None  # a program's warps at most (None: the device's cap, _warp_cap)
        self.maxnreg: int | None = None  # NVIDIA: registers per thread at most (None: the compiler's choice)
        self.seed = 0  # random.*'s stream (the population's own seed: Population.seed)
        self.batch = PULL["batch"]
        self.trace = None
        self.overrun = torch.zeros(1, device=self.device, dtype=torch.int32)
        self._launchers = {}
        self._launches = _launches(self.device)
        self._launches.runtimes.add(self)
        # the ring's status after each run, copied into one of status_slots pinned host slots on its stream and
        # judged once landed, in run order (see the module docstring); a run whose slot is still in flight waits
        # for it -- so the host runs at most status_slots runs ahead of what the sizing has seen
        self.status_slots = 4
        # the rings' sizing (pooled: blocks per arena, records per overflow log; per-cell: the main run
        # capacity), from every run's need: grow when need x grow_at exceeds the size; shrink (pooled) when
        # need x grow_at stays below half the size for shrink_window runs; either way to need x grow_to
        self.grow_at, self.grow_to, self.shrink_window = 1.5, 2.0, 200
        self.growths: list[tuple[int, str, int]] = []  # (run id, "main" / "side" / "arena" / "log", new size)
        self.resizes: list[tuple[int, int, int]] = []  # pooled: (run id, blocks per arena, log records per cell)
        self.stops: list[int] = []  # run ids that stopped (the ring too small) and were rerun
        self._pending = collections.deque()
        self._wanted = None  # pooled: (blocks, log records, run id) for the next _Launches.resize
        signature = kernel.kernel.params
        self._constexpr_names = [p.name for p in signature if p.is_constexpr]
        self.connect([], [], n_slots=1)

    @property
    def uniform_emission(self) -> bool:
        return self.kernel.uniform_emission

    def connect(self, sources: list, topologies: list, n_slots: int, slot_width: float = 0.1, log_cap: int = 0,
                main_cap: int | None = None, side_cap: int | None = None, pool: tuple[int, int] | None = None,
                min_delays: list[float] | None = None):
        """pool: (block, blocks per group) -- the runs in per-group arenas (Ring); None: per-cell runs.
        min_delays: per connection its minimum delay as the host schedules with it (a gated run's due time;
        default 0)."""
        caps = {k: v for k, v in (("main_cap", main_cap), ("side_cap", side_cap)) if v}
        self.ring = Ring(self.n_groups, n_slots, slot_width, self.device, pool=pool, **caps)
        self.wmark = torch.full((self.n_groups,), -math.inf, device=self.device)
        self.plan = FusedPlan([s.history for s in sources], topologies, self.n_groups, self.device,
                              min_delays=min_delays)
        # the event state (fused.py: _gate_skip, _ev_close): per run parity the delivery position, the runs
        # made, per group its first slot still holding records
        self.ev = torch.full((2, 2 + self.n_groups), 2147483647, device=self.device, dtype=torch.int32)
        self.ev[:, :2] = 0
        self.uniform_sources = all(s.uniform_emission for s in sources)
        self.log_cap = log_cap
        self.log = torch.zeros(1 + 5 * max(log_cap, 1), device=self.device, dtype=torch.int32)
        self._launchers = {}  # new buffers (their sizes are part of the AMD backend's specialization)
        self._judge_pending(None, wait=True)  # (the old ring's)
        self._alloc_slots(self.status_slots)
        # the most blocks / log records any run needed, and the runs judged, since the last shrink check
        self._window = [0, 0, 0]

    def _alloc_slots(self, n: int) -> None:
        """n slots of the runs' status, written by the kernel straight into pinned host memory -- per slot,
        per group [blocks held, most logged, lost bits, done] (fused.py: _run_status), a run's slot its run
        id mod n -- and as many of the direct schedule (run()'s entries, read by the kernel from pinned host
        memory too: naer2/schedule.py). The runs still pending are judged first."""
        self._judge_pending(None, wait=True)
        cuda = self.device.type == "cuda"
        self.status_slots = n
        self._status_host = torch.zeros(n, self.n_groups, 4, dtype=torch.int32, pin_memory=cuda)
        self._status_np = self._status_host.numpy()
        self._sched_host = torch.zeros(n, RUN_WORDS, dtype=torch.int32, pin_memory=cuda)
        self._sched_np = self._sched_host.numpy()

    def run(self, t_start, t_end, deliver_until=None, round_id=None, num_warps=None, stream=None):
        """One run over [t_start, t_end) in round `round_id` (default: the
        run id): take in what the sources emitted in earlier rounds, deliver
        every ring slot that ends by `deliver_until` (default t_end), update,
        send, emit. stream: a torch.cuda.Stream to launch on (default: the
        current one). num_warps: default one lane per neuron, up to 8 warps;
        never more lanes than the group has neurons (see _warps)."""
        num_warps = self._warps(num_warps)
        self._judge_pending(stream, wait=False)
        cuda = self.device.type == "cuda"
        if cuda:
            while len(self._pending) >= self.status_slots:  # the slot this run's kernel writes: judged first
                self._judge(self._pending.popleft())
        self.run_id += 1
        k_lo, k_hi = self.ring.ring_pos, self.delivers_until(deliver_until if deliver_until is not None else t_end)
        slot = self.run_id % self.status_slots  # (free: its previous run judged above)
        write_entry(self._sched_np[slot], self.run_id, round_id if round_id is not None else self.run_id, k_lo,
                    k_hi, t_start, t_end)
        # (the state tensors as launched: a rerun reads what this launch reads, whatever replaces them later)
        params = (self._sched_host, slot, self.run_id, num_warps, tuple(self.state[a] for a in self.kernel.attrs))
        self._launch(params, stream)
        self.ring.ring_pos = max(k_lo, k_hi + 1)
        if cuda:
            self._launches.journal[(id(self), self.run_id)] = (self, params)
            self._pending.append(self.run_id)

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

    def runs_made(self) -> int:
        """The runs this runtime's kernels made (a skipped gated run doesn't count; a sync)."""
        return int(self.ev[(self.run_id + 1) % 2, 1]) if self.run_id else 0

    def sync_ring_pos(self) -> None:
        """The ring's delivery position from the device (after planned gated runs: only their kernels know
        which ran; a sync)."""
        self.ring.ring_pos = int(self.ev[(self.run_id + 1) % 2, 0])

    def delivers_until(self, t: float) -> int:
        """The last ring slot a run delivering up to time t delivers (its slots end by t)."""
        return int((t + 1e-6) // self.ring.slot_width) - 1

    def launch_entry(self, sched: torch.Tensor, i: int, run_id: int, stream=None, num_warps=None,
                     bookkeeping: bool = True, status: torch.Tensor | None = None) -> None:
        """A planned run: entry i of a schedule (naer2/schedule.py) -- everything the run needs besides
        this runtime's own buffers is in it; run_id is the entry's (for the status bookkeeping).
        bookkeeping False: the launch alone (a graph capture: the bookkeeping follows each replay, note_window).
        status: where the kernel writes its status (the entry's slot in it; None: the runtime's host slots)."""
        if not bookkeeping:
            self._launch(self.entry_params(sched, i, run_id, num_warps, status), stream)
            return
        self.free_slot()
        params = self.entry_params(sched, i, run_id, num_warps)
        self._launch(params, stream)
        if self.device.type == "cuda":
            self._launches.journal[(id(self), run_id)] = (self, params)
            self._pending.append(run_id)

    def entry_params(self, sched: torch.Tensor, i: int, run_id: int, num_warps=None, status=None) -> tuple:
        """A planned run's launch params (_launch, _launch_args): entry i of schedule sched; status: where the
        kernel writes its status (None: the runtime's host slots). The state tensors as of now: a rerun reads
        what this launch reads, whatever replaces them later."""
        return (sched, i, run_id, self._warps(num_warps), tuple(self.state[a] for a in self.kernel.attrs), status)

    def free_slot(self) -> None:
        """Before a run that writes a status slot: the slot judged free (the oldest pending runs judged)."""
        if self.device.type == "cuda":
            while len(self._pending) >= self.status_slots:  # the slot this run's kernel writes: judged first
                self._judge(self._pending.popleft())

    def note_window(self, part: _WindowPart) -> None:
        """A replayed window's runs of this runtime (naer2/graph.py): pending as one, judged together
        (_judge_part); the window's journal entry (_Window) holds them for a rerun."""
        part.window.parts[id(self)] = part
        self._pending.append(part)

    def _n_pending(self) -> int:
        return sum(len(p) if isinstance(p, _WindowPart) else 1 for p in self._pending)

    def make_room(self, n: int) -> None:
        """Status slots for n more runs: the oldest pending runs judged (waited for) until n fit."""
        while self._pending and self._n_pending() + n > self.status_slots:
            self._judge(self._pending.popleft())

    def _launch_args(self, params) -> tuple[list, int, bool]:
        """One run's kernel arguments (the generated kernel's, constexprs aside: _constexprs), its warps, and
        whether its state tensors are 16-byte aligned (the specialization every call meets) -- for its own
        kernel (_launch) or its share of a fused round's (naer2/fusion.py)."""
        sched, sched_i, run_id, num_warps, state = params[:5]
        status = params[5] if len(params) > 5 else None
        if status is None:
            status = self._status_host if self.device.type == "cuda" else self.overrun
        args = list(state) + [
            self.n, self.plan.plan, self.plan.n_conns, self.ring.buf, self.wmark, self.overrun, self._launches.ctl,
            status, self.status_slots, sched, sched_i,
            self.ev, self.ring.slot_width, self.received, self.n_received, self.log, self.log_cap,
            self.history.buf, self.history.cap, self.trace.buf if self.trace is not None else self.log, self.seed]
        return args, num_warps, all(not (t.data_ptr() & 15) for t in state)

    def _launch(self, params, stream) -> None:
        """One run's kernel launch: entry sched_i of schedule sched (run(): the direct schedule; a planned
        window: naer2/schedule.py; a rerun: _Launches.recover)."""
        args, num_warps, aligned = self._launch_args(params)
        # what selects the compiled variant, besides the plan, ring and history (fixed between connect()s)
        key = (num_warps, self.waves_per_eu, self.maxnreg, self.count_received, self.batch, self.stream_block,
               self.unroll, self.log_cap > 0)
        launch = self._launchers.get(key)
        if launch is not None and aligned:
            launch(args, stream.cuda_stream if stream is not None else None)
        else:
            consts = self._constexprs(num_warps)
            # waves_per_eu: an AMD backend option (the CUDA backend rejects it)
            # maxnreg: the NVIDIA backend's (fewer registers: more programs resident, maybe spills)
            amd = {"waves_per_eu": self.waves_per_eu} if torch.version.hip else (
                {"maxnreg": self.maxnreg} if self.maxnreg else {})
            with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                compiled = self.kernel.kernel[(self.n_groups,)](*args, **consts, num_warps=num_warps, **amd)
            if aligned:
                self._launchers[key] = _direct_launch(compiled, self.n_groups,
                                                      tuple(consts[c] for c in self._constexpr_names))

    # ---- the ring's status and size (see the module docstring)

    def _landed(self, run) -> bool:
        """Every group of run `run` wrote its done word (finished or stopped) into the run's host slot
        (fused.py: _run_status -- 2 run_id, + 1 if stopped; the slot's earlier runs wrote less); a window part
        (_WindowPart): every group of every one of its runs, into the window's host copy."""
        if isinstance(run, _WindowPart):
            return bool((run.status_host[:, :, 3] >= 2 * run.run_ids[:, None]).all())
        return int(self._status_np[run % self.status_slots, :, 3].min()) >= 2 * run

    def _wait_landed(self, run) -> None:
        """Until run `run`'s status has landed (a run id, or a window part): polling the host a little, then
        waiting for the device (its kernels done, their stores visible) -- a kernel that didn't write raises,
        it doesn't hang."""
        deadline = time.perf_counter() + 2e-3
        while not self._landed(run):
            if time.perf_counter() > deadline:
                torch.cuda.synchronize()
                if not self._landed(run):
                    what = (f"runs {int(run.run_ids[0])}-{int(run.run_ids[-1])}" if isinstance(run, _WindowPart)
                            else f"run {run}")
                    raise RuntimeError(f"{what}' kernels finished without writing their status")
                return

    def _judge_pending(self, stream, wait: bool) -> None:
        """Every pending run whose status has landed (wait: every one), in run order."""
        while self._pending and (wait or self._landed(self._pending[0])):
            self._judge(self._pending.popleft())

    def _judge(self, run) -> None:
        """One run's status (waited for); `run`: a run id, or a replayed window's part (_WindowPart: its runs
        judged together -- the most of each column over all of them). Not every group finished -- this run
        stopped (its ring too small), or another one before it did: _Launches.recover. Finished: the run's
        launch is complete, and the ring follows its need (_follow_need)."""
        self._wait_landed(run)
        if isinstance(run, _WindowPart):
            # per column, the most any group of any run wrote: blocks held, records logged, lost bits (any),
            # done (odd: stopped)
            blocks, most, lost, _ = (int(v) for v in run.status_host.reshape(-1, 4).max(0))
            if (run.status_host[:, :, 3] & 1).any():
                self._launches.recover()
                return
            window = run.window
            window.parts.pop(id(self), None)
            if not window.parts:
                self._launches.journal.pop(("window", id(window)), None)
            self._follow_need(blocks, most, lost, int(run.run_ids[-1]), len(run))
            return
        blocks, most, lost, done = (int(v) for v in self._status_np[run % self.status_slots].max(0))
        if done & 1:
            self._launches.recover()
            return
        self._launches.journal.pop((id(self), run), None)
        self._follow_need(blocks, most, lost, run, 1)

    def _follow_need(self, blocks: int, most: int, lost: int, run_id: int, n_runs: int) -> None:
        """The ring following what complete runs needed -- n_runs of them, run_id the last (pooled: the most
        blocks any group held, the most records any cell logged; per-cell: the longest run) -- see the
        module docstring. The general pull's lost links (a full side run, or a main run / arena out of room:
        it takes room one record at a time and can't stop) double that size."""
        ring = self.ring
        main_full = side_full = lost > 0  # (the general pull: either kind grows)
        if not ring.pool_b:
            longest = blocks
            main_cap = ring.main_cap
            if longest * self.grow_at > main_cap or main_full:
                main_cap = max(-(-math.ceil(longest * self.grow_to) // 4) * 4, 2 * main_cap if main_full else 0)
            if main_cap != ring.main_cap or side_full:
                self._grow_cells(main_cap, 2 * ring.side_cap if side_full else None, run_id, main_full or side_full)
            return
        if main_full:  # (the general pull: the arena ran out, links lost)
            blocks = max(blocks, 2 * ring.pool_nb)
        w = self._window
        w[0], w[1], w[2] = max(w[0], blocks), max(w[1], most), w[2] + n_runs
        nb, log_cap = self._sizes(blocks, most)
        grew = nb > ring.pool_nb or log_cap > ring.log_cap
        if not grew and w[2] >= self.shrink_window:
            floor_nb, floor_log = self._floors()
            if w[0] * self.grow_at < ring.pool_nb / 2:
                nb = max(math.ceil(w[0] * self.grow_to), floor_nb)
            if w[1] * self.grow_at < ring.log_cap / 2:
                log_cap = max(math.ceil(w[1] * self.grow_to), floor_log)
            self._window = [0, 0, 0]
        if (nb, log_cap) != (ring.pool_nb, ring.log_cap):
            self._want(nb, log_cap, run_id)
            self._launches.resize()
        if main_full or side_full:
            ring.flags[0] = 0
            ring.flags[3] = 0
            warnings.warn(f"a ring ran out in run {run_id} (the general pull: it can't stop a run): the links it "
                          f"couldn't take were lost; the arenas are now {ring.pool_nb} blocks of {ring.pool_b} "
                          f"records", RuntimeWarning, stacklevel=4)

    def _floors(self) -> tuple[int, int]:
        """Pooled: the smallest arena (every slot its reserve, a partly used block, one more) and log (one
        block: a run hardly ever logs more in quiet activity, and a burst stops a run at worst)."""
        return self.ring.n_slots * (POOL_RESERVE + 1) + 1, self.ring.pool_b

    def _sizes(self, blocks: int, most: int) -> tuple[int, int]:
        """Pooled: the arena and log sizes for a run's need -- grown to need x grow_to where need x grow_at
        exceeds them, else as they are; once one grows, the other too wherever it has less than grow_to x
        its need (one resize, one copy of the arena, for both)."""
        ring = self.ring
        floor_nb, floor_log = self._floors()
        nb, log_cap = ring.pool_nb, ring.log_cap
        at = self.grow_to if blocks * self.grow_at > nb or most * self.grow_at > log_cap else None
        if at and blocks * at > nb:
            nb = max(math.ceil(blocks * self.grow_to), floor_nb)
        if at and most * at > log_cap:
            log_cap = max(math.ceil(most * self.grow_to), floor_log)
        return nb, log_cap

    def _want(self, nb: int, log_cap: int, run_id: int) -> None:
        """Pooled: the arenas and logs to these sizes, at the next _Launches.resize (or recovery)."""
        self._wanted = (nb, log_cap, run_id)

    def _apply_wanted(self) -> None:
        """(_Launches.resize / recover: the device idle) The wanted sizes (Ring.resize)."""
        if self._wanted is None:
            return
        (nb, log_cap, run_id), ring = self._wanted, self.ring
        self._wanted = None
        if nb < ring.pool_nb:  # never below what the groups hold now (at most what their last pull held)
            nb = max(nb, int(ring.stats[:, 0].max()))
        if (nb, log_cap) == (ring.pool_nb, ring.log_cap):
            return
        grew = (nb > ring.pool_nb, log_cap > ring.log_cap)
        ring.resize(pool_nb=nb, log_cap=log_cap)
        self.resizes.append((self.run_id, nb, log_cap))
        if grew[0]:
            self.growths.append((run_id, "arena", ring.arena))
        elif grew[1]:
            self.growths.append((run_id, "log", log_cap))

    def _grow_stopped(self) -> None:
        """(_Launches.recover, the device idle) A ring that stopped a run grows to what that run needs, its
        flag cleared."""
        ring = self.ring
        if not int(ring.flags[4]):
            return
        self.stops.append(self.run_id)
        stats = ring.stats.cpu()
        if not ring.pool_b:
            longest = int(stats[:, 0].max())
            self._grow_cells(-(-math.ceil(longest * self.grow_to) // 4) * 4, None, self.run_id, False)
        else:
            blocks, most = int(stats[:, 0].max()), int(stats[:, 1].max())
            if blocks >= 1 << 24:
                raise RuntimeError(f"a ring run needs more than {POOL_MAX_BLOCKS} blocks of {ring.pool_b} records "
                                   f"in one slot (Population.ring_block: larger blocks)")
            self._want(*self._sizes(blocks, most), self.run_id)  # (applied by the recovery)
        ring.flags[4] = 0

    def _grow_cells(self, main_cap: int | None, side_cap: int | None, run_id: int, lost: bool) -> None:
        """Per-cell runs: larger capacities (Ring.grow), the device idle (see _Launches.resize)."""
        ring = self.ring
        torch.cuda.synchronize()
        ring.grow(main_cap=main_cap, side_cap=side_cap)
        ring.flags[0] = 0
        ring.flags[3] = 0
        for kind, cap in (("main", main_cap), ("side", side_cap)):
            if cap:
                self.growths.append((run_id, kind, cap))
        if lost:
            warnings.warn(f"a ring run overflowed in run {run_id} (the general pull: it can't stop a run): its "
                          f"links beyond the capacity were lost; the capacities are now main {ring.main_cap}, "
                          f"side {ring.side_cap}", RuntimeWarning, stacklevel=4)

    def _constexprs(self, num_warps) -> dict:
        p2, lanes = triton.next_power_of_2, 32 * num_warps
        consts = dict(
            GROUP_SIZE=self.group_size, K=self.history.K, SK=self.plan.K, KP=p2(self.plan.K),
            UNIFORM_SOURCES=self.uniform_sources, N_SLOTS=self.ring.n_slots,
            LOG=self.log_cap > 0, STREAM=self.stream_block, U=self.unroll,
            SPT=max(self.group_size // lanes, 1), NUM_WARPS=num_warps, MAX_CONNS=p2(max(self.plan.n_conns, 1)),
            LOG_CJ=(p2(max(self.plan.n_conns, 1)) * p2(self.plan.K)).bit_length() - 1,
            COUNT=self.count_received, BATCH=self.batch, NSP=p2(self.ring.n_slots),
            TRACE=self.trace.level if self.trace is not None else 0,
            TRACE_CAP=self.trace.cap if self.trace is not None else 1,
            POOL_B=self.ring.pool_b,
            **self.kernel.consts)
        return {c: consts[c] for c in self._constexpr_names if c in consts}

    def settle(self) -> None:
        """Before reading what runs wrote (on the host): every runtime's runs complete -- a stopped one rerun
        (_Launches.settle)."""
        if self.device.type == "cuda":
            self._launches.settle()

    def check(self) -> None:
        """Judge every run's ring status (waiting for them: settle(); a
        stopped run rerun), then raise on what remains flagged: far-future
        arrivals, the uniform contract, history overrun."""
        self.settle()
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
        self.settle()
        n = min(int(self.log[0]), self.log_cap)
        rec = self.log[1:1 + 4 * self.log_cap].view(-1, 4)[:n].tolist()
        e = self.log[1 + 4 * self.log_cap:].view(torch.float32)[:n].tolist()
        return [(k, ei, c, s, t) for (k, c, s, t), ei in zip(rec, e)]

    def emitted(self) -> list[tuple[float, int]]:
        """The last run's spikes, [(e, node)] in flat-list order."""
        self.settle()
        return self.history.spikes(self.run_id)
