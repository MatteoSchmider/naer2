"""
A runtime's runs between launch and known complete: their launch parameters (RunParams), the status slots
their kernels write, and the judging of what they wrote.

Without a host sync: every run's kernel writes its status straight into one of `slots` pinned host slots (per
group: the blocks it holds, the most it logged, lost bits, done -- fused.py: _run_status; no copy, no event); a
replayed window's runs into device rows its graph copies to the host (naer2/graph.py: one WindowPart per
runtime). Later runs judge every slot that has landed, in run order, and a run whose slot is still pending
waits for it -- so the ring's size follows the runs at most `slots` behind (settle() judges them all). A run
judged complete leaves the journal (naer2/sim/recovery.py) and its need goes to the runtime's sizer
(naer2/sim/sizing.py); a stopped one starts a recovery.
"""

import collections
import dataclasses
import time

import numpy as np
import torch

from ..schedule import RUN_WORDS


@dataclasses.dataclass(frozen=True, slots=True)
class RunParams:
    """One run's launch: entry `index` of schedule `sched` (a runtime's direct schedule, a planned window's
    entries: naer2/schedule.py), its run id, its warps, the state tensors as launched (a rerun reads what this
    launch read, whatever replaces them later), and where its kernel writes its status (None: the runtime's
    host slots)."""

    sched: torch.Tensor
    index: int
    run_id: int
    num_warps: int
    state: tuple
    status: torch.Tensor | None = None


class WindowPart:
    """One runtime's runs of a replayed window (naer2/graph.py), pending as one: their run ids (in order), their
    entries' indices in the window's entry buffer, the status rows its kernels wrote (device) and the window's
    last node copied to the host -- judged together."""

    __slots__ = ("window", "run_ids", "entries", "sched", "status", "status_host")

    def __init__(self, window, run_ids, entries, sched, status, status_host):
        self.window, self.run_ids, self.entries = window, run_ids, entries
        self.sched, self.status, self.status_host = sched, status, status_host

    def __len__(self) -> int:
        return len(self.run_ids)


class Ledger:
    """See the module docstring. owner: the runtime (its recovery, its ring's sizing: PopulationRuntime.ran)."""

    def __init__(self, owner, n_groups: int, device: torch.device, slots: int = 4):
        self.owner, self.n_groups, self.cuda = owner, n_groups, device.type == "cuda"
        self.pending = collections.deque()  # run ids and WindowParts, in run order
        self.alloc(slots)

    # ---- status slots

    def alloc(self, n: int) -> None:
        """n slots of the runs' status, written by the kernel straight into pinned host memory -- per slot,
        per group [blocks held, most logged, lost bits, done] (fused.py: _run_status), a run's slot its run
        id mod n -- and as many of the direct schedule (PopulationRuntime.run's entries, read by the kernel from
        pinned host memory too: naer2/schedule.py). The runs still pending are judged first."""
        if self.pending:
            self.judge_pending(wait=True)
        self.slots = n
        self.status_host = torch.zeros(n, self.n_groups, 4, dtype=torch.int32, pin_memory=self.cuda)
        self._status_np = self.status_host.numpy()
        self.sched_host = torch.zeros(n, RUN_WORDS, dtype=torch.int32, pin_memory=self.cuda)
        self.sched_np = self.sched_host.numpy()

    def ensure(self, n: int) -> None:
        """At least n slots (a power of two)."""
        if self.slots < n:
            self.alloc(1 << (n - 1).bit_length())

    def slot(self, run_id: int) -> int:
        return run_id % self.slots

    # ---- runs pending

    def launched(self, params: RunParams) -> None:
        """A direct or planned run launched: pending, journaled for a rerun."""
        if self.cuda:
            self.owner.recovery.launched(self.owner, params)
            self.pending.append(params.run_id)

    def launched_window(self, part: WindowPart) -> None:
        """A replayed window's runs of this runtime: pending as one, judged together; the window's journal
        entry (Window) holds them for a rerun."""
        part.window.parts[id(self.owner)] = part
        self.pending.append(part)

    def clear(self) -> None:
        self.pending.clear()

    def n_pending(self) -> int:
        return sum(len(p) if isinstance(p, WindowPart) else 1 for p in self.pending)

    def free_slot(self) -> None:
        """Before a run that writes a status slot: the slot judged free (the oldest pending runs judged)."""
        if self.cuda:
            while len(self.pending) >= self.slots:
                self.judge(self.pending.popleft())

    def make_room(self, n: int) -> None:
        """Status slots for n more runs: the oldest pending runs judged (waited for) until n fit."""
        while self.pending and self.n_pending() + n > self.slots:
            self.judge(self.pending.popleft())

    def judge_windows(self, keep: int = 1) -> None:
        """Replayed windows pending at most `keep` (judged, waited for, oldest first, with any runs before
        them): a capture turn's buffers (its entries, its statuses) are the window's until it is judged, and
        every other window reuses them (naer2/graph.py) -- a window two turns back must be judged before the
        next replay writes its entries."""
        while sum(isinstance(p, WindowPart) for p in self.pending) > keep:
            self.judge(self.pending.popleft())

    def judge_pending(self, wait: bool) -> None:
        """Every pending run whose status has landed (wait: every one), in run order."""
        while self.pending and (wait or self._landed(self.pending[0])):
            self.judge(self.pending.popleft())

    # ---- judging

    def _landed(self, run) -> bool:
        """Every group of run `run` wrote its done word (finished or stopped) into the run's host slot
        (fused.py: _run_status -- 2 run_id, + 1 if stopped; the slot's earlier runs wrote less); a window part:
        every group of every one of its runs, into the window's host copy."""
        if isinstance(run, WindowPart):
            return bool((run.status_host[:, :, 3] >= 2 * run.run_ids[:, None]).all())
        return int(self._status_np[run % self.slots, :, 3].min()) >= 2 * run

    def _wait_landed(self, run) -> None:
        """Until run `run`'s status has landed (a run id, or a window part): polling the host a little, then
        waiting for the device (its kernels done, their stores visible) -- a kernel that didn't write raises,
        it doesn't hang."""
        deadline = time.perf_counter() + 2e-3
        while not self._landed(run):
            if time.perf_counter() > deadline:
                torch.cuda.synchronize()
                if not self._landed(run):
                    what = (f"runs {int(run.run_ids[0])}-{int(run.run_ids[-1])}" if isinstance(run, WindowPart)
                            else f"run {run}")
                    raise RuntimeError(f"{what}' kernels finished without writing their status")
                return

    def judge(self, run) -> None:
        """One run's status (waited for); `run`: a run id, or a replayed window's part (its runs judged
        together -- the most of each column over all of them). Not every group finished -- this run stopped
        (its ring too small), or another one before it did: a recovery. Finished: the run's launch is
        complete, and its need goes to the owner (PopulationRuntime.ran)."""
        self._wait_landed(run)
        owner = self.owner
        if isinstance(run, WindowPart):
            rows = run.status_host
            run_id, n_runs = int(run.run_ids[-1]), len(run)
        else:
            rows = self._status_np[run % self.slots]
            run_id, n_runs = run, 1
        # per column, the most any group (of any run) wrote: blocks held, records logged, done (odd: stopped);
        # the lost bits of any
        blocks, most, _, _ = (int(v) for v in rows.reshape(-1, 4).max(0))
        lost = int(np.bitwise_or.reduce(rows[..., 2], axis=None))
        owner.raise_on_arena_check(lost)
        if (rows[..., 3] & 1).any():
            owner.recovery.recover()
            return
        if isinstance(run, WindowPart):
            owner.recovery.window_complete(run.window, owner)
        else:
            owner.recovery.complete(owner, run)
        owner.ran(blocks, most, lost, run_id, n_runs)
