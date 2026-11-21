"""
Captured windows: a planned window's launches (Network.plan -> naer2/schedule.py) recorded once as a device
graph and replayed for every window of the same shape -- which populations run in which round, gated or not
-- the runs' own values (run ids, rounds, slots, windows) in entries the kernels read, written before each
replay. torch.cuda.CUDAGraph: CUDA graphs on NVIDIA, HIP graphs on ROCm; Triton launches on the current
stream, so stream capture records them (the populations' streams joined through the round's events: one
graph branch per population, a round's runs concurrent).

  WindowGraph(network, schedule)   captures the window twice (nothing runs yet), each capture with its own
                                   entry buffers, one per runtime, at the addresses that graph holds
  .replay(schedule)                the next capture's turn: the schedule's entries into its buffers (a copy
                                   from pinned staging, on the replay's stream: after the previous replay),
                                   then its graph

The runs' statuses (fused.py: _run_status) go into device memory, each run its own row (its entry's status
slot: its index), and the window's last node copies them into pinned host memory, where the host judges them
-- not stored straight into host memory by every program as a direct run's are: on NVIDIA a kernel finishing
with host-bound stores outstanding stalls the kernels still running for ~15-20 us (PD14 on the 4090, a
round's GPU time 65 -> 33 us; tools/pd14_graph_gpu.py --slowest, --status-device).

Ring resizing stays outside a window: a run too large for its ring stops (and is rerun once the window's
statuses are judged, the ring grown); a resized ring's new buffer means a new capture.
"""

import numpy as np
import torch

from .runtime import _Window, _WindowPart


class WindowGraph:
    """Two captures of one window shape, each with its own entry buffers, replayed in turn: a window's
    entries stay as they were until the window after next (its statuses judged by then -- a stopped run's
    rerun reads them)."""

    def __init__(self, network, schedule, barrier: bool = False, fused=None):
        """barrier: a tiny kernel on the capture stream between rounds -- every round's kernels before it,
        every next round's kernels after it (9 + 9 edges a round, not 9 x 9: what a graph is instantiated
        from, its edges -- torch has no empty node to stand in). fused: a FusedRound (naer2/fusion.py) --
        every round it covers one kernel on the capture stream (no fork, no join, no barrier needed)."""
        device = network.device
        self._bar = torch.zeros(1, dtype=torch.int32, device=device) if barrier else None
        self.addresses = {key: self._addresses(rt) for key, rt in self._runtimes(schedule).items()}
        runtimes = self._rts = self._runtimes(schedule)
        self.buffers, self.staging, self.graphs = [], [], []
        self.status, self.status_pinned, self.status_host = [], [], []  # (device rows; their host copy; its numpy)
        self._slots = {key: np.arange(len(rows), dtype=np.int32) for key, rows in schedule.rows.items()}
        torch.cuda.synchronize()
        for _ in range(2):
            buffers = {key: torch.from_numpy(rows).to(device) for key, rows in schedule.rows.items()}
            # per runtime, a row per run: written by the kernels, copied to the host by the window's last node
            status = {key: torch.zeros(len(rows), runtimes[key].n_groups, 4, dtype=torch.int32, device=device)
                      for key, rows in schedule.rows.items()}
            status_host = {key: torch.zeros(st.shape, dtype=torch.int32).pin_memory() for key, st in status.items()}
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                if barrier:  # (one root node: the first round's kernels after it, as every later round's)
                    self._bar.add_(1)
                for runs in schedule.launches:
                    if fused is not None and fused.covers([pop._runtime for pop, _, _ in runs]):
                        by_rt = {id(pop._runtime): (i, run_id) for pop, i, run_id in runs}
                        fused.launch([rt.entry_params(buffers[id(rt)], *by_rt[id(rt)], status=status[id(rt)])
                                      for rt in fused.runtimes], torch.cuda.current_stream())
                        continue
                    network._concurrently(
                        (pop, lambda stream, pop=pop, i=i, run_id=run_id, b=buffers, st=status:
                         pop._runtime.launch_entry(b[id(pop._runtime)], i, run_id, stream, bookkeeping=False,
                                                   status=st[id(pop._runtime)]))
                        for pop, i, run_id in runs)
                    if barrier:
                        self._bar.add_(1)
                for key, st in status.items():  # (after every run: the round's join, or the last barrier)
                    status_host[key].copy_(st, non_blocking=True)
            self.status.append(status)
            self.status_pinned.append(status_host)
            self.status_host.append({key: h.numpy() for key, h in status_host.items()})
            self.buffers.append(buffers)
            self.staging.append({key: torch.from_numpy(rows.copy()).pin_memory() for key, rows in schedule.rows.items()})
            self.graphs.append(graph)
        self._turn = 0

    @staticmethod
    def _addresses(rt) -> tuple:
        """What the graph holds of a runtime by address: its ring's buffer and its state tensors (a resize, or
        a state tensor replaced between windows, means a new capture)."""
        return (rt.ring.buf.data_ptr(),) + tuple(t.data_ptr() for t in rt.state.values())

    @staticmethod
    def _runtimes(schedule) -> dict:
        return schedule.runtimes

    def valid(self, schedule) -> bool:
        """Whether the captured graph still fits: every runtime's ring and state where it was captured."""
        return all(self._addresses(rt) == self.addresses.get(key) for key, rt in self._runtimes(schedule).items())

    def replay(self, schedule) -> None:
        turn, self._turn = self._turn, self._turn ^ 1
        buffers = self.buffers[turn]
        for key, rows in schedule.rows.items():  # (the copy on the replay's stream: after the last replay)
            stage = self.staging[turn][key]
            np.copyto(stage.numpy(), rows)
            stage.numpy()[:, 7] = self._slots[key]  # (each run's status row: its entry's index)
            buffers[key].copy_(stage, non_blocking=True)
        self.graphs[turn].replay()
        # the window's runs pending as one per runtime (judged together), one journal entry for a rerun
        status, status_host = self.status[turn], self.status_host[turn]
        window = _Window(lambda: [(pop._runtime, i, run_id) for runs in schedule.launches for pop, i, run_id in runs])
        for key, rows in schedule.rows.items():
            rt = self._rts[key]
            rt.note_window(_WindowPart(window, rows[:, 0].astype(np.int64), self._slots[key], buffers[key],
                                       status[key], status_host[key]))
        next(iter(self._rts.values()))._launches.journal[("window", id(window))] = window
