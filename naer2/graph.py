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

Ring resizing stays outside a window: a run too large for its ring stops (and is rerun once the window's
statuses are judged, the ring grown); a resized ring's new buffer means a new capture.
"""

import numpy as np
import torch


class WindowGraph:
    """Two captures of one window shape, each with its own entry buffers, replayed in turn: a window's
    entries stay as they were until the window after next (its statuses judged by then -- a stopped run's
    rerun reads them)."""

    def __init__(self, network, schedule):
        device = network.device
        self.addresses = {key: self._addresses(rt) for key, rt in self._runtimes(schedule).items()}
        self.buffers, self.staging, self.graphs = [], [], []
        torch.cuda.synchronize()
        for _ in range(2):
            buffers = {key: torch.from_numpy(rows).to(device) for key, rows in schedule.rows.items()}
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for runs in schedule.launches:
                    network._concurrently(
                        (pop, lambda stream, pop=pop, i=i, run_id=run_id, b=buffers: pop._runtime.launch_entry(
                            b[id(pop._runtime)], i, run_id, stream, bookkeeping=False))
                        for pop, i, run_id in runs)
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
        return {id(pop._runtime): pop._runtime for runs in schedule.launches for pop, _, _ in runs}

    def valid(self, schedule) -> bool:
        """Whether the captured graph still fits: every runtime's ring and state where it was captured."""
        return all(self._addresses(rt) == self.addresses.get(key) for key, rt in self._runtimes(schedule).items())

    def replay(self, schedule) -> None:
        turn, self._turn = self._turn, self._turn ^ 1
        buffers = self.buffers[turn]
        for key, rows in schedule.rows.items():  # (the copy on the replay's stream: after the last replay)
            stage = self.staging[turn][key]
            np.copyto(stage.numpy(), rows)
            buffers[key].copy_(stage, non_blocking=True)
        self.graphs[turn].replay()
        for runs in schedule.launches:
            for pop, i, run_id in runs:
                pop._runtime.note_run(buffers[id(pop._runtime)], i, run_id)
