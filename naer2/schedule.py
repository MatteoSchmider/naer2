"""
Run schedules: everything a run of a population's fused kernel needs that its runtime doesn't hold itself --
its run id and round, the ring slots it delivers, its window, whether it is gated -- as data the kernel
reads (fused.py: _run_entry), not as launch arguments. So a launch passes the same arguments every time but
the entry's index, and a planned window of runs can be filled ahead (and captured, replayed).

One entry is RUN_WORDS int32 words:
  [run id, round id, first slot delivered (k_lo), last slot delivered (k_hi), t_start (float32 bits),
   t_end (float32 bits), gated, status slot (-1: the run id mod the runtime's status slots)]

  - a direct run (PopulationRuntime.run): one entry in the runtime's own pinned host schedule, its slot the
    run id mod the runtime's status slots (the kernel reads it from host memory; a slot is reused only
    once its run's status has been judged)
  - a planned window (Schedule, from Network.plan): every population's runs of the window, numbered and
    their ring positions advanced on the host, in device memory, uploaded once
  - a captured window (naer2/graph.py): the same entries copied into a graph's own buffers before replay

A run too large for its ring inside a planned or captured window stops (and every run after it), and is rerun
once the window's statuses are judged, the ring grown (naer2/sim/recovery.py: Recovery.recover).
"""

import math

import numpy as np
import torch

RUN_WORDS = 8  # (fused.py: RUN_WORDS, _run_entry)


def write_entry(row, run_id: int, round_id: int, k_lo: int, k_hi: int, t_start: float, t_end: float,
                gated: int = 0) -> None:
    """One entry into an int32 row (a numpy view of RUN_WORDS words)."""
    row[0], row[1], row[2], row[3] = run_id, round_id, k_lo, k_hi
    row[4] = np.float32(t_start).view(np.int32)
    row[5] = np.float32(t_end).view(np.int32)
    row[6], row[7] = gated, -1  # (status slot -1: the run id's, fused.py _status_slot)


class Schedule:
    """A planned window (Network.plan's launches) as run schedules, one per population: its runs numbered on
    from its runtime's run id, each run's ring slots from its ring's position advanced run by run (as
    PopulationRuntime.run does it), its round from the network's round id -- in device memory (one upload).
    launches[i]: launch i's runs as (population, entry index), in the plan's order. commit() moves the
    runtimes on past the window (run ids, ring positions), as if its runs had been made one by one."""

    def __init__(self, network, launches: list, device, upload: bool = True):
        self.launches = []
        rows, run_ids, ring_pos = {}, {}, {}
        for i, launch in enumerate(launches):
            round_id = network.round_id + i + 1
            runs = []
            for pop, t_start, t_end, gated in launch:
                rt = pop._runtime
                if rt is None:
                    raise RuntimeError("Schedule: every population needs its runtime (run a round first)")
                key = id(rt)
                run_id = run_ids.get(key, rt.run_id) + 1
                k_lo = ring_pos.get(key, rt.ring.ring_pos)
                k_hi = rt.delivers_until(t_end)
                run_ids[key], ring_pos[key] = run_id, max(k_lo, k_hi + 1)
                row = np.zeros(RUN_WORDS, dtype=np.int32)
                write_entry(row, run_id, round_id, k_lo, k_hi, t_start, t_end, int(gated))
                rows.setdefault(key, (rt, []))[1].append(row)
                runs.append((pop, len(rows[key][1]) - 1, run_id))
            self.launches.append(runs)
        self.rows = {key: np.stack(r) for key, (rt, r) in rows.items()}  # (host: [runs, RUN_WORDS] int32)
        self.buf = {key: torch.from_numpy(r).to(device) for key, r in self.rows.items()} if upload else {}
        self._ends = {key: (rows[key][0], run_ids[key], ring_pos[key]) for key in rows}
        self.runtimes = {key: rt for key, (rt, _) in rows.items()}  # (runtime id -> runtime)
        self.n_runs = {key: len(r) for key, (rt, r) in rows.items()}

    def signature(self) -> tuple:
        """The window's shape: per launch, which runtimes run (in order), their entries' indices, gated --
        what a captured graph of it depends on (the entries' values don't: the kernels read them)."""
        return tuple(tuple((id(pop._runtime), i, int(self.rows[id(pop._runtime)][i, 6])) for pop, i, _ in runs)
                     for runs in self.launches)

    def entries(self, rt) -> torch.Tensor:
        return self.buf[id(rt)]

    def commit(self) -> None:
        for rt, run_id, pos in self._ends.values():
            rt.run_id, rt.ring.ring_pos = run_id, pos


def window_slots(schedule: Schedule) -> dict:
    """Per runtime, the status slots a window needs: one per run (judged once the window is done)."""
    return {key: 1 << math.ceil(math.log2(max(n, 1))) for key, n in schedule.n_runs.items()}
