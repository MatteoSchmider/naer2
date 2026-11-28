"""
A ring's sizing: its arenas (blocks per group) and overflow logs (records per cell) follow the runs' need --
the most blocks any group held, the most records any cell logged (Little's law at work: the arrival rate x
how long a record stays) -- with headroom and hysteresis: grow when need x grow_at exceeds the size, shrink
when need x grow_at stays below half of it for shrink_window runs, either to need x grow_to (Ring.resize); a
run that doesn't fit stops and reruns (naer2/sim/recovery.py). The policy only: what to resize to is noted
(wanted), and applied once the device is idle (Recovery.resize, recover).
"""

import math

from ..ring import FLAG_STOPPED, MAX_BLOCKS, RESERVE


class RingSizer:
    def __init__(self):
        self.grow_at, self.grow_to, self.shrink_window = 1.5, 2.0, 200
        self.growths: list[tuple[int, str, int]] = []  # (run id, "arena" / "log", new size)
        self.resizes: list[tuple[int, int, int]] = []  # (run id, blocks per arena, log records per cell)
        self.stops: list[int] = []  # run ids that stopped (the ring too small) and were rerun
        self.wanted = None  # (blocks, log records, run id), applied at the next resize
        self.reset()

    def reset(self) -> None:
        """(A new ring) the most blocks / log records any run needed, and the runs judged, since the last
        shrink check."""
        self._window = [0, 0, 0]

    def follow(self, ring, blocks: int, most: int, run_id: int, n_runs: int) -> bool:
        """The ring following what complete runs needed -- n_runs of them, run_id the last (the most blocks any
        group held, the most records any cell logged). Whether a resize is wanted now."""
        w = self._window
        w[0], w[1], w[2] = max(w[0], blocks), max(w[1], most), w[2] + n_runs
        nb, log_cap = self._sizes(ring, blocks, most)
        grew = nb > ring.n_blocks or log_cap > ring.log_cap
        if not grew and w[2] >= self.shrink_window:
            floor_nb, floor_log = self._floors(ring)
            if w[0] * self.grow_at < ring.n_blocks / 2:
                nb = max(math.ceil(w[0] * self.grow_to), floor_nb)
            if w[1] * self.grow_at < ring.log_cap / 2:
                log_cap = max(math.ceil(w[1] * self.grow_to), floor_log)
            self.reset()
        if (nb, log_cap) == (ring.n_blocks, ring.log_cap):
            return False
        self.wanted = (nb, log_cap, run_id)
        return True

    def stopped(self, ring, run_id: int) -> None:
        """(Recovery, the device idle) A ring that stopped a run grows to what that run needs, its flag cleared."""
        if not int(ring.flags[FLAG_STOPPED]):
            return
        self.stops.append(run_id)
        stats = ring.stats.cpu()
        blocks, most = int(stats[:, 0].max()), int(stats[:, 1].max())
        if blocks >= 1 << 24:
            raise RuntimeError(f"a ring run needs more than {MAX_BLOCKS} blocks of {ring.block} records "
                               f"in one slot (Population.ring_block: larger blocks)")
        self.wanted = (*self._sizes(ring, blocks, most), run_id)
        ring.flags[FLAG_STOPPED] = 0

    def apply(self, ring, run_id: int) -> None:
        """(Recovery.resize / recover: the device idle) The wanted sizes (Ring.resize); run_id: the runtime's
        last run."""
        if self.wanted is None:
            return
        (nb, log_cap, at), self.wanted = self.wanted, None
        if nb < ring.n_blocks:  # never below what the groups hold now (at most what their last pull held)
            nb = max(nb, int(ring.stats[:, 0].max()))
        if (nb, log_cap) == (ring.n_blocks, ring.log_cap):
            return
        grew = (nb > ring.n_blocks, log_cap > ring.log_cap)
        ring.resize(n_blocks=nb, log_cap=log_cap)
        self.resizes.append((run_id, nb, log_cap))
        if grew[0]:
            self.growths.append((at, "arena", ring.arena))
        elif grew[1]:
            self.growths.append((at, "log", log_cap))

    @staticmethod
    def _floors(ring) -> tuple[int, int]:
        """The smallest arena (every slot its reserve, a partly used block, one more) and log (one
        block: a run hardly ever logs more in quiet activity, and a burst stops a run at worst)."""
        return ring.n_slots * (RESERVE + 1) + 1, ring.block

    def _sizes(self, ring, blocks: int, most: int) -> tuple[int, int]:
        """The arena and log sizes for a run's need -- grown to need x grow_to where need x grow_at
        exceeds them, else as they are; once one grows, the other too wherever it has less than grow_to x
        its need (one resize, one copy of the arena, for both)."""
        floor_nb, floor_log = self._floors(ring)
        nb, log_cap = ring.n_blocks, ring.log_cap
        at = self.grow_to if blocks * self.grow_at > nb or most * self.grow_at > log_cap else None
        if at and blocks * at > nb:
            nb = max(math.ceil(blocks * self.grow_to), floor_nb)
        if at and most * at > log_cap:
            log_cap = max(math.ceil(most * self.grow_to), floor_log)
        return nb, log_cap
