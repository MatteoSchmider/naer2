"""
A population's emission history: what its fused kernel (naer2/greenfield/
fused.py) writes on every run and what its consumers pull from.

K run slots; run r writes slot r % K. Per slot a header and the run's spikes
as one flat list:
  header [K, 8]   run id (0: never) and the round the run ran in -- one
                  64-bit word, written and read whole, so a consumer never
                  sees a new run id with an old round --, spike count,
                  window end (float32), earliest e (float32), 3 words pad
  node   [K, cap] the run's spikes' population-local neurons
  time   [K, cap] their emission times e (float32)
  overflow [1]    a run emitted more than cap spikes
A program with n spikes writes them at [base, base + n), base = what its
count atomic returned: no gaps, order across programs arbitrary (consumers
order what they need themselves).

Visibility: a run's spikes become visible from the next round on -- a
consumer takes a slot only if it is tagged with an earlier round than its
own, so a source's run in flight in the same round is never read. Run r
clears slot (r + 1) % K for the next run (run id 0, count 0; clearing at a
run's own start would race with that run's atomics), so a slot's run stays
readable for K - 2 further runs: K = the most runs a source makes between
two reads of its slowest consumer + 2 (3 when every population runs every
round). A consumer that falls further behind sees a gap in the run ids it
finds and raises its overrun flag.
"""

import math

import torch
import triton
import triton.language as tl

HDR_WORDS = 8


@triton.jit
def history_ptrs(buf, slot, K, cap):
    """Typed pointers into a history buffer for one slot: its tag (int64:
    round << 32 | run id), count, window end, earliest e, node list, time
    list; and the overflow flag."""
    hdr = buf + 8 * slot
    node = buf + 8 * K + slot * cap
    time = (buf + 8 * K + K * cap + slot * cap).to(tl.pointer_type(tl.float32))
    overflow = buf + 8 * K + 2 * K * cap
    return (hdr.to(tl.pointer_type(tl.int64)), hdr + 2, (hdr + 3).to(tl.pointer_type(tl.float32)),
            (hdr + 4).to(tl.pointer_type(tl.float32)), node, time, overflow)


@triton.jit
def history_tag(run_id, round_id):
    return (tl.cast(round_id, tl.int64) << 32) | (tl.cast(run_id, tl.int64) & 0xFFFFFFFF)


class EmissionHistory:
    """See the module docstring. cap: spikes per run (default: one per neuron)."""

    def __init__(self, n_neurons: int, depth: int = 3, device="cuda", cap: int | None = None):
        if depth < 3:
            raise ValueError("an emission history needs at least 3 run slots (see naer2/greenfield/history.py)")
        self.K, self.cap = depth, cap or n_neurons
        K, c = self.K, self.cap
        self.buf = torch.zeros(HDR_WORDS * K + 2 * K * c + 1, device=device, dtype=torch.int32)
        self.header = self.buf[:HDR_WORDS * K].view(K, HDR_WORDS)
        self.header[:, 4] = torch.tensor(math.inf, dtype=torch.float32).view(torch.int32)
        self.header[:, 1] = -1  # round -1: never
        self.node = self.buf[HDR_WORDS * K:HDR_WORDS * K + K * c].view(K, c)
        self.time = self.buf[HDR_WORDS * K + K * c:HDR_WORDS * K + 2 * K * c].view(torch.float32).view(K, c)
        self.overflow = self.buf[HDR_WORDS * K + 2 * K * c:]

    def _slot(self, run_id: int) -> int:
        slot = run_id % self.K
        if int(self.header[slot, 0]) != run_id:
            raise KeyError(f"run {run_id} is not in the history any more (or not yet)")
        return slot

    def summary(self, run_id: int) -> tuple[int, float]:
        """(spike count, earliest emission time) of run `run_id`; one readback."""
        h = self.header[self._slot(run_id)]
        n, earliest = int(h[2]), float(h[4:5].view(torch.float32)[0])
        return n, earliest

    def spikes(self, run_id: int) -> list[tuple[float, int]]:
        """Run `run_id`'s spikes, [(e, node)] in flat-list order."""
        slot = self._slot(run_id)
        n = min(int(self.header[slot, 2]), self.cap)
        return list(zip(self.time[slot, :n].tolist(), self.node[slot, :n].tolist()))

    def window_end(self, run_id: int) -> float:
        return float(self.header[self._slot(run_id), 3:4].view(torch.float32)[0])

    def round_of(self, run_id: int) -> int:
        return int(self.header[self._slot(run_id), 1])
