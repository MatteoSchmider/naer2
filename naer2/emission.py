"""
A population's emission output: what its update kernel writes on every run
and what every consumer's intake reads (see naer2/intake.py).

One persistent int32 buffer per population, with a fixed layout that the
update kernel and intake both reach through `emission_ptrs`:
  node    [n_blocks * cap]  population-local neuron of each spike
  time    [n_blocks * cap]  emission time e (float32); each block's valid
                            entries sorted ascending by e (sorted once, at
                            the source, so intake only merges)
  count   [n_blocks]        valid entries per block: block b's are
                            [b * cap, b * cap + count[b]); setting it
                            invalidates the rest
  state   [2]               run id (0 = never ran), window end of the last
                            run (float32) -- intake compares the run id with
                            the one it last read to find new spikes
  summary [2 + 2]           per run parity (run id % 2): spike count,
                            earliest e (float32); the run with parity p
                            resets parity 1 - p for the next run
  overflow [1]              a block emitted more than cap spikes
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def emission_ptrs(buf, n_blocks, cap):
    """Typed pointers into an emission buffer: node, time, count, run id,
    window end, summary counts [2], summary earliest e [2], overflow."""
    time = buf + n_blocks * cap
    count = time + n_blocks * cap
    state = count + n_blocks
    summary = state + 2
    return (
        buf, time.to(tl.pointer_type(tl.float32)), count,
        state, (state + 1).to(tl.pointer_type(tl.float32)),
        summary, (summary + 2).to(tl.pointer_type(tl.float32)), summary + 4,
    )


class EmissionBuffer:
    """See the module docstring. Views: node, time, count, state, overflow;
    cap, n_blocks."""

    def __init__(self, n_blocks: int, cap: int, device):
        self.n_blocks, self.cap = n_blocks, cap
        s = n_blocks * cap
        self.buf = torch.zeros(2 * s + n_blocks + 2 + 4 + 1, device=device, dtype=torch.int32)
        self.node = self.buf[:s]
        self.time = self.buf[s:2 * s].view(torch.float32)
        self.count = self.buf[2 * s:2 * s + n_blocks]
        self.state = self.buf[2 * s + n_blocks:2 * s + n_blocks + 2]
        self._sum_count = self.buf[2 * s + n_blocks + 2:2 * s + n_blocks + 4]
        self._sum_min = self.buf[2 * s + n_blocks + 4:2 * s + n_blocks + 6].view(torch.float32)
        self._sum_min.fill_(math.inf)
        self.overflow = self.buf[2 * s + n_blocks + 6:]

    @property
    def run_id(self) -> int:
        return int(self.state[0])

    @property
    def window_end(self) -> float:
        return float(self.state[1:].view(torch.float32)[0])

    def summary(self, run_id: int) -> tuple[int, float]:
        """(spike count, earliest emission time) of run `run_id`; one readback."""
        p = run_id % 2
        n, earliest = torch.stack([self._sum_count[p].to(torch.float32), self._sum_min[p]]).tolist()
        return int(n), earliest
