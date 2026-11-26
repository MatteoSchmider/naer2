"""
A population's emission history: what its fused kernel (naer2/transpiler/
templates/fused.py) writes on every run and what its consumers pull from.

K run slots; run r writes slot r % K. Per slot a header, and the run's spikes
in one region per producing program (G programs, R spikes each):
  header [K, 8]   run id (0: never) and the round the run ran in -- one
                  64-bit word, written and read whole, so a consumer never
                  sees a new run id with an old round --, spike count (all
                  programs'), window end (float32), earliest e (float32), 3
                  words pad
  counts [K, G]   per program its spikes (past R: the overflow)
  node   [K, G R] the run's spikes' population-local neurons, program g's at
                  [g R, g R + its count)
  time   [K, G R] their emission times e (float32)
  overflow [1]    a program emitted more than R spikes in a run
Every program writes only its own region and count: the run's spikes in a
fixed order -- the regions in program order, each in the program's own order
-- whatever the programs' timing (SPIKES.md: no atomics placing them, no
program waiting on another). A consumer reads a run's counts and scans them.
That's for producers whose spikes need ordering (non-uniform emission: in
receive, at precise times); a uniform producer (every spike at its window's
start, PD14) keeps one flat list -- one region, G = 1, its count the header's,
the programs' pieces placed by an atomic in any order (regions=False: the
consumer's batched pull at its old cost; SPIKES.md, TODO.md).

Visibility: a run's spikes become visible from the next round on -- a
consumer takes a slot only if it is tagged with an earlier round than its
own, so a source's run in flight in the same round is never read. Run r
clears slot (r + 1) % K's header for the next run (run id 0, count 0;
clearing at a run's own start would race with that run's atomics); the
programs' region counts each program clears itself at its own run's start
(its only writer -- a consumer reads them mid-pull, after the header: fused.py
_region_reset). So a slot's run stays
readable for K - 2 further runs: K = the most runs a source makes between
two reads of its slowest consumer + 2 (3 when every population runs every
round). A consumer that falls further behind sees a gap in the run ids it
finds and raises its overrun flag.
"""

# Depth K -- the most runs a source makes between two reads of its slowest consumer + 2 --
#   comes from the scheduler, which knows the populations' run cadences.
# cap: a neuron can fire any number of times per run (in receive, after every delivery, at several send sites --
#   D-3), so cap is sized per population (Population.history_cap; default one spike per neuron), a run beyond it
#   raising the overflow flag (PopulationRuntime.check raises)
# What an event-driven consumer's scheduling needs of a source run (spike count, earliest e)
#   is the slot header, read on the host -- a sync per run, kept for now (D-13)
import math

import torch
import triton
import triton.language as tl

HDR_WORDS = 8


@triton.jit
def history_ptrs(buf, slot, K, G, R):
    """Typed pointers into a history buffer (G programs, R spikes a region) for one slot: its tag (int64:
    round << 32 | run id), count, window end, earliest e, the programs' counts, the node list, the time list
    (program g's region at g R); and the overflow flag."""
    hdr = buf + 8 * slot
    counts = buf + 8 * K + slot * G
    node = buf + 8 * K + K * G + slot * G * R
    time = (buf + 8 * K + K * G + K * G * R + slot * G * R).to(tl.pointer_type(tl.float32))
    overflow = buf + 8 * K + K * G + 2 * K * G * R
    return (hdr.to(tl.pointer_type(tl.int64)), hdr + 2, (hdr + 3).to(tl.pointer_type(tl.float32)),
            (hdr + 4).to(tl.pointer_type(tl.float32)), counts, node, time, overflow)


@triton.jit
def history_tag(run_id, round_id):
    return (tl.cast(round_id, tl.int64) << 32) | (tl.cast(run_id, tl.int64) & 0xFFFFFFFF)


class EmissionHistory:
    """See the module docstring. group_size: the producer's (G = its programs); cap: spikes per run (default:
    one per neuron), spread over the programs -- R = ceil(cap / G) a region. regions False: one flat list (G =
    1, R = cap: a uniform producer)."""

    def __init__(self, n_neurons: int, depth: int = 3, device="cuda", cap: int | None = None,
                 group_size: int = 256, regions: bool = True):
        if depth < 3:
            raise ValueError("an emission history needs at least 3 run slots (see naer2/history.py)")
        self.K, self.regions = depth, regions
        self.G = (-(-n_neurons // group_size) if n_neurons else 1) if regions else 1
        if not regions:
            self.R = cap or max(n_neurons, 1)
        else:
            self.R = -(-cap // self.G) if cap else min(group_size, max(n_neurons, 1))
            if self.R > 4 * group_size:  # (fused.py _sort_region: a region's order through LDS)
                raise ValueError(f"a history region holds at most 4 x the group size ({4 * group_size}) spikes a "
                                 f"run, not {self.R} (history_cap {cap} over {self.G} programs)")
        self.cap = self.G * self.R
        K, G, c = self.K, self.G, self.cap
        o_cnt, o_node = HDR_WORDS * K, HDR_WORDS * K + K * G
        o_time = o_node + K * c
        self.buf = torch.zeros(o_time + K * c + 1, device=device, dtype=torch.int32)
        self.header = self.buf[:o_cnt].view(K, HDR_WORDS)
        self.header[:, 4] = torch.tensor(math.inf, dtype=torch.float32).view(torch.int32)
        self.header[:, 1] = -1  # round -1: never
        self.counts = self.buf[o_cnt:o_node].view(K, G)
        self.node = self.buf[o_node:o_time].view(K, c)
        self.time = self.buf[o_time:o_time + K * c].view(torch.float32).view(K, c)
        self.overflow = self.buf[o_time + K * c:]

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
        """Run `run_id`'s spikes, [(e, node)] in the history's order (the regions in program order)."""
        slot = self._slot(run_id)
        n = self.counts[slot].clamp(max=self.R).tolist() if self.regions else [min(int(self.header[slot, 2]), self.R)]
        times, nodes = self.time[slot].tolist(), self.node[slot].tolist()
        return [(times[g * self.R + i], nodes[g * self.R + i]) for g in range(self.G) for i in range(n[g])]

    def window_end(self, run_id: int) -> float:
        return float(self.header[self._slot(run_id), 3:4].view(torch.float32)[0])

    def round_of(self, run_id: int) -> int:
        return int(self.header[self._slot(run_id), 1])
