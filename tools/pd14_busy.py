"""PD14 in quiet activity: a round's wall time free-running, and the GPU's busy time per round from the
in-kernel trace's wall-clock stamps (naer2/trace.py: every program's start and end, 10 ns ticks, comparable
across kernels and streams -- AMD's REALTIME, NVIDIA's %globaltimer), so it works where profilers don't.

    python tools/pd14_busy.py G [--rounds=300] [--rdna2]
    (on a cloud instance: cloud/remote run 'python tools/pd14_busy.py 256')

Busy: per round, each kernel spans its programs' earliest start to latest end; the round's busy time is the
union of its kernels' spans (overlapping kernels count once). From a second free-running pass: every run's
trace header copied on its stream after its launch (no sync -- syncing per round lets an iGPU drop its clock).
"""

import os
import statistics
import sys
import time

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch

import pd14_network as P

G = int(sys.argv[1])
ROUNDS = 300
for a in sys.argv[2:]:
    if a.startswith("--rounds="):
        ROUNDS = int(a.split("=")[1])

obj = P._network(G)
pops = P._runtimes("network", obj)
r = 0
for r in range(1005):
    P._advance("network", obj, r)
    if r % 500 == 0:
        torch.cuda.synchronize()
for p in pops.values():
    P._runtime("network", p).check()
torch.cuda.synchronize()
r = 1005
for k in range(r, r + 100):  # warm
    P._advance("network", obj, k)
torch.cuda.synchronize()
r += 100
t1 = time.perf_counter()
for k in range(r, r + ROUNDS):
    P._advance("network", obj, k)
torch.cuda.synchronize()
wall = (time.perf_counter() - t1) / ROUNDS
r += ROUNDS

from naer2.sim import runtime as R
for p in pops.values():
    P._runtime("network", p).set_trace(cap=8, level=1)  # (only the header's wall clock is used)
for k in range(r, r + 5):
    P._advance("network", obj, k)
torch.cuda.synchronize()
r += 5
# every run traces into its own slice of one buffer made up front (the launch takes the runtime's current
# trace buffer: pointing it at the round's slice costs a Python assignment -- no copy, no sync, no allocation),
# so the pass stays free-running and the GPU at its working clock
SLICES = {}
for p in pops.values():
    rt_ = P._runtime("network", p)
    big = torch.zeros(ROUNDS, rt_.trace.buf.numel(), device=rt_.trace.buf.device, dtype=torch.int32)
    SLICES[id(rt_)] = (big, [big[i] for i in range(ROUNDS)])
_launch = R.PopulationRuntime.launch
def _launch_slice(self, params, stream):
    big, views = SLICES[id(self)]
    self.trace.buf = views[(params.run_id - r0) % ROUNDS]
    _launch(self, params, stream)
R.PopulationRuntime.launch = _launch_slice
r0 = r
t2 = time.perf_counter()
for k in range(r, r + ROUNDS):
    P._advance("network", obj, k)
torch.cuda.synchronize()
wall_traced = (time.perf_counter() - t2) / ROUNDS
R.PopulationRuntime.launch = _launch
SNAPS = []
for p in pops.values():
    rt_ = P._runtime("network", p)
    big, _ = SLICES[id(rt_)]
    hdrs = big.view(ROUNDS, rt_.trace.n_programs, -1)[:, :, :4].cpu()
    for i in range(ROUNDS):
        SNAPS.append((i, hdrs[i]))
by_round = {}
for rnd, hdr in SNAPS:
    rt_ = hdr.cpu().contiguous().view(torch.int64)  # [programs, 2]: start, end
    st, en = rt_[:, 0], rt_[:, 1]
    ok = (st > 0) & (en > 0)
    if ok.any():
        by_round.setdefault(rnd, []).append((int(st[ok].min()), int(en[ok].max())))
busy, summed, kernel_max = [], [], []
for rnd, spans in by_round.items():
    spans.sort()
    u, ca, cb = 0, None, None
    for a, b in spans:
        if cb is None or a > cb:
            if cb is not None:
                u += cb - ca
            ca, cb = a, b
        else:
            cb = max(cb, b)
    u += (cb - ca) if cb is not None else 0
    busy.append(u * 10e-3)  # 10 ns ticks -> us
    summed.append(sum(b - a for a, b in spans) * 10e-3)
    kernel_max.append(max(b - a for a, b in spans) * 10e-3)
dev = torch.cuda.get_device_name(0)
print(f"{dev} G={G}, quiet from 100 ms, {ROUNDS} rounds: wall {wall * 1e6:.0f} us/round free-running "
      f"({wall_traced * 1e6:.0f} traced); GPU busy median {statistics.median(busy):.0f} us/round (p90 {sorted(busy)[int(0.9 * len(busy))]:.0f}), "
      f"kernels summed {statistics.median(summed):.0f}, overlap x{statistics.median(summed) / statistics.median(busy):.2f}, "
      f"longest kernel {statistics.median(kernel_max):.0f} (max {max(kernel_max):.0f})")
