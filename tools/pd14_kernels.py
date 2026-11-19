"""PD14's kernels one by one in quiet activity: per population, the fused kernel's time (median / slowest) from
the profiler, and the round's wall clock free-running -- the device-side floor against the host's.

    python tools/pd14_kernels.py G [--warps=N] [--rounds=300]
    (on a cloud instance: cloud/remote run 'python tools/pd14_kernels.py 256')

Kernels map to populations by launch order: Network.step launches every population once per round, in a fixed
order (list(network.populations)); a GPU that doesn't overlap them (the RTX 4090, host-bound) keeps that order.
"""

import os
import statistics
import sys
import time

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
import torch
from torch.profiler import ProfilerActivity, profile

import pd14_network as P
from naer2 import runtime as R

G = int(sys.argv[1])
ROUNDS = 300
for a in sys.argv[2:]:
    if a.startswith("--warps="):
        R._warp_cap = lambda device, _w=int(a.split("=")[1]): _w
    if a.startswith("--rounds="):
        ROUNDS = int(a.split("=")[1])

t0 = time.time()
obj = P._network(G)
net = obj[0]
pops = P._runtimes("network", obj)
names = {id(p): n for n, p in pops.items()}
order = [names[id(p)] for p in list(net.populations)]
r = 0
for r in range(1005):  # into quiet activity (100 ms), every kernel compiled, sizes settled
    P._advance("network", obj, r)
    if r % 500 == 0:
        torch.cuda.synchronize()
for p in pops.values():
    P._runtime("network", p).check()
torch.cuda.synchronize()
r = 1005
t1 = time.perf_counter()
for k in range(r, r + 1000):
    P._advance("network", obj, k)
torch.cuda.synchronize()
wall = (time.perf_counter() - t1) / 1000
r += 1000
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for k in range(r, r + ROUNDS):
        P._advance("network", obj, k)
    torch.cuda.synchronize()
ks = sorted((e.time_range.start, e.time_range.end) for e in prof.events()
            if e.device_type.name == "CUDA" and "fused_kernel" in e.name)
per = {n: [] for n in order}
rounds = len(ks) // len(order)
for i in range(rounds * len(order)):
    a, b = ks[i]
    per[order[i % len(order)]].append(b - a)
rt0 = P._runtime("network", list(net.populations)[0])
print(f"{torch.cuda.get_device_name(0)} G={G} ({rt0._warps(None)} warps), quiet from 100 ms: wall {wall * 1e6:.0f} "
      f"us/round free-running; {rounds} profiled rounds [{time.time() - t0:.0f} s]")
print(f"  {'pop':5s} {'median':>7s} {'max':>7s}  {'neurons':>7s} {'programs':>8s}  (us)")
for n in sorted(order, key=lambda n: -statistics.median(per[n])):
    rt = P._runtime("network", pops[n])
    print(f"  {n:5s} {statistics.median(per[n]):7.1f} {max(per[n]):7.1f}  {rt.n:7d} {rt.n_groups:8d}")
med = [statistics.median(v) for v in per.values()]
print(f"  sum of medians {sum(med):.0f} us/round, largest median {max(med):.1f}, slowest kernel "
      f"{max(max(v) for v in per.values()):.1f} us")
