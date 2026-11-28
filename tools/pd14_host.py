"""The host's share of PD14's captured windows (run_graph): where a window's host time goes, part by part --
planning (Network.plan), the schedule's entries (naer2/schedule.py), making room for the window's statuses
(judging the older window's: waits for it), the replay (entries copied, graph launched), the runs'
bookkeeping (Ledger.launched_window), the commit -- and cProfile's top functions over the same windows.

    python tools/pd14_host.py [--window=100] [--windows=20] [--warmup=1000] [--profile] [--rdna2]

Host time per round, against the GPU's (the replay's events): the wall clock is the larger of the two.
"""

import cProfile
import os
import pstats
import sys
import time

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch

import naer2.graph as GR
import naer2.network as N
import naer2.sim.ledger as LG
import naer2.schedule as S
import pd14_network as P

opts = {a.split("=")[0][2:]: int(a.split("=")[1]) for a in sys.argv if a.startswith("--") and "=" in a}
WINDOW, WINDOWS, WARMUP = opts.get("window", 100), opts.get("windows", 20), opts.get("warmup", 1000)

obj = P._network(256)
net = obj[0]
net.graph_barrier = True
for r in range(WARMUP):
    P._advance("network", obj, r)
    if r % 500 == 0:
        torch.cuda.synchronize()
for _ in range(3):
    net.run_graph(max_launches=WINDOW)
torch.cuda.synchronize()

parts = {}


def timed(owner, name, label):
    f = getattr(owner, name)

    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            parts[label] = parts.get(label, 0.0) + time.perf_counter() - t

    setattr(owner, name, g)


timed(N.Network, "plan", "plan")
timed(S.Schedule, "__init__", "schedule")
timed(S.Schedule, "commit", "commit")
timed(LG.Ledger, "make_room", "make room (judging)")
timed(LG.Ledger, "launched_window", "launched_window (bookkeeping)")
timed(GR.WindowGraph, "replay", "replay (incl. launched_window)")
gpu = []
real_replay = torch.cuda.CUDAGraph.replay


def replay(self):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    real_replay(self)
    e.record()
    gpu.append((s, e))


torch.cuda.CUDAGraph.replay = replay
prof = cProfile.Profile() if "--profile" in sys.argv else None
if prof:
    prof.enable()
t0 = time.perf_counter()
n = 0
for _ in range(WINDOWS):
    t = time.perf_counter()
    n += net.run_graph(max_launches=WINDOW)
    parts["run_graph (all)"] = parts.get("run_graph (all)", 0.0) + time.perf_counter() - t
torch.cuda.synchronize()
wall = time.perf_counter() - t0
if prof:
    prof.disable()
gpu_us = sum(s.elapsed_time(e) for s, e in gpu) * 1e3 / n
print(f"{torch.cuda.get_device_name(0)}: {WINDOWS} windows of {WINDOW} rounds, fused={net.fuse_rounds}: wall "
      f"{wall / n * 1e6:.1f} us/round, the graphs on the GPU {gpu_us:.1f} us/round")
print("  host time per round, by part (us; nested parts are inside their parents):")
for label, v in sorted(parts.items(), key=lambda kv: -kv[1]):
    print(f"    {label:28s} {v / n * 1e6:7.1f}")
if prof:
    pstats.Stats(prof).sort_stats("tottime").print_stats(18)
