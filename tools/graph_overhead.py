"""What a round's shape costs in a captured graph, apart from the kernels' own work: PD14's round (9 kernels
of its populations' program counts, the barrier kernel after them, as naer2/graph.py captures it) against
one kernel of all their programs a round -- each program busy for a fixed time (spinning on the clock), so
the difference is the graph's: nodes, edges, waves, launch latency.

    python tools/graph_overhead.py [--us=15] [--rounds=100] [--warps=4] [--rdna2]

Per shape: the replay's GPU time per round (events around the replay), the fixed busy time subtracted.
"""

import os
import sys

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch
import triton
import triton.language as tl

from naer2.trace import realtime

opts = {a.split("=")[0][2:]: float(a.split("=")[1]) for a in sys.argv if a.startswith("--") and "=" in a}
US, ROUNDS, WARPS = opts.get("us", 15.0), int(opts.get("rounds", 100)), int(opts.get("warps", 4))
GRIDS = (86, 81, 57, 23, 22, 19, 12, 5, 4)  # PD14 at G=256: its populations' programs


@triton.jit
def busy(out, ticks):
    """Every program until `ticks` of the wall clock (10 ns) have passed since it started."""
    t0 = realtime()
    t = t0
    while t - t0 < ticks:
        t = realtime()
    tl.store(out + tl.program_id(0), (t - t0).to(tl.int32))


def capture(shape, out, bar, ticks):
    side = [torch.cuda.Stream() for _ in GRIDS]
    ready, done = torch.cuda.Event(), [torch.cuda.Event() for _ in GRIDS]
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g.capture_begin(capture_error_mode="thread_local")
        bar.add_(1)
        for _ in range(ROUNDS):
            if shape == "fused":
                busy[(sum(GRIDS),)](out, ticks, num_warps=WARPS)
            else:
                ready.record(s)
                for st, e, n in zip(side, done, GRIDS):
                    st.wait_event(ready)
                    with torch.cuda.stream(st):
                        busy[(n,)](out, ticks, num_warps=WARPS)
                    e.record(st)
                for e in done:
                    s.wait_event(e)
                bar.add_(1)
        g.capture_end()
    return g


out = torch.zeros(sum(GRIDS), dtype=torch.int32, device="cuda")
bar = torch.zeros(1, dtype=torch.int32, device="cuda")
ticks = int(US * 100)
busy[(1,)](out, ticks, num_warps=WARPS)  # (compiled before the captures)
torch.cuda.synchronize()
print(f"{torch.cuda.get_device_name(0)}: {ROUNDS} rounds a graph, every program busy {US} us, {WARPS} warps")
for shape in ("9 kernels + barrier", "fused"):
    g = capture(shape, out, bar, ticks)
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / ROUNDS)
    times.sort()
    print(f"  {shape:20s}: {times[len(times) // 2]:6.1f} us/round (min {times[0]:.1f}), "
          f"{times[len(times) // 2] - US:5.1f} beyond the busy time")
