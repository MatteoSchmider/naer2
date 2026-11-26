"""What capturing a window costs, step by step, and whether it overlaps a running replay: PD14 warmed up,
one window captured and replayed (graph A); then a further window's schedule (planned, not committed: it is
never replayed) captured into graph B -- alone, and again while A replays.

    python tools/pd14_capture_cost.py [--window=100] [--warmup=1000] [--G=256] [--serial] [--barrier] [--rdna2]

Capture steps (host time): record (the launches recorded, no work done), end (capture ended: the graph
built), instantiate (the executable graph). No torch.cuda.synchronize around the capture (torch.cuda.graph
does one; this uses capture_begin/capture_end directly).
"""

import os
import sys
import time

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch

import pd14_network as P
from naer2.schedule import Schedule

opts = {a.split("=")[0][2:]: int(a.split("=")[1]) for a in sys.argv if a.startswith("--") and "=" in a}
WINDOW, WARMUP, G = opts.get("window", 100), opts.get("warmup", 1000), opts.get("G", 256)

BARRIER = "--barrier" in sys.argv  # B with a barrier kernel between rounds (as WindowGraph(barrier=True))
SERIAL = "--serial" in sys.argv  # B recorded on one stream (no fork/join events: a chain of kernel nodes)


def _serial(work):
    s = torch.cuda.current_stream()
    for _, fn in work:
        fn(s)


obj = P._network(G)
net = obj[0]
for r in range(WARMUP):
    P._advance("network", obj, r)
    if r % 500 == 0:
        torch.cuda.synchronize()
for _ in range(3):  # graph A captured, replayed twice
    net.run_graph(max_launches=WINDOW)
torch.cuda.synchronize()
graph_a = next(iter(net._graphs.values()))
n_kernels = sum(len(runs) for runs in net.plan(WINDOW, commit=False))


def capture(stream):
    """Graph B: the next window's schedule (uncommitted) recorded on stream, step by step."""
    tp = time.perf_counter()
    launches = net.plan(WINDOW, commit=False)
    sched = Schedule(net, launches, net.device, upload=False)
    global t_plan
    t_plan = time.perf_counter() - tp
    global t_more
    ta = time.perf_counter()
    pinned = {key: torch.from_numpy(rows).pin_memory() for key, rows in sched.rows.items()}
    tb = time.perf_counter()
    # (the capture holds only the buffers' addresses: their entries are written before each replay)
    bufs = {key: torch.empty(rows.shape, dtype=torch.int32, device=net.device) for key, rows in sched.rows.items()}
    tc = time.perf_counter()
    stream.synchronize()
    td = time.perf_counter()
    try:
        g = torch.cuda.CUDAGraph(keep_graph=True)
    except TypeError:
        g = None
    t0 = time.perf_counter()
    t_more = dict(pin=tb - ta, alloc_upload=tc - tb, side_sync=td - tc, graph_new=t0 - td)
    with torch.cuda.stream(stream):
        if g is None:
            g = torch.cuda.CUDAGraph()
        g.capture_begin(capture_error_mode="thread_local")
        for runs in sched.launches:
            (_serial if SERIAL else net._concurrently)(
                (pop, lambda s, pop=pop, i=i, run_id=run_id: pop._runtime.launch_entry(
                    bufs[id(pop._runtime)], i, run_id, s, bookkeeping=False))
                for pop, i, run_id in runs)
            if BARRIER:
                bar.add_(1)
        t1 = time.perf_counter()
        g.capture_end()
    t2 = time.perf_counter()
    try:
        g.instantiate()
    except RuntimeError:
        pass  # (no keep_graph: capture_end instantiated already)
    t3 = time.perf_counter()
    return g, bufs, (t1 - t0, t2 - t1, t3 - t2)


bar = torch.zeros(1, dtype=torch.int32, device=net.device)
side = torch.cuda.Stream()
for _ in range(2):  # (a first capture warms whatever is cached)
    capture(side)
kept = [capture(side) for _ in range(3)]
steps = [k[2] for k in kept]
rec, end, inst = (min(s[i] for s in steps) * 1e3 for i in range(3))
print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}: a {WINDOW}-round window, {n_kernels} kernels")
print(f"  capture alone (host, ms): record {rec:.1f} ({rec * 1e3 / n_kernels:.0f} us/kernel), end {end:.1f}, "
      f"instantiate {inst:.1f}, total {rec + end + inst:.1f}")


def window():
    """A real window (run_graph: planned, its entries written, graph A replayed, committed -- for a clocked
    network it returns without waiting), its GPU time measured."""
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    net.run_graph(max_launches=WINDOW)
    e.record()
    return s, e


window()
torch.cuda.synchronize()
t = time.perf_counter()
s, e = window()
t_enq = time.perf_counter()
torch.cuda.synchronize()
alone = time.perf_counter() - t
print(f"  a window alone: GPU {s.elapsed_time(e):.1f} ms, host until enqueued {(t_enq - t) * 1e3:.1f} ms, "
      f"wall {alone * 1e3:.1f} ms")
t = time.perf_counter()
s, e = window()
t_enq = time.perf_counter()
graph_b, bufs_b, (r, en, i) = capture(side)  # (kept: destroying a graph may wait for the device)
t_cap = time.perf_counter()
t_done = None
if not e.query():
    t_done = "the window still running when B was ready"
torch.cuda.synchronize()
both = time.perf_counter() - t
print(f"  a window, then B captured while it runs: capture {(t_cap - t_enq) * 1e3:.1f} ms (record {r * 1e3:.1f}, "
      f"end {en * 1e3:.1f}, instantiate {i * 1e3:.1f}), window GPU {s.elapsed_time(e):.1f} ms, wall {both * 1e3:.1f} ms "
      f"(serial: {(alone + (rec + end + inst) / 1e3) * 1e3:.1f}); plan + schedule {t_plan * 1e3:.1f} ms"
      + (f"; {t_done}" if t_done else ""))
print("  before the capture (ms): " + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in t_more.items()))
print(f"  graphs captured by run_graph meanwhile: {net._captures}")
