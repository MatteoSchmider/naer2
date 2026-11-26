"""One captured PD14 window's replay under Nsight Systems, round by round: per round its kernels' durations
and what lies between them on the GPU -- where a replay's time per round goes (the longest kernel, the hops
between dependent graph nodes, the barrier).

    nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node -o OUT \\
        python tools/pd14_graph_trace.py run [--window=100] [--warmup=1000] [--windows=1] [--barrier] [--fuse | --unfused]
    nsys stats --report cuda_gpu_trace --format csv --output OUT OUT.nsys-rep
    python tools/pd14_graph_trace.py analyze OUT_cuda_gpu_trace.csv [--window=100]
(cloud/remote run: the three in one go, see the commit's notes)

run: warm up stepped, three windows through run_graph (captured, replayed), then one more with the profiler on.
analyze: the trace's kernels in start order, split into rounds (by the barrier kernel, else 9 kernels a round).
"""

import csv
import os
import statistics
import sys

opts = {a.split("=")[0][2:]: int(a.split("=")[1]) for a in sys.argv if a.startswith("--") and "=" in a}
WINDOW, WARMUP = opts.get("window", 100), opts.get("warmup", 1000)

if sys.argv[1] == "run":
    sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
    import torch

    import pd14_network as P

    obj = P._network(256)
    net = obj[0]
    net.graph_barrier = "--barrier" in sys.argv
    for r in range(WARMUP):
        P._advance("network", obj, r)
        if r % 500 == 0:
            torch.cuda.synchronize()
    if "--fuse" in sys.argv or "--unfused" in sys.argv:  # (else as PD14's definition sets it: net.py)
        net.fuse_rounds = "--fuse" in sys.argv
    for _ in range(3):
        net.run_graph(max_launches=WINDOW)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(opts.get("windows", 1)):
        net.run_graph(max_launches=WINDOW)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("traced one window; captures:", net._captures)
    sys.exit(0)

rows = list(csv.DictReader(open(sys.argv[2])))
key = {k.lower(): k for k in rows[0]}
start_k = next(key[k] for k in key if k.startswith("start"))
dur_k = next(key[k] for k in key if k.startswith("duration"))
name_k = next(key[k] for k in key if k == "name")
kernels = []
for r in rows:
    name = r[name_k]
    if "memcpy" in name.lower() or "memset" in name.lower():
        continue
    kernels.append((float(r[start_k]), float(r[dur_k]), name))
kernels.sort()
unit = 1e-3  # (nsys csv: ns)
is_bar = [("elementwise" in n or "vectorized" in n or "add" in n.lower()) and "fused" not in n.lower()
          for _, _, n in kernels]
rounds, cur = [], []
for k, bar in zip(kernels, is_bar):
    if bar:
        rounds.append((cur, k))
        cur = []
    else:
        cur.append(k)
        if not any(is_bar) and len(cur) == 9:
            rounds.append((cur, None))
            cur = []
print(f"{len(kernels)} kernels, {len(rounds)} rounds, names: "
      + ", ".join(sorted({n[:40] for _, _, n in kernels})[:12]))
per = {k: [] for k in ("span", "longest", "spread", "to barrier", "barrier", "barrier to next")}
for i, (ks, bar) in enumerate(rounds):
    if not ks:
        continue
    first = min(s for s, _, _ in ks)
    last_end = max(s + d for s, d, _ in ks)
    nxt = rounds[i + 1][0] if i + 1 < len(rounds) and rounds[i + 1][0] else None
    end = min(s for s, _, _ in nxt) if nxt else None
    if end is not None:
        per["span"].append((end - first) * unit)
    per["longest"].append(max(d for _, d, _ in ks) * unit)
    per["spread"].append((max(s for s, _, _ in ks) - first) * unit)  # (the round's kernels' starts: apart by)
    if bar is not None:
        per["to barrier"].append((bar[0] - last_end) * unit)
        per["barrier"].append(bar[1] * unit)
        if end is not None:
            per["barrier to next"].append((end - bar[0] - bar[1]) * unit)
print("per round, us: " + ", ".join(
    f"{k} median {statistics.median(v):.1f} (max {max(v):.1f})" for k, v in per.items() if v))
ks0 = sorted(rounds[len(rounds) // 2][0])
t0 = ks0[0][0]
print("a middle round's kernels (start +us, duration us): " + ", ".join(
    f"{(s - t0) * unit:.1f}+{d * unit:.1f}" for s, d, _ in ks0))
durs = sorted(d * unit for ks, _ in rounds for _, d, _ in ks)
print(f"all fused kernels: median {statistics.median(durs):.1f} us, p90 {durs[int(len(durs) * .9)]:.1f}, "
      f"max {durs[-1]:.1f}; their sum per round {sum(durs) / len(rounds):.1f} us")
