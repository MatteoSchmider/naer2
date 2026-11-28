"""PD14's round tail: what makes a round's slowest programs slow -- per program its duration (traced wall
clock) against its group's work that round (links delivered to its neurons: count_received), over one-round
graphs (all of a round's programs launched at once, as a captured window runs them).

    python tools/pd14_tail.py [--G=256] [--Gpop=I_4:128,...] [--rounds=40] [--warmup=1000] [--rdna2]

Per population: programs, median links a program, median / p90 / max duration; the fit duration ~ a + b x
links over all programs (a: the fixed cost of a program, b: per link); which populations the rounds' last
programs belong to.
"""

import collections
import os
import statistics
import sys

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch

import pd14_network as P

opts = {a.split("=")[0][2:]: int(a.split("=")[1]) for a in sys.argv
        if a.startswith("--") and "=" in a and not a.startswith("--Gpop=")}
G, ROUNDS, WARMUP = opts.get("G", 256), opts.get("rounds", 40), opts.get("warmup", 1000)

obj = P._network(G)
net = obj[0]
net.executor.barrier = True
pops = P._runtimes("network", obj)
P.group_sizes(pops, next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--Gpop=")), None))
for p in pops.values():
    p.count_received = True  # (the links each neuron received: its runtime counts them)
for r in range(WARMUP):
    P._advance("network", obj, r)
    if r % 500 == 0:
        torch.cuda.synchronize()
for p in pops.values():
    P._runtime("network", p).set_trace(cap=1024, level=1)
for _ in range(3):  # (the traced variants compiled outside a capture)
    P._advance("network", obj, 0)
for _ in range(3):
    net.run_graph(max_launches=1)
torch.cuda.synchronize()

rows = collections.defaultdict(list)  # pop -> [(links, duration us)]
last = collections.Counter()  # pop of each round's last-ending program
ends = []
prev = {n: P._runtime("network", p).n_received.clone() for n, p in pops.items()}
for _ in range(ROUNDS):
    net.run_graph(max_launches=1)
    torch.cuda.synchronize()
    t0, latest = None, (0, None)
    for name, p in pops.items():
        rt = P._runtime("network", p)
        cur = rt.n_received.clone()
        per_group = torch.nn.functional.pad(cur - prev[name], (0, rt.n_groups * rt.group_size - rt.n))
        links = per_group.view(rt.n_groups, rt.group_size).sum(1).tolist()
        prev[name] = cur
        for gi, prog in enumerate(rt.traced()):
            if not prog["start"]:
                continue
            rows[name].append((links[gi], (prog["end"] - prog["start"]) / 100))
            t0 = prog["start"] if t0 is None else min(t0, prog["start"])
            if prog["end"] > latest[0]:
                latest = (prog["end"], name)
    last[latest[1]] += 1
    ends.append((latest[0] - t0) / 100)

print(f"{torch.cuda.get_device_name(0)} G={G}, {ROUNDS} one-round graphs: a round's last end after its first start "
      f"median {statistics.median(ends):.1f} us, max {max(ends):.1f}; the last program's population: "
      + ", ".join(f"{n} {c}" for n, c in last.most_common()))
print(f"  {'pop':5s} {'programs':>8s} {'links/prog':>10s} {'dur median':>10s} {'p90':>6s} {'max':>6s}")
for name in sorted(rows, key=lambda n: -statistics.median(d for _, d in rows[n])):
    xs = rows[name]
    d = sorted(x for _, x in xs)
    print(f"  {name:5s} {len(xs) // ROUNDS:8d} {statistics.median(l for l, _ in xs):10.0f} "
          f"{statistics.median(d):10.1f} {d[int(.9 * len(d))]:6.1f} {d[-1]:6.1f}")
xs = [x for v in rows.values() for x in v]
n = len(xs)
mx, my = sum(l for l, _ in xs) / n, sum(d for _, d in xs) / n
b = sum((l - mx) * (d - my) for l, d in xs) / max(sum((l - mx) ** 2 for l, _ in xs), 1e-9)
a = my - b * mx
r2 = 1 - sum((d - a - b * l) ** 2 for l, d in xs) / max(sum((d - my) ** 2 for _, d in xs), 1e-9)
print(f"  fit over all programs: duration ~ {a:.1f} us + {b * 1e3:.2f} ns x links (R^2 {r2:.2f})")
