"""PD14's tail rounds: why some rounds take longer -- per round (fused one-round graphs, level-2 traced) its
critical path (the first program's start to the last one's end), and what its last program and the whole
round did: spikes taken in, links appended, slots and stream blocks delivered. The slowest rounds against the
median ones; which of those grow with the critical path.

    python tools/pd14_tail_rounds.py [--rounds=200] [--warmup=1000] [--unfused] [--rdna2]
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

opts = {a.split("=")[0][2:]: int(a.split("=")[1]) for a in sys.argv if a.startswith("--") and "=" in a}
ROUNDS, WARMUP = opts.get("rounds", 200), opts.get("warmup", 1000)

obj = P._network(256)
net = obj[0]
net.graph_barrier = True
pops = P._runtimes("network", obj)
for r in range(WARMUP):
    P._advance("network", obj, r)
    if r % 500 == 0:
        torch.cuda.synchronize()
for p in pops.values():
    P._runtime("network", p).set_trace(cap=2048, level=2)
for _ in range(3):  # (the traced variants compiled outside a capture)
    P._advance("network", obj, 0)
if "--unfused" in sys.argv:  # (else as PD14's definition sets it: net.py)
    net.fuse_rounds = False
for _ in range(3):
    net.run_graph(max_launches=1)
torch.cuda.synchronize()


def program(prog):
    """A traced program's counts: spikes taken in, links appended, slots, stream blocks, links delivered."""
    c = collections.Counter()
    for tag, aux, _ in prog["events"]:
        if tag == "spikes":
            c["spikes"] += aux
        elif tag == "appended":
            c["appended"] += aux
        elif tag == "slot":
            c["slots"] += 1
            c["delivered"] += aux
        elif tag == "streamed":
            c["blocks"] += 1
    return c


rounds = []
for _ in range(ROUNDS):
    net.run_graph(max_launches=1)
    torch.cuda.synchronize()
    t0, last, total = None, None, collections.Counter()
    durs = {}
    for name, p in pops.items():
        for gi, prog in enumerate(P._runtime("network", p).traced()):
            if not prog["start"] or not prog["events"]:
                continue
            c = program(prog)
            total.update(c)
            t0 = prog["start"] if t0 is None else min(t0, prog["start"])
            if last is None or prog["end"] > last[0]:
                last = (prog["end"], name, gi, c, (prog["end"] - prog["start"]) / 100)
            durs.setdefault(name, []).append((prog["end"] - prog["start"]) / 100)
    crit = (last[0] - t0) / 100
    rounds.append(dict(crit=crit, pop=last[1], last=last[3], last_dur=last[4], total=total,
                       med=statistics.median(d for v in durs.values() for d in v)))

rounds.sort(key=lambda r: r["crit"])
n = len(rounds)
print(f"{torch.cuda.get_device_name(0)}, {n} {'fused' if net.fuse_rounds else 'unfused'} one-round graphs: a round's "
      f"critical path median {rounds[n // 2]['crit']:.1f} us, p90 {rounds[int(.9 * n)]['crit']:.1f}, max "
      f"{rounds[-1]['crit']:.1f}")


def describe(group, label):
    g = lambda f: statistics.mean(f(r) for r in group)
    pops_ = collections.Counter(r["pop"] for r in group).most_common(4)
    print(f"  {label:18s} crit {g(lambda r: r['crit']):5.1f} | last program {g(lambda r: r['last_dur']):5.1f} us "
          f"(the round's median program {g(lambda r: r['med']):4.1f}): spikes {g(lambda r: r['last']['spikes']):5.1f}, "
          f"appended {g(lambda r: r['last']['appended']):5.0f}, slots {g(lambda r: r['last']['slots']):3.1f}, "
          f"delivered {g(lambda r: r['last']['delivered']):5.0f}, blocks {g(lambda r: r['last']['blocks']):3.1f} | "
          f"the round: spikes {g(lambda r: r['total']['spikes']):6.0f}, appended {g(lambda r: r['total']['appended']):6.0f}"
          f" | last: " + " ".join(f"{p} {c}" for p, c in pops_))


k = max(n // 10, 1)
describe(rounds[:k], "fastest 10%")
describe(rounds[n // 2 - k // 2:n // 2 + k // 2 + 1], "median 10%")
describe(rounds[-k:], "slowest 10%")
crit = [r["crit"] for r in rounds]


def corr(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** .5
    sy = sum((y - my) ** 2 for y in ys) ** .5
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sx * sy, 1e-9)


print("  correlation with the critical path: " + ", ".join(
    f"{label} {corr(crit, [f(r) for r in rounds]):+.2f}" for label, f in (
        ("round spikes", lambda r: r["total"]["spikes"]), ("round appended", lambda r: r["total"]["appended"]),
        ("last's appended", lambda r: r["last"]["appended"]), ("last's delivered", lambda r: r["last"]["delivered"]),
        ("last's slots", lambda r: r["last"]["slots"]), ("median program", lambda r: r["med"]))))
