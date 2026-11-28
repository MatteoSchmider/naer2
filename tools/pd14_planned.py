"""PD14 through Network.run_planned (planned windows: the runs' entries in a Schedule, no per-round host
scheduling) or Network.run_graph (each window replayed from a captured graph) against Network.step: the same
rounds either way, then every population's state compared.

    python tools/pd14_planned.py {step,planned,graph} OUT.pt [--rounds=1000] [--window=1000] [--G=256] [--warmup=5] [--barrier] [--rdna2]
    python tools/pd14_planned.py compare A.pt B.pt
(each mode in its own process; on a cloud instance through cloud/remote run)
"""

import os
import sys
import time

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch

import pd14_network as P

opts = {a.split("=")[0][2:]: int(a.split("=")[1]) for a in sys.argv if a.startswith("--") and "=" in a}
ROUNDS, WINDOW, G = opts.get("rounds", 1000), opts.get("window", 1000), opts.get("G", 256)
WARMUP = opts.get("warmup", 5)  # stepped rounds first (every runtime built; past the arenas' growth)

if sys.argv[1] == "compare":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    print(f"{a['mode']} ({a['rounds']} rounds, {a['us']:.0f} us/round) vs {b['mode']} ({b['rounds']} rounds, "
          f"{b['us']:.0f} us/round)")
    if "received" in a and "received" in b:
        print("  links received:", ", ".join(f"{n} {a['received'][n]}/{b['received'][n]}" for n in a["received"]))
    for name in a["state"]:
        for attr, x in a["state"][name].items():
            y = b["state"][name][attr]
            same = torch.equal(x, y)
            diff = "" if same else f", {int((x != y).sum())} of {x.numel()} differ, max {float((x - y).abs().max()):.3g}"
            print(f"  {name:5s} {attr:12s} {'identical' if same else 'DIFFERENT'}{diff}")
    sys.exit(0)

mode, out = sys.argv[1], sys.argv[2]
obj = P._network(G)
net = obj[0]
net.executor.barrier = "--barrier" in sys.argv  # (graph mode: a barrier kernel between rounds, naer2/graph.py)
pops = P._runtimes("network", obj)
for p in pops.values():
    p.count_received = True  # (the links each population received: compared as activity)
for r in range(WARMUP):  # every runtime built (run_planned needs them)
    P._advance("network", obj, r)
torch.cuda.synchronize()
t0 = time.perf_counter()
if mode == "step":
    for r in range(WARMUP, WARMUP + ROUNDS):
        P._advance("network", obj, r)
else:
    run = net.run_planned if mode == "planned" else net.run_graph
    done = 0
    while done < ROUNDS:
        t1 = time.perf_counter()
        n = run(max_launches=min(WINDOW, ROUNDS - done))
        if done == 0:
            torch.cuda.synchronize()
            first = time.perf_counter() - t1
            print(f"  first window ({n} rounds, {mode}: captures included): {first * 1e3:.0f} ms", flush=True)
            t_rest = time.perf_counter()
        done += n
    torch.cuda.synchronize()
    if ROUNDS > WINDOW:
        print(f"  the other windows: {(time.perf_counter() - t_rest) / (ROUNDS - WINDOW) * 1e6:.0f} us/round"
              + (f", {len(net.executor.graphs)} window shapes, captures: {net.executor.captures}" if mode == "graph" else ""),
              flush=True)
torch.cuda.synchronize()
us = (time.perf_counter() - t0) / ROUNDS * 1e6
for p in pops.values():
    P._runtime("network", p).check()
state = {n: {a: t.detach().cpu().clone() for a, t in P._runtime("network", p).state.items()} for n, p in pops.items()}
received = {n: int(P._runtime("network", p).n_received.sum()) for n, p in pops.items()}
torch.save(dict(mode=mode, rounds=ROUNDS, us=us, state=state, received=received), out)
print(f"{torch.cuda.get_device_name(0)} {mode}: {ROUNDS} rounds, {us:.0f} us/round (G={G}) -> {out}")
