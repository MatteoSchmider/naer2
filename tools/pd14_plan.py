"""Network.plan on PD14: plan the next 1000 launches (rounds) as a dry run, time it, then step the same
network as many rounds and check every population ran exactly as planned (PD14 is clocked throughout).

    python tools/pd14_plan.py            (on a cloud instance: cloud/remote run 'python tools/pd14_plan.py')
"""
import sys, time, os
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
import torch
import pd14_network as P
from naer2.sim.scheduler import Scheduler
obj = P._network(256)
net = obj[0]
names = {id(p): n for n, p in P._runtimes("network", obj).items()}
for r in range(5):  # into the run (every runtime built)
    P._advance("network", obj, r)
torch.cuda.synchronize()
t = time.perf_counter()
launches = net.plan(max_launches=1000)
dt = time.perf_counter() - t
planned = [sorted((names[id(p)], round(a, 6), round(b, 6)) for p, a, b, g in L) for L in launches]
print(f"plan: {len(launches)} launches ({sum(len(L) for L in launches)} kernel runs, {launches[0][0][1]:.1f} "
      f"-> {launches[-1][0][2]:.1f} ms) in {dt * 1e3:.0f} ms host ({dt / len(launches) * 1e6:.0f} us a launch)")
ran = []
orig = Scheduler.take
def rec(states, t):  # (stepped from here on: every round's runs as step() takes them)
    runs = orig(states, t)
    ran.extend((net.round_id, names[id(p)], round(a, 6), round(b, 6)) for p, a, b, _ in runs)
    return runs
Scheduler.take = staticmethod(rec)
for r in range(5, 5 + len(launches)):
    P._advance("network", obj, r)
torch.cuda.synchronize()
by_round = {}
for rid, n, a, b in ran:
    by_round.setdefault(rid, []).append((n, a, b))
actual = [sorted(v) for _, v in sorted(by_round.items())]
print(f"step: {len(actual)} rounds; plan == step: {planned == actual}")
if planned != actual:
    for i, (x, y) in enumerate(zip(planned, actual)):
        if x != y:
            print(" first difference at launch", i, x[:3], y[:3]); break
