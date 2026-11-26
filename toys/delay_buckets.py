"""
Toy model of the bucketed to-deliver buffer (TODO.md, item 1c), outside the
library, to reason about with concrete numbers.

Network: three source populations S1-S3 (4 neurons each, timestep 0.1;
every step 2 of each source's neurons fire, stamped at the source's run
time, no emission delay) all feeding one consumer X (6 neurons, timestep
--timestep). Every source neuron has 3 links into X with per-link delays
spread evenly over all buckets (random within each).

Per connection, delays are quantized on the residual d - D_MIN with
resolution RES, anchored at D_MIN: bucket j holds delays in
[D_MIN + j*RES, D_MIN + (j+1)*RES), and delivers at the MEAN delay of its
links. With D_MIN = 0.1, D_MAX = 0.6, RES = 0.1 that is 5 buckets.

Pending buffer: a hash table keyed by arrival slot, slot = floor(a / RES).
Intake expands every new spike into one entry per bucket of its connection,
entry = (emission time, source, source neuron, bucket j), arrival
a = e + mean_delay[j], and drops it straight into its slot -- no
comparisons. A run of X with window [t, t + T) takes the slots inside the
window in slot order; within a slot entries keep insertion order, which is
emission order without emission delays. So delivery is in arrival order (to
slot resolution) with no comparison sort anywhere.

Round order, as in Network.step: X runs (if due) on what is pending, THEN
the sources' spikes of this round are taken in. An entry whose slot lies
before the window being processed was taken in too late for the run that
should have delivered it; it is delivered now and marked LATE.

    python toys/delay_buckets.py --timestep 0.1
    python toys/delay_buckets.py --timestep 0.2 --links
    python toys/delay_buckets.py --timestep 0.2 --brief
"""

import argparse
import math
import random
from collections import defaultdict

RES = 0.1
D_MIN, D_MAX = 0.1, 0.6
SOURCE_T = 0.1
N_SRC, N_X, LINKS_PER_NEURON, FIRING_PER_STEP = 4, 6, 3, 2
EPS = 1e-9


def slot_of(a: float) -> int:
    return math.floor(a / RES + EPS)


def build(seed: int):
    """Per source: rows[neuron] = [(target, delay, bucket)], and the mean
    delay of each bucket over the whole connection."""
    rng = random.Random(seed)
    n_buckets = round((D_MAX - D_MIN) / RES)
    conns = {}
    for name in ("S1", "S2", "S3"):
        rows = {}
        k = 0
        for n in range(N_SRC):
            links = []
            for target in rng.sample(range(N_X), LINKS_PER_NEURON):
                j = k % n_buckets
                k += 1
                d = round(D_MIN + (j + rng.random()) * RES, 3)
                links.append((target, d, j))
            rows[n] = links
        by_bucket = defaultdict(list)
        for links in rows.values():
            for _, d, j in links:
                by_bucket[j].append(d)
        mean = {j: sum(ds) / len(ds) for j, ds in sorted(by_bucket.items())}
        conns[name] = (rows, mean)
    return conns


def run(timestep: float, until: float, show_links: bool, seed: int, brief: bool) -> None:
    conns = build(seed)
    for name, (_, mean) in conns.items():
        print(f"{name} -> X bucket mean delays: " + "  ".join(f"j{j}={m:.3f}" for j, m in mean.items()))

    pending: dict[int, list] = defaultdict(list)  # slot -> [entry]
    n_rounds = round(until / SOURCE_T)
    for r in range(n_rounds + 1):
        t = round(r * SOURCE_T, 9)

        # X runs first, on what is already pending
        if abs(t / timestep - round(t / timestep)) < EPS:
            lo, hi = slot_of(t), slot_of(t + timestep)  # window [t, t + T) = slots lo .. hi-1
            due = sorted(k for k in pending if k < hi)
            entries = [(k, e) for k in due for e in pending.pop(k)]
            new = sum(1 for _, e in entries if e[5] == round(t - SOURCE_T, 9))
            late = sum(1 for k, _ in entries if k < lo)
            print(f"\nX run at t={t:.1f}  window [{t:.1f}, {t + timestep:.1f})  slots {lo}..{hi - 1}: "
                  f"{len(entries)} entries ({new} from the last intake, {len(entries) - new} older"
                  f"{f', {late} LATE' if late else ''})")
            for k, (e_time, src, neuron, j, a, taken) in ([] if brief else entries):
                flag = "  LATE" if k < lo else ""
                print(f"   slot {k}  a={a:.3f}  {src}#{neuron} e={e_time:.1f} j{j}  (taken in at {taken:.1f}){flag}")
                if show_links:
                    for target, d, jj in conns[src][0][neuron]:
                        if jj == j:
                            print(f"        -> X#{target}  link delay {d:.3f}, delivered at {a:.3f}")
            if not brief:
                print(f"   pending after: {sum(len(v) for v in pending.values())} entries in slots {sorted(pending)}")

        # sources run at every 0.1 and emit, stamped at t; intake takes them in
        for name, (rows, mean) in conns.items():
            for i in range(FIRING_PER_STEP):
                neuron = (r * FIRING_PER_STEP + i) % N_SRC
                for j, m in mean.items():
                    a = t + m
                    pending[slot_of(a)].append((t, name, neuron, j, a, t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep", type=float, default=0.1, help="consumer X's timestep")
    parser.add_argument("--until", type=float, default=0.8)
    parser.add_argument("--links", action="store_true", help="also expand entries to their links")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--brief", action="store_true", help="one line per run")
    args = parser.parse_args()
    run(args.timestep, args.until, args.links, args.seed, args.brief)
