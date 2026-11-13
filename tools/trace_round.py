"""Where the PD14 round's GPU time goes, from the kernels' own stamps
(naer2/trace.py): every program of every population's fused kernel on one
wall-clock axis.

    python tools/trace_round.py G [--rounds 10] [--unroll 8] [--heat N] [--no-counters] [--level 1|2|3]

Two modes, each over --rounds rounds after warm-up:
  concurrent  as in production (populations on their own streams): the
              round's span and how busy the GPU is
  isolated    every kernel alone (synchronized): its span, its programs'
              durations and occupancy, and its phases (critical program and
              all programs); --level 2 and 3 split the pull intake further
"""
import argparse
import os
import statistics as st
import sys
from collections import defaultdict

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
ap = argparse.ArgumentParser()
ap.add_argument("G", type=int)
ap.add_argument("--rounds", type=int, default=10)
ap.add_argument("--unroll", type=int, default=None)
ap.add_argument("--rdna2", action="store_true")
ap.add_argument("--main-cap", type=int, default=None)
ap.add_argument("--heat", type=int, default=0,
                help="untraced rounds before each mode, and HEAT // 20 between measured rounds, so power "
                     "management keeps the clock up (short bursty runs stay at the lowest DPM level)")
ap.add_argument("--no-counters", action="store_true", help="compile the stand-in receive counters out (PD14's kernel)")
ap.add_argument("--stream", type=int, default=None)
ap.add_argument("--level", type=int, default=1, help="trace level: 2 also stamps inside the pull, 3 inside the append")
args = ap.parse_args()
if args.rdna2:
    import rdna2  # noqa: F401

import torch

from pd14_setup import PD14

US = 0.01  # REALTIME tick: 10 ns
med = st.median
b = PD14(args.G, unroll=args.unroll, stream=args.stream, main_cap=args.main_cap,
         count_received=not args.no_counters)
_masks = [b.fire_mask(r) for r in range(200)]


class _Fires:  # round r's fire times, made on use from a cyclic pool of fire masks
    def __getitem__(self, r):
        return b.fire_at(_masks[r % len(_masks)], r)


fires = _Fires()
r = 0
for _ in range(20):
    b.round(r, fires[r]); r += 1
for p in b.pops.values():
    p.set_trace(1024, level=args.level)
for _ in range(3):  # compile the traced kernels
    b.round(r, fires[r]); r += 1
torch.cuda.synchronize()


def collect():
    return {n: p.traced() for n, p in b.pops.items()}


def ran(progs):
    return [q for q in progs if q["end"] > 0]


def reset():
    for p in b.pops.values():
        p.trace.buf.zero_()


def heat(n):
    global r
    if n:
        for _ in range(n):
            b.round(r, fires[r]); r += 1
        torch.cuda.synchronize()


# ---- concurrent: the round timeline
timeline = []
heat(args.heat)
for _ in range(args.rounds):
    heat(args.heat // 20)
    reset()
    b.round(r, fires[r]); r += 1
    torch.cuda.synchronize()
    timeline.append(collect())

# ---- isolated: each kernel alone
iso = defaultdict(list)
heat(args.heat)
for _ in range(args.rounds):
    heat(args.heat // 20)
    reset()
    t = 10.0 + 0.1 * r
    for n, p in b.pops.items():
        p.run(t, t + 0.1, fires[r][n], b.zeros[n], uniform=True, num_warps=b.warps, round_id=r)
        torch.cuda.synchronize()
    r += 1
    for k, v in collect().items():
        iso[k].append(v)
b.check()


def union(intervals):
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


p0 = next(iter(b.pops.values()))
print(f"{torch.cuda.get_device_name(0)} fused G={args.G} U={p0.unroll} stream={p0.stream} level={args.level}, "
      f"{args.rounds} rounds per mode\n")
print("CONCURRENT (production): median over rounds, us")
rows = defaultdict(list)
for rd in timeline:
    allp = [(q["start"], q["end"]) for progs in rd.values() for q in ran(progs)]
    t0, t1 = min(s for s, _ in allp), max(e for _, e in allp)
    rows["round span (first program start -> last end)"].append((t1 - t0) * US)
    rows["  GPU busy (union of all programs)"].append(union(allp) * US)
for k, v in rows.items():
    print(f"  {k:48s} {med(v):8.1f}")


PHASES = {("start", "state"): "load state", ("state", "slot"): "to first slot",
          ("start", "pulled"): "pull intake", ("pulled", "state"): "load state",
          # TRACE 2: the pull
          ("start", "runs"): "pull: fresh runs", ("runs", "spikes"): "pull: spike loads",
          ("appended", "spikes"): "pull: spike loads", ("spikes", "found"): "pull: segment lookup",
          ("found", "appended"): "pull: append", ("appended", "pulled"): "pull: e-group barrier",
          ("runs", "pulled"): "pull: no spikes",
          # TRACE 3: the append's chain, per tile
          ("found", "a:segments"): "append: segment words", ("a:stored", "a:segments"): "append: segment words",
          ("a:segments", "a:mean"): "append: bucket mean", ("a:mean", "a:claimed"): "append: claim (LDS)",
          ("a:claimed", "a:targets"): "append: link targets", ("a:stored", "a:targets"): "append: link targets",
          ("a:targets", "a:stored"): "append: store records", ("a:stored", "appended"): "append: tile end",
          ("a:claimed", "appended"): "append: no links",
          # delivery
          ("slot", "streamed"): "records -> LDS (first block)", ("walked", "streamed"): "records -> LDS (next blocks)",
          ("streamed", "walked"): "walk", ("delivered", "sent"): "propagate + send", ("sent", "end"): "summary + state"}


def phases(events):
    out = defaultdict(int)
    for (a, _, ca), (bb, _, cb) in zip(events, events[1:]):
        key = PHASES.get((a, bb))
        if key is None:
            key = "empty slots" if a == "slot" else "between blocks" if a == "walked" else f"{a}->{bb}"
        out[key] += cb - ca
    return out


print("\nISOLATED (each kernel alone): median over rounds")
print(f"  {'population':10s} {'progs':>5s} {'span us':>8s} {'prog med':>8s} {'prog max':>8s} {'alive':>5s} "
      f"{'last start':>10s} {'MHz':>5s}  work")
totals = defaultdict(int)
for n, rounds in sorted(iso.items()):
    span, pmed, pmax, alive, lstart, mhz, links = [], [], [], [], [], [], []
    crit = defaultdict(list)
    for progs in rounds:
        ps = ran(progs)
        t0, t1 = min(q["start"] for q in ps), max(q["end"] for q in ps)
        span.append((t1 - t0) * US)
        d = [(q["end"] - q["start"]) * US for q in ps]
        pmed.append(med(d)); pmax.append(max(d))
        lstart.append((max(q["start"] for q in ps) - t0) * US)
        ev = sorted([(q["start"], 1) for q in ps] + [(q["end"], -1) for q in ps])
        a = m = 0
        for _, s_ in ev:
            a += s_; m = max(m, a)
        alive.append(m)
        for q in ps:
            if q["end"] > q["start"] and q["events"] and q["events"][-1][2]:
                mhz.append(q["events"][-1][2] / ((q["end"] - q["start"]) * US))
            for key, cyc in phases(q["events"]).items():
                totals[key] += cyc
        last = max(ps, key=lambda q: q["end"])
        for key, cyc in phases(last["events"]).items():
            crit[key].append(cyc)
        links.append(sum(x for q in ps for t_, x, _ in q["events"] if t_ == "walked"))
    f = med(mhz) if mhz else 1
    print(f"  {n:10s} {len(rounds[0]):5d} {med(span):8.1f} {med(pmed):8.1f} {med(pmax):8.1f} "
          f"{max(alive):5d} {med(lstart):10.1f} {f:5.0f}  links {med(links):,.0f}")
    top = sorted(((key, med(v) / f) for key, v in crit.items()), key=lambda kv: -kv[1])[:5]
    print(f"  {'':10s} last-finishing program, us: " + ", ".join(f"{k_} {v:.1f}" for k_, v in top))
tot = sum(totals.values())
print("  share of all programs' cycles: " + ", ".join(
    f"{k_} {v / tot:.0%}" for k_, v in sorted(totals.items(), key=lambda kv: -kv[1]) if v / tot >= 0.01))
