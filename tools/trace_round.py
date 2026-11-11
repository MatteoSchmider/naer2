"""Where the PD14 round's GPU time goes, from the kernels' own stamps
(naer2/trace.py): every program of every update and intake kernel on one
wall-clock axis.

    python tools/trace_round.py G [--backend gluon|triton] [--rounds 10] [--unroll 8]

Two modes, each over --rounds rounds after warm-up:
  concurrent  as in production (populations on their own streams): the
              round's timeline -- phase windows, gaps, how busy the GPU is
  isolated    every kernel alone (synchronized): its span, its programs'
              durations and occupancy, and its phases (critical program and
              all programs)
"""
import argparse
import os
import statistics as st
import sys
from collections import defaultdict

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
ap = argparse.ArgumentParser()
ap.add_argument("G", type=int)
ap.add_argument("--backend", default="gluon", choices=["triton", "gluon"])
ap.add_argument("--rounds", type=int, default=10)
ap.add_argument("--unroll", type=int, default=None)
ap.add_argument("--rdna2", action="store_true")
ap.add_argument("--main-cap", type=int, default=None)
ap.add_argument("--heat", type=int, default=0,
                help="untraced rounds before each mode, and HEAT // 20 between measured rounds, so power "
                     "management keeps the clock up (short bursty runs stay at the lowest DPM level)")
ap.add_argument("--entry-block", type=int, default=None)
ap.add_argument("--no-counters", action="store_true", help="compile the stand-in receive counters out (PD14's kernel)")
ap.add_argument("--stream", type=int, default=None)
args = ap.parse_args()
if args.rdna2:
    import rdna2  # noqa: F401

import torch

from naer2.intake import INTAKE_TRACE_TAGS
from naer2.trace import Trace
from pd14_setup import PD14

US = 0.01  # REALTIME tick: 10 ns
med = st.median
b = PD14(args.G, args.backend, unroll=args.unroll, stream=args.stream, main_cap=args.main_cap,
         entry_block=args.entry_block, count_received=not args.no_counters)
_masks = [b.fire_mask(r) for r in range(200)]


class _Fires:  # round r's fire times, made on use from a cyclic pool of fire masks
    def __getitem__(self, r):
        return b.fire_at(_masks[r % len(_masks)], r)


fires = _Fires()
r = 0
for _ in range(20):
    b.round(r, fires[r]); r += 1
for p in b.pops.values():
    p.set_trace(1, 1024)
itraces = {n: Trace(p.plan.n_progs, 256) for n, p in b.pops.items()}
for _ in range(3):  # compile the traced kernels
    b.round(r, fires[r], intake_traces=itraces); r += 1
torch.cuda.synchronize()


def collect():
    return ({("update", n): p.traced() for n, p in b.pops.items()},
            {("intake", n): t.decode(INTAKE_TRACE_TAGS) for n, t in itraces.items()})


def ran(progs):
    return [q for q in progs if q["end"] > 0]


def reset():
    for p in b.pops.values():
        p.trace.zero_()
    for t in itraces.values():
        t.buf.zero_()


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
    b.round(r, fires[r], intake_traces=itraces); r += 1
    torch.cuda.synchronize()
    u, i = collect()
    timeline.append({**u, **i})

# ---- isolated: each kernel alone
iso = defaultdict(list)
heat(args.heat)
for _ in range(args.rounds):
    heat(args.heat // 20)
    reset()
    t = 10.0 + 0.1 * r
    for n, p in b.pops.items():
        p.run(t, t + 0.1, fires[r][n], b.zeros[n], uniform=True, num_warps=b.warps_update)
        torch.cuda.synchronize()
    for n, p in b.pops.items():
        p.plan.run(True, num_warps=b.warps_intake, trace=itraces[n])
        torch.cuda.synchronize()
    r += 1
    u, i = collect()
    for k, v in {**u, **i}.items():
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
print(f"{torch.cuda.get_device_name(0)} {args.backend} G={args.G} U={p0.unroll} "
      f"chunk={p0.entry_block} stream={p0.stream}, "
      f"{args.rounds} rounds per mode\n")
print("CONCURRENT (production): median over rounds, us")
rows = defaultdict(list)
for rd in timeline:
    allp = [(q["start"], q["end"]) for progs in rd.values() for q in ran(progs)]
    t0, t1 = min(s for s, _ in allp), max(e for _, e in allp)
    upd = [(q["start"], q["end"]) for (kind, _), progs in rd.items() if kind == "update" for q in ran(progs)]
    itk = [(q["start"], q["end"]) for (kind, _), progs in rd.items() if kind == "intake" for q in ran(progs)]
    rows["round span (first program start -> last end)"].append((t1 - t0) * US)
    rows["  GPU busy (union of all programs)"].append(union(allp) * US)
    rows["  update phase window"].append((max(e for _, e in upd) - min(s for s, _ in upd)) * US)
    rows["  gap update end -> first intake start"].append((min(s for s, _ in itk) - max(e for _, e in upd)) * US)
    rows["  intake phase window"].append((max(e for _, e in itk) - min(s for s, _ in itk)) * US)
for k, v in rows.items():
    print(f"  {k:48s} {med(v):8.1f}")


def phases(events, kind):
    out = defaultdict(int)
    for (a, _, ca), (bb, _, cb) in zip(events, events[1:]):
        if kind == "update":
            key = {("start", "state"): "load state", ("state", "slot"): "to first slot",
                   ("slot", "headers"): "entry headers", ("headers", "streamed"): "gather -> LDS",
                   ("streamed", "walked"): "walk", ("delivered", "sent"): "propagate + send",
                   ("sent", "end"): "summary + state"}.get((a, bb))
            if key is None:
                key = "empty slots" if a == "slot" else "between blocks" if a == "walked" else f"{a}->{bb}"
        else:
            key = {("start", "sources"): "read sources", ("sources", "tile"): "first tile",
                   ("tile", "tile"): "later tiles", ("tile", "end"): "finish",
                   ("sources", "end"): "no work"}.get((a, bb), f"{a}->{bb}")
        out[key] += cb - ca
    return out


print("\nISOLATED (each kernel alone): median over rounds")
print(f"  {'kernel':14s} {'progs':>5s} {'span us':>8s} {'prog med':>8s} {'prog max':>8s} {'alive':>5s} "
      f"{'last start':>10s} {'MHz':>5s}  work")
for kind in ("update", "intake"):
    totals = defaultdict(int)
    for (k, n), rounds in sorted(iso.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if k != kind:
            continue
        span, pmed, pmax, alive, lstart, mhz, work = [], [], [], [], [], [], defaultdict(list)
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
                for key, cyc in phases(q["events"], kind).items():
                    totals[key] += cyc
            last = max(ps, key=lambda q: q["end"])
            for key, cyc in phases(last["events"], kind).items():
                crit[key].append(cyc)
            if kind == "update":
                work["links"].append(sum(x for t_, x, _ in sum((q["events"] for q in ps), []) if t_ == "walked"))
            else:
                tiles = [x for q in ps for t_, x, _ in q["events"] if t_ == "tile"]
                work["tiles"].append(len(tiles)); work["segments"].append(sum(tiles))
        f = med(mhz) if mhz else 1
        w = " ".join(f"{key} {med(v):,.0f}" for key, v in work.items())
        print(f"  {kind + ' ' + n:14s} {len(rounds[0]):5d} {med(span):8.1f} {med(pmed):8.1f} {med(pmax):8.1f} "
              f"{max(alive):5d} {med(lstart):10.1f} {f:5.0f}  {w}")
        top = sorted(((key, med(v) / f) for key, v in crit.items()), key=lambda kv: -kv[1])[:5]
        print(f"  {'':14s} last-finishing program, us: " + ", ".join(f"{k_} {v:.1f}" for k_, v in top))
    tot = sum(totals.values())
    print(f"  {kind}: share of all programs' cycles: " + ", ".join(
        f"{k_} {v / tot:.0%}" for k_, v in sorted(totals.items(), key=lambda kv: -kv[1]) if v / tot >= 0.01) + "\n")
