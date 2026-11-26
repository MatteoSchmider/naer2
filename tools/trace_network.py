"""Where PD14's round time goes through Network.step, from the fused
kernels' own stamps (naer2/trace.py), per phase -- for two ring kinds side by
side: per-cell runs (the harness's capacities) and the default per-group
arenas (Population.ring_block / ring_in_flight).

    python tools/trace_network.py [--ms 100] [--rounds 20] [--level 1|2|3] [--G 256] [--rdna2]

From --ms on, --rounds traced rounds (one network step each, synchronized,
the stamps read back after every round). Per phase: the cycles of every
program of every population summed over the rounds (the GPU's work, not its
wall time -- programs overlap), as per round means; and the round's span
(first program start to last end) and GPU busy time. --level 2 splits the
pull, 3 also the append's chain per tile.
"""
import argparse
import os
import statistics as st
import subprocess
import sys
from collections import defaultdict

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]

PHASES = {("start", "state"): "load state", ("state", "slot"): "to first slot",
          ("start", "pulled"): "pull intake", ("pulled", "state"): "load state",
          ("start", "runs"): "pull: fresh runs", ("runs", "spikes"): "pull: spike loads",
          ("appended", "spikes"): "pull: spike loads", ("spikes", "found"): "pull: segment lookup",
          ("found", "appended"): "pull: append", ("appended", "pulled"): "pull: e-group barrier",
          ("runs", "pulled"): "pull: no spikes",
          ("found", "a:segments"): "append: segment words", ("a:stored", "a:segments"): "append: segment words",
          ("a:segments", "a:mean"): "append: bucket mean", ("a:mean", "a:claimed"): "append: claim (LDS)",
          ("a:claimed", "a:targets"): "append: link targets", ("a:stored", "a:targets"): "append: link targets",
          ("a:targets", "a:stored"): "append: store records", ("a:stored", "appended"): "append: tile end",
          ("a:claimed", "appended"): "append: no links",
          ("slot", "streamed"): "records -> LDS (first block)", ("walked", "streamed"): "records -> LDS (next blocks)",
          ("streamed", "walked"): "walk", ("delivered", "sent"): "propagate + send", ("sent", "end"): "summary + state"}
US = 0.01  # REALTIME tick: 10 ns


def phases(events):
    out = defaultdict(int)
    for (a, _, ca), (b, _, cb) in zip(events, events[1:]):
        key = PHASES.get((a, b))
        if key is None:
            key = "empty slots + freeing" if a == "slot" else "between blocks" if a == "walked" else f"{a}->{b}"
        out[key] += cb - ca
    return out


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


def measure(kind, ms, rounds, level, G):
    """One ring kind in this process: {phase: cycles per round}, span and busy (us)."""
    import torch
    from pd14_network import RES, _network

    net, pops = _network(G, harness_caps=(kind == "cells"))
    while net.t < ms - 1e-9:
        net.step(RES)
    torch.cuda.synchronize()
    for p in pops.values():
        p.runtime.check()  # a pending growth now, not in the traced rounds
        p.runtime.set_trace(1024, level=level)
    for _ in range(3):  # compile the traced kernels
        net.step(RES)
    torch.cuda.synchronize()
    cyc, span, busy = defaultdict(int), [], []
    for _ in range(rounds):
        for p in pops.values():
            p.runtime.trace.buf.zero_()
        net.step(RES)
        torch.cuda.synchronize()
        allp = []
        for p in pops.values():
            for q in p.runtime.traced():
                if q["end"] <= 0:
                    continue
                allp.append((q["start"], q["end"]))
                for k, v in phases(q["events"]).items():
                    cyc[k] += v
        span.append((max(e for _, e in allp) - min(s for s, _ in allp)) * US)
        busy.append(union(allp) * US)
    for p in pops.values():
        p.runtime.check()
    return {k: v / rounds for k, v in cyc.items()}, st.median(span), st.median(busy)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=100.0)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--G", type=int, default=256)
    ap.add_argument("--rdna2", action="store_true")
    ap.add_argument("--kind", choices=["cells", "arena"], help=argparse.SUPPRESS)  # one kind (a child process)
    args = ap.parse_args()
    if args.rdna2:
        import rdna2  # noqa: F401
    if args.kind:  # the child: print its result as a repr line
        print("RESULT", repr(measure(args.kind, args.ms, args.rounds, args.level, args.G)), flush=True)
        sys.exit(0)
    res = {}
    for kind in ("cells", "arena"):  # each in its own process: two PD14s don't fit next to each other
        out = subprocess.run([sys.executable, __file__, "--kind", kind, "--ms", str(args.ms), "--rounds",
                              str(args.rounds), "--level", str(args.level), "--G", str(args.G)]
                             + (["--rdna2"] if args.rdna2 else []), capture_output=True, text=True)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            sys.exit(f"{kind} failed:\n{out.stderr[-3000:]}")
        res[kind] = eval(line[7:])
    (pc, sc, bc), (pa, sa, ba) = res["cells"], res["arena"]
    print(f"PD14 through Network.step, G={args.G}, from {args.ms} ms, {args.rounds} rounds, level {args.level}")
    print(f"  round span      cells {sc:7.1f} us   arena {sa:7.1f} us")
    print(f"  GPU busy        cells {bc:7.1f} us   arena {ba:7.1f} us")
    print(f"  {'phase (all programs, per round)':34s} {'cells Mcyc':>10s} {'arena Mcyc':>10s} {'diff':>8s}")
    keys = sorted(set(pc) | set(pa), key=lambda k: -(pa.get(k, 0) + pc.get(k, 0)))
    tc, ta = sum(pc.values()), sum(pa.values())
    for k in keys:
        c, a = pc.get(k, 0) / 1e6, pa.get(k, 0) / 1e6
        if max(c, a) >= 0.005 * tc / 1e6:
            print(f"  {k:34s} {c:10.2f} {a:10.2f} {a - c:+8.2f}")
    print(f"  {'total':34s} {tc / 1e6:10.2f} {ta / 1e6:10.2f} {(ta - tc) / 1e6:+8.2f}")
