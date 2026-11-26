"""PD14's fused kernels phase by phase in quiet activity, from the in-kernel trace (naer2/trace.py): per
population and program, the cycles of the pull, the delivery, update + send and the emission; at level 2 also
inside the pull (fresh source runs, spike blocks, their segments, the appends).

    python tools/pd14_phases.py G [--level=1|2] [--rounds=30] [--mhz=SM_CLOCK] [--graph] [--dissect (level 3)]
    (on a cloud instance: cloud/remote run 'python tools/pd14_phases.py 256 --level=2')

Cycles are the SM clock's (NVIDIA %clock, AMD SHADER_CYCLES); --mhz converts them to us (default: nvidia-smi's
current SM clock, else none). Tracing costs a little itself (a store per event), level 2 more.
"""

import os
import statistics
import subprocess
import sys

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
import torch

import pd14_network as P

G = int(sys.argv[1])
LEVEL, ROUNDS, MHZ = 1, 30, None
for a in sys.argv[2:]:
    if a.startswith("--level="):
        LEVEL = int(a.split("=")[1])
    if a.startswith("--rounds="):
        ROUNDS = int(a.split("=")[1])
    if a.startswith("--mhz="):
        MHZ = float(a.split("=")[1])

obj = P._network(G)
net = obj[0]
pops = P._runtimes("network", obj)
r = 0
for r in range(1005):
    P._advance("network", obj, r)
    if r % 500 == 0:
        torch.cuda.synchronize()
for p in pops.values():
    P._runtime("network", p).check()
for p in pops.values():
    P._runtime("network", p).set_trace(cap=4096, level=LEVEL)
for k in range(1005, 1010):  # the traced variants compiled
    P._advance("network", obj, k)
torch.cuda.synchronize()
GRAPH = "--graph" in sys.argv  # one-round graphs (naer2/graph.py, the barrier): a round's kernels all at once
if GRAPH:
    net.graph_barrier = True
    for _ in range(3):
        net.run_graph(max_launches=1)
    torch.cuda.synchronize()
if MHZ is None:
    try:
        MHZ = float(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.split()[0])
    except Exception:
        MHZ = None

DISSECT = LEVEL >= 3 and "--dissect" in sys.argv  # (the temporary f:* / d:* marks, fused.py)
FIX = ("appended->bar", "fixup", "fixup grew", "count stores", "->pulled")
DEL = ("setup", "stage", "walk", "->barrier", "free", "->next", "slots", "stream blks")
dis = {}


def _dissect(n, ev):
    """One program's fix-up and delivery parts (cycles) into dis[n]."""
    d = dis.setdefault(n, {k: [] for k in FIX + DEL})
    t = {}
    for tag, aux, cyc in ev:
        t.setdefault(tag, (aux, cyc))
    app = [cyc for tag, _, cyc in ev if tag == "appended"]
    if {"f:bar", "f:fixed", "f:stored", "pulled"} <= set(t) and app:
        d["appended->bar"].append(t["f:bar"][1] - app[-1])
        d["fixup"].append(t["f:fixed"][1] - t["f:bar"][1])
        d["fixup grew"].append(100.0 * (t["f:fixed"][0] > 0) * (MHZ or 1))
        d["count stores"].append(t["f:stored"][1] - t["f:fixed"][1])
        d["->pulled"].append(t["pulled"][1] - t["f:stored"][1])
    if "state" in t and "slot" in t:
        d["setup"].append(t["slot"][1] - t["state"][1])
    prev, slots, blocks = None, 0, 0
    for tag, aux, cyc in ev:
        if tag == "slot":
            slots += 1
        elif tag == "streamed" and prev is not None and prev[0] in ("slot", "d:bar"):
            d["stage"].append(cyc - prev[1])
            blocks += 1
        elif tag == "walked" and prev is not None and prev[0] == "streamed":
            d["walk"].append(cyc - prev[1])
        elif tag == "d:bar" and prev is not None and prev[0] == "walked":
            d["->barrier"].append(cyc - prev[1])
        elif tag == "d:freed" and prev is not None and prev[0] in ("d:bar", "slot"):
            d["free"].append(cyc - prev[1])
        elif tag in ("slot", "delivered") and prev is not None and prev[0] == "d:freed":
            d["->next"].append(cyc - prev[1])
        if tag in ("slot", "streamed", "walked", "d:bar", "d:freed", "delivered"):
            prev = (tag, cyc)
    d["slots"].append(slots * (MHZ or 1))
    d["stream blks"].append(blocks * (MHZ or 1))


PHASES = ("pull", "delivery", "update+send", "emission")
stats = {n: {ph: [] for ph in PHASES + ("total", "blocks", "slots")} for n in pops}
worst = {n: (0, None) for n in pops}
for k in range(1010, 1010 + ROUNDS):
    if GRAPH:
        net.run_graph(max_launches=1)
    else:
        P._advance("network", obj, k)
    torch.cuda.synchronize()
    for n, p in pops.items():
        for prog in P._runtime("network", p).traced():
            ev = prog["events"]
            if not ev or prog["dropped"]:
                continue
            at = {}
            for tag, aux, cyc in ev:
                at.setdefault(tag, cyc)
            if not {"start", "state", "delivered", "sent", "end"} <= set(at):
                continue
            pulled = at.get("pulled", at["state"])
            st = stats[n]
            st["pull"].append(pulled - at["start"])
            st["delivery"].append(at["delivered"] - at["state"])
            st["update+send"].append(at["sent"] - at["delivered"])
            st["emission"].append(at["end"] - at["sent"])
            total = at["end"] - at["start"]
            st["total"].append(total)
            st["blocks"].append(sum(1 for tag, _, _ in ev if tag == "spikes"))
            st["slots"].append(sum(1 for tag, _, _ in ev if tag == "slot"))
            if total > worst[n][0]:
                worst[n] = (total, ev)
            if DISSECT:
                _dissect(n, ev)

us = (lambda c: f"{c / MHZ:6.2f}") if MHZ else (lambda c: f"{c:7.0f}")
unit = "us" if MHZ else "cycles"
print(f"{torch.cuda.get_device_name(0)} G={G}, trace level {LEVEL}, {ROUNDS} quiet rounds"
      f"{' (one-round graphs)' if GRAPH else ' (stepped)'}"
      f"{f', SM clock {MHZ:.0f} MHz' if MHZ else ''}: per program, median ({unit})")
print(f"  {'pop':5s} " + " ".join(f"{ph:>11s}" for ph in PHASES) + f" {'total':>9s} {'worst':>9s}"
      + (f" {'spike blks':>10s}" if LEVEL >= 2 else "") + f" {'slots':>6s}")
for n in sorted(pops, key=lambda n: -statistics.median(stats[n]["total"] or [0])):
    st = stats[n]
    if not st["total"]:
        continue
    row = " ".join(f"{us(statistics.median(st[ph])):>11s}" for ph in PHASES)
    blocks = f" {statistics.median(st['blocks']):10.0f}" if LEVEL >= 2 else ""
    print(f"  {n:5s} {row} {us(statistics.median(st['total'])):>9s} {us(worst[n][0]):>9s}{blocks}"
          f" {statistics.median(st['slots']):6.0f}")
if LEVEL >= 3:  # inside the append, per tile: the steps' durations, median over all tiles and programs
    steps = ("a:segments", "a:mean", "a:claimed", "a:targets", "a:stored")
    dur = {st: [] for st in steps + ("tile",)}
    tiles = []
    for n, p in pops.items():
        for prog in P._runtime("network", p).traced():
            ev = prog["events"]
            for i, (tag, aux, cyc) in enumerate(ev):
                if tag != "a:segments":
                    continue
                prev = ev[i - 1][2] if i else cyc
                seq = {"start": prev}
                for tag2, aux2, cyc2 in ev[i:i + 12]:
                    if tag2 in steps and tag2 not in seq:
                        seq[tag2] = cyc2
                    if tag2 == "a:stored":
                        break
                order_ = ["start"] + [st for st in steps if st in seq]
                for a_, b_ in zip(order_, order_[1:]):
                    dur[b_].append(seq[b_] - seq[a_])
                dur["tile"].append(seq.get("a:stored", seq["start"]) - seq["start"])
                n_it = 0
                for t2, _, _ in ev[i + 1:]:
                    if t2 == "a:targets":
                        n_it += 1
                    elif t2 in ("a:segments", "appended", "pulled"):
                        break
                tiles.append(n_it)
    print(f"  inside the append (last round), per tile, median ({unit}): " + ", ".join(
        f"{st}: {us(statistics.median(dur[st])).strip()}" for st in steps + ("tile",) if dur[st]))
    import collections
    hist = collections.Counter(tiles)
    print(f"  {len(dur['tile'])} tiles; link iterations per tile (count: tiles): "
          + ", ".join(f"{k}: {v}" for k, v in sorted(hist.items())))
if LEVEL >= 3:  # a spike block's parts, per population: spikes -> found, found -> its first tile, its tiles,
    # its last tile -> appended (median over the blocks of the last round)
    parts = ("spikes->found", "found->tile 1", "tiles", "last tile->appended", "tiles a block")
    print(f"  a spike block's parts, median ({unit}): {'pop':5s} " + " ".join(f"{s:>19s}" for s in parts))
    for n, p in pops.items():
        got = {s: [] for s in parts}
        for prog in P._runtime("network", p).traced():
            ev = prog["events"]
            i = 0
            while i < len(ev):
                if ev[i][0] != "spikes":
                    i += 1
                    continue
                j = i + 1
                blk = [ev[i]]
                while j < len(ev) and ev[j][0] not in ("spikes", "pulled"):
                    blk.append(ev[j])
                    if ev[j][0] == "appended":
                        break
                    j += 1
                i = j
                tags = [t for t, _, _ in blk]
                if "found" not in tags or "appended" not in tags:
                    continue
                t = {tag: cyc for tag, _, cyc in reversed(blk)}  # (first occurrence)
                seg = [cyc for tag, _, cyc in blk if tag == "a:segments"]
                stored = [cyc for tag, _, cyc in blk if tag == "a:stored"]
                got["spikes->found"].append(t["found"] - t["spikes"])
                if seg and stored:
                    first_start = max(cyc for tag, _, cyc in blk if cyc <= seg[0] and tag != "a:segments")
                    got["found->tile 1"].append(first_start - t["found"])
                    got["tiles"].append(stored[-1] - first_start)
                    got["last tile->appended"].append(t["appended"] - stored[-1])
                    got["tiles a block"].append(len(seg) * (MHZ or 1))
        print(f"  {'':34s}{n:5s} " + " ".join(
            f"{us(statistics.median(v)).strip() if v else '-':>19s}" for v in got.values()))
if DISSECT:  # medians over every round's programs; "fixup grew": % of programs; slots, blocks: counts
    for title, keys in (("fix-up", FIX), ("delivery", DEL)):
        print(f"  {title} parts, median ({unit}): {'pop':5s} " + " ".join(f"{k:>13s}" for k in keys))
        for n in pops:
            if n not in dis:
                continue
            row = []
            for k in keys:
                v = dis[n][k]
                if not v:
                    row.append("-")
                elif k == "fixup grew":
                    row.append(f"{statistics.mean(v) / (MHZ or 1):.0f}%")
                elif k in ("slots", "stream blks"):
                    row.append(f"{statistics.mean(v) / (MHZ or 1):.2f}")
                else:
                    row.append(us(statistics.median(v)).strip())
            print(f"  {'':30s}{n:5s} " + " ".join(f"{x:>13s}" for x in row))
if LEVEL >= 2:  # the pull's and delivery's steps, per population, median over its programs (last round)
    steps = ("header chain", "per spike block", "fix-up", "state->slot", "per stream block", "update+send")
    print(f"  steps (last round), median ({unit}): {'pop':5s} " + " ".join(f"{s:>13s}" for s in steps))
    for n, p in pops.items():
        got = {s: [] for s in steps}
        for prog in P._runtime("network", p).traced():
            ev = prog["events"]
            if not ev:
                continue
            at = {}
            for tag, _, cyc in ev:
                at.setdefault(tag, cyc)
            if "runs" in at:
                got["header chain"].append(at["runs"] - at["start"])
            blk_start = None
            for tag, _, cyc in ev:
                if tag == "spikes":
                    blk_start = cyc
                elif tag == "appended" and blk_start is not None:
                    got["per spike block"].append(cyc - blk_start)
                    blk_start = None
            appended = [cyc for tag, _, cyc in ev if tag == "appended"]
            if appended and "pulled" in at:
                got["fix-up"].append(at["pulled"] - appended[-1])
            if "state" in at and "slot" in at:
                got["state->slot"].append(at["slot"] - at["state"])
            st = None
            for tag, _, cyc in ev:
                if tag == "streamed":
                    st = cyc
                elif tag == "walked" and st is not None:
                    got["per stream block"].append(cyc - st)
            if "sent" in at and "delivered" in at:
                got["update+send"].append(at["sent"] - at["delivered"])
        print(f"  {'':34s}{n:5s} " + " ".join(
            f"{us(statistics.median(v)).strip() if v else '-':>13s}" for v in got.values()))
if LEVEL >= 2:  # the slowest program of the slowest population: its event sequence
    n = max(pops, key=lambda n: worst[n][0])
    ev = worst[n][1]
    print(f"  slowest program ({n}): " + ", ".join(
        f"{tag}{'(' + str(aux) + ')' if aux else ''}@{us(cyc).strip()}" for tag, aux, cyc in ev[:40]))
