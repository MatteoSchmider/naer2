"""PD14's fused kernels phase by phase in quiet activity, from the in-kernel trace (naer2/trace.py): per
population and program, the cycles of the pull, the delivery, update + send and the emission; at level 2 also
inside the pull (fresh source runs, spike blocks, their segments, the appends).

    python tools/pd14_phases.py G [--level=1|2] [--rounds=30] [--mhz=SM_CLOCK]
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
if MHZ is None:
    try:
        MHZ = float(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.split()[0])
    except Exception:
        MHZ = None

PHASES = ("pull", "delivery", "update+send", "emission")
stats = {n: {ph: [] for ph in PHASES + ("total", "blocks", "slots")} for n in pops}
worst = {n: (0, None) for n in pops}
for k in range(1010, 1010 + ROUNDS):
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

us = (lambda c: f"{c / MHZ:6.2f}") if MHZ else (lambda c: f"{c:7.0f}")
unit = "us" if MHZ else "cycles"
print(f"{torch.cuda.get_device_name(0)} G={G}, trace level {LEVEL}, {ROUNDS} quiet rounds"
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
                tiles.append(sum(1 for t2, _, _ in ev[i:i + 200] if t2 == "a:targets"))
    print(f"  inside the append (last round), per tile, median ({unit}): " + ", ".join(
        f"{st}: {us(statistics.median(dur[st])).strip()}" for st in steps + ("tile",) if dur[st]))
    print(f"  {len(dur['tile'])} tiles; target/store iterations per tile: median "
          f"{statistics.median([sum(1 for _ in [0])]) if False else ''}(see a:targets counts)")
if LEVEL >= 2:  # the slowest program of the slowest population: its event sequence
    n = max(pops, key=lambda n: worst[n][0])
    ev = worst[n][1]
    print(f"  slowest program ({n}): " + ", ".join(
        f"{tag}{'(' + str(aux) + ')' if aux else ''}@{us(cyc).strip()}" for tag, aux, cyc in ev[:40]))
