"""Pull-intake variants of the fused kernel (runs batched or not, directory
or binary search) compared fairly: one process, one
PD14 network, the variants alternating round by round, the clock kept up by
untraced rounds in between. Per variant, from the kernels' own stamps:
  alone   sum over populations of the fused kernel's span, each run alone
  conc    the round's window, all populations concurrent
  pull    median per program of the pull-intake phase (us, from cycles)
plus the group directories' memory and build time.

    python tools/fused_sweep.py [G] [--rounds 10] [--rdna2]
"""
import os
import statistics as st
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
G = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 256
ROUNDS = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 10
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401

import torch

from naer2.greenfield.fused import fused_kernel
from pd14_setup import PD14

CONFIGS = [(True, True), (True, False)]  # (batch, directory)
NAMES = {(True, True): "batch+dir", (True, False): "batch+search", (False, True): "per run+dir",
         (False, False): "per run+search"}
HEAT = 5
t0 = time.time()
b = PD14(G, fused=True, count_received=False)
build_s = time.time() - t0
dir_mb = sum(t.directory()[0].numel() * 4 for p in b.pops.values() for t in p.plan.topologies) / 2**20
seg_mb = sum(t.buf.numel() * 4 for p in b.pops.values() for t in p.plan.topologies) / 2**20
masks = [b.fire_mask(r) for r in range(200)]
r = 0


def fire(r):
    return b.fire_at(masks[r % len(masks)], r)


def heat(n=HEAT):
    global r
    for _ in range(n):
        b.round(r, fire(r)); r += 1


def configure(cfg):
    for p in b.pops.values():
        p.batch, p.use_directory = cfg


def span(progs):
    ps = [q for q in progs if q["end"] > 0]
    return min(q["start"] for q in ps), max(q["end"] for q in ps)


def pull_us(progs):
    out = []
    for q in progs:
        ev = {tag: cyc for tag, _, cyc in q["events"]}
        if "pulled" in ev and q["end"] > q["start"] and q["events"][-1][2]:
            mhz = q["events"][-1][2] / ((q["end"] - q["start"]) * 0.01)
            out.append(ev["pulled"] / mhz)
    return out


def measure(concurrent):
    global r
    for p in b.pops.values():
        p.trace.buf.zero_()
    f = fire(r)
    if concurrent:
        b.update(r, f)
        torch.cuda.synchronize()
        spans = [span(p.traced()) for p in b.pops.values()]
        out = (max(e for _, e in spans) - min(s for s, _ in spans)) * 0.01
        pulls = []
    else:
        out, pulls = 0.0, []
        t = 10.0 + 0.1 * r
        for n, p in b.pops.items():
            p.run(t, t + 0.1, f[n], b.zeros[n], uniform=True, round_id=r)
            torch.cuda.synchronize()
            tr = p.traced()
            s, e = span(tr)
            out += (e - s) * 0.01
            pulls += pull_us(tr)
    r += 1
    return out, pulls


heat(20)
for p in b.pops.values():
    p.set_trace(1024)
regs = {}
for cfg in CONFIGS:  # compile every variant
    configure(cfg)
    measure(False); measure(True)
    k = [k for cache in fused_kernel.device_caches.values() for k in cache[0].values()][-1]
    regs[cfg] = (k.n_regs, k.n_spills)
heat(300)
res = {cfg: ([], [], []) for cfg in CONFIGS}
for _ in range(ROUNDS):
    for cfg in CONFIGS:
        configure(cfg)
        heat()
        a, pulls = measure(False)
        res[cfg][0].append(a)
        res[cfg][2].extend(pulls)
        heat()
        res[cfg][1].append(measure(True)[0])
b.check()
print(f"{torch.cuda.get_device_name(0)} fused G={G}, {ROUNDS} rounds per variant, interleaved; medians, us")
print(f"  topologies {seg_mb:.0f} MB + group directories {dir_mb:.0f} MB; network + topologies + "
      f"directories built in {build_s:.0f}s")
print(f"  {'pull intake':22s} {'alone':>7s} {'conc':>7s} {'pull/prog':>9s} {'VGPRs':>6s} {'spills':>6s}")
for cfg in CONFIGS:
    a, c, pulls = res[cfg]
    print(f"  {NAMES[cfg]:22s} {st.median(a):7.0f} {st.median(c):7.0f} {st.median(pulls):9.1f} "
          f"{regs[cfg][0]:6d} {regs[cfg][1]:6d}")
