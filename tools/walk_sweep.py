"""Delivery-walk configurations of the Gluon update kernel (chunk size,
stream length, links per LDS read), compared fairly:
one process, one PD14 network, the configurations alternating round by
round. Per configuration, from the kernels' own stamps (naer2/trace.py):
  alone   sum over populations of the update kernel's span, each run alone
  conc    the update phase window with all populations concurrent
plus the kernel's VGPRs and LDS (G=256 unless given).

    python tools/walk_sweep.py [G] [--rounds 12] [--rdna2]
"""
import os
import statistics as st
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
G = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 256
ROUNDS = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 12
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401

import torch

from pd14_setup import PD14
from test.update_gluon import update_kernel_gluon

CONFIGS = [  # (entries per chunk, stream, U, stand-in receive counters, waves_per_eu target or 0)
    (64, 128, 8, False, 16), (128, 256, 8, False, 0), (128, 256, 8, False, 12), (128, 256, 8, False, 14),
    (64, 128, 8, False, 12), (64, 128, 8, False, 14),
]
HEAT = 5  # untraced rounds before each measured one: keep the clock up
b = PD14(G, "gluon")
_masks = [b.fire_mask(r) for r in range(200)]


class _Fires:
    def __getitem__(self, r):
        return b.fire_at(_masks[r % len(_masks)], r)


fires = _Fires()
r = 0
for _ in range(20):
    b.round(r, fires[r]); r += 1
for p in b.pops.values():
    p.set_trace(1, 1024)


CURRENT = [None]


def configure(cfg):
    CURRENT[0] = cfg
    for p in b.pops.values():
        p.entry_block, p.stream, p.unroll, p.count_received, p.waves_per_eu = cfg


def heat(n=HEAT):
    global r
    for _ in range(n):
        b.round(r, fires[r]); r += 1


def span(progs):
    ps = [q for q in progs if q["end"] > 0]
    return min(q["start"] for q in ps), max(q["end"] for q in ps)


def alive(progs):
    ev = sorted([(q["start"], 1) for q in progs if q["end"] > 0] + [(q["end"], -1) for q in progs if q["end"] > 0])
    a = m = 0
    for _, d in ev:
        a += d
        m = max(m, a)
    return m


most_alive = {}


def round_alone():
    global r
    t = 10.0 + 0.1 * r
    total = 0.0
    for n, p in b.pops.items():
        p.trace.zero_()
        p.run(t, t + 0.1, fires[r][n], b.zeros[n], uniform=True, num_warps=b.warps_update)
        torch.cuda.synchronize()
        tr = p.traced()
        s, e = span(tr)
        total += (e - s) * 0.01
        if n == "E_4":
            most_alive[CURRENT[0]] = max(most_alive.get(CURRENT[0], 0), alive(tr))
    b.intake(); r += 1
    return total


def round_conc():
    global r
    for p in b.pops.values():
        p.trace.zero_()
    b.update(r, fires[r])
    torch.cuda.synchronize()
    spans = [span(p.traced()) for p in b.pops.values()]
    b.intake(); r += 1
    return (max(e for _, e in spans) - min(s for s, _ in spans)) * 0.01


regs = {}
for cfg in CONFIGS:  # compile and warm up every configuration
    configure(cfg)
    round_alone(); round_conc()
    k = [k for cache in update_kernel_gluon.device_caches.values() for k in cache[0].values()][-1]
    regs[cfg] = (k.n_regs, k.metadata.shared, k.n_spills)
res = {cfg: ([], []) for cfg in CONFIGS}
heat(300)
for _ in range(ROUNDS):
    for cfg in CONFIGS:
        configure(cfg)
        heat()
        res[cfg][0].append(round_alone())
        heat()
        res[cfg][1].append(round_conc())
b.check()
print(f"{torch.cuda.get_device_name(0)} G={G}, {ROUNDS} rounds per configuration, interleaved; medians, us")
print(f"  {'chunk':>5s} {'stream':>6s} {'U':>3s} {'counters':>8s} {'waves':>5s} {'alone':>7s} {'conc':>7s} "
      f"{'VGPRs':>6s} {'spills':>6s} {'E_4 alive':>9s}")
for cfg in CONFIGS:
    a, c = res[cfg]
    print(f"  {cfg[0]:5d} {cfg[1]:6d} {cfg[2]:3d} {'yes' if cfg[3] else 'no':>8s} {cfg[4] or '-':>5} {st.median(a):7.0f} "
          f"{st.median(c):7.0f} {regs[cfg][0]:6d} {regs[cfg][2]:6d} {most_alive.get(cfg, 0):5d}/86")
