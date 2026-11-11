"""Uniform intake configurations (programs, tile shape) compared fairly:
one process, one PD14
network, configurations alternating round by round, the clock kept up by
untraced rounds in between. Per configuration, from the kernels' stamps:
  alone   sum over consumers of the intake kernel's span, each run alone
  conc    the intake phase window, all consumers concurrent

    python tools/intake_sweep.py [G] [--rounds 8] [--rdna2]
"""
import os
import statistics as st
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
G = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 256
ROUNDS = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 8
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401

import torch

from naer2.intake import INTAKE_TRACE_TAGS
from naer2.trace import Trace
from pd14_setup import PD14

b = PD14(G)
CUS = torch.cuda.get_device_properties(0).multi_processor_count
masks = [b.fire_mask(r) for r in range(200)]
r = 0


def fire(r):
    return b.fire_at(masks[r % len(masks)], r)


def heat(n):
    global r
    for _ in range(n):
        b.round(r, fire(r)); r += 1


SHAPES = [(1, 128), (2, 128), (4, 128), (1, 64), (2, 64), (1, 256)]  # (items per step, segments per item)
PROGS = [CUS * k for k in (2, 4, 8, 16, 32)]
CONFIGS = [(p, i, s) for p in PROGS for i, s in SHAPES]
traces = {n: Trace(max(PROGS), 64) for n in b.pops}


def configure(cfg):
    n_progs, items, segs = cfg
    for p in b.pops.values():
        p.plan.n_progs, p.plan.items, p.plan.item_segs = n_progs, items, segs


def span(t, n_progs):
    ps = [q for q in t.decode(INTAKE_TRACE_TAGS)[:n_progs] if q["end"] > 0]
    return min(q["start"] for q in ps), max(q["end"] for q in ps)


def measure(cfg, concurrent):
    global r
    for t in traces.values():
        t.buf.zero_()
    b.update(r, fire(r))
    torch.cuda.synchronize()
    if concurrent:
        b.intake(traces=traces)
        torch.cuda.synchronize()
        s = [span(t, cfg[0]) for t in traces.values()]
        out = (max(e for _, e in s) - min(a for a, _ in s)) * 0.01
    else:
        out = 0.0
        for n, p in b.pops.items():
            p.plan.run(True, num_warps=b.warps_intake, trace=traces[n])
            torch.cuda.synchronize()
            a, e = span(traces[n], cfg[0])
            out += (e - a) * 0.01
    r += 1
    return out


heat(20)
for cfg in CONFIGS:  # compile every configuration
    configure(cfg)
    measure(cfg, False)
heat(300)
res = {cfg: ([], []) for cfg in CONFIGS}
for _ in range(ROUNDS):
    for cfg in CONFIGS:
        heat(5)
        configure(cfg)
        res[cfg][0].append(measure(cfg, False))
        heat(5)
        configure(cfg)
        res[cfg][1].append(measure(cfg, True))
b.check()
print(f"{torch.cuda.get_device_name(0)} G={G}, {CUS} CUs (as torch counts), {ROUNDS} rounds per configuration; medians, us")
print(f"  {'programs':>8s} {'items':>5s} {'segs':>5s} {'alone':>7s} {'conc':>7s}")
for cfg in CONFIGS:
    a, c = res[cfg]
    print(f"  {cfg[0]:8d} {cfg[1]:5d} {cfg[2]:5d} {st.median(a):7.0f} {st.median(c):7.0f}")
