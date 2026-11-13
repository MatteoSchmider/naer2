"""PD14 spike-management round trip, outside the library: per round one
fused kernel per cortical population (pull intake, delivery, emission),
populations on their own streams (or one after another on one).

    python tools/pd14_roundtrip.py G [--warps 4] [--sequential] [--rounds 50]
        [--unroll 8] [--stream 256] [--rdna2] [--breakdown]

--breakdown: host issue time per round (median), with the launch queue
filling up and with the GPU idle at each round's start, and the round alone
(synchronized).

--rdna2: for the RX 6600 eGPU, with
    HSA_OVERRIDE_GFX_VERSION_2=10.3.0 HIP_VISIBLE_DEVICES=1
"""
import argparse
import os
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
ap = argparse.ArgumentParser()
ap.add_argument("G", type=int)
ap.add_argument("--warps", type=int, default=4)
ap.add_argument("--sequential", action="store_true")
ap.add_argument("--rounds", type=int, default=50)
ap.add_argument("--unroll", type=int, default=None)
ap.add_argument("--stream", type=int, default=None)
ap.add_argument("--rdna2", action="store_true")
ap.add_argument("--breakdown", action="store_true")
args = ap.parse_args()
if args.rdna2:
    import rdna2  # noqa: F401

import torch

from pd14_setup import PD14

WARMUP, ROUNDS, CONC = 20, args.rounds, not args.sequential
b = PD14(args.G, unroll=args.unroll, stream=args.stream, warps=args.warps)
# fire patterns: a pool of masks reused cyclically, turned into each round's
# (absolute) fire times up front, so the timed loop launches only the kernels
masks = [b.fire_mask(r) for r in range(min(WARMUP + 4 * ROUNDS, 200))]
fires = [b.fire_at(masks[r % len(masks)], r) for r in range(WARMUP + 4 * ROUNDS)]
for r in range(WARMUP):
    b.round(r, fires[r], CONC)
torch.cuda.synchronize()
t0 = time.perf_counter()
for r in range(WARMUP, WARMUP + ROUNDS):
    b.round(r, fires[r], CONC)
torch.cuda.synchronize()
per_round = (time.perf_counter() - t0) / ROUNDS
b.check()
p0 = next(iter(b.pops.values()))
print(f"{torch.cuda.get_device_name(0)}: fused G={args.G:4d} U={p0.unroll} S={p0.stream} warps={args.warps} "
      f"{'streams' if CONC else 'sequential':10s} per round {per_round * 1e3:6.2f} ms  "
      f"(~{b.delivered() / (WARMUP + ROUNDS):,.0f} links delivered/round; {b.links / 1e6:.0f}M links, "
      f"locality {b.links / b.segments:.2f}, topologies built in {b.build_s:.0f}s)", flush=True)

if args.breakdown:
    med = lambda v: sorted(v)[len(v) // 2] * 1e3
    issue, idle, alone = [], [], []
    r = WARMUP + ROUNDS
    for _ in range(ROUNDS):  # launches asynchronous, the queue filling up
        t1 = time.perf_counter()
        b.round(r, fires[r], CONC)
        issue.append(time.perf_counter() - t1)
        r += 1
    for _ in range(ROUNDS):  # the GPU idle at the round's start: issue, then the round to its end
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        b.round(r, fires[r], CONC)
        t2 = time.perf_counter()
        torch.cuda.synchronize()
        idle.append(t2 - t1)
        alone.append(time.perf_counter() - t1)
        r += 1
    print(f"  host issue per round {med(issue):.2f} ms (queue filling), {med(idle):.2f} ms (GPU idle) | "
          f"round alone {med(alone):.2f} ms (including one sync)")
