"""PD14 spike-management round trip, outside the library: per round every
cortical population's green-field update kernel (delivery + emission), then
every intake kernel, populations on their own streams (or sequential).

    python tools/pd14_roundtrip.py G [--backend triton|gluon] [--warps-update 4]
        [--warps-intake 4] [--sequential] [--rounds 50] [--unroll 8]
        [--stream 64] [--rdna2] [--breakdown]

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
ap.add_argument("--backend", default="triton", choices=["triton", "gluon"])
ap.add_argument("--warps-update", type=int, default=4)
ap.add_argument("--warps-intake", type=int, default=4)
ap.add_argument("--sequential", action="store_true")
ap.add_argument("--rounds", type=int, default=50)
ap.add_argument("--unroll", type=int, default=None)
ap.add_argument("--stream", type=int, default=None)
ap.add_argument("--entry-block", type=int, default=None)
ap.add_argument("--rdna2", action="store_true")
ap.add_argument("--breakdown", action="store_true")
args = ap.parse_args()
if args.rdna2:
    import rdna2  # noqa: F401

import torch

from pd14_setup import PD14

WARMUP, ROUNDS, CONC = 20, args.rounds, not args.sequential
b = PD14(args.G, args.backend, args.unroll, args.stream, args.warps_update, args.warps_intake,
         entry_block=args.entry_block)
# fire patterns: a pool of masks reused cyclically, turned into each round's
# (absolute) fire times up front, so the timed loop launches only the kernels
masks = [b.fire_mask(r) for r in range(min(WARMUP + 3 * ROUNDS, 200))]
fires = [b.fire_at(masks[r % len(masks)], r) for r in range(WARMUP + 3 * ROUNDS)]
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
print(f"{torch.cuda.get_device_name(0)}: {args.backend:6s} G={args.G:4d} U={p0.unroll} "
      f"S={p0.stream} chunk={p0.entry_block} warps update={args.warps_update} intake={args.warps_intake} "
      f"{'streams' if CONC else 'sequential':10s} per round {per_round * 1e3:6.2f} ms  "
      f"(~{b.delivered() / (WARMUP + ROUNDS):,.0f} links delivered/round; {b.links / 1e6:.0f}M links, "
      f"locality {b.links / b.segments:.2f}, topologies built in {b.build_s:.0f}s)", flush=True)

if args.breakdown:
    # host issue time per round (launches are asynchronous), then each phase
    # on its own (synchronized), rounds continuing in simulated time
    issue = []
    for r in range(WARMUP + ROUNDS, WARMUP + 2 * ROUNDS):
        t1 = time.perf_counter()
        b.round(r, fires[r], CONC)
        issue.append(time.perf_counter() - t1)
    upd, itk = [], []
    for r in range(WARMUP + 2 * ROUNDS, WARMUP + 3 * ROUNDS):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        b.update(r, fires[r], CONC)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        b.intake(CONC)
        torch.cuda.synchronize()
        upd.append(t2 - t1)
        itk.append(time.perf_counter() - t2)
    med = lambda v: sorted(v)[len(v) // 2] * 1e3
    print(f"  host issue per round {med(issue):.2f} ms | phases alone: update {med(upd):.2f} ms, "
          f"intake {med(itk):.2f} ms (each including one sync)")
