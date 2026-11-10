"""PD14 spike-management round trip, outside the library: per round every
cortical population's green-field update kernel (delivery + emission), then
every intake kernel, populations on their own streams (or sequential).

    python tools/pd14_roundtrip.py G [--warps-update 4] [--warps-intake 4]
        [--sequential] [--rounds 50] [--walk 2] [--unroll 8] [--rdna2]
        [--breakdown]

--rdna2: for the RX 6600 eGPU, with
    HSA_OVERRIDE_GFX_VERSION_2=10.3.0 HIP_VISIBLE_DEVICES=1
"""
import argparse
import math
import os
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
ap = argparse.ArgumentParser()
ap.add_argument("G", type=int)
ap.add_argument("--warps-update", type=int, default=4)
ap.add_argument("--warps-intake", type=int, default=4)
ap.add_argument("--sequential", action="store_true")
ap.add_argument("--rounds", type=int, default=50)
ap.add_argument("--walk", type=int, default=2)
ap.add_argument("--unroll", type=int, default=8)
ap.add_argument("--rdna2", action="store_true")
ap.add_argument("--backend", default="triton", choices=["triton", "gluon"])
ap.add_argument("--stream", type=int, default=64)
ap.add_argument("--breakdown", action="store_true")
args = ap.parse_args()
if args.rdna2:
    import rdna2  # noqa: F401

import torch

from naer2.topology_new import TopologyNew
from test.networks.PD14.net import build, description
import test.test_update_emission_gpu as T
from test.test_update_emission_gpu import Pop

T.BACKEND = args.backend

G, CONC = args.G, not args.sequential
torch.cuda.set_device(0)
torch.manual_seed(0)
dev, RES, WARMUP, ROUNDS = "cuda", 0.1, 20, args.rounds
rate = dict(E_23=0.0, I_23=2.2, E_4=5.0, I_4=6.0, E_5=8.5, I_5=8.5, E_6=1.7, I_6=7.6)
names = dict(zip(description["populations"]["size"], description["populations"]["name"]))
MAIN_CAP = 8 * G

net = build()
cortical = {p: names[len(p.shape)] for p in net.populations if len(p.shape) in names}
pops = {name: Pop(len(p.shape), group_size=G) for p, name in cortical.items()}
for p in pops.values():
    p.walk, p.unroll, p.stream = args.walk, args.unroll, args.stream
t0 = time.time()
links = segs = 0
for p, name in cortical.items():
    conns = [c for c in p.incoming if c._source in cortical]
    topos = [TopologyNew(c.fabric.topology, len(p.shape), G, dev, delay=c.fabric.delay, resolution=RES,
                         attrs={"weight": c.fabric.weight}) for c in conns]
    links += sum(t.n_links for t in topos)
    segs += sum(t.n_segments for t in topos)
    n_slots = int((max(float(t.bucket_mean.max()) for t in topos) + 0.2) / RES) + 2
    pops[name].connect([pops[cortical[c._source]] for c in conns], topos, n_slots,
                       main_cap=MAIN_CAP, side_cap=16)
torch.cuda.synchronize()
build_s = time.time() - t0
del net

g = torch.Generator(device=dev).manual_seed(1)
fire = [{name: torch.where(torch.rand(pop.n, device=dev, generator=g) < rate[name] * 1e-4,
                           torch.full((pop.n,), 10.0 + RES * r, device=dev),
                           torch.full((pop.n,), math.inf, device=dev))
         for name, pop in pops.items()} for r in range(WARMUP + 3 * ROUNDS)]
zeros = {name: torch.zeros(pop.n, device=dev) for name, pop in pops.items()}
streams = {name: torch.cuda.Stream() for name in pops}


def phase(fns, concurrent):
    if not concurrent:
        for fn in fns.values():
            fn()
        return
    main = torch.cuda.current_stream()
    for name, fn in fns.items():
        streams[name].wait_stream(main)
        with torch.cuda.stream(streams[name]):
            fn()
    for name in fns:
        main.wait_stream(streams[name])


def round_(r, concurrent=CONC):
    t = 10.0 + RES * r
    phase({n: (lambda p=p, n=n: p.run(t, t + RES, fire[r][n], zeros[n], uniform=True, num_warps=args.warps_update))
           for n, p in pops.items()}, concurrent)
    phase({n: (lambda p=p: p.plan.run(True, num_warps=args.warps_intake)) for n, p in pops.items()}, concurrent)


for r in range(WARMUP):
    round_(r)
torch.cuda.synchronize()
t0 = time.perf_counter()
for r in range(WARMUP, WARMUP + ROUNDS):
    round_(r)
torch.cuda.synchronize()
per_round = (time.perf_counter() - t0) / ROUNDS
for p in pops.values():
    p.ring.check()
delivered = sum(int(p.n_received.sum()) for p in pops.values()) / (WARMUP + ROUNDS)
print(f"{torch.cuda.get_device_name(0)}: {args.backend:6s} G={G:4d} walk={args.walk} U={args.unroll} S={args.stream} warps update={args.warps_update} "
      f"intake={args.warps_intake} {'streams' if CONC else 'sequential':10s} per round {per_round * 1e3:6.2f} ms  "
      f"(~{delivered:,.0f} links delivered/round; {links / 1e6:.0f}M links, locality {links / segs:.2f}, "
      f"topologies built in {build_s:.0f}s)", flush=True)

if args.breakdown:
    # host issue time per round (launches are asynchronous), then wall clock
    # of each kernel kind alone, rounds continuing in simulated time
    issue = []
    for r in range(WARMUP + ROUNDS, WARMUP + 2 * ROUNDS):
        t1 = time.perf_counter()
        round_(r)
        issue.append(time.perf_counter() - t1)
    torch.cuda.synchronize()
    # each phase on its own (synchronized), populations still concurrent
    upd, itk = [], []
    for r in range(WARMUP + 2 * ROUNDS, WARMUP + 3 * ROUNDS):
        t = 10.0 + RES * r
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        phase({n: (lambda p=p, n=n: p.run(t, t + RES, fire[r][n], zeros[n], uniform=True,
                                          num_warps=args.warps_update)) for n, p in pops.items()}, CONC)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        phase({n: (lambda p=p: p.plan.run(True, num_warps=args.warps_intake)) for n, p in pops.items()}, CONC)
        torch.cuda.synchronize()
        upd.append(t2 - t1)
        itk.append(time.perf_counter() - t2)
    med = lambda v: sorted(v)[len(v) // 2] * 1e3
    print(f"  host issue per round {med(issue):.2f} ms | phases alone: update {med(upd):.2f} ms, "
          f"intake {med(itk):.2f} ms (each including one sync)")
