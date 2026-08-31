"""
PD14 integration test — builds the full Potjans & Diesmann 2014 cortical
microcircuit and reports network statistics.

Run from the project root:
    python -m test.pd14
    PYTHONPATH=. python test/pd14.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test.networks.PD14 import build

t0 = time.time()
net = build()
elapsed = time.time() - t0

pops = sorted(net.populations, key=lambda p: int(p.shape[0][1]), reverse=True)
conns = sorted(net.connections, key=lambda c: int(c._source.shape[0][1]), reverse=True)

print(f"\n{'='*60}")
print(f"Network: {len(pops)} populations, {len(conns)} connections  [{elapsed:.1f}s]")

print(f"\nPopulations:")
for pop in pops:
    name = type(pop.instance).__name__
    size = int(pop.shape[0][1])
    print(f"  {name:<20} n={size:>6}  dyn={list(pop.dynamic_attrs)}")

pop = pops[0]
name = type(pop.instance).__name__
size = int(pop.shape[0][1])
static = {k: v for k, v in pop.instance.__dict__.items() if k not in pop.dynamic_attrs}
print(f"\n--- Population: {name} (n={size}) ---")
print(f"  static:  {static}")
print(f"  dynamic: {list(pop.dynamic_attrs)}")
if hasattr(pop, "potential"):
    t = pop.potential
    print(f"  potential: mean={t.mean():.2f}, std={t.std():.2f}, range=[{t.min():.2f}, {t.max():.2f}]")

conn = conns[0]
src, tgt = conn._source, conn._target
print(
    f"\n--- Connection: {type(src.instance).__name__}(n={int(src.shape[0][1])}) "
    f"→ {type(tgt.instance).__name__}(n={int(tgt.shape[0][1])}) ---"
)
print(f"  edge:   {type(conn.fabric.instance).__name__}")
print(
    f"  weight: mean={conn.fabric.weight.mean():.5f}, std={conn.fabric.weight.std():.5f}, "
    f"range=[{conn.fabric.weight.min():.5f}, {conn.fabric.weight.max():.5f}]"
)
print(
    f"  delay:  mean={conn.fabric.delay.mean():.3f}, std={conn.fabric.delay.std():.3f}, "
    f"range=[{conn.fabric.delay.min():.3f}, {conn.fabric.delay.max():.3f}]"
)

topo = conn.fabric.topology
fan_out = topo.source_offsets[1:] - topo.source_offsets[:-1]
print(f"\n--- Topology ---")
print(f"  n_sources={len(topo.source_offsets)-1}, n_links={len(topo)}")
print(f"  fan-out: mean={fan_out.float().mean():.1f}, min={fan_out.min()}, max={fan_out.max()}")
for i in range(min(3, len(topo.source_offsets) - 1)):
    s, e = topo.source_offsets[i].item(), topo.source_offsets[i + 1].item()
    targets = topo.target_indices[s:e].tolist()
    suffix = "..." if len(targets) > 8 else ""
    print(f"  source[{i}] → {targets[:8]}{suffix}")
print(f"{'='*60}")
