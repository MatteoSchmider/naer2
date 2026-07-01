import logging

logging.basicConfig(
    format="{%(filename)s:%(lineno)d} %(levelname)s - %(message)s", level=logging.DEBUG
)
# to let 780M (gfx1103) be used, fake gfx1100 with 'export HSA_OVERRIDE_GFX_VERSION=11.0.0'

from test import PD14_net

net = PD14_net.build()

pops = sorted(net.populations, key=lambda p: int(p.shape[0][1]), reverse=True)
conns = sorted(net.connections, key=lambda c: int(c.source.shape[0][1]), reverse=True)

print(f"\n{'='*60}")
print(f"Network: {len(pops)} populations, {len(conns)} connections")

print(f"\nPopulations:")
for pop in pops:
    name = type(pop.instance).__name__
    size = int(pop.shape[0][1])
    print(f"  {name:<20} n={size:>6}  dyn={list(pop.dynamic_attrs)}")

# Largest cortical population
pop = pops[0]
name = type(pop.instance).__name__
size = int(pop.shape[0][1])
static = {k: v for k, v in pop.instance.__dict__.items() if k not in pop.dynamic_attrs}
print(f"\n--- Population: {name} (n={size}) ---")
print(f"  static:  {static}")
print(f"  dynamic: {list(pop.dynamic_attrs)}")
if hasattr(pop, "potential"):
    t = pop.potential
    print(f"  potential: n={t.numel()}, mean={t.mean():.2f}, std={t.std():.2f}, range=[{t.min():.2f}, {t.max():.2f}]")

# Largest connection
conn = conns[0]
src = conn.source
tgt = conn._target
print(f"\n--- Connection: {type(src.instance).__name__}(n={int(src.shape[0][1])}) → {type(tgt.instance).__name__}(n={int(tgt.shape[0][1])}) ---")
print(f"  edge:   {type(conn._fabric.instance).__name__}")
print(f"  weight: n={conn.weight.numel()}, mean={conn.weight.mean():.5f}, std={conn.weight.std():.5f}, range=[{conn.weight.min():.5f}, {conn.weight.max():.5f}]")
print(f"  delay:  n={conn.delay.numel()}, mean={conn.delay.mean():.3f}, std={conn.delay.std():.3f}, range=[{conn.delay.min():.3f}, {conn.delay.max():.3f}]")
print(f"{'='*60}")
