"""
Hybrid scheduling showcase — one network, three populations of 2,000
neurons each, mixing clock-driven and event-driven dynamics, deliberately
outside the SNN domain to show the scheduling mechanism is generic (see
TODO.md).

  generator     -- only a send(): fires every call it's due for, on its
                    own explicit Population.timestep. No state.
  integrator    -- clock-driven via its own explicit Population.timestep
                    (coarser than the generator's), sums the arrival time
                    of every delivery.
  differentiator -- event-driven: no update(), no explicit timestep -- its
                    cadence comes entirely from its incoming connection's
                    delay (a small integer, 3). Tracks the difference
                    between consecutive arrival times.

Both retain more than one generator tick's worth of deliveries before
they're due to consume them (2 and 3 respectively), exercising the
scheduling step (combine()) at 2,000-wide populations.

Nodes read `time` rather than a `source.<attr>` on purpose: a receive
hook's `source.attr` reflects the source's CURRENT state, not a snapshot
pinned to that delivery -- while a delivery waits to be scheduled it would reflect the source's
latest value by the time it's finally read, not its value as of that
delivery. `time` is fixed forever once recorded. See TODO.md.

The first step compiles several Triton kernels at this population size;
cost varies with whether Triton's on-disk kernel cache is already warm
from a previous run (can be a few seconds, or close to a minute cold).
Steady state (reported below) is what actually answers "how fast is this."

Run from the project root:
    python -m showcases.hybrid_scheduling
    PYTHONPATH=. python showcases/hybrid_scheduling.py
"""
import os
import sys
import time as time_mod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import naer2 as n2
from naer2.types import Region
from naer2.types.points import IndexPoints
from naer2.types.position import Position

N = 2000
GROUP_SIZE = 256  # power of 2; keeps the spike-management kernels' compile time reasonable at this N


@n2.pattern
def one_to_one(pos: Position, src: Region, tgt=None) -> Region:
    """N independent parallel lanes -- every lane runs the same computation."""
    return IndexPoints(pos.coords.to(torch.int64).unsqueeze(0))


@n2.node(receive=None)
class clock_generator:
    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        n2.spike()


@n2.node
class integrator:
    def __init__(self):
        self.total: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.receives.time, edge: n2.receives.edge, source: n2.receives.source):
        self.total += time


@n2.node
class differentiator:
    def __init__(self):
        self.last_time: float = 0.0
        self.diff: float = 0.0
        self.n_received: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.receives.time, edge: n2.receives.edge, source: n2.receives.source):
        self.diff = time - self.last_time
        self.last_time = time
        self.n_received += 1.0


if not torch.cuda.is_available():
    print("No CUDA/ROCm device visible -- this showcase needs real hardware.")
    sys.exit(0)

torch.cuda.set_device(0)

# carry_cap/per_src_cap must exceed N (every neuron fires every round it's
# due); see Population.__init__ and TODO.md.
gen = clock_generator() * N
gen.group_size = GROUP_SIZE
gen.local_spike_cap = GROUP_SIZE
gen.carry_cap = 128
gen.per_src_cap = 128
gen.timestep = 1.0

integ = integrator() * N
integ.total = torch.zeros(N)
integ.group_size = GROUP_SIZE
integ.local_spike_cap = GROUP_SIZE
integ.carry_cap = 128
integ.per_src_cap = 2048
integ.timestep = 2.0

diff = differentiator() * N
diff.last_time = torch.zeros(N)
diff.diff = torch.zeros(N)
diff.n_received = torch.zeros(N)
diff.group_size = GROUP_SIZE
diff.local_spike_cap = GROUP_SIZE
diff.carry_cap = 128
diff.per_src_cap = 2048

gen >> one_to_one >> integ
conn_diff = gen >> one_to_one >> diff
conn_diff.delay = 3

network = gen.network

print(f"{'='*70}")
print(f"Hybrid scheduling: {N}-wide generator -> integrator (timestep=2.0) / "
      f"differentiator (delay=3)")

t0 = time_mod.time()
network.step(1.0)
torch.cuda.synchronize()
compile_s = time_mod.time() - t0

iters = 14
t0 = time_mod.time()
for _ in range(iters):
    network.step(1.0)
torch.cuda.synchronize()
steady_ms = (time_mod.time() - t0) / iters * 1e3

print(f"first step (incl. kernel compile): {compile_s:.1f}s")
print(f"steady state: {steady_ms:.2f} ms/step ({1000/steady_ms:.0f} steps/s)")
print()

assert bool((integ.total == integ.total[0]).all())
assert bool((diff.n_received == diff.n_received[0]).all())
assert bool((diff.diff == diff.diff[0]).all())
print(f"all {N} lanes agree                      ✓")

# integrator: due at t=1 (nothing yet), then every 2.0 (t=3,5,...,15),
# summing the two retained arrival times each time. Cumulative total after
# the tick ending at t=2k+1 is sum(1..2k) = k(2k+1).
assert integ.last_update == 15.0
assert integ.total[0].item() == float(7 * 15)
print(f"integrator.total[0] = {integ.total[0].item():.0f}  (= sum(1..14) = 7*15) ✓")

# differentiator: due every delay=3 (t=1,4,7,10,13), retaining exactly 3
# arrival times each time -- a constant 1.0 difference between consecutive
# integer ticks, and exactly 3 more received per tick (none dropped).
assert diff.last_update == 13.0
assert diff.diff[0].item() == 1.0
assert diff.n_received[0].item() == 12.0
print(f"differentiator.diff[0] = {diff.diff[0].item():.1f}  (constant, derivative of a unit ramp) ✓")
print(f"differentiator.n_received[0] = {diff.n_received[0].item():.0f}  (4 ticks x 3 retained, none dropped) ✓")
print(f"{'='*70}")
