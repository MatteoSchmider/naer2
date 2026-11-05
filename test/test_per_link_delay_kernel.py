"""
Real-hardware test of the per-link-delay (intra-timestep transmission delay) update
kernel (naer2/transpiler/templates/per_link_delay_update_kernel.py) against
a real network, built the normal way through the DSL -- no synthetic
Triton-level fixtures, matching test_gpu_hardware.py's style.

One source neuron (A, a `counter` primed to fire exactly once, at exactly
t=1.0, with no extra delay encoded in its own send() thresholds -- see
test_gpu_hardware.py's identical trick) connects all-to-all to four target
neurons (B, `delay_probe`): weight alternates [1, 0, 1, 0], delay places
each target in a different one of B's 4 buckets (bucket_width = B.timestep
/ B.n_delay_buckets = 1.0 / 4 = 0.25):

    target  weight  delay  bucket  expected arrival = emission(1.0) + bucket*0.25
      0        1     0.00     0      1.00  -> fires (weight 1 crosses threshold 0.5)
      1        0     0.25     1      1.25  -> never fires (weight 0)
      2        1     0.50     2      1.50  -> fires
      3        0     0.75     3      1.75  -> never fires

Buckets are walked in ascending order inside one kernel call (bucket 0's
whole link range fully processed -- receive, then send -- before bucket 1's
starts, see the template's module docstring), so target 0's spike is
written into B's local spike buffer strictly before target 2's: checking
that order directly is this test's way of confirming "buckets go through
one after another," not just that the final set of who-fired is right.

Skipped entirely when no CUDA/ROCm device is visible to torch (no CPU
reference for the compiled kernel, same as test_gpu_hardware.py).
"""

import pytest
import torch

from test.nodes import counter, delay_probe
from test.patterns import all_to_all

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a real GPU (CUDA/ROCm); correctness independent of real "
    "hardware is already covered by the Triton CPU interpreter elsewhere",
)


@pytest.fixture
def net():
    torch.cuda.set_device(0)
    A = counter() * 1
    A.potential = torch.tensor([15.0])  # crosses only the lowest threshold -> bare n2.spike() at exactly t

    B = delay_probe() * 4
    B.potential = torch.zeros(4)
    B.timestep = 1.0
    B.n_delay_buckets = 4

    conn = A >> all_to_all >> B
    conn.weight = torch.tensor([1.0, 0.0, 1.0, 0.0])
    conn.delay = torch.tensor([0.0, 0.25, 0.5, 0.75])

    return A.network, A, B, conn


def test_per_link_delay_population_compiles_to_the_real_bucketed_kernel(net):
    _, A, B, conn = net
    assert B.uses_per_link_delay_kernel
    module = B.compile()
    assert module.body[0].name == "per_link_delay_update_kernel"


def test_targets_fire_at_their_own_bucket_s_arrival_time_in_ascending_order(net):
    network, A, B, conn = net

    network.step(1.0)  # t=1.0: A fires; nothing merged into B yet (gather runs before update)
    network.step(1.0)  # t=2.0: B's own due tick (period=1.0) consumes A's now-due spike

    count = int(B._fresh.count[0].item())
    nodes = B._fresh.node[:count].tolist()
    times = B._fresh.time[:count].tolist()

    assert list(zip(nodes, times)) == [(0, 1.0), (2, 1.5)], (
        "only the weight-1 targets (buckets 0 and 2) should fire, each at its own "
        "bucket's arrival time, in ascending bucket order"
    )


def test_weight_zero_targets_never_cross_threshold(net):
    network, A, B, conn = net

    for _ in range(5):
        network.step(1.0)

    assert B.potential[1].item() == 0.0
    assert B.potential[3].item() == 0.0
