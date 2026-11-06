"""
Real-hardware test of the per-link-delay (intra-timestep transmission delay) update
kernel (naer2/transpiler/templates/per_link_delay_update_kernel.py) against
a real network, built the normal way through the DSL -- no synthetic
Triton-level fixtures, matching test_gpu_hardware.py's style.

One source neuron (A, a `counter` primed to fire exactly once, at exactly
t=1.0, with no extra delay encoded in its own send() thresholds -- see
test_gpu_hardware.py's identical trick) connects all-to-all to four target
neurons (B, `delay_probe`): weight alternates [1, 0, 1, 0], delay places
each target in a different one of B's 4 buckets (bucket width =
B.delay_resolution = 0.25):

    target  weight  delay  bucket  delivery time = diagonal centre (4 + bucket + 1) * 0.25
      0        1     0.00     0      1.25  -> fires (weight 1 crosses threshold 0.5)
      1        0     0.25     1      1.50  -> never fires (weight 0)
      2        1     0.50     2      1.75  -> fires
      3        0     0.75     3      2.00  -> never fires

(emission 1.0 is emission bin 4; delivery time is the (emission bin, delay
bucket) diagonal's centre, see the template's module docstring.) Diagonals
are walked in ascending order inside one kernel call, so target 0's spike
is written into B's local spike buffer strictly before target 2's.
test_deliveries_follow_arrival_order_not_emission_order below covers the
case that actually needs the convolution: two spikes with different
emission times whose arrivals interleave.

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
    B.delay_resolution = 0.25

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

    assert list(zip(nodes, times)) == [(0, 1.25), (2, 1.75)], (
        "only the weight-1 targets (buckets 0 and 2) should fire, each at its own "
        "bucket's arrival time, in ascending bucket order"
    )


def test_deliveries_follow_arrival_order_not_emission_order():
    """A0 fires first (1.0) on a long delay (0.75 -> bucket 3), A1 fires
    later (1.5) on no delay (bucket 0). Arrival order is the reverse of
    emission order: A1's delivery (diagonal 6 -> 1.75) must be processed
    before A0's (diagonal 7 -> 2.0). Processing in emission order -- each
    spike walking all its buckets before the next -- would fire B0 first.
    B's input is set directly so both spikes land in one kernel call."""
    torch.cuda.set_device(0)
    A = counter() * 2
    A.potential = torch.zeros(2)
    B = delay_probe() * 2
    B.potential = torch.zeros(2)
    B.timestep = 1.0
    B.delay_resolution = 0.25

    conn = A >> all_to_all >> B
    # links in source-major order: A0->B0, A0->B1, A1->B0, A1->B1
    conn.weight = torch.tensor([1.0, 0.0, 0.0, 1.0])
    conn.delay = torch.tensor([0.75, 0.0, 0.0, 0.0])

    dev = "cuda:0"
    B._merged = (
        torch.tensor([1.0, 1.5], device=dev),
        torch.tensor([0, 1], device=dev, dtype=torch.int32),
        torch.tensor([0, 0], device=dev, dtype=torch.int32),
        torch.tensor([2], device=dev, dtype=torch.int32),
    )
    B._execute_update_kernel(2.0)

    count = int(B._fresh.count[0].item())
    fired = list(zip(B._fresh.node[:count].tolist(), B._fresh.time[:count].tolist()))
    assert fired == [(1, 1.75), (0, 2.0)]


def test_inter_timestep_delays_deliver_once_each_in_their_own_later_call():
    """Delays beyond B's own period (1.0): A fires once at t=1.0 (emission
    bin 4, w=0.25); delays 0.0 / 1.25 / 2.5 are buckets 0 / 5 / 10 ->
    diagonals 4 / 9 / 14 -> delivery times 1.25 / 2.5 / 3.75. B runs at
    t=2, 3, 4, ...: each call may only deliver diagonals with time <= its t,
    so each target fires in a different call, exactly once, and the spike
    leaves B's pending input once its last bucket is delivered."""
    torch.cuda.set_device(0)
    A = counter() * 1
    A.potential = torch.tensor([15.0])
    B = delay_probe() * 3
    B.potential = torch.zeros(3)
    B.timestep = 1.0
    B.delay_resolution = 0.25

    conn = A >> all_to_all >> B
    conn.weight = torch.ones(3)
    conn.delay = torch.tensor([0.0, 1.25, 2.5])
    network = A.network

    network.step(1.0)  # t=1.0: A fires
    fired_per_call = []
    for _ in range(5):  # B runs at t=2..6
        network.step(1.0)
        count = int(B._fresh.count[0].item())
        fired_per_call.append(
            list(zip(B._fresh.node[:count].tolist(), B._fresh.time[:count].tolist()))
        )

    assert fired_per_call == [[(0, 1.25)], [(1, 2.5)], [(2, 3.75)], [], []]
    assert B._merged is None


def test_weight_zero_targets_never_cross_threshold(net):
    network, A, B, conn = net

    for _ in range(5):
        network.step(1.0)

    assert B.potential[1].item() == 0.0
    assert B.potential[3].item() == 0.0
