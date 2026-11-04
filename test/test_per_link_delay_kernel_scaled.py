"""
Same correctness check as test_per_link_delay_kernel.py (one source firing
once, targets split weight-1/weight-0 across delay buckets, bucket order
checked explicitly), but scaled to N=4000 targets so the update kernel
genuinely launches multiple workgroups (n_blocks = N / group_size = 125,
with the default group_size=32) instead of the single-block N=4 case.

Every tile-size knob (group_size, n_delay_buckets, spike_block) is left at
its real default -- this used to need group_size=32/n_delay_buckets=4
specifically to dodge a since-fixed compile-time blowup (see TODO.md: the
densify step's one-hot compare-and-reduce, and tl.static_range's forced
unroll of the per-spike loop, both replaced). n_blocks itself was always a
pure runtime launch parameter with no compile-time cost, which is why N
alone is enough to genuinely exercise multiple workgroups.
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

N = 4000


@pytest.fixture
def net():
    torch.cuda.set_device(0)
    A = counter() * 1
    A.potential = torch.tensor([15.0])

    B = delay_probe() * N
    B.potential = torch.zeros(N)
    B.timestep = 1.0
    # group_size/n_delay_buckets/spike_block all stay at their real defaults
    # -> n_blocks = 125, multiple workgroups

    conn = A >> all_to_all >> B
    idx = torch.arange(N)
    conn.weight = (idx % 2 == 0).float()  # even targets get weight 1, odd weight 0
    conn.delay = (idx % B.n_delay_buckets).float() * (1.0 / B.n_delay_buckets)

    return A.network, A, B, conn


def test_population_compiles_to_the_real_bucketed_kernel_at_scale(net):
    _, A, B, conn = net
    assert B.uses_per_link_delay_kernel
    assert B.n_blocks == 125
    module = B.compile()
    assert module.body[0].name == "per_link_delay_update_kernel"


def test_targets_fire_at_their_own_bucket_s_arrival_time_in_ascending_order(net):
    network, A, B, conn = net

    network.step(1.0)
    network.step(1.0)

    # B._fresh is the raw, per-block BlockedSpikes output (see
    # naer2/spike_gather.py): block b's valid entries are
    # node[b*cap : b*cap+count[b]] (already global neuron indices -- the
    # kernel stores `offs`, not a block-local index), not one flat
    # 0..count[0] range -- that only coincided for the earlier N=4,
    # single-block (n_blocks=1) test. Unpack every block here since
    # n_blocks=125 at this scale.
    fresh = B._fresh
    cap = fresh.cap
    nodes, times = [], []
    by_block: dict[int, list[tuple[int, float]]] = {}
    for b in range(fresh.count.shape[0]):
        c = int(fresh.count[b].item())
        block_nodes = fresh.node[b * cap : b * cap + c].tolist()
        block_times = fresh.time[b * cap : b * cap + c].tolist()
        nodes += block_nodes
        times += block_times
        by_block[b] = list(zip(block_nodes, block_times))

    n_buckets = B.n_delay_buckets
    expected_nodes = [i for i in range(N) if i % 2 == 0]
    expected_bucket = [(i % n_buckets) for i in expected_nodes]
    expected_times = [1.0 + bk * (1.0 / n_buckets) for bk in expected_bucket]

    assert sorted(nodes) == expected_nodes, (
        "exactly the even (weight-1) targets should fire, regardless of "
        "block/bucket ordering"
    )

    # Within each block, bucket order must be ascending: buckets are walked
    # 0, 1, 2, ... in sequence inside one kernel call (see
    # per_link_delay_update_kernel.py's module docstring), so each block's
    # own firings must already come out in non-decreasing time order.
    for block, firings in by_block.items():
        times_in_block = [t for _, t in firings]
        assert times_in_block == sorted(times_in_block), (
            f"block {block}'s firings are not in ascending bucket/time order: {firings}"
        )

    # Exact arrival time per target, independent of ordering.
    got = dict(zip(nodes, times))
    want = dict(zip(expected_nodes, expected_times))
    assert got == want


def test_weight_zero_targets_never_cross_threshold(net):
    network, A, B, conn = net

    for _ in range(5):
        network.step(1.0)

    odd = torch.arange(1, N, 2)
    assert torch.all(B.potential[odd] == 0.0)
