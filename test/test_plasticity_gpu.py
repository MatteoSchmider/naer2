"""
Per-link edge state is GPU-resident (Fabric.bind_device): the kernel's
tl.store of a plastic update must persist across launches, reading
conn.fabric.<attr> between steps returns the live values in canonical link
order, and assigning it re-initializes them.

S (clock_source) fires every step; T (counter) ignores edge.weight, so the
weight only changes through plastic_synapse's forward hook (+factor per
delivery). Spikes reach T one step after emission (gather runs before
update), so after n steps every link has seen n - 1 deliveries.

Skipped entirely when no CUDA/ROCm device is visible to torch.
"""

import pytest
import torch

from test.edges import plastic_synapse
from test.nodes import clock_source, counter
from test.patterns import all_to_all

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a real GPU (CUDA/ROCm) to launch the compiled kernel",
)


@pytest.fixture
def net():
    torch.cuda.set_device(0)
    S = clock_source() * 2
    T = counter() * 3
    T.potential = torch.zeros(3)
    conn = S >> plastic_synapse(0.5, 0.1) @ all_to_all >> T
    return S.network, conn


def test_plastic_updates_persist_across_launches(net):
    network, conn = net
    for _ in range(4):
        network.step(1.0)

    weight = conn.fabric.weight
    assert weight.device.type == "cuda"
    assert torch.allclose(weight.cpu(), torch.full((6,), 0.5 + 3 * 0.1))


def test_assigning_reinitializes_the_live_values(net):
    network, conn = net
    network.step(1.0)
    network.step(1.0)  # one delivery so far: 0.6

    conn.fabric.weight = torch.arange(6, dtype=torch.float32)
    network.step(1.0)

    assert torch.allclose(conn.fabric.weight.cpu(), torch.arange(6, dtype=torch.float32) + 0.1)
