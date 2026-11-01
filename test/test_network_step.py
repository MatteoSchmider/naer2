"""
First test of Network.step() actually driving real, compiled kernels
end-to-end -- no manual set_fresh_output()/compact()/merge_incoming() calls
anywhere, unlike test_network_gather.py (which tests the lifecycle wiring in
isolation with static spike data) or test_gpu_hardware.py (which launches one
population's kernel by hand). This is the first time Population.
_execute_gather_kernel/_execute_update_kernel and Network.step()'s three-phase
choreography (gather all -> merge all -> update due) run for real.

Reuses test/networks/counter_net (A -> B, all-to-all, counter node: update
-1, receive +10, send fires >10/>20/>30 with delay +0/+10/+20) specifically
because every quantity is a small integer, so the expected potentials after N
calls to network.step() can be worked out by hand -- see test_gpu_hardware.py
for the same reasoning applied to one bare kernel launch.

A.potential starts at 15 on neuron 0 only (just above the first threshold,
comfortably below the second), everything else at 0; B starts at all 0.

Hand-derived trace (t_step=1.0 each call; both populations have autonomous
dynamics -- counter.update always assigns potential -- so both are always
due and sub_dt never needs to split):

  step() #1, t: 0.0 -> 1.0
    gather (both): nothing carried yet, nothing due.
    merge (B): empty (A has nothing due yet).
    update:
      A: n_spikes=0; update -1 -> neuron0 14, rest -1; send: 14 > 10 (not
         > 20/30) -> bare n2.spike() records t=1.0, neuron0 resets to 0.
         A.potential = [0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
         A's fresh output: one spike (node=0, time=1.0).
      B: n_spikes=0; update -1 -> all -1; no sends.
         B.potential = [-1]*10

  step() #2, t: 1.0 -> 2.0
    gather (A): folds the (node=0, time=1.0) spike into carry; 1.0 <= 2.0 ->
      due now. gather (B): nothing due.
    merge (B): one entry (time=1.0, src=0, conn=0).
    update:
      A: n_spikes=0; update -1 -> neuron0 0-1=-1, rest -1-1=-2; no sends.
         A.potential = [-1, -2, -2, -2, -2, -2, -2, -2, -2, -2]
      B: n_spikes=1, source node 0, all-to-all -> delivers to every B
         neuron. receive (+10) at the spike's own time (1.0): -1+10=9;
         send check right after receive: 9 > 10? no. Final pass at t=2.0:
         update -1 -> 9-1=8; send: 8 > 10? no.
         B.potential = [8]*10

Skipped entirely when no CUDA/ROCm device is visible to torch (same reason
as test_gpu_hardware.py/test_network_gather.py: no CPU reference for the
compiled kernel or the spike-management kernels).
"""

import pytest
import torch

from test.networks.counter_net import build

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a real GPU (CUDA/ROCm); correctness independent of real "
    "hardware is already covered by the Triton CPU interpreter elsewhere",
)


@pytest.fixture
def net():
    torch.cuda.set_device(0)
    return build()


def test_step_drives_real_kernels_end_to_end_with_one_round_pipeline_latency(net):
    network, pops = net
    A, B = pops["A"], pops["B"]

    A.potential = torch.tensor([15.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    B.potential = torch.zeros(10)

    network.step(1.0)
    assert A.potential.tolist() == [0.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    assert B.potential.tolist() == [-1.0] * 10
    assert A._fresh.count.tolist() == [1], "A's spike hasn't surfaced to a gather yet"

    network.step(1.0)
    assert A.potential.tolist() == [-1.0, -2, -2, -2, -2, -2, -2, -2, -2, -2]
    assert B.potential.tolist() == [8.0] * 10, \
        "B must have received A's spike (+10) automatically, with no manual wiring"


def test_tensors_move_to_the_network_device_lazily(net):
    network, pops = net
    A, B = pops["A"], pops["B"]
    assert A.potential.device.type == "cpu" and B.potential.device.type == "cpu"

    network.step(1.0)

    assert A.potential.device.type == network.device.type
    assert B.potential.device.type == network.device.type
