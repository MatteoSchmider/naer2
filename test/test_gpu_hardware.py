"""
First tests that actually launch a compiled naer2 kernel on real GPU
hardware via Triton, rather than only inspecting its generated AST (that's
test_coverage.py's job) or running it through the Triton CPU interpreter.

Uses test/networks/counter_net (the `counter` node: update subtracts 1,
receive adds 10 flat, send fires at thresholds 10/20/30 with delays
0/+10/+20) specifically because every number involved is a small integer,
so the expected spike times/counts and final potential can be worked out by
hand from the starting potential and the incoming spikes alone.

Skipped entirely when no CUDA/ROCm device is visible to torch.
"""

import ast
import importlib.util
import pathlib
import tempfile

import pytest
import torch

from test.networks.counter_net import build

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a real GPU (CUDA/ROCm); correctness independent of real "
    "hardware is already covered by the Triton CPU interpreter elsewhere",
)

DEVICE = "cuda:0"
GROUP_SIZE, SPIKE_BLOCK, LINK_BLOCK, LOCAL_SPIKE_CAP = 4, 4, 16, 64


def _load_update_kernel(population):
    """Compile `population`'s kernel AST to source and import it for real,
    returning the loaded @triton.jit function (not just its AST)."""
    module = population.compile()
    source = "import triton\nimport triton.language as tl\n\n" + ast.unparse(module) + "\n"
    path = pathlib.Path(tempfile.mkdtemp()) / "naer2_gpu_kernel.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.no_delay_update_kernel


def _launch(fn, conn, N, n_blocks, potential, t, spike_time=(), spike_src=(), spike_conn=()):
    n_spikes = len(spike_time)
    bt = conn.fabric.topology.blocked(GROUP_SIZE, N)
    args = dict(
        potential_ptr=torch.tensor(potential, device=DEVICE, dtype=torch.float32),
        n_neurons=N,
        spike_time_ptr=torch.tensor(spike_time or [0.0], device=DEVICE, dtype=torch.float32),
        spike_src_ptr=torch.tensor(spike_src or [0], device=DEVICE, dtype=torch.int32),
        spike_conn_ptr=torch.tensor(spike_conn or [0], device=DEVICE, dtype=torch.int32),
        n_spikes=n_spikes,
        conn0_src_offsets_ptr=bt.offsets.flatten().to(device=DEVICE, dtype=torch.int64),
        conn0_target_indices_ptr=bt.target_local.to(device=DEVICE, dtype=torch.int32),
        t=t,
        local_spike_node_ptr=torch.zeros(n_blocks * LOCAL_SPIKE_CAP, device=DEVICE, dtype=torch.int32),
        local_spike_time_ptr=torch.zeros(n_blocks * LOCAL_SPIKE_CAP, device=DEVICE, dtype=torch.float32),
        local_spike_count_ptr=torch.zeros(n_blocks, device=DEVICE, dtype=torch.int32),
    )
    args = {k: v for k, v in args.items() if k in fn.arg_names}
    fn[(n_blocks,)](**args, GROUP_SIZE=GROUP_SIZE, SPIKE_BLOCK=SPIKE_BLOCK, LINK_BLOCK=LINK_BLOCK, LOCAL_SPIKE_CAP=LOCAL_SPIKE_CAP)
    torch.cuda.synchronize()

    # local_spike_{node,time,count}_ptr together are exactly a BlockedSpikes
    # (naer2/spike_gather.py) -- flatten its GROUP-blocked, unsorted layout
    # into a plain (node, time) list for the hand-derived assertions below.
    counts = args["local_spike_count_ptr"].tolist()
    nodes_blocked = args["local_spike_node_ptr"].tolist()
    times_blocked = args["local_spike_time_ptr"].tolist()
    spikes = []
    for g, c in enumerate(counts):
        for i in range(c):
            spikes.append((nodes_blocked[g * LOCAL_SPIKE_CAP + i], times_blocked[g * LOCAL_SPIKE_CAP + i]))
    spikes.sort()
    return args["potential_ptr"].tolist(), spikes


@pytest.fixture(scope="module")
def pops():
    torch.cuda.set_device(0)
    _, pops = build()
    return pops


def test_final_pass_only_no_incoming_spikes(pops):
    """No spike consumption at all (n_spikes=0) -- just update/send/flush.

    potential -= 1, then checked against thresholds 10/20/30:
      node 0: 35 -> 34, > 30            -> fires at t + 20, resets to 0
      node 2: 25 -> 24, > 20 (not > 30) -> fires at t + 10, resets to 0
      everyone else: 0 -> -1, no fire
    """
    N, n_blocks = 10, 3
    B = pops["B"]
    fn = _load_update_kernel(B)
    potential0 = [35, 0, 25, 0, 0, 0, 0, 0, 0, 0]

    final_potential, spikes = _launch(fn, B.incoming[0], N, n_blocks, potential0, t=3.0)

    assert spikes == [(0, 23.0), (2, 13.0)]
    expected_potential = [0.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    assert final_potential == expected_potential


def test_consumes_real_incoming_spike_through_link_walk(pops):
    """One real incoming spike (not hand-seeded potential) from A, all-to-all
    into B, so every B neuron's receive() runs for real through the
    compiled CSR/link-tiling walk and adds +10 -- this is the path Triton
    never ran until now (n_spikes was 0 in the previous smoke test).

    send's threshold checks are lowered into the SAME link-walk dispatch
    block as receive (see test_coverage.py's test_spike_lowers_to_local_...:
    a node's emission sites all land in one link loop), so they run again
    right after this one spike's receive -- using the spike's own time
    (4.5) for the delay math -- *before* update's -1 ever applies. update
    then runs once more, unconditionally, in the final pass, followed by
    send's checks a second time against the post-update values.

    potential before this call: [0, 0, 11, 21, 31, 0, 0, 0, 0, 0]
    +10 (one incoming spike, all-to-all -> hits every neuron):
                                 [10, 10, 21, 31, 41, 10, 10, 10, 10, 10]
    send check #1, right after receive, time = 4.5 (the spike's own time):
      node 2: 21 -> > 20 (not > 30) -> fires at 4.5 + 10 = 14.5, resets to 0
      node 3: 31 -> > 30            -> fires at 4.5 + 20 = 24.5, resets to 0
      node 4: 41 -> > 30            -> fires at 4.5 + 20 = 24.5, resets to 0
      everyone else: 10, no fire
    -1 (update, final pass), applied to ALL neurons regardless of the reset above:
                                 [9,  9,  -1, -1, -1,  9,  9,  9,  9,  9]
    send check #2, final pass, time = t (5.0): nothing left crosses 10 -> no new spikes.
    """
    N, n_blocks = 10, 3
    B = pops["B"]
    fn = _load_update_kernel(B)
    potential0 = [0, 0, 11, 21, 31, 0, 0, 0, 0, 0]

    final_potential, spikes = _launch(
        fn, B.incoming[0], N, n_blocks, potential0, t=5.0,
        spike_time=[4.5], spike_src=[0], spike_conn=[0],
    )

    assert spikes == [(2, 14.5), (3, 24.5), (4, 24.5)]
    expected_potential = [9.0, 9.0, -1.0, -1.0, -1.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    assert final_potential == expected_potential
