"""
Captured windows (Network.run_graph, naer2/graph.py) whose runs stop: a ring too small for a run inside a
replayed window -- the window's runs are pending as one per runtime and judged together (runtime._WindowPart),
its journal entry one for the whole window (runtime._Window): the stop found when the window is judged, every
run not known complete rerun through its population's own kernel, in launch order, the ring grown -- and the
result the one with rings large enough from the start (the links received exactly, the node state up to float
rounding: test_fusion._same_state).
"""

import pytest
import torch

from test.networks.mini_ei import build
from test.test_fusion import _same_state

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernels"
)


def _run(small: bool, fuse: bool):
    torch.manual_seed(0)
    network, pops = build()
    for pop in pops.values():
        pop.count_received = True
        pop.potential = torch.normal(-52.0, 4.0, pop.potential.shape)
        for conn in pop.incoming:
            conn.delay = 0.5
    if small:  # E's arena at its minimum: 4-record blocks, nothing for spikes in flight
        pops["E"].ring_block, pops["E"].ring_in_flight = 4, 0.0
    network.step(0.1)
    network.fuse_rounds = fuse
    for _ in range(4):
        network.run_graph(max_launches=10)
    for pop in pops.values():
        pop.check()
    state = {(n, a): t.detach().cpu().clone() for n, p in pops.items() for a, t in p.runtime.state.items()}
    received = {n: p.runtime.n_received.detach().cpu().clone() for n, p in pops.items()}
    return state, received, pops["E"].runtime


@pytest.mark.parametrize("fuse", [False, True], ids=["unfused", "fused"])
def test_gap_a_stop_inside_a_captured_window_is_rerun(fuse):
    """GAP (TODO.md, resizing in planned and captured windows): fails before window bookkeeping and after --
    the rerun ends in a consumer's history overrun, and the next capture fails; windows need rings that fit
    (PD14: after its warm-up) until stops inside them are handled."""
    state, received, _ = _run(small=False, fuse=fuse)
    state_s, received_s, rt = _run(small=True, fuse=fuse)
    assert rt.stops and rt.growths, "E's ring stopped a run and grew"
    assert sum(int(r.sum()) for r in received.values()) > 0
    for n in received:
        assert torch.equal(received[n], received_s[n]), f"{n}: received counts differ"
    _same_state(state, state_s)
