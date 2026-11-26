"""
while loops in hooks (IfTransformer: a loop for as long as any lane's condition holds, each lane in it until its
own fails): the lowered source, what isn't lowered, and on the GPU a loop that fires any number of times a run
(its spike site emitted where it stands) and random.* drawing anew every iteration (RandomLoweringTransformer:
the loop's counter in the key).
"""

import math
import random

import pytest
import torch

import naer2 as n2
from naer2.runtime import PopulationRuntime
from naer2.transpiler.templates.fused_template import FusedKernelTemplate


@n2.node(receive=None)
class _burst:
    """Fires `left` spikes a run, `gap` apart from next_spike (a loop with a spike inside)."""

    def __init__(self, gap: float = 0.01):
        self.gap = gap
        self.left: float = 0.0
        self.next_spike: float = 0.0

    def update(self, time: n2.time):
        while self.left > 0.5:
            n2.spike(self.next_spike)
            self.next_spike += self.gap
            self.left -= 1.0

    def send(self, time: n2.time):
        pass


@n2.node(receive=None)
class _draws:
    """Adds up three draws (random.random() inside a loop)."""

    def __init__(self):
        self.i: float = 0.0
        self.total: float = 0.0

    def update(self, time: n2.time):
        while self.i < 3.0:
            self.total += random.random()
            self.i += 1.0

    def send(self, time: n2.time):
        pass


@n2.node(receive=None)
class _breaking:
    def __init__(self):
        self.x: float = 0.0

    def update(self, time: n2.time):
        while self.x < 3.0:
            self.x += 1.0
            if self.x > 1.5:
                break

    def send(self, time: n2.time):
        pass


def test_a_while_loop_lowers_to_a_masked_loop():
    k = FusedKernelTemplate(_burst() * 10).build()
    src = k.source
    assert "_it0 = gl.to_tensor(0)" in src
    assert "while _any_lane(_m0):" in src
    assert "_m0 = _m0 & (left > 0.5)" in src
    # the spike site inside: emitted where it stands (any number of times a run), not flushed after send
    assert "_emit_spikes(_m0, next_spike + gl.zeros([GROUP_SIZE], gl.float32, layout=LN)" in src
    assert "_fire0" not in src
    assert not k.uniform_emission
    draws = FusedKernelTemplate(_draws() * 10).build().source
    assert "_it0.to(gl.int64) *" in draws  # (the loop's iteration in the draw's key)


def test_what_a_loop_cannot_hold():
    with pytest.raises(NotImplementedError, match="break in a while loop"):
        FusedKernelTemplate(_breaking() * 10).build()


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm)")


def _run_once(pop, **state):
    gk = FusedKernelTemplate(pop).build()
    defaults = {a: v for a, v in pop.instance.__dict__.items() if a in pop.dynamic_attrs}
    rt = PopulationRuntime(gk, len(pop.shape), group_size=256, device="cuda:0", state={**defaults, **state},
                           history_cap=4 * len(pop.shape))
    rt.run(1.0, 1.1, round_id=0)
    rt.check()
    return rt


@gpu
def test_a_loop_fires_any_number_of_times_a_run():
    """Every neuron j fires j % 4 times, 0.01 apart from 1.0 + j / 1000: each spike at its own time."""
    n = 600
    left = (torch.arange(n) % 4).float()
    start = 1.0 + torch.arange(n) / 1000.0
    rt = _run_once(_burst() * n, left=left, next_spike=start.clone())
    want = sorted((float(torch.tensor(float(start[j]) + 0.01 * i, dtype=torch.float32)), j)
                  for j in range(n) for i in range(int(left[j])))
    got = sorted(rt.emitted())
    assert len(got) == len(want) == int(left.sum())
    assert all(g[1] == w[1] and math.isclose(g[0], w[0], abs_tol=1e-5) for g, w in zip(got, want))
    assert torch.all(rt.state["left"] == 0)


@gpu
def test_a_loop_draws_anew_every_iteration():
    """Three draws per neuron, summed: independent draws vary as 3 / 12 = 0.25 (one draw three times: 0.75)."""
    n = 4096
    rt = _run_once(_draws() * n)
    total = rt.state["total"].cpu()
    assert torch.all(rt.state["i"] == 3.0)
    assert abs(float(total.mean()) - 1.5) < 0.05
    assert 0.2 < float(total.var()) < 0.3, float(total.var())
