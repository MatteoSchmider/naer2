"""
Emission histories (naer2/history.py, SPIKES.md): a non-uniform producer writes each program's spikes into its own
region of the run, ordered by (e, node) at the run's end -- the same order whatever the warps' timing; a uniform
one keeps one flat list.
"""

import pytest
import torch

import naer2 as n2
from naer2.runtime import PopulationRuntime
from naer2.transpiler.templates.fused_template import FusedKernelTemplate

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm)")


@n2.node(receive=None)
class _late_first:
    """Neuron j fires `left` times from next_spike, 0.003 apart -- times falling with j, the loop emitting
    lane sets (each run's spikes far from (e, node) order as emitted)."""

    def __init__(self):
        self.left: float = 0.0
        self.next_spike: float = 0.0

    def update(self, time: n2.time):
        while self.left > 0.5:
            n2.spike(self.next_spike)
            self.next_spike += 0.003
            self.left -= 1.0

    def send(self, time: n2.time):
        pass


@n2.node(receive=None)
class _bare:
    def __init__(self):
        self.fire: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        if self.fire > 0.5:
            n2.spike()
            self.fire = 0.0


def _run(node, n, group_size=128, **state):
    pop = node() * n
    gk = FusedKernelTemplate(pop).build()
    defaults = {a: v for a, v in pop.instance.__dict__.items() if a in pop.dynamic_attrs}
    rt = PopulationRuntime(gk, n, group_size=group_size, device="cuda:0", state={**defaults, **state},
                           history_cap=2 * n)
    rt.run(1.0, 1.1, round_id=0)
    rt.check()
    return rt


def test_a_non_uniform_producer_s_regions_come_ordered():
    n = 500
    left = (torch.arange(n) % 3).float()
    start = 1.09 - torch.arange(n) * 1e-4  # (later neurons earlier)
    rt = _run(_late_first, n, left=left, next_spike=start.clone())
    h = rt.history
    assert h.regions and h.G == 4 and h.R == 250  # (history_cap 1000 over 4 programs)
    spikes = rt.emitted()
    assert len(spikes) == int(left.sum())
    # program by program (its neurons 128 g .. 128 g + 127): exactly its spikes, in (e, node) order
    pos = 0
    for g in range(h.G):
        k = int(h.counts[1, g])
        region = spikes[pos:pos + k]
        pos += k
        assert all(g * 128 <= j < (g + 1) * 128 for _, j in region)
        assert region == sorted(region)
    assert pos == len(spikes)


def test_a_non_uniform_producer_s_history_is_the_same_every_time():
    n = 500
    left = (torch.arange(n) % 3).float()
    start = 1.09 - torch.arange(n) * 1e-4
    runs = [_run(_late_first, n, left=left, next_spike=start.clone()).emitted() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_a_uniform_producer_keeps_one_flat_list():
    rt = _run(_bare, 300, fire=(torch.arange(300) % 2).float())
    assert not rt.history.regions and rt.history.G == 1
    assert sorted(j for _, j in rt.emitted()) == list(range(1, 300, 2))


def test_a_slot_s_old_run_stays_whole_until_its_next_run():
    """A run clears the next slot's header for its successor, but not the old run's region counts there: a
    consumer K - 1 runs behind still reads them mid-pull (history.py; each program clears its own count at its
    own run's start -- fused.py _region_reset)."""
    n = 300
    pop = _late_first() * n
    gk = FusedKernelTemplate(pop).build()
    rt = PopulationRuntime(gk, n, group_size=128, device="cuda:0", depth=3,
                           state={"left": torch.ones(n), "next_spike": torch.full((n,), 1.0)})
    for r in range(3):  # runs 1..3: slots 1, 2, 0; only run 1 fires
        rt.run(1.0 + 0.1 * r, 1.1 + 0.1 * r, round_id=r)
        if r == 0:
            rt.state["left"].fill_(0.0)
    torch.cuda.synchronize()
    h = rt.history
    assert int(h.counts[1].sum()) == n  # run 1's regions, untouched by run 3's close (slot 1 is its next)
    rt.run(1.3, 1.4, round_id=3)  # run 4 writes slot 1: its programs clear their counts first
    torch.cuda.synchronize()
    assert int(h.counts[1].sum()) == 0
