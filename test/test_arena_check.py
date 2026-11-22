"""
The device-side arena check (PopulationRuntime.arena_check, fused.py: ARENA_CHECK): every block a pull pops and
delivery frees against a held count per (group, block) -- silent through stops, reruns and resizes (the
counts rebuilt from the tables), and the first violation reported once the arena breaks (a held block on a
free stack: the next pop takes it twice).
"""

import pytest
import torch

from test.networks.mini_ei import build

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernels"
)


def _net():
    torch.manual_seed(0)
    network, pops = build()
    for pop in pops.values():
        pop.potential = torch.normal(-52.0, 4.0, pop.potential.shape)
        for conn in pop.incoming:
            conn.delay = 0.5
    pops["E"].ring_block, pops["E"].ring_in_flight = 4, 0.0  # (E's arena at its minimum: it stops and grows)
    network.step(0.1)
    for pop in pops.values():
        pop.runtime.arena_check = True
    return network, pops


def test_the_arena_check_is_silent_through_stops_and_resizes():
    network, pops = _net()
    network.step(3.0)
    for pop in pops.values():
        pop.check()
    rt = pops["E"].runtime
    assert rt.stops and rt.growths, "E stopped a run and its ring grew"
    for name, pop in pops.items():
        assert pop.runtime.arena_errors() is None, f"{name}: {pop.runtime.arena_errors()}"


def test_a_held_block_on_a_free_stack_is_reported():
    network, pops = _net()  # (right after the first round: E's early runs grow past their reserves -- pops)
    rt = pops["E"].runtime
    rt.settle()
    ring = rt.ring
    held = {}
    for g in range(ring.n_groups):  # every block a group could pop: one it holds (its first cell's first)
        held[g] = int(ring.table[g * ring.n_slots, 0])
        ring.stack[g, :int(ring.top[g])] = held[g]
    with pytest.raises(RuntimeError, match="arena check found a violation"):  # (raised when the run is judged)
        network.step(1.0)
    err = rt.arena_errors()
    assert err is not None and err["kind"] == "pop" and err["block"] == held[err["group"]], err
