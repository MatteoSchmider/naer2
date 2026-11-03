"""
End-to-end, real-hardware test of mixed scheduling: clock-driven
(autonomous) and event-driven populations, with and without delay, sharing
one network and one Network.step() loop (see TODO.md). Covers three
behaviors together, in one hand-verified scenario:

  1. An event-driven population (no update() dynamics) with an explicit
     per-link delay only re-checks at that delay's cadence -- pre-existing
     behavior (Population._min_incoming_delay/Network._compute_effective_
     periods), confirmed here against a real, repeating source rather than
     reasoned about abstractly.
  2. An event-driven population with NO delay set on its incoming
     connection inherits its SOURCE's own cadence -- previously this
     connection contributed nothing (math.inf), which combined with no
     autonomous dynamics and no explicit timestep meant "never due," the
     same silent-failure shape as the PD14 T/poisson gap. Now it correctly
     syncs to whatever the source's own effective period is.
  3. Retention: a population that ticks slower than the network's sub_dt
     must not lose anything that became due while it was waiting --
     `Network.step()` used to unconditionally overwrite `_merged` every
     sub-step, discarding whatever a not-yet-due population hadn't
     consumed yet. Found by direct reproduction (a slow consumer only ever
     saw the LAST sub-step's contribution, never what arrived in between)
     and fixed via spike_gather.combine(), which accumulates instead of
     overwriting until the consuming population's own update kernel runs
     and clears it.

Network: `A` (clock_source, autonomous, fires all 3 of its neurons
unconditionally every call it's due for -- counter doesn't serve this,
since it only fires once, crosses threshold, resets to 0, and nothing
refills it) feeds two independent event-driven targets all-to-all:
`Y_delay` (explicit delay=3.0) and `Y_nodelay` (no delay at all).

Hand-derived trace (t_step=1.0 each call; A has no explicit timestep, so
its autonomous-default period is 0.0 -- always due):

  Y_nodelay inherits A's period (0.0) through its undelayed connection, so
  it's due every single step, same cadence as A -- but gather always runs
  BEFORE update in the same step() call, so a given step's own fresh spikes
  aren't visible to anyone until the NEXT step (see network.py's docstring).
  Each round A fires, all 3 of its neurons deliver to all 3 of Y_nodelay's
  neurons (+3 per neuron per round):
    step 1: nothing due yet -> count 0
    step 2..7: +3 each step -> 3, 6, 9, 12, 15, 18

  Y_delay's period is the connection's explicit delay (3.0), so it's only
  due at t=4 and t=7 (last_update jumps 1 -> 4 -> 7). Between ticks it
  still gathers/merges every step (phase 1/2 are unconditional for every
  population), accumulating via combine() rather than losing anything:
    by t=4: 3 rounds' worth due (steps 1,2,3) -> +9 -> count 9
    by t=7: 3 more rounds' worth (steps 4,5,6) -> +9 -> count 18
  Without the retention fix, Y_delay would have seen only the LAST
  sub-step's contribution each time it finally ticked (3, not 9) -- see
  test_retention_across_sub_steps_loses_nothing below, which pins the
  exact numbers the bug used to produce.

Skipped entirely when no CUDA/ROCm device is visible to torch (no CPU
reference for the compiled kernels, same as the rest of this tier).
"""

import pytest
import torch

from test.edges import static_synapse
from test.nodes import clock_source, ed_relay
from test.patterns import all_to_all

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a real GPU (CUDA/ROCm); no CPU reference for the compiled kernels",
)


@pytest.fixture
def net():
    torch.cuda.set_device(0)
    A = clock_source() * 3
    Y_delay = ed_relay() * 3
    Y_nodelay = ed_relay() * 3
    A.tick = torch.zeros(3)
    Y_delay.count = torch.zeros(3)
    Y_nodelay.count = torch.zeros(3)

    conn_delay = A >> static_synapse(0.0) @ all_to_all >> Y_delay
    conn_delay.delay = 3.0
    A >> static_synapse(0.0) @ all_to_all >> Y_nodelay

    return A.network, A, Y_delay, Y_nodelay


def test_event_driven_with_delay_ticks_at_its_own_delay_cadence(net):
    network, A, Y_delay, Y_nodelay = net

    for _ in range(7):
        network.step(1.0)

    assert Y_delay.last_update == 7.0
    assert Y_delay.count.tolist() == [18.0, 18.0, 18.0]


def test_event_driven_with_no_delay_syncs_to_its_source_s_own_cadence(net):
    network, A, Y_delay, Y_nodelay = net

    expected = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
    for i in range(7):
        network.step(1.0)
        assert Y_nodelay.last_update == network.t, "no delay -> due every step, same as A"
        assert Y_nodelay.count.tolist() == [expected[i]] * 3


def test_retention_across_sub_steps_loses_nothing(net):
    """The actual regression test for the bug: without combine()-based
    retention, Y_delay only ever sees whatever was due on the single
    sub-step it happens to tick on (3 per tick), silently dropping the
    other 2 rounds' worth every time -- confirmed by direct reproduction
    before this fix landed. With retention, every round it was waiting
    through is preserved."""
    network, A, Y_delay, Y_nodelay = net

    for i in range(1, 8):
        network.step(1.0)
        if i in (4, 7):
            assert Y_delay.last_update == float(i)
    assert Y_delay.count.tolist() == [18.0, 18.0, 18.0], (
        "3 rounds x 3 deliveries x 2 ticks = 18; a dropped round would show 9 (one "
        "tick's worth only) or 15 (losing exactly one round)"
    )
