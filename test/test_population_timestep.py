"""
Population.timestep / Network._compute_effective_periods /
Network._sub_dt / Network._validate_schedulable -- the explicit,
user-settable per-population re-check cadence, and how it combines with
autonomous dynamics and incoming delay across the whole network (see
TODO.md).

Covers the gap found while reasoning about PD14: a population driven
entirely by send()'s own clock (no incoming connections, no update()
dynamics -- e.g. a Poisson background generator) was never due under the
old has_autonomous_dynamics/_min_incoming_delay logic alone, since neither
applies to it. `timestep` is the user's explicit way to say "check this
population on its own clock."

Also covers the companion gap: a connection with NO delay set at all used
to contribute nothing (an undelayed connection should mean "tick whenever
my source does," not "this connection doesn't exist for scheduling
purposes") -- see _compute_effective_periods' recursive resolution.

Pure scheduling-logic tests -- no GPU needed, no kernel launch (never calls
_execute_update_kernel). See test_mixed_scheduling.py for the real,
end-to-end, GPU-required version of all of this (including the retention
fix for a slow-ticking consumer).
"""

import math

import torch

from test.networks.counter_net import build as build_counter
from test.networks.PD14.poisson import PD14_poisson


def test_autonomous_population_defaults_to_always_due():
    # unchanged existing behavior: has_autonomous_dynamics with no explicit
    # timestep set means period 0.0, i.e. every sub-step, regardless of t.
    network, pops = build_counter()
    A = pops["A"]
    assert A.timestep is None
    periods = network._compute_effective_periods()
    assert periods[A] == 0.0
    assert A._is_due(0.0001, periods[A])
    assert A._is_due(1e9, periods[A])


def test_explicit_timestep_overrides_autonomous_default_to_a_coarser_cadence():
    network, pops = build_counter()
    A = pops["A"]
    A.timestep = 5.0
    A.last_update = 0.0
    periods = network._compute_effective_periods()
    assert periods[A] == 5.0
    assert not A._is_due(4.9, periods[A])
    assert A._is_due(5.0, periods[A])


def test_explicit_timestep_is_the_only_way_a_source_only_population_becomes_due():
    # PD14_poisson: no incoming connections, update() is a no-op -- the
    # exact real case that motivated this (see TODO.md). Without an
    # explicit timestep, has_autonomous_dynamics is False and
    # _min_incoming_delay() is inf, so this population could never be due.
    T = PD14_poisson(0.12, 0.0, 1000.0) * 10
    assert not T.has_autonomous_dynamics
    assert T._min_incoming_delay() == math.inf
    assert not T._is_due(1e9, math.inf), "no timestep set -- must never be due"

    T.timestep = 2.0
    T.last_update = 0.0
    assert not T._is_due(1.9, T.timestep)
    assert T._is_due(2.0, T.timestep)


def test_network_sub_dt_respects_explicit_timestep_over_t_step():
    network, pops = build_counter()
    A, B = pops["A"], pops["B"]
    A.timestep = 0.3
    periods = network._compute_effective_periods()
    assert network._sub_dt(1.0, periods) == 0.3


def test_validate_schedulable_raises_for_a_never_due_population():
    import pytest

    network, _ = build_counter()
    T = PD14_poisson(0.12, 0.0, 1000.0) * 10
    T.network = network
    network.populations.add(T)
    periods = network._compute_effective_periods()
    assert periods[T] == math.inf
    with pytest.raises(ValueError, match="never be due"):
        network._validate_schedulable(periods)


def test_validate_schedulable_passes_once_timestep_is_set():
    network, _ = build_counter()
    T = PD14_poisson(0.12, 0.0, 1000.0) * 10
    T.network = network
    network.populations.add(T)
    T.timestep = 1.0
    periods = network._compute_effective_periods()
    network._validate_schedulable(periods)  # must not raise


# ------------------------------------------------------------------ no-delay inheritance
#
# Each of these needs a genuinely event-driven TARGET (has_autonomous_dynamics
# False), since an autonomous population's period is fixed at 0.0 regardless
# of its incoming connections -- counter (used above) is always autonomous,
# so it can't exercise the incoming-delay branch of _compute_effective_periods
# at all. ed_relay (update() is a no-op) is the node for that; the source can
# be anything.


def _ed_pair():
    """A (autonomous source, counter) -> Y (event-driven target, ed_relay),
    all-to-all, no delay set by default."""
    from test.edges import static_synapse
    from test.nodes import counter, ed_relay
    from test.patterns import all_to_all

    A = counter() * 3
    Y = ed_relay() * 3
    A.potential = torch.zeros(3)
    conn = A >> static_synapse(0.0) @ all_to_all >> Y
    return A.network, A, Y, conn


def test_connection_with_an_explicit_delay_sets_the_target_s_period():
    network, A, Y, conn = _ed_pair()
    conn.delay = 2.5
    periods = network._compute_effective_periods()
    assert periods[A] == 0.0  # A is autonomous, unaffected by any of this
    assert periods[Y] == 2.5


def test_connection_with_no_delay_inherits_the_source_s_own_period():
    network, A, Y, conn = _ed_pair()
    assert getattr(conn.fabric, "delay", None) is None
    A.timestep = 0.4  # give A a real, non-default period to inherit
    periods = network._compute_effective_periods()
    assert periods[A] == 0.4
    assert periods[Y] == 0.4, "undelayed connection must sync to its source's own period"


def test_no_delay_from_an_autonomous_source_gives_the_target_period_zero():
    network, A, Y, conn = _ed_pair()
    # A has no explicit timestep -> autonomous default period 0.0 -> Y
    # (undelayed) inherits that, i.e. also ticks every sub-step.
    periods = network._compute_effective_periods()
    assert periods[A] == 0.0
    assert periods[Y] == 0.0


def test_mixed_incoming_connections_take_the_minimum_across_them():
    from test.edges import static_synapse
    from test.nodes import counter
    from test.patterns import all_to_all

    _, A, Y, conn = _ed_pair()
    conn.delay = 5.0  # A -> Y now has a coarse explicit delay

    C = counter() * 5
    C.potential = torch.zeros(5)
    C.timestep = 0.2  # a second, finer source into Y
    C >> static_synapse(0.0) @ all_to_all >> Y
    # Connection.target() merges networks via `self._source.network += ...`,
    # i.e. into C's own Network object -- re-fetch after all connections
    # are made rather than trusting a reference captured earlier.
    network = Y.network

    periods = network._compute_effective_periods()
    assert periods[Y] == 0.2, "Y must track whichever incoming connection is finest"


def test_fixed_point_iteration_terminates_for_recurrent_no_delay_connectivity():
    # X <-> Y, both event-driven (no autonomous dynamics), both undelayed,
    # except X has an explicit anchor timestep -- must not infinite-loop
    # on the cycle, and both ends should settle on the anchor's period.
    from test.edges import static_synapse
    from test.nodes import ed_relay
    from test.patterns import all_to_all

    X = ed_relay() * 3
    Y = ed_relay() * 3
    X.timestep = 0.25
    X >> static_synapse(0.0) @ all_to_all >> Y
    Y >> static_synapse(0.0) @ all_to_all >> X
    network = X.network

    periods = network._compute_effective_periods()
    assert periods[X] == 0.25
    assert periods[Y] == 0.25
