"""
Population.timestep / Network._sub_dt / Network._validate_schedulable --
the explicit, user-settable per-population re-check cadence (see TODO.md).

Covers the gap found while reasoning about PD14: a population driven
entirely by send()'s own clock (no incoming connections, no update()
dynamics -- e.g. a Poisson background generator) was never due under the
old has_autonomous_dynamics/_min_incoming_delay logic alone, since neither
applies to it. `timestep` is the user's explicit way to say "check this
population on its own clock."

Pure scheduling-logic tests -- no GPU needed, no kernel launch (never calls
_execute_update_kernel).
"""

import math

import torch

from test.networks.counter_net import build as build_counter
from test.networks.PD14.poisson import PD14_poisson


def test_autonomous_population_defaults_to_always_due():
    # unchanged existing behavior: has_autonomous_dynamics with no explicit
    # timestep set means "every sub-step", regardless of t.
    _, pops = build_counter()
    A = pops["A"]
    assert A.timestep is None
    assert A._is_due(0.0001)
    assert A._is_due(1e9)


def test_explicit_timestep_overrides_autonomous_default_to_a_coarser_cadence():
    _, pops = build_counter()
    A = pops["A"]
    A.timestep = 5.0
    A.last_update = 0.0
    assert not A._is_due(4.9)
    assert A._is_due(5.0)


def test_explicit_timestep_is_the_only_way_a_source_only_population_becomes_due():
    # PD14_poisson: no incoming connections, update() is a no-op -- the
    # exact real case that motivated this (see TODO.md). Without an
    # explicit timestep, has_autonomous_dynamics is False and
    # _min_incoming_delay() is inf, so this population could never be due.
    T = PD14_poisson(0.12, 0.0, 1000.0) * 10
    assert not T.has_autonomous_dynamics
    assert T._min_incoming_delay() == math.inf
    assert not T._is_due(1e9), "no timestep set -- must never be due"

    T.timestep = 2.0
    T.last_update = 0.0
    assert not T._is_due(1.9)
    assert T._is_due(2.0)


def test_network_sub_dt_respects_explicit_timestep_over_t_step():
    network, pops = build_counter()
    A, B = pops["A"], pops["B"]
    A.timestep = 0.3
    assert network._sub_dt(1.0) == 0.3


def test_validate_schedulable_raises_for_a_never_due_population():
    import pytest

    network, _ = build_counter()
    T = PD14_poisson(0.12, 0.0, 1000.0) * 10
    T.network = network
    network.populations.add(T)
    with pytest.raises(ValueError, match="never be due"):
        network._validate_schedulable()


def test_validate_schedulable_passes_once_timestep_is_set():
    network, _ = build_counter()
    T = PD14_poisson(0.12, 0.0, 1000.0) * 10
    T.network = network
    network.populations.add(T)
    T.timestep = 1.0
    network._validate_schedulable()  # must not raise
