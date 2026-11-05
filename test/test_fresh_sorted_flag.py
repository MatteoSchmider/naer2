"""
Population.fresh_sorted -- whether compact() may skip sorting a population's
fresh output. Only bare n2.spike() (which always records the kernel's `t`)
guarantees one shared timestamp per launch; any explicit spike time, in any
hook, must turn the fast path off.
"""

from test.edges import static_synapse
from test.nodes import counter, delay_probe, lif_ed
from test.patterns import all_to_all


def test_only_bare_spikes_is_sorted():
    assert (lif_ed(-50.0, -65.0, 10.0, 2.0) * 4).fresh_sorted


def test_explicit_spike_time_in_send_is_not_sorted():
    assert not (delay_probe() * 4).fresh_sorted


def test_any_explicit_time_among_bare_spikes_is_not_sorted():
    # counter mixes bare n2.spike() with n2.spike(time + 10.0) etc.
    assert not (counter() * 4).fresh_sorted


def test_incoming_connection_with_plain_edge_keeps_it_sorted():
    A = lif_ed(-50.0, -65.0, 10.0, 2.0) * 4
    B = lif_ed(-50.0, -65.0, 10.0, 2.0) * 4
    A >> static_synapse(1.0) @ all_to_all >> B
    assert B.fresh_sorted
