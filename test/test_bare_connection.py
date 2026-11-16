"""
A connection with no edge class at all (`A >> pattern >> B`, no `@n2.edge`
instance anywhere in the chain) -- naer2/fabric.py's Fabric(instance=None)
and naer2/connection.py's ConnectionBuilder._ensure_fabric/Connection.
__setattr__.

conn.<name> = tensor is the validated way to populate a per-link tensor on
one: it only stores the tensor if the hook at the connection's target port
actually reads edge.<name> (after reversing fabric_map), since with no edge
class there's no other check (a real edge class's declared dynamic_attrs)
to catch a typo or an unused tensor. conn.fabric.<name> = tensor
(unchecked) works on a bare connection or a real one alike.

delay is exempt from the "some hook must read edge.<name>" check: it's read
directly off fabric by the framework itself (Population._min_delay,
Fabric.simulation's delay buckets), never via edge.delay inside hook code.

No GPU: the generated kernel is only built, not run.
"""

import pytest
import torch

from test.networks.PD14.neuron import PD14_neuron
from test.nodes import counter
from test.patterns import all_to_all


def _counter_pair(n=5):
    A = counter() * n
    B = counter() * n
    A.potential = torch.zeros(n)
    B.potential = torch.zeros(n)
    return A, B


def test_bare_connection_builds_without_an_edge_class():
    A, B = _counter_pair()
    conn = A >> all_to_all >> B
    assert conn.shape == (25,)
    assert conn.fabric.dynamic_attrs == {}
    assert conn.fabric.edge == {"forward": None, "backward": None}


def test_delay_is_exempt_from_the_hook_usage_check():
    A, B = _counter_pair()
    conn = A >> all_to_all >> B
    delay = torch.rand(conn.shape) + 0.1
    conn.delay = delay
    assert torch.equal(conn.fabric.delay, delay)
    assert B._min_delay(conn) == pytest.approx(float(delay.min()))


def test_the_delay_shorthand_sets_the_fabric_s_delay():
    A, B = _counter_pair()
    conn = (A >> all_to_all >> B) | 0.5
    assert conn.fabric.delay == 0.5
    conn.delay(0.7)
    assert conn.fabric.delay == 0.7


def test_setting_an_unused_attr_raises():
    # counter.receive never reads edge.weight -- storing it would be dead.
    A, B = _counter_pair()
    conn = A >> all_to_all >> B
    with pytest.raises(ValueError, match="weight"):
        conn.weight = torch.rand(conn.shape)


def test_setting_an_attr_the_receive_hook_actually_reads_wires_into_the_kernel():
    A, B = PD14_neuron(K_cortical=10) * 5, PD14_neuron(K_cortical=10) * 5
    conn = A >> all_to_all >> B
    conn.weight = torch.rand(conn.shape)
    kernel = B.fused_kernel()
    assert kernel.edge_attr == "weight", "the per-link weight is the attr the kernel stages per link"
    assert "edge_weight = w" in kernel.source


def test_fabric_attr_assignment_still_works_unchecked_on_a_bare_connection():
    A, B = _counter_pair()
    conn = A >> all_to_all >> B
    conn.fabric.anything_at_all = torch.rand(conn.shape)
    assert conn.fabric.anything_at_all.shape == conn.shape
