"""
A connection with no edge class at all (`A >> pattern >> B`, no `@n2.edge`
instance anywhere in the chain) -- naer2/fabric.py's Fabric(instance=None)
and naer2/connection.py's ConnectionBuilder._ensure_fabric/Connection.
__setattr__.

Before this, `A >> pattern >> B` crashed with an unhelpful AttributeError
('NoneType' object has no attribute 'topology') -- an edge class was
mandatory even for the common "just a per-link tensor, no learning rule"
case, which forced trivial classes like static_synapse/PD14_synapse to
exist purely to satisfy the DSL. Now a bare connection works, and
conn.<name> = tensor is the validated way to populate a per-link tensor on
one: it only stores the tensor if some hook at the connection's target port
actually reads edge.<name> (after reversing fabric_map) -- the same
resolution Population.compile() itself does -- since with no edge class
there's no other check (a real edge class's declared dynamic_attrs) to
catch a typo or an unused tensor. conn.fabric.<name> = tensor (unchecked)
still works exactly as before, on a bare connection or a real one alike.

delay is exempt from the "some hook must read edge.<name>" check: it's read
directly off fabric by the framework itself (Population.
_min_incoming_delay, uses_per_link_delay_kernel), never via edge.delay
inside hook code.

Pure AST/metadata work -- no GPU needed, like test_coverage.py and
test_update_kernel_selection.py.
"""

import pytest
import torch

from test.nodes import counter, lif_ed
from test.patterns import all_to_all


def _counter_pair(n=5):
    A = counter() * n
    B = counter() * n
    A.potential = torch.zeros(n)
    B.potential = torch.zeros(n)
    return A, B


def _lif_pair(n=5):
    A = lif_ed(1.0, 0.0, 10.0, 2.0) * n
    B = lif_ed(1.0, 0.0, 10.0, 2.0) * n
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
    conn.delay = torch.rand(conn.shape)
    assert conn.fabric.delay.shape == conn.shape
    assert B.uses_per_link_delay_kernel


def test_setting_an_unused_attr_raises():
    # counter.receive never reads edge.weight -- storing it would be dead.
    A, B = _counter_pair()
    conn = A >> all_to_all >> B
    with pytest.raises(ValueError, match="weight"):
        conn.weight = torch.rand(conn.shape)


def test_setting_an_attr_the_receive_hook_actually_reads_wires_into_the_kernel():
    A, B = _lif_pair()
    conn = A >> all_to_all >> B
    conn.weight = torch.rand(conn.shape)

    module = B.compile()
    arg_names = {a.arg for a in module.body[0].args.args}
    assert "conn0_weight_ptr" in arg_names


def test_fabric_attr_assignment_still_works_unchecked_on_a_bare_connection():
    A, B = _counter_pair()
    conn = A >> all_to_all >> B
    conn.fabric.anything_at_all = torch.rand(conn.shape)
    assert conn.fabric.anything_at_all.shape == conn.shape
