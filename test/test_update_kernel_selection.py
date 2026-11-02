"""
Population.uses_per_link_delay_kernel / compile()'s variant selection
(naer2/population.py, naer2/transpiler/templates/no_delay_update_kernel.py,
naer2/transpiler/templates/per_link_delay_update_kernel.py -- see TODO.md).

Pure AST/metadata work -- no GPU needed, unlike the kernels themselves.
Reuses test/networks/counter_net (A -> B, all-to-all) purely as a stand-in;
its hooks are irrelevant here, only the presence/shape of conn.fabric.delay
matters.
"""

import pytest
import torch

from test.networks.counter_net import build


def test_no_delay_by_default():
    _, pops = build()
    B = pops["B"]
    assert not B.uses_per_link_delay_kernel
    module = B.compile()
    assert module.body[0].name == "no_delay_update_kernel"


def test_uniform_scalar_delay_stays_on_the_no_delay_kernel():
    _, pops = build()
    B = pops["B"]
    B.incoming[0].fabric.delay = 1.5
    assert not B.uses_per_link_delay_kernel
    module = B.compile()
    assert module.body[0].name == "no_delay_update_kernel"


def test_a_real_per_link_delay_tensor_switches_the_whole_population():
    _, pops = build()
    B = pops["B"]
    B.incoming[0].fabric.delay = torch.rand(B.incoming[0].shape)
    assert B.uses_per_link_delay_kernel

    with pytest.raises(NotImplementedError, match="per-link-delay"):
        B.compile()


def test_one_delayed_incoming_connection_switches_the_whole_population():
    """Population-wide, not per-connection: even a population with a SECOND,
    non-delayed incoming connection switches entirely once ANY incoming
    connection is per-link-delayed -- see per_link_delay_update_kernel.py's
    module docstring for why (edge callbacks become disallowed everywhere on
    that population, not just on the delayed connection)."""
    from test.edges import static_synapse
    from test.nodes import counter
    from test.patterns import all_to_all

    _, pops = build()
    A, B = pops["A"], pops["B"]
    C = counter() * 5
    C.potential = torch.zeros(5)
    C >> static_synapse(0.0) @ all_to_all >> B

    assert B.incoming[0]._source is A and B.incoming[1]._source is C
    B.incoming[1].fabric.delay = torch.rand(B.incoming[1].shape)

    assert B.uses_per_link_delay_kernel
    with pytest.raises(NotImplementedError):
        B.compile()
