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


def test_delay_exceeding_the_population_s_own_period_raises_not_implemented():
    """counter is autonomous with no explicit timestep -> period 0.0 ("every
    sub-step") -- any nonzero delay is an inter-timestep transmission delay
    by definition, which isn't implemented (only intra-timestep is)."""
    _, pops = build()
    B = pops["B"]
    B.incoming[0].fabric.delay = torch.rand(B.incoming[0].shape)

    with pytest.raises(NotImplementedError, match="inter-timestep"):
        B.compile()


def test_delay_within_the_population_s_own_timestep_compiles_for_real():
    """The in-scope case: every delay value stays below the target's own
    explicit timestep -> compiles to the real per-link-delay kernel instead
    of raising."""
    _, pops = build()
    B = pops["B"]
    B.timestep = 1.0
    B.incoming[0].fabric.delay = torch.rand(B.incoming[0].shape) * 0.9  # all < 1.0

    module = B.compile()
    assert module.body[0].name == "per_link_delay_update_kernel"


def test_edge_class_with_dynamic_attrs_raises_clearly_in_delay_mode():
    """A connection whose edge class has a real forward/backward-assigned
    per-link attr (plastic_synapse) can never actually mutate it once the
    population is in per-link-delay mode -- see
    per_link_delay_update_kernel.py. This must raise a clear, specific
    error naming the edge type, not fall through to the stub's generic
    NotImplementedError."""
    from test.edges import plastic_synapse
    from test.nodes import counter
    from test.patterns import all_to_all

    A = counter() * 5
    B = counter() * 5
    A.potential = torch.zeros(5)
    B.potential = torch.zeros(5)
    conn = A >> plastic_synapse(0.0, 0.1) @ all_to_all >> B
    conn.fabric.delay = torch.rand(conn.shape)

    with pytest.raises(ValueError, match="plastic_synapse"):
        B.compile()


def test_static_edge_with_a_tensor_override_is_fine_in_delay_mode():
    """A static edge (no forward/backward assignment) that merely has a
    tensor OVERRIDE set from outside (conn.fabric.weight = tensor(...),
    never written back by any hook) has no dynamic_attrs at all -- the
    check above must not flag it, since nothing here was ever claimed to
    change. Still hits the stub's NotImplementedError (the kernel itself
    isn't implemented yet), not the ValueError."""
    from test.edges import static_synapse
    from test.nodes import counter
    from test.patterns import all_to_all

    A = counter() * 5
    B = counter() * 5
    A.potential = torch.zeros(5)
    B.potential = torch.zeros(5)
    conn = A >> static_synapse(0.0) @ all_to_all >> B
    conn.fabric.weight = torch.rand(conn.shape)
    conn.fabric.delay = torch.rand(conn.shape)

    with pytest.raises(NotImplementedError):
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
