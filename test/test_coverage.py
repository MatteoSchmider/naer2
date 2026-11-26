"""
General compilation coverage smoke-test.

Builds a 4-population network exercising:
  - A -> B -> C chain:
      - flexible @ chaining order (port / mapping / pattern in any order)
      - fabric_map: edge.w -> real attr 'weight', on all connections
      - source_map: source.src_pot -> real attr 'potential', on the B->C connection
      - two differently-named receive ports ("positive" / "negative")
      - a static synapse (A->B) with a per-link weight tensor
      - a plastic synapse (B->C) whose forward hook mutates weight per link
  - A, B, and C all -> D, through the SAME receive port ("positive"), A->D
    and B->D through a static synapse, C->D through a plastic one
  - relay's spike sites: two in send (one with an explicit time), one in
    update, one (never firing) in pos_receive

The DSL side (fan-in, port routing, the maps) is checked directly. The
generated fused kernels (Population.fused_kernel): A's -- no incoming
connections, every spike site in update / send -- is checked on its source,
and B's, a spike inside delivery (D-3); C and D need what the template
doesn't generate yet (edge forward hooks; source attrs, D-10), so they are
one gap test, failing until those close (TODO.md).

No GPU: generating a kernel only builds its source and the gluon.jit
function.
"""

import pytest

from test.networks.coverage import build


@pytest.fixture(scope="module")
def pops():
    _, pops = build()
    return pops


@pytest.fixture(scope="module")
def src_A(pops):
    return pops["A"].fused_kernel().source


def test_population_fan_in_counts(pops):
    A, B, C, D = pops["A"], pops["B"], pops["C"], pops["D"]
    assert len(A.incoming) == 0
    assert len(B.incoming) == 1
    assert len(C.incoming) == 1
    assert len(D.incoming) == 3
    assert D.incoming[0]._source is A and D.incoming[1]._source is B and D.incoming[2]._source is C


def test_target_port_routing(pops):
    B, C, D = pops["B"], pops["C"], pops["D"]
    assert B.incoming[0]._target_port == "positive"
    assert C.incoming[0]._target_port == "negative"
    assert all(conn._target_port == "positive" for conn in D.incoming)


def test_fabric_map_edge_w_to_weight(pops):
    B, C, D = pops["B"], pops["C"], pops["D"]
    assert B.incoming[0]._fabric_map == {"weight": "w"}
    assert C.incoming[0]._fabric_map == {"weight": "w"}
    assert all(conn._fabric_map == {"weight": "w"} for conn in D.incoming)


def test_source_map_src_pot_to_potential(pops):
    B, C = pops["B"], pops["C"]
    assert B.incoming[0]._source_map is None
    assert C.incoming[0]._source_map == {"potential": "src_pot"}


def test_only_plastic_synapses_have_edge_state(pops):
    B, C, D = pops["B"], pops["C"], pops["D"]
    assert B.incoming[0].fabric.dynamic_attrs == {}
    assert set(C.incoming[0].fabric.dynamic_attrs) == {"weight"}
    assert [set(c.fabric.dynamic_attrs) for c in D.incoming] == [set(), set(), {"weight"}]


def test_a_population_without_connections_has_no_receive(pops, src_A):
    k = pops["A"].fused_kernel()
    assert k.attrs == ["potential"] and k.state_attrs == [] and k.edge_attr is None
    assert 'def _receive(state, hit, time, w, srcword, ectx, nodes):\n    """Nothing (no incoming connection carries one)."""' in src_A


def test_every_update_and_send_spike_site_is_emitted(src_A):
    # update's (never firing) site, then send's three: two in the far-past-threshold branch, one in the elif
    assert "_m0 = mask & (potential < -1000000.0)" in src_A, "update's site, rooted at the group mask"
    assert "_m0 = mask & (potential > 2 * 5)" in src_A
    assert "_m1 = ~_m0 & mask & (potential > 5)" in src_A, "the elif chains off the first branch's mask"
    assert [f"_fire{i} = " in src_A for i in range(4)] == [True] * 4
    assert src_A.count("_emit_spikes(") == 4
    assert src_A.index("_emit_spikes(") > src_A.index("gl.store(potential_ptr"), "emitted after the state"


def test_spike_times(pops, src_A):
    # bare n2.spike(): the window's start (D-4); n2.spike(time + 1.0): verbatim, so not uniform
    assert src_A.count("= t_start + gl.zeros([GROUP_SIZE], gl.float32, layout=LN)") == 3
    assert "_e2 = time + 1.0 + gl.zeros([GROUP_SIZE], gl.float32, layout=LN)" in src_A
    assert not pops["A"].fused_kernel().uniform_emission


def test_lowered_completely(src_A):
    assert "n2.spike" not in src_A and "tl." not in src_A and "self." not in src_A


def test_a_spike_inside_delivery_generates(pops):
    """B: pos_receive's n2.spike() (D-3) -- emitted inside delivery, straight into the history; every hook
    lowered, the edge map reversed."""
    k = pops["B"].fused_kernel()
    assert k.edge_attr == "weight" and not k.uniform_emission
    assert "_emit_delivered(" in k.source and "_emit_ctx(pid, n_groups, history, hist_cap, run_id, K)" in k.source
    for leftover in ("edge.w", "n2.spike", "math.", "tl."):
        assert leftover not in k.source


# ------------------------------------------------------------ gaps (fail until closed)


@pytest.mark.parametrize("label", ["C", "D"])
def test_gap_coverage_network_generates(pops, label):
    """TODO.md: C -- an edge forward hook, source.src_pot (D-10); D -- three
    connections, one with an edge forward hook. Every hook lowered, the maps
    reversed."""
    k = pops[label].fused_kernel()  # raises NotImplementedError today
    assert k.edge_attr == "weight"
    for leftover in ("edge.w", "source.src_pot", "n2.spike", "math.", "tl."):
        assert leftover not in k.source
