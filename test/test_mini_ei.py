"""
Mini E/I compilation smoke-test.

Builds a 2-population E/I circuit (100 E + 25 I neurons, 4 connections) and
compiles the merged update-receive-send kernel for both populations.

Expected kernel structure:
  - ONE outer for-loop over the unified sorted spike buffer
  - Inner tl.static_range, guarded by blk_mask[j], containing:
      * time / spike_src_j / spike_conn_j loaded as scalars for this spike slot
      * An if/elif dispatch on spike_conn_j (warp-uniform — same value for
        every lane in this program, so this branches with no divergence),
        one branch per incoming connection, each with its own scalar CSR
        range lookup using that connection's own directly-named pointers:
          if spike_conn_j == 0:
            link_start = tl.load(conn0_src_offsets_ptr + _row + spike_src_j)
            link_end   = tl.load(conn0_src_offsets_ptr + _row + spike_src_j + 1)
            for link_tile in range(link_start, link_end, LINK_BLOCK):
              <<edge forward on link lanes; densify links -> neuron lanes>>
              <<receive + send on neuron lanes, masked by is_target>>
          elif spike_conn_j == 1: ...
  - Final propagation block (time = t) after the spike loop
  - edge.weight replaced by locally loaded scalar weight
  - current = current + weight * is_target  (ReceiveMaskTransformer)
  - Signature: spike_src_ptr (not spike_tgt_ptr), conn{i}_src_offsets_ptr,
    conn{i}_target_indices_ptr

This is an integration test: it goes through the real node/edge/pattern/
connection chain and compiles real kernels, then asserts on their structure.
It does not invoke the real Triton compiler or run anything — that's a later,
separate system-test concern (see TODO.md); correctness here is checked by
inspecting the generated AST.
"""

import ast

import pytest

from test.networks.mini_ei import build


@pytest.fixture(scope="module")
def pops():
    _, pops = build()
    return pops


@pytest.fixture(scope="module")
def E_module(pops):
    return pops["E"].compile()


@pytest.fixture(scope="module")
def I_module(pops):
    return pops["I"].compile()


@pytest.fixture(scope="module")
def func_E(E_module):
    return E_module.body[0]


@pytest.fixture(scope="module")
def src_E(E_module):
    return ast.unparse(E_module)


def test_populations_have_expected_incoming_connections(pops):
    E_pop, I_pop = pops["E"], pops["I"]
    assert len(E_pop.incoming) == 2
    assert len(I_pop.incoming) == 2
    for conn in (*E_pop.incoming, *I_pop.incoming):
        assert conn.fabric.topology is not None
        assert conn.fabric.topology.target_indices.shape[0] > 0


def test_I_kernel_also_compiles(I_module):
    assert isinstance(I_module, ast.Module)
    assert isinstance(I_module.body[0], ast.FunctionDef)


def test_one_outer_loop_over_unified_spike_buffer(func_E):
    outer_loops = sum(1 for s in func_E.body if isinstance(s, ast.For))
    assert outer_loops == 1


def test_dispatches_via_if_elif_on_spike_conn(func_E, src_E):
    # spike_conn_j is warp-uniform (same value for every lane in this program),
    # so this branch is not divergent, and each branch runs that connection's
    # own transpiled AST with its own directly-named pointers — no dynamic
    # pointer selection needed, the branch itself is the selection.
    has_spike_conn_dispatch = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "spike_conn_j"
        for node in ast.walk(func_E)
    )
    assert has_spike_conn_dispatch
    assert "spike_conn_j == 0" in src_E
    assert "spike_conn_j == 1" in src_E


def test_scalar_per_branch_csr_range_lookup(src_E):
    # scalar lookup inside each branch, using that connection's own pointer
    # directly — no block-level vectorized pre-gather, no per-connection
    # link_start_i/link_end_i vectors — indexed by this program's row of the
    # blocked [n_blocks, N_src + 1] offsets table
    assert "link_start = tl.load(conn0_src_offsets_ptr + _row + spike_src_j)" in src_E
    assert "link_end = tl.load(conn0_src_offsets_ptr + _row + spike_src_j + 1)" in src_E
    assert "link_start = tl.load(conn1_src_offsets_ptr + _row + spike_src_j)" in src_E
    assert "_row = pid.to(tl.int64) * " in src_E
    assert "range(link_start, link_end, LINK_BLOCK)" in src_E
    assert "link_start_0" not in src_E, "old block-level vectorized gather should be gone"


def test_links_densified_into_per_neuron_is_target(src_E):
    assert "is_target" in src_E
    assert "_hit = (tl.arange(0, GROUP_SIZE)[:, None] == _tgt[None, :]) & _lk_valid[None, :] & mask[:, None]" in src_E
    assert "is_target = tl.sum(_hit.to(tl.int32), axis=1) > 0" in src_E
    assert "_own" not in src_E, "ownership is structural in the blocked layout — no _own mask"


def test_edge_weight_loaded_as_link_tile_then_reduced(src_E):
    assert "edge.weight" not in src_E, "AttrPrefixTransformer missed edge.weight"
    assert "conn0_weight_ptr" in src_E
    assert "lk_weight = tl.load(conn0_weight_ptr + _lk, mask=_lk_valid, other=0)" in src_E
    assert "edge_weight = tl.sum(tl.where(_hit, lk_weight[None, :], 0), axis=1)" in src_E


def test_receive_masked_by_is_target(src_E):
    assert "tl.where(is_target, current + edge_weight, current)" in src_E


def test_update_send_masked_by_is_target(src_E):
    assert "tl.where(is_target" in src_E


def test_event_driven_timing(src_E):
    assert "time = tl.load(spike_time_ptr + _s)" in src_E
    assert "time = t" in src_E


def test_signature_has_unified_buffer_and_csr_args(func_E):
    arg_names = {a.arg for a in func_E.args.args}
    assert "spike_src_ptr" in arg_names
    assert "spike_conn_ptr" in arg_names
    assert "spike_tgt_ptr" not in arg_names, "old spike_tgt_ptr should be gone"
    assert "spike_link_ptr" not in arg_names, "old spike_link_ptr should be gone"
    assert "conn0_src_offsets_ptr" in arg_names
    assert "conn0_target_indices_ptr" in arg_names
    assert "conn1_src_offsets_ptr" in arg_names


def test_spike_lowers_to_local_buffer_write_no_atomic(func_E, src_E):
    # Per-emission-site writes go into this program's private local_spike_*
    # slice with no atomic (plain scalar local_spike_count, since no other
    # program instance touches this slice). PD14_neuron calls bare n2.spike()
    # (no time arg), so the time=None default should record t, the kernel's
    # own timestep boundary, not the contextual time.
    arg_names = {a.arg for a in func_E.args.args}
    assert {"local_spike_node_ptr", "local_spike_time_ptr",
            "out_spike_node_ptr", "out_spike_time_ptr", "out_spike_count_ptr"} <= arg_names
    assert "out_spike_ptr" not in arg_names, "old placeholder out_spike_ptr should be gone"
    assert "LOCAL_SPIKE_CAP" in arg_names
    assert "tl.store(local_spike_node_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, offs, mask=_m0)" in src_E
    assert "tl.store(local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, t, mask=_m0)" in src_E, \
        "bare n2.spike() should default to recording t, not the contextual time"


def test_exactly_one_atomic_add_per_program_instance(src_E):
    # Exactly one tl.atomic_add call (the end-of-kernel flush), not one per
    # spike check — aggregation is the whole point of the local buffer.
    assert src_E.count("tl.atomic_add(") == 1
    assert "_global_base = tl.atomic_add(out_spike_count_ptr, local_spike_count)" in src_E
    assert "tl.arange(0, LOCAL_SPIKE_CAP)" in src_E
    assert "tl.store(out_spike_node_ptr + _global_base + _flush_idx, _flush_node, mask=_flush_mask)" in src_E


def test_per_link_fire_check_rooted_at_is_target(src_E):
    # Must be rooted at is_target, not the whole-group mask — otherwise a
    # non-target lane already above threshold would leak a spike record.
    assert "_m0 = is_target & (potential > -50.0)" in src_E


def test_math_exp_lowered_to_tl_exp(src_E):
    # PD14's update hook calls math.exp on tensors; MathLoweringTransformer
    # must have rewritten it for the kernel to even be well-formed.
    assert "math." not in src_E
    assert "tl.exp(" in src_E
