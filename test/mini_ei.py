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
"""

import ast

from test.networks.mini_ei import build
from test.triton_check import compiles


def _show_kernel(name: str, module: ast.Module):
    print(f"\n{'─'*60}")
    print(f"  {name} kernel")
    print(f"{'─'*60}")
    for line in ast.unparse(module).split(";"):
        print(line.strip())
    print()


def main():
    network, pops = build()
    E_pop = pops["E"]
    I_pop = pops["I"]

    print(f"E population: {len(E_pop.incoming)} incoming connections")
    for i, conn in enumerate(E_pop.incoming):
        src_name = "E" if conn._source is E_pop else "I"
        n_links = conn.fabric.topology.target_indices.shape[0] if conn.fabric.topology else 0
        print(f"  conn[{i}]: {src_name}→E  links={n_links}  "
              f"state={list(conn.fabric.state)}  dynamic={list(conn.fabric.dynamic_attrs)}")

    print(f"\nI population: {len(I_pop.incoming)} incoming connections")
    for i, conn in enumerate(I_pop.incoming):
        src_name = "E" if conn._source is E_pop else "I"
        n_links = conn.fabric.topology.target_indices.shape[0] if conn.fabric.topology else 0
        print(f"  conn[{i}]: {src_name}→I  links={n_links}  "
              f"state={list(conn.fabric.state)}  dynamic={list(conn.fabric.dynamic_attrs)}")

    print("\n── compiling E kernel ──")
    E_module = E_pop.compile()

    print("\n── compiling I kernel ──")
    I_module = I_pop.compile()

    _show_kernel("E population", E_module)
    _show_kernel("I population", I_module)

    func_E = E_module.body[0]
    func_I = I_module.body[0]
    src_E = ast.unparse(E_module)

    # ── structural checks ─────────────────────────────────────────────────

    # one outer for-loop (unified spike buffer)
    outer_loops_E = sum(1 for s in func_E.body if isinstance(s, ast.For))
    assert outer_loops_E == 1, f"expected 1 outer loop in E kernel, got {outer_loops_E}"
    print(f"E kernel: {outer_loops_E} outer for-loop (expected 1) ✓")

    # if/elif dispatch on spike_conn_j — one branch per incoming connection.
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
    assert has_spike_conn_dispatch, "E kernel missing spike_conn if-elif dispatch"
    assert "spike_conn_j == 0" in src_E, "conn0 dispatch branch missing"
    assert "spike_conn_j == 1" in src_E, "conn1 dispatch branch missing"
    print("if/elif dispatch on spike_conn_j (warp-uniform, not divergent) ✓")

    # scalar CSR range lookup inside each branch, using that connection's own
    # pointer directly — no block-level vectorized pre-gather, no per-connection
    # link_start_i/link_end_i vectors
    # ...indexed by this program's row of the blocked [n_blocks, N_src + 1] offsets table
    assert "link_start = tl.load(conn0_src_offsets_ptr + _row + spike_src_j)" in src_E
    assert "link_end = tl.load(conn0_src_offsets_ptr + _row + spike_src_j + 1)" in src_E
    assert "link_start = tl.load(conn1_src_offsets_ptr + _row + spike_src_j)" in src_E
    assert "_row = pid.to(tl.int64) * " in src_E, "block-row stride missing"
    assert "range(link_start, link_end, LINK_BLOCK)" in src_E, "per-branch link-tile loop missing"
    assert "link_start_0" not in src_E, "old block-level vectorized gather should be gone"
    print("scalar per-branch CSR range lookup, own pointer per connection ✓")

    # is_target masking computed before update/receive
    assert "is_target" in src_E, "is_target mask missing"
    assert "_hit = (tl.arange(0, GROUP_SIZE)[:, None] == _tgt[None, :]) & _lk_valid[None, :] & mask[:, None]" in src_E, \
        "one-hot link->neuron densify missing"
    assert "is_target = tl.sum(_hit.to(tl.int32), axis=1) > 0" in src_E, "is_target reduce missing"
    assert "_own" not in src_E, "ownership is structural in the blocked layout — no _own mask"
    print("links densified into per-neuron is_target via one-hot compare-reduce ✓")

    # edge.weight gone, locally loaded as edge_weight
    assert "edge.weight" not in src_E, "AttrPrefixTransformer missed edge.weight"
    assert "conn0_weight_ptr" in src_E, "conn0_weight_ptr missing from kernel signature"
    assert "lk_weight = tl.load(conn0_weight_ptr + _lk, mask=_lk_valid, other=0)" in src_E, \
        "per-link tile load missing"
    assert "edge_weight = tl.sum(tl.where(_hit, lk_weight[None, :], 0), axis=1)" in src_E, \
        "per-neuron edge_weight reduce missing"
    print("edge.weight → per-link lk_weight tile, reduced to per-neuron edge_weight ✓")

    # receive masking: additive spike delivery masked by is_target
    assert "tl.where(is_target, current + edge_weight, current)" in src_E, "receive masking pattern wrong"
    print("tl.where(is_target, current + edge_weight, current) ✓")

    # update section wrapped with tl.where(is_target, ...) inside link loop
    assert "tl.where(is_target" in src_E, "update/send not masked by is_target inside link loop"
    print("tl.where(is_target, ...) masking in update/send ✓")

    # event-driven timing
    assert "time = tl.load(spike_time_ptr + _s)" in src_E, "event-driven time assignment missing"
    assert "time = t" in src_E, "final propagation missing"
    print("event-driven time = per-spike scalar load + final time = t ✓")

    # signature: new buffer args, CSR topology args
    arg_names = {a.arg for a in func_E.args.args}
    assert "spike_src_ptr"  in arg_names, "spike_src_ptr missing"
    assert "spike_conn_ptr" in arg_names, "spike_conn_ptr missing"
    assert "spike_tgt_ptr"  not in arg_names, "old spike_tgt_ptr should be gone"
    assert "spike_link_ptr" not in arg_names, "old spike_link_ptr should be gone"
    assert "conn0_src_offsets_ptr"    in arg_names, "conn0_src_offsets_ptr missing"
    assert "conn0_target_indices_ptr" in arg_names, "conn0_target_indices_ptr missing"
    assert "conn1_src_offsets_ptr"    in arg_names, "conn1_src_offsets_ptr missing"
    print("signature: unified buffer + CSR topology args correct ✓")

    # n2.spike() lowering: per-emission-site writes go into this program's private
    # local_spike_{node,time}_ptr slice with NO atomic (plain scalar local_spike_count,
    # since no other program instance touches this slice) — only the one-time flush
    # at the very end does a single tl.atomic_add per program instance.
    # PD14_neuron calls bare n2.spike() (no time arg), so per the new interface
    # (time=None default -> records t, the kernel's own timestep boundary) this
    # should emit `t`, not the contextual `time` the old exact=True default used.
    assert "local_spike_node_ptr" in arg_names, "local_spike_node_ptr missing"
    assert "local_spike_time_ptr" in arg_names, "local_spike_time_ptr missing"
    assert "out_spike_node_ptr" in arg_names, "out_spike_node_ptr missing"
    assert "out_spike_time_ptr" in arg_names, "out_spike_time_ptr missing"
    assert "out_spike_count_ptr" in arg_names, "out_spike_count_ptr missing"
    assert "out_spike_ptr" not in arg_names, "old placeholder out_spike_ptr should be gone"
    assert "LOCAL_SPIKE_CAP" in arg_names, "LOCAL_SPIKE_CAP constexpr missing"
    assert "tl.store(local_spike_node_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, offs, mask=_m0)" in src_E
    assert "tl.store(local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, t, mask=_m0)" in src_E, \
        "bare n2.spike() should default to recording t, not the contextual time"
    print("n2.spike() -> per-program local buffer write (no atomic), default time=t ✓")

    # exactly one atomic_add per program instance (the flush step), not one per
    # spike check — aggregation is the whole point of the local buffer
    assert src_E.count("tl.atomic_add(") == 1, \
        "expected exactly one tl.atomic_add call (the end-of-kernel flush)"
    assert "_global_base = tl.atomic_add(out_spike_count_ptr, local_spike_count)" in src_E
    assert "tl.arange(0, LOCAL_SPIKE_CAP)" in src_E, "flush's vectorized copy-out missing"
    assert "tl.store(out_spike_node_ptr + _global_base + _flush_idx, _flush_node, mask=_flush_mask)" in src_E
    print("exactly one atomic_add per program instance (end-of-kernel flush) ✓")

    # per-link spike gating must be rooted at is_target, not the whole-group mask —
    # otherwise a non-target lane already above threshold would leak a spike record
    assert "_m0 = is_target & (potential > -50.0)" in src_E, \
        "per-link fire-check mask should be seeded from is_target (MaskAliasTransformer)"
    print("per-link fire-check correctly rooted at is_target, not mask ✓")

    # PD14's update hook calls math.exp on tensors; MathLoweringTransformer must
    # have rewritten it to tl.exp for these to compile at all
    assert "math." not in src_E and "tl.exp(" in src_E, "math.exp should be lowered to tl.exp"
    compiles(E_module, "E")
    compiles(I_module, "I")
    print("E and I kernels compile through Triton (math.exp -> tl.exp) ✓")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
