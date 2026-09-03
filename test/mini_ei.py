"""
Mini E/I compilation smoke-test.

Builds a 2-population E/I circuit (100 E + 25 I neurons, 4 connections) and
compiles the merged update-receive-send kernel for both populations.

Expected kernel structure:
  - ONE outer for-loop over the unified sorted spike buffer
  - Block-level vectorized gather of CSR range endpoints per connection:
      link_start_0 = tl.load(conn0_src_offsets_ptr + spike_src, mask=..., other=0)
      link_end_0   = tl.load(conn0_src_offsets_ptr + spike_src + 1, ...)
  - Inner tl.static_range, guarded by blk_mask[j], containing:
      * time = spike_t[j]
      * One per-connection for-loop:
          for link_idx in range(link_start_0[j], link_end_0[j]):
            is_target = mask & (offs == tgt)
            <<update+send masked by is_target>>
            <<edge forward + receive>>
      (wrong-connection entries pre-zeroed → range(0,0), no if-elif dispatch)
  - Final propagation block (time = t) after the spike loop
  - edge.weight replaced by locally loaded scalar weight
  - current = current + weight * is_target  (ReceiveMaskTransformer)
  - Signature: spike_src_ptr (not spike_tgt_ptr), conn{i}_src_offsets_ptr,
    conn{i}_target_indices_ptr
"""

import ast

from test.networks.mini_ei import build


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

    # no if-elif spike_conn dispatch — replaced by separate link loops
    has_spike_conn_dispatch = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Subscript)
        and isinstance(node.test.left.value, ast.Name)
        and node.test.left.value.id == "spike_conn"
        for node in ast.walk(func_E)
    )
    assert not has_spike_conn_dispatch, "E kernel still has spike_conn if-elif dispatch"
    print("No spike_conn if-elif dispatch (replaced by empty-range link loops) ✓")

    # vectorized gather of CSR range endpoints
    assert "link_start_0" in src_E, "vectorized link_start_0 gather missing"
    assert "link_end_0"   in src_E, "vectorized link_end_0 gather missing"
    assert "link_start_1" in src_E, "vectorized link_start_1 gather missing"
    assert "link_end_1"   in src_E, "vectorized link_end_1 gather missing"
    assert "spike_conn == 0" in src_E, "gather mask for conn 0 missing"
    assert "spike_conn == 1" in src_E, "gather mask for conn 1 missing"
    print("Vectorized CSR range gather (link_start_i / link_end_i) ✓")

    # per-connection link loops using the pre-fetched ranges
    assert "range(link_start_0[j], link_end_0[j])" in src_E, "conn0 link loop missing"
    assert "range(link_start_1[j], link_end_1[j])" in src_E, "conn1 link loop missing"
    print("Per-connection link loops range(link_start_i[j], link_end_i[j]) ✓")

    # is_target masking computed before update/receive
    assert "is_target" in src_E, "is_target mask missing"
    assert "is_target = mask & (offs == tgt)" in src_E, "is_target assignment pattern wrong"
    print("is_target = mask & (offs == tgt) ✓")

    # edge.weight gone, locally loaded as edge_weight
    assert "edge.weight" not in src_E, "AttrPrefixTransformer missed edge.weight"
    assert "conn0_weight_ptr" in src_E, "conn0_weight_ptr missing from kernel signature"
    assert "edge_weight = tl.load(conn0_weight_ptr" in src_E, "edge_weight local load missing"
    print("edge.weight → edge_weight (locally loaded) ✓")

    # receive masking: additive spike delivery masked by is_target
    assert "tl.where(is_target, current + edge_weight, current)" in src_E, "receive masking pattern wrong"
    print("tl.where(is_target, current + edge_weight, current) ✓")

    # update section wrapped with tl.where(is_target, ...) inside link loop
    assert "tl.where(is_target" in src_E, "update/send not masked by is_target inside link loop"
    print("tl.where(is_target, ...) masking in update/send ✓")

    # event-driven timing
    assert "spike_t[j]" in src_E, "event-driven time assignment missing"
    assert "time = t" in src_E, "final propagation missing"
    print("event-driven time = spike_t[j] + final time = t ✓")

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

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
