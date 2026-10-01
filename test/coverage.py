"""
General compilation coverage smoke-test (formerly the "counter" network test).

Builds a 4-population network exercising:
  - A -> B -> C chain:
      - flexible @ chaining order (port / mapping / pattern in any order)
      - fabric_map: edge.w -> real attr 'weight', on all static-synapse connections
      - source_map: source.src_pot -> real attr 'potential', on the B->C connection
      - two differently-named receive ports ("positive" / "negative")
      - a static synapse (A->B) with a per-link weight *tensor* rather than a
        shared scalar — "static" means the tensor never changes over the
        simulation, not that every link has the same value, so this must still
        load per-link from conn0_weight_ptr rather than inline a literal
      - a plastic synapse (B->C) whose forward hook mutates weight per link
  - A, B, and C all -> D, through the SAME receive port ("positive"):
      three independent source populations, three independent CSR structures,
      dispatched at runtime via an if/elif chain on spike_conn, funneled into
      one receive hook — coverage for multiple input sites landing on one
      population. A->D and B->D share an edge type (static_synapse); C->D
      uses a DIFFERENT edge type (plastic_synapse) — this is the case that
      exercises genuinely different per-connection code (not just different
      topology), since plastic_synapse's forward hook mutates weight and
      static_synapse's doesn't.

Expected kernel structure highlights:
  - conn._target_port routes to the correct receive hook per connection
  - edge.w -> edge_weight (fabric_map reversed to the real attr name)
  - source.src_pot -> source_potential (source_map reversed to the real attr name)
  - all static-synapse connections load edge_weight from conn{i}_weight_ptr
    (tensor-backed — either because forward assigns it, or because it was
    given an explicit per-link override from outside)
  - source.src_pot gather-loads B's real potential from conn0_source_potential_ptr,
    indexed by spike_src_j, once per connection per spike (not per link)
  - is_target masking applied throughout
  - D's kernel dispatches spike_conn_j across three branches (conn0=A,
    conn1=B, conn2=C), each with its own CSR topology + weight pointers and
    its own transpiled edge-forward AST (a no-op for the two static synapses,
    a real per-link mutation for the plastic one), all routed to pos_receive
"""

import ast

from test.networks.coverage import build
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
    A, B, C, D = pops["A"], pops["B"], pops["C"], pops["D"]

    print(f"A: {len(A.incoming)} incoming connections")
    print(f"B: {len(B.incoming)} incoming connections")
    for i, conn in enumerate(B.incoming):
        print(f"  conn[{i}]: port={conn._target_port!r} "
              f"fabric_map={conn._fabric_map!r} source_map={conn._source_map!r}")
    print(f"C: {len(C.incoming)} incoming connections")
    for i, conn in enumerate(C.incoming):
        print(f"  conn[{i}]: port={conn._target_port!r} "
              f"fabric_map={conn._fabric_map!r} source_map={conn._source_map!r}")
    print(f"D: {len(D.incoming)} incoming connections")
    src_names = {A: "A", B: "B", C: "C"}
    for i, conn in enumerate(D.incoming):
        print(f"  conn[{i}]: source={src_names[conn._source]} port={conn._target_port!r} "
              f"fabric_map={conn._fabric_map!r} source_map={conn._source_map!r}")

    print("\n── compiling A kernel ──")
    A_module = A.compile()
    print("\n── compiling B kernel ──")
    B_module = B.compile()
    print("\n── compiling C kernel ──")
    C_module = C.compile()
    print("\n── compiling D kernel ──")
    D_module = D.compile()

    _show_kernel("A population", A_module)
    _show_kernel("B population", B_module)
    _show_kernel("C population", C_module)
    _show_kernel("D population", D_module)

    src_B = ast.unparse(B_module)
    src_C = ast.unparse(C_module)
    src_D = ast.unparse(D_module)

    # ── structural checks ─────────────────────────────────────────────────

    assert len(A.incoming) == 0, "A should have no incoming connections"
    assert len(B.incoming) == 1, "B should have exactly 1 incoming connection"
    assert len(C.incoming) == 1, "C should have exactly 1 incoming connection"
    assert len(D.incoming) == 3, "D should have exactly 3 incoming connections"
    print("population fan-in counts correct ✓")

    # port routing (str chained onto the edge, in either position)
    assert B.incoming[0]._target_port == "positive", "B's connection should use the 'positive' port"
    assert C.incoming[0]._target_port == "negative", "C's connection should use the 'negative' port"
    assert all(conn._target_port == "positive" for conn in D.incoming), \
        "both of D's connections should use the 'positive' port"
    print("target_port routing correct (positive / negative) ✓")

    # fabric_map: edge.w -> edge_weight, on all static-synapse connections
    assert B.incoming[0]._fabric_map == {"weight": "w"}
    assert C.incoming[0]._fabric_map == {"weight": "w"}
    assert all(conn._fabric_map == {"weight": "w"} for conn in D.incoming)
    assert "edge.w" not in src_B and "edge.weight" not in src_B
    assert "edge_weight" in src_B
    assert "edge_weight" in src_C
    assert "edge_weight" in src_D
    print("fabric_map: edge.w -> edge_weight (reversed rename) ✓")

    # weight is tensor-backed on every connection, so all load edge_weight from
    # a real per-link pointer rather than inlining a literal. B's and D's static
    # synapse weights are static (forward never assigns them) but were still
    # given an explicit per-link tensor via conn.fabric.weight = torch.rand(...),
    # so they must load like any dynamic attr — Fabric.tensor_attrs is what
    # decides this, not dynamic_attrs alone.
    assert "conn0_weight_ptr" in {a.arg for a in B.compile().body[0].args.args}, \
        "B's weight is tensor-backed and needs a conn0_weight_ptr signature arg"
    assert "lk_weight = tl.load(conn0_weight_ptr + _lk, mask=_lk_valid, other=0)" in src_B, \
        "B's weight should load per-link, not inline a literal, since a tensor was assigned"
    assert "lk_weight = tl.load(conn0_weight_ptr + _lk, mask=_lk_valid, other=0)" in src_C
    assert "edge_weight = tl.sum(tl.where(_hit, lk_weight[None, :], 0), axis=1)" in src_B, \
        "receive should see the per-neuron reduction of the per-link tile"
    print("connections load edge_weight from conn{i}_weight_ptr (tensor-backed) ✓")

    # source_map: source.src_pot -> source_potential, C's connection only
    assert B.incoming[0]._source_map is None, "B's connection should have no source_map"
    assert C.incoming[0]._source_map == {"potential": "src_pot"}
    assert "source.src_pot" not in src_C
    assert "source_potential" in src_C
    assert "source_potential" not in src_B
    print("source_map: source.src_pot -> source_potential (reversed rename) ✓")

    # source.src_pot is a real source-population attr (B.potential) and must be
    # gather-loaded per spike (indexed by spike_src_j), not left undefined —
    # only C's connection references it, so only C gets the pointer/load.
    assert "conn0_source_potential_ptr" in {a.arg for a in C.compile().body[0].args.args}, \
        "C's connection references source.src_pot -> needs a conn0_source_potential_ptr arg"
    assert "conn0_source_potential_ptr" not in {a.arg for a in B.compile().body[0].args.args}, \
        "B's connection never references source.* -> should get no source pointer arg"
    assert "source_potential = tl.load(conn0_source_potential_ptr + spike_src_j)" in src_C, \
        "source_potential should be gather-loaded once per spike (per connection), not left unbound"
    print("source.src_pot -> source_potential gather-loaded from conn0_source_potential_ptr ✓")

    # is_target masking present throughout
    assert "is_target" in src_B and "is_target" in src_C and "is_target" in src_D
    print("is_target masking present ✓")

    # plastic_synapse forward hook (dynamic edge state) on C's connection only.
    # This is an unconditional per-link scalar update, not masked by is_target:
    # weight belongs to a link alone, not to the GROUP_SIZE target-neuron lanes.
    assert "conn0_weight_ptr" in src_C
    assert "lk_weight += 0.1" in src_C, "plastic forward runs on link lanes as lk_weight"
    assert "tl.where(is_target, lk_weight" not in src_C, "edge forward should not be is_target-masked"
    # In the blocked layout a program's slice of a row holds only links into its
    # own neurons, so the write-back needs no ownership mask — only the ragged
    # tile end. (With the old whole-row walk, all P programs updated every link.)
    assert "tl.store(conn0_weight_ptr + _lk, lk_weight, mask=_lk_valid)" in src_C, \
        "weight write-back should be masked to valid tile lanes only"
    assert "_own" not in src_C, "ownership is structural in the blocked layout"
    assert "tl.store(conn0_weight_ptr" not in src_B, "static synapse never writes weights back"
    print("plastic_synapse forward on link lanes (lk_weight += 0.1), stored per link (ownership is structural) ✓")

    # n2.spike() lowering: per-emission-site writes go into this program's private
    # local_spike_{node,time}_ptr slice, no atomic — only the one-time flush at the
    # very end does a single tl.atomic_add per program instance (see mini_ei.py's
    # dedicated check for that; here we just confirm the per-site write and timing).
    # send() calls bare n2.spike() (no time arg) -> new interface's time=None
    # default records t, the kernel's own timestep boundary, not the contextual
    # time the old exact=True default used.
    arg_names_C = {a.arg for a in C_module.body[0].args.args}
    assert {"local_spike_node_ptr", "local_spike_time_ptr"} <= arg_names_C, \
        "per-program local spike scratch args missing"
    assert {"out_spike_node_ptr", "out_spike_time_ptr", "out_spike_count_ptr"} <= arg_names_C, \
        "compacted output spike buffer args missing"
    assert "out_spike_ptr" not in arg_names_C, "old placeholder out_spike_ptr should be gone"
    assert "LOCAL_SPIKE_CAP" in arg_names_C, "LOCAL_SPIKE_CAP constexpr missing"
    assert "tl.store(local_spike_node_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, offs, mask=_m0)" in src_C
    assert "tl.store(local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, t, mask=_m0)" in src_C, \
        "bare n2.spike() in send() should default to recording t"
    print("n2.spike() -> per-program local buffer write (no atomic), default time=t ✓")

    # send's delayed extra spike: n2.spike(time + 1.0) records the explicit time
    # expression verbatim (per-link: the arriving spike's time plus a delay);
    # the plain n2.spike() sites record t.
    for label, src in (("B", src_B), ("C", src_C), ("D", src_D)):
        assert "tl.store(local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, time + 1.0, mask=_m0)" in src, label
    print("n2.spike(time + delay) in send -> explicit time expression recorded verbatim ✓")

    # Coverage-only spike sites with thresholds that can never trigger: one in
    # receive (rooted at is_target, records the arriving spike's time) and one in
    # update (runs in the final pass, rooted at the whole-group mask).
    assert "_m0 = is_target & (potential > 1000000.0)" in src_B and "_m0 = is_target & (potential > 1000000.0)" in src_D
    assert "_m0 = mask & (potential < -1000000.0)" in src_B, "spike emission from update (final pass) missing"
    print("n2.spike from receive and from update lowered (coverage-only thresholds) ✓")

    # per-link fire-checks must be rooted at is_target, not the whole-group mask,
    # else a non-target lane already past threshold leaks a spurious spike record.
    # This applies to send's fire-check (via _make_target_section; B, C, D); a
    # receive hook's own if-branch goes through emit_for_spike_loop's
    # MaskAliasTransformer the same way, but the current relay fixture has none.
    for label, src in (("B", src_B), ("C", src_C), ("D", src_D)):
        assert "_m0 = is_target & (potential > 2 * 5)" in src, \
            f"{label}: send's far-past-threshold branch should be seeded from is_target (MaskAliasTransformer)"
        assert "_m1 = ~_m0 & is_target & (potential > 5)" in src, \
            f"{label}: send's elif branch should chain off the first mask, still rooted at is_target"
    print("per-link fire-checks (send and receive) correctly rooted at is_target ✓")

    # Regression: exactly one emission per spike site written in the hooks. In
    # relay, send has three (two in the far-past-threshold case, one in the elif)
    # and pos_receive has one (a never-firing coverage-only site), so a link tile
    # has 4 for B/D (positive port) and 3 for C (negative port has no spike). A
    # receive hook that spikes without resetting would double-emit alongside send.
    def emissions_per_link_loop(module):
        return [
            ast.unparse(n).count("tl.cumsum(")
            for n in ast.walk(module.body[0])
            if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "link_tile"
        ]
    assert emissions_per_link_loop(B_module) == [4], emissions_per_link_loop(B_module)
    assert emissions_per_link_loop(C_module) == [3], emissions_per_link_loop(C_module)
    assert emissions_per_link_loop(D_module) == [4, 4, 4], emissions_per_link_loop(D_module)
    print("no duplicate spike emission inside link loops ✓")

    # ── D: three independent input sites into one population ────────────────

    arg_names_D = {a.arg for a in D_module.body[0].args.args}
    assert D.incoming[0]._source is A and D.incoming[1]._source is B and D.incoming[2]._source is C, \
        "expected D.incoming order [A->D, B->D, C->D] matching build() insertion order"

    # three fully independent CSR structures (conn0=A->D, conn1=B->D, conn2=C->D),
    # each with its own topology pointers and its own weight pointer
    assert {"conn0_src_offsets_ptr", "conn0_target_indices_ptr", "conn0_weight_ptr"} <= arg_names_D
    assert {"conn1_src_offsets_ptr", "conn1_target_indices_ptr", "conn1_weight_ptr"} <= arg_names_D
    assert {"conn2_src_offsets_ptr", "conn2_target_indices_ptr", "conn2_weight_ptr"} <= arg_names_D
    print("D has three independent CSR structures (conn0=A, conn1=B, conn2=C) ✓")

    # if/elif dispatch on spike_conn_j, one branch per connection, each with
    # its own scalar CSR range lookup via that connection's own pointer
    assert "spike_conn_j == 0" in src_D and "spike_conn_j == 1" in src_D and "spike_conn_j == 2" in src_D, \
        "D should dispatch all 3 connections via if/elif on spike_conn_j"
    func_D = D_module.body[0]
    link_loops_D = [
        node for node in ast.walk(func_D)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "link_tile"
    ]
    assert len(link_loops_D) == 3, f"expected 3 link loops in D's kernel, got {len(link_loops_D)}"
    print("D dispatches spike_conn_j across 3 branches (conn0, conn1, conn2) ✓")

    # conn0 and conn1 (static_synapse) never mutate weight; conn2 (plastic_synapse)
    # does — same receive port, genuinely different edge-forward code per connection
    assert "lk_weight = tl.load(conn2_weight_ptr + _lk, mask=_lk_valid, other=0)" in src_D
    assert "lk_weight += 0.1" in src_D, "conn2's plastic_synapse forward hook missing from D's kernel"
    assert src_D.count("lk_weight += 0.1") == 1, \
        "only conn2 (plastic_synapse) should mutate weight; conn0/conn1 (static_synapse) must not"
    print("conn2's plastic forward hook present alongside conn0/conn1's static (no-op) forward ✓")

    # ── the generated kernels must actually compile ─────────────────────────
    for label, module in (("A", A_module), ("B", B_module), ("C", C_module), ("D", D_module)):
        compiles(module, label)
    print("all four kernels compile through Triton (frontend -> IR -> PTX, fake sm_80) ✓")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
