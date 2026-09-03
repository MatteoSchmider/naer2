"""
Template for the merged update-receive-send Triton kernel.

Every Population compiles to exactly this kernel.  The template provides the
fixed outer structure; hook transpilation fills in the variable sections.

Kernel signature:

    @triton.jit
    def update_kernel(
        # per-neuron dynamic state
        <attr>_ptr, ..., n_neurons,

        # merged sorted spike buffer (one entry per source neuron firing)
        spike_time_ptr,  # [n_spikes] float32 — sorted arrival times
        spike_src_ptr,   # [n_spikes] int32   — source neuron index
        spike_conn_ptr,  # [n_spikes] int32   — incoming connection id (0-based)
        n_spikes,

        # per-connection topology (CSR) and per-link edge state
        conn{i}_src_offsets_ptr,      # [N_src_i + 1] int64 — CSR row pointers
        conn{i}_target_indices_ptr,   # [M_i] int32         — CSR column indices
        conn{i}_{attr}_ptr, ...,      # [M_i] float32       — per-link edge attrs

        t,             # current kernel timestamp
        out_spike_ptr, # output spike flags (TODO: n2.spike() lowering)

        GROUP_SIZE: tl.constexpr,
        SPIKE_BLOCK: tl.constexpr,
    ):

Kernel body — two parts:

    ┌─ Part 1: Event-driven spike loop ─────────────────────────────────────┐
    │  for blk_start in range(0, n_spikes, SPIKE_BLOCK):                    │
    │    load SPIKE_BLOCK entries from unified buffer                        │
    │    # vectorized CSR range pre-fetch per connection:                    │
    │    link_start_{i} = tl.load(conn{i}_src_offsets_ptr + spike_src,      │
    │                             mask=blk_mask & (spike_conn == i), other=0)│
    │    link_end_{i}   = tl.load(conn{i}_src_offsets_ptr + spike_src + 1,  │
    │                             mask=blk_mask & (spike_conn == i), other=0)│
    │    for j in tl.static_range(SPIKE_BLOCK):         ← compile-unrolled  │
    │      if blk_mask[j]:                              ← skip padding       │
    │        time = spike_t[j]                                               │
    │        # one link loop per connection — empty range if wrong conn:     │
    │        for link_idx in range(link_start_0[j], link_end_0[j]):         │
    │          tgt    = target_indices[link_idx]                             │
    │          weight = weight_ptr[link_idx]                                 │
    │          is_target = mask & (offs == tgt)                              │
    │          <<< update_section — propagate target to spike time >>>       │
    │          <<< send_section   — fire check at spike time >>>             │
    │          <<< edge forward for conn 0 >>>                               │
    │          <<< receive for conn 0 >>>                                    │
    │        for link_idx in range(link_start_1[j], link_end_1[j]): ...     │
    └────────────────────────────────────────────────────────────────────────┘

    ┌─ Part 2: Final propagation ────────────────────────────────────────────┐
    │  time = t                                                               │
    │  <<< update_section — propagate from last spike time to t >>>          │
    │  <<< send_section   — final fire check >>>                             │
    └────────────────────────────────────────────────────────────────────────┘

The spike buffer stores one entry per SOURCE NEURON FIRING, not one per
delivery.  All targets of that source neuron are found by walking the CSR
topology inside the kernel.  The GROUP_SIZE SPMD threads are the target
neurons; each checks whether it is the target of the current link via the
`is_target = mask & (offs == tgt)` mask.

When spike_conn[j] != c, the pre-loaded link_start_c[j] == link_end_c[j] == 0
(the gather mask zeroes out wrong-connection entries), so range(0, 0) is empty.
No if-elif dispatch is needed.

Per-link delays are intentionally ignored here — all deliveries from a source
spike are processed at the same arrival time from the spike buffer.  This is a
known approximation to be revisited when delay handling is implemented.

Known gaps:
  - n2.spike() lowering: send section's tl.store uses wrong variable names.
  - edge backward: deferred, needs reverse CSR.
  - source.attr references: SourceAttrTransformer emits placeholders only.
"""

import ast
import copy

from ..transformers import TargetMaskTransformer

_EXPR = lambda src: ast.parse(src, mode="eval").body
_STMTS = lambda src: ast.parse(src).body


class UpdateKernelTemplate:
    """
    Assembles the ast.Module for the merged update-receive-send kernel.

    incoming_with_hooks — list of (Connection, recv_stmts, edge_fwd_stmts) in
    the same order as population.incoming.
    """

    def __init__(self, context, incoming_with_hooks: list):
        self.ctx = context
        self.incoming = incoming_with_hooks  # [(conn, recv_stmts, edge_fwd_stmts)]

    # ------------------------------------------------------------------ public

    def build(self, update_section: list, send_section: list) -> ast.Module:
        body: list[ast.stmt] = []

        # SPMD setup
        body += _STMTS("""\
pid  = tl.program_id(0)
offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
mask = offs < n_neurons
""")

        # load per-neuron dynamic state
        for attr in self.ctx.dynamic_attrs:
            body += _STMTS(f"{attr} = tl.load({attr}_ptr + offs, mask=mask)")

        # Part 1: event-driven spike loop.
        # Sections are deepcopied because they appear twice (inside loop + final pass).
        body += self._spike_loop(
            copy.deepcopy(update_section),
            copy.deepcopy(send_section),
        )

        # Part 2: final propagation from last spike time to current kernel time t.
        body += _STMTS("time = t")
        body += update_section
        body += send_section
        # TODO: edge backward (needs reverse CSR)
        # TODO: n2.spike() → out_spike_ptr accumulation

        # store updated per-neuron state
        for attr in self.ctx.dynamic_attrs:
            body += _STMTS(f"tl.store({attr}_ptr + offs, {attr}, mask=mask)")

        module = ast.Module(body=[self._func(body)], type_ignores=[])
        ast.fix_missing_locations(module)
        return module

    # ------------------------------------------------------------------ spike loop

    def _spike_loop(self, update_section: list, send_section: list) -> list[ast.stmt]:
        """Outer loop over the unified spike buffer; inner static_range per spike.

        CSR range endpoints are pre-fetched via vectorized gather at the block level.
        Wrong-connection entries are zeroed by the gather mask, producing empty
        range(0, 0) without any if-elif dispatch.
        """
        outer_load: list[ast.stmt] = [
            *_STMTS("blk      = blk_start + tl.arange(0, SPIKE_BLOCK)"),
            *_STMTS("blk_mask = blk < n_spikes"),
            *_STMTS(
                "spike_t    = tl.load(spike_time_ptr + blk, mask=blk_mask, other=t)"
            ),
            *_STMTS(
                "spike_src  = tl.load(spike_src_ptr  + blk, mask=blk_mask, other=-1)"
            ),
            *_STMTS(
                "spike_conn = tl.load(spike_conn_ptr + blk, mask=blk_mask, other=-1)"
            ),
        ]

        # Vectorized gather of CSR range endpoints per connection.
        # mask=blk_mask & (spike_conn == i) zeroes entries for other connections,
        # making range(link_start_i[j], link_end_i[j]) empty for wrong connections.
        for i in range(len(self.incoming)):
            outer_load += _STMTS(
                f"link_start_{i} = tl.load("
                f"conn{i}_src_offsets_ptr + spike_src, "
                f"mask=blk_mask & (spike_conn == {i}), other=0)"
            )
            outer_load += _STMTS(
                f"link_end_{i} = tl.load("
                f"conn{i}_src_offsets_ptr + spike_src + 1, "
                f"mask=blk_mask & (spike_conn == {i}), other=0)"
            )

        # Per-spike body: time assignment then one link loop per connection.
        per_spike_body: list[ast.stmt] = [*_STMTS("time = spike_t[j]")]
        for i, (conn, recv_stmts, edge_fwd_stmts) in enumerate(self.incoming):
            per_spike_body += self._csr_loop(
                i, conn, recv_stmts, edge_fwd_stmts, update_section, send_section
            )

        inner_for = ast.For(
            target=ast.Name(id="j", ctx=ast.Store()),
            iter=_EXPR("tl.static_range(SPIKE_BLOCK)"),
            body=[
                ast.If(
                    test=_EXPR("blk_mask[j]"),
                    body=per_spike_body,
                    orelse=[],
                )
            ],
            orelse=[],
        )

        return [
            ast.For(
                target=ast.Name(id="blk_start", ctx=ast.Store()),
                iter=_EXPR("range(0, n_spikes, SPIKE_BLOCK)"),
                body=outer_load + [inner_for],
                orelse=[],
            )
        ]

    # ------------------------------------------------------------------ csr loop

    def _csr_loop(
        self,
        conn_idx: int,
        conn,
        recv_stmts: list,
        edge_fwd_stmts: list,
        update_section: list,
        send_section: list,
    ) -> list[ast.stmt]:
        """Walk all targets of spike_src[j] in connection conn_idx via CSR.

        Range endpoints were pre-loaded at block level via gather; link_start_i[j]
        and link_end_i[j] are scalar elements of those SPIKE_BLOCK-wide vectors.
        When this is the wrong connection, both are 0 so the loop body is skipped.

        Structure per link:
          tgt    = target_indices[link_idx]       — scalar target neuron index
          weight = weight_ptr[link_idx], …        — scalar per-link edge attrs
          is_target = mask & (offs == tgt)        — [GROUP_SIZE] bool mask
          <<< update (masked by is_target) >>>    — propagate only target neuron
          <<< send   (masked by is_target) >>>    — fire check only for target
          <<< edge forward >>>                    — update edge state scalars
          <<< link stores >>>                     — write back dynamic edge attrs
          <<< receive >>>                         — accumulate PSC into target
        """
        i = conn_idx

        # Load target index and all per-link fabric state (scalars).
        # Edge attrs are loaded as edge_{attr} to match the AttrPrefixTransformer("edge")
        # rewriting applied to hook bodies (edge.weight → edge_weight).
        link_loads: list[ast.stmt] = [
            *_STMTS(f"tgt = tl.load(conn{i}_target_indices_ptr + link_idx)"),
        ]
        for attr in conn.fabric.state:
            link_loads += _STMTS(
                f"edge_{attr} = tl.load(conn{i}_{attr}_ptr + link_idx)"
            )

        # is_target mask: selects which GROUP_SIZE threads are this link's target
        is_target_stmt: list[ast.stmt] = _STMTS("is_target = mask & (offs == tgt)")

        # update+send masked to operate only on the target neuron
        target_update = self._make_target_section(update_section)
        target_send = self._make_target_section(send_section)

        # Store back only dynamic attrs (learning rules may have modified them)
        link_stores: list[ast.stmt] = []
        for attr in conn.fabric.dynamic_attrs:
            link_stores += _STMTS(
                f"tl.store(conn{i}_{attr}_ptr + link_idx, edge_{attr})"
            )

        link_body = (
            link_loads
            + is_target_stmt
            + target_update
            + target_send
            + list(edge_fwd_stmts)
            + link_stores
            + list(recv_stmts)
        )

        return [
            ast.For(
                target=ast.Name(id="link_idx", ctx=ast.Store()),
                iter=_EXPR(f"range(link_start_{i}[j], link_end_{i}[j])"),
                body=link_body or [ast.Pass()],
                orelse=[],
            )
        ]

    # ------------------------------------------------------------------ helpers

    def _make_target_section(self, stmts: list) -> list:
        """Deepcopy stmts and wrap dynamic-attr assignments with is_target masking."""
        return TargetMaskTransformer(set(self.ctx.dynamic_attrs)).transform(
            copy.deepcopy(stmts)
        )

    # ------------------------------------------------------------------ signature

    def _signature_args(self) -> list[ast.arg]:
        args: list[ast.arg] = []

        # per-neuron state pointers
        for attr in self.ctx.dynamic_attrs:
            args.append(ast.arg(arg=f"{attr}_ptr"))
        args.append(ast.arg(arg="n_neurons"))

        # unified sorted spike buffer (one entry per source neuron firing)
        args += [
            ast.arg(arg="spike_time_ptr"),
            ast.arg(arg="spike_src_ptr"),
            ast.arg(arg="spike_conn_ptr"),
            ast.arg(arg="n_spikes"),
        ]

        # per-connection CSR topology + per-link edge state
        for i, (conn, _, _) in enumerate(self.incoming):
            args += [
                ast.arg(arg=f"conn{i}_src_offsets_ptr"),
                ast.arg(arg=f"conn{i}_target_indices_ptr"),
            ]
            for attr in conn.fabric.state:
                args.append(ast.arg(arg=f"conn{i}_{attr}_ptr"))

        # timestamp and output spike buffer
        args += [ast.arg(arg="t"), ast.arg(arg="out_spike_ptr")]

        # constexpr tile sizes
        args += [
            ast.arg(arg="GROUP_SIZE", annotation=_EXPR("tl.constexpr")),
            ast.arg(arg="SPIKE_BLOCK", annotation=_EXPR("tl.constexpr")),
        ]
        return args

    def _func(self, body: list[ast.stmt]) -> ast.FunctionDef:
        return ast.FunctionDef(
            name="update_kernel",
            args=ast.arguments(
                posonlyargs=[],
                args=self._signature_args(),
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[_EXPR("triton.jit")],
            returns=None,
        )
