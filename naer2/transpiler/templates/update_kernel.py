"""
Template for the merged update-receive-send Triton kernel.

Every Population compiles to exactly this kernel.  The template provides the
fixed outer structure; hook transpilation fills in the three variable sections.

Kernel signature (dynamically constructed from the population's state and its
incoming connections):

    @triton.jit
    def update_kernel(
        # per-neuron dynamic state — one ptr per dynamic attribute
        <attr>_ptr, ...,
        n_neurons,

        # per-incoming-connection spike buffers (produced by the gather kernel)
        # one set per connection, ordered by self.incoming
        spike_tgt_{i}_ptr,   # [n_spikes_i] int32  — target neuron index
        spike_link_{i}_ptr,  # [n_spikes_i] int32  — flat link index → edge state row
        n_spikes_{i},
        # per-link edge state for connection i (weight, delay, …)
        conn{i}_{attr}_ptr, ...,

        # current timestamp
        t,
        # output: one bool per neuron indicating whether it fired this step
        out_spike_ptr,

        GROUP_SIZE: tl.constexpr,
        SPIKE_BLOCK: tl.constexpr,
    ):

Kernel body — two parts:

    ┌─ Part 1: Receive ──────────────────────────────────────────────────────┐
    │  For each incoming connection i (unrolled at compile time):            │
    │    for blk_start in range(0, n_spikes_i, SPIKE_BLOCK):                │
    │      load tgt[SPIKE_BLOCK], link[SPIKE_BLOCK]                         │
    │      for j in tl.static_range(SPIKE_BLOCK):                           │
    │        load edge attrs: weight = conn{i}_weight_ptr[link[j]], …       │
    │        is_target = blk_mask[j] & (offs == tgt[j])   # [GROUP_SIZE]   │
    │        <<< receive_section — transpiled from receive hook >>>          │
    └────────────────────────────────────────────────────────────────────────┘

    ┌─ Part 2: Update + Send ────────────────────────────────────────────────┐
    │  <<< update_section — transpiled from update hook >>>                  │
    │  <<< send_section   — transpiled from send hook   >>>                  │
    └────────────────────────────────────────────────────────────────────────┘

The receive section runs inside the spike loop where is_target (a [GROUP_SIZE]
bool mask) and per-link edge attributes are local.  An EdgeAttrTransformer
(TODO) will rewrite edge.weight → weight etc. in the hook's AST before it
reaches this template.  The update and send sections run over the full group
with no spike context — they see only per-neuron state and the timestamp t.
"""

import ast

_EXPR  = lambda src: ast.parse(src, mode="eval").body
_STMTS = lambda src: ast.parse(src).body


class UpdateKernelTemplate:
    """
    Assembles the ast.Module for the merged update-receive-send kernel.

    incoming_with_hooks — list of (Connection, list[ast.stmt]) in the same
    order as population.incoming.  Each entry pairs one incoming connection
    with the receive-hook stmts that should run for its spikes.
    """

    def __init__(self, context, incoming_with_hooks: list):
        self.ctx = context
        self.incoming = incoming_with_hooks  # [(conn, recv_stmts), ...]

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

        # Part 1: receive — one loop per incoming connection (compile-time unroll)
        for conn_idx, (conn, recv_stmts) in enumerate(self.incoming):
            body += self._receive_loop(conn_idx, conn, recv_stmts)

        # Part 2: update + send
        body += update_section
        body += send_section

        # store updated per-neuron state
        for attr in self.ctx.dynamic_attrs:
            body += _STMTS(f"tl.store({attr}_ptr + offs, {attr}, mask=mask)")

        module = ast.Module(body=[self._func(body)], type_ignores=[])
        ast.fix_missing_locations(module)
        return module

    # ------------------------------------------------------------------ parts

    def _receive_loop(self, conn_idx: int, conn, recv_stmts: list) -> list[ast.stmt]:
        """Spike delivery loop for one incoming connection."""
        n_var  = f"n_spikes_{conn_idx}"
        tgt_p  = f"spike_tgt_{conn_idx}_ptr"
        link_p = f"spike_link_{conn_idx}_ptr"

        # load per-link edge attributes before calling the receive section
        edge_loads: list[ast.stmt] = []
        for attr in conn.fabric.state:
            edge_loads += _STMTS(
                f"{attr} = tl.load(conn{conn_idx}_{attr}_ptr + link[j])"
            )

        inner_body: list[ast.stmt] = [
            *edge_loads,
            *_STMTS("is_target = blk_mask[j] & (offs == tgt[j])"),
            *(recv_stmts or [ast.Pass()]),
        ]

        inner_for = ast.For(
            target=ast.Name(id="j", ctx=ast.Store()),
            iter=_EXPR("tl.static_range(SPIKE_BLOCK)"),
            body=inner_body,
            orelse=[],
        )

        outer_body: list[ast.stmt] = [
            *_STMTS(f"blk      = blk_start + tl.arange(0, SPIKE_BLOCK)"),
            *_STMTS(f"blk_mask = blk < {n_var}"),
            *_STMTS(f"tgt  = tl.load({tgt_p}  + blk, mask=blk_mask, other=-1)"),
            *_STMTS(f"link = tl.load({link_p} + blk, mask=blk_mask, other=-1)"),
            inner_for,
        ]

        return [ast.For(
            target=ast.Name(id="blk_start", ctx=ast.Store()),
            iter=_EXPR(f"range(0, {n_var}, SPIKE_BLOCK)"),
            body=outer_body,
            orelse=[],
        )]

    # ------------------------------------------------------------------ signature

    def _signature_args(self) -> list[ast.arg]:
        args: list[ast.arg] = []

        # per-neuron state pointers
        for attr in self.ctx.dynamic_attrs:
            args.append(ast.arg(arg=f"{attr}_ptr"))
        args.append(ast.arg(arg="n_neurons"))

        # per-connection spike buffers and per-link edge state
        for i, (conn, _) in enumerate(self.incoming):
            args += [
                ast.arg(arg=f"spike_tgt_{i}_ptr"),
                ast.arg(arg=f"spike_link_{i}_ptr"),
                ast.arg(arg=f"n_spikes_{i}"),
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
