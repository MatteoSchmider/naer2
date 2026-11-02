"""
Template for the merged update-receive-send Triton kernel -- the NO-DELAY
variant: every incoming delivery from one firing source neuron lands at that
spike's own recorded time, for every link out of that source alike. Edge
forward/backward callbacks (learned/plastic per-link state) are fully
supported here.

A Population compiles to this kernel unless any incoming connection carries
a real per-link delay tensor (Population.uses_per_link_delay_kernel), in
which case it compiles to the sibling per_link_delay_update_kernel.py
instead -- see that module's docstring and TODO.md for why the two are
separate rather than one kernel handling both: once per-link delay is
real, a single delivery's processing cost and shape changes enough (one
entry per synapse activation instead of per firing neuron, no CSR walk,
gather/scatter instead of a block-local vector load/store) that forcing
both through one template would complicate this, simpler, far more common
case for a feature only some connections use.

The template provides the fixed outer structure; hook transpilation fills
in the variable sections.

Kernel signature:

    @triton.jit
    def no_delay_update_kernel(
        # per-neuron dynamic state
        <attr>_ptr, ..., n_neurons,

        # merged sorted spike buffer (one entry per source neuron firing)
        spike_time_ptr,  # [n_spikes] float32 — sorted arrival times
        spike_src_ptr,   # [n_spikes] int32   — source neuron index
        spike_conn_ptr,  # [n_spikes] int32   — incoming connection id (0-based)
        n_spikes,

        # per-connection BLOCKED topology (Topology.blocked(GROUP_SIZE, n_neurons))
        # and per-link edge state, all in the same block-major link order
        conn{i}_src_offsets_ptr,      # [n_blocks * (N_src_i + 1)] int64 — absolute
                                       #   positions into the link arrays, row `pid`
                                       #   is this program's; flattened from 2-D
        conn{i}_target_indices_ptr,   # [M_i] int32         — block-local target index
        conn{i}_{attr}_ptr, ...,      # [M_i] float32       — per-link edge attrs
                                       #   (BlockedTopology.to_blocked of the canonical tensor)
        conn{i}_source_{attr}_ptr,... # [N_src_i] float32   — source population state
                                       #   (only for source.attr names the hooks use)

        t,                     # current kernel timestamp

        # per-program private spike scratch (exclusive to this program instance —
        # no atomic needed to write into it; see IfTransformer._lower_spike)
        local_spike_node_ptr,  # [num_programs * LOCAL_SPIKE_CAP] int32
        local_spike_time_ptr,  # [num_programs * LOCAL_SPIKE_CAP] float32

        # this program's valid count within its own local_spike_{node,time}_ptr
        # slice — written once, at the very end (see the flush step below).
        # local_spike_{node,time,count}_ptr together are exactly a BlockedSpikes
        # (naer2/spike_gather.py): one GROUP of entries per program instance,
        # ready for Population.set_fresh_output()/compact() to consume directly
        # with no separate compaction pass — no atomic needed here at all, since
        # this is a plain scalar store into this program's own exclusive slot.
        local_spike_count_ptr, # [num_programs] int32

        GROUP_SIZE: tl.constexpr,
        SPIKE_BLOCK: tl.constexpr,
        LINK_BLOCK: tl.constexpr,       # links processed per tile within one spike's row
        LOCAL_SPIKE_CAP: tl.constexpr,  # per-program spike scratch capacity; power of 2
    ):

Kernel body — two parts:

    ┌─ Part 1: Event-driven spike loop ─────────────────────────────────────┐
    │  for blk_start in range(0, n_spikes, SPIKE_BLOCK):                    │
    │    load SPIKE_BLOCK entries from unified buffer                        │
    │    for j in tl.static_range(SPIKE_BLOCK):         ← compile-unrolled  │
    │      if blk_mask[j]:                              ← skip padding       │
    │        time = spike_t[j]                                               │
    │        if spike_conn[j] == 0:       ← warp-uniform: same value for     │
    │                                        every lane in this program, so   │
    │                                        this branch is NOT divergent     │
    │          link_start = tl.load(conn0_src_offsets_ptr + spike_src[j])    │
    │          link_end   = tl.load(conn0_src_offsets_ptr + spike_src[j]+1)  │
    │          source_{attr} = tl.load(conn0_source_{attr}_ptr + spike_src[j])│
    │          for link_idx in range(link_start, link_end):                  │
    │            tgt    = target_indices[link_idx]                           │
    │            weight = weight_ptr[link_idx]                               │
    │            is_target = mask & (offs == tgt)                            │
    │            <<< edge forward for conn 0 >>>                             │
    │            <<< receive for conn 0 >>>                                  │
    │            <<< send_section — fire check, after this spike's receive >>>│
    │        elif spike_conn[j] == 1:                                        │
    │          ... conn 1's own pointers, own CSR range, own transpiled AST │
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

The update hook is NOT run per spike delivery — only receive (state change from
this spike) and send (fire check) run per link, in that order.  A node that
needs its own state propagated to the incoming spike's exact time before
applying the delivery must do that itself, explicitly, inside its receive
hook — the framework does not inject update() there.  update() still runs
once in Part 2, propagating from the last spike time to the timestep boundary t.

Each spike's connection is dispatched via an explicit if/elif chain on
spike_conn[j], not by pre-gathering every connection's CSR range and letting
wrong-connection ranges come out empty (an earlier design). spike_conn[j] is
the same value for every lane in the GROUP_SIZE tile — a property of which
spike is being processed, not of which lane's neuron is the target — so
branching on it causes no warp divergence, unlike is_target (which genuinely
varies per lane and is why THAT is handled with tl.where masking instead of
branching). Dispatching explicitly also means each branch uses that
connection's own directly-named pointers and its own transpiled receive/edge-
forward AST with no indirection: different connections can have entirely
different edge types with entirely different code, not just different
topology, so a per-connection branch is not just a topology-routing
convenience but the actual mechanism for running the right code at all. It
does mean total code size scales with (number of connections) × SPIKE_BLOCK,
since tl.static_range unrolls the whole dispatch per spike slot and every
branch's code must be present regardless of which one runs for a given spike
— if that ever matters, SPIKE_BLOCK is the knob to shrink, not the dispatch
shape.

Per-link delays are intentionally ignored here — all deliveries from a source
spike are processed at the same arrival time from the spike buffer.  This is a
known approximation to be revisited when delay handling is implemented.

source.attr is resolved like edge.attr: a name the source population's own
dynamic_attrs contains gets a per-spike gather load (source_{attr}, indexed by
spike_src[j] — one load per connection, not per link, since it doesn't depend
on link_idx); a name that's static on the source population inlines as a
literal instead. Only names a connection's hooks actually reference get a
load — see population.compile()'s use of UsedMemberFinder.

n2.spike(time=...) (see naer2.sends.spike) lowers to a write of (node_id,
timestamp) into this program's own private slice of local_spike_{node,time}_ptr
(pid * LOCAL_SPIKE_CAP + local offset), bumping local_spike_count — a plain
scalar, no atomic needed, since no other program instance ever touches this
slice. See IfTransformer._lower_spike. Exactly once per program instance, at
the very end of the kernel, that count is stored to local_spike_count_ptr[pid]
— a plain scalar store, still no atomic, since pid is this program's own
exclusive slot. Nothing is compacted into a shared buffer here at all: the
three local_spike_* arrays, taken together, are already exactly the shape
Population.set_fresh_output()/compact() expect (BlockedSpikes — one GROUP's
worth of entries per program instance, unsorted, not even unblocked) — see
naer2/spike_gather.py and naer2/population.py. LOCAL_SPIKE_CAP must be sized
generously enough that one program instance's GROUP_SIZE neurons can't
plausibly emit more spikes than that within a single kernel call — there is
no overflow check.

Known gaps:
  - edge backward: deferred, needs reverse CSR.
"""

import ast
import copy

from ..transformers import MaskAliasTransformer, TargetMaskTransformer

_EXPR = lambda src: ast.parse(src, mode="eval").body
_STMTS = lambda src: ast.parse(src).body


class NoDelayUpdateKernelTemplate:
    """
    Assembles the ast.Module for the merged update-receive-send kernel (the
    no-per-link-delay variant -- see the module docstring).

    incoming_with_hooks — list of (Connection, recv_stmts, edge_fwd_stmts,
    source_dynamic_attrs, edge_tensor_attrs) in the same order as population.incoming.
    source_dynamic_attrs is the set of real source-population attr names this
    connection's hooks reference that are dynamic (need a per-spike gather load).
    edge_tensor_attrs is the set of real edge attr names this connection's hooks
    (receive/forward/backward) actually reference that need a per-link tensor
    load — assigned by forward/backward, or given an explicit per-link override
    from outside — as opposed to ones that inline as a literal constant.
    """

    def __init__(self, context, incoming_with_hooks: list):
        self.ctx = context
        self.incoming = incoming_with_hooks  # [(conn, recv_stmts, edge_fwd_stmts, source_dynamic_attrs, edge_tensor_attrs)]

    # ------------------------------------------------------------------ public

    def build(self, update_section: list, send_section: list) -> ast.Module:
        body: list[ast.stmt] = []

        # SPMD setup
        body += _STMTS("""\
pid  = tl.program_id(0)
offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
mask = offs < n_neurons
""")

        # Per-program spike accumulator: local_spike_count is a plain (non-atomic)
        # loop-carried scalar, since local_spike_{node,time}_ptr's pid-indexed
        # slice belongs exclusively to this program instance. See
        # IfTransformer._lower_spike and the flush step at the end of this method.
        body += _STMTS("local_spike_count = tl.cast(0, tl.int32)")

        # load per-neuron dynamic state
        for attr in self.ctx.dynamic_attrs:
            body += _STMTS(f"{attr} = tl.load({attr}_ptr + offs, mask=mask)")

        # Part 1: event-driven spike loop.
        # update_section is NOT run per spike delivery, only in Part 2 below —
        # only send_section runs per link (after receive), so only it needs a
        # deepcopy here (it appears twice: inside the loop and in Part 2).
        body += self._spike_loop(copy.deepcopy(send_section))

        # Part 2: final propagation from last spike time to current kernel time t.
        body += _STMTS("time = t")
        body += update_section
        body += send_section
        # TODO: edge backward (needs reverse CSR)

        # store updated per-neuron state
        for attr in self.ctx.dynamic_attrs:
            body += _STMTS(f"tl.store({attr}_ptr + offs, {attr}, mask=mask)")

        # Publish this program's own spike count — a plain scalar store into
        # this program's exclusive slot, no atomic, no copy: local_spike_node_ptr
        # /local_spike_time_ptr already hold this program's entries exactly
        # where a BlockedSpikes expects them (see the module docstring).
        body += _STMTS("tl.store(local_spike_count_ptr + pid, local_spike_count)")

        module = ast.Module(body=[self._func(body)], type_ignores=[])
        ast.fix_missing_locations(module)
        return module

    # ------------------------------------------------------------------ spike loop

    def _spike_loop(self, send_section: list) -> list[ast.stmt]:
        """Outer loop over the unified spike buffer; inner static_range per spike.

        Each spike slot loads its own three scalars straight from the buffer —
        Triton can't index a tensor by an integer to pull one element out, so a
        block-wide vector load followed by spike_t[j] doesn't compile.
        Each spike is dispatched to its connection via an explicit if/elif
        chain on spike_conn_j — see _conn_dispatch and the module docstring
        for why that's safe (warp-uniform) and necessary (different
        connections can run genuinely different code, not just different CSR
        topology).
        """
        per_spike_body: list[ast.stmt] = [
            *_STMTS("time = tl.load(spike_time_ptr + _s)"),
            *_STMTS("spike_src_j = tl.load(spike_src_ptr + _s)"),
            *_STMTS("spike_conn_j = tl.load(spike_conn_ptr + _s)"),
            *self._conn_dispatch(send_section),
        ]

        inner_for = ast.For(
            target=ast.Name(id="j", ctx=ast.Store()),
            iter=_EXPR("tl.static_range(SPIKE_BLOCK)"),
            body=[
                *_STMTS("_s = blk_start + j"),
                ast.If(test=_EXPR("_s < n_spikes"), body=per_spike_body, orelse=[]),
            ],
            orelse=[],
        )

        return [
            ast.For(
                target=ast.Name(id="blk_start", ctx=ast.Store()),
                iter=_EXPR("range(0, n_spikes, SPIKE_BLOCK)"),
                body=[inner_for],
                orelse=[],
            )
        ]

    # ------------------------------------------------------------------ conn dispatch

    def _conn_dispatch(self, send_section: list) -> list[ast.stmt]:
        """if/elif chain on spike_conn[j], one branch per incoming connection.

        spike_conn[j] is the same value for every lane in this program's
        GROUP_SIZE tile — a property of which spike is being processed, not of
        which lane's neuron is the target — so this branch is warp-uniform and
        causes no divergence (unlike is_target, which is genuinely per-lane and
        is why that's handled with tl.where masking instead of branching).
        Each branch is self-contained: it looks up its own CSR range with its
        own directly-named pointers and runs its own transpiled receive/edge-
        forward AST, so nothing needs to dynamically pick among different
        connections' pointers — the branch itself is the selection.
        """
        def build(idx: int) -> list[ast.stmt]:
            if idx >= len(self.incoming):
                return []
            conn, recv_stmts, edge_fwd_stmts, source_dynamic_attrs, edge_tensor_attrs = (
                self.incoming[idx]
            )
            branch_body = self._csr_branch(
                idx, conn, recv_stmts, edge_fwd_stmts,
                source_dynamic_attrs, edge_tensor_attrs, send_section,
            )
            return [ast.If(
                test=_EXPR(f"spike_conn_j == {idx}"),
                body=branch_body,
                orelse=build(idx + 1),
            )]

        return build(0)

    def _csr_branch(
        self,
        conn_idx: int,
        conn,
        recv_stmts: list,
        edge_fwd_stmts: list,
        source_dynamic_attrs: set,
        edge_tensor_attrs: set,
        send_section: list,
    ) -> list[ast.stmt]:
        """One spike_conn_j == conn_idx branch: this connection's CSR walk.

        The links are laid out block-major (see Topology.blocked): program `pid`
        owns target neurons [pid * GROUP_SIZE, (pid + 1) * GROUP_SIZE), and its
        slice of a source's row contains exactly the links into that range. So
        this program walks only its own links — not the whole row, discarding
        the rest — and ownership is structural: no masking, and targets are
        stored as block-local indices (0..GROUP_SIZE-1).

        That row is walked LINK_BLOCK links at a time. Per tile
        there are two lane spaces, kept apart by naming:
          link lanes   [LINK_BLOCK]  lk_{attr}: one edge value per link — the
                                     edge forward hook runs here, and its
                                     result is stored back per link
          neuron lanes [GROUP_SIZE]  edge_{attr}: what receive sees — this
                                     program's neurons, each carrying the edge
                                     value of the link delivering to it, if any
        The two are bridged by a one-hot compare-and-reduce that turns "these
        links target these neurons" into a per-neuron delivery mask and
        per-neuron edge values, so receive/send run ONCE per tile on the
        register-resident neuron state instead of once per link.

        Structure:
          link_start, link_end = this program's slice of spike_src_j's row  — scalar
          source_{attr} = source_ptr[spike_src_j]   — scalar, once per branch
          for link_tile in range(link_start, link_end, LINK_BLOCK):
            _tgt, lk_{attr}   = tile of target ids / per-link edge attrs
            <<< edge forward >>>       on link lanes
            <<< link stores >>>        per link, masked only for the ragged tile end
            _hit[G, L]                 = neuron i is link l's target
            is_target[G], edge_{attr}[G] = reduce over links
            <<< receive >>>            on neuron lanes, masked by is_target
            <<< send >>>               fire check, after receive

        Assumes a source's targets are unique within its row (no multapses):
        each neuron then receives at most one delivery per spike, which is what
        makes the reduce exact and per-target ordering unchanged. A duplicate
        target within one tile would deliver the sum of its links' values as
        a single delivery.

        update is deliberately NOT run here — only in the final propagation pass
        (Part 2). A node that needs its own state advanced to the spike's exact
        time before this delivery must do so explicitly inside its receive hook.
        """
        i = conn_idx

        # Scalar CSR range for spike_src_j, and the per-spike source state
        # gather — both depend only on spike_src_j, so once per branch.
        topology = conn.fabric.topology
        if topology is None:
            raise ValueError(f"connection {i} has no topology; cannot compile its link walk")
        row_stride = len(topology.source_offsets)  # N_src + 1
        range_loads: list[ast.stmt] = _STMTS(
            # this program's row of the [n_blocks, N_src + 1] offsets table; int64
            # so pid * stride can't overflow for large populations
            f"_row = pid.to(tl.int64) * {row_stride}\n"
            f"link_start = tl.load(conn{i}_src_offsets_ptr + _row + spike_src_j)\n"
            f"link_end = tl.load(conn{i}_src_offsets_ptr + _row + spike_src_j + 1)"
        )
        source_loads: list[ast.stmt] = []
        for attr in source_dynamic_attrs:
            source_loads += _STMTS(
                f"source_{attr} = tl.load(conn{i}_source_{attr}_ptr + spike_src_j)"
            )

        # Link tile: target ids plus every per-link tensor-backed edge attr this
        # connection's hooks reference (edge_tensor_attrs, from
        # population.compile()'s UsedMemberFinder pass). Attrs outside that set
        # inline as literals instead. Loaded on link lanes as lk_{attr}.
        link_loads: list[ast.stmt] = _STMTS(
            "_lk = link_tile + tl.arange(0, LINK_BLOCK)\n"
            "_lk_valid = _lk < link_end\n"
            f"_tgt = tl.load(conn{i}_target_indices_ptr + _lk, mask=_lk_valid, other=-1)"
        )
        for attr in edge_tensor_attrs:
            link_loads += _STMTS(
                f"lk_{attr} = tl.load(conn{i}_{attr}_ptr + _lk, mask=_lk_valid, other=0)"
            )

        # Write back only dynamic attrs (learning rules may have modified them).
        # Every link in this program's range targets one of its own neurons, so
        # no ownership masking is needed — the blocked layout makes it structural.
        link_stores: list[ast.stmt] = []
        for attr in conn.fabric.dynamic_attrs:
            link_stores += _STMTS(
                f"tl.store(conn{i}_{attr}_ptr + _lk, lk_{attr}, mask=_lk_valid)"
            )

        # Densify: which of this program's neurons is a target of which link in
        # the tile, then reduce per-link edge values down to per-neuron ones.
        densify: list[ast.stmt] = _STMTS(
            "_hit = (tl.arange(0, GROUP_SIZE)[:, None] == _tgt[None, :]) & _lk_valid[None, :] & mask[:, None]\n"
            "is_target = tl.sum(_hit.to(tl.int32), axis=1) > 0"
        )
        for attr in edge_tensor_attrs:
            densify += _STMTS(
                f"edge_{attr} = tl.sum(tl.where(_hit, lk_{attr}[None, :], 0), axis=1)"
            )

        # send (fire check) masked to the delivered neurons, run after receive
        # so it sees this spike's effect on their state
        target_send = self._make_target_section(send_section)

        link_body = (
            link_loads
            + list(edge_fwd_stmts)
            + link_stores
            + densify
            + list(recv_stmts)
            + target_send
        )

        link_loop = ast.For(
            target=ast.Name(id="link_tile", ctx=ast.Store()),
            iter=_EXPR("range(link_start, link_end, LINK_BLOCK)"),
            body=link_body or [ast.Pass()],
            orelse=[],
        )
        return range_loads + source_loads + [link_loop]

    # ------------------------------------------------------------------ helpers

    def _make_target_section(self, stmts: list) -> list:
        """Deepcopy stmts and restrict them to the current link's target neuron.

        MaskAliasTransformer rewrites IfTransformer's root mask seed (the bare
        name "mask") to "is_target", so every branch computed within — and
        anything gated by one, including a spike emission — only fires for
        the actual target lane instead of any lane whose own state happens to
        already satisfy the branch's condition. TargetMaskTransformer then
        handles the separate case of top-level (not inside any if) dynamic-attr
        assignments, which IfTransformer never touches.
        """
        stmts = MaskAliasTransformer().transform(copy.deepcopy(stmts))
        return TargetMaskTransformer(set(self.ctx.dynamic_attrs)).transform(stmts)

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

        # per-connection CSR topology + per-link edge state + source population state
        for i, (conn, _, _, source_dynamic_attrs, edge_tensor_attrs) in enumerate(
            self.incoming
        ):
            args += [
                ast.arg(arg=f"conn{i}_src_offsets_ptr"),
                ast.arg(arg=f"conn{i}_target_indices_ptr"),
            ]
            for attr in edge_tensor_attrs:
                args.append(ast.arg(arg=f"conn{i}_{attr}_ptr"))
            for attr in source_dynamic_attrs:
                args.append(ast.arg(arg=f"conn{i}_source_{attr}_ptr"))

        # timestamp and this program's private spike scratch/output (see the
        # flush step at the end of build() and the module docstring: these
        # three arrays together are exactly a BlockedSpikes)
        args += [
            ast.arg(arg="t"),
            ast.arg(arg="local_spike_node_ptr"),
            ast.arg(arg="local_spike_time_ptr"),
            ast.arg(arg="local_spike_count_ptr"),
        ]

        # constexpr tile sizes
        args += [
            ast.arg(arg="GROUP_SIZE", annotation=_EXPR("tl.constexpr")),
            ast.arg(arg="SPIKE_BLOCK", annotation=_EXPR("tl.constexpr")),
            ast.arg(arg="LINK_BLOCK", annotation=_EXPR("tl.constexpr")),
            ast.arg(arg="LOCAL_SPIKE_CAP", annotation=_EXPR("tl.constexpr")),
        ]
        return args

    def _func(self, body: list[ast.stmt]) -> ast.FunctionDef:
        return ast.FunctionDef(
            name="no_delay_update_kernel",
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
