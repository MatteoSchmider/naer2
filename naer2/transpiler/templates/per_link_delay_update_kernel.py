"""
Template for the merged update-receive-send Triton kernel -- the PER-LINK-
DELAY variant, intra-timestep transmission delay only (see TODO.md): a
connection whose links carry a real per-link delay tensor, with every delay
value less than the target population's own timestep. Inter-timestep
transmission delay (delay >= timestep) is not implemented -- Population.compile() raises
before this template is ever reached in that case.

Each connection's links are statically split by delay bucket (bucket j
holds delays in [j*w, (j+1)*w), w = timestep / n_buckets; Topology.
blocked(..., delay=..., n_buckets=..., timestep=...), see naer2/topology.py).
The spike loop is a convolution over (emission bin, delay bucket): incoming
spikes are already emission-sorted, so binning their emission times on an
absolute grid of width w makes each bin a contiguous range of the buffer
(bin_start_ptr, computed by the host). Pair (bin i, bucket j) arrives in
[(i+j)*w, (i+j+2)*w), so walking anti-diagonals k = i+j in increasing order
delivers in arrival order up to one bucket width -- two deliveries can only
be out of order across adjacent diagonals, by less than w:

    for k in diagonals:                       time = (k+1)*w  (monotone)
      for j in range(n_buckets):              bucket j, emission bin k-j
        for each spike s in bin k-j:
          <conn dispatch>: walk s's links in bucket j's sub-range

Every delivery on diagonal k uses the same time, (k+1)*w -- the emission-
bin centre plus the bucket centre -- so time never decreases between
deliveries (no negative dt for the model). Every (spike, bucket) pair is
still walked exactly once, and the bucket is a runtime index, so the link
body exists once per connection rather than once per bucket. See TODO.md.

Everything per link -- tiling, densify, receive, send -- is exactly the
no-delay template's.

Edge forward/backward callbacks are disallowed on every incoming
connection of a population using this kernel (checked in Population.
compile(), before this template is built) -- not because bucketing
reshuffles anything (it doesn't: links stay in their static, permanent
positions, so writing back per-link state is exactly as cheap as the
no-delay kernel's), but because that restriction was already shipped
under the (different, since-abandoned) continuous-delay design and
hasn't been revisited for this one yet. See TODO.md.
"""

import ast
import copy

from ..transformers import MaskAliasTransformer, TargetMaskTransformer

_EXPR = lambda src: ast.parse(src, mode="eval").body
_STMTS = lambda src: ast.parse(src).body


class PerLinkDelayUpdateKernelTemplate:
    """
    Assembles the ast.Module for the merged update-receive-send kernel (the
    per-link-delay, intra-timestep transmission delay variant -- see the module
    docstring).

    incoming_with_hooks — same shape as NoDelayUpdateKernelTemplate's:
    [(Connection, recv_stmts, edge_fwd_stmts, source_dynamic_attrs,
    edge_tensor_attrs), ...] in population.incoming order. edge_fwd_stmts
    is expected empty for every connection here (Population.compile()
    raises before building this template otherwise).

    n_buckets/bucket_width: this population's own delay-bucket count and
    bucket width (timestep / n_buckets) -- same for every incoming
    connection, since both are properties of the TARGET population, not
    of any one connection.
    """

    def __init__(self, context, incoming_with_hooks: list, n_buckets: int, bucket_width: float):
        self.ctx = context
        self.incoming = incoming_with_hooks
        self.n_buckets = n_buckets
        self.bucket_width = bucket_width

    # ------------------------------------------------------------------ public

    def build(self, update_section: list, send_section: list) -> ast.Module:
        body: list[ast.stmt] = []

        body += _STMTS("""\
pid  = tl.program_id(0)
offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
mask = offs < n_neurons
""")
        body += _STMTS("local_spike_count = tl.cast(0, tl.int32)")

        for attr in self.ctx.dynamic_attrs:
            body += _STMTS(f"{attr} = tl.load({attr}_ptr + offs, mask=mask)")

        body += self._spike_loop(copy.deepcopy(send_section))

        body += _STMTS("time = t")
        body += update_section
        body += send_section

        for attr in self.ctx.dynamic_attrs:
            body += _STMTS(f"tl.store({attr}_ptr + offs, {attr}, mask=mask)")

        body += _STMTS("tl.store(local_spike_count_ptr + pid, local_spike_count)")

        module = ast.Module(body=[self._func(body)], type_ignores=[])
        ast.fix_missing_locations(module)
        return module

    # ------------------------------------------------------------------ spike loop (unchanged from the no-delay template)

    def _spike_loop(self, send_section: list) -> list[ast.stmt]:
        """The (emission bin, delay bucket) convolution -- see the module
        docstring. bin_start_ptr[m] : bin_start_ptr[m + 1] is emission bin
        first_bin + m's range in the emission-sorted spike buffer. All
        loops are real loops (runtime bounds), nothing is unrolled."""
        per_spike_body: list[ast.stmt] = [
            *_STMTS("spike_src_j = tl.load(spike_src_ptr + _s)"),
            *_STMTS("spike_conn_j = tl.load(spike_conn_ptr + _s)"),
            *self._conn_dispatch(send_section),
        ]

        spike_for = ast.For(
            target=ast.Name(id="_s", ctx=ast.Store()),
            iter=_EXPR("range(_s_start, _s_end)"),
            body=per_spike_body,
            orelse=[],
        )

        bucket_for = ast.For(
            target=ast.Name(id="_bk", ctx=ast.Store()),
            iter=_EXPR(f"range({self.n_buckets})"),
            body=[
                *_STMTS("_m = _k - _bk"),
                ast.If(
                    test=_EXPR("(_m >= 0) & (_m < n_bins)"),
                    body=[
                        *_STMTS(
                            "_s_start = tl.load(bin_start_ptr + _m)\n"
                            "_s_end = tl.load(bin_start_ptr + _m + 1)"
                        ),
                        spike_for,
                    ],
                    orelse=[],
                ),
            ],
            orelse=[],
        )

        return [
            ast.For(
                target=ast.Name(id="_k", ctx=ast.Store()),
                iter=_EXPR(f"range(n_bins + {self.n_buckets - 1})"),
                body=[
                    # diagonal k's centre: emission-bin centre + bucket centre
                    *_STMTS(f"time = (first_bin + _k + 1) * {self.bucket_width!r}"),
                    bucket_for,
                ],
                orelse=[],
            )
        ]

    # ------------------------------------------------------------------ conn dispatch (unchanged from the no-delay template)

    def _conn_dispatch(self, send_section: list) -> list[ast.stmt]:
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

    # ------------------------------------------------------------------ CSR branch -- the one real difference

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
        """One spike_conn_j == conn_idx branch, walking only delay bucket
        `_bk`'s sub-range (a runtime index set by the convolution loop, see
        _spike_loop). The sub-range comes from `conn{i}_src_offsets_ptr`'s
        bucket axis (Topology.blocked(..., n_buckets=...) flattens
        [n_blocks, n_buckets, N_src + 1] row-major, so bucket b's row for
        program `pid` sits at flat offset `(pid * n_buckets + b) * (N_src +
        1)`). Everything else -- link tiling, densify, receive, send -- is
        exactly the no-delay template's.
        """
        i = conn_idx
        topology = conn.fabric.topology
        if topology is None:
            raise ValueError(f"connection {i} has no topology; cannot compile its link walk")
        n_src = len(topology.source_offsets) - 1
        row_stride = n_src + 1

        source_loads: list[ast.stmt] = []
        for attr in source_dynamic_attrs:
            source_loads += _STMTS(
                f"source_{attr} = tl.load(conn{i}_source_{attr}_ptr + spike_src_j)"
            )

        stmts: list[ast.stmt] = [
            *source_loads,
            *_STMTS(
                f"_row = (pid.to(tl.int64) * {self.n_buckets} + _bk) * {row_stride}\n"
                f"link_start = tl.load(conn{i}_src_offsets_ptr + _row + spike_src_j)\n"
                f"link_end = tl.load(conn{i}_src_offsets_ptr + _row + spike_src_j + 1)"
            ),
        ]

        link_loads: list[ast.stmt] = _STMTS(
            "_lk = link_tile + tl.arange(0, LINK_BLOCK)\n"
            "_lk_valid = _lk < link_end\n"
            f"_tgt = tl.load(conn{i}_target_indices_ptr + _lk, mask=_lk_valid, other=-1)"
        )
        for attr in edge_tensor_attrs:
            link_loads += _STMTS(
                f"lk_{attr} = tl.load(conn{i}_{attr}_ptr + _lk, mask=_lk_valid, other=0)"
            )

        link_stores: list[ast.stmt] = []
        for attr in conn.fabric.dynamic_attrs:
            link_stores += _STMTS(
                f"tl.store(conn{i}_{attr}_ptr + _lk, lk_{attr}, mask=_lk_valid)"
            )

        # Densify: tl.histogram (is_target) + scratch-buffer atomic
        # scatter-add (edge_{attr}) instead of a one-hot compare-and-reduce
        # -- see no_delay_update_kernel.py's _csr_branch docstring and
        # TODO.md for why. scratch_ptr is reused across every tile and attr
        # in this kernel call -- each use is self-contained (zero, scatter,
        # barrier, load).
        densify: list[ast.stmt] = _STMTS(
            "is_target = (tl.histogram(_tgt, GROUP_SIZE, mask=_lk_valid) > 0) & mask"
        )
        for attr in edge_tensor_attrs:
            densify += _STMTS(
                f"tl.store(scratch_ptr + offs, 0.0, mask=mask)\n"
                f"tl.debug_barrier()\n"
                f"tl.atomic_add(scratch_ptr + pid * GROUP_SIZE + _tgt, lk_{attr}, mask=_lk_valid, scope='cta')\n"
                f"tl.debug_barrier()\n"
                f"edge_{attr} = tl.load(scratch_ptr + offs)"
            )

        target_send = self._make_target_section(send_section)
        link_body = (
            link_loads
            + list(copy.deepcopy(edge_fwd_stmts))
            + link_stores
            + densify
            + list(copy.deepcopy(recv_stmts))
            + target_send
        )
        stmts.append(ast.For(
            target=ast.Name(id="link_tile", ctx=ast.Store()),
            iter=_EXPR("range(link_start, link_end, LINK_BLOCK)"),
            body=link_body or [ast.Pass()],
            orelse=[],
        ))
        return stmts

    # ------------------------------------------------------------------ helpers (unchanged from the no-delay template)

    def _make_target_section(self, stmts: list) -> list:
        stmts = MaskAliasTransformer().transform(copy.deepcopy(stmts))
        return TargetMaskTransformer(set(self.ctx.dynamic_attrs)).transform(stmts)

    # ------------------------------------------------------------------ signature (unchanged from the no-delay template)

    def _signature_args(self) -> list[ast.arg]:
        args: list[ast.arg] = []

        for attr in self.ctx.dynamic_attrs:
            args.append(ast.arg(arg=f"{attr}_ptr"))
        args.append(ast.arg(arg="n_neurons"))

        # spike_time_ptr isn't read: the convolution only needs the
        # emission-bin boundaries (bin_start_ptr/first_bin/n_bins, computed
        # by the host from the sorted times), and delivery time comes from
        # the diagonal, not the spike.
        args += [
            ast.arg(arg="spike_src_ptr"),
            ast.arg(arg="spike_conn_ptr"),
            ast.arg(arg="bin_start_ptr"),
            ast.arg(arg="first_bin"),
            ast.arg(arg="n_bins"),
        ]

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

        args += [
            ast.arg(arg="t"),
            ast.arg(arg="local_spike_node_ptr"),
            ast.arg(arg="local_spike_time_ptr"),
            ast.arg(arg="local_spike_count_ptr"),
            # per-program [GROUP_SIZE] scratch, reused by every densify step
            # (every tile/attr) in this kernel call -- see _csr_branch.
            ast.arg(arg="scratch_ptr"),
        ]

        args += [
            ast.arg(arg="GROUP_SIZE", annotation=_EXPR("tl.constexpr")),
            ast.arg(arg="LINK_BLOCK", annotation=_EXPR("tl.constexpr")),
            ast.arg(arg="LOCAL_SPIKE_CAP", annotation=_EXPR("tl.constexpr")),
        ]
        return args

    def _func(self, body: list[ast.stmt]) -> ast.FunctionDef:
        return ast.FunctionDef(
            name="per_link_delay_update_kernel",
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
