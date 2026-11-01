import math
from logging import debug

import torch

from .network import Network
from .nodes import Node, node_types
from .spike_gather import BlockedSpikes, CarryBuffer, CompactedSpikes
from .spike_gather import compact as _compact
from .spike_gather import merge as _merge
from .types import Region


class Population:
    def __init__(self, instance, shape):
        # TODO: actually verify that shape is integer or one of our allowed Regions with the Region constructor call
        self.shape: Region = Region(shape)

        # Check that instance's type was decorated and registered
        if instance.__class__ not in node_types.keys():
            raise ValueError("node needs to be decorated by the @naer.node decorator")
        self.node: Node = node_types[instance.__class__]

        # Check that the instance has all dynamic members initialized
        self.dynamic_attrs: dict[str, type] = {}
        assigned_members = (
            self.node.send.assigned_members | self.node.update.assigned_members
        )
        for receive_func in self.node.receive.values():
            assigned_members |= receive_func.assigned_members
        for member in assigned_members:
            if member not in instance.__dict__:
                raise ValueError(f"Node instance needs to have a member {member}")
            else:
                self.dynamic_attrs[member] = instance.__dict__[member].__class__
        self.instance = instance
        self.incoming: list = []  # Connection objects whose target is this population
        self.last_update: float = -math.inf

        # Spike lifecycle (see naer2/spike_gather.py and TODO.md): _carry
        # persists across calls, _fresh/_due are this round's only.
        self._carry: CarryBuffer | None = None
        self._fresh: BlockedSpikes | None = None
        self._due: CompactedSpikes | None = None

        self.network: Network = Network(self)

        # Debug
        # debug(self.dynamic_attrs)

        # need to decide:
        # attributes may be dynamic if they are changed in a connection as well!
        # => only collect ALL attributes here
        # => potentially let connection ADD attributes?
        # => when is the decision made which ones are dynamic and which are static?

        # why should a connection add attributes?
        # => stdp may want to write to a buffer of pre-firing times in a target neuron
        # => this is the basis of our whole speedup in simulation!
        # => so to not make the target neuron code depend on this, we need to be able to add an array to it
        # => the connection needs to add the array to the population from the edge code
        # => edge code needs to be analyzed, together with the mappings to identify which unresolved attributes are left and need to be added

        # STEP 1:
        # this whole issue arose from trying to verify source and edge mappings in connection
        # so for that problem, just provide all static and dynamic attributes
        # in fact, as we see here now, we can only make a final decision on which is which, once we have all connections from and to a population
        # ideally, the connection "adds" these to the appropriate population
        # => we may add to source from edge and target code
        # => we may add to target from edge code

        # focus on verifying mappings for now.
        # a mapping can only mean that the thing it is renaming already exists, otherwise it will simply be added
        # when something is added we can worry about that later on.
        # we need to provide a  way to access all known attributes from the population!
        # we also need a way to check which attributes from source the edge code uses

        # =====> we have the assignedmemberfinder!
        # instead of looking for assignments on members of self, look for assignments on members of source, target, time, etc.!
        # => we can even have a separate ast analyzer which looks for usages overall, not just assignments
        # so we can find which source attributes the edge and target code actually USE and which ones they ASSIGN and also which ones in total are present in either population and the edge
        # overall we do need something like the Population class for Edges too, wherein this analysis can be found, their multiplicity is generated from the Connection class, because it depends upon the pattern
        # so the edge class may have its edge type known and the ast analysis accessed before knowing how many edges there actually are, since the connection may still be incomplete but already trying to verify things
        # overall, since the edge class would be for verification and compilation, just like the Population class is for now,
        # it would make sense to remove any start/stop control from the population class and delegate this exclusively to a network class!
        # since the network class is singly responsible for that, it should consist only of that and not at all of compilation, that is handled by the individual populations
        # the edge code also needs to be integrated in the population code of course, but this may happen at a later stage and it needs information from both the edge and population classes
        # so merging only makes sense by letting the population own the edge at some point.....
        # networks owns populations and connections, but populations own edges but edges are part of connections????

        # STEP 2:
        # JIT-compile only just before running

    @property
    def ports(self):
        return self.node.receive.keys()

    @property
    def has_autonomous_dynamics(self) -> bool:
        """True when the update hook modifies state, so the network clock must drive it."""
        return bool(self.node.update.assigned_members)

    def _min_incoming_delay(self) -> float:
        """Smallest per-link delay across all incoming connections."""
        min_d = math.inf
        for conn in self.incoming:
            d = getattr(conn.fabric, "delay", None)
            if d is None:
                continue
            min_d = min(
                min_d,
                float(d.min().item()) if isinstance(d, torch.Tensor) else float(d),
            )
        return min_d

    def advance(self, t: float):
        """Called by Network.step — run both kernels if an update is due at time t."""
        if self.has_autonomous_dynamics:
            self._execute_gather_kernel(t)
            self._execute_update_kernel(t)
            self.last_update = t
        else:
            min_d = self._min_incoming_delay()
            if min_d < math.inf and t - self.last_update >= min_d:
                self._execute_gather_kernel(t)
                self._execute_update_kernel(t)
                self.last_update = t

    def _execute_gather_kernel(self, t: float):
        """Kernel 1 stub: collect incoming spikes from all incoming connections,
        interleave and sort them by arrival time, producing a merged spike buffer
        (spike_tgt, spike_link per connection) for the update kernel.
        """
        pass

    def _execute_update_kernel(self, t: float):
        """Kernel 2 stub: merged update-receive-send.
        Consumes the spike buffer produced by the gather kernel, applies receive
        hooks per connection, then runs the update and send hooks across all neurons.
        """
        pass

    def set_fresh_output(self, blocked: BlockedSpikes) -> None:
        """Hand off this round's freshly emitted, GROUP-blocked spikes for
        compact() to fold in. This is the real integration seam
        `_execute_update_kernel`'s own flush step will eventually call
        automatically, once it stops globally compacting into one flat
        array and leaves its local buffers blocked instead (see TODO.md) —
        until then, callers (tests, for now) set this directly."""
        self._fresh = blocked

    def compact(self, t_hi: float, carry_cap: int = 64) -> CompactedSpikes:
        """This population's own lifecycle step (naer2/spike_gather.py):
        fold this round's `_fresh` (see set_fresh_output) into `_carry`
        (created on first call, persists across calls on this population),
        split on `t_hi`, store the due entries as `_due` for any target's
        merge_incoming() to read this round, and return them.

        Must run for every population before ANY of them call
        merge_incoming() — a target's merge reads its sources' `_due`,
        which this call just (re)produces; see Network.gather().
        """
        if self._fresh is None:
            raise ValueError(f"{self!r} has no fresh spike output set for this round (see set_fresh_output)")
        if self._carry is None:
            self._carry = CarryBuffer(carry_cap, self._fresh.time.device)
        self._due = _compact(self._fresh, self._carry, t_hi)
        return self._due

    def merge_incoming(self, per_src_cap: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """This population's merge step (naer2/spike_gather.py): combine
        every incoming connection's source population's already-compact()ed
        `_due` (this round) into the sorted (time, src, conn) buffer this
        population's own update kernel would consume. conn ids are each
        connection's index in `self.incoming`, matching compile()'s
        conn{i}_... pointer numbering. Every source must have already run
        compact() this round."""
        sources = []
        for conn in self.incoming:
            src_due = conn._source._due
            if src_due is None:
                raise ValueError(f"{conn._source!r} has not run compact() this round yet")
            sources.append(src_due)
        conns = list(range(len(self.incoming)))
        return _merge(sources, conns, per_src_cap)

    def __rshift__(self, right):
        from .connection import ConnectionBuilder

        return ConnectionBuilder(self).connect(right)

    def __matmul__(self, mapping):
        """A @ {real_name: hook_name} — source_map, before any edge is known.

        Returns an incomplete ConnectionBuilder (no fabric/target yet) meant to
        be finished with >>, e.g. A @ {"potential": "src_pot"} >> edge @ pattern >> B.

        TODO: this claims @ on every Population unconditionally, which is why
        `>> B @ {mapping}` (bare, no parens) can't mean "post-target fabric_map
        on B" — @ binds tighter than >>, so `B @ {mapping}` evaluates first and
        hits this method, treating B as a brand-new source. Post-target mapping
        currently requires explicit parens: `(A >> edge @ pattern >> B) @ {...}`.
        A prior design used deliberately different operator binding strengths
        (mirroring how | already binds looser than >> for delay()) to make bare
        post-target chaining just work; we dropped that in favor of the current
        type-dispatch design since it's more robust to a user not remembering
        precedence rules. If we want bare post-target @ back, revisit that.
        """
        from .connection import ConnectionBuilder

        if not isinstance(mapping, dict):
            raise ValueError("Population @ mapping expects a dict (source_map)")
        return ConnectionBuilder(self).source_map(mapping)

    def compile(self) -> "ast.Module":
        """Compile the merged update-receive-send kernel for this population.

        Pairs each incoming connection with its transpiled receive-hook stmts,
        then feeds them — along with the update and send sections — into
        UpdateKernelTemplate to produce the final ast.Module.
        """
        import ast

        from .transpiler.analyzers.used_members import UsedMemberFinder
        from .transpiler.context import TranspilerContext
        from .transpiler.hook_transpiler import HookTranspiler
        from .transpiler.templates.update_kernel import UpdateKernelTemplate

        ctx = TranspilerContext(
            static_attrs={
                k: v
                for k, v in self.instance.__dict__.items()
                if k not in self.dynamic_attrs
            },
            dynamic_attrs=self.dynamic_attrs,
        )

        # one transpiled receive + edge-forward section per incoming connection
        incoming_with_hooks = []
        for conn in self.incoming:
            port = conn._target_port or "receive"
            hook = self.node.receive.get(port)
            edge_fwd_hook = conn.fabric.edge.get("forward")
            edge_bwd_hook = conn.fabric.edge.get("backward")

            # Which source.attr names (real, post-source_map) do this connection's
            # hooks actually reference? Only those need a per-spike gather load —
            # a source population can have dynamic attrs no connection here uses.
            source_rename = {
                hook_name: real for real, hook_name in (conn._source_map or {}).items()
            }
            used_source_names = set()
            if hook is not None:
                used_source_names |= UsedMemberFinder(hook.ast, "source").run()
            if edge_fwd_hook is not None:
                used_source_names |= UsedMemberFinder(edge_fwd_hook.ast, "source").run()
            real_source_attrs = {source_rename.get(n, n) for n in used_source_names}

            source_pop = conn._source
            for attr in real_source_attrs:
                if attr not in source_pop.dynamic_attrs and attr not in source_pop.instance.__dict__:
                    raise ValueError(
                        f"source.{attr} referenced but '{attr}' does not exist on {source_pop!r}"
                    )
            source_dynamic_attrs = {a for a in real_source_attrs if a in source_pop.dynamic_attrs}
            source_static_attrs = {
                a: source_pop.instance.__dict__[a]
                for a in real_source_attrs
                if a not in source_dynamic_attrs
            }

            # Which edge attrs (real, post-fabric_map) does this connection actually
            # reference — via edge.attr in receive, or self.attr in forward/backward?
            # Fabric.dynamic_attrs (forward/backward's *assigned* members) alone is
            # not a reliable signal here: an attr can be read-only in backward (not
            # yet wired into the kernel, but still worth accounting for now) without
            # ever being assigned anywhere, or referenced only from receive and
            # nowhere in the edge's own code at all — UsedMemberFinder catches both,
            # same treatment as source.attr above.
            edge_rename = {
                hook_name: real for real, hook_name in (conn._fabric_map or {}).items()
            }
            used_edge_names = set()
            if hook is not None:
                used_edge_names |= UsedMemberFinder(hook.ast, "edge").run()
            real_used_edge_from_receive = {edge_rename.get(n, n) for n in used_edge_names}
            used_self_forward = (
                UsedMemberFinder(edge_fwd_hook.ast, "self").run() if edge_fwd_hook else set()
            )
            used_self_backward = (
                UsedMemberFinder(edge_bwd_hook.ast, "self").run() if edge_bwd_hook else set()
            )
            referenced_edge_attrs = (
                real_used_edge_from_receive | used_self_forward | used_self_backward
            )

            for attr in referenced_edge_attrs:
                if attr not in conn.fabric.dynamic_attrs and attr not in conn.fabric.static_attrs:
                    raise ValueError(
                        f"edge.{attr}/self.{attr} referenced but '{attr}' does not exist on {conn.fabric!r}"
                    )
            # Tensor-backed = assigned by forward/backward, or given an explicit
            # per-link override from outside (conn.fabric.state) — same test as
            # Fabric.tensor_attrs, just restricted to names actually referenced here.
            edge_tensor_attrs = {
                a for a in referenced_edge_attrs
                if a in conn.fabric.dynamic_attrs or a in conn.fabric.state
            }
            edge_constant_attrs = {
                a: conn.fabric.static_attrs[a]
                for a in referenced_edge_attrs
                if a not in edge_tensor_attrs
            }

            recv_stmts = (
                HookTranspiler(hook, ctx).emit_for_spike_loop(
                    set(self.dynamic_attrs),
                    fabric_map=conn._fabric_map,
                    source_map=conn._source_map,
                    edge_static_attrs=edge_constant_attrs,
                    source_static_attrs=source_static_attrs,
                )
                if hook
                else []
            )

            # edge forward hook (updates per-link edge state inside the spike loop)
            if edge_fwd_hook is not None and edge_fwd_hook.assigned_members:
                edge_ctx = TranspilerContext(
                    static_attrs=edge_constant_attrs,
                    dynamic_attrs=edge_tensor_attrs,
                )
                edge_fwd_stmts = HookTranspiler(
                    edge_fwd_hook, edge_ctx, self_prefix="lk_"
                ).emit_for_spike_loop(
                    edge_tensor_attrs,
                    source_map=conn._source_map,
                    source_static_attrs=source_static_attrs,
                    mask_target=False,
                    lane_mask="_lk_valid",
                )
            else:
                edge_fwd_stmts = []

            incoming_with_hooks.append(
                (conn, recv_stmts, edge_fwd_stmts, source_dynamic_attrs, edge_tensor_attrs)
            )

        update_section = HookTranspiler(self.node.update, ctx).emit()
        send_section = HookTranspiler(self.node.send, ctx).emit()

        return UpdateKernelTemplate(ctx, incoming_with_hooks).build(
            update_section, send_section
        )
