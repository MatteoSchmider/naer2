"""
Connection builder and Connection.

A Connection joins two populations via a Fabric (edge type) and a Pattern
(connectivity rule). The operator shorthand:

    pop_a >> edge_instance @ pattern >> pop_b

evaluates (@ binds tighter than >>) as:

    ConnectionBuilder(pop_a)
        .connect(_PartialConnection(edge, pattern))
        .connect(pop_b)                              # builds Topology eagerly

When the target population is not known in advance, target_from_pattern infers
its size from the pattern's output and creates the population in one step:

    conn, pop_b = (pop_a >> edge @ pattern).target_from_pattern(node_instance)

Delay is set on the finished Connection, either as an attribute or via |:

    conn.delay = delay_tensor        # per-link tensor
    conn = ... >> pop_b | 1.5        # pipe shorthand, homogeneous scalar

MAPPINGS
  source_map  — renames source attributes before edge/target see them
  fabric_map  — renames edge attributes before target sees them
"""

from typing import Callable, Set

import torch

from ._partial import _PartialConnection
from .edges import edge_types
from .fabric import Fabric
from .nodes import node_types
from .patterns import pattern_types
from .population import Population
from .topology import Topology


class ConnectionBuilder:
    def __init__(self, source: Population):
        self._source: Population = source
        self._target: Population = None
        self._fabric: Fabric = None
        self._pattern: Callable = None
        self._target_port: str = None
        self._source_map: dict[str, str | Set[str]] = None
        self._fabric_map: dict[str, str] = None
        self._delay: int | float | torch.Tensor = None

    # ------------------------------------------------------------------ properties

    @property
    def fabric(self) -> Fabric:
        return self._fabric

    @property
    def _topology(self):
        return self._fabric.topology if self._fabric is not None else None

    def _ensure_fabric(self) -> Fabric:
        """A pattern with no edge at all (`A >> pattern >> B`) is valid --
        it just means no forward/backward hooks and no per-link tensor
        other than whatever gets set directly on the connection afterwards
        (see Connection.__setattr__, Fabric(instance=None))."""
        if self._fabric is None:
            self._fabric = Fabric()
        return self._fabric

    # ------------------------------------------------------------------ builder API

    def target(self, target: Population) -> "Connection":
        if not isinstance(target, Population):
            if target.__class__ in node_types:
                if self._pattern is None:
                    self._target = Population(target, 1)
                else:
                    topology, inferred_region = Topology.build(
                        self._pattern, self._source.shape
                    )
                    self._ensure_fabric().topology = topology
                    self._target = Population(target, inferred_region)
            else:
                raise ValueError(
                    f"target {target!r} is not an instance of a node or a Population"
                )
        else:
            self._target = target
            if self._pattern is not None:
                self._ensure_fabric().topology = Topology.build(
                    self._pattern, self._source.shape, target.shape
                )
        if (
            self._target_port is not None
            and self._target_port not in self._target.ports
        ):
            raise ValueError(
                f"target_port '{self._target_port}' not in {self._target!r}"
            )

        self._source.network += self._target.network
        conn = Connection(self)
        self._source.network.connections.add(conn)
        self._target.incoming.append(conn)
        return conn

    def _set_fabric(self, fabric: "Fabric | object") -> "ConnectionBuilder":
        if not isinstance(fabric, Fabric):
            if fabric.__class__ in edge_types:
                fabric = Fabric(fabric)
            else:
                raise ValueError(f"fabric {fabric!r} is not a Fabric")
        self._fabric = fabric
        return self

    def pattern(self, pattern: Callable) -> "ConnectionBuilder":
        if pattern not in pattern_types and not callable(pattern):
            raise ValueError(f"pattern {pattern!r} is not a registered Pattern")
        self._pattern = pattern
        return self

    def target_port(self, target_port: str) -> "ConnectionBuilder":
        if not isinstance(target_port, str):
            raise ValueError("target port must be a string")
        # Port name may be set before the target is known (e.g. in the pre->>
        # edge chain); validated against the target's ports once it is set.
        if self._target is not None and target_port not in self._target.ports:
            raise ValueError(f"target_port '{target_port}' not in {self._target!r}")
        self._target_port = target_port
        return self

    def source_map(self, source_map: dict[str, str | Set[str]]) -> "ConnectionBuilder":
        if not isinstance(source_map, dict):
            raise ValueError("source_map must be a dict")
        self._source_map = {**(self._source_map or {}), **source_map}
        return self

    def fabric_map(self, fabric_map: dict[str, str]) -> "ConnectionBuilder":
        if not isinstance(fabric_map, dict):
            raise ValueError("fabric_map must be a dict")
        self._fabric_map = {**(self._fabric_map or {}), **fabric_map}
        return self

    # ------------------------------------------------------------------ connect dispatch

    def connect(self, to) -> "ConnectionBuilder | Connection":
        if isinstance(to, _PartialConnection):
            self._fabric = (
                Fabric(to.edge) if not isinstance(to.edge, Fabric) else to.edge
            )
            self._pattern = to.pattern
            if to.port is not None:
                self.target_port(to.port)
            if to.fabric_map:
                self.fabric_map(to.fabric_map)
            return self
        if isinstance(to, Population):
            return self.target(to)
        if to.__class__ in node_types:
            return self.target(to)  # target() handles size-1 vs infer
        if to.__class__ in edge_types:
            return self._set_fabric(Fabric(to))
        if to in pattern_types or type(to) in pattern_types or callable(to):
            return self.pattern(to)
        raise ValueError(f"Cannot connect to {to!r}")

    def __rshift__(self, to) -> "ConnectionBuilder | Connection":
        return self.connect(to)

    def at(self, place) -> "ConnectionBuilder":
        if isinstance(place, str):
            self.target_port(place)
            return self
        if not isinstance(place, dict):
            raise ValueError("at() expects a port name (str) or mapping (dict)")
        # Disambiguation is purely by whether the edge (fabric) is known yet:
        # before an edge is set, a mapping renames source attributes; once the
        # edge is set, a mapping renames edge attributes for the target's receive.
        if self._fabric is None:
            self.source_map(place)
        else:
            self.fabric_map(place)
        return self

    def __matmul__(self, mapping_or_port) -> "ConnectionBuilder":
        return self.at(mapping_or_port)

    def delay(self, value: int | float | torch.Tensor) -> "ConnectionBuilder":
        if type(value) not in (int, float):
            if not isinstance(value, torch.Tensor):
                raise ValueError("delay must be an int, float or tensor")
            elif value.shape != self.shape:
                raise ValueError(f"delay tensor must be shape {self.shape}")
        self._delay = value
        return self

    def __or__(self, value) -> "ConnectionBuilder":
        return self.delay(value)

    @property
    def shape(self) -> tuple[int, ...]:
        if self._topology is None:
            raise ValueError("shape is not available before the target is set")
        return self._topology.shape


class Connection:
    """A fully specified connection between two populations."""

    def __init__(self, builder: ConnectionBuilder):
        self.builder = builder

    def __setattr__(self, name, value):
        if name == "builder":
            object.__setattr__(self, name, value)
            return
        if name == "delay":
            # delay is read directly off fabric by the framework itself
            # (Population._min_incoming_delay, uses_per_link_delay_kernel),
            # never via edge.delay inside a hook -- so it's exempt from the
            # "some receive hook must reference this name" check below.
            setattr(self.builder._ensure_fabric(), name, value)
            return
        self._set_edge_attr(name, value)

    def _set_edge_attr(self, name, value) -> None:
        """conn.<name> = tensor -- the validated, ergonomic alternative to
        conn.fabric.<name> = tensor (still available, unchecked), and what
        makes an edge class optional for the common "just a per-link
        tensor, no learning rule" case (see Fabric(instance=None)).

        Only stores it if the target's receive hook at this connection's
        port actually reads edge.<name> (after reversing fabric_map, same
        resolution Population.compile() itself does) -- storing a tensor
        nothing will ever load is almost certainly a mistake, and with no
        edge class there's no other check (a real edge class's dynamic_attrs
        validation) to catch it.
        """
        from .transpiler.analyzers.used_members import UsedMemberFinder

        builder = self.builder
        target = builder._target
        port = builder._target_port or "receive"
        hook = target.node.receive.get(port) if target is not None else None
        if hook is None:
            raise ValueError(
                f"cannot set '{name}': target has no receive hook at port '{port}'"
            )
        used_edge_names = UsedMemberFinder(hook.ast, "edge").run()
        edge_rename = {hook_name: real for real, hook_name in (builder._fabric_map or {}).items()}
        real_used_edge_attrs = {edge_rename.get(n, n) for n in used_edge_names}
        if name not in real_used_edge_attrs:
            raise ValueError(
                f"'{name}' is not used as edge.{name} by {target!r}'s receive hook "
                f"(port '{port}') -- nothing would read this tensor"
            )
        setattr(builder._ensure_fabric(), name, value)

    def delay(self, value):
        self.builder.delay(value)
        return self

    def __or__(self, value):
        self.builder.delay(value)
        return self

    def __matmul__(self, other) -> "Connection":
        # Post-target mapping/port, once the whole chain is already a Connection:
        #   (A >> edge @ pattern >> B) @ {"weight": "w"}
        # The edge is always already set by this point, so a dict here is
        # unambiguously a fabric_map (at()'s own disambiguation confirms this).
        #
        # TODO: this only fires with the explicit parens shown above. Written
        # bare — `A >> edge @ pattern >> B @ {"weight": "w"}` — @ binds tighter
        # than >>, so `B @ {"weight": "w"}` evaluates first, hits
        # Population.__matmul__, and produces a source_map builder for B as a
        # brand-new source instead of a fabric_map on this Connection. A prior
        # design used deliberately different operator binding strengths
        # (mirroring how | already binds looser than >> for delay(), enabling
        # `... >> B | 1.5` to work bare) so post-target @ would just work
        # without parens; we dropped that for the current type-dispatch design,
        # which is more robust to a user not remembering precedence rules
        # (see Population.__matmul__'s TODO). Revisit if bare post-target
        # chaining turns out to matter enough.
        self.builder.at(other)
        return self

    def __getattr__(self, name):
        return getattr(self.builder, name)
