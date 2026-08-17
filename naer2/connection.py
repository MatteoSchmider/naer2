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
        self._topology: Topology = None
        self._target_port: str = None
        self._source_map: dict[str, str | Set[str]] = None
        self._fabric_map: dict[str, str] = None
        self._delay: int | float | torch.Tensor = None

    # ------------------------------------------------------------------ builder API

    def target(self, target: Population) -> "Connection":
        if not isinstance(target, Population):
            if target.__class__ in node_types:
                if self._pattern is None:
                    self._target = Population(target, 1)
                else:
                    self._topology, inferred_region = Topology.build(
                        self._pattern, self._source.shape
                    )
                    self._target = Population(target, inferred_region)
            else:
                raise ValueError(
                    f"target {target!r} is not an instance of a node or a Population"
                )
        else:
            self._target = target
            if self._pattern is not None:
                self._topology = Topology.build(
                    self._pattern, self._source.shape, target.shape
                )
        self._source.network += self._target.network
        conn = Connection(self)
        self._source.network.connections.add(conn)
        return conn

    def fabric(self, fabric: Fabric) -> "ConnectionBuilder":
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
        if self._target is None:
            raise ValueError(f"Cannot set target_port before target is set")
        if target_port not in self._target.ports:
            raise ValueError(f"target_port '{target_port}' not in {self._target!r}")
        self._target_port = target_port
        return self

    def source_map(self, source_map: dict[str, str | Set[str]]) -> "ConnectionBuilder":
        if not isinstance(source_map, dict):
            raise ValueError("source_map must be a dict")
        self._source_map = source_map
        return self

    def fabric_map(self, fabric_map: dict[str, str]) -> "ConnectionBuilder":
        if not isinstance(fabric_map, dict):
            raise ValueError("fabric_map must be a dict")
        self._fabric_map = fabric_map
        return self

    # ------------------------------------------------------------------ connect dispatch

    def connect(self, to) -> "ConnectionBuilder | Connection":
        if isinstance(to, _PartialConnection):
            self._fabric = (
                Fabric(to.edge) if not isinstance(to.edge, Fabric) else to.edge
            )
            self._pattern = to.pattern
            return self
        if isinstance(to, Population):
            return self.target(to)
        if to.__class__ in node_types:
            return self.target(Population(to, 1))
        if to.__class__ in edge_types:
            return self.fabric(Fabric(to))
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
        if self._target is None or self._fabric is None:
            self.source_map(place)
            return self
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

    def delay(self, value):
        self.builder.delay(value)
        return self

    def __or__(self, value):
        self.builder.delay(value)
        return self

    def __getattr__(self, name):
        return getattr(self.builder, name)
