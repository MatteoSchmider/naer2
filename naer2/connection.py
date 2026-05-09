"""
Connection builder and Connection.

A Connection joins two populations via a Fabric (edge + its multiplicity) and a
Pattern (connectivity rule). The builder API is chainable and also supports the
operator shorthand:

    pop_a >> edge_instance @ pattern >> pop_b

which evaluates (due to @ > >> precedence) as:

    pop_a >> (edge_instance @ pattern) >> pop_b
     └─ ConnectionBuilder(pop_a)
         .connect(_PartialConnection(edge, pattern))   # sets fabric + pattern
         .connect(pop_b)                               # sets target, returns Connection

MAPPINGS
Two maps can be set on a connection:
  source_map  — renames source attributes before they are visible to edge and
                target code. One source name can map to two aliases, the first
                used in edge code and the second in target code:
                {"strength": ("edge_strength", "target_strength")}
  fabric_map  — renames edge attributes before they are visible to target code:
                {"w": "weight"}
"""

from typing import Callable, Set

from ._partial import _PartialConnection
from .edges import edge_types
from .fabric import Fabric
from .nodes import node_types
from .patterns import pattern_types
from .population import Population


class ConnectionBuilder:
    def __init__(self, source: Population):
        self.source: Population = source
        # Private state — underscore-prefixed so builder methods (target, fabric,
        # pattern …) are not shadowed by same-named instance attributes.
        self._target: Population = None
        self._fabric: Fabric = None
        self._pattern: Callable = None
        self._target_port: str = None
        self._source_map: dict[str, str | Set[str]] = None
        self._fabric_map: dict[str, str] = None

    # ------------------------------------------------------------------ builder API

    def target(self, target: Population) -> "Connection":
        if not isinstance(target, Population):
            if target.__class__ in node_types:
                target = Population(target, 1)
            else:
                raise ValueError(f"target {target!r} is not a Population")
        self._target = target
        self.source.network += target.network
        conn = Connection(self)
        self.source.network.connections.add(conn)
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
            raise ValueError(f"Cannot set target port '{target_port}' before target is set")
        if target_port not in self._target.ports:
            raise ValueError(f"target port '{target_port}' does not exist in {self._target!r}")
        self._target_port = target_port
        return self

    def source_map(self, source_map: dict[str, str | Set[str]]) -> "ConnectionBuilder":
        if not isinstance(source_map, dict):
            raise ValueError("source_map must be a dict")
        self._source_map = source_map
        return self

    def edge_map(self, edge_map: dict[str, str]) -> "ConnectionBuilder":
        if not isinstance(edge_map, dict):
            raise ValueError("edge_map must be a dict")
        self._fabric_map = edge_map
        return self

    # ------------------------------------------------------------------ connect dispatch

    def connect(self, to) -> "ConnectionBuilder | Connection":
        """
        Dispatch-based connect: figures out what `to` is and applies it.

        Accepts:
          _PartialConnection  — unpacks fabric + pattern (result of edge @ pattern)
          Population          — sets as target, finalises the connection
          node class instance — auto-wrapped into a size-1 Population, then target
          edge instance       — wraps in Fabric and sets as fabric
          pattern callable    — sets as pattern
        """
        if isinstance(to, _PartialConnection):
            self._fabric = Fabric(to.edge) if not isinstance(to.edge, Fabric) else to.edge
            self._pattern = to.pattern
            return self
        if isinstance(to, Population):
            return self.target(to)
        if to.__class__ in node_types:
            return self.target(Population(to, 1))
        if to.__class__ in edge_types:
            return self.fabric(Fabric(to))
        if to in pattern_types or callable(to):
            return self.pattern(to)
        raise ValueError(f"Cannot connect to {to!r}")

    def __rshift__(self, to) -> "ConnectionBuilder | Connection":
        return self.connect(to)

    # ------------------------------------------------------------------ at / @

    def at(self, place) -> "ConnectionBuilder":
        """
        Specify port or aliasing for the most recently set endpoint.

        String argument → target port name.
        Dict argument   → source_map or fabric_map depending on context
                          (see MAPPINGS in module docstring).
        """
        if isinstance(place, str):
            self.target_port(place)
            return self
        if not isinstance(place, dict):
            raise ValueError("at() expects a port name (str) or mapping (dict)")
        if self._target is None:
            self.source_map(place)
            return self
        if self._fabric is None:
            self.source_map(place)
            return self
        # Both fabric and target set: treat as fabric_map
        self.edge_map(place)
        return self

    def __matmul__(self, mapping_or_port) -> "ConnectionBuilder":
        return self.at(mapping_or_port)


class Connection:
    """
    A fully or partially specified connection between two populations.

    While not finished, attribute access is forwarded to the builder so the
    connection can continue to be configured after creation.
    """

    def __init__(self, builder: ConnectionBuilder):
        self.builder = builder
        self.finished = False

    def finish(self):
        self.finished = True

    def __getattr__(self, name):
        if not self.finished:
            return getattr(self.builder, name)
        raise AttributeError(f"Connection is finished; cannot access '{name}'")
