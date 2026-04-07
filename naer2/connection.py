"""
stdp_synapse(1, 2, 3)
    .source(src_pop, mapping={"w": "weight"})
    .target(dst_pop, port="excitatory", mapping={"s": "strength"})
    .mapping({"vel": "velocity"})
    .pattern(AllToAll())
    .connect()
"""

# how should connection syntax look like?
# pop_a >> Edge(1.0) @ AllToAll() >> pop_b
# with customization if needed:
# conn.source.map(to_edge={"x": "attr_x"})
# conn.target.port("input").map(to_edge={"w": "weight"})

# net = node1() @ pattern * edge1() + pattern * edge10() >> node2() @ pattern: edge2() >> node3() @ pattern: edge3() >> node4()
# net.layers = [node1, node2, node3, node4]
# net.connections = [(pattern, edge1), (pattern, edge2), (pattern, edge3)]
# net.layers[0] @ pattern: edge4() >> net.layers[0]

# >> operator on population: needs a partial connection after?
# problem: operator precedence rules apply here...
# => brackets and a one-lineable syntax would still be nice though...

# let's get some order here:
# => a pattern applied to a node can generate a population
# => a pattern applied to a population can generate a connection with no behaviour
# => a pattern applied to a connection overwrites the previous pattern
# => an edge applied to a node generates a connection with no connections

# the above list is incomplete but this line of thinking is leading somewhere:
# A set of rules of what needs to be known at each stage.
# 1.) a connection can only start from a population, a node is an implicit population of size 1
# 2.) a valid connection only needs a source and a target
# 3.) a connection can be started in three ways:
#     a.) applying a pattern
#     b.) applying an edge
#     c.) applying a target population
# 3.) of all the above, only c.) is already a valid connection, with no connections of no behaviour, the other two are incomplete
# 4.) a complete connection may be modified just as an incomplete one
# 5.) lazy check: connection is a

# population.connect(pattern) -> ConnectionBuilder
# population.connect(edge) -> ConnectionBuilder
# population.connect(target) -> Connection

# ConnectionBuilder.source(source, mapping = None) -> ConnectionBuilder
# ConnectionBuilder.pattern(pattern) -> ConnectionBuilder
# ConnectionBuilder.edge(edge) -> ConnectionBuilder
# ConnectionBuilder.target(target, port = None, mapping = None) -> Connection

# Connection has internal ConnectionBuilder
# or better yet: ConnectionBuilder has internal Connection
# Connection implements builder functions on itself
# ConnectionBuilder is just a wrapper around Connection whilst it is incomplete


# MAPPINGS:
# 1.) source -> edge for mapping source attributes into the edge code
# 2.) source -> target for mapping source attributes into the target code
# 3.) target -> edge for mapping target attributes into the edge code


from typing import Callable, Set
from .edges import edge_types
from .fabric import Fabric
from .nodes import node_types
from .patterns import pattern_types
from .population import Population


class ConnectionBuilder:
    def __init__(self, source: Population):
        self.source: Population = source
        self.target: Population = None
        self.fabric: Fabric = None
        self.pattern: Callable = None
        self.target_port: str = None
        self.source_map: dict[str, str | Set[str]] = None
        self.fabric_map: dict[str, str] = None

    def source(self, source: Population):
        if not isinstance(source, Population):
            if source.__class__ in node_types:
                source = Population(source, 1)
            else:
                raise ValueError("source " + source + " is not a Population")
        self.source = source
        return self

    def target(self, target: Population):
        if not isinstance(target, Population):
            if target.__class__ in node_types:
                target = Population(target, 1)
            else:
                raise ValueError("target " + target + " is not a Population")
        self.target = target
        self.source.network += target.network
        conn = Connection(self)
        self.source.network.connections += conn
        return conn

    def fabric(self, fabric: Fabric):
        if not isinstance(fabric, Fabric):
            if fabric.__class__ in edge_types:
                fabric = Fabric(fabric)
            else:
                raise ValueError("fabric " + fabric + " is not a Fabric")
        self.fabric = fabric
        return self

    def pattern(self, pattern: Callable):
        # TODO: verify pattern, if it generates a target, generate the target population
        if pattern.__class__ not in pattern_types:
            raise ValueError("pattern " + pattern + " is not a Pattern")
        self.pattern = pattern
        return self

    def target_port(self, target_port: str):
        if not isinstance(target_port, str):
            raise ValueError("target port must be a string")
        if self.target is None:
            raise ValueError(
                "Can not set target port " + target_port + " if target is not set"
            )
        if target_port not in self.target.ports:
            raise ValueError(
                "target port " + target_port + " does not exist in target" + self.target
            )
        self.target_port = target_port
        return self

    def source_map(self, source_map: dict[str, str | Set[str]]):
        if not isinstance(source_map, dict):
            raise ValueError("source map " + source_map + " must be a dictionary")
        for key, value in source_map.items():
            if not isinstance(key, str):
                raise ValueError("source map key " + key + " must be a string")
            if key not in self.source.attributes:
                raise ValueError(
                    "source map key " + key + " does not exist in source" + self.source
                )
            if not isinstance(value, str) and not isinstance(value, set):
                raise ValueError(
                    "source map value " + value + " must be a string or set"
                )
        self.source_map = source_map
        return self

    def edge_map(self, edge_map: dict[str, str]):
        if not isinstance(edge_map, dict):
            raise ValueError("edge map " + edge_map + " must be a dictionary")
        for key, value in edge_map.items():
            if not isinstance(key, str):
                raise ValueError("edge map key " + key + " must be a string")
            if key not in self.fabric.attributes:
                raise ValueError(
                    "edge map key " + key + " does not exist in edge" + self.fabric
                )
            if not isinstance(value, str):
                raise ValueError("edge map value " + value + " must be a string")
        self.fabric_map = edge_map
        return self

    def connect(self, to):
        try:
            return self.pattern(to)
        except ValueError:
            pass
        try:
            return self.fabric(to)
        except ValueError:
            pass
        try:
            return self.target(to)
        except ValueError:
            pass
        raise ValueError("connection target to (" + str(to) + ") is not a valid target")

    def __rshift__(self, to):
        self.connect(to)
        return self

    def at(self, place):
        # TODO: build out context aware decision-making here, use above functions
        if isinstance(place, str):
            self.target_port(place)
            return
        elif not isinstance(place, dict):
            raise ValueError("place is not a target port or mapping")
        if self.target is None:
            # can only set source + edge ports here
            self.map_source_edge(place)
            return
        if self.fabric is None:
            # can only set source + target ports here
            self.map_source_target(place)
            return
        # This case is the hardest to decide
        # we already have source, edge and target, so we are likely at the end of an @-chain
        # the last @ must therefore be a target-edge mapping or a target port
        # BUT: we could also be dealing with a source-target mapping or even a source-edge mapping, so what gives?
        # example 1: source >> edge >> target @ "excitatory" # port can be decided from type str
        # example 2: source >> edge >> target @ {"w_edge": "w_target"} # "default" assumption, mapping corresponds to edge-target-mapping
        # example 3: source >> edge >> target @ {"strength_source": "strength_target"} # "bad" scenario, mapping corresponds to source-target-mapping
        # example 3.5: source >> target @ {"strength_source": "strength_target"} >> edge # "good" scenario, mapping corresponds to source-target-mapping
        # example 4: source >> edge >> target @ {"strength_source": "strength_edge"} # "bad" scenario, mapping corresponds to source-edge-mapping
        # example 4.5: source >> edge @ {"strength_source": "strength_edge"} >> target # "good" scenario, mapping corresponds to source-edge-mapping
        # RULE 1: after edge and target are both set, a source-edge mapping is not assumed anymore!
        # RULE 2: we therefore need to decide between a source-target mapping and an edge-target mapping

        # NEW APPROACH:
        # only have two maps:
        # source_map and edge_map
        # where source_map maps into both edge and target code
        # to do this, a source_map can have multiple aliases for one variable name
        # e.g. {"strength": ("stdp_trength", "lif_prev_strength")}
        # we enforce that the entries are all tuples of size two and strings, as well as the names being strings
        # we enforce that the first name is used in edge code and the second name in target code
        # for edge_map, we enforce that it is a singular string-tostring map
        # for both maps, we need to enforce that the names actually exist!
        # TODO: implement new map approach and add validation helpers in population and edge
        # => requires some eager compilation at this point, but only do it now, not later on!
        # => we can have the attribute accessor lazyly compile only once we want to know the variable names maybe?
        # => or do a later check only at compilation? => NO NO NO, users need feedback!
        # In general: switch to eager compilation NOW

    def ____matmul__(self, mapping_or_port):
        self.at(mapping_or_port)
        return self


class Connection:
    def __init__(self, builder: ConnectionBuilder):
        self.builder = builder
        self.finished = False

    def finish(self):
        self.finished = True

    def __getattr__(self, name):
        if not self.finished:
            return getattr(self.builder, name)
        elif name in self.builder.__dict__:
            raise AttributeError("Connection already finished")
