from logging import debug

from .network import Network
from .nodes import Node, node_types
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

        self.network: Network = Network(self)

        # Debug
        debug(self.dynamic_attrs)

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

    def __rshift__(self, right):
        from .connection import ConnectionBuilder

        return ConnectionBuilder(self).connect(right)

    def compile(self):  # returns ast.Module
        import ast

        from .transpiler.context import TranspilerContext
        from .transpiler.hook_transpiler import HookTranspiler
        from .transpiler.kernel_builder import KernelBuilder

        ctx = TranspilerContext(
            static_attrs={
                k: v
                for k, v in self.instance.__dict__.items()
                if k not in self.dynamic_attrs
            },
            dynamic_attrs=self.dynamic_attrs,
        )

        kernel_ast = (
            KernelBuilder(ctx)
            .add_update(HookTranspiler(self.node.update, ctx).emit())
            .build()
        )

        return kernel_ast
