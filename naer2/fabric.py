from .edges import edge_types

_INTERNAL = frozenset({
    'instance', 'edge', 'dynamic_attrs', 'static_attrs', 'topology', 'state'
})


class Fabric:
    """
    A fully-instantiated edge bundle: type info, CSR topology, and per-link state.

    Fabric owns everything that belongs to one directed connection:
      - edge       : hook ASTs and type annotations (from the @n2.edge decorator)
      - topology   : CSR layout (source_offsets [N+1], target_indices [M])
      - state      : per-link tensors, e.g. state['weight'] has shape [M]
      - dynamic_attrs / static_attrs : which edge attributes are per-link vs constant

    Per-link tensors are accessed as plain attributes — fabric.weight, fabric.delay —
    and are routed to fabric.state by __setattr__/__getattr__.  The reserved names
    (topology, state, instance, edge, …) behave as normal attributes.

    Topology is set by ConnectionBuilder.target() after Topology.build() completes.
    Per-link tensors are populated by the user (or a future initializer) afterwards.
    """

    def __init__(self, instance):
        if instance.__class__ not in edge_types:
            raise ValueError("edge needs to be decorated by the @n2.edge decorator")

        object.__setattr__(self, 'instance', instance)
        object.__setattr__(self, 'edge', edge_types[instance.__class__])
        object.__setattr__(self, 'topology', None)
        object.__setattr__(self, 'state', {})

        dynamic_attrs: dict[str, type] = {}
        assigned = (
            self.edge["forward"].assigned_members
            | self.edge["backward"].assigned_members
        )
        for member in assigned:
            if member not in instance.__dict__:
                raise ValueError(f"Edge instance needs to have a member '{member}'")
            dynamic_attrs[member] = instance.__dict__[member].__class__

        object.__setattr__(self, 'dynamic_attrs', dynamic_attrs)
        object.__setattr__(self, 'static_attrs', {
            k: v for k, v in instance.__dict__.items()
            if k not in dynamic_attrs
        })

    # ------------------------------------------------------------------ derived

    @property
    def tensor_attrs(self) -> set[str]:
        """Attrs that need a real per-link tensor: assigned by forward/backward
        (dynamic_attrs), or given an explicit per-link override from outside
        (state) even though nothing in the edge code ever assigns them — e.g.
        a static synapse's weight set as conn.fabric.weight = tensor(...).

        static_attrs is computed once at __init__ from the raw instance dict
        and never updated when a later `fabric.attr = tensor` overrides one of
        those names, so it alone isn't enough to tell "needs a load" apart from
        "inline as a literal constant"; this property is the actual answer.
        """
        return set(self.dynamic_attrs) | set(self.state)

    @property
    def constant_attrs(self) -> dict[str, object]:
        """Static attrs with no tensor override — safe to inline as AST literals."""
        return {k: v for k, v in self.static_attrs.items() if k not in self.state}

    # ------------------------------------------------------------------ state routing

    def __setattr__(self, name, value):
        if name in _INTERNAL:
            object.__setattr__(self, name, value)
        else:
            object.__getattribute__(self, 'state')[name] = value

    def __getattr__(self, name):
        try:
            state = object.__getattribute__(self, 'state')
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        if name in state:
            return state[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
