import torch

from .edges import edge_types

_INTERNAL = frozenset({
    'instance', 'edge', 'dynamic_attrs', 'static_attrs', 'topology', 'state',
    'perm', 'device_state',
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

    Once the target population first launches, every per-link tensor its
    kernel references moves to the GPU for good (bind_device): it lives in
    device_state, in the kernel's blocked link order, and is the only live
    copy -- the kernel reads it and writes plastic updates back into it.
    Reading fabric.<attr> after that returns it in canonical order (a GPU
    gather, for analysis between steps); assigning re-initializes it.

    Topology is set by ConnectionBuilder.target() after Topology.build() completes.
    Per-link tensors are populated by the user (or a future initializer) afterwards.

    instance=None (no @n2.edge class at all) makes a bare Fabric: no forward/
    backward hooks, no dynamic/static attrs of its own. This is what lets a
    connection skip an edge class entirely for the common "just one or more
    plain per-link tensors, no learning rule" case — see
    Connection.__setattr__, which is the validated, ergonomic way to populate
    one (conn.weight = tensor(...) rather than conn.fabric.weight = ...);
    nothing loads such a tensor unless some hook actually references
    edge.<name>, exactly as for a real edge class's unused attrs.
    """

    def __init__(self, instance=None):
        if instance is not None and instance.__class__ not in edge_types:
            raise ValueError("edge needs to be decorated by the @n2.edge decorator")

        object.__setattr__(self, 'instance', instance)
        object.__setattr__(
            self,
            'edge',
            edge_types[instance.__class__] if instance is not None else {"forward": None, "backward": None},
        )
        object.__setattr__(self, 'topology', None)
        object.__setattr__(self, 'state', {})
        object.__setattr__(self, 'perm', None)
        object.__setattr__(self, 'device_state', {})

        dynamic_attrs: dict[str, type] = {}
        if instance is not None:
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
            k: v for k, v in (instance.__dict__.items() if instance is not None else ())
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
        return set(self.dynamic_attrs) | set(self.state) | set(self.device_state)

    @property
    def constant_attrs(self) -> dict[str, object]:
        """Static attrs with no tensor override — safe to inline as AST literals."""
        return {k: v for k, v in self.static_attrs.items() if k not in self.tensor_attrs}

    # ------------------------------------------------------------------ device residency

    def bind_device(self, perm: torch.Tensor, attrs, device) -> None:
        """Move `attrs` onto `device` in blocked order `perm` (canonical
        link index of each blocked position), once. An attr with no tensor
        set (e.g. a plastic edge's weight left at its scalar default) is
        broadcast from the edge instance's own value."""
        perm = perm.to(device=device, dtype=torch.int64)
        object.__setattr__(self, 'perm', perm)
        for attr in attrs:
            if attr in self.device_state:
                continue
            if attr in self.state:
                canonical = torch.as_tensor(self.state.pop(attr))
            else:
                canonical = torch.full((len(perm),), float(getattr(self.instance, attr)))
            self.device_state[attr] = canonical.to(device=device, dtype=torch.float32)[perm]

    # ------------------------------------------------------------------ state routing

    def __setattr__(self, name, value):
        if name in _INTERNAL:
            object.__setattr__(self, name, value)
        elif name in self.device_state:
            blocked = self.device_state[name]
            blocked.copy_(torch.as_tensor(value).to(blocked.device, blocked.dtype)[self.perm])
        else:
            object.__getattribute__(self, 'state')[name] = value

    def __getattr__(self, name):
        try:
            state = object.__getattribute__(self, 'state')
            device_state = object.__getattribute__(self, 'device_state')
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        if name in device_state:
            blocked = device_state[name]
            canonical = torch.empty_like(blocked)
            canonical[object.__getattribute__(self, 'perm')] = blocked
            return canonical
        if name in state:
            return state[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
