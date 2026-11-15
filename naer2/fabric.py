import torch

from .edges import edge_types

_INTERNAL = frozenset({
    'instance', 'edge', 'dynamic_attrs', 'static_attrs', 'topology', 'state',
    'perm', 'device_state', 'sim', '_sim_key',
})
PLACEHOLDER = "_no_attr"  # the per-link attr a connection gets when its receive reads none (see simulation())


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

    On the fused kernel's path the connection's SimulationTopology (built
    once by simulation(), for its consumer) holds the per-link attrs the
    consumer's kernel reads: from then on they live there only, on the
    device in simulation order, and fabric.<attr> reads / assigns them in
    canonical order through its perm. (bind_device / device_state are the
    old templates' path, until they go.)
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
        object.__setattr__(self, 'sim', None)
        object.__setattr__(self, '_sim_key', None)

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
        sim_attrs = set(self.sim.attrs) - {PLACEHOLDER} if self.sim is not None else set()
        return set(self.dynamic_attrs) | (set(self.state) - {"delay"}) | set(self.device_state) | sim_attrs

    @property
    def constant_attrs(self) -> dict[str, object]:
        """Static attrs with no tensor override — safe to inline as AST literals."""
        return {k: v for k, v in self.static_attrs.items() if k not in self.tensor_attrs}

    # ------------------------------------------------------------------ the simulation topology

    def simulation(self, n_targets: int, group_size: int, device, resolution: float | None = None,
                   attrs: list[str] = ()):
        """The connection's SimulationTopology (naer2/topology.py) for its
        consumer: the canonical topology restructured at the consumer's
        group size, per-link delays quantized at `resolution` (the finest
        its consumer needs, TODO.md 1e D-12), carrying `attrs` -- the
        per-link attrs the consumer's kernel reads, the one it stages first
        (a scalar edge attr broadcast to every link). With none, one
        placeholder attr: the kernel's per-link attr load is unconditional.
        Built once: those attrs then live there only (fabric.<attr> reads
        and assigns them in canonical order); the same call again returns
        it, a different one raises (a connection has one consumer)."""
        from .topology import SimulationTopology

        key = (n_targets, group_size, str(torch.device(device)), resolution, tuple(attrs))
        if self.sim is not None:
            if key != self._sim_key:
                raise ValueError(f"the simulation topology is already built for {self._sim_key}, not {key}")
            return self.sim
        if self.topology is None:
            raise ValueError("the connection has no topology yet")
        n_links = len(self.topology)
        delay = self.state.get("delay")
        if isinstance(delay, torch.Tensor):
            if resolution is None:
                raise ValueError("a per-link delay needs the consumer's slot resolution")
            delay = delay.float()
        values = {}
        for name in attrs or [PLACEHOLDER]:
            if name == PLACEHOLDER:
                values[name] = torch.zeros(n_links)
            elif name in self.state:
                values[name] = torch.as_tensor(self.state[name], dtype=torch.float32).expand(n_links)
            elif self.instance is not None and name in self.instance.__dict__:
                values[name] = torch.full((n_links,), float(getattr(self.instance, name)))
            else:
                raise ValueError(f"per-link attr '{name}' has no value on {self!r}")
        sim = SimulationTopology(self.topology, n_targets, group_size, device, delay=delay,
                                 resolution=resolution if isinstance(delay, torch.Tensor) else None, attrs=values)
        for name in values:
            self.state.pop(name, None)  # it lives in the simulation topology from now on
        object.__setattr__(self, 'sim', sim)
        object.__setattr__(self, '_sim_key', key)
        return sim

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
        elif self.sim is not None and name in self.sim.attrs and name != PLACEHOLDER:
            value = torch.as_tensor(value, dtype=torch.float32)
            if value.shape != (self.sim.n_links,):
                raise ValueError(f"'{name}' must be a tensor of {self.sim.n_links} links, one per link")
            self.sim.write(name, value)
        elif name == "delay" and self.sim is not None:
            raise ValueError("the delay is fixed once the simulation topology is built (it is its delay buckets)")
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
        sim = object.__getattribute__(self, 'sim')
        if sim is not None and name in sim.attrs and name != PLACEHOLDER:
            return sim.read(name)
        if name in device_state:
            blocked = device_state[name]
            canonical = torch.empty_like(blocked)
            canonical[object.__getattribute__(self, 'perm')] = blocked
            return canonical
        if name in state:
            return state[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
