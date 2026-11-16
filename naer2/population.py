import ast
import importlib.util
import math
import pathlib
import tempfile
from logging import debug

import torch

from .network import Network
from .nodes import Node, node_types
from .types import Region


def _same_time(a: float, b: float) -> bool:
    """Equality of two run times up to float rounding: 0.2 * 3 and 0.3 + 0.3
    must count as the same round. inf only equals inf."""
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


class Population:
    def __init__(
        self,
        instance,
        shape,
        group_size: int = 32,
        spike_block: int = 32,
        link_block: int = 64,
        local_spike_cap: int = 64,
        carry_cap: int = 64,
        per_src_cap: int = 64,
        timestep: float | None = None,
        delay_resolution: float | None = None,
    ):
        # TODO: actually verify that shape is integer or one of our allowed Regions with the Region constructor call
        self.shape: Region = Region(shape)

        # Triton tile sizes for this population's compiled kernel (see
        # naer2/transpiler/templates/update_kernel.py). Defaults sized for
        # small test networks; override per-population for real scale.
        self.group_size = group_size
        self.spike_block = spike_block
        self.link_block = link_block
        self.local_spike_cap = local_spike_cap

        # Spike-lifecycle capacities (naer2/spike_gather.py): carry_cap caps
        # how many not-yet-due spikes this population can hold across
        # rounds; per_src_cap caps how many due entries any ONE incoming
        # connection can contribute to a merge. Both were hardcoded to 64 at
        # every call site until this population-level override existed --
        # see TODO.md's "compact()/merge() capacity" entry. Must exceed the
        # population's own size for an all-firing-at-once worst case (e.g.
        # every neuron spiking the same round); defaults preserve existing
        # small-test-network behavior exactly.
        self.carry_cap = carry_cap
        self.per_src_cap = per_src_cap

        # This population's own, explicit re-check interval -- independent of
        # (and taking priority over) both has_autonomous_dynamics' "every
        # sub-step" default and _min_incoming_delay()'s event-driven floor.
        # None (default) preserves existing behavior exactly: an autonomous
        # population (real update() dynamics) runs every sub-step, a purely
        # event-driven one only when due by incoming delay. Set this when
        # neither applies -- e.g. a population whose send() fires on its own
        # clock with no incoming connections at all (a Poisson background
        # generator) has no other way to ever become due; see
        # Network._validate_schedulable and TODO.md.
        self.timestep = timestep

        # Width w of the per-link-delay kernel's delay buckets and emission
        # bins (naer2/transpiler/templates/per_link_delay_update_kernel.py),
        # only used when some incoming connection carries a per-link delay
        # tensor. None (default) adopts the network's sub-step; fixed once
        # this population first compiles, together with the bucket count and
        # the per-bucket topology splits (see _delay_layout).
        self.delay_resolution = delay_resolution

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
        # Scheduling (see _next_run_time, _relax_horizon):
        # _horizon: no spike of this population before it is still unknown
        #   (clocked/update populations: where their last run computed to;
        #   event-driven ones: derived every round from their inputs).
        # _bound: no input arrival before it is still unknown.
        # _run_times: pending arrivals of an event-driven population --
        #   intake appends arrival + the connection's minimum delay for a
        #   connection whose source emitted new spikes; later per-link
        #   arrivals are found while the update kernel expands spikes to
        #   their links, which then records the next one.
        self._horizon: float = 0.0
        self._bound: float = math.inf
        self._run_times: list[float] = []
        self._min_delays: dict = {}
        # Intake bookkeeping: _run_id counts this population's runs (0 =
        # never run); _read_run[conn] is the source run intake last read
        # from conn, so a source is read once per run even when rounds
        # repeat at the same time; _summary caches (run id, n spikes,
        # earliest emission time) of this population's own last run.
        self._run_id: int = 0
        self._read_run: dict = {}
        self._summary: tuple[int, int, float] | None = None

        self._delay_layout_cache: tuple[float, int] | None = None
        # compile()'s (the old templates', kept for the AST tests until they go -- TODO.md 1e step 6)
        self._compiled_incoming_meta: list[tuple[set, set]] = []

        # The fused kernel's runtime (naer2/runtime.py), built on the first
        # run: the kernel generated from this population's hooks
        # (fused_kernel()), its dynamic attrs as the runtime's state (the
        # same tensors), its history, ring and plan of incoming
        # connections. main_cap / side_cap / history_depth: the ring's run
        # capacities and the history's depth (None: their defaults).
        self._fused = None
        self._runtime = None
        self._connected = False
        self.main_cap: int | None = None
        self.side_cap: int | None = None
        # the emission history's depth; None: the network's derived one
        # (Network._plan_histories), 3 without a network step
        self.history_depth: int | None = None
        self._derived_depth: int = 3
        # the runtime's receive counters (runtime.received / n_received: summed weight and links per neuron), a
        # development aid -- off by default, a little work per delivered link
        self.count_received: bool = False

        self.network: Network = Network(self)

    @property
    def ports(self):
        return self.node.receive.keys()

    @property
    def has_autonomous_dynamics(self) -> bool:
        """True when the update hook modifies state, so the network clock must drive it."""
        return bool(self.node.update.assigned_members)

    @property
    def n_blocks(self) -> int:
        """Number of program instances this population's kernel launches
        with -- also the number of GROUPs (BlockedSpikes' n_groups)."""
        return -(-len(self.shape) // self.group_size)

    @property
    def uses_per_link_delay_kernel(self) -> bool:
        """True when any incoming connection carries a real per-link delay
        tensor, as opposed to no delay or one uniform scalar for the whole
        connection -- selects which update-kernel variant compile() produces
        (see naer2/transpiler/templates/). This is population-wide: if any
        ONE incoming connection is per-link-delayed, every incoming
        connection's edge forward/backward callbacks are disallowed, not
        just that connection's own -- see per_link_delay_update_kernel.py's
        module docstring for why."""
        return any(
            isinstance(getattr(conn.fabric, "delay", None), torch.Tensor)
            for conn in self.incoming
        )

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

    def _delay_layout(self) -> tuple[float, int]:
        """(w, n_buckets) for the per-link-delay kernel, computed once and
        then fixed: w is delay_resolution if set, else the network's
        sub-step (the last step()'s, or -- when compiling before any
        step() -- the finest positive effective period in the network);
        n_buckets covers the largest incoming delay."""
        if self._delay_layout_cache is None:
            w = self.delay_resolution or self.network.sub_dt
            if w is None:
                positive = [
                    p
                    for p in self.network._compute_effective_periods().values()
                    if 0.0 < p < math.inf
                ]
                if not positive:
                    raise ValueError(
                        f"{type(self.instance).__name__} population needs a delay "
                        f"resolution before any network.step() has run: set "
                        f"population.delay_resolution explicitly."
                    )
                w = min(positive)
            max_bucket = 0
            for conn in self.incoming:
                d = getattr(conn.fabric, "delay", None)
                if isinstance(d, torch.Tensor):
                    max_bucket = max(max_bucket, int((d / w).floor().max()))
                elif d is not None:
                    max_bucket = max(max_bucket, math.floor(float(d) / w))
            self._delay_layout_cache = (w, max_bucket + 1)
        return self._delay_layout_cache

    def _is_due(self, t: float, period: float) -> bool:
        """Whether this population's update kernel needs to run at time t,
        given `period` -- the coarsest interval it can safely go without
        being checked, computed network-wide by
        Network._compute_effective_periods() (0.0 for autonomous with no
        override, self.timestep if set, otherwise derived from incoming
        delay -- recursively, through undelayed connections, from upstream
        populations' own periods). A single `t - last_update >= period`
        check covers every other case uniformly: period 0.0 means "every
        sub-step" (always true), since last_update only ever moves forward.
        period == math.inf ("never due" -- see Network._validate_schedulable,
        which should already have raised before this matters) is special-
        cased explicitly rather than left to fall out of the arithmetic:
        last_update starts at -inf, so t - (-inf) is +inf, and +inf >= +inf
        is True in Python -- the opposite of "never due."

        See Network.step(), which calls this after every population's
        gather phase has already run (gather is unconditional, every
        sub-step, for every population: a population's carried spikes may
        need to surface for a downstream target even on a round it itself
        has nothing new to compute)."""
        if period == math.inf:
            return False
        return t - self.last_update >= period

    def _ensure_on_device(self, device) -> None:
        """Move this population's own per-neuron dynamic-attr tensors onto
        `device` in place (idempotent), so they're the right device both to
        read and to tl.store into from this point on. Called on `self`
        before this population's own kernel launches, and on a connection's
        `_source` before reading its state as a source_{attr} pointer --
        since the "lazy move on first kernel launch" policy must hold
        regardless of which population's kernel happens to run first on a
        given round.

        A dynamic attr build()-time code never explicitly broadcast into a
        per-neuron tensor (e.g. `pop.potential = tensor` is common, but a
        node can have several dynamic attrs and nothing requires every one
        of them to be set that way) is broadcast here instead, lazily, from
        the node instance's own scalar default (the value its __init__ set)
        -- same timing as the device move, since both are "whatever this
        attr needs before the first real kernel read/write touches it"."""
        n = len(self.shape)
        for attr, py_type in self.dynamic_attrs.items():
            if attr not in self.__dict__:
                default = getattr(self.instance, attr)
                dtype = (
                    torch.bool
                    if py_type is bool
                    else torch.int32 if py_type is int else torch.float32
                )
                setattr(self, attr, torch.full((n,), default, dtype=dtype))
            tensor = getattr(self, attr)
            if tensor.device != device:
                setattr(self, attr, tensor.to(device))

    @property
    def _event_driven(self) -> bool:
        return self.timestep is None and not self.has_autonomous_dynamics

    # (The emission contract of the old templates, kept for reference until they go:
    #   Every run OVERWRITES this population's emission buffer; it is valid
    #   until the population's next run, and consumers read it once per run
    #   (intake, via _run_id), in the round it was written.
    #   - node  [n_blocks * local_spike_cap] int32  population-local index
    #   - time  [n_blocks * local_spike_cap] float32 emission time e: >= the
    #           run's window start, possibly past its end (emission delay);
    #           each block's valid entries SORTED ascending by e (sorted in
    #           the kernel before writing out: once at the source, so every
    #           consumer's intake only merges, never sorts; across blocks
    #           there is no order -- no grid-wide barrier in Triton)
    #   - count [n_blocks] int32  valid entries of block b are
    #           [b * cap, b * cap + count[b]); setting it invalidates the rest
    #   - summary [2] float32  (number of spikes, earliest emission time),
    #           written by the kernel itself -- read back once per run by
    #           _emission_summary, which computes it on the host side until
    #           then
    #   - a block emitting more than local_spike_cap spikes must flag
    #     overflow, not write past its slice
    def _emission_summary(self) -> tuple[int, float]:
        """(number of spikes, earliest emission time) of this population's
        last run: its history's header, read back once per run and shared by
        all consumers (a host sync -- TODO.md 1e, D-13)."""
        if self._summary is None or self._summary[0] != self._run_id:
            if self._runtime is None or self._runtime.run_id == 0:
                return 0, math.inf  # no kernel ran: nothing emitted
            n, earliest = self._runtime.history.summary(self._runtime.run_id)
            self._summary = (self._run_id, n, earliest)
        return self._summary[1], self._summary[2]

    def _intake(self, t: float) -> None:
        """End of every round (Network.step): take in what every source that
        ran since this population last read it emitted. Host side: read
        each source once per run, and for an event-driven population
        schedule a run at the source's earliest emission + the connection's
        minimum delay (later per-link arrivals are found by the update
        kernel)."""
        for conn in self.incoming:
            source = conn._source
            if source._run_id == 0 or self._read_run.get(conn) == source._run_id:
                continue
            self._read_run[conn] = source._run_id
            if not self._event_driven:
                continue  # its own clock or its inputs' horizon runs it: no summary needed
            n, earliest = source._emission_summary()  # a host sync (TODO.md 1e, D-13)
            if n:
                self._run_times.append(earliest + self._min_delay(conn))
        # (The spikes themselves: the fused kernel's pull takes them in at
        # this population's next run, straight from the sources' histories.)

    def _check_schedulable(self) -> None:
        """An update() population needs a clock: its own timestep, or
        inputs bounding how far it may integrate."""
        if self.timestep is None and self.has_autonomous_dynamics and not self.incoming:
            raise ValueError(
                f"{type(self.instance).__name__} population has update() dynamics but "
                f"neither an explicit timestep nor incoming connections, so nothing "
                f"determines how far each update may integrate: set population.timestep."
            )

    def _min_delay(self, conn) -> float:
        """Smallest delay on `conn` (0.0 if it has none), cached: delays
        are static and a per-link tensor can be very large. A float32 tensor
        holds ~7 significant digits: its minimum is read back to those (0.1
        stays 0.1, not 0.10000000149 -- which would put this population's
        runs a hair after its sources' and split every round in two)."""
        if conn not in self._min_delays:
            d = getattr(conn.fabric, "delay", None)
            if d is None:
                self._min_delays[conn] = 0.0
            elif isinstance(d, torch.Tensor):
                m = float(d.min())
                self._min_delays[conn] = float(f"{m:.7g}") if d.dtype in (torch.float32, torch.float16,
                                                                          torch.bfloat16) else m
            else:
                self._min_delays[conn] = float(d)
        return self._min_delays[conn]

    def _reset_horizon(self) -> None:
        """Start of a round's horizon fixed point: derived values restart at
        inf and only shrink while relaxing (see Network.step)."""
        if self.incoming:
            self._bound = math.inf
        if self._event_driven and self.last_update != -math.inf:
            self._horizon = math.inf

    def _relax_horizon(self) -> bool:
        """One relaxation step: input bound = min over incoming connections
        of (source horizon + minimum delay); an event-driven population
        emits nothing before its earliest unprocessed arrival, so its own
        horizon is min(bound, earliest pending arrival). True if anything
        changed."""
        bound = min(
            (c._source._horizon + self._min_delay(c) for c in self.incoming),
            default=math.inf,
        )
        horizon = self._horizon
        if self._event_driven and self.last_update != -math.inf:
            horizon = min(bound, min(self._run_times, default=math.inf))
        changed = bound != self._bound or horizon != self._horizon
        self._bound, self._horizon = bound, horizon
        return changed

    def _next_run_time(self) -> float:
        """When this population next wants its update kernel to run, given
        this round's relaxed horizons; math.inf for "not until something
        changes". Never earlier than the network's current time.

          - never run yet: 0.0, for every population
          - explicit timestep: its horizon (last run + timestep)
          - update() dynamics, no timestep: as soon as its input bound is
            past its horizon, i.e. it can integrate further
          - event-driven: its earliest pending arrival, once the input
            bound has reached it"""
        if self.last_update == -math.inf:
            return 0.0
        if self.timestep is not None:
            return self._horizon
        if self.has_autonomous_dynamics:
            if self._bound > self._horizon and not _same_time(self._bound, self._horizon):
                return max(self._horizon, self.network.t)
            return math.inf
        arrival = min(self._run_times, default=math.inf)
        if arrival <= self._bound or _same_time(arrival, self._bound):
            return arrival
        return math.inf

    def _execute_update_kernel(self, t: float, stream=None) -> None:
        """Offered every round's time t by Network.step(); runs only if t is
        this population's own next run time. The run's window ends at
        t + timestep for a clocked population, else at its input bound."""
        if not _same_time(self._next_run_time(), t):
            return
        self.last_update = t
        self._run_id += 1
        window_end = t + self.timestep if self.timestep is not None else self._bound
        self._window = (t, window_end)
        if not self._event_driven:
            self._horizon = window_end
        self._run_times = [
            r for r in self._run_times if r > window_end and not _same_time(r, window_end)
        ]

        self.run(t, window_end, round_id=self.network.round_id, stream=stream)

    def _note_pending_arrivals(self) -> None:
        """An event-driven population runs when an arrival is due: its ring
        knows the next one after a run -- the first slot still holding
        records (later per-link arrivals of spikes already taken in). A run
        at that slot's end is the first that can deliver it. One readback of
        the ring's counts (the host sync event-driven scheduling has anyway,
        TODO.md 1e, D-13) -- after the round's join (Network.step): the run
        is on this population's stream."""
        if self._runtime is None:
            return  # no kernel ran: nothing pending
        ring = self._runtime.ring
        pending = (ring.main_count + ring.side_count).view(ring.n_groups, ring.n_slots).amax(0).tolist()
        for k in range(ring.ring_pos, ring.ring_pos + ring.n_slots):
            if pending[k % ring.n_slots]:
                arrival = (k + 1) * ring.slot_width
                if arrival not in self._run_times:
                    self._run_times.append(arrival)
                return

    # ------------------------------------------------------------------ the fused kernel's runtime

    def fused_kernel(self):
        """This population's FusedKernel, generated from its hooks (naer2/
        transpiler/templates/fused_template.py), once."""
        if self._fused is None:
            from .transpiler.templates.fused_template import FusedKernelTemplate

            self._fused = FusedKernelTemplate(self).build()
        return self._fused

    @property
    def runtime(self):
        """The runtime (naer2/runtime.py), connected to its sources' (built on first use)."""
        self._ensure_runtime()
        return self._runtime

    def _create_runtime(self, device) -> None:
        """The runtime itself, its state this population's dynamic attrs
        (the same tensors: what the kernel writes, the attrs show) -- not
        yet connected."""
        if self._runtime is not None:
            return
        from .runtime import PopulationRuntime

        self._ensure_on_device(device)
        kernel = self.fused_kernel()
        self._runtime = PopulationRuntime(kernel, len(self.shape), group_size=self.group_size,
                                          depth=self.history_depth or self._derived_depth, device=device,
                                          state={a: getattr(self, a) for a in kernel.attrs})

    def _resolution(self, conn) -> float:
        """A connection's delay quantization: its own, else the network's."""
        return conn.resolution if conn.resolution is not None else self.network.resolution

    def _slot_width(self) -> float:
        """The ring's slot width: the finest resolution its incoming
        connections require (TODO.md 1e, D-12; population.delay_resolution
        overrides); without incoming connections the network's."""
        if self.delay_resolution is not None:
            return float(self.delay_resolution)
        return min((self._resolution(c) for c in self.incoming), default=self.network.resolution)

    def _ensure_runtime(self) -> None:
        """Every runtime of the population and its sources created, then
        this one connected: each incoming connection's SimulationTopology
        (Fabric.simulation, at this population's group size and slot width,
        carrying the per-link attr its kernel stages) and its source's
        runtime."""
        device = self.network.device
        self._create_runtime(device)
        if self._connected:
            return
        kernel, w = self._runtime.kernel, self._slot_width()
        sources, topologies = [], []
        for conn in self.incoming:
            conn._source._create_runtime(device)
            sources.append(conn._source._runtime)
            topologies.append(conn.fabric.simulation(
                len(self.shape), self.group_size, device, resolution=self._resolution(conn),
                attrs=[kernel.edge_attr] if kernel.edge_attr is not None else []))
        # the ring covers the longest incoming delay (its bucket mean) and one run's window ahead
        longest = max((float(t.bucket_mean.max()) for t in topologies), default=0.0)
        window = self.timestep if self.timestep is not None else w
        n_slots = math.ceil(longest / w) + math.ceil(window / w) + 3
        caps = {k: v for k, v in (("main_cap", self.main_cap), ("side_cap", self.side_cap)) if v}
        self._runtime.connect(sources, topologies, n_slots, slot_width=w, **caps)
        self._connected = True

    def run(self, t_start: float, t_end: float, deliver_until: float | None = None, round_id: int | None = None,
            stream=None) -> None:
        """One run of the fused kernel over [t_start, t_end) in round
        `round_id`: what the sources emitted in earlier rounds taken in,
        every slot due by deliver_until (default t_end) delivered, update,
        send, emission into the history. The scheduler's entry
        (_execute_update_kernel) and the direct one."""
        self._ensure_runtime()
        self._runtime.count_received = self.count_received
        self._runtime.run(t_start, t_end, deliver_until=deliver_until, round_id=round_id, stream=stream)

    def check(self) -> None:
        """Raise on what the last runs flagged (ring overflow, far-future arrivals, history overrun)."""
        if self._runtime is not None:
            self._runtime.check()

    def emitted(self) -> list[tuple[float, int]]:
        """The last run's spikes, [(e, node)] in flat-list order."""
        return self._runtime.emitted()

    def __setattr__(self, name, value):
        # a dynamic attr assigned once the runtime exists: the runtime's state follows (same tensor)
        runtime = self.__dict__.get("_runtime")
        if runtime is not None and name in runtime.state:
            value = torch.as_tensor(value).to(runtime.device)
            runtime.state[name] = value
        object.__setattr__(self, name, value)

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

        Selects between the two update-kernel variants first
        (uses_per_link_delay_kernel) -- see per_link_delay_update_kernel.py's
        module docstring for why that switch disallows edge callbacks
        population-wide rather than per-connection. Otherwise, pairs each
        incoming connection with its transpiled receive-hook stmts, then
        feeds them — along with the update and send sections — into
        NoDelayUpdateKernelTemplate to produce the final ast.Module.
        """
        import ast

        from .transpiler.analyzers.used_members import ReadMemberFinder, UsedMemberFinder
        from .transpiler.context import TranspilerContext
        from .transpiler.hook_transpiler import HookTranspiler
        from .transpiler.templates.no_delay_update_kernel import (
            NoDelayUpdateKernelTemplate,
        )

        ctx = TranspilerContext(
            static_attrs={
                k: v
                for k, v in self.instance.__dict__.items()
                if k not in self.dynamic_attrs
            },
            dynamic_attrs=self.dynamic_attrs,
        )

        if self.uses_per_link_delay_kernel:
            for conn in self.incoming:
                if conn.fabric.dynamic_attrs:
                    edge_name = type(conn.fabric.instance).__name__
                    raise ValueError(
                        f"{type(self.instance).__name__} population uses a per-link-delay "
                        f"kernel (some incoming connection has a real per-link delay "
                        f"tensor), which disallows edge forward/backward callbacks on "
                        f"every incoming connection -- but edge type '{edge_name}' "
                        f"declares forward/backward-assigned per-link attrs "
                        f"{sorted(conn.fabric.dynamic_attrs)}, which can never actually "
                        f"change under this kernel. Use an edge with no forward/backward "
                        f"assignment and set any needed per-link tensor directly instead, "
                        f"e.g. conn.fabric.weight = tensor(...)."
                    )

        # Send after a delivery only where that delivery can change send's
        # outcome: the connection's receive writes a node attr send reads.
        # Otherwise (e.g. PD14_neuron: receive writes current, send reads
        # potential) the per-delivery send is left out and send runs once,
        # after update -- so a send whose effect doesn't depend on what
        # receive changes (unconditional side effects, or only `time`) runs
        # once per window, not once per delivery.
        send_reads = ReadMemberFinder(self.node.send.ast, "self").run()
        send_after_delivery = []

        # one transpiled receive + edge-forward section per incoming connection
        incoming_with_hooks = []
        for conn in self.incoming:
            port = conn._target_port or "receive"
            hook = self.node.receive.get(port)
            send_after_delivery.append(hook is not None and bool(hook.assigned_members & send_reads))
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
                if (
                    attr not in source_pop.dynamic_attrs
                    and attr not in source_pop.instance.__dict__
                ):
                    raise ValueError(
                        f"source.{attr} referenced but '{attr}' does not exist on {source_pop!r}"
                    )
            source_dynamic_attrs = {
                a for a in real_source_attrs if a in source_pop.dynamic_attrs
            }
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
            real_used_edge_from_receive = {
                edge_rename.get(n, n) for n in used_edge_names
            }
            used_self_forward = (
                UsedMemberFinder(edge_fwd_hook.ast, "self").run()
                if edge_fwd_hook
                else set()
            )
            used_self_backward = (
                UsedMemberFinder(edge_bwd_hook.ast, "self").run()
                if edge_bwd_hook
                else set()
            )
            referenced_edge_attrs = (
                real_used_edge_from_receive | used_self_forward | used_self_backward
            )

            for attr in referenced_edge_attrs:
                if (
                    attr not in conn.fabric.dynamic_attrs
                    and attr not in conn.fabric.static_attrs
                    and attr not in conn.fabric.tensor_attrs
                ):
                    raise ValueError(
                        f"edge.{attr}/self.{attr} referenced but '{attr}' does not exist on {conn.fabric!r}"
                    )
            # Tensor-backed = assigned by forward/backward, or given an explicit
            # per-link override from outside (conn.fabric.state) — same test as
            # Fabric.tensor_attrs, just restricted to names actually referenced here.
            edge_tensor_attrs = {
                a
                for a in referenced_edge_attrs
                if a in conn.fabric.tensor_attrs
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
                (
                    conn,
                    recv_stmts,
                    edge_fwd_stmts,
                    source_dynamic_attrs,
                    edge_tensor_attrs,
                )
            )

        update_section = HookTranspiler(self.node.update, ctx).emit()
        send_section = HookTranspiler(self.node.send, ctx).emit()

        # Which source/edge attrs each connection's compiled branch actually
        # needs a conn{i}_... pointer for -- _execute_update_kernel reads
        # this back to build the real launch args with matching names.
        self._compiled_incoming_meta = [
            (source_dynamic_attrs, edge_tensor_attrs)
            for (
                _,
                _,
                _,
                source_dynamic_attrs,
                edge_tensor_attrs,
            ) in incoming_with_hooks
        ]

        if self.uses_per_link_delay_kernel:
            from .transpiler.templates.per_link_delay_update_kernel import (
                PerLinkDelayUpdateKernelTemplate,
            )

            w, n_buckets = self._delay_layout()
            return PerLinkDelayUpdateKernelTemplate(
                ctx, incoming_with_hooks, n_buckets, w, send_after_delivery
            ).build(update_section, send_section)

        return NoDelayUpdateKernelTemplate(ctx, incoming_with_hooks, send_after_delivery).build(
            update_section, send_section
        )
