import itertools
import math

import torch

from .network import Network
from .nodes import Node, node_types
from .types import Region


_SEEDS = itertools.count(1)  # (Population.seed: the populations numbered as they are made)


class Population:
    def __init__(
        self,
        instance,
        shape,
        group_size: int = 32,
        timestep: float | None = None,
        delay_resolution: float | None = None,
    ):
        # TODO: actually verify that shape is integer or one of our allowed Regions with the Region constructor call
        self.shape: Region = Region(shape)

        # neurons per program of the fused kernel (one group: its ring, its share of every link)
        self.group_size = group_size

        # This population's own clock: a run every timestep (its window
        # [t, t + timestep)). None (default): an update() population runs as
        # far as its inputs' horizon allows, an event-driven one when an
        # arrival is due (naer2/sim/scheduler.py: PopSchedule.next_run_time); an update() population without
        # incoming connections (a source) every smallest delay of its outgoing
        # ones (_own_timestep).
        self.timestep = timestep

        # The ring's slot width; None (default): the finest resolution of the
        # incoming connections (see _slot_width).
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
        # (its scheduling -- last run, horizon, pending run times: the network's, naer2/sim/scheduler.py)
        self._min_delays: dict = {}
        self._max_delays: dict = {}
        # (its last run's emission summary, read back once per run: _emission_summary)
        self._summary: tuple[int, int, float] | None = None

        # The fused kernel's runtime (naer2/sim/runtime.py), built on the first
        # run: the kernel generated from this population's hooks
        # (fused_kernel()), its dynamic attrs as the runtime's state (the
        # same tensors), its history, ring and plan of incoming
        # connections.
        # The ring (naer2/ring.py): every group's runs in one arena of
        # ring_block-record blocks, starting at ring_in_flight of the group's
        # incoming links (the largest group's) + per slot its reserved blocks
        # and a partly used one -- by default the reserves alone: the
        # runtime sizes the arenas from what the runs need (naer2/sim/sizing.py;
        # PD14: quiet activity needs little more than the reserves, the
        # thalamic volley ~4x that).
        self._fused = None
        self._runtime = None
        self._connected = False
        self.ring_block: int = 256
        self.ring_in_flight: float = 0.0
        # the emission history's depth; None: the network's derived one
        # (Scheduler.history_depth), 3 without a network step
        self.history_depth: int | None = None
        self._derived_depth: int = 3
        # the spikes one run can emit (its history's cap); None: one per neuron -- more where neurons fire inside
        # delivery, any number of times a run (D-3); past it the run's spikes are lost and check() raises
        self.history_cap: int | None = None
        # the runtime's receive counters (runtime.received / n_received: summed weight and links per neuron), a
        # development aid -- off by default, a little work per delivered link
        self.count_received: bool = False
        # receive's time where the incoming delays vary per link (D-1, BRUNEL.md): every link of a ring slot at
        # one time, its "center" (statistically unbiased) or its "end" (causal: never before an actual arrival)
        self.slot_time: str = "center"
        # random.* draws from this population's own stream (RandomLoweringTransformer): distinct by default
        # (populations numbered as they are made: the same network built the same way draws the same numbers),
        # the same stream where set equal
        self.seed: int = next(_SEEDS)

        self.network: Network = Network(self)

    @property
    def ports(self):
        return self.node.receive.keys()

    @property
    def has_autonomous_dynamics(self) -> bool:
        """True when the update hook modifies state, so the network clock must drive it."""
        return bool(self.node.update.assigned_members)

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

    def _own_timestep(self) -> float | None:
        """The population's clock: its explicit timestep; else, for update() dynamics without incoming
        connections (a source: e.g. a Poisson generator), the smallest delay of its outgoing connections
        (BRUNEL.md, step 4) -- a run [t, t + D) emits nothing a consumer receives before t + D, so no consumer
        ever waits on a run of it it couldn't have had (and the longest window that holds for). None: no clock
        of its own (event-driven, an update() population its inputs bound, a source without outgoing
        connections or with a zero delay)."""
        if self.timestep is not None:
            return self.timestep
        if not (self.has_autonomous_dynamics and not self.incoming):
            return None
        d = min((pop._min_delay(c) for pop in self.network.populations for c in pop.incoming if c._source is self),
                default=0.0)
        return d if d > 0 else None

    def _max_window(self) -> float:
        """The longest window a run of this population can have: its clock (_own_timestep); else at most, over
        its incoming connections, a source's longest window + the connection's minimum delay (its window ends at
        its input bound, min over the sources' horizons + their delays: a horizon no further than a run of the
        source past its start) -- a fixed point over the network (cycles bounded by any clocked path); inf
        without one."""
        pops = list(self.network.populations)
        clocks = {p: p._own_timestep() for p in pops}
        W = {p: clocks[p] if clocks[p] is not None else math.inf for p in pops}
        for _ in range(len(pops) + 1):
            changed = False
            for p in pops:
                if clocks[p] is not None or not p.incoming:
                    continue
                v = min(W.get(c._source, math.inf) + p._min_delay(c) for c in p.incoming)
                if v < W[p]:
                    W[p], changed = v, True
            if not changed:
                break
        return W[self]

    def _ring_window(self) -> float:
        """The window the ring covers ahead of a run's start: its own clock (else the slot width), and every
        source's run's (an intake can take in a whole run, its spikes spanning its window): a clocked source's
        clock (exact), an event-driven one's longest window (_max_window). LIMITATION: an update() source
        without a clock (bounded by its inputs, e.g. PD14's neurons) counts as one slot width -- in lockstep
        with a clock its runs are (the fixed point's bound would be loose there: 2-3x PD14's 0.1)."""
        w = self._slot_width()
        own = self._own_timestep()
        windows = [own if own is not None else w]
        for c in self.incoming:
            src = c._source
            if src._own_timestep() is not None:
                windows.append(src._own_timestep())
            elif src._event_driven and math.isfinite(src._max_window()):
                windows.append(src._max_window())
        return max(windows)

    @property
    def last_update(self) -> float:
        """The start of this population's last run as the network scheduled it (-inf: none yet)."""
        return self.network.scheduler.state(self).last_update

    # ---- the device's feedback to the scheduler (naer2/sim/scheduler.py): readbacks, host syncs (TODO.md, D-13)

    def _emission_summary(self) -> tuple[int, float]:
        """(number of spikes, earliest emission time) of this population's
        last run: its history's header, read back once per run and shared by
        all consumers."""
        rt = self._runtime
        if rt is None or rt.run_id == 0:
            return 0, math.inf  # no kernel ran: nothing emitted
        if self._summary is None or self._summary[0] != rt.run_id:
            rt.settle()  # (a stopped run rerun first)
            self._summary = (rt.run_id, *rt.history.summary(rt.run_id))
        return self._summary[1], self._summary[2]

    def _pending_arrival(self) -> tuple[float, float] | None:
        """An event-driven population runs when an arrival is due: its ring
        knows the next one after a run -- the first slot still holding
        records (later per-link arrivals of spikes already taken in). A run
        at that slot's end is the first that can deliver it: (that run time,
        its earliest spike -- the slot's start); a queue population's, exact
        (its queues' next arrival, from the run's event state: fused.py
        _ev_close). None: nothing pending. After the round's join
        (Network.step): the run is on this population's stream."""
        if self._runtime is None:
            return None  # no kernel ran: nothing pending
        self._runtime.settle()  # (a stopped run rerun first)
        ring = self._runtime.ring
        if self._runtime.queues:
            nxt = self._runtime.next_arrival()
            return (nxt, nxt) if math.isfinite(nxt) else None
        pending = ring.run_count.view(ring.n_groups, ring.n_slots).amax(0).tolist()  # (per slot, over the groups)
        ring.first_pending = math.inf  # (none: the ring is empty -- its next run starts from its own start)
        for k in range(ring.ring_pos, ring.ring_pos + ring.n_slots):
            if pending[k % ring.n_slots]:
                ring.first_pending = k
                return (k + 1) * ring.slot_width, k * ring.slot_width  # (a run at the slot's end)
        return None

    def _max_delay(self, conn) -> float:
        """Largest delay on `conn` (0.0 if it has none), cached (as _min_delay)."""
        if conn not in self._max_delays:
            d = getattr(conn.fabric, "delay", None)
            if d is None:
                self._max_delays[conn] = 0.0
            elif isinstance(d, torch.Tensor):
                m = float(d.max())
                self._max_delays[conn] = float(f"{m:.7g}") if d.dtype in (torch.float32, torch.float16,
                                                                          torch.bfloat16) else m
            else:
                self._max_delays[conn] = float(d)
        return self._max_delays[conn]

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
        """The runtime (naer2/sim/runtime.py), connected to its sources' (built on first use)."""
        self._ensure_runtime()
        return self._runtime

    def _create_runtime(self, device) -> None:
        """The runtime itself, its state this population's dynamic attrs
        (the same tensors: what the kernel writes, the attrs show) -- not
        yet connected."""
        if self._runtime is not None:
            return
        from .sim.runtime import PopulationRuntime

        self._ensure_on_device(device)
        kernel = self.fused_kernel()
        self._runtime = PopulationRuntime(kernel, len(self.shape), group_size=self.group_size,
                                          depth=self.history_depth or self._derived_depth, device=device,
                                          state={a: getattr(self, a) for a in kernel.attrs},
                                          history_cap=self.history_cap)
        self._runtime.seed = self.seed

    def _resolution(self, conn) -> float:
        """A connection's delay quantization: its own, else the network's."""
        return conn.resolution if conn.resolution is not None else self.network.resolution

    def _homogeneous_delay(self, conn) -> float | None:
        """A connection's delay where it has one value for every link (a number, or a tensor of one value), read
        back as _min_delay does; None where it varies per link."""
        d = getattr(conn.fabric, "delay", None)
        if d is None or not isinstance(d, torch.Tensor):
            return float(d) if d is not None else 0.0
        if d.numel() and bool((d == d.reshape(-1)[0]).all()):
            return self._min_delay(conn)
        return None

    def _delivery_time(self) -> tuple[float, float, float]:
        """receive's time (T_A, T_C, T_B: T_A e + T_C k + T_B for a record emitted at e, delivered from slot k;
        D-1, BRUNEL.md, fused.py _deliver_link) by the incoming delays: one homogeneous delay D over every
        incoming link -- exact, e + D (emission order is arrival order); otherwise the slot's center or end
        (slot_time), every link of a slot at one time. (Queues: each connection's delay added by delivery
        itself, exact -- these constants unused.)"""
        delays = [self._homogeneous_delay(c) for c in self.incoming]
        if None not in delays and len(set(delays)) <= 1:
            return 1.0, 0.0, delays[0] if delays else 0.0
        if self.slot_time not in ("center", "end"):
            raise ValueError(f"slot_time is 'center' or 'end', not {self.slot_time!r}")
        w = self._slot_width()
        return 0.0, w, (0.5 if self.slot_time == "center" else 1.0) * w

    def _queues(self) -> bool:
        """Every incoming connection one delay value (any values): a queue per connection, exact -- else the slotted
        ring (any per-link delays: a slot's time). SPIKES.md. (A queue's arrival, e + D_c, its delivery computes
        itself: _delivery_time's constants stay a slotted ring's.)"""
        return bool(self.incoming) and all(self._homogeneous_delay(c) is not None for c in self.incoming)

    def _slot_width(self) -> float:
        """The ring's slot width: the finest resolution its incoming
        connections require (TODO.md, D-12; population.delay_resolution
        overrides); without incoming connections the network's."""
        if self.delay_resolution is not None:
            return float(self.delay_resolution)
        return min((self._resolution(c) for c in self.incoming), default=self.network.resolution)

    def _ring_arena(self, topologies, n_slots: int) -> tuple[int, int]:
        """(block, blocks per group) of the ring's arenas (see __init__)."""
        n_groups = -(-len(self.shape) // self.group_size)
        links = sum((t.links_per_group() for t in topologies), torch.zeros(n_groups, dtype=torch.int64))
        from .ring import RESERVE

        records = math.ceil(self.ring_in_flight * int(links.max())) if len(links) else 0
        # + per slot its reserved blocks and one partly used one (see naer2/ring.py)
        return self.ring_block, -(-records // self.ring_block) + n_slots * (RESERVE + 1) + 1

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
        # the ring covers the longest incoming delay (its bucket mean) and one run's window ahead -- its own, or a
        # source's, the longer: one intake can take in a source's run whose spikes span its window, so arrivals
        # reach up to a source window + the longest delay past this run's start (_max_window)
        longest = max((float(t.bucket_mean.max()) for t in topologies), default=0.0)
        n_slots = math.ceil(longest / w) + math.ceil(self._ring_window() / w - 1e-9) + 3
        min_delays = [self._min_delay(c) for c in self.incoming]
        if self._queues():  # (a queue per connection: one cell each, no slots -- SPIKES.md)
            self._runtime.connect(sources, topologies, len(self.incoming), slot_width=w,
                                  arena=self._ring_arena(topologies, len(self.incoming)), min_delays=min_delays,
                                  queues=True)
        else:
            self._runtime.connect(sources, topologies, n_slots, slot_width=w,
                                  arena=self._ring_arena(topologies, n_slots), min_delays=min_delays)
        self._connected = True

    def run(self, t_start: float, t_end: float, deliver_until: float | None = None, round_id: int | None = None,
            stream=None) -> None:
        """One run of the fused kernel over [t_start, t_end) in round
        `round_id`: what the sources emitted in earlier rounds taken in,
        every slot due by deliver_until (default t_end) delivered, update,
        send, emission into the history -- the direct one (a stepped round's:
        naer2/sim/executor.py)."""
        self._prepared_runtime().run(t_start, t_end, deliver_until=deliver_until, round_id=round_id, stream=stream)

    def _prepared_runtime(self):
        """The runtime, built and connected, its settings this population's (before a run)."""
        self._ensure_runtime()
        self._runtime.count_received = self.count_received
        return self._runtime

    def check(self) -> None:
        """Raise on what the last runs flagged (ring overflow, far-future arrivals, history overrun)."""
        if self._runtime is not None:
            self._runtime.check()

    def emitted(self) -> list[tuple[float, int]]:
        """The last run's spikes, [(e, node)] in flat-list order."""
        return self._runtime.emitted()

    def __setattr__(self, name, value):
        if name[0] == "_":  # the internal bookkeeping (the scheduler sets these every round): never a dynamic attr
            object.__setattr__(self, name, value)
            return
        # a dynamic attr assigned once the runtime exists: the runtime's state follows (same tensor)
        runtime = self.__dict__.get("_runtime")
        if runtime is not None and name == "seed":
            runtime.seed = int(value)
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

        This claims @ on every Population unconditionally, which is why
        `>> B @ {mapping}` (bare, no parens) can't mean "post-target fabric_map
        on B" — @ binds tighter than >>, so `B @ {mapping}` evaluates first and
        hits this method, treating B as a brand-new source (the chain then
        raises, ConnectionBuilder.connect, naming the fix). Post-target mapping
        requires explicit parens: `(A >> edge @ pattern >> B) @ {...}`.
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
