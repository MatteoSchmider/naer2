from __future__ import annotations
import math
from typing import TYPE_CHECKING, Set

import torch

if TYPE_CHECKING:
    from .connection import Connection
    from .population import Population

from .kernel import Kernel


class Network:
    def __init__(self, population: Population):
        self.populations: Set[Population] = {population}
        self.connections: Set[Connection] = set()
        self.kernels: Set[Kernel] = set()
        self.t: float = 0.0
        # The sub-step the last step() ran at -- populations without an
        # explicit delay_resolution adopt it (see Population._delay_layout).
        self.sub_dt: float | None = None
        # Device every population's state lazily moves onto on its first
        # real kernel launch (Population._ensure_on_device) -- see TODO.md.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def merge(self, other: Network):
        for population in other.populations:
            population.network = self
        self.populations.update(other.populations)
        self.connections.update(other.connections)
        return self

    def __add__(self, other):
        return self.merge(other)

    def __iadd__(self, other):
        return self.merge(other)

    def _compute_effective_periods(self) -> dict[Population, float]:
        """For every population, the coarsest interval it can safely go
        without being checked:

          - its own explicit `timestep` (Population.__init__), if set --
            always wins, never revised below;
          - else 0.0 if it has autonomous dynamics (real update() state) --
            needs checking every sub-step, also never revised;
          - else the minimum, across its incoming connections, of: that
            connection's own delay (scalar or tensor) if it has one, or
            -- if it has NO delay set at all -- the SOURCE population's own
            effective period, recursively. An undelayed connection has no
            timing of its own; it just means "tick whenever my source
            does." A population with no incoming connections and none of
            the above gets `math.inf` ("never due" -- see
            _validate_schedulable, which is what should catch this before
            it matters).

        Fixed-point iteration handles recurrent connectivity (a population
        feeding back into something that, through some chain of undelayed
        connections, feeds back into it): periods only ever shrink here,
        bounded below by 0.0, so this always terminates, and a population
        with an explicit/autonomous period is never revised once set, so
        cycles through those are harmless anchors rather than infinite
        regress.
        """
        periods: dict[Population, float] = {}
        fixed: dict[Population, bool] = {}
        for pop in self.populations:
            if pop.timestep is not None:
                periods[pop] = pop.timestep
                fixed[pop] = True
            elif pop.has_autonomous_dynamics:
                periods[pop] = 0.0
                fixed[pop] = True
            else:
                periods[pop] = math.inf
                fixed[pop] = False

        changed = True
        while changed:
            changed = False
            for pop in self.populations:
                if fixed[pop]:
                    continue
                best = periods[pop]
                for conn in pop.incoming:
                    d = getattr(conn.fabric, "delay", None)
                    if d is None:
                        cand = periods[conn._source]
                    else:
                        cand = float(d.min().item()) if isinstance(d, torch.Tensor) else float(d)
                    if cand < best:
                        best = cand
                if best < periods[pop]:
                    periods[pop] = best
                    changed = True
        return periods

    def _sub_dt(self, t_step: float, periods: dict[Population, float]) -> float:
        """Smallest interval the network must broadcast at: `t_step`
        itself, unless some population's own effective period (see
        _compute_effective_periods) is finer. A period of exactly 0.0
        (autonomous, no override) doesn't constrain this -- such a
        population just runs on whatever cadence other populations
        require."""
        sub_dt = t_step
        for period in periods.values():
            if 0.0 < period < sub_dt:
                sub_dt = period
        return sub_dt

    def _validate_schedulable(self, periods: dict[Population, float]) -> None:
        """Raise clearly, before stepping, for a population that could
        never become due under any t_step (effective period is `math.inf`)
        -- e.g. a Poisson background generator (send() fires on its own
        clock, no incoming connections, update() a no-op) needs `timestep`
        set explicitly, since nothing else could ever make it due."""
        for pop, period in periods.items():
            if period == math.inf:
                raise ValueError(
                    f"{type(pop.instance).__name__} population would never be due: "
                    f"no autonomous update() dynamics, no incoming connection (or only "
                    f"undelayed ones into other never-due populations), and no explicit "
                    f"timestep set. Set population.timestep explicitly if this population "
                    f"needs to be checked on its own clock (e.g. a Poisson background "
                    f"generator driven by send() alone)."
                )

    def gather(self, t_hi: float, carry_cap: int = 64, per_src_cap: int = 64) -> dict:
        """Two-phase spike management across the whole network (see
        naer2/spike_gather.py and TODO.md): every population's compact()
        first — a source can feed multiple targets, so this must run
        exactly once per population, not once per consumer, and every
        source's compaction must finish before any target's merge reads it
        — then merge_incoming() for every population that has incoming
        connections. Populations with no incoming connections have nothing
        to merge and are omitted from the returned dict.

        Every population must have had set_fresh_output() called on it this
        round before this runs. Phase ordering is enforced by Python-level
        sequencing here, not an explicit GPU sync: both phases launch on the
        same (default) CUDA stream, so in-order execution already
        guarantees phase 1 finishes before phase 2's kernels run.
        """
        for pop in self.populations:
            pop.compact(t_hi, carry_cap)
        results = {}
        for pop in self.populations:
            if pop.incoming:
                results[pop] = pop.merge_incoming(per_src_cap)
        return results

    def step(self, t_step: float):
        """Advance the network by t_step.

        Splits t_step into sub-steps when any population's own effective
        period (Population.timestep, autonomous dynamics, or incoming
        delay -- see _compute_effective_periods, which also resolves an
        undelayed connection to its source's own period) is finer than
        t_step, then runs each sub-time as three network-wide phases, same
        ordering as gather() (see its docstring for why phase 1 must finish
        for every population before phase 2 reads any of them) plus a
        third phase for the kernels themselves:

          1. gather  — every population's own compact() step
             (Population._execute_gather_kernel), unconditionally: a
             population not due for its own update this round may still
             have a carried spike surfacing for a downstream target.
          2. merge   — every population with incoming connections combines
             its sources' now-compact()ed output into this round's input
             spike buffer (Population.merge_incoming). If a PREVIOUS
             round's merge result is still sitting unconsumed on
             `_merged` (this population wasn't due yet to run its update
             kernel), it is combine()d with -- not overwritten by -- this
             round's result: a population ticking slower than the
             network's sub_dt must still see everything that became due
             while it was waiting, not just whatever happened to be due on
             the one sub-step it finally runs. The result is stashed on
             Population._merged for phase 3 to consume (and clear).
          3. update  — populations actually due at this sub-time
             (Population._is_due, given this round's precomputed periods)
             run their real compiled kernel (Population.
             _execute_update_kernel), consuming _merged and publishing
             freshly emitted spikes for the NEXT sub-time's phase 1 to
             fold in.

        All three phases launch on the same (default) CUDA stream per
        population, same reasoning as gather(): in-order execution is the
        only sync needed, no explicit barrier.
        """
        from .spike_gather import combine

        periods = self._compute_effective_periods()
        self._validate_schedulable(periods)
        sub_dt = self._sub_dt(t_step, periods)
        self.sub_dt = sub_dt
        t_end = self.t + t_step
        while self.t < t_end - 1e-10:
            self.t = min(self.t + sub_dt, t_end)
            for pop in self.populations:
                pop._execute_gather_kernel(self.t)
            for pop in self.populations:
                if pop.incoming:
                    fresh_merged = pop.merge_incoming(pop.per_src_cap)
                    pop._merged = (
                        combine(pop._merged, fresh_merged)
                        if pop._merged is not None
                        else fresh_merged
                    )
            for pop in self.populations:
                if pop._is_due(self.t, periods[pop]):
                    pop._execute_update_kernel(self.t)
                    pop.last_update = self.t
