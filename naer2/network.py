from __future__ import annotations
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

    def _sub_dt(self, t_step: float) -> float:
        """Smallest interval the network must broadcast at.

        For populations with autonomous dynamics the network clock is always
        needed, so they don't constrain sub_dt.  For purely input-driven
        populations the minimum incoming delay is the coarsest interval at
        which they can safely skip updates, so it sets the floor.
        """
        sub_dt = t_step
        for pop in self.populations:
            if pop.has_autonomous_dynamics:
                continue
            d = pop._min_incoming_delay()
            if 0.0 < d < sub_dt:
                sub_dt = d
        return sub_dt

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

        Splits t_step into sub-steps when any input-only population has a
        minimum incoming delay smaller than t_step, then runs each sub-time
        as three network-wide phases, same ordering as gather() (see its
        docstring for why phase 1 must finish for every population before
        phase 2 reads any of them) plus a third phase for the kernels
        themselves:

          1. gather  — every population's own compact() step
             (Population._execute_gather_kernel), unconditionally: a
             population not due for its own update this round may still
             have a carried spike surfacing for a downstream target.
          2. merge   — every population with incoming connections combines
             its sources' now-compact()ed output into this round's input
             spike buffer (Population.merge_incoming), stashed on
             Population._merged for phase 3 to consume.
          3. update  — populations actually due at this sub-time
             (Population._is_due) run their real compiled kernel
             (Population._execute_update_kernel), consuming _merged and
             publishing freshly emitted spikes for the NEXT sub-time's
             phase 1 to fold in.

        All three phases launch on the same (default) CUDA stream per
        population, same reasoning as gather(): in-order execution is the
        only sync needed, no explicit barrier.
        """
        sub_dt = self._sub_dt(t_step)
        t_end = self.t + t_step
        while self.t < t_end - 1e-10:
            self.t = min(self.t + sub_dt, t_end)
            for pop in self.populations:
                pop._execute_gather_kernel(self.t)
            for pop in self.populations:
                if pop.incoming:
                    pop._merged = pop.merge_incoming()
            for pop in self.populations:
                if pop._is_due(self.t):
                    pop._execute_update_kernel(self.t)
                    pop.last_update = self.t
