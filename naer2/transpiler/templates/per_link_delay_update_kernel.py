"""
Stub for the per-link-delay variant of the merged update-receive-send kernel
-- see no_delay_update_kernel.py's module docstring for the split this is
the other half of, and TODO.md for the design discussion in progress.

Selected by Population.compile() (via Population.uses_per_link_delay_kernel)
whenever ANY incoming connection carries a real per-link delay tensor. That
switch is population-wide, not per-connection: once it's on, edge
forward/backward callbacks are disallowed on EVERY incoming connection of
this population, including ones with no delay of their own. This is a
deliberate limitation, not a temporary gap: a per-link delivery, once
expanded and merged into arrival-time order, no longer sits at a stable
position a plastic/learned edge attribute could be written back to without
an expensive scatter -- see the design discussion in TODO.md for why this
was judged not worth supporting rather than supported awkwardly. A
population needing both per-link delay on one connection and a learned
(write-back) edge attribute on another isn't representable; split it into
two populations if that's ever needed.

Still being designed, not implemented: expand each firing source's row
(pre-sorted by delay once, at connection-build time, so a single spike's
own expansion is already arrival-time ordered) and combine arrival-ordered
across sources and across connections with a dedicated fixed-width
merge-network kernel (not a full re-sort), landing on one single-target-
per-entry buffer the update kernel then processes with no CSR walk at all
-- each GPU lane becomes one delivery rather than one target neuron, so
per-neuron state access becomes a gather/scatter by each lane's own target
id, guarded so that two lanes delivering to the same target are never
processed together (two different links from the same source, at the same
source time, can never collide -- see naer2/population.py's no-multapses
note -- so the mask only needs to check for distinct lanes landing on the
same target, nothing about source/time).
"""


class PerLinkDelayUpdateKernelTemplate:
    """Not implemented yet -- see the module docstring."""

    def __init__(self, ctx, population):
        self.ctx = ctx
        self.population = population

    def build(self):
        raise NotImplementedError(
            "per-link-delay update kernel is not implemented yet -- see TODO.md"
        )
