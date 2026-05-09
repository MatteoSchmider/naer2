from collections.abc import Callable

from ..patterns import pattern_types
from ..types import Hook, Position, Region


def pattern(
    cb: Callable[[Position, Region], Region],
) -> Callable[[Position, Region], Region]:
    """
    Decorator to register a callable as a connection pattern.

    A pattern is a pure function with signature:
        (position: Position, source_region: Region) -> Region

    where:
      - position: the coordinates of one specific source neuron within the
        source population. Type mirrors the source Region: IndexPosition for
        IndexBox populations, ContinuousPosition for spatially embedded ones.
      - source_region: the full extent of the source population, providing
        structural context (dimensionality, bounds) so the pattern can compute
        relative offsets, distances, strides, etc.
      - return value: a Region describing which target neurons this source
        neuron connects to. A Box return allows procedural weight addressing
        and optional auto-generation of the target population. A Points return
        requires an explicit adjacency structure at JIT time.

    Patterns must be pure (deterministic, no side effects) so that the JIT
    compiler can rely on repeated calls producing identical connectivity,
    which is required for consistent weight offset addressing.

    Type checking of the callback parameters against Position and Region is
    deferred to connection build time, when the actual source population's
    Region type is known and can be matched against what the pattern declares.
    """
    if not callable(cb):
        raise ValueError(
            f"pattern decorator can only be applied to callables (got {type(cb).__name__})"
        )

    pattern_types[cb] = Hook("pattern", cb).verify({Position, Region})

    return cb
