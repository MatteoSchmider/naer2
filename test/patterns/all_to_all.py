import naer2 as n2
from naer2.types import Region
from naer2.types.position import Position


@n2.pattern
def all_to_all(position: Position, source_region: Region, target_region=None) -> Region:
    """
    Connect every source neuron to all neurons in the target population.

    If target_region is provided (the common case when both populations exist),
    it is returned directly — every source neuron projects to the full target.

    If target_region is None, source_region is returned as a proxy. The JIT
    compiler interprets this as "generate the target population with the same
    shape as the source", enabling implicit population creation from the pattern
    alone (useful for constructing symmetric or auto-sized layers).
    """
    if target_region is not None:
        return target_region
    return source_region
