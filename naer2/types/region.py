import torch
from abc import ABCMeta

from .box import Box, IndexBox, _BaseBox
from .points import Points, _BasePoints


class Region(metaclass=ABCMeta):
    """
    Base type and factory for all neuron space regions.

    A Region describes the spatial structure of a set of neurons — either as a
    bounding box (Box) for regularly structured populations, or as an explicit
    point set (Points) for irregular or biologically-derived layouts.

    Regions serve two roles in the architecture:
      1. Population shape — passed to Population() to define neuron count and
         layout. Scalars and tuples are automatically promoted to IndexBox.
      2. Pattern interface — the source_region argument to a pattern callback
         provides structural context about the source space; the return value
         describes which target neurons a given source neuron connects to.
         A Box return enables procedural weight addressing (no explicit
         adjacency matrix). A Points return requires an explicit adjacency
         structure to be generated at JIT time.

    isinstance(x, Region) returns True for any Box or Points instance.

    Construction shortcuts (all produce IndexBox):
      Region(1000)        → IndexBox([[0, 1000]])           # flat 1D population
      Region((H, W))      → IndexBox([[0, H], [0, W]])      # 2D grid
      Region(tensor)      → Box(tensor) or Points(tensor)   # from raw tensor
      Region(some_region) → some_region                     # identity
    """

    def __new__(cls, *args, **kwargs):
        # Identity: already a Region
        if len(args) == 1 and isinstance(args[0], Region):
            return args[0]
        # int → 1D flat population [[0, n]]
        if len(args) == 1 and isinstance(args[0], int):
            return IndexBox(torch.tensor([[0, args[0]]], dtype=torch.int64))
        # tuple of ints → n-D grid [[0, d0], [0, d1], ...]
        if len(args) == 1 and isinstance(args[0], tuple) and all(isinstance(d, int) for d in args[0]):
            return IndexBox(torch.tensor([[0, d] for d in args[0]], dtype=torch.int64))
        # raw tensor → Box by dtype, fall back to Points
        try:
            return Box(*args, **kwargs)
        except (ValueError, TypeError):
            return Points(*args, **kwargs)


# Register concrete base types so isinstance(x, Region) works without
# requiring _BaseBox/_BasePoints to import from region.py (which would be circular)
Region.register(_BaseBox)
Region.register(_BasePoints)
