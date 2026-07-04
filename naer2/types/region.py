from abc import ABCMeta

import torch

from .box import Box, IndexBox, _BaseBox
from .points import Points, _BasePoints


class Region(metaclass=ABCMeta):
    """
    Base type and factory for all node space regions.

    A Region describes the spatial structure of a set of nodes — either as a
    bounding box (Box) for regularly structured populations, or as an explicit
    point set (Points) for irregular layouts.

    Construction shortcuts:
      Region(1000)        → IndexBox([[0, 1000, 1]])              # flat 1D, step=1
      Region((H, W))      → IndexBox([[0,H,1],[0,W,1]])           # n-D grid, step=1
      Region(tensor)      → Box(tensor) or Points(tensor)         # from raw tensor
      Region(some_region) → some_region                           # identity

    Box tensors may be [ndim, 2] (start, stop) or [ndim, 3] (start, stop, step).
    A missing step column is filled with the default (1 for IndexBox, 1.0 for
    ContinuousBox) automatically.
    """

    def __new__(cls, *args, **kwargs):
        # Identity: already a Region
        if len(args) == 1 and isinstance(args[0], Region):
            return args[0]
        # int → 1D flat population [[0, n]]
        if len(args) == 1 and isinstance(args[0], int):
            return IndexBox(torch.tensor([[0, args[0]]], dtype=torch.int64))
        # tuple of ints → n-D grid [[0, d0], [0, d1], ...]
        if (
            len(args) == 1
            and isinstance(args[0], tuple)
            and all(isinstance(d, int) for d in args[0])
        ):
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
