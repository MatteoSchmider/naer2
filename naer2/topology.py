import inspect

import torch

from naer2.types.position import Position

from .types.box import _BaseBox
from .types.region import Region


class Topology:
    """
    Sparse directed mapping from source nodes to target nodes, in CSR format.

    Built by running a connection pattern once per source point. Each source
    point is independent — the build is embarrassingly parallel and will be
    moved to GPU in the future.

    source_offsets  [N_src + 1] int64 — CSR row pointers
    target_indices  [n_links]   int64 — flat target node indices

    TODO: parallelize build; nD target flattening; support all Region subtypes
    """

    def __init__(self, source_offsets: torch.Tensor, target_indices: torch.Tensor):
        self.source_offsets = source_offsets
        self.target_indices = target_indices

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.target_indices),)

    def __len__(self) -> int:
        return len(self.target_indices)

    @staticmethod
    def _build_target_region(pattern, source_region: Region):
        target_points = []
        for pos in enumerate(source_region.points):
            for point in pattern(Position(pos), source_region):
                target_points.append(point)
        return Region(set(target_points))

    @staticmethod
    def build(pattern, source_region: Region, target_region: Region = None):
        """
        Run pattern for every source point and build CSR topology.

        Returns Topology when target_region is given.
        Returns (Topology, inferred_target_region) when target_region is None,
        where the inferred region is the smallest 1D IndexBox covering all
        generated indices. Use this form for self-defining patterns that produce
        absolute target indices without needing a pre-specified target extent.

        Patterns with a third parameter (target_region) receive it even when
        None. Patterns with only (position, source_region) never receive it.
        """
        takes_target = len(inspect.signature(pattern.__call__).parameters) > 2

        offsets = torch.zeros(len(source_region) + 1, dtype=torch.int64)
        rows: list[torch.Tensor] = []

        if target_region is None:
            target_region = Topology._build_target_region(pattern, source_region)

        for i, pos in enumerate(source_region.points):
            result = (
                pattern(pos, source_region, target_region)
                if takes_target
                else pattern(pos, source_region)
            )
            # generate instantaneous target index rows
            if target_region is not None:
                row = torch.zeros(result.shape[0], dtype=torch.int64)
                for i, target_pos in enumerate(result):
                    row[i] = (target_region.points == target_pos.coords).nonzero(
                        as_tuple=True
                    )[0]
            else:
                row = torch.zeros(0, dtype=torch.int64)
            rows.append(row)
            offsets[i + 1] = offsets[i] + len(row)
        target_indices = (
            torch.cat(rows) if offsets[-1] > 0 else torch.zeros(0, dtype=torch.int64)
        )
        topology = Topology(offsets, target_indices)
        return topology, target_region
