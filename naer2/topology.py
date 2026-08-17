import inspect

import torch

from .types.box import _BaseBox
from .types.points import _BasePoints
from .types.region import Region


def _make_index_fn(target_region):
    """
    Build a coordinate → linear-index function for the given target region.

    Called once per Topology.build() invocation so per-point lookup is fast:
    - Box: row-major arithmetic (O(1) per point, no memory beyond strides).
    - Points: dict lookup built upfront (O(n_target) once, O(1) per lookup).

    Returned function signature: (coords: [n, ndim] Tensor) -> [n] int64 Tensor
    """
    if isinstance(target_region, _BaseBox):
        starts  = target_region.start.clone()
        steps   = target_region.step.clone()
        sizes   = target_region.sizes.clone()
        ndim    = len(sizes)

        if ndim == 1:
            # Fast path: single dimension — skip broadcasting and the .sum() reduction.
            s0 = int(starts[0].item())
            p0 = int(steps[0].item())
            if s0 == 0 and p0 == 1:
                def to_index(coords: torch.Tensor) -> torch.Tensor:
                    if len(coords) == 0:
                        return torch.zeros(0, dtype=torch.int64)
                    return coords[:, 0].to(torch.int64)
            else:
                def to_index(coords: torch.Tensor) -> torch.Tensor:
                    if len(coords) == 0:
                        return torch.zeros(0, dtype=torch.int64)
                    return (coords[:, 0].to(torch.int64) - s0) // p0
            return to_index

        strides = torch.ones(ndim, dtype=torch.int64)
        for d in range(ndim - 2, -1, -1):
            strides[d] = strides[d + 1] * sizes[d + 1]

        def to_index(coords: torch.Tensor) -> torch.Tensor:
            if len(coords) == 0:
                return torch.zeros(0, dtype=torch.int64)
            local = (coords.to(torch.int64) - starts) // steps  # [n, ndim]
            return (local * strides).sum(dim=1)                  # [n]

        return to_index

    if isinstance(target_region, _BasePoints):
        lookup = {
            tuple(target_region.coords[i].tolist()): i
            for i in range(len(target_region))
        }

        def to_index(coords: torch.Tensor) -> torch.Tensor:
            if len(coords) == 0:
                return torch.zeros(0, dtype=torch.int64)
            return torch.tensor(
                [lookup[tuple(c.tolist())] for c in coords], dtype=torch.int64
            )

        return to_index

    raise TypeError(f"Cannot build index function for {type(target_region).__name__}")


def _result_coords(result) -> torch.Tensor:
    """Extract [n, ndim] coordinate tensor from a pattern result."""
    if isinstance(result, _BasePoints):
        return result.coords
    if isinstance(result, _BaseBox):
        return result.points
    return torch.zeros((0, 1), dtype=torch.int64)


def _infer_region(coord_rows: list[torch.Tensor]) -> Region:
    """
    Derive the smallest IndexBox covering all coordinates in coord_rows.

    For 1D: Region(max + 1).  For nD: Region((max_0 + 1, max_1 + 1, ...)).
    """
    non_empty = [r for r in coord_rows if len(r) > 0]
    if not non_empty:
        return Region(0)
    all_coords = torch.cat(non_empty)            # [total, ndim]
    maxes = all_coords.max(dim=0).values.tolist()
    sizes = tuple(int(m) + 1 for m in maxes)
    return Region(sizes[0] if len(sizes) == 1 else sizes)


class Topology:
    """
    Sparse directed mapping from source nodes to target nodes, in CSR format.

    Built by running a connection pattern once per source point. Patterns work
    in coordinate space — results are coordinates in the target region, converted
    to linear indices by the builder using row-major arithmetic (Box targets) or
    a pre-built dict lookup (Points targets).

    source_offsets  [N_src + 1] int64 — CSR row pointers
    target_indices  [n_links]   int64 — flat target node indices

    TODO: parallelize build (multiprocessing → GPU); support Points source regions
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
    def build(pattern, source_region: Region, target_region: Region = None):
        """
        Run pattern for every source point and build CSR topology.

        Returns Topology when target_region is given.
        Returns (Topology, inferred_target_region) when target_region is None,
        collecting all output coordinates in one pass and deriving the smallest
        bounding box that covers them.

        Patterns with a third parameter receive target_region (even when None);
        patterns with only (position, source_region) never receive it.

        If the pattern class exposes target_region(source_region) analytically,
        that is called instead of the infer pass (option B — not yet implemented).
        """
        takes_target = len(inspect.signature(pattern.__call__).parameters) > 2

        def call(pos):
            return (
                pattern(pos, source_region, target_region)
                if takes_target
                else pattern(pos, source_region)
            )

        if target_region is not None:
            to_index = _make_index_fn(target_region)
            rows: list[torch.Tensor] = []
            off = [0]
            for pos in source_region:
                row = to_index(_result_coords(call(pos)))
                rows.append(row)
                off.append(off[-1] + len(row))
            target_indices = (
                torch.cat(rows) if off[-1] > 0
                else torch.zeros(0, dtype=torch.int64)
            )
            return Topology(torch.tensor(off, dtype=torch.int64), target_indices)

        # Infer path: collect coords in one pass, derive target, convert.
        coord_rows: list[torch.Tensor] = []
        for pos in source_region:
            coord_rows.append(_result_coords(call(pos)))

        inferred = _infer_region(coord_rows)
        to_index = _make_index_fn(inferred)
        rows = [to_index(r) for r in coord_rows]
        off = [0]
        for row in rows:
            off.append(off[-1] + len(row))
        target_indices = (
            torch.cat(rows) if off[-1] > 0
            else torch.zeros(0, dtype=torch.int64)
        )
        return Topology(torch.tensor(off, dtype=torch.int64), target_indices), inferred
