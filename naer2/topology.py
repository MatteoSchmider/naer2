import torch

from .types.box import _BaseBox
from .types.points import _BasePoints
from .types.region import Region


class Topology:
    """
    Sparse directed mapping from source nodes to target nodes.

    Built eagerly on the CPU by iterating every point in the source region
    and running the pattern once per point. Stored in CSR format so per-source
    fan-out is O(1) to look up.

    The build loop is embarrassingly parallel: each source point is independent.
    Future GPU version: launch one thread per source point, call
    source_region.position_at(program_id) to get the position, run the pattern
    as a Triton kernel, and write targets into a pre-allocated output buffer.

    Attributes:
        source_offsets  [N_src + 1] int64 — CSR row pointers
        target_indices  [n_links]   int64 — flat target node indices

    TODO:
    - parallelize build over source points (multiprocessing, then GPU)
    - nD target indexing: flatten target position → linear index via target strides
    - support all Region subtypes as both source and target
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
    def build(pattern, source_region: Region, target_region: Region) -> "Topology":
        """Run pattern for every source point and accumulate results into CSR."""
        offsets = [0]
        rows: list[torch.Tensor] = []

        for pos in source_region:
            result = pattern(pos, source_region, target_region)

            if isinstance(result, _BasePoints):
                targets = result.coords[:, 0].to(torch.int64)
            elif isinstance(result, _BaseBox):
                targets = torch.arange(
                    int(result.start[0].item()),
                    int(result.stop[0].item()),
                    int(result.step[0].item()),
                    dtype=torch.int64,
                )
            else:
                targets = torch.zeros(0, dtype=torch.int64)

            rows.append(targets)
            offsets.append(offsets[-1] + len(targets))

        source_offsets = torch.tensor(offsets, dtype=torch.int64)
        target_indices = (
            torch.cat(rows)
            if rows and any(len(r) > 0 for r in rows)
            else torch.zeros(0, dtype=torch.int64)
        )
        return Topology(source_offsets, target_indices)
