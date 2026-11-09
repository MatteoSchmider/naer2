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

    def blocked(
        self,
        group_size: int,
        n_targets: int,
        bucket: torch.Tensor | None = None,
    ) -> "BlockedTopology":
        """
        Re-lay this topology out per target block, for the update kernel.

        Program `b` of the update kernel owns target neurons
        [b * group_size, (b + 1) * group_size). Walking the canonical
        source-major CSR makes every program traverse a source's whole row and
        discard the links that aren't its own; here links are reordered
        block-major (then source-major inside a block) so a program's
        traversal of a source's row contains only its own links.

        group_size is the kernel's GROUP_SIZE and must match at launch —
        the layout is meaningless for any other value.

        bucket (one delay-bucket id per link, canonical order; see
        delay_buckets) additionally sorts each (block, source) row by bucket
        and keeps each link's id in `bucket` -- the offsets stay the plain
        [n_blocks, N_src + 1] table, so memory is O(links + blocks *
        sources), not O(blocks * buckets * sources).
        """
        n_src = len(self.source_offsets) - 1
        n_blocks = -(-n_targets // group_size)
        if len(self.target_indices) and int(self.target_indices.max()) >= n_targets:
            raise ValueError("topology has a target index >= n_targets")
        if bucket is not None and len(bucket) != len(self.target_indices):
            raise ValueError("bucket must have exactly one entry per link")

        row = (self.target_indices // group_size) * n_src + torch.repeat_interleave(
            torch.arange(n_src), self.source_offsets.diff()
        )
        counts = torch.bincount(row, minlength=n_blocks * n_src)
        n_buckets = int(bucket.max()) + 1 if bucket is not None and len(bucket) else 1
        if bucket is not None:
            # stable: keeps target order within a (row, bucket)
            perm = torch.argsort(row * n_buckets + bucket.to(torch.int64), stable=True)
            bucket = bucket[perm].to(torch.int16)
        else:
            perm = torch.argsort(row, stable=True)  # stable: keeps target order within a row
        del row

        starts = torch.cumsum(counts, 0) - counts
        block_base = torch.zeros(n_blocks + 1, dtype=torch.int64)
        block_base[1:] = torch.cumsum(counts.view(n_blocks, n_src).sum(1), 0)

        # offsets[b, s] : offsets[b, s + 1] are block b's links out of source s.
        # Sorted order is block-major, so the end of a block's last source row is
        # the start of the next block — no separate end table needed.
        offsets = torch.empty(n_blocks, n_src + 1, dtype=torch.int64)
        offsets[:, :n_src] = starts.view(n_blocks, n_src)
        offsets[:, n_src] = block_base[1:]

        target_local = (self.target_indices[perm] % group_size).to(torch.int32)
        return BlockedTopology(
            offsets, target_local, perm.to(torch.int32), block_base, group_size, n_targets,
            n_buckets=n_buckets, bucket=bucket,
        )

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
        takes_target = len(inspect.signature(pattern).parameters) > 2

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


class BlockedTopology:
    """
    A Topology laid out block-major for the update kernel (see Topology.blocked).

    offsets       [n_blocks, N_src + 1] int64 — ABSOLUTE positions into the
                  flat link arrays: block b's links out of source s are
                  positions offsets[b, s] : offsets[b, s + 1] (sorted by
                  delay bucket when delay-bucketed)
    target_local  [n_links] int32 — target index within its block, 0..group_size-1
                  (the global target is block * group_size + target_local)
    perm          [n_links] int32 — perm[k] is the canonical link index of
                  blocked position k
    block_base    [n_blocks + 1] int64 — block b owns positions
                  block_base[b] : block_base[b + 1]
    n_buckets     number of delay buckets (1 without bucketing)
    bucket        [n_links] int16 — each blocked link's delay bucket, or None
                  without bucketing

    Per-link tensors (weights, ...) must be laid out the same way: to_blocked()
    before handing them to the kernel, from_blocked() to read results back in
    canonical link order. A per-group view of any blocked array is just
    array[block_slice(b)] — a zero-copy slice, not a separate allocation.
    """

    def __init__(
        self, offsets, target_local, perm, block_base, group_size, n_targets,
        n_buckets: int = 1, bucket: torch.Tensor | None = None,
    ):
        self.offsets = offsets
        self.target_local = target_local
        self.perm = perm
        self.block_base = block_base
        self.group_size = group_size
        self.n_targets = n_targets
        self.n_buckets = n_buckets
        self.bucket = bucket

    @property
    def n_blocks(self) -> int:
        return self.offsets.shape[0]

    def block_slice(self, b: int) -> slice:
        return slice(int(self.block_base[b]), int(self.block_base[b + 1]))

    def to_blocked(self, per_link: torch.Tensor) -> torch.Tensor:
        return per_link[self.perm]

    def from_blocked(self, blocked: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(blocked)
        out[self.perm] = blocked
        return out


class DelayBuckets:
    """One connection's static delay quantization (see delay_buckets).

    bucket      [n_links] int64, canonical link order: each link's bucket
    mean        [num_buckets] float32: mean delay of each bucket's links --
                the delay a bucket delivers at
    d_min       the connection's minimum delay (the scheduling lookahead)
    resolution  bucket width (grid), or the average width (equal-count;
                feedback only)
    """

    def __init__(self, bucket, mean, d_min, resolution):
        self.bucket, self.mean, self.d_min, self.resolution = bucket, mean, d_min, resolution

    @property
    def num_buckets(self) -> int:
        return len(self.mean)


def delay_buckets(
    delay: torch.Tensor, resolution: float | None = None, num_buckets: int | None = None
) -> DelayBuckets:
    """Quantize one connection's per-link delays into buckets. Only the
    residual above the minimum delay d_min is quantized (d_min is already
    the scheduling lookahead):

      resolution    a grid anchored at d_min: bucket j holds residuals in
                    [j * resolution, (j + 1) * resolution)
      num_buckets   equal-count buckets: boundaries at quantiles of the
                    residuals, so each holds ~len(delay) / num_buckets links
                    (links with equal delays always share a bucket, so a
                    bucket can come out larger and the count smaller)

    Exactly one of the two must be given; the other is reported (for
    num_buckets, `resolution` is the average bucket width). Empty buckets
    are dropped and the ids renumbered, so every bucket has links and a
    mean delay."""
    if (resolution is None) == (num_buckets is None):
        raise ValueError("give exactly one of resolution or num_buckets")
    delay = delay.to(torch.float64)
    d_min, d_max = float(delay.min()), float(delay.max())
    residual = delay - d_min
    if resolution is not None:
        bucket = torch.floor(residual / resolution + 1e-9).to(torch.int64)
    else:
        sorted_res = torch.sort(residual).values
        idx = (torch.arange(1, num_buckets) * len(sorted_res)) // num_buckets
        bounds = torch.unique(sorted_res[idx])
        bounds = bounds[bounds > 0]
        bucket = torch.searchsorted(bounds, residual, right=True)
    used, bucket = torch.unique(bucket, return_inverse=True)
    n = len(used)
    mean = torch.zeros(n, dtype=torch.float64).index_add_(0, bucket, delay)
    mean /= torch.bincount(bucket, minlength=n).to(torch.float64)
    if num_buckets is not None:
        resolution = (d_max - d_min) / n if d_max > d_min else 0.0
    return DelayBuckets(bucket, mean.to(torch.float32), d_min, resolution)
