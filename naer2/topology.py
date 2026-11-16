import inspect

import torch
import triton
import triton.language as tl

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
        idx = (torch.arange(1, num_buckets, device=delay.device) * len(sorted_res)) // num_buckets
        bounds = torch.unique(sorted_res[idx])
        bounds = bounds[bounds > 0]
        bucket = torch.searchsorted(bounds, residual, right=True)
    used, bucket = torch.unique(bucket, return_inverse=True)
    n = len(used)
    mean = torch.zeros(n, dtype=torch.float64, device=delay.device).index_add_(0, bucket, delay)
    mean /= torch.bincount(bucket, minlength=n).to(torch.float64)
    if num_buckets is not None:
        resolution = (d_max - d_min) / n if d_max > d_min else 0.0
    return DelayBuckets(bucket, mean.to(torch.float32), d_min, resolution)


# ------------------------------------------------------------ simulation form

# One SimulationTopology per Connection, built from its Fabric (Fabric.simulation: topology, per-link delay,
#   and the edge attrs the consumer's kernel reads as `attrs`) at the consumer's group size and slot
#   resolution (D-12) -- the device copy of a per-link attr lives here, in simulation order, `perm` giving
#   canonical order for user access
# GAP (D-7): attrs become one block per link -- link l's attrs contiguous,
#   [n_links, n_attrs] instead of [n_attrs, n_links] -- so one pointer per connection reaches them all (D-7)


@triton.jit
def topology_ptrs(buf, n_sources, n_segments, n_buckets, n_links):
    """Typed pointers into a SimulationTopology buffer: seg_offsets, seg_start,
    seg_gb, bucket_mean, link_target (int16, read & 0xFFFF), attrs."""
    seg_offsets = buf
    seg_start = seg_offsets + n_sources + 1
    seg_gb = seg_start + n_segments + 1
    mean = seg_gb + n_segments
    link_target = mean + n_buckets
    attrs = link_target + (n_links + 1) // 2
    return (
        seg_offsets, seg_start, seg_gb, mean.to(tl.pointer_type(tl.float32)),
        link_target.to(tl.pointer_type(tl.int16)), attrs.to(tl.pointer_type(tl.float32)),
    )


@triton.jit
def directory_ptrs(buf, n_sources, n_pairs):
    """Typed pointers into a SimulationTopology.directory() buffer: dir_offsets,
    dir_first, words ([n_sources, W, 2]: bitmap word, earlier count)."""
    dir_first = buf + n_sources + 1
    return buf, dir_first, dir_first + n_pairs + 1


class SimulationTopology:
    """A connection's simulation-side topology: the user's canonical links,
    restructured once, on the device, into what the fused kernel's pull and
    delivery walk (see naer2/transpiler/templates/fused.py).

    Canonical form (what the user and Topology.build give): one row per source
    neuron, its links sorted by target index; every per-link tensor (targets,
    delays, weights, ...) in that order. Simulation form: links sorted by
    (source, target group, delay bucket), target order kept within each piece
    -- one stable sort. Each non-empty (source, group, bucket) piece is a
    SEGMENT, a contiguous range of links all going into one group with one
    delay. Delays are consumed by this: only each bucket's mean delay is kept,
    and which bucket a link belongs to is where it sits.

    One int32 buffer, layout reached through `topology_ptrs`:
      seg_offsets [n_sources + 1]   segments of source s: seg_offsets[s] : seg_offsets[s + 1]
      seg_start   [n_segments + 1]  links of segment i: seg_start[i] : seg_start[i + 1]
      seg_gb      [n_segments]      (group << 16) | bucket -- read the group as
                                    (gb >> 16) & 0xFFFF (bit 31 may be set)
      bucket_mean [n_buckets]       mean delay of each bucket's links (float32)
      link_target [n_links]         target within its group, uint16, two per word
      attrs       [n_attrs, n_links] per-link attributes in simulation order (float32)
    Everything is linear in sources, segments and links -- nothing dense over
    (source, group, bucket). `perm` (simulation position -> canonical link) is
    kept separately, for reading and writing attributes in canonical order.

    topology      the canonical Topology (CSR, rows sorted by target)
    n_targets     target population size
    group_size    neurons per update-kernel program (<= 65536)
    delay         None (no delay), a float (homogeneous), or a per-link
                  tensor in canonical order, quantized with exactly one of
                  resolution / num_buckets (see delay_buckets)
    attrs         per-link tensors in canonical order (e.g. weight)
    """

    def __init__(
        self,
        topology: Topology,
        n_targets: int,
        group_size: int,
        device,
        delay: torch.Tensor | float | None = None,
        resolution: float | None = None,
        num_buckets: int | None = None,
        attrs: dict[str, torch.Tensor] | None = None,
    ):
        attrs = attrs or {}
        device = torch.device(device)
        if group_size > 65536:
            raise ValueError("group_size must fit a uint16 target index (<= 65536)")
        self.group_size = group_size
        self.n_sources = len(topology.source_offsets) - 1
        self.n_groups = -(-n_targets // group_size)
        self.n_links = len(topology.target_indices)
        if self.n_groups >= 65536:
            raise ValueError("more than 65535 target groups don't fit the (group << 16 | bucket) packing")

        offsets = topology.source_offsets.to(device)
        target = topology.target_indices.to(device=device, dtype=torch.int64)
        source = torch.repeat_interleave(torch.arange(self.n_sources, device=device), offsets.diff())
        group = target // group_size

        if isinstance(delay, torch.Tensor):
            db = delay_buckets(delay.to(device), resolution=resolution, num_buckets=num_buckets)
            bucket, mean, self.d_min, self.resolution = db.bucket, db.mean, db.d_min, db.resolution
        else:
            d = float(delay) if delay is not None else 0.0
            bucket = torch.zeros(self.n_links, device=device, dtype=torch.int64)
            mean = torch.tensor([d], device=device, dtype=torch.float32)
            self.d_min, self.resolution = d, 0.0
        self.n_buckets = len(mean)
        if self.n_buckets >= 65536:
            raise ValueError("more than 65535 delay buckets don't fit the (group << 16 | bucket) packing")

        # (source, group, bucket); canonical order is (source, target), so a
        # stable sort keeps target order within every segment
        key = (source * self.n_groups + group) * self.n_buckets + bucket
        perm = torch.argsort(key, stable=True)
        sorted_key = key[perm]
        del key, source, group, bucket
        first = torch.ones(self.n_links, device=device, dtype=torch.bool)
        first[1:] = sorted_key[1:] != sorted_key[:-1]
        seg_first = first.nonzero().flatten()
        self.n_segments = len(seg_first)
        seg_key = sorted_key[seg_first]
        seg_source = seg_key // (self.n_groups * self.n_buckets)
        seg_group = (seg_key // self.n_buckets) % self.n_groups
        seg_bucket = seg_key % self.n_buckets
        del sorted_key, first, seg_key

        n_attr_words = len(attrs) * self.n_links
        link_words = (self.n_links + 1) // 2
        size = (self.n_sources + 1) + (self.n_segments + 1) + self.n_segments + self.n_buckets + link_words + n_attr_words
        self.buf = torch.zeros(size, device=device, dtype=torch.int32)
        o = 0
        self.seg_offsets = self.buf[o:o + self.n_sources + 1]; o += self.n_sources + 1
        self.seg_start = self.buf[o:o + self.n_segments + 1]; o += self.n_segments + 1
        self.seg_gb = self.buf[o:o + self.n_segments]; o += self.n_segments
        self.bucket_mean = self.buf[o:o + self.n_buckets].view(torch.float32); o += self.n_buckets
        self.link_target = self.buf[o:o + link_words].view(torch.int16)[: self.n_links]; o += link_words
        self._attr_offset = o
        attr_block = self.buf[o:o + n_attr_words].view(torch.float32).view(len(attrs), self.n_links)

        self.seg_offsets.copy_(torch.searchsorted(seg_source, torch.arange(self.n_sources + 1, device=device)))
        self.seg_start[: self.n_segments].copy_(seg_first)
        self.seg_start[self.n_segments] = self.n_links
        self.seg_gb.copy_((seg_group << 16) | seg_bucket)
        self.bucket_mean.copy_(mean)
        self.link_target.copy_((target[perm] % group_size).to(torch.int32).to(torch.int16))
        self.perm = perm.to(torch.int32)
        self.attrs = {}
        for i, (name, values) in enumerate(attrs.items()):
            self.attrs[name] = attr_block[i]
            self.write(name, values)

    def directory(self) -> tuple[torch.Tensor, int, int]:
        """Per source, which groups it reaches and where their segments
        start -- for pull intake, which needs a source's segments into ONE
        group (see naer2/transpiler/templates/fused.py). Built once on request, as one
        int32 buffer (directory_ptrs):
          dir_offsets [n_sources + 1]  pairs of source s: dir_offsets[s] : dir_offsets[s + 1]
          dir_first   [n_pairs + 1]    first segment of each (source, group) pair, in segment
                                       order; a pair's segments end where the next pair's start
          words       [n_sources, W, 2] per source and 32-group word: its bitmap of reached
                                       groups, and how many reached groups the earlier words hold
        A spike of source s finds group g's segments in three dependent
        steps: word (s, g // 32) and dir_offsets[s] -> dir_first at
        dir_offsets[s] + earlier count + popcount of the lower bits ->
        the segments; a clear bit means no segments in g. Memory: one word
        per (source, reached group) pair plus 2 * ceil(n_groups / 32) per
        source. Returns (buffer, n_pairs, W)."""
        if getattr(self, "_directory", None) is None:
            dev = self.buf.device
            counts = self.seg_offsets.diff().to(torch.int64)
            seg_source = torch.repeat_interleave(torch.arange(self.n_sources, device=dev), counts)
            seg_group = ((self.seg_gb >> 16) & 0xFFFF).to(torch.int64)
            key = seg_source * self.n_groups + seg_group
            first = torch.ones(self.n_segments, device=dev, dtype=torch.bool)
            first[1:] = key[1:] != key[:-1]
            pair_seg = first.nonzero().flatten()
            n_pairs = len(pair_seg)
            pair_source, pair_group = seg_source[pair_seg], seg_group[pair_seg]
            W = -(-self.n_groups // 32)
            bits = torch.zeros(self.n_sources * W, device=dev, dtype=torch.int64)
            word = pair_source * W + pair_group // 32
            bits.index_add_(0, word, torch.ones_like(word) << (pair_group % 32))  # distinct bits: sum = or
            per_word = torch.bincount(word, minlength=self.n_sources * W).view(self.n_sources, W)
            prefix = per_word.cumsum(1) - per_word
            buf = torch.zeros((self.n_sources + 1) + (n_pairs + 1) + 2 * self.n_sources * W, device=dev,
                              dtype=torch.int32)
            buf[: self.n_sources + 1] = torch.searchsorted(pair_source, torch.arange(self.n_sources + 1, device=dev))
            o = self.n_sources + 1
            buf[o:o + n_pairs] = pair_seg.to(torch.int32)
            buf[o + n_pairs] = self.n_segments
            o += n_pairs + 1
            words = buf[o:].view(self.n_sources, W, 2)
            words[:, :, 0] = (((bits + 2**31) % 2**32) - 2**31).to(torch.int32).view(self.n_sources, W)
            words[:, :, 1] = prefix.to(torch.int32)
            self._directory = (buf, n_pairs, W)
        return self._directory

    def link_addresses(self) -> tuple[int, int, int]:
        """Device addresses of seg_start, link_target and the attribute block
        (attribute i of link l at attrs + 4 * (i * n_links + l)), for kernels
        that keep them per connection instead of recomputing the layout."""
        return self.seg_start.data_ptr(), self.link_target.data_ptr(), self.buf.data_ptr() + 4 * self._attr_offset

    def write(self, name: str, canonical: torch.Tensor) -> None:
        """Set a per-link attribute from canonical link order."""
        view = self.attrs[name]
        view.copy_(canonical.to(device=view.device, dtype=torch.float32)[self.perm.to(torch.int64)])

    def read(self, name: str) -> torch.Tensor:
        """A per-link attribute in canonical link order (a device gather)."""
        view = self.attrs[name]
        out = torch.empty_like(view)
        out[self.perm.to(torch.int64)] = view
        return out

    def segments(self) -> list[tuple[int, int, int, int, int]]:
        """(source, group, bucket, link start, link end) of every segment,
        in order -- for inspection and tests, not the simulation."""
        offsets, start, gb = self.seg_offsets.tolist(), self.seg_start.tolist(), self.seg_gb.tolist()
        out = []
        for s in range(self.n_sources):
            for i in range(offsets[s], offsets[s + 1]):
                out.append((s, (gb[i] >> 16) & 0xFFFF, gb[i] & 0xFFFF, start[i], start[i + 1]))
        return out

    def locality(self) -> float:
        """Links per segment: how well connectivity and delays line up with
        target groups (higher is better; ~1 is the per-link regime).

        TODO (choosing group_size automatically, at reintegration): with the
        scalar receive every wave of a program executes every link (~9 VALU
        per link per wave), so delivery's ALU work is ~ links * group_size /
        32, independent of locality; per-entry costs (headers, the stream
        gather) and intake's work are ~ entries = links / locality(group_size),
        and fewer, larger programs shorten the per-program chains. The
        optimum is set by latency and occupancy more than by ALU work and
        depends on the device: PD14, 0.1ms buckets, full round trip -- 780M:
        G=32 3.40, 128 1.76, 256 1.31, 512 1.29 ms/round (one-hot receive
        before: best 3.11 at G=128); RX 6600 (one-hot): best at G=32. Model
        it as  a * links * G + b * entries(G) + latency(n_groups, per-program
        chain), fitted per device, subject to n_groups >= a few programs per
        compute unit; entries(G) can be estimated by building at a small G
        and counting how many of a source's segments share a larger group."""
        return self.n_links / max(self.n_segments, 1)
