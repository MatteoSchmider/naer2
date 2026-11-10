"""
A connection's simulation-side topology: the user's canonical links,
restructured once, on the device, into what intake and the update kernel
walk (see TODO.md, the spike-management redesign).

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
"""

import torch
import triton
import triton.language as tl

from .topology import Topology, delay_buckets


@triton.jit
def topology_ptrs(buf, n_sources, n_segments, n_buckets, n_links):
    """Typed pointers into a TopologyNew buffer: seg_offsets, seg_start,
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


class TopologyNew:
    """See the module docstring.

    topology      the canonical Topology (CSR, rows sorted by target)
    n_targets     target population size
    group_size    neurons per update-kernel program (<= 65536)
    delay         None (no delay), a float (homogeneous), or a per-link
                  tensor in canonical order, quantized with exactly one of
                  resolution / num_buckets (see topology.delay_buckets)
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
