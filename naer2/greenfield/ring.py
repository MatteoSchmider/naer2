"""
A consumer's rings of arrival slots: where its fused kernel's pull intake
puts every link it takes in, and where delivery reads them from (naer2/
greenfield/fused.py).

One ring per target GROUP (the neurons one program owns). Slot k of a
group's ring covers arrivals in [k * slot_width, (k + 1) * slot_width) and
lives at index k % N_SLOTS. Each slot holds
  - a main run: records appended in emission order, and
  - a side run: out-of-order records (emission-delayed, or below the
    watermark), kept sorted by emission time by insertion.
A record is one LINK: emission time e, (connection << 24) | source neuron,
the link's index in its connection (its attributes, e.g. the weight) and
its target within the group -- delivery goes from the record straight to
the receive. A record goes to the main runs if e >= the watermark (largest
e ever appended to a group's main runs) and it isn't emission-delayed (e
before its source's run window end); otherwise to the side runs.

An arrival before the ring position (already delivered) goes into the first
undelivered slot (late, the timestep quantization); one at or past ring
position + N_SLOTS is far future (flag 1, not implemented yet); a full run
sets flag 0; a uniform-emission run that would need a side run sets flag 2.
"""

import torch
import triton
import triton.language as tl

# TODO: run capacities are fixed constants for now; a burst (e.g. PD14's
# 902-spike thalamic volley) lands in every slot of its delay range at once.
# They should become population-specific and adjust themselves on overflow.
DEFAULT_MAIN_CAP = 2048
DEFAULT_SIDE_CAP = 512
_N_FLAGS = 3  # run overflow, far future, uniform contract violated


@triton.jit
def _f32(p):
    return p.to(tl.pointer_type(tl.float32))


@triton.jit
def _ring(ring, n_groups, N_SLOTS: tl.constexpr, MAIN_CAP: tl.constexpr, SIDE_CAP: tl.constexpr):
    """Typed pointers into a Ring buffer, see Ring for the layout: per run
    kind (main, side) the four fields of a record -- time, source word,
    link index, target -- then the per-cell counts; then watermark and
    flags. Field f of the record with index i (see _rec) is at
    field_ptr + 4 * i."""
    cells = n_groups * N_SLOTS
    main_time = ring
    main_src = ring + 1
    main_lo = ring + 2
    main_n = ring + 3
    side_time = ring + 4 * cells * MAIN_CAP
    side_src = side_time + 1
    side_lo = side_time + 2
    side_n = side_time + 3
    main_count = side_time + 4 * cells * SIDE_CAP
    side_count = main_count + cells
    watermark = side_count + cells
    return (
        _f32(main_time), main_src, main_lo, main_n, main_count,
        _f32(side_time), side_src, side_lo, side_n, side_count,
        _f32(watermark), watermark + 1,
    )


@triton.jit
def _rec(cell, pos, n_cells):
    """Record index of entry pos of a cell's run. Runs are cut into cache
    lines of 4 records and line b of every cell sits next to line b of all
    the others: the live front of all cells is one dense region (the pull's
    appends and delivery's reads stay in few lines and pages), bursts reach
    further back. Capacities are multiples of 4."""
    return ((pos >> 2) * n_cells + cell) * 4 + (pos & 3)


class Ring:
    """One consumer's rings of arrival slots, one ring per target group
    (see the module docstring), as one int32 buffer: per (group, slot) cell
    a main run of MAIN_CAP records and a side run of SIDE_CAP, each one
    16-byte record (time, (conn << 24) | source, link index, target),
    interleaved by cache line across cells (see _rec); then main counts,
    side counts, the watermark (float32) and the flags. Cell of (group g,
    slot k): g * n_slots + k % n_slots. main / side: the four fields as
    copies in run order, record pos of cell c at c * cap + pos (for
    inspection and tests)."""

    def __init__(self, n_groups: int, n_slots: int, slot_width: float, device,
                 main_cap: int = DEFAULT_MAIN_CAP, side_cap: int = DEFAULT_SIDE_CAP):
        self.n_groups, self.n_slots, self.slot_width = n_groups, n_slots, slot_width
        if main_cap % 4 or side_cap % 4:
            raise ValueError("run capacities must be multiples of 4 (entries are interleaved by cache line)")
        self.main_cap, self.side_cap = main_cap, side_cap
        cells = n_groups * n_slots
        m, s = cells * main_cap, cells * side_cap
        self.buf = torch.zeros(4 * m + 4 * s + 2 * cells + 1 + _N_FLAGS, device=device, dtype=torch.int32)
        self._records = (self.buf[:4 * m].view(m, 4), self.buf[4 * m:4 * m + 4 * s].view(s, 4))
        o = 4 * m + 4 * s
        self.main_count = self.buf[o:o + cells]
        self.side_count = self.buf[o + cells:o + 2 * cells]
        self.watermark = self.buf[o + 2 * cells:o + 2 * cells + 1].view(torch.float32)
        self.watermark.fill_(float("-inf"))
        self.flags = self.buf[o + 2 * cells + 1:]
        self.ring_pos = 0  # first slot not yet delivered

    def cell(self, group: int, slot: int) -> int:
        return group * self.n_slots + slot % self.n_slots

    def _fields(self, rec: torch.Tensor, cap: int):
        cells = self.n_groups * self.n_slots
        cell = torch.arange(cells, device=rec.device).repeat_interleave(cap)
        pos = torch.arange(cap, device=rec.device).repeat(cells)
        r = rec[((pos >> 2) * cells + cell) * 4 + (pos & 3)]  # see _rec
        return r.view(torch.float32)[:, 0], r[:, 1], r[:, 2], r[:, 3]

    @property
    def main(self):
        """(time, source word, link index, target) of the main runs, run order."""
        return self._fields(self._records[0], self.main_cap)

    @property
    def side(self):
        return self._fields(self._records[1], self.side_cap)

    def check(self) -> None:
        overflow, far, uniform = self.flags.tolist()
        if overflow:
            raise RuntimeError(
                f"a ring run exceeded its capacity (main {self.main_cap}, side {self.side_cap} entries)"
            )
        if far:
            raise NotImplementedError(
                f"an arrival lies at or beyond {self.n_slots} slots ahead of the ring "
                f"position (e.g. a long emission delay): the far-future list isn't implemented"
            )
        if uniform:
            raise RuntimeError(
                "a uniform-emission run took in a spike below the watermark or past its "
                "source's run window: the uniform contract (bare n2.spike() only) was violated"
            )
