"""
naer2/intake.py's intake kernel against a plain-Python reference of the same
rules (see the module docstring there): gather the sources' per-block
sorted emission runs, merge them in emission order, and insert one ring
entry per delay bucket -- main run if e >= the watermark and not
emission-delayed, side run (kept sorted) otherwise; late arrivals into the
first undelivered slot; far future and full slots raise.

Order among entries with equal emission times is free, so each slot is
compared as a multiset, plus an explicit check that both runs are sorted.
"""

import math
import random

import pytest
import torch

from naer2.intake import Ring, intake
from naer2.spike_gather import BlockedSpikes

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernel"
)

DEVICE = "cuda:0"


@pytest.fixture(autouse=True, scope="module")
def _device():
    torch.cuda.set_device(0)


def _emission(blocks: list[list[tuple[float, int]]], cap: int) -> BlockedSpikes:
    """blocks[b] = [(time, node), ...], sorted by time (the emission
    contract); padded to cap with garbage that must never be read."""
    n = len(blocks)
    time = torch.full((n * cap,), float("nan"))
    node = torch.full((n * cap,), -1, dtype=torch.int32)
    count = torch.zeros(n, dtype=torch.int32)
    for b, entries in enumerate(blocks):
        assert [t for t, _ in entries] == sorted(t for t, _ in entries)
        for i, (t, s) in enumerate(entries):
            time[b * cap + i], node[b * cap + i] = t, s
        count[b] = len(entries)
    return BlockedSpikes(node.to(DEVICE), time.to(DEVICE), count.to(DEVICE), cap)


class Reference:
    def __init__(self, n_slots, slot_width):
        self.n_slots, self.w = n_slots, torch.tensor(slot_width, dtype=torch.float32)
        self.main = {k: [] for k in range(n_slots)}
        self.side = {k: [] for k in range(n_slots)}
        self.mark = -math.inf
        self.far = False

    def intake(self, ring_pos, sources, active, window_end, bucket_mean):
        batch = []
        for c, (src, a) in enumerate(zip(sources, active)):
            if not a:
                continue
            count, time, node = src.count.tolist(), src.time.tolist(), src.node.tolist()
            for b, n in enumerate(count):
                for i in range(min(n, src.cap)):
                    batch.append((time[b * src.cap + i], node[b * src.cap + i], c))
        for e, node, c in sorted(batch, key=lambda x: x[0]):
            to_main = e >= self.mark and e < window_end[c]
            if to_main:
                self.mark = e
            for j, m in enumerate(bucket_mean[c]):
                arrival = torch.tensor(e, dtype=torch.float32) + torch.tensor(m, dtype=torch.float32)
                k = max(int(torch.floor(arrival / self.w)), ring_pos)
                if k - ring_pos >= self.n_slots:
                    self.far = True
                    continue
                (self.main if to_main else self.side)[k % self.n_slots].append((e, node, c, j))


def _ring_contents(ring: Ring, which: str):
    time, node, conn, bucket = (t.cpu() for t in getattr(ring, which))
    counts = getattr(ring, f"{which}_count").tolist()
    out = {}
    for k, n in enumerate(counts):
        sl = slice(k * ring.slot_cap, k * ring.slot_cap + n)
        out[k] = list(zip(time[sl].tolist(), node[sl].tolist(), conn[sl].tolist(), bucket[sl].tolist()))
    return out


def _check(ring: Ring, ref: Reference):
    for which in ("main", "side"):
        got = _ring_contents(ring, which)
        want = getattr(ref, which)
        for k in range(ring.n_slots):
            times = [e for e, *_ in got[k]]
            assert times == sorted(times), f"{which} run of slot {k} is not sorted: {times}"
            g = sorted((torch.tensor(e).item(), n, c, j) for e, n, c, j in got[k])
            w = sorted((torch.tensor(e, dtype=torch.float32).item(), n, c, j) for e, n, c, j in want[k])
            assert g == w, f"{which} run of slot {k}"
    assert ring.watermark.item() == pytest.approx(ref.mark) if ref.mark != -math.inf else True


def _toy_round(t: float):
    """Toy model (toys/delay_buckets.py): 3 sources, 2 spikes each, all
    stamped at the round time, one block per neuron pair."""
    sources = [_emission([[(t, 0)], [(t, 1)]], cap=4) for _ in range(3)]
    return sources, [True] * 3, [t + 0.1] * 3


BUCKET_MEAN = [[0.15, 0.27, 0.33, 0.48, 0.56], [0.13, 0.26, 0.34, 0.46, 0.57], [0.18, 0.24, 0.38, 0.43, 0.57]]


@pytest.mark.parametrize("uniform_time", [True, False])
def test_toy_rounds_fill_main_runs_only(uniform_time):
    ring, ref = Ring(8, 0.1, DEVICE), Reference(8, 0.1)
    for r in range(3):
        t = round(0.1 * r, 6)
        sources, active, window_end = _toy_round(t)
        intake(ring, sources, active, window_end, BUCKET_MEAN, uniform_time=uniform_time)
        ref.intake(0, sources, active, window_end, BUCKET_MEAN)
    _check(ring, ref)
    assert ring.side_count.sum().item() == 0
    assert ring.main_count.sum().item() == 3 * 6 * 5


@pytest.mark.parametrize("seed", range(5))
def test_unsorted_blocks_and_emission_delays_use_side_runs(seed):
    """Exact (unsorted across blocks) emission times, some emission-delayed
    past their source's window, and a second intake whose spikes lie below
    the watermark set by the first."""
    rng = random.Random(seed)
    ring, ref = Ring(64, 0.1, DEVICE), Reference(64, 0.1)
    means = [[0.1 + 0.1 * j + rng.random() * 0.05 for j in range(rng.randint(1, 6))] for _ in range(3)]
    for t in (1.0, 1.05, 1.2):
        sources = []
        for c in range(3):
            blocks = []
            for _ in range(5):
                times = sorted(t + rng.random() * (0.6 if rng.random() < 0.2 else 0.2) for _ in range(rng.randint(0, 4)))
                blocks.append([(e, rng.randrange(100)) for e in times])
            sources.append(_emission(blocks, cap=4))
        active, window_end = [True, True, c != 1], [t + 0.2] * 3
        intake(ring, sources, active, window_end, means)
        ref.intake(0, sources, active, window_end, means)
    _check(ring, ref)
    assert ring.side_count.sum().item() > 0


def test_uniform_kernel_matches_reference_across_many_programs():
    """Sources with many blocks are split over several programs of the
    uniform kernel (32 blocks each); the result must not depend on that."""
    rng = random.Random(7)
    ring, ref = Ring(32, 0.1, DEVICE), Reference(32, 0.1)
    for t in (1.0, 1.1, 1.2):
        sources = [
            _emission([[(t, rng.randrange(1000))] * rng.choice([0, 0, 0, 1, 2]) for _ in range(n_blocks)], cap=4)
            for n_blocks in (200, 70, 33)
        ]
        intake(ring, sources, [True] * 3, [t + 0.1] * 3, BUCKET_MEAN, uniform_time=True)
        ref.intake(0, sources, [True] * 3, [t + 0.1] * 3, BUCKET_MEAN)
    _check(ring, ref)
    assert ring.main_count.sum().item() > 0


@pytest.mark.parametrize("uniform_time", [True, False])
def test_persistent_plan_reads_each_source_run_exactly_once(uniform_time):
    """With one IntakePlan over several rounds, intake decides on the device
    which sources ran since it last read them (source run id vs last-read
    id, double-buffered): a source that didn't run again must not be read
    again, even though its emission buffer still holds the old spikes."""
    from naer2.intake import IntakePlan, source_state

    ring = Ring(32, 0.1, DEVICE)
    a, b = _emission([[(1.0, 0)], [(1.0, 1)]], cap=4), _emission([[(1.0, 2)]], cap=4)
    states = [source_state(DEVICE), source_state(DEVICE)]
    plan = IntakePlan(ring, [a, b], states, BUCKET_MEAN[:2])

    def ran(i, t, window_end):  # what the source's update kernel will do on the device
        src = (a, b)[i]  # its spikes (the non-nan entries), restamped t
        src.time[:] = torch.where(torch.isnan(src.time), src.time, torch.full_like(src.time, t))
        states[i][0] += 1
        states[i][1:].view(torch.float32)[0] = window_end

    ran(0, 1.0, 1.1); ran(1, 1.0, 1.1)
    plan.run(uniform_time)   # both fresh: 3 spikes x 5 buckets
    ran(0, 1.1, 1.2)
    plan.run(uniform_time)   # only a: +2 spikes x 5 buckets
    plan.run(uniform_time)   # nothing ran
    ring.check()
    assert ring.main_count.sum().item() == (3 + 2) * 5


SIDE_BOUND_ROUNDS = ((1.2, 1.3), (1.0, 1.1), (1.25, 1.25))  # (t, window end)


def test_general_kernel_puts_below_watermark_and_delayed_batches_in_side_runs():
    """Equal-time batches that still need side runs: below the watermark
    (the round at 1.0 after 1.2) and emission-delayed (window end at the
    emission time)."""
    ring, ref = Ring(32, 0.1, DEVICE), Reference(32, 0.1)
    for t, window_end in SIDE_BOUND_ROUNDS:
        sources = [_emission([[(t, 0)], [(t, 1)]], cap=4) for _ in range(3)]
        intake(ring, sources, [True] * 3, [window_end] * 3, BUCKET_MEAN)
        ref.intake(0, sources, [True] * 3, [window_end] * 3, BUCKET_MEAN)
    _check(ring, ref)
    assert ring.side_count.sum().item() == 2 * 3 * 2 * 5


@pytest.mark.parametrize("t, window_end", SIDE_BOUND_ROUNDS[1:])
def test_uniform_kernel_rejects_batches_needing_side_runs(t, window_end):
    """uniform_time promises bare n2.spike() sources, whose spikes never
    need a side run; a batch that would must fail loudly, not mis-order."""
    ring = Ring(32, 0.1, DEVICE)
    first = [_emission([[(1.2, 0)]], cap=4) for _ in range(3)]
    intake(ring, first, [True] * 3, [1.3] * 3, BUCKET_MEAN, uniform_time=True)
    sources = [_emission([[(t, 0)], [(t, 1)]], cap=4) for _ in range(3)]
    with pytest.raises(RuntimeError, match="uniform contract"):
        intake(ring, sources, [True] * 3, [window_end] * 3, BUCKET_MEAN, uniform_time=True)


def test_arrivals_before_the_ring_position_land_in_its_first_slot():
    ring, ref = Ring(8, 0.1, DEVICE), Reference(8, 0.1)
    ring.ring_pos = 4  # slots up to 3 already delivered
    sources, active, window_end = _toy_round(0.1)  # arrivals 0.23 .. 0.67 -> slots 2 .. 6
    intake(ring, sources, active, window_end, BUCKET_MEAN)
    ref.intake(4, sources, active, window_end, BUCKET_MEAN)
    _check(ring, ref)
    assert ring.main_count[4].item() == 3 * 2 * 3  # buckets landing in slots 2, 3 and 4


def test_far_future_arrivals_raise_not_implemented():
    ring = Ring(4, 0.1, DEVICE)  # covers only 0.4 ahead
    sources, active, window_end = _toy_round(0.0)
    with pytest.raises(NotImplementedError, match="far-future"):
        intake(ring, sources, active, window_end, BUCKET_MEAN)


def test_full_slot_raises():
    ring = Ring(8, 0.1, DEVICE, slot_cap=5)
    sources, active, window_end = _toy_round(0.0)  # 6 spikes into every bucket's slot
    with pytest.raises(RuntimeError, match="capacity"):
        intake(ring, sources, active, window_end, BUCKET_MEAN)
