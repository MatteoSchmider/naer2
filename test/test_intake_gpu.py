"""
naer2/intake.py's intake kernels against a plain-Python reference of the
same rules (see the module docstring there): gather the sources' per-group
sorted emission runs, merge them in emission order, and expand every spike
into one entry per segment of its source (naer2/topology_new.py), each into
its group's ring at the slot of e + its bucket's mean delay -- main run if
e >= the watermark and not emission-delayed, side run (kept sorted)
otherwise; late arrivals into the first undelivered slot; far future and
full runs raise.

Order among entries with equal emission times is free, so each (group,
slot) cell is compared as a multiset, plus an explicit check that both runs
are sorted.
"""

import math
import random

import pytest
import torch

from naer2.emission import EmissionBuffer
from naer2.intake import IntakePlan, Ring
from naer2.topology import Topology
from naer2.topology_new import TopologyNew

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernel"
)

DEVICE = "cuda:0"


@pytest.fixture(autouse=True, scope="module")
def _device():
    torch.cuda.set_device(0)


# ------------------------------------------------------------ building blocks


def _emission(groups: list[list[tuple[float, int]]], cap: int) -> EmissionBuffer:
    """An EmissionBuffer holding groups[g] = [(time, node), ...], sorted by
    time (the emission contract); padded to cap with garbage that must never
    be read. Its run id is 0 (never ran) until _ran marks it."""
    em = EmissionBuffer(len(groups), cap, DEVICE)
    time = torch.full((len(groups) * cap,), float("nan"))
    node = torch.full((len(groups) * cap,), -1, dtype=torch.int32)
    count = torch.zeros(len(groups), dtype=torch.int32)
    for g, entries in enumerate(groups):
        assert [t for t, _ in entries] == sorted(t for t, _ in entries)
        for i, (t, s) in enumerate(entries):
            time[g * cap + i], node[g * cap + i] = t, s
        count[g] = len(entries)
    em.time.copy_(time)
    em.node.copy_(node)
    em.count.copy_(count)
    return em


def _ran(em: EmissionBuffer, window_end: float) -> None:
    """What the source's update kernel does on every run: bump its run id,
    record its window end."""
    em.state[0] += 1
    em.state[1:].view(torch.float32).fill_(window_end)


def _topology(n_src, n_tgt, group_size, delay="random", k=6, seed=0, weight=False):
    """A random connection: every source gets up to k distinct targets
    (sorted, canonical), per-link delays in [0.1, 0.6) at resolution 0.1
    ("random"), one homogeneous delay (a float), or none; with `weight`, a
    random per-link weight attribute."""
    g = torch.Generator().manual_seed(seed)
    rows = [torch.sort(torch.randperm(n_tgt, generator=g)[: torch.randint(1, k + 1, (1,), generator=g).item()]).values
            for _ in range(n_src)]
    topo = Topology(torch.tensor([0] + [len(r) for r in rows]).cumsum(0), torch.cat(rows))
    attrs = {"weight": torch.rand(len(topo), generator=g)} if weight else None
    if delay == "random":
        d = torch.rand(len(topo), generator=g) * 0.5 + 0.1
        return TopologyNew(topo, n_tgt, group_size, DEVICE, delay=d, resolution=0.1, attrs=attrs)
    return TopologyNew(topo, n_tgt, group_size, DEVICE, delay=delay, attrs=attrs)


def intake(ring, sources, active, window_end, topologies, uniform_time=False):
    """One-shot intake: the active sources ran (window_end[i]), the others
    never did; a fresh IntakePlan reads them; flags checked right away."""
    for em, a, w in zip(sources, active, window_end):
        em.state.zero_()
        if a:
            _ran(em, w)
    IntakePlan(ring, sources, topologies).run(uniform_time)
    ring.check()


class Reference:
    def __init__(self, n_groups, n_slots, slot_width):
        self.n_slots, self.w = n_slots, torch.tensor(slot_width, dtype=torch.float32)
        cells = range(n_groups * n_slots)
        self.main = {k: [] for k in cells}
        self.side = {k: [] for k in cells}
        self.mark = -math.inf
        self.far = False

    def intake(self, ring_pos, sources, active, window_end, topologies):
        batch = []
        for c, (src, a) in enumerate(zip(sources, active)):
            if not a:
                continue
            count, time, node = src.count.tolist(), src.time.tolist(), src.node.tolist()
            for g, n in enumerate(count):
                for i in range(min(n, src.cap)):
                    batch.append((time[g * src.cap + i], node[g * src.cap + i], c))
        for e, s, c in sorted(batch, key=lambda x: x[0]):
            to_main = e >= self.mark and e < window_end[c]
            if to_main:
                self.mark = e
            t = topologies[c]
            offsets, gb, mean = t.seg_offsets.tolist(), t.seg_gb.tolist(), t.bucket_mean.tolist()
            for seg in range(offsets[s], offsets[s + 1]):
                group, j = (gb[seg] >> 16) & 0xFFFF, gb[seg] & 0xFFFF
                arrival = torch.tensor(e, dtype=torch.float32) + torch.tensor(mean[j], dtype=torch.float32)
                k = max(int(torch.floor(arrival / self.w)), ring_pos)
                if k - ring_pos >= self.n_slots:
                    self.far = True
                    continue
                cell = group * self.n_slots + k % self.n_slots
                (self.main if to_main else self.side)[cell].append((e, (c << 24) | s, seg))


def _ring_contents(ring: Ring, which: str):
    time, src, seg = (t.cpu() for t in getattr(ring, which))
    counts = getattr(ring, f"{which}_count").tolist()
    cap = ring.main_cap if which == "main" else ring.side_cap
    return {
        k: list(zip(time[k * cap:k * cap + n].tolist(), src[k * cap:k * cap + n].tolist(),
                    seg[k * cap:k * cap + n].tolist()))
        for k, n in enumerate(counts)
    }


def _check(ring: Ring, ref: Reference):
    for which in ("main", "side"):
        got = _ring_contents(ring, which)
        want = getattr(ref, which)
        for k in got:
            times = [e for e, *_ in got[k]]
            assert times == sorted(times), f"{which} run of cell {k} is not sorted: {times}"
            g = sorted(got[k])
            w = sorted((torch.tensor(e, dtype=torch.float32).item(), s, seg) for e, s, seg in want[k])
            assert g == w, f"{which} run of cell {k}"
    if ref.mark != -math.inf:
        assert ring.watermark.item() == pytest.approx(ref.mark)


# ------------------------------------------------------------ tests

# toy shape (toys/delay_buckets.py): 3 sources of 2 neurons, 2 spikes each
# per round (one per source group), into a consumer of 8 neurons in 2 groups


def _toy_topologies():
    return [_topology(2, 8, 4, seed=c) for c in range(3)]


def _toy_round(t: float):
    sources = [_emission([[(t, 0)], [(t, 1)]], cap=4) for _ in range(3)]
    return sources, [True] * 3, [t + 0.1] * 3


@pytest.mark.parametrize("uniform_time", [True, False])
def test_toy_rounds_fill_main_runs_only(uniform_time):
    topos = _toy_topologies()
    ring, ref = Ring(2, 8, 0.1, DEVICE), Reference(2, 8, 0.1)
    for r in range(3):
        t = round(0.1 * r, 6)
        sources, active, window_end = _toy_round(t)
        intake(ring, sources, active, window_end, topos, uniform_time=uniform_time)
        ref.intake(0, sources, active, window_end, topos)
    _check(ring, ref)
    assert ring.side_count.sum().item() == 0
    assert ring.main_count.sum().item() == 3 * sum(t.n_segments for t in topos)  # every segment, every round


@pytest.mark.parametrize("delay", [None, 0.25])
def test_no_or_homogeneous_delay_is_one_entry_per_source_and_group(delay):
    """One bucket: each spike reaches each of its groups with exactly one
    entry, all in the slot of e + delay."""
    topos = [_topology(8, 64, 16, delay=delay, k=40, seed=c) for c in range(2)]
    ring, ref = Ring(4, 8, 0.1, DEVICE), Reference(4, 8, 0.1)
    ring.ring_pos = 10  # e = 1.0 is slot 10
    sources = [_emission([[(1.0, 0), (1.0, 3)], [(1.0, 5)]], cap=4) for _ in range(2)]
    intake(ring, sources, [True, True], [1.1, 1.1], topos, uniform_time=True)
    ref.intake(10, sources, [True, True], [1.1, 1.1], topos)
    _check(ring, ref)
    slot = int(math.floor((1.0 + (delay or 0.0)) / 0.1 + 1e-6))
    for g in range(4):
        others = [ring.main_count[ring.cell(g, k)].item() for k in range(8) if k != slot % 8]
        assert others == [0] * 7


@pytest.mark.parametrize("seed", range(5))
def test_unsorted_groups_and_emission_delays_use_side_runs(seed):
    """Exact (unsorted across groups) emission times, some emission-delayed
    past their source's window, and a second intake whose spikes lie below
    the watermark set by the first."""
    rng = random.Random(seed)
    topos = [_topology(20, 64, 16, seed=seed * 10 + c) for c in range(3)]
    ring, ref = Ring(4, 64, 0.1, DEVICE), Reference(4, 64, 0.1)
    for t in (1.0, 1.05, 1.2):
        sources = []
        for c in range(3):
            groups = []
            for _ in range(5):
                times = sorted(t + rng.random() * (0.6 if rng.random() < 0.2 else 0.2) for _ in range(rng.randint(0, 4)))
                groups.append([(e, rng.randrange(20)) for e in times])
            sources.append(_emission(groups, cap=4))
        active, window_end = [True, True, t != 1.05], [t + 0.2] * 3
        intake(ring, sources, active, window_end, topos)
        ref.intake(0, sources, active, window_end, topos)
    _check(ring, ref)
    assert ring.side_count.sum().item() > 0


def test_uniform_kernel_matches_reference_across_many_programs():
    """Sources with many groups are split over several programs of the
    uniform kernel (32 source groups each); the result must not depend on
    that."""
    rng = random.Random(7)
    sizes = (200, 70, 33)
    topos = [_topology(n * 4, 64, 16, seed=c) for c, n in enumerate(sizes)]
    ring, ref = Ring(4, 32, 0.1, DEVICE), Reference(4, 32, 0.1)
    for t in (1.0, 1.1, 1.2):
        sources = [
            _emission([[(t, g * 4 + i) for i in range(rng.choice([0, 0, 0, 1, 2]))] for g in range(n)], cap=4)
            for n in sizes
        ]
        intake(ring, sources, [True] * 3, [t + 0.1] * 3, topos, uniform_time=True)
        ref.intake(0, sources, [True] * 3, [t + 0.1] * 3, topos)
    _check(ring, ref)
    assert ring.main_count.sum().item() > 0


@pytest.mark.parametrize("uniform_time", [True, False])
def test_persistent_plan_reads_each_source_run_exactly_once(uniform_time):
    """With one IntakePlan over several rounds, intake decides on the device
    which sources ran since it last read them (source run id vs last-read
    id, double-buffered): a source that didn't run again must not be read
    again, even though its emission buffer still holds the old spikes."""
    topos = _toy_topologies()[:2]
    ring = Ring(2, 32, 0.1, DEVICE)
    a, b = _emission([[(1.0, 0)], [(1.0, 1)]], cap=4), _emission([[(1.0, 1)]], cap=4)
    plan = IntakePlan(ring, [a, b], topos)

    def ran(i, t, window_end):
        src = (a, b)[i]  # its spikes (the non-nan entries), restamped t
        src.time[:] = torch.where(torch.isnan(src.time), src.time, torch.full_like(src.time, t))
        _ran(src, window_end)

    def segments(t, s):
        return int(t.seg_offsets[s + 1] - t.seg_offsets[s])

    ran(0, 1.0, 1.1); ran(1, 1.0, 1.1)
    plan.run(uniform_time)   # both fresh
    ran(0, 1.1, 1.2)
    plan.run(uniform_time)   # only a
    plan.run(uniform_time)   # nothing ran
    ring.check()
    a_spike = segments(topos[0], 0) + segments(topos[0], 1)
    assert ring.main_count.sum().item() == 2 * a_spike + segments(topos[1], 1)


SIDE_BOUND_ROUNDS = ((1.2, 1.3), (1.0, 1.1), (1.25, 1.25))  # (t, window end)


def test_general_kernel_puts_below_watermark_and_delayed_batches_in_side_runs():
    """Equal-time batches that still need side runs: below the watermark
    (the round at 1.0 after 1.2) and emission-delayed (window end at the
    emission time)."""
    topos = _toy_topologies()
    ring, ref = Ring(2, 32, 0.1, DEVICE), Reference(2, 32, 0.1)
    for t, window_end in SIDE_BOUND_ROUNDS:
        sources = [_emission([[(t, 0)], [(t, 1)]], cap=4) for _ in range(3)]
        intake(ring, sources, [True] * 3, [window_end] * 3, topos)
        ref.intake(0, sources, [True] * 3, [window_end] * 3, topos)
    _check(ring, ref)
    assert ring.side_count.sum().item() == 2 * sum(t.n_segments for t in topos)


@pytest.mark.parametrize("t, window_end", SIDE_BOUND_ROUNDS[1:])
def test_uniform_kernel_rejects_batches_needing_side_runs(t, window_end):
    """uniform_time promises bare n2.spike() sources, whose spikes never
    need a side run; a batch that would must fail loudly, not mis-order."""
    topos = _toy_topologies()
    ring = Ring(2, 32, 0.1, DEVICE)
    first = [_emission([[(1.2, 0)]], cap=4) for _ in range(3)]
    intake(ring, first, [True] * 3, [1.3] * 3, topos, uniform_time=True)
    sources = [_emission([[(t, 0)], [(t, 1)]], cap=4) for _ in range(3)]
    with pytest.raises(RuntimeError, match="uniform contract"):
        intake(ring, sources, [True] * 3, [window_end] * 3, topos, uniform_time=True)


def test_arrivals_before_the_ring_position_land_in_its_first_slot():
    topos = _toy_topologies()
    ring, ref = Ring(2, 8, 0.1, DEVICE), Reference(2, 8, 0.1)
    ring.ring_pos = 4  # slots up to 3 already delivered
    sources, active, window_end = _toy_round(0.1)  # arrivals 0.2 .. 0.7 -> slots 2 .. 6
    intake(ring, sources, active, window_end, topos)
    ref.intake(4, sources, active, window_end, topos)
    _check(ring, ref)
    late = sum(len(ref.main[ring.cell(g, 4)]) for g in range(2))
    assert late > 0 and sum(ring.main_count[ring.cell(g, 4)].item() for g in range(2)) == late


def test_far_future_arrivals_raise_not_implemented():
    ring = Ring(2, 4, 0.1, DEVICE)  # covers only 0.4 ahead
    sources, active, window_end = _toy_round(0.0)
    with pytest.raises(NotImplementedError, match="far-future"):
        intake(ring, sources, active, window_end, _toy_topologies())


def test_full_run_raises():
    topos = [_topology(2, 8, 8, delay=0.15, k=8, seed=c) for c in range(3)]  # one group, one bucket
    ring = Ring(1, 8, 0.1, DEVICE, main_cap=5)
    sources, active, window_end = _toy_round(0.0)  # 6 spikes into the same cell
    with pytest.raises(RuntimeError, match="capacity"):
        intake(ring, sources, active, window_end, topos)


def test_uniform_intake_tracing_changes_nothing():
    """The traced uniform intake fills the ring exactly like the untraced one
    and stamps every taken-in spike."""
    from naer2.intake import INTAKE_TRACE_TAGS
    from naer2.trace import Trace

    rings = []
    for traced in (False, True):
        em = _emission([[(1.0, g * 8), (1.0, g * 8 + 1), (1.0, g * 8 + 5)] for g in range(4)], 16)
        n_src = 4 * 8
        _ran(em, 1.1)
        topo = _topology(n_src, 64, 32, delay=0.3, k=6, seed=3)
        ring = Ring(topo.n_groups, 8, 0.1, DEVICE)
        plan = IntakePlan(ring, [em], [topo])
        trace = Trace(plan.n_progs, 64) if traced else None
        plan.run(True, trace=trace)
        rings.append(ring.main_count.cpu().clone())
        if traced:
            progs = trace.decode(INTAKE_TRACE_TAGS)
            spikes = sum(1 for p in progs for tag, _, _ in p["events"] if tag == "inserted")
            assert spikes == 12
            assert all(p["events"][0][0] == "start" and p["events"][-1][0] == "end" for p in progs)
    assert torch.equal(rings[0], rings[1])
