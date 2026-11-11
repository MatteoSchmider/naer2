"""
Tests of the green-field update kernel (naer2/greenfield): emission (the
EmissionBuffer contract: per group sorted, flat list, summary, state) and
delivery (every link in the run that covers its slot, per group in slot
then emission order, summing exactly the right weights), fed through the
real intake kernel and checked against test_intake_gpu.py's reference end
to end. Every test runs with three delivery-walk configurations.
"""

import math
import random

import pytest
import torch

import naer2.greenfield.pop as greenfield_pop
from naer2.greenfield import Pop as _Pop
from naer2.intake import IntakePlan, Ring
from naer2.topology_new import TopologyNew
from test.test_intake_gpu import Reference, _check, _topology

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernel"
)

DEVICE = "cuda:0"
GROUP_SIZE = 32


@pytest.fixture(autouse=True, scope="module")
def _device():
    torch.cuda.set_device(0)


@pytest.fixture(autouse=True, params=[(128, 256, 8), (32, 32, 8), (64, 128, 16)],
                ids=["default", "small-stream", "chunk64-u16"])
def _walk(monkeypatch, request):
    """Every test with the default delivery walk, a stream shorter than a
    chunk's links (chunks continue in further stream blocks), and a smaller
    chunk with a wider walk group."""
    entry_block, stream, unroll = request.param
    monkeypatch.setattr(greenfield_pop, "WALK", dict(entry_block=entry_block, stream=stream, unroll=unroll))


def Pop(n, cap=64, group_size=GROUP_SIZE):
    return _Pop(n, cap=cap, group_size=group_size, device=DEVICE)


# ------------------------------------------------------------ emission


def _fire_pattern(n, t_start, t_end, p, rng):
    """Per neuron: a fire time inside the window with probability p, else inf."""
    return torch.tensor([t_start + rng.random() * (t_end - t_start) if rng.random() < p else math.inf for _ in range(n)])


def test_uniform_emission_buffer_state_and_summary():
    rng = random.Random(0)
    pop = Pop(200)
    fire = _fire_pattern(200, 1.0, 1.1, 0.3, rng)
    pop.run(1.0, 1.1, fire, uniform=True)

    want = sorted((1.0, i) for i in range(200) if fire[i] < math.inf)
    got = sorted(entry for entries in pop.emitted().values() for entry in entries)
    assert [(pytest.approx(t), i) for t, i in want] == got
    for g, entries in pop.emitted().items():
        assert all(g * GROUP_SIZE <= i < (g + 1) * GROUP_SIZE for _, i in entries)
    assert pop.emission.state[0].item() == 1
    assert pop.emission.state[1:].view(torch.float32).item() == pytest.approx(1.1)
    assert pop.emission.summary(1) == (len(want), pytest.approx(1.0))
    assert sorted(pop.flat()) == got  # the flat list: the same spikes, no gaps
    assert torch.allclose(pop.last_time, torch.full((200,), 1.1, device=DEVICE))


def test_exact_times_with_emission_delays_are_sorted_per_group():
    rng = random.Random(1)
    pop = Pop(200)
    fire = _fire_pattern(200, 1.0, 1.1, 0.5, rng)
    delay = torch.tensor([rng.choice([0.0, 0.0, 0.03, 0.4]) for _ in range(200)])
    pop.run(1.0, 1.1, fire, delay, uniform=False)

    e = (fire.to(torch.float32) + delay.to(torch.float32)).tolist()  # the kernel's float32 sum
    for g, entries in pop.emitted().items():
        times = [t for t, _ in entries]
        assert times == sorted(times), f"group {g} not sorted: {times}"
        want = sorted((e[i], i) for i in range(g * GROUP_SIZE, min((g + 1) * GROUP_SIZE, 200)) if fire[i] < math.inf)
        assert sorted(entries) == want
    fired = [i for i in range(200) if fire[i] < math.inf]
    assert pop.emission.summary(1) == (len(fired), min(e[i] for i in fired))
    assert sorted(pop.flat()) == sorted((e[i], i) for i in fired)


def test_summary_is_per_run():
    rng = random.Random(2)
    pop = Pop(100)
    pop.run(1.0, 1.1, _fire_pattern(100, 1.0, 1.1, 0.5, rng))
    fire2 = _fire_pattern(100, 1.1, 1.2, 0.1, rng)
    pop.run(1.1, 1.2, fire2)
    assert pop.emission.summary(2) == (int((fire2 < math.inf).sum()), pytest.approx(1.1))
    assert sorted(n for _, n in pop.flat()) == [i for i in range(100) if fire2[i] < math.inf]  # run 2's only
    pop.run(1.2, 1.3, torch.full((100,), math.inf))
    assert pop.emission.summary(3) == (0, math.inf)


def test_group_overflow_is_flagged_not_written_past():
    pop = Pop(64, cap=8)
    pop.run(1.0, 1.1, torch.full((64,), 1.05))  # 32 spikes per group, cap 8
    assert pop.emission.overflow.item() == 1
    assert pop.emission.count.tolist() == [8, 8]


@pytest.mark.parametrize("uniform", [True, False])
def test_emission_feeds_intake_end_to_end(uniform):
    """Update kernel -> emission buffer -> real intake kernel -> ring, over
    three rounds (the second population doesn't run in round 2), against
    the intake reference."""
    rng = random.Random(3)
    a, b = Pop(150), Pop(90)
    # into a consumer of 64 neurons in 4 groups: per-link and homogeneous delay
    topologies = [_topology(150, 64, 16, seed=0), _topology(90, 64, 16, delay=0.2, seed=1)]
    ring, ref = Ring(4, 32, 0.1, DEVICE), Reference(4, 32, 0.1)
    plan = IntakePlan(ring, [a.emission, b.emission], topologies)
    for r, t in enumerate((1.0, 1.1, 1.2)):
        ran = [a] if r == 1 else [a, b]
        for pop in ran:
            pop.run(t, t + 0.1, _fire_pattern(pop.n, t, t + 0.1, 0.3, rng),
                    torch.tensor([rng.choice([0.0, 0.0, 0.2]) for _ in range(pop.n)]), uniform=uniform)
        plan.run(uniform)
        ref.intake(0, [a.emission, b.emission], [p in ran for p in (a, b)], [t + 0.1] * 2, topologies)
    ring.check()
    _check(ring, ref)
    assert ring.main_count.sum().item() > 0


def test_kernel_takes_a_segment_topology():
    """The delivery interface: a real incoming connection's TopologyNew is
    accepted (walking it is the next step)."""
    pop = Pop(96)
    pop.topo = _topology(50, 96, GROUP_SIZE, seed=4)
    pop.run(1.0, 1.1, torch.full((96,), math.inf))
    assert pop.emission.summary(1) == (0, math.inf)


# ------------------------------------------------------------ delivery


def _expected_deliveries(topo: TopologyNew, c: int, spikes, ring_pos: int, slot_width: float):
    """Per spike (e, source) of connection c taken in at ring position
    ring_pos: every link it delivers, as (slot, e, connection, source,
    target, weight) -- through its segments, like intake and the walk."""
    w = torch.tensor(slot_width, dtype=torch.float32)
    offsets, start, gb = topo.seg_offsets.tolist(), topo.seg_start.tolist(), topo.seg_gb.tolist()
    mean, target, weight = topo.bucket_mean.tolist(), topo.link_target.tolist(), topo.attrs["weight"].tolist()
    out = []
    for e, s in spikes:
        for seg in range(offsets[s], offsets[s + 1]):
            group, j = (gb[seg] >> 16) & 0xFFFF, gb[seg] & 0xFFFF
            arrival = torch.tensor(e, dtype=torch.float32) + torch.tensor(mean[j], dtype=torch.float32)
            k = max(int(torch.floor(arrival / w)), ring_pos)
            for link in range(start[seg], start[seg + 1]):
                out.append((k, e, c, s, group * topo.group_size + (target[link] & 0xFFFF), weight[link]))
    return out


@pytest.mark.parametrize("long_segments", [False, True])
@pytest.mark.parametrize("uniform", [True, False])
def test_deliveries_arrive_in_their_run_in_slot_then_emission_order(uniform, long_segments):
    """Sources A, B -> consumer C (3 groups) over six rounds, in network
    order: every population's update kernel (C delivers its due slots),
    then intake. Every delivered link must come out in the run that covers
    its slot, per group in slot order and within a slot in emission order,
    and C's stand-in receive must sum exactly those weights."""
    rng = random.Random(5)
    a, b, consumer = Pop(150), Pop(90), Pop(96)
    topologies = [  # per-link delays (A -> C) and one homogeneous delay (B -> C)
        _topology(a.n, 96, GROUP_SIZE, delay="random", k=24, seed=10, weight=True),
        # long_segments: ~30 links per (source, group), several U-link walk steps per entry
        _topology(b.n, 96, GROUP_SIZE, delay=0.2, k=90 if long_segments else 24, seed=11, weight=True),
    ]
    consumer.connect([a, b], topologies, n_slots=16, log_cap=200_000)

    expected_by_run, delivered_by_run = [], []
    for r in range(6):
        t = round(1.0 + 0.1 * r, 6)
        before = len(consumer.deliveries())
        consumer.run(t, t + 0.1, torch.full((96,), math.inf), uniform=uniform)
        delivered_by_run.append(consumer.deliveries()[before:])
        for src in (a, b):
            src.run(t, t + 0.1, _fire_pattern(src.n, t, t + 0.1, 0.3, rng),
                    torch.tensor([rng.choice([0.0, 0.0, 0.15]) for _ in range(src.n)]), uniform=uniform)
        ring_pos = consumer.ring.ring_pos
        consumer.plan.run(uniform)
        consumer.ring.check()
        spikes = [[(e, s) for entries in src.emitted().values() for e, s in entries] for src in (a, b)]
        expected_by_run.append([d for c in range(2) for d in
                                _expected_deliveries(topologies[c], c, spikes[c], ring_pos, 0.1)])

    # which run delivers which slot: run r delivers slots up to (t_r + 0.1) / 0.1 - 1
    due_until = [int(round((1.0 + 0.1 * r + 0.1) / 0.1)) - 1 for r in range(6)]
    pending = [d for run in expected_by_run for d in run]
    for r, delivered in enumerate(delivered_by_run):
        lo = due_until[r - 1] + 1 if r else -10**9
        want = sorted((k, e, c, s, tg) for k, e, c, s, tg, _ in pending if lo <= k <= due_until[r])
        got = sorted(delivered)
        assert got == want, f"run {r}"
        for grp in range(consumer.n_groups):
            order = [(k, e) for k, e, c, s, tg in delivered if tg // GROUP_SIZE == grp]
            assert order == sorted(order), f"run {r}, group {grp}: not in slot then emission order"

    delivered_all = [d for run in delivered_by_run for d in run]
    assert delivered_all, "nothing was delivered"
    sums = torch.zeros(96, dtype=torch.float64)
    counts = torch.zeros(96, dtype=torch.int64)
    for k, e, c, s, tg, wt in pending:
        if k <= due_until[-1]:
            sums[tg] += wt
            counts[tg] += 1
    assert torch.equal(consumer.n_received.cpu().long(), counts)
    assert torch.allclose(consumer.received.cpu().double(), sums, atol=1e-4)
    # delivered cells are freed: every cell not holding a still-pending slot is empty
    still_pending = {consumer.ring.cell(tg // GROUP_SIZE, k) for k, e, c, s, tg, _ in pending if k > due_until[-1]}
    counts_now = (consumer.ring.main_count + consumer.ring.side_count).tolist()
    assert all(n == 0 for cell, n in enumerate(counts_now) if cell not in still_pending)
    assert all(counts_now[cell] > 0 for cell in still_pending)


# ------------------------------------------------------------ tracing


def test_tracing_changes_nothing():
    """The traced kernel delivers what the untraced one does and stamps
    every phase; the walk accounts for every link staged into the stream."""
    results = []
    for traced in (False, True):
        rng = random.Random(5)
        a, consumer = Pop(150), Pop(96)
        topo = _topology(a.n, 96, GROUP_SIZE, delay="random", k=40, seed=10, weight=True)
        consumer.connect([a], [topo], n_slots=16)
        if traced:
            consumer.set_trace(512)
        runs = []
        for r in range(6):
            t = round(1.0 + 0.1 * r, 6)
            consumer.run(t, t + 0.1, torch.full((96,), math.inf))
            if traced:
                runs.append(consumer.traced())
            a.run(t, t + 0.1, _fire_pattern(a.n, t, t + 0.1, 0.3, rng), uniform=True)
            consumer.plan.run(True)
        results.append((consumer.n_received.cpu(), consumer.received.cpu(), runs))
    assert torch.equal(results[0][0], results[1][0])
    # uniform intake appends equal-e entries by parallel atomics: their order,
    # hence the float summation order, varies run to run
    assert torch.allclose(results[0][1], results[1][1], atol=1e-4)
    for run in results[1][2]:
        for prog in run:
            tags = [tag for tag, _, _ in prog["events"]]
            assert prog["dropped"] == 0 and tags[0] == "start" and tags[-1] == "end"
            assert {"state", "delivered", "sent"} <= set(tags)
            assert prog["end"] >= prog["start"] > 0
            streamed = sum(aux for tag, aux, _ in prog["events"] if tag == "streamed")
            walked = sum(aux for tag, aux, _ in prog["events"] if tag == "walked")
            assert streamed == walked
