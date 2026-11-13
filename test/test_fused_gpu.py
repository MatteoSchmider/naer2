"""
Tests of the fused green-field kernel (naer2/greenfield/fused.py): emission
into the K-deep history, pull intake + delivery end to end (every link in
the run that covers its slot, per group in slot then emission order,
summing exactly the right weights), the round visibility rule (results
don't depend on the order populations run in within a round), consumers
reading less often than their sources run, overrun detection, tracing.
"""

import math
import random

import pytest
import torch

import naer2.greenfield.fused as fused
from naer2.greenfield.fused import FusedPop
from naer2.topology_new import TopologyNew
from test.topologies import random_topology as _topology

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernel"
)

DEVICE = "cuda:0"
GROUP_SIZE = 32


@pytest.fixture(autouse=True, scope="module")
def _device():
    torch.cuda.set_device(0)


@pytest.fixture(autouse=True, params=[(256, 8), (32, 8)], ids=["default", "small-stream"])
def _walk(monkeypatch, request):
    stream, unroll = request.param
    monkeypatch.setattr(fused, "WALK", dict(stream=stream, unroll=unroll))


@pytest.fixture(autouse=True, params=[(True, True), (True, False), (False, True), (False, False)],
                ids=["batch-dir", "batch-search", "runs-dir", "runs-search"])
def _pull(monkeypatch, request):
    """Pull intake: fresh runs batched by emission time or one by one;
    segments through the group directory or by binary search."""
    batch, directory = request.param
    monkeypatch.setattr(fused, "PULL", dict(batch=batch, directory=directory))


def Pop(n, depth=3):
    return FusedPop(n, group_size=GROUP_SIZE, depth=depth, device=DEVICE)


def _fire_pattern(n, t_start, t_end, p, rng):
    """Per neuron: a fire time inside the window with probability p, else inf."""
    return torch.tensor([t_start + rng.random() * (t_end - t_start) if rng.random() < p else math.inf for _ in range(n)])


# ------------------------------------------------------------ emission


@pytest.mark.parametrize("uniform", [True, False])
def test_emission_into_the_history(uniform):
    rng = random.Random(0)
    pop = Pop(200)
    fire = _fire_pattern(200, 1.0, 1.1, 0.4, rng)
    delay = torch.tensor([rng.choice([0.0, 0.03, 0.4]) for _ in range(200)])
    pop.run(1.0, 1.1, fire, delay, uniform=uniform, round_id=7)
    e = [1.0 if uniform else float(torch.tensor(fire[i].item(), dtype=torch.float32) + torch.tensor(
        delay[i].item(), dtype=torch.float32)) for i in range(200)]
    fired = [i for i in range(200) if fire[i] < math.inf]
    got = {node: t for t, node in pop.emitted()}
    assert sorted(got) == fired and len(pop.emitted()) == len(fired)  # every spike once, no gaps
    assert all(got[i] == pytest.approx(e[i]) for i in fired)
    assert pop.history.summary(1) == (len(fired), pytest.approx(min(e[i] for i in fired)))
    h = pop.history.header
    assert int(h[1, 0]) == 1 and pop.history.round_of(1) == 7  # slot 1: run 1, round 7
    assert pop.history.window_end(1) == pytest.approx(1.1)
    assert int(h[2, 0]) == 0 and int(h[2, 2]) == 0  # the next run's slot, cleared ahead
    assert torch.allclose(pop.last_time, torch.full((200,), 1.1, device=DEVICE))


def test_history_keeps_recent_runs_only():
    rng = random.Random(1)
    pop = Pop(100, depth=3)
    fires = []
    for r in range(4):
        t = 1.0 + 0.1 * r
        fires.append(_fire_pattern(100, t, t + 0.1, 0.2, rng))
        pop.run(t, t + 0.1, fires[-1], round_id=r)
    assert sorted(n for _, n in pop.history.spikes(4)) == [i for i in range(100) if fires[3][i] < math.inf]
    assert sorted(n for _, n in pop.history.spikes(3)) == [i for i in range(100) if fires[2][i] < math.inf]
    with pytest.raises(KeyError):  # run 2's slot was cleared by run 4 (for run 5)
        pop.history.spikes(2)


# ------------------------------------------------------------ delivery


def _expected_deliveries(topo: TopologyNew, c: int, spikes, ring_pos: int, slot_width: float):
    """Per spike (e, source) of connection c taken in at ring position
    ring_pos: every link it delivers, as (slot, e, connection, source,
    target, weight)."""
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


def _check_deliveries(consumer, expected_by_read, delivered_by_run, due_until, n_targets):
    pending = [d for run in expected_by_read for d in run]
    for r, delivered in enumerate(delivered_by_run):
        lo = due_until[r - 1] + 1 if r else -10**9
        want = sorted((k, e, c, s, tg) for k, e, c, s, tg, _ in pending if lo <= k <= due_until[r])
        assert sorted(delivered) == want, f"run {r}"
        for grp in range(consumer.n_groups):
            order = [(k, e) for k, e, c, s, tg in delivered if tg // GROUP_SIZE == grp]
            assert order == sorted(order), f"run {r}, group {grp}: not in slot then emission order"
    assert any(delivered_by_run), "nothing was delivered"
    sums = torch.zeros(n_targets, dtype=torch.float64)
    counts = torch.zeros(n_targets, dtype=torch.int64)
    for k, e, c, s, tg, wt in pending:
        if k <= due_until[-1]:
            sums[tg] += wt
            counts[tg] += 1
    assert torch.equal(consumer.n_received.cpu().long(), counts)
    assert torch.allclose(consumer.received.cpu().double(), sums, atol=1e-4)


@pytest.mark.parametrize("consumer_first", [True, False])
@pytest.mark.parametrize("long_segments", [False, True])
@pytest.mark.parametrize("uniform", [True, False])
def test_pull_and_deliver_end_to_end(uniform, long_segments, consumer_first):
    """Sources A, B -> consumer C (3 groups), six rounds, every population
    running once per round -- C before or after its sources within the round
    (the outcome must not depend on it: C only ever sees earlier rounds).
    Spikes emitted in round r are taken in by C's run in round r + 1."""
    rng = random.Random(5)
    a, b, consumer = Pop(150), Pop(90), Pop(96)
    topologies = [  # per-link delays (A -> C) and one homogeneous delay (B -> C)
        _topology(a.n, 96, GROUP_SIZE, delay="random", k=24, seed=10, weight=True),
        _topology(b.n, 96, GROUP_SIZE, delay=0.2, k=90 if long_segments else 24, seed=11, weight=True),
    ]
    # run capacities count links (one ring record per link): emission-delayed spikes with long
    # segments put a few thousand links into a side run
    consumer.connect([a, b], topologies, n_slots=16, log_cap=200_000, side_cap=4096)

    expected_by_read, delivered_by_run = [], []
    for r in range(6):
        t = round(1.0 + 0.1 * r, 6)
        fires = {src: (_fire_pattern(src.n, t, t + 0.1, 0.3, rng),
                       torch.tensor([rng.choice([0.0, 0.0, 0.15]) for _ in range(src.n)])) for src in (a, b)}

        def run_consumer():
            ring_pos = consumer.ring.ring_pos  # where this run's intake lands late arrivals
            before = len(consumer.deliveries())
            consumer.run(t, t + 0.1, torch.full((96,), math.inf), uniform=uniform, round_id=r)
            delivered_by_run.append(consumer.deliveries()[before:])
            return ring_pos

        if consumer_first:
            ring_pos = run_consumer()
        for src in (a, b):
            src.run(t, t + 0.1, *fires[src], uniform=uniform, round_id=r)
        if not consumer_first:
            ring_pos = run_consumer()
        consumer.check()
        if r:  # this run took in the sources' previous round
            expected_by_read.append([d for c, src in enumerate((a, b)) for d in
                                     _expected_deliveries(topologies[c], c, prev_spikes[c], ring_pos, 0.1)])
        prev_spikes = [src.history.spikes(src.run_id) for src in (a, b)]
    due_until = [int(round((1.0 + 0.1 * r + 0.1) / 0.1)) - 1 for r in range(6)]
    _check_deliveries(consumer, expected_by_read, delivered_by_run, due_until, 96)


@pytest.mark.parametrize("depth", [4, 3])
def test_a_consumer_reading_every_other_round(depth):
    """The source runs every round, the consumer every other one (a 0.2 ms
    window), and the source first within the round -- the worst case of
    running concurrently: when the consumer reads, the source's run of the
    round is in its history but not visible, and one more slot is cleared.
    Two earlier runs pile up between reads: depth 4 keeps them, at depth 3
    the consumer must notice what it lost."""
    rng = random.Random(7)
    src, consumer = Pop(120, depth=depth), Pop(64)
    topo = _topology(src.n, 64, GROUP_SIZE, delay="random", k=20, seed=3, weight=True)
    consumer.connect([src], [topo], n_slots=24, log_cap=100_000)
    expected, delivered, pending_spikes = [], [], []
    for r in range(8):
        t = round(1.0 + 0.1 * r, 6)
        src.run(t, t + 0.1, _fire_pattern(src.n, t, t + 0.1, 0.3, rng), round_id=r)
        if r % 2 == 0:
            ring_pos = consumer.ring.ring_pos
            before = len(consumer.deliveries())
            consumer.run(t, t + 0.2, torch.full((64,), math.inf), round_id=r)
            delivered.append(consumer.deliveries()[before:])
            expected.append(_expected_deliveries(topo, 0, pending_spikes, ring_pos, 0.1))
            pending_spikes = []
        pending_spikes += src.history.spikes(src.run_id)  # visible from the next round on
    if depth == 3:
        with pytest.raises(RuntimeError, match="fell behind"):
            consumer.check()
        return
    consumer.check()
    due_until = [int(round((1.0 + 0.1 * r + 0.2) / 0.1)) - 1 for r in range(0, 8, 2)]
    _check_deliveries(consumer, expected, delivered, due_until, 64)


# ------------------------------------------------------------ tracing


def test_tracing_changes_nothing():
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
            consumer.run(t, t + 0.1, torch.full((96,), math.inf), round_id=r)
            if traced:
                runs.append(consumer.traced())
            a.run(t, t + 0.1, _fire_pattern(a.n, t, t + 0.1, 0.3, rng), round_id=r)
        results.append((consumer.n_received.cpu(), consumer.received.cpu(), runs))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.allclose(results[0][1], results[1][1], atol=1e-4)
    for run in results[1][2]:
        for prog in run:
            tags = [tag for tag, _, _ in prog["events"]]
            assert prog["dropped"] == 0 and tags[0] == "start" and tags[-1] == "end"
            assert {"pulled", "delivered", "sent"} <= set(tags)
            streamed = sum(aux for tag, aux, _ in prog["events"] if tag == "streamed")
            walked = sum(aux for tag, aux, _ in prog["events"] if tag == "walked")
            assert streamed == walked
