"""The uniform-emission tests of test_update_emission_gpu.py, rerun on the
Gluon update kernel (test/update_gluon.py)."""

import pytest

import test.test_update_emission_gpu as T


@pytest.fixture(autouse=True, params=[(128, 256, 8), (32, 32, 8), (64, 128, 16)],
                ids=["default", "small-stream", "chunk64-u16"])
def _gluon(monkeypatch, request):
    """Every test on the Gluon kernel: its default walk, a stream shorter
    than a chunk's links (chunks continue in further stream blocks), a
    smaller chunk with a wider walk group."""
    monkeypatch.setattr(T, "BACKEND", "gluon")
    entry_block, stream, unroll = request.param
    monkeypatch.setitem(T.WALK, "gluon", dict(entry_block=entry_block, stream=stream, unroll=unroll))


def test_uniform_emission_buffer_state_and_summary():
    T.test_uniform_emission_buffer_state_and_summary()


def test_summary_is_per_run():
    T.test_summary_is_per_run()


def test_group_overflow_is_flagged_not_written_past():
    T.test_group_overflow_is_flagged_not_written_past()


def test_emission_feeds_intake_end_to_end():
    T.test_emission_feeds_intake_end_to_end(True)


@pytest.mark.parametrize("long_segments", [False, True])
def test_deliveries_arrive_in_their_run_in_slot_then_emission_order(long_segments):
    T.test_deliveries_arrive_in_their_run_in_slot_then_emission_order(True, long_segments)


def test_tracing_changes_nothing():
    """The traced Gluon kernel delivers what the untraced one does and
    stamps every phase."""
    import math
    import random

    import torch

    results = []
    for traced in (False, True):
        rng = random.Random(5)
        a, consumer = T.Pop(150), T.Pop(96)
        topo = T._topology(a.n, 96, T.GROUP_SIZE, delay="random", k=40, seed=10, weight=True)
        consumer.connect([a], [topo], n_slots=16)
        if traced:
            consumer.set_trace(1, 512)
        runs = []
        for r in range(6):
            t = round(1.0 + 0.1 * r, 6)
            consumer.run(t, t + 0.1, torch.full((96,), math.inf))
            if traced:
                runs.append(consumer.traced())
            a.run(t, t + 0.1, T._fire_pattern(a.n, t, t + 0.1, 0.3, rng), uniform=True)
            consumer.plan.run(True)
        results.append((consumer.n_received.cpu(), consumer.received.cpu(), runs))
    assert torch.equal(results[0][0], results[1][0])
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
