"""The uniform-emission tests of test_update_emission_gpu.py, rerun on the
Gluon update kernel (test/update_gluon.py)."""

import pytest

import test.test_update_emission_gpu as T


@pytest.fixture(autouse=True)
def _gluon(monkeypatch):
    monkeypatch.setattr(T, "BACKEND", "gluon")


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
