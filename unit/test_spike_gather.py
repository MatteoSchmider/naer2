"""
Unit tests for naer2.spike_gather: merging blocked per-population spike
output into one unblocked, time-sorted (spike_time, spike_src, spike_conn)
buffer, restricted to a window.

Pure CPU, pure torch — no network, no Triton. This pins the task's contract
(the exact input/output shapes, and the window's semantics) and proves it's
solvable before any thought goes into what algorithm belongs in a kernel;
see naer2/spike_gather.py.
"""

import torch

from naer2.spike_gather import BlockedSpikes, merge_blocked_spikes, merge_blocked_spikes_kway

# "Include everything" window, for tests that aren't about windowing itself.
_ALL = (float("-inf"), float("inf"))


def _blocked(groups: list[list[tuple[float, int]]], cap: int, conn: int) -> BlockedSpikes:
    """Build a BlockedSpikes from groups = [[(time, local_src), ...], ...],
    one inner list per group, padding up to `cap` with unwritten scratch."""
    n_groups = len(groups)
    node = torch.full((n_groups * cap,), -1, dtype=torch.int64)  # scratch value, never read
    time = torch.full((n_groups * cap,), float("nan"), dtype=torch.float32)  # scratch value, never read
    count = torch.zeros(n_groups, dtype=torch.int64)
    for g, entries in enumerate(groups):
        assert len(entries) <= cap, "fixture error: more entries than the group's own cap"
        count[g] = len(entries)
        for i, (t, src) in enumerate(entries):
            time[g * cap + i] = t
            node[g * cap + i] = src
    return BlockedSpikes(node, time, count, cap, conn)


def _as_triples(time: torch.Tensor, src: torch.Tensor, conn: torch.Tensor) -> list[tuple[float, int, int]]:
    return list(zip(time.tolist(), src.tolist(), conn.tolist()))


def _as_sorted_set(time: torch.Tensor, src: torch.Tensor, conn: torch.Tensor) -> list[tuple[float, int, int]]:
    # for comparing the two implementations: same entries, same time order,
    # but they don't promise the same tie-break order, so sort fully before comparing
    return sorted(_as_triples(time, src, conn))


def test_single_source_single_group():
    a = _blocked([[(0.5, 3), (0.25, 7)]], cap=4, conn=0)
    time, src, conn = merge_blocked_spikes([a], _ALL)
    assert _as_triples(time, src, conn) == [(0.25, 7, 0), (0.5, 3, 0)]


def test_empty_groups_contribute_nothing():
    # a group with count 0 must contribute zero entries, even though its slot
    # still holds unwritten scratch (-1 / nan) beyond the count
    a = _blocked([[], [(0.25, 2)], []], cap=4, conn=0)
    time, src, conn = merge_blocked_spikes([a], _ALL)
    assert _as_triples(time, src, conn) == [(0.25, 2, 0)]


def test_full_group_at_capacity():
    # count == cap exactly — no off-by-one truncating the last valid entry
    a = _blocked([[(0.25, 0), (0.5, 1), (0.75, 2)]], cap=3, conn=0)
    time, src, conn = merge_blocked_spikes([a], _ALL)
    assert _as_triples(time, src, conn) == [(0.25, 0, 0), (0.5, 1, 0), (0.75, 2, 0)]


def test_out_of_order_within_group_still_globally_sorted():
    # an explicit n2.spike(time + delay) can land a later timestamp earlier in
    # the buffer than an in-order one emitted after it — group-local order is
    # not meaningful, and the merge must not assume it is
    a = _blocked([[(5.0, 0), (1.0, 1), (3.0, 2)]], cap=4, conn=0)
    time, src, conn = merge_blocked_spikes([a], _ALL)
    assert time.tolist() == sorted(time.tolist())
    assert _as_triples(time, src, conn) == [(1.0, 1, 0), (3.0, 2, 0), (5.0, 0, 0)]


def test_two_sources_merged_and_tagged_with_their_own_conn():
    a = _blocked([[(0.25, 0)], [(0.75, 1)]], cap=2, conn=0)
    b = _blocked([[(0.5, 0)], [(0.125, 1)]], cap=2, conn=1)
    time, src, conn = merge_blocked_spikes([a, b], _ALL)
    assert _as_triples(time, src, conn) == [
        (0.125, 1, 1),
        (0.25, 0, 0),
        (0.5, 0, 1),
        (0.75, 1, 0),
    ]


def test_three_or_more_sources_with_ragged_group_counts():
    # mismatched cap and group counts across sources, some empty groups —
    # exercises that nothing assumes every source has the same shape
    a = _blocked([[(1.0, 0)], [], [(4.0, 2)]], cap=2, conn=0)
    b = _blocked([[(2.0, 1), (2.5, 3)]], cap=4, conn=1)
    c = _blocked([[], [(3.0, 0)]], cap=1, conn=2)
    time, src, conn = merge_blocked_spikes([a, b, c], _ALL)
    assert time.tolist() == sorted(time.tolist())
    assert _as_triples(time, src, conn) == [
        (1.0, 0, 0),
        (2.0, 1, 1),
        (2.5, 3, 1),
        (3.0, 0, 2),
        (4.0, 2, 0),
    ]


def test_ties_keep_every_entry_not_just_one():
    a = _blocked([[(1.0, 5)]], cap=1, conn=0)
    b = _blocked([[(1.0, 9)]], cap=1, conn=1)
    time, src, conn = merge_blocked_spikes([a, b], _ALL)
    assert sorted(_as_triples(time, src, conn)) == [(1.0, 5, 0), (1.0, 9, 1)]


def test_total_count_matches_sum_of_group_counts():
    a = _blocked([[(1.0, 0), (2.0, 1)], [(3.0, 2)]], cap=4, conn=0)
    b = _blocked([[], [(4.0, 0)], [(5.0, 1), (6.0, 2)]], cap=4, conn=1)
    time, src, conn = merge_blocked_spikes([a, b], _ALL)
    assert len(time) == len(src) == len(conn) == 6


def test_no_sources_is_empty_not_an_error():
    time, src, conn = merge_blocked_spikes([], _ALL)
    assert len(time) == 0 and len(src) == 0 and len(conn) == 0


def test_all_empty_groups_is_empty_not_an_error():
    a = _blocked([[], []], cap=3, conn=0)
    time, src, conn = merge_blocked_spikes([a], _ALL)
    assert len(time) == 0 and len(src) == 0 and len(conn) == 0


# ── windowing ────────────────────────────────────────────────────────────────
# window = (t_lo, t_hi]: this is the whole point of conditioning the sort on a
# window at all — a population's buffer can hold spikes not due yet (an
# explicit n2.spike(time + delay)) that a later call must pick up instead, and
# this function must never silently include or drop the wrong ones.

def test_window_excludes_not_yet_due_future_spikes():
    # a delayed spike timestamped beyond t_hi must not appear in this call's
    # output — it's not due yet, a later call (with a later window) gets it
    a = _blocked([[(1.0, 0), (5.0, 1)]], cap=4, conn=0)
    time, src, conn = merge_blocked_spikes([a], window=(0.0, 2.0))
    assert _as_triples(time, src, conn) == [(1.0, 0, 0)]


def test_window_excludes_already_consumed_past_spikes():
    # a spike at or before t_lo belongs to a previous call's window and must
    # not reappear — defensive on this function's part, since a population
    # that cleared correctly wouldn't hand these back in anyway
    a = _blocked([[(1.0, 0), (3.0, 1)]], cap=4, conn=0)
    time, src, conn = merge_blocked_spikes([a], window=(2.0, 5.0))
    assert _as_triples(time, src, conn) == [(3.0, 1, 0)]


def test_window_boundaries_are_lo_exclusive_hi_inclusive():
    a = _blocked([[(2.0, 0), (5.0, 1)]], cap=4, conn=0)
    # exactly at t_lo: excluded (belongs to the window that just ended)
    time, src, conn = merge_blocked_spikes([a], window=(2.0, 5.0))
    assert _as_triples(time, src, conn) == [(5.0, 1, 0)]
    # exactly at t_hi: included (this is the window that reaches it)
    time, src, conn = merge_blocked_spikes([a], window=(0.0, 2.0))
    assert _as_triples(time, src, conn) == [(2.0, 0, 0)]


def test_window_with_multiple_sources_each_partially_filtered():
    # each source has one spike inside the window and one outside it, on
    # opposite sides — the filter is per-entry, not per-source
    a = _blocked([[(1.0, 0), (9.0, 1)]], cap=4, conn=0)  # 9.0 too late
    b = _blocked([[(-1.0, 0), (4.0, 1)]], cap=4, conn=1)  # -1.0 too early
    time, src, conn = merge_blocked_spikes([a, b], window=(0.0, 5.0))
    assert _as_triples(time, src, conn) == [(1.0, 0, 0), (4.0, 1, 1)]


def test_window_can_exclude_a_source_entirely():
    a = _blocked([[(1.0, 0)]], cap=2, conn=0)
    b = _blocked([[(100.0, 0)]], cap=2, conn=1)  # entirely outside the window
    time, src, conn = merge_blocked_spikes([a, b], window=(0.0, 5.0))
    assert _as_triples(time, src, conn) == [(1.0, 0, 0)]


def test_window_excluding_everything_is_empty_not_an_error():
    a = _blocked([[(1.0, 0), (2.0, 1)]], cap=4, conn=0)
    time, src, conn = merge_blocked_spikes([a], window=(10.0, 20.0))
    assert len(time) == 0 and len(src) == 0 and len(conn) == 0


# ── k-way merge tree: structural correctness against the flat baseline ───────
# merge_blocked_spikes_kway doesn't yet implement any of the GPU-specific
# techniques discussed (near-sortedness, merge-path) — on CPU both its
# building blocks fall back to a generic sort. What's being validated here is
# the TREE STRUCTURE itself: pairing streams and recursing must produce
# exactly the same set of entries as one flat sort over everything, for any k.

def _kway_matches_baseline(sources: list[BlockedSpikes], window: tuple[float, float]) -> None:
    expected = _as_sorted_set(*merge_blocked_spikes(sources, window))
    got = _as_sorted_set(*merge_blocked_spikes_kway(sources, window))
    assert got == expected
    time, _, _ = merge_blocked_spikes_kway(sources, window)
    assert time.tolist() == sorted(time.tolist())


def test_kway_matches_baseline_k0_no_sources():
    _kway_matches_baseline([], _ALL)


def test_kway_matches_baseline_k1_one_source():
    # zero merge rounds: goes straight to _sort_one_stream
    a = _blocked([[(5.0, 0), (1.0, 1), (3.0, 2)]], cap=4, conn=0)
    _kway_matches_baseline([a], _ALL)


def test_kway_matches_baseline_k2_one_merge():
    a = _blocked([[(0.25, 0)], [(0.75, 1)]], cap=2, conn=0)
    b = _blocked([[(0.5, 0)], [(0.125, 1)]], cap=2, conn=1)
    _kway_matches_baseline([a, b], _ALL)


def test_kway_matches_baseline_k3_odd_one_out_carried_forward():
    # two rounds: (stream0, stream1) merge, then merge that with stream2
    a = _blocked([[(1.0, 0)]], cap=2, conn=0)
    b = _blocked([[(2.0, 0)]], cap=2, conn=1)
    c = _blocked([[(0.5, 0)]], cap=2, conn=2)
    _kway_matches_baseline([a, b, c], _ALL)


def test_kway_matches_baseline_k5_multiple_rounds():
    sources = [_blocked([[(float(k * 10 + g), 0) for g in range(2)]], cap=2, conn=k) for k in range(5)]
    _kway_matches_baseline(sources, _ALL)


def test_kway_matches_baseline_with_window_and_ragged_groups():
    a = _blocked([[(1.0, 0)], [], [(4.0, 2), (9.0, 3)]], cap=2, conn=0)  # 9.0 outside window
    b = _blocked([[(2.0, 1), (2.5, 3)]], cap=4, conn=1)
    c = _blocked([[], [(3.0, 0)]], cap=1, conn=2)
    _kway_matches_baseline([a, b, c], window=(0.0, 5.0))


def test_kway_matches_baseline_out_of_order_within_stream():
    # disorder within one stream must still end up globally sorted after
    # _sort_one_stream, same as the baseline requires of its own input
    a = _blocked([[(5.0, 0), (1.0, 1), (3.0, 2)]], cap=4, conn=0)
    b = _blocked([[(4.0, 0), (2.0, 1)]], cap=4, conn=1)
    _kway_matches_baseline([a, b], _ALL)
