"""
Spike management: merging per-population blocked spike outputs into the
unblocked, time-sorted (spike_time, spike_src, spike_conn) buffer the update
kernel's event-driven spike loop consumes (see transpiler/templates/update_kernel.py).

The source side stays blocked — no compaction/flush happens in the source
population's kernel — because spike management has to touch every produced
spike anyway to sort and route it; compacting first would be a second,
redundant pass over the same data. See TODO.md for the fuller reasoning
(this is a reasoned-but-unbenchmarked choice: no GPU is available in this
environment, and the Triton CPU interpreter checks correctness, not cost).

Lifecycle note (not handled here, see TODO.md): a population's spike buffer
can hold entries beyond the current window (an explicit n2.spike(time+delay)
not due yet) that must survive to a later call, and entries already consumed
by a previous call that must not be double-counted. merge_blocked_spikes only
reads `sources` and filters by `window` — it never mutates them. Clearing
consumed entries and retaining not-yet-due ones is lifecycle management that
can only be done by the population the spikes belong to (it alone knows what
it has already handed out and what it's still holding), not by this merge
step, and isn't implemented yet.

Two reference implementations, both CPU/torch only, no Triton:
  merge_blocked_spikes       — the simplest correct thing: filter, concatenate
                                everything, one flat sort. The baseline/oracle.
  merge_blocked_spikes_kway  — same contract, structured as a k-way merge tree
                                instead, to validate that structure's
                                correctness before any GPU-specific technique
                                (per-stream odd-even-transposition sort
                                exploiting near-sortedness, pairwise
                                merge-path avoiding a full re-sort) gets built
                                on top of it. See its own docstring for what
                                it does and doesn't demonstrate yet.
"""

import torch


class BlockedSpikes:
    """
    One source population's raw blocked spike output, tagged with the
    spike_conn index it represents for a particular target population.

    node  [n_groups * cap] int64   — population-local source neuron index
                                     (what the kernel calls `offs`)
    time  [n_groups * cap] float32 — spike time
    count [n_groups]       int64   — valid entries in group g are
                                     node[g*cap : g*cap + count[g]] (and the
                                     same slice of time); anything past that
                                     is unwritten scratch, never read
    cap   int — LOCAL_SPIKE_CAP this population's kernel used
    conn  int — the spike_conn tag to stamp onto every entry from this
                source, for the target population this merge is building a
                buffer for (one BlockedSpikes per incoming connection, not
                per source population — the same population feeding two
                different connections needs two tags)
    """

    def __init__(self, node: torch.Tensor, time: torch.Tensor, count: torch.Tensor, cap: int, conn: int):
        self.node = node
        self.time = time
        self.count = count
        self.cap = cap
        self.conn = conn


def _empty() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(0, dtype=torch.float32),
        torch.zeros(0, dtype=torch.int64),
        torch.zeros(0, dtype=torch.int64),
    )


def _filtered_entries(
    s: BlockedSpikes, window: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This source's valid, windowed entries, flattened across its groups and
    tagged with its conn id. Returns (time, src, conn), possibly length 0."""
    t_lo, t_hi = window
    times, srcs = [], []
    n_groups = s.count.shape[0]
    for g in range(n_groups):
        c = int(s.count[g])
        if c == 0:
            continue
        base = g * s.cap
        t = s.time[base : base + c]
        keep = (t > t_lo) & (t <= t_hi)
        if not bool(keep.any()):
            continue
        times.append(t[keep])
        srcs.append(s.node[base : base + c][keep])

    if not times:
        return _empty()

    time = torch.cat(times)
    src = torch.cat(srcs)
    conn = torch.full((time.shape[0],), s.conn, dtype=torch.int64)
    return time, src, conn


def merge_blocked_spikes(
    sources: list[BlockedSpikes],
    window: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge several connections' blocked spike outputs into one unblocked,
    time-sorted (spike_time, spike_src, spike_conn) buffer, restricted to
    `window = (t_lo, t_hi]`: entries with `time <= t_lo` (already consumed by
    an earlier call) or `time > t_hi` (not due yet — e.g. an explicit delayed
    spike) are excluded. This function only reads `sources`; it has no way to
    tell the difference between "already consumed" and "not due yet" beyond
    the window bounds it's given, and makes no attempt to — see the lifecycle
    note at the top of this file.

    This is deliberately the simplest possible correct implementation —
    filter, concatenate every valid entry, then one full sort — meant to
    establish ground truth and prove the task is solvable from this input
    shape before any thought goes into what algorithm actually belongs in a
    kernel. It's the oracle `merge_blocked_spikes_kway` is checked against.

    It does NOT assume entries are already time-sorted within a group or
    across groups: an explicit n2.spike(time + delay) can emit a timestamp
    out of the order it was produced in, so group-local order can't be
    relied on, and this doesn't rely on it.

    Ties (equal timestamps, e.g. from different connections) come out stable
    with respect to `sources` order, but that's this implementation's
    incidental behavior, not a guarantee — nothing here or downstream
    currently depends on how ties are broken.
    """
    parts = [_filtered_entries(s, window) for s in sources]
    parts = [p for p in parts if p[0].shape[0] > 0]
    if not parts:
        return _empty()

    time = torch.cat([p[0] for p in parts])
    src = torch.cat([p[1] for p in parts])
    conn = torch.cat([p[2] for p in parts])
    order = torch.argsort(time, stable=True)
    return time[order], src[order], conn[order]


_Stream = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _sort_one_stream(time: torch.Tensor, src: torch.Tensor, conn: torch.Tensor) -> _Stream:
    """Sort one connection's filtered, windowed entries by time.

    Placeholder for an odd-even-transposition-sort-based approach that would
    exploit near-sortedness directly (cost tracking how far out of place
    entries are, not array size) — that's a GPU-kernel-level technique, left
    as a later refinement (see TODO.md). This just calls a generic sort.
    """
    order = torch.argsort(time, stable=True)
    return time[order], src[order], conn[order]


def _merge_pairwise(a: _Stream, b: _Stream) -> _Stream:
    """Merge two already-sorted-by-time streams into one sorted stream.

    On CPU this is a plain sort of the concatenation. A true O(n+m)
    merge-path algorithm (binary search along the merge diagonal) belongs in
    the eventual GPU kernel, not here — this function's job is to validate
    that pairing streams and recursing (the k-way merge TREE structure)
    produces the same result as the flat baseline, not yet to demonstrate the
    performance benefit of exploiting near-sortedness.
    """
    time = torch.cat([a[0], b[0]])
    src = torch.cat([a[1], b[1]])
    conn = torch.cat([a[2], b[2]])
    order = torch.argsort(time, stable=True)
    return time[order], src[order], conn[order]


def merge_blocked_spikes_kway(
    sources: list[BlockedSpikes],
    window: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Same contract and result as merge_blocked_spikes, but structured as a
    k-way merge tree: each connection's filtered entries are sorted once
    (_sort_one_stream), then combined via ceil(log2(k)) rounds of pairwise
    merges (_merge_pairwise) instead of one flat sort over everything.

    This exists to validate the TREE STRUCTURE — see
    test_kway_matches_naive_baseline in unit/test_spike_gather.py, which
    checks both implementations agree on the same inputs (as sets of
    entries; the two don't promise the same tie-break order). It does NOT
    yet demonstrate a performance difference from merge_blocked_spikes: both
    _sort_one_stream and _merge_pairwise fall back to a generic sort on CPU.
    Exploiting near-sortedness per-stream and avoiding a full re-sort per
    merge are both GPU-kernel-level techniques, out of scope here.

    The tree degenerates correctly at small k, with no special-casing needed:
    k=0 (no sources) returns empty; k=1 goes straight to _sort_one_stream,
    zero merge rounds; k=2 is one _merge_pairwise call, no tree; k=3 does two
    rounds (merge two of them, then merge that result with the third).
    """
    streams: list[_Stream] = []
    for s in sources:
        time, src, conn = _filtered_entries(s, window)
        if time.shape[0] == 0:
            continue
        streams.append(_sort_one_stream(time, src, conn))

    if not streams:
        return _empty()

    while len(streams) > 1:
        paired: list[_Stream] = []
        i = 0
        while i + 1 < len(streams):
            paired.append(_merge_pairwise(streams[i], streams[i + 1]))
            i += 2
        if i < len(streams):
            paired.append(streams[i])  # odd one out, carried to the next round untouched
        streams = paired

    return streams[0]
