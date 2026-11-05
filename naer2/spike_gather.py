"""
Spike management, GPU-resident: two Triton kernels, no CPU fallback.

Measured on real hardware (see TODO.md's "Stay on GPU" entry) that a
GPU-resident sort beats a CPU round trip at every size tested, including the
smallest -- so there's no CPU reference to validate against here, unlike
naer2/population.py's node/edge logic (which is checked against the Triton
CPU interpreter). These kernels are checked by actually running them on a
GPU; see test/test_spike_gather_gpu.py (skipped without one).

Two separate concerns, two separate kernels:

  compact() -- owned by a single source population. Combines whatever it's
    still holding from previous calls (`carry`, already sorted) with this
    call's freshly emitted, GROUP-blocked spikes (`fresh`, a BlockedSpikes:
    not sorted, not even unblocked), splits the result on `t_hi` into "due
    now" and "not yet due", returns the due entries (flat, sorted, ready to
    read) and OVERWRITES carry in place with the compacted remainder. A
    source commonly feeds multiple targets (e.g. one population feeding two
    different connections) -- compacting (and sorting) it once here, rather
    than once per downstream consumer, is the whole point of this split.
    `conn` tagging is NOT done here: it's a per-(target, connection)
    concept, not something intrinsic to a source's own output, so it's
    merge()'s job, not compact()'s.

  merge() -- owned by a single target population. Takes several already-
    compacted (flat, sorted, untagged) sources, one per incoming connection,
    tags each with that connection's id, and interleaves them into the one
    sorted (time, src, conn) buffer the target's own update kernel consumes
    in its per-spike dispatch loop. Because every input is already sorted,
    this never re-sorts a source stream that compact() already sorted for a
    previous consumer -- it only needs to combine already-sorted streams.
    (For now it does that by re-sorting the concatenation rather than a true
    merge-path; see the module docstring note below each kernel for what's
    deferred.)

All kernels run as a single Triton program (grid=(1,)).

compact() does NOT use tl.sort: tl.sort builds its whole bitonic network
unrolled at trace time over one TOTAL_CAP-wide register tile, and at real
population sizes (TOTAL_CAP 65536 for PD14) that took tens of minutes to
compile. Instead it works through global memory in BLOCK-sized chunks with
real loops: densify fresh, sort it (a looped bitonic network -- skipped
entirely when the population only emits bare n2.spike(), see
Population.fresh_sorted), then merge it with the already-sorted carry by
binary-search rank. Sizes are runtime arguments, so one compiled kernel
serves every population size. Output is sorted by time only; ties have no
defined order.

merge()/combine() still use the original trick to turn a sort into a single
`tl.sort` call (which only orders one tensor of keys, no attached payload):
pack every field being carried into one sortable integer key, sort that,
unpack after.
float times are made integer-sortable via the standard order-preserving bit
flip (flip the sign bit for positives, invert all bits for negatives);
everything else (a population-local node index, a connection id) is small
enough to ride along packed into the key's low bits.
"""

import torch
import triton
import triton.language as tl

_SIGN_BIT = tl.constexpr(0x80000000)
_ALL_ONES = tl.constexpr(0xFFFFFFFF)
_INF = tl.constexpr(float("inf"))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _to_ordered_u32(time):
    """float32 bits -> uint32 that sorts the same way the floats do."""
    bits = time.to(tl.uint32, bitcast=True)
    sign_set = (bits & _SIGN_BIT) != 0
    mask = tl.where(sign_set, _ALL_ONES, _SIGN_BIT).to(tl.uint32)
    return bits ^ mask


@triton.jit
def _from_ordered_u32(u):
    """Inverse of _to_ordered_u32."""
    sign_set = (u & _SIGN_BIT) == 0
    mask = tl.where(sign_set, _ALL_ONES, _SIGN_BIT).to(tl.uint32)
    return (u ^ mask).to(tl.float32, bitcast=True)


class BlockedSpikes:
    """One source population's raw, freshly emitted, GROUP-blocked spike
    output for this call -- compact()'s `fresh` argument. Not sorted, not
    even unblocked; see naer2/population.py (not yet wired up) for where
    this would come from for real.

    node  [n_groups * cap] int32   -- population-local source neuron index
    time  [n_groups * cap] float32 -- spike time
    count [n_groups]       int32   -- valid entries in group g are
                                      node[g*cap : g*cap + count[g]] (and the
                                      same slice of time); anything past that
                                      is unwritten scratch, never read
    cap   int -- LOCAL_SPIKE_CAP this population's kernel used
    """

    def __init__(self, node: torch.Tensor, time: torch.Tensor, count: torch.Tensor, cap: int):
        self.node = node
        self.time = time
        self.count = count
        self.cap = cap


def empty_blocked_spikes(n_groups: int, cap: int, device) -> BlockedSpikes:
    """A BlockedSpikes reporting zero spikes for every group -- used as the
    "nothing new this round" placeholder (Population's first-ever gather
    call, and every round a population's update kernel doesn't run)."""
    node = torch.zeros((n_groups * cap,), device=device, dtype=torch.int32)
    time = torch.zeros((n_groups * cap,), device=device, dtype=torch.float32)
    count = torch.zeros((n_groups,), device=device, dtype=torch.int32)
    return BlockedSpikes(node, time, count, cap)


class CarryBuffer:
    """A population's persistent, cross-call lifecycle state: whatever it's
    holding that wasn't due as of its last compact() call, already sorted.
    compact() overwrites `time`/`node`/`count` in place every call.

    time/node are pre-allocated to `cap`; only the first `count` entries are
    valid, and they're always sorted ascending by time -- that invariant is
    exactly why merge() never needs to re-sort a compacted source.
    """

    def __init__(self, cap: int, device):
        self.cap = cap
        self.time = torch.full((cap,), float("inf"), device=device, dtype=torch.float32)
        self.node = torch.zeros((cap,), device=device, dtype=torch.int32)
        self.count = torch.zeros((1,), device=device, dtype=torch.int32)


class CompactedSpikes:
    """One source's compact()ed, flat, sorted, untagged "due now" output --
    merge()'s per-connection input. Only `time`/`node`[:count] are valid."""

    def __init__(self, time: torch.Tensor, node: torch.Tensor, count: torch.Tensor):
        self.time = time
        self.node = node
        self.count = count


@triton.jit
def _bit_length(n):
    """Number of binary-search halvings that always converge over [0, n]."""
    b = tl.cast(0, tl.int32)
    while n > 0:
        n = n >> 1
        b += 1
    return b


@triton.jit
def _rank(arr_ptr, n, x, iters, INCLUSIVE: tl.constexpr):
    """Per element of x: how many of the sorted arr[0:n] are < x (or <= x if
    INCLUSIVE). Vectorized binary search as a real loop over global memory --
    a register tensor can't be gathered from at arbitrary indices."""
    lo = tl.zeros_like(x).to(tl.int32)
    hi = lo + n
    for _ in range(iters):
        active = lo < hi
        mid = (lo + hi) // 2
        v = tl.load(arr_ptr + mid, mask=active, other=0.0)
        if INCLUSIVE:
            right = v <= x
        else:
            right = v < x
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    return lo


@triton.jit
def _bitonic_sort_inplace(time_ptr, node_ptr, n, BLOCK: tl.constexpr):
    """Sort time_ptr[0:n] ascending (node_ptr riding along), in place, as a
    bitonic network over a power-of-two padded scratch buffer (caller
    guarantees capacity). Every stage is the same compare-exchange at a
    different stride, so this is a real loop through global memory -- not
    tl.sort's trace-time-unrolled register network, whose compile time
    explodes with width."""
    p = tl.cast(1, tl.int32)
    while p < n:
        p *= 2
    for start in range(n, p, BLOCK):
        i = start + tl.arange(0, BLOCK)
        tl.store(time_ptr + i, _INF, mask=i < p)
    tl.debug_barrier()

    k = tl.cast(2, tl.int32)
    while k <= p:
        j = k // 2
        while j > 0:
            for start in range(0, p, BLOCK):
                i = start + tl.arange(0, BLOCK)
                partner = i ^ j
                # each pair is owned by its lower index, so no lane in this
                # stage ever touches an element another lane writes
                lead = (i < p) & (partner > i)
                ti = tl.load(time_ptr + i, mask=lead, other=0.0)
                tp = tl.load(time_ptr + partner, mask=lead, other=0.0)
                ni = tl.load(node_ptr + i, mask=lead, other=0)
                np_ = tl.load(node_ptr + partner, mask=lead, other=0)
                ascending = (i & k) == 0
                swap = lead & tl.where(ascending, ti > tp, ti < tp)
                tl.store(time_ptr + i, tp, mask=swap)
                tl.store(time_ptr + partner, ti, mask=swap)
                tl.store(node_ptr + i, np_, mask=swap)
                tl.store(node_ptr + partner, ni, mask=swap)
            tl.debug_barrier()
            j = j // 2
        k *= 2


@triton.jit
def _emit_merged(
    pos, time, node, m, due_count,
    due_time_ptr, due_node_ptr, new_carry_time_ptr, new_carry_node_ptr, CARRY_CAP,
):
    is_due = pos < due_count
    tl.store(due_time_ptr + pos, time, mask=m & is_due)
    tl.store(due_node_ptr + pos, node, mask=m & is_due)
    cpos = pos - due_count
    keep = m & ~is_due & (cpos < CARRY_CAP)
    tl.store(new_carry_time_ptr + cpos, time, mask=keep)
    tl.store(new_carry_node_ptr + cpos, node, mask=keep)


@triton.jit
def _compaction_kernel(
    carry_time_ptr, carry_node_ptr, carry_count_ptr,
    fresh_time_ptr, fresh_node_ptr, fresh_count_ptr,
    dense_time_ptr, dense_node_ptr, group_off_ptr,
    due_time_ptr, due_node_ptr, due_count_ptr,
    new_carry_time_ptr, new_carry_node_ptr,
    t_hi, N_GROUPS, CARRY_CAP,
    GROUP_CAP: tl.constexpr, FRESH_SORTED: tl.constexpr, BLOCK: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK)

    # 1. densify fresh: exclusive prefix sum of per-group counts gives each
    #    group's offset in one contiguous array
    running = tl.cast(0, tl.int32)
    for gs in range(0, N_GROUPS, BLOCK):
        gi = gs + lanes
        gm = gi < N_GROUPS
        c = tl.minimum(tl.load(fresh_count_ptr + gi, mask=gm, other=0), GROUP_CAP)
        tl.store(group_off_ptr + gi, running + tl.cumsum(c, axis=0) - c, mask=gm)
        running += tl.sum(c, axis=0)
    n_fresh = running
    tl.debug_barrier()

    for s0 in range(0, N_GROUPS * GROUP_CAP, BLOCK):
        s = s0 + lanes
        g = s // GROUP_CAP
        local = s % GROUP_CAP
        in_range = s < N_GROUPS * GROUP_CAP
        c = tl.minimum(tl.load(fresh_count_ptr + g, mask=in_range, other=0), GROUP_CAP)
        v = in_range & (local < c)
        dst = tl.load(group_off_ptr + g, mask=v, other=0) + local
        tl.store(dense_time_ptr + dst, tl.load(fresh_time_ptr + s, mask=v), mask=v)
        tl.store(dense_node_ptr + dst, tl.load(fresh_node_ptr + s, mask=v), mask=v)
    tl.debug_barrier()

    # 2. fresh must be sorted before merging; with only bare n2.spike() in a
    #    population's hooks every fresh time is the same `t`, already sorted
    if not FRESH_SORTED:
        _bitonic_sort_inplace(dense_time_ptr, dense_node_ptr, n_fresh, BLOCK)
        tl.debug_barrier()

    # 3. merge sorted carry with sorted fresh; since both are sorted, the
    #    due/carry split point is just how many of each are <= t_hi
    n_carry = tl.minimum(tl.load(carry_count_ptr), CARRY_CAP)
    due_count = tl.cast(0, tl.int32)
    for start in range(0, n_carry, BLOCK):
        i = start + lanes
        due_count += tl.sum((tl.load(carry_time_ptr + i, mask=i < n_carry, other=_INF) <= t_hi).to(tl.int32), axis=0)
    for start in range(0, n_fresh, BLOCK):
        i = start + lanes
        due_count += tl.sum((tl.load(dense_time_ptr + i, mask=i < n_fresh, other=_INF) <= t_hi).to(tl.int32), axis=0)

    # an element's merged position = its own index + how many of the other
    # sequence precede it (ties: carry first, so carry counts fresh strictly
    # less, fresh counts carry less-or-equal -- every position is unique)
    iters_fresh = _bit_length(n_fresh)
    iters_carry = _bit_length(n_carry)
    for start in range(0, n_carry, BLOCK):
        i = start + lanes
        m = i < n_carry
        time = tl.load(carry_time_ptr + i, mask=m, other=_INF)
        node = tl.load(carry_node_ptr + i, mask=m, other=0)
        pos = i + _rank(dense_time_ptr, n_fresh, time, iters_fresh, INCLUSIVE=False)
        _emit_merged(pos, time, node, m, due_count,
                     due_time_ptr, due_node_ptr, new_carry_time_ptr, new_carry_node_ptr, CARRY_CAP)
    for start in range(0, n_fresh, BLOCK):
        i = start + lanes
        m = i < n_fresh
        time = tl.load(dense_time_ptr + i, mask=m, other=_INF)
        node = tl.load(dense_node_ptr + i, mask=m, other=0)
        pos = i + _rank(carry_time_ptr, n_carry, time, iters_carry, INCLUSIVE=True)
        _emit_merged(pos, time, node, m, due_count,
                     due_time_ptr, due_node_ptr, new_carry_time_ptr, new_carry_node_ptr, CARRY_CAP)

    tl.store(due_count_ptr, due_count)
    # unclamped on purpose: a count above CARRY_CAP marks an overflow (the
    # excess was dropped above) instead of hiding it
    tl.store(carry_count_ptr, n_carry + n_fresh - due_count)


_COMPACT_BLOCK = 1024


def compact(fresh: BlockedSpikes, carry: CarryBuffer, t_hi: float, fresh_sorted: bool = False) -> CompactedSpikes:
    """Fold `fresh` into `carry` (mutated: its buffers are replaced by the
    new carry-forward remainder), return the entries due by `t_hi`, sorted
    by time (ties in no particular order).

    fresh_sorted=True promises every valid fresh entry is already in time
    order once the groups are concatenated -- true whenever all of them
    share one timestamp, which is what a population emitting only bare
    n2.spike() produces (see Population.fresh_sorted). It skips the sort;
    the default False is always correct."""
    device = fresh.time.device
    n_groups = fresh.count.shape[0]
    fresh_cap = n_groups * fresh.cap

    dense_time = torch.empty(max(_next_pow2(fresh_cap), 1), device=device, dtype=torch.float32)
    dense_node = torch.empty_like(dense_time, dtype=torch.int32)
    group_off = torch.empty(max(n_groups, 1), device=device, dtype=torch.int32)

    due_time = torch.zeros(carry.cap + fresh_cap, device=device, dtype=torch.float32)
    due_node = torch.zeros(carry.cap + fresh_cap, device=device, dtype=torch.int32)
    due_count = torch.zeros(1, device=device, dtype=torch.int32)
    # carry is read by binary search while the merge writes the new
    # remainder, so the remainder goes into fresh buffers that replace it
    new_carry_time = torch.full((carry.cap,), float("inf"), device=device, dtype=torch.float32)
    new_carry_node = torch.zeros(carry.cap, device=device, dtype=torch.int32)

    _compaction_kernel[(1,)](
        carry.time, carry.node, carry.count,
        fresh.time, fresh.node, fresh.count,
        dense_time, dense_node, group_off,
        due_time, due_node, due_count,
        new_carry_time, new_carry_node,
        t_hi, n_groups, carry.cap,
        GROUP_CAP=fresh.cap, FRESH_SORTED=fresh_sorted, BLOCK=_COMPACT_BLOCK,
    )
    carry.time = new_carry_time
    carry.node = new_carry_node
    return CompactedSpikes(due_time, due_node, due_count)


@triton.jit
def _merge_kernel(
    src_time_ptr, src_node_ptr, src_count_ptr, src_conn_ptr,
    out_time_ptr, out_node_ptr, out_conn_ptr, out_count_ptr,
    K: tl.constexpr, PER_SRC_CAP: tl.constexpr, TOTAL_CAP: tl.constexpr,
):
    idx = tl.arange(0, TOTAL_CAP)
    src_i = idx // PER_SRC_CAP
    local = idx % PER_SRC_CAP
    in_range = idx < K * PER_SRC_CAP

    count_i = tl.load(src_count_ptr + src_i, mask=in_range, other=0)
    valid = in_range & (local < count_i)
    time = tl.load(src_time_ptr + idx, mask=valid, other=0.0)
    node = tl.load(src_node_ptr + idx, mask=valid, other=0)
    conn = tl.load(src_conn_ptr + src_i, mask=in_range, other=0)
    time = tl.where(valid, time, _INF)

    low = (conn.to(tl.uint32) << 24) | (node.to(tl.uint32) & 0xFFFFFF)
    key = (_to_ordered_u32(time).to(tl.uint64) << 32) | low.to(tl.uint64)
    key = tl.sort(key)
    sorted_node = (key & 0xFFFFFF).to(tl.int32)
    sorted_conn = ((key >> 24) & 0xFF).to(tl.int32)
    sorted_time = _from_ordered_u32((key >> 32).to(tl.uint32))

    total = tl.sum(valid.to(tl.int32), axis=0)
    out_mask = idx < total
    tl.store(out_time_ptr + idx, sorted_time, mask=out_mask)
    tl.store(out_node_ptr + idx, sorted_node, mask=out_mask)
    tl.store(out_conn_ptr + idx, sorted_conn, mask=out_mask)
    tl.store(out_count_ptr, total)


def merge(
    sources: list[CompactedSpikes], conns: list[int], per_src_cap: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine several compact()ed sources (each tagged with its own
    connection id) into one sorted (time, src, conn) buffer, plus the valid
    count. `per_src_cap` must be >= every source's actual count -- this
    packs sources into a fixed [K, per_src_cap] layout rather than reading
    each source's own (possibly different) buffer length, see TODO.md for
    what a non-fixed-capacity version would need.

    node (0 <= node < 2**24) and conn (0 <= conn < 2**8) are packed into one
    key's low bits alongside the sort -- both comfortably within range for
    any realistic population size or per-population connection count.
    """
    assert len(sources) == len(conns)
    k = len(sources)
    device = sources[0].time.device if sources else torch.device("cpu")
    total_cap = _next_pow2(max(k, 1) * per_src_cap)

    src_time = torch.zeros(k * per_src_cap, device=device, dtype=torch.float32)
    src_node = torch.zeros(k * per_src_cap, device=device, dtype=torch.int32)
    src_count = torch.zeros(k, device=device, dtype=torch.int32)
    src_conn = torch.tensor(conns, device=device, dtype=torch.int32)
    for i, s in enumerate(sources):
        # .item() forces a host sync per source -- fine for this standalone
        # wrapper, but exactly the per-transfer latency floor TODO.md warns
        # about if done once per connection in the real integration. A real
        # version should pack sources into one [K, per_src_cap] buffer with
        # a small kernel/copy instead of this Python loop.
        n = int(s.count[0].item())
        src_time[i * per_src_cap : i * per_src_cap + n] = s.time[:n]
        src_node[i * per_src_cap : i * per_src_cap + n] = s.node[:n]
        src_count[i] = n

    out_time = torch.zeros(total_cap, device=device, dtype=torch.float32)
    out_node = torch.zeros(total_cap, device=device, dtype=torch.int32)
    out_conn = torch.zeros(total_cap, device=device, dtype=torch.int32)
    out_count = torch.zeros(1, device=device, dtype=torch.int32)

    _merge_kernel[(1,)](
        src_time, src_node, src_count, src_conn,
        out_time, out_node, out_conn, out_count,
        K=k, PER_SRC_CAP=per_src_cap, TOTAL_CAP=total_cap,
    )
    return out_time, out_node, out_conn, out_count


@triton.jit
def _combine_kernel(
    a_time_ptr, a_node_ptr, a_conn_ptr, a_count_ptr,
    b_time_ptr, b_node_ptr, b_conn_ptr, b_count_ptr,
    out_time_ptr, out_node_ptr, out_conn_ptr, out_count_ptr,
    CAP_A: tl.constexpr, CAP_B: tl.constexpr, TOTAL_CAP: tl.constexpr,
):
    idx = tl.arange(0, TOTAL_CAP)
    is_a = idx < CAP_A
    a_count = tl.load(a_count_ptr)
    b_count = tl.load(b_count_ptr)

    a_valid = is_a & (idx < a_count)
    time_a = tl.load(a_time_ptr + idx, mask=is_a, other=0.0)
    node_a = tl.load(a_node_ptr + idx, mask=is_a, other=0)
    conn_a = tl.load(a_conn_ptr + idx, mask=is_a, other=0)

    b_idx = idx - CAP_A
    is_b = (idx >= CAP_A) & (idx < CAP_A + CAP_B)
    b_valid = is_b & (b_idx < b_count)
    time_b = tl.load(b_time_ptr + b_idx, mask=is_b, other=0.0)
    node_b = tl.load(b_node_ptr + b_idx, mask=is_b, other=0)
    conn_b = tl.load(b_conn_ptr + b_idx, mask=is_b, other=0)

    valid = tl.where(is_a, a_valid, b_valid)
    time = tl.where(is_a, time_a, time_b)
    node = tl.where(is_a, node_a, node_b)
    conn = tl.where(is_a, conn_a, conn_b)
    time = tl.where(valid, time, _INF)

    low = (conn.to(tl.uint32) << 24) | (node.to(tl.uint32) & 0xFFFFFF)
    key = (_to_ordered_u32(time).to(tl.uint64) << 32) | low.to(tl.uint64)
    key = tl.sort(key)
    sorted_node = (key & 0xFFFFFF).to(tl.int32)
    sorted_conn = ((key >> 24) & 0xFF).to(tl.int32)
    sorted_time = _from_ordered_u32((key >> 32).to(tl.uint32))

    total = tl.sum(valid.to(tl.int32), axis=0)
    out_mask = idx < total
    tl.store(out_time_ptr + idx, sorted_time, mask=out_mask)
    tl.store(out_node_ptr + idx, sorted_node, mask=out_mask)
    tl.store(out_conn_ptr + idx, sorted_conn, mask=out_mask)
    tl.store(out_count_ptr, total)


def combine(
    a: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine two already (time, node, conn)-tagged, sorted streams (each a
    merge()-shaped (time, node, conn, count) tuple) into one sorted stream.

    Unlike merge(), neither input gets a connection id assigned here --
    both already carry their own per-entry conn, since both are themselves
    merge() (or combine()) outputs. This is what retention needs: a
    population not yet due for its own update must not lose a round's
    merge_incoming() result just because a later round's overwrites it --
    see Network.step() and TODO.md.

    CAP_A/CAP_B are each input's LIVE count, not its allocated size -- an
    earlier version used the allocated size (matching merge()'s own
    per_src_cap convention), which seemed like the more obvious choice
    but is wrong here specifically: merge()/combine() always allocate a
    full padded buffer regardless of how much is actually due, so sizing
    off that padding instead of the live count made capacity compound
    roughly geometrically across repeated retention rounds (combine(2048,
    2048) -> 4096 -> 8192 -> ... even when almost nothing was actually
    pending) -- found by it making compact()/merge()'s single-block
    tl.sort compile time blow up (minutes, not seconds) at a population
    size of just 1,000-2,000 once a population needed to retain across a
    couple of sub-steps. Reading the live count costs one extra .item()
    sync per side (same latency-floor caveat merge()'s own host wrapper
    already carries, see its docstring) but keeps the buffer honest.
    """
    a_time, a_node, a_conn, a_count = a
    b_time, b_node, b_conn, b_count = b
    cap_a = max(int(a_count[0].item()), 1)
    cap_b = max(int(b_count[0].item()), 1)
    device = a_time.device
    total_cap = _next_pow2(max(cap_a + cap_b, 1))

    out_time = torch.zeros(total_cap, device=device, dtype=torch.float32)
    out_node = torch.zeros(total_cap, device=device, dtype=torch.int32)
    out_conn = torch.zeros(total_cap, device=device, dtype=torch.int32)
    out_count = torch.zeros(1, device=device, dtype=torch.int32)

    _combine_kernel[(1,)](
        a_time, a_node, a_conn, a_count,
        b_time, b_node, b_conn, b_count,
        out_time, out_node, out_conn, out_count,
        CAP_A=cap_a, CAP_B=cap_b, TOTAL_CAP=total_cap,
    )
    return out_time, out_node, out_conn, out_count
