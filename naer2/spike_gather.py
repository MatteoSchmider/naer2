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

Both kernels run as a single Triton program (grid=(1,)): realistic spike
counts here are small -- tens to a few hundred per population per timestep,
see TODO.md's scale reasoning -- so one block sorting everything at once is
simpler than tiling, and was fast enough (tens to low hundreds of us,
measured) that tiling isn't worth it unless a future scale actually needs
it. Both use the same trick to turn a sort into a single `tl.sort` call
(which only orders one tensor of keys, no attached payload): pack every
field being carried into one sortable integer key, sort that, unpack after.
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
def _compaction_kernel(
    carry_time_ptr, carry_node_ptr, carry_count_ptr,
    fresh_time_ptr, fresh_node_ptr, fresh_count_ptr,
    due_time_ptr, due_node_ptr, due_count_ptr,
    t_hi,
    N_GROUPS: tl.constexpr, GROUP_CAP: tl.constexpr, CARRY_CAP: tl.constexpr, TOTAL_CAP: tl.constexpr,
):
    idx = tl.arange(0, TOTAL_CAP)
    carry_count = tl.load(carry_count_ptr)

    is_carry = idx < CARRY_CAP
    is_fresh = (idx >= CARRY_CAP) & (idx < CARRY_CAP + N_GROUPS * GROUP_CAP)

    time_c = tl.load(carry_time_ptr + idx, mask=is_carry, other=0.0)
    node_c = tl.load(carry_node_ptr + idx, mask=is_carry, other=0)
    carry_valid = is_carry & (idx < carry_count)

    fresh_idx = idx - CARRY_CAP
    g = fresh_idx // GROUP_CAP
    local = fresh_idx % GROUP_CAP
    fresh_count_g = tl.load(fresh_count_ptr + g, mask=is_fresh, other=0)
    fresh_valid = is_fresh & (local < fresh_count_g)
    time_f = tl.load(fresh_time_ptr + fresh_idx, mask=fresh_valid, other=0.0)
    node_f = tl.load(fresh_node_ptr + fresh_idx, mask=fresh_valid, other=0)

    valid = tl.where(is_carry, carry_valid, fresh_valid)
    time = tl.where(is_carry, time_c, time_f)
    node = tl.where(is_carry, node_c, node_f)
    time = tl.where(valid, time, _INF)

    key = (_to_ordered_u32(time).to(tl.uint64) << 32) | node.to(tl.uint64)
    key = tl.sort(key)
    sorted_node = (key & 0xFFFFFFFF).to(tl.int32)
    sorted_time = _from_ordered_u32((key >> 32).to(tl.uint32))

    fresh_total = tl.sum(fresh_valid.to(tl.int32), axis=0)
    total_real = carry_count + fresh_total
    due_count = tl.sum((sorted_time <= t_hi).to(tl.int32), axis=0)
    new_carry_count = total_real - due_count

    due_mask = idx < due_count
    tl.store(due_time_ptr + idx, sorted_time, mask=due_mask)
    tl.store(due_node_ptr + idx, sorted_node, mask=due_mask)

    carry_store_idx = idx - due_count
    carry_mask = (idx >= due_count) & (idx < due_count + new_carry_count)
    tl.store(carry_time_ptr + carry_store_idx, sorted_time, mask=carry_mask)
    tl.store(carry_node_ptr + carry_store_idx, sorted_node, mask=carry_mask)

    tl.store(due_count_ptr, due_count)
    tl.store(carry_count_ptr, new_carry_count)


def compact(fresh: BlockedSpikes, carry: CarryBuffer, t_hi: float) -> CompactedSpikes:
    """Fold `fresh` into `carry` (mutated in place: becomes the new
    carry-forward remainder), return the entries due by `t_hi`."""
    n_groups = fresh.count.shape[0]
    total_cap = _next_pow2(carry.cap + n_groups * fresh.cap)

    due_time = torch.zeros(total_cap, device=fresh.time.device, dtype=torch.float32)
    due_node = torch.zeros(total_cap, device=fresh.time.device, dtype=torch.int32)
    due_count = torch.zeros(1, device=fresh.time.device, dtype=torch.int32)

    _compaction_kernel[(1,)](
        carry.time, carry.node, carry.count,
        fresh.time, fresh.node, fresh.count,
        due_time, due_node, due_count,
        t_hi,
        N_GROUPS=n_groups, GROUP_CAP=fresh.cap, CARRY_CAP=carry.cap, TOTAL_CAP=total_cap,
    )
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
