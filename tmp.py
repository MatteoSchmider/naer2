import triton
import triton.language as tl


@triton.jit
def update_kernel(
    potential_ptr,
    n_neurons,
    spike_time_ptr,
    spike_src_ptr,
    spike_conn_ptr,
    n_spikes,
    conn0_src_offsets_ptr,
    conn0_target_indices_ptr,
    conn0_weight_ptr,
    conn1_src_offsets_ptr,
    conn1_target_indices_ptr,
    conn1_weight_ptr,
    conn2_src_offsets_ptr,
    conn2_target_indices_ptr,
    conn2_weight_ptr,
    t,
    local_spike_node_ptr,
    local_spike_time_ptr,
    out_spike_node_ptr,
    out_spike_time_ptr,
    out_spike_count_ptr,
    GROUP_SIZE: tl.constexpr,
    SPIKE_BLOCK: tl.constexpr,
    LINK_BLOCK: tl.constexpr,
    LOCAL_SPIKE_CAP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = offs < n_neurons
    local_spike_count = tl.cast(0, tl.int32)
    potential = tl.load(potential_ptr + offs, mask=mask)
    for blk_start in range(0, n_spikes, SPIKE_BLOCK):
        for j in tl.static_range(SPIKE_BLOCK):
            _s = blk_start + j
            if _s < n_spikes:
                time = tl.load(spike_time_ptr + _s)
                spike_src_j = tl.load(spike_src_ptr + _s)
                spike_conn_j = tl.load(spike_conn_ptr + _s)
                if spike_conn_j == 0:
                    link_start = tl.load(conn0_src_offsets_ptr + spike_src_j)
                    link_end = tl.load(conn0_src_offsets_ptr + spike_src_j + 1)
                    for link_tile in range(link_start, link_end, LINK_BLOCK):
                        _lk = link_tile + tl.arange(0, LINK_BLOCK)
                        _lk_valid = _lk < link_end
                        _tgt = tl.load(
                            conn0_target_indices_ptr + _lk, mask=_lk_valid, other=-1
                        )
                        _own = (
                            _lk_valid
                            & (_tgt >= pid * GROUP_SIZE)
                            & (_tgt < (pid + 1) * GROUP_SIZE)
                        )
                        lk_weight = tl.load(
                            conn0_weight_ptr + _lk, mask=_lk_valid, other=0
                        )
                        _hit = (
                            (offs[:, None] == _tgt[None, :])
                            & _lk_valid[None, :]
                            & mask[:, None]
                        )
                        is_target = tl.sum(_hit.to(tl.int32), axis=1) > 0
                        edge_weight = tl.sum(
                            tl.where(_hit, lk_weight[None, :], 0), axis=1
                        )
                        potential = tl.where(
                            is_target, potential + edge_weight, potential
                        )
                        _m0 = is_target & (potential > 5)
                        _n_fire = _m0.to(tl.int32)
                        _local_pos = (
                            local_spike_count + tl.cumsum(_n_fire, axis=0) - _n_fire
                        )
                        local_spike_count = local_spike_count + tl.sum(_n_fire, axis=0)
                        tl.store(
                            local_spike_node_ptr + pid * LOCAL_SPIKE_CAP + _local_pos,
                            offs,
                            mask=_m0,
                        )
                        tl.store(
                            local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _local_pos,
                            t,
                            mask=_m0,
                        )
                        potential = tl.where(
                            is_target, tl.where(_m0, 0, potential), potential
                        )
                elif spike_conn_j == 1:
                    link_start = tl.load(conn1_src_offsets_ptr + spike_src_j)
                    link_end = tl.load(conn1_src_offsets_ptr + spike_src_j + 1)
                    for link_tile in range(link_start, link_end, LINK_BLOCK):
                        _lk = link_tile + tl.arange(0, LINK_BLOCK)
                        _lk_valid = _lk < link_end
                        _tgt = tl.load(
                            conn1_target_indices_ptr + _lk, mask=_lk_valid, other=-1
                        )
                        _own = (
                            _lk_valid
                            & (_tgt >= pid * GROUP_SIZE)
                            & (_tgt < (pid + 1) * GROUP_SIZE)
                        )
                        lk_weight = tl.load(
                            conn1_weight_ptr + _lk, mask=_lk_valid, other=0
                        )
                        _hit = (
                            (offs[:, None] == _tgt[None, :])
                            & _lk_valid[None, :]
                            & mask[:, None]
                        )
                        is_target = tl.sum(_hit.to(tl.int32), axis=1) > 0
                        edge_weight = tl.sum(
                            tl.where(_hit, lk_weight[None, :], 0), axis=1
                        )
                        potential = tl.where(
                            is_target, potential + edge_weight, potential
                        )
                        _m0 = is_target & (potential > 5)
                        _n_fire = _m0.to(tl.int32)
                        _local_pos = (
                            local_spike_count + tl.cumsum(_n_fire, axis=0) - _n_fire
                        )
                        local_spike_count = local_spike_count + tl.sum(_n_fire, axis=0)
                        tl.store(
                            local_spike_node_ptr + pid * LOCAL_SPIKE_CAP + _local_pos,
                            offs,
                            mask=_m0,
                        )
                        tl.store(
                            local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _local_pos,
                            t,
                            mask=_m0,
                        )
                        potential = tl.where(
                            is_target, tl.where(_m0, 0, potential), potential
                        )
                elif spike_conn_j == 2:
                    link_start = tl.load(conn2_src_offsets_ptr + spike_src_j)
                    link_end = tl.load(conn2_src_offsets_ptr + spike_src_j + 1)
                    for link_tile in range(link_start, link_end, LINK_BLOCK):
                        _lk = link_tile + tl.arange(0, LINK_BLOCK)
                        _lk_valid = _lk < link_end
                        _tgt = tl.load(
                            conn2_target_indices_ptr + _lk, mask=_lk_valid, other=-1
                        )
                        _own = (
                            _lk_valid
                            & (_tgt >= pid * GROUP_SIZE)
                            & (_tgt < (pid + 1) * GROUP_SIZE)
                        )
                        lk_weight = tl.load(
                            conn2_weight_ptr + _lk, mask=_lk_valid, other=0
                        )
                        lk_weight += 0.1
                        tl.store(conn2_weight_ptr + _lk, lk_weight, mask=_own)
                        _hit = (
                            (offs[:, None] == _tgt[None, :])
                            & _lk_valid[None, :]
                            & mask[:, None]
                        )
                        is_target = tl.sum(_hit.to(tl.int32), axis=1) > 0
                        edge_weight = tl.sum(
                            tl.where(_hit, lk_weight[None, :], 0), axis=1
                        )
                        potential = tl.where(
                            is_target, potential + edge_weight, potential
                        )
                        _m0 = is_target & (potential > 5)
                        _n_fire = _m0.to(tl.int32)
                        _local_pos = (
                            local_spike_count + tl.cumsum(_n_fire, axis=0) - _n_fire
                        )
                        local_spike_count = local_spike_count + tl.sum(_n_fire, axis=0)
                        tl.store(
                            local_spike_node_ptr + pid * LOCAL_SPIKE_CAP + _local_pos,
                            offs,
                            mask=_m0,
                        )
                        tl.store(
                            local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _local_pos,
                            t,
                            mask=_m0,
                        )
                        potential = tl.where(
                            is_target, tl.where(_m0, 0, potential), potential
                        )
    time = t
    potential *= 0.9
    _m0 = mask & (potential > 5)
    _n_fire = _m0.to(tl.int32)
    _local_pos = local_spike_count + tl.cumsum(_n_fire, axis=0) - _n_fire
    local_spike_count = local_spike_count + tl.sum(_n_fire, axis=0)
    tl.store(local_spike_node_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, offs, mask=_m0)
    tl.store(local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, t, mask=_m0)
    potential = tl.where(_m0, 0, potential)
    tl.store(potential_ptr + offs, potential, mask=mask)
    _global_base = tl.atomic_add(out_spike_count_ptr, local_spike_count)
    _flush_idx = tl.arange(0, LOCAL_SPIKE_CAP)
    _flush_mask = _flush_idx < local_spike_count
    _flush_node = tl.load(
        local_spike_node_ptr + pid * LOCAL_SPIKE_CAP + _flush_idx,
        mask=_flush_mask,
        other=0,
    )
    _flush_time = tl.load(
        local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _flush_idx,
        mask=_flush_mask,
        other=0.0,
    )
    tl.store(
        out_spike_node_ptr + _global_base + _flush_idx, _flush_node, mask=_flush_mask
    )
    tl.store(
        out_spike_time_ptr + _global_base + _flush_idx, _flush_time, mask=_flush_mask
    )
