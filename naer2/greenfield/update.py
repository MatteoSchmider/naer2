"""
The green-field update kernel (Gluon), hand-written outside population.py:
the PD14 neuron kernel stripped down to what the spike-management redesign
needs -- delivery, propagation, emission -- before it goes back into the
transpiler templates (and gets fused with intake).

  state:     only what the receive touches (PD14: current) is live during
             delivery; the rest is loaded after it -- registers are
             occupancy, occupancy is what hides the delivery's load chains
  delivery:  every program walks its own group's due slots k_lo..k_hi of the
             consumer's rings (naer2/intake.py) in slot order; each ring
             entry is one segment of a connection's TopologyNew, carrying
             its link range.
               main run only (always for uniform emission): ENTRY_BLOCK
                 entries at a time become one link stream in delivery order
                 (entry, then link), gathered one link per lane, staged in
                 LDS and walked U links per broadcast LDS read
               main and side runs (emission-delayed or late entries): merged
                 by emission time with two cursors, each entry's links read
                 straight from memory, U per round trip
             every link received in order by its target lane (the scalar
             receive: lanes compare against the link's target, no
             cross-lane traffic). Delivered cells are freed.
  emission:  naer2/emission.py's contract -- per group sorted by (e, node)
             (uniform emission: already sorted; otherwise every spike is
             written straight at its rank), the flat spike list placed by
             the summary-count atomic, the summary, the source state
  stand-in:  receive bookkeeping (summed weight, link count, optional log of
             every delivered link for the order checks; COUNT / LOG) and
             send (a neuron emits iff its given fire time lies in the run's
             window; e = t_start for uniform emission, else fire time +
             emission delay)

Layouts: LT = [GROUP_SIZE, U] puts the neurons over lanes and warps and the
U links of a walk group whole inside every thread, so a walk group is one
broadcast LDS read and picking link u out of it is a register select.

The Triton helpers (pointer layouts of the emission, ring and plan buffers)
are called as they are: Gluon takes triton.jit functions, their tensors pick
up the layouts of their inputs.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from naer2.emission import emission_flat_ptrs, emission_ptrs
from naer2.intake import _plan, _rec, _ring
from naer2.trace import mark, pin, pin_s, trace_begin, trace_end

UPDATE_TRACE_TAGS = ("start", "state", "slot", "headers", "streamed", "walked", "delivered", "sent", "end")


@gluon.jit
def _f32p(p):
    return p.to(gl.pointer_type(gl.float32))


# ------------------------------------------------- receive


@gluon.jit
def _receive(t, w, srcword, e, current, received, n_received, lane, mask, k, pid,
             log_ptr, log_e, log_cap, GROUP_SIZE: gl.constexpr, LOG: gl.constexpr, COUNT: gl.constexpr):
    """ONE link: t its target within the group (uniform over the program,
    -1 for none), w its weight. Every lane compares against t and the
    target lane runs the receive. A multapse's weights arrive one link at a
    time, in link order."""
    hit = (lane == t) & mask
    current = gl.where(hit, current + w, current)  # PD14 receive
    if COUNT:  # the stand-in's bookkeeping (tests), not PD14
        received = gl.where(hit, received + w, received)
        n_received = n_received + hit.to(gl.int32)
    if LOG:
        if t >= 0:
            at_log = gl.atomic_add(log_ptr, 1)
            if at_log < log_cap:
                rec = log_ptr + 1 + 4 * at_log
                gl.store(rec, k)
                gl.store(rec + 1, srcword >> 24)
                gl.store(rec + 2, srcword & 0xFFFFFF)
                gl.store(rec + 3, pid * GROUP_SIZE + t)
                gl.store(log_e + at_log, e)
    return current, received, n_received


# ------------------------------------------------- delivery: main runs (the link stream)


@gluon.jit
def _headers(main_lo, main_n, main_src, main_time, cell, b0, nm, n_cells, ei, LOG: gl.constexpr):
    """The ring records of one chunk: entries b0 .. b0 + ENTRY_BLOCK of the
    cell's main run."""
    em_ = b0 + ei < nm
    at = _rec(cell, b0 + ei, n_cells)
    lo = gl.load(main_lo + 4 * at, mask=em_, other=0)
    n = gl.load(main_n + 4 * at, mask=em_, other=0)
    srcw = gl.load(main_src + 4 * at, mask=em_, other=0)
    if LOG:
        ev = gl.load(main_time + 4 * at, mask=em_, other=0.0)
    else:
        ev = gl.zeros_like(lo).to(gl.float32)
    return lo, n, srcw, ev


@gluon.jit
def _links(lo, n, srcw, ev, s0, qi, ltp_tab, atp_tab, STREAM: gl.constexpr, LQ: gl.constexpr, LOG: gl.constexpr):
    """Stream block s0 of a chunk: per stream position its entry (histogram
    of the entries' starts + scan), then its link's target and weight.
    Each entry's two link addresses are formed once, with the stream offset
    folded in (base + lo - pos), so a stream position needs just two
    gathers by entry and its own position. Returns (target or -1, weight,
    source word, e, links in the block, links in the chunk)."""
    pos = gl.cumsum(n, axis=0) - n
    n_links = gl.sum(n, axis=0)
    live = n > 0
    c = srcw >> 24
    delta = (lo - pos).to(gl.int64)
    t_base = gl.gather(ltp_tab, c, 0) + 2 * delta   # byte addresses: int16 targets
    w_base = gl.gather(atp_tab, c, 0) + 4 * delta   # float32 weights
    q = s0 + qi
    qv = q < n_links
    rel = pos - s0
    before = gl.sum((live & (rel < 0)).to(gl.int32), axis=0)
    starts = gl.histogram(rel, STREAM, mask=live & (rel >= 0) & (rel < STREAM), layout=LQ)
    ent = gl.maximum(before + gl.cumsum(starts, axis=0) - 1, 0)
    q64 = q.to(gl.int64)
    ltp = (gl.gather(gl.convert_layout(t_base, LQ), ent, 0) + 2 * q64).to(gl.pointer_type(gl.int16))
    atp = (gl.gather(gl.convert_layout(w_base, LQ), ent, 0) + 4 * q64).to(gl.pointer_type(gl.float32))
    t_q = gl.where(qv, gl.load(ltp, mask=qv, other=0).to(gl.int32) & 0xFFFF, -1)
    w_q = gl.load(atp, mask=qv, other=0.0)
    if LOG:
        srcw_q = gl.gather(gl.convert_layout(srcw, LQ), ent, 0)
        e_q = gl.gather(gl.convert_layout(ev, LQ), ent, 0)
    else:
        srcw_q = gl.zeros_like(t_q)
        e_q = gl.zeros_like(w_q)
    return t_q, w_q, srcw_q, e_q, gl.minimum(n_links - s0, STREAM), n_links


@gluon.jit
def _stage(s_tgt, s_w, s_src, s_e, t_q, w_q, srcw_q, e_q, STREAM: gl.constexpr, U: gl.constexpr,
           LG: gl.constexpr, LOG: gl.constexpr):
    s_tgt.store(gl.convert_layout(gl.reshape(t_q, [STREAM // U, U]), LG))
    s_w.store(gl.convert_layout(gl.reshape(w_q, [STREAM // U, U]), LG))
    if LOG:
        s_src.store(gl.convert_layout(gl.reshape(srcw_q, [STREAM // U, U]), LG))
        s_e.store(gl.convert_layout(gl.reshape(e_q, [STREAM // U, U]), LG))
    gl.barrier()


@gluon.jit
def _walk(r_tgt, r_w, r_src, r_e, nq, current, received, n_received, lane, mask, ui, k, pid,
          log_ptr, log_e, log_cap, GROUP_SIZE: gl.constexpr, U: gl.constexpr, LU: gl.constexpr,
          LOG: gl.constexpr, COUNT: gl.constexpr):
    """The staged stream block's nq links, U per broadcast LDS read,
    received one by one by their target lanes."""
    for g in range(0, (nq + U - 1) // U):
        tg = r_tgt.index(g).load(LU)
        wg = r_w.index(g).load(LU)
        if LOG:
            sg = r_src.index(g).load(LU)
            eg = r_e.index(g).load(LU)
        for u in gl.static_range(U):
            t = gl.sum(gl.where(ui == u, tg, 0), axis=0)    # in-thread: a register select
            w = gl.sum(gl.where(ui == u, wg, 0.0), axis=0)
            if LOG:
                srcword = gl.sum(gl.where(ui == u, sg, 0), axis=0)
                e = gl.sum(gl.where(ui == u, eg, 0.0), axis=0)
            else:
                srcword = 0
                e = 0.0
            current, received, n_received = _receive(
                t, w, srcword, e, current, received, n_received, lane, mask, k, pid,
                log_ptr, log_e, log_cap, GROUP_SIZE, LOG, COUNT)
    return current, received, n_received


# ------------------------------------------------- delivery: main and side runs merged


@gluon.jit
def _merge(main_time, main_src, main_lo, main_n, side_time, side_src, side_lo, side_n, cell, nm, ns, n_cells,
           link_target_addr, attr_addr, current, received, n_received, lane, mask, k, pid,
           log_ptr, log_e, log_cap, GROUP_SIZE: gl.constexpr, U: gl.constexpr,
           LOG: gl.constexpr, COUNT: gl.constexpr):
    """A slot with side entries: its main and side runs merged by emission
    time with two cursors (main first on ties), each entry's links read
    straight from memory, U per round trip, received in order."""
    im = gl.to_tensor(0)
    js = gl.to_tensor(0)
    for _ in range(nm + ns):
        at_m = 4 * _rec(cell, im, n_cells)
        at_s = 4 * _rec(cell, js, n_cells)
        mv = im < nm
        sv = js < ns
        em = gl.load(main_time + at_m, mask=mv, other=float("inf"))
        es = gl.load(side_time + at_s, mask=sv, other=float("inf"))
        take = mv & (em <= es)
        e = gl.where(take, em, es)
        srcword = gl.where(take, gl.load(main_src + at_m, mask=mv, other=0), gl.load(side_src + at_s, mask=sv, other=0))
        lo = gl.where(take, gl.load(main_lo + at_m, mask=mv, other=0), gl.load(side_lo + at_s, mask=sv, other=0))
        n = gl.where(take, gl.load(main_n + at_m, mask=mv, other=0), gl.load(side_n + at_s, mask=sv, other=0))
        im = im + take.to(gl.int32)
        js = js + (1 - take.to(gl.int32))
        c = srcword >> 24
        ltp = gl.load(link_target_addr + c).to(gl.pointer_type(gl.int16)) + lo
        atp = gl.load(attr_addr + c).to(gl.pointer_type(gl.float32)) + lo
        for l0 in range(0, n, U):
            for u in gl.static_range(U):
                ok = l0 + u < n
                t = gl.where(ok, gl.load(ltp + l0 + u, mask=ok, other=0).to(gl.int32) & 0xFFFF, -1)
                w = gl.load(atp + l0 + u, mask=ok, other=0.0)
                current, received, n_received = _receive(
                    t, w, srcword, e, current, received, n_received, lane, mask, k, pid,
                    log_ptr, log_e, log_cap, GROUP_SIZE, LOG, COUNT)
    return current, received, n_received


# ------------------------------------------------- emission


@gluon.jit
def _rank(e, node, fits, s_e, s_n, r_e, r_n, ui, GROUP_SIZE: gl.constexpr, U: gl.constexpr,
          LU: gl.constexpr, LS: gl.constexpr, LG: gl.constexpr):
    """Per neuron its rank by (e, node) among the group's emitting neurons
    (fits): the group's keys go to LDS once, every neuron counts, U per
    broadcast read, the keys ordered before its own. Distinct nodes make
    the ranks distinct; non-emitting neurons count as (inf, node) and never
    order before an emitting one."""
    key_e = gl.where(fits, e, float("inf"))
    s_e.store(gl.convert_layout(gl.reshape(gl.convert_layout(key_e, LS), [GROUP_SIZE // U, U]), LG))
    s_n.store(gl.convert_layout(gl.reshape(gl.convert_layout(node, LS), [GROUP_SIZE // U, U]), LG))
    gl.barrier()
    rank = gl.zeros_like(node)
    for g in range(GROUP_SIZE // U):
        eg = r_e.index(g).load(LU)
        ng = r_n.index(g).load(LU)
        for u in gl.static_range(U):
            ej = gl.sum(gl.where(ui == u, eg, 0.0), axis=0)
            nj = gl.sum(gl.where(ui == u, ng, 0), axis=0)
            rank += ((ej < key_e) | ((ej == key_e) & (nj < node))).to(gl.int32)
    gl.barrier()
    return rank


# ------------------------------------------------- the kernel


@gluon.jit(do_not_specialize=["n_neurons", "n_conns", "k_lo", "k_hi", "run_id", "log_cap"])
def update_kernel(
    t_ref_end_ptr, potential_ptr, current_ptr, last_time_ptr, n_neurons,
    plan, n_conns, ring, k_lo, k_hi, slot_width,             # delivery input
    received_ptr, n_received_ptr, log_ptr, log_cap,          # stand-in receive, delivery log
    fire_time_ptr, emit_delay_ptr,                           # stand-in send
    emission, run_id, t_start, t_end, trace_ptr,
    GROUP_SIZE: gl.constexpr, LOCAL_SPIKE_CAP: gl.constexpr, UNIFORM_TIME: gl.constexpr,
    N_SLOTS: gl.constexpr, MAIN_CAP: gl.constexpr, SIDE_CAP: gl.constexpr,
    ENTRY_BLOCK: gl.constexpr, LOG: gl.constexpr,
    STREAM: gl.constexpr, U: gl.constexpr, SPT: gl.constexpr, NUM_WARPS: gl.constexpr, MAX_CONNS: gl.constexpr,
    SPQ: gl.constexpr, SPE: gl.constexpr, COUNT: gl.constexpr = True,
    TRACE: gl.constexpr = 0, TRACE_CAP: gl.constexpr = 1,
):
    gl.static_assert(STREAM % U == 0, "STREAM must be a multiple of U")
    gl.static_assert(GROUP_SIZE % U == 0, "GROUP_SIZE must be a multiple of U")
    # layouts (see the module docstring); SPT = neurons per thread
    LT: gl.constexpr = gl.BlockedLayout([SPT, U], [32, 1], [NUM_WARPS, 1], [1, 0])
    LN: gl.constexpr = gl.SliceLayout(1, LT)     # [GROUP_SIZE]: one neuron per lane
    LU: gl.constexpr = gl.SliceLayout(0, LT)     # [U]: a walk group, whole in every thread
    LQ: gl.constexpr = gl.BlockedLayout([SPQ], [32], [NUM_WARPS], [0])  # [STREAM]: stream positions over lanes
    LE: gl.constexpr = gl.BlockedLayout([SPE], [32], [NUM_WARPS], [0])  # [ENTRY_BLOCK]: entries over lanes
    LS: gl.constexpr = gl.BlockedLayout([SPT], [32], [NUM_WARPS], [0])  # LN as a plain layout: scans, reshapes
    SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    # LDS buffers as [rows, U] (one walk group each): stored whole through a
    # 2-D view, read a row at a time with a dynamic index (index() keeps the
    # layout's rank, so the buffer's layout is a row's)
    SM2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    LG: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [NUM_WARPS, 1], [1, 0])  # [rows, U] to store

    pid = gl.program_id(0)
    n_groups = gl.num_programs(0)
    n_ev = gl.to_tensor(0)
    if TRACE:  # see naer2/trace.py; tags UPDATE_TRACE_TAGS
        trace = trace_begin(trace_ptr, pid, TRACE_CAP)
        n_ev = mark(trace, n_ev, 0, 0, TRACE_CAP)
    lane = gl.arange(0, GROUP_SIZE, layout=LN)
    offs = pid * GROUP_SIZE + lane
    mask = offs < n_neurons
    ui = gl.arange(0, U, layout=LU)
    (spike_node, spike_time, spike_count, state_run, state_end,
     sum_count, sum_min, overflow) = emission_ptrs(emission, n_groups, LOCAL_SPIKE_CAP)

    current = gl.load(current_ptr + offs, mask=mask)
    if TRACE:
        current = pin(current)
        n_ev = mark(trace, n_ev, 1, 0, TRACE_CAP)

    # ---- delivery
    (_em_addr, _topo_addr, _seg_start_addr, link_target_addr, attr_addr, _eng, _ecap, _tns, _tnseg,
     _tnb, _tnl, _max_segs, _last_read) = _plan(plan, n_conns)
    (main_time, main_src, main_lo, main_n, main_count, side_time, side_src, side_lo, side_n, side_count,
     _watermark, _flags) = _ring(ring, n_groups, N_SLOTS, MAIN_CAP, SIDE_CAP)
    if COUNT:
        received = gl.load(received_ptr + offs, mask=mask, other=0.0)
        n_received = gl.load(n_received_ptr + offs, mask=mask, other=0)
    else:
        received = gl.zeros_like(current)
        n_received = gl.zeros_like(lane)
    log_e = (log_ptr + 4 * log_cap + 1).to(gl.pointer_type(gl.float32))
    ei = gl.arange(0, ENTRY_BLOCK, layout=LE)
    qi = gl.arange(0, STREAM, layout=LQ)
    ci = gl.arange(0, MAX_CONNS, layout=LQ)
    s_tgt = gl.allocate_shared_memory(gl.int32, [STREAM // U, U], SM2)
    s_w = gl.allocate_shared_memory(gl.float32, [STREAM // U, U], SM2)
    r_tgt = s_tgt.reinterpret(gl.int32, [STREAM // U, U], SM)
    r_w = s_w.reinterpret(gl.float32, [STREAM // U, U], SM)
    if LOG:
        s_src = gl.allocate_shared_memory(gl.int32, [STREAM // U, U], SM2)
        s_e = gl.allocate_shared_memory(gl.float32, [STREAM // U, U], SM2)
    else:  # placeholders, never touched
        s_src = s_tgt
        s_e = s_w
    r_src = s_src.reinterpret(gl.int32, [STREAM // U, U], SM)
    r_e = s_e.reinterpret(gl.float32, [STREAM // U, U], SM)
    n_cells = n_groups * N_SLOTS
    for k in range(k_lo, k_hi + 1):
        cell = pid * N_SLOTS + k % N_SLOTS
        nm = gl.minimum(gl.load(main_count + cell), MAIN_CAP)
        ns = gl.minimum(gl.load(side_count + cell), SIDE_CAP)
        if TRACE:
            nm = pin_s(nm)
            n_ev = mark(trace, n_ev, 2, nm, TRACE_CAP)
        if ns == 0:
            for b0 in range(0, nm, ENTRY_BLOCK):
                # a chunk of the slot's entries, each a link range of its connection
                n_links = gl.sum(gl.load(main_n + 4 * _rec(cell, b0 + ei, n_cells), mask=b0 + ei < nm, other=0),
                                 axis=0)
                if TRACE:
                    n_ev = mark(trace, n_ev, 3, n_links, TRACE_CAP)
                for s0 in range(0, n_links, STREAM):  # almost always one block
                    # the chunk's records (L1-hot) and the connections' base
                    # pointers are loaded where consumed
                    h_lo, h_n, h_srcw, h_ev = _headers(main_lo, main_n, main_src, main_time, cell, b0, nm,
                                                       n_cells, ei, LOG)
                    ltp_tab = gl.load(link_target_addr + ci, mask=ci < n_conns, other=0)
                    atp_tab = gl.load(attr_addr + ci, mask=ci < n_conns, other=0)
                    t_q, w_q, srcw_q, e_q, nq, _n = _links(h_lo, h_n, h_srcw, h_ev, s0, qi, ltp_tab, atp_tab,
                                                           STREAM, LQ, LOG)
                    _stage(s_tgt, s_w, s_src, s_e, t_q, w_q, srcw_q, e_q, STREAM, U, LG, LOG)
                    if TRACE:
                        n_ev = mark(trace, n_ev, 4, nq, TRACE_CAP)
                    current, received, n_received = _walk(
                        r_tgt, r_w, r_src, r_e, nq, current, received, n_received, lane, mask, ui, k, pid,
                        log_ptr, log_e, log_cap, GROUP_SIZE, U, LU, LOG, COUNT)
                    if TRACE:
                        current = pin(current)
                        n_ev = mark(trace, n_ev, 5, nq, TRACE_CAP)
                    gl.barrier()  # the stream is refilled next
        else:
            current, received, n_received = _merge(
                main_time, main_src, main_lo, main_n, side_time, side_src, side_lo, side_n, cell, nm, ns, n_cells,
                link_target_addr, attr_addr, current, received, n_received, lane, mask, k, pid,
                log_ptr, log_e, log_cap, GROUP_SIZE, U, LOG, COUNT)
        gl.store(main_count + cell, 0)
        gl.store(side_count + cell, 0)
    if COUNT:
        gl.store(received_ptr + offs, received, mask=mask)
        gl.store(n_received_ptr + offs, n_received, mask=mask)
    if TRACE:
        n_ev = mark(trace, n_ev, 6, 0, TRACE_CAP)

    # ---- final propagation to the window end (PD14)
    t_ref_end = gl.load(t_ref_end_ptr + offs, mask=mask)
    potential = gl.load(potential_ptr + offs, mask=mask)
    last_time = gl.load(last_time_ptr + offs, mask=mask)
    time = t_end
    dt = time - last_time
    last_time = gl.zeros_like(last_time) + time
    P_V = gl.exp(-dt / 10.0)
    P_I = gl.exp(-dt / 0.5)
    P_VI = 40.0 * 0.5 / (10.0 - 0.5) * (P_V - P_I)
    _m0 = mask & (time < t_ref_end)
    potential = gl.where(_m0, -65.0, potential)
    potential = gl.where(~_m0 & mask, -38.30576 + P_V * (potential - -38.30576) + P_VI * current, potential)
    current = P_I * current

    # ---- stand-in send: emit per the given fire times
    fire = gl.load(fire_time_ptr + offs, mask=mask, other=float("inf"))
    fires = mask & (fire >= t_start) & (fire < t_end)
    if UNIFORM_TIME:
        e = gl.zeros_like(fire) + t_start
    else:
        e = fire + gl.load(emit_delay_ptr + offs, mask=mask, other=0.0)
    _n_fire = fires.to(gl.int32)
    _local_pos = gl.convert_layout(gl.cumsum(gl.convert_layout(_n_fire, LS), axis=0), LN) - _n_fire
    local_spike_count = gl.sum(_n_fire, axis=0)
    _fits = fires & (_local_pos < LOCAL_SPIKE_CAP)
    # emission contract: the group's entries sorted by (e, node) -- uniform
    # emission is in node order already; otherwise each spike goes straight
    # to its rank (most groups emit 0 or 1 spikes per run)
    slot = _local_pos
    if not UNIFORM_TIME:
        s_ke = gl.allocate_shared_memory(gl.float32, [GROUP_SIZE // U, U], SM2)
        s_kn = gl.allocate_shared_memory(gl.int32, [GROUP_SIZE // U, U], SM2)
        if local_spike_count > 1:
            slot = _rank(e, offs, _fits, s_ke, s_kn, s_ke.reinterpret(gl.float32, [GROUP_SIZE // U, U], SM),
                         s_kn.reinterpret(gl.int32, [GROUP_SIZE // U, U], SM), ui, GROUP_SIZE, U, LU, LS, LG)
    gl.store(spike_node + pid * LOCAL_SPIKE_CAP + slot, offs, mask=_fits)
    gl.store(spike_time + pid * LOCAL_SPIKE_CAP + slot, e, mask=_fits)
    potential = gl.where(fires, -65.0, potential)
    t_ref_end = gl.where(fires, time + 2.0, t_ref_end)

    gl.store(t_ref_end_ptr + offs, t_ref_end, mask=mask)
    gl.store(potential_ptr + offs, potential, mask=mask)
    gl.store(current_ptr + offs, current, mask=mask)
    gl.store(last_time_ptr + offs, last_time, mask=mask)
    if TRACE:
        n_ev = mark(trace, n_ev, 7, 0, TRACE_CAP)

    # ---- summary, flat spike list, source state
    if local_spike_count > LOCAL_SPIKE_CAP:
        gl.atomic_max(overflow, 1)
    n_emit = gl.minimum(local_spike_count, LOCAL_SPIKE_CAP)
    gl.store(spike_count + pid, n_emit)
    p = run_id % 2
    base = gl.atomic_add(sum_count + p, n_emit)  # this group's place in the flat spike list
    flat_node, flat_time = emission_flat_ptrs(emission, n_groups, LOCAL_SPIKE_CAP)
    gl.store(flat_node + base + _local_pos, offs, mask=_fits)
    gl.store(flat_time + base + _local_pos, e, mask=_fits)
    gl.atomic_min(sum_min + p, gl.min(gl.where(_fits, e, float("inf")), axis=0))
    if pid == 0:
        gl.store(sum_count + (1 - p), 0)
        gl.store(sum_min + (1 - p), float("inf"))
        gl.store(state_run, run_id)
        gl.store(state_end, t_end)
    if TRACE:
        n_ev = mark(trace, n_ev, 8, 0, TRACE_CAP)
        trace_end(trace, n_ev)
