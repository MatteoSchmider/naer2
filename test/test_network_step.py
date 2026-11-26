"""
Network.step() driving the fused kernels end to end (TODO.md, runtime
integration step 4): the horizon scheduling deciding who runs when, every
run of a round launched with the round's id (a run reads its sources'
earlier rounds only) on its population's stream through one event hub, the
intake host-side only (an event-driven consumer scheduled from its sources'
emission summaries and its ring's pending arrivals), the history depth
derived from the timesteps.

A clocked commanded source (test/nodes/commanded.py) fires once, at t = 0
(bare n2.spike(): at its window's start), into an event-driven commanded
consumer (update() assigns nothing: it runs when an arrival is due); the
consumer's receive adds up its links' weights in `current`.
"""

import pytest
import torch

from test.networks.PD14.connection import PD14_connection
from test.nodes import commanded

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernels"
)


def _net(delay, n_src=40, n_dst=60, seed=0):
    torch.manual_seed(seed)
    A, B = commanded() * n_src, commanded() * n_dst
    A.timestep = 1.0
    B.count_received = True
    conn = A >> PD14_connection(0.3) >> B
    conn.weight = torch.rand(conn.shape)
    conn.delay = delay(conn) if callable(delay) else delay
    fires = torch.rand(n_src, generator=torch.Generator().manual_seed(seed + 1)) < 0.4
    A.fire_now = fires.float()  # consumed by A's first run, at t = 0
    return A.network, A, B, conn, fires


def _expected(conn, fires, n):
    offs, tgt = conn.fabric.topology.source_offsets, conn.fabric.topology.target_indices.long()
    weight = conn.fabric.weight.cpu().double()
    count, total = torch.zeros(n, dtype=torch.int64), torch.zeros(n, dtype=torch.float64)
    for s in fires.nonzero().flatten().tolist():
        count.index_add_(0, tgt[offs[s]:offs[s + 1]], torch.ones(int(offs[s + 1] - offs[s]), dtype=torch.int64))
        total.index_add_(0, tgt[offs[s]:offs[s + 1]], weight[offs[s]:offs[s + 1]])
    return count, total


def _delivered(B, conn, fires):
    count, total = _expected(conn, fires, 60)
    assert count.sum() > 0
    assert torch.equal(B.runtime.n_received.cpu().long(), count)
    assert torch.allclose(B.current.cpu().double(), total, atol=1e-4)


def test_an_event_driven_consumer_runs_when_its_input_arrives():
    network, A, B, conn, fires = _net(0.5)
    network.step(0.4)
    assert int(B.runtime.n_received.sum()) == 0  # the arrival (0.5) isn't due yet
    network.step(1.6)
    _delivered(B, conn, fires)
    A.check()
    B.check()


def test_later_per_link_arrivals_are_scheduled_from_the_ring():
    """Delays 0.5 and 2.5: the summary schedules the consumer for the
    earliest arrival (0.5) -- a run whose window ends at its input bound,
    1.5 -- and only its ring knows the 2.5 arrivals are still pending (the
    source's later runs emit nothing)."""
    network, A, B, conn, fires = _net(lambda c: torch.where(torch.rand(c.shape) < 0.5, 0.5, 2.5))
    network.step(2.0)
    count, _ = _expected(conn, fires, 60)
    assert 0 < int(B.runtime.n_received.sum()) < int(count.sum())  # the 0.5 links only, so far
    network.step(2.0)
    _delivered(B, conn, fires)


@pytest.mark.parametrize("mode", ["planned", "graph"])
@pytest.mark.parametrize("delay", ["0.5", "0.5 and 2.5"])
def test_planned_windows_gate_an_event_driven_consumer(delay, mode):
    """The same consumer through Network.run_planned / run_graph (a captured window replayed; after one
    step, every runtime built): its launches
    are gated -- one at every time an arrival could be due -- and its kernel runs only those with something
    due (fused.py: _gate_skip); every link arrives, exactly, among many gated launches that skipped."""
    d = 0.5 if delay == "0.5" else (lambda c: torch.where(torch.rand(c.shape) < 0.5, 0.5, 2.5))
    stepped, _, B_step, _, _ = _net(d)  # the same network stepped: how many runs B makes there
    stepped.step(4.0)
    network, A, B, conn, fires = _net(d)
    network.step(0.05)  # the first round (t = 0): every runtime built
    gated = sum(1 for launch in network.plan(until=4.0) for pop, *_, g in launch if pop is B and g)
    (network.run_planned if mode == "planned" else network.run_graph)(until=4.0)
    A.check()
    B.check()
    _delivered(B, conn, fires)
    made = B.runtime.runs_made()
    assert made == B_step.runtime.runs_made() and made < 1 + gated  # (the first round's run + those due)


def _two_delay_net(seed=0):
    """Two sources into one consumer, each connection one homogeneous delay (0.5, 2.5): two delay classes."""
    torch.manual_seed(seed)
    A1, A2, B = commanded() * 40, commanded() * 30, commanded() * 60
    A1.timestep = A2.timestep = 1.0
    B.count_received = True
    conns, fires = [], []
    for i, (A, d) in enumerate(((A1, 0.5), (A2, 2.5))):
        conn = A >> PD14_connection(0.3) >> B
        conn.weight = torch.rand(conn.shape)
        conn.delay = d
        f = torch.rand(len(A.shape), generator=torch.Generator().manual_seed(seed + 1 + i)) < 0.4
        A.fire_now = f.float()
        conns.append(conn)
        fires.append(f)
    return A1.network, B, conns, fires


@pytest.mark.parametrize("mode", ["step", "planned", "graph"])
def test_several_homogeneous_delays_through_the_network(mode):
    """Several homogeneous delays end to end: the consumer's connections a queue each (SPIKES.md), scheduled from
    its queues' next arrival (the 2.5 arrivals: only its queues know them), every link delivered -- stepped,
    and through planned and captured windows."""
    network, B, conns, fires = _two_delay_net()
    if mode == "step":
        network.step(2.0)
        assert B.runtime.queues and B.runtime.ring.n_slots == 2  # (a queue per connection: SPIKES.md)
        first, _ = _expected(conns[0], fires[0], 60)
        assert int(B.runtime.n_received.sum()) == int(first.sum())  # the 0.5 links only, so far
        network.step(2.0)
    else:
        network.step(0.05)
        (network.run_planned if mode == "planned" else network.run_graph)(until=4.0)
    B.check()
    count = sum(_expected(c, f, 60)[0] for c, f in zip(conns, fires))
    total = sum(_expected(c, f, 60)[1] for c, f in zip(conns, fires))
    assert torch.equal(B.runtime.n_received.cpu().long(), count)
    assert torch.allclose(B.current.cpu().double(), total, atol=1e-4)


def _periodic_net(delays=(0.5,), period=0.7, seed=0):
    """An update()-only source with precise spike times and no timestep (test/nodes/periodic_source.py: each
    neuron at its phase, then every period) into event-driven consumers, one per delay."""
    from test.nodes import periodic_source

    torch.manual_seed(seed)
    A = periodic_source(period=period) * 50
    phases = 0.05 + 0.6 * torch.rand(50)
    A.next_spike = phases.clone()
    conns, Bs = [], []
    for d in delays:
        B = commanded() * 60
        B.count_received = True
        conn = A >> PD14_connection(0.3) >> B
        conn.weight = torch.rand(conn.shape)
        conn.delay = d
        conns.append(conn)
        Bs.append(B)
    return A.network, A, Bs, conns, phases


def _periodic_expected(conn, phases, period, delay, until):
    """Per target, the links of every spike (phase + k period) arriving before `until`, and their weights."""
    offs, tgt = conn.fabric.topology.source_offsets, conn.fabric.topology.target_indices.long()
    weight = conn.fabric.weight.cpu().double()
    count, total = torch.zeros(60, dtype=torch.int64), torch.zeros(60, dtype=torch.float64)
    for s in range(len(phases)):
        n = 0
        while float(phases[s]) + period * n + delay < until - 1e-6:
            count.index_add_(0, tgt[offs[s]:offs[s + 1]], torch.ones(int(offs[s + 1] - offs[s]), dtype=torch.int64))
            total.index_add_(0, tgt[offs[s]:offs[s + 1]], weight[offs[s]:offs[s + 1]])
            n += 1
    return count, total


@pytest.mark.parametrize("mode", ["step", "planned", "graph"])
def test_a_source_runs_every_smallest_outgoing_delay(mode):
    """BRUNEL.md, step 4: an update() population without inputs or a timestep (a source) runs every smallest
    delay of its outgoing connections (0.3 here, of 0.3 and 0.5): its runs [t, t + 0.3), every spike at its
    exact time -- each consumer receives exactly the spikes arriving before its last window's end, stepped,
    planned and captured."""
    network, A, Bs, conns, phases = _periodic_net(delays=(0.5, 0.3))
    if mode == "step":
        network.step(5.0)
    else:
        network.step(0.05)
        (network.run_planned if mode == "planned" else network.run_graph)(until=5.0)
    assert A.timestep is None and A._own_timestep() == 0.3
    assert A.runtime.runs_made() == round(A._horizon / 0.3)
    for B, conn, d in zip(Bs, conns, (0.5, 0.3)):
        B.check()
        count, total = _periodic_expected(conn, phases, 0.7, d, B._window[1])
        assert count.sum() > 1000
        assert torch.equal(B.runtime.n_received.cpu().long(), count), (d, B._window)
        assert torch.allclose(B.current.cpu().double(), total, atol=1e-3)


def test_a_source_without_outgoing_delay_needs_a_timestep():
    from test.nodes import periodic_source

    A = periodic_source() * 10
    with pytest.raises(ValueError, match="set population.timestep"):
        A.network.step(1.0)


@pytest.mark.parametrize("mode", ["step", "planned", "graph"])
def test_an_event_driven_consumer_idle_longer_than_its_ring(mode):
    """An event-driven consumer idle past its ring's length (its inputs arrive around 0.6, then 3.6; the ring:
    13 slots of 0.1): its next run starts from its own start, not its last run's end -- the slots between are
    empty, the later arrivals aren't far future (stepped: Ring.first_pending; gated: fused.py _run_from).
    (Idle much longer, its sources' runs outgrow its history: the slow-consumer gap, test_gap_history_grows.)"""
    network, A, (B,), (conn,), phases = _periodic_net(delays=(0.5,), period=3.0)
    if mode == "step":
        network.step(5.0)
    else:
        network.step(0.05)
        (network.run_planned if mode == "planned" else network.run_graph)(until=5.0)
    B.check()
    assert B.runtime.ring.ring_slots * B.runtime.ring.slot_width < 2.0
    count, total = _periodic_expected(conn, phases, 3.0, 0.5, B._window[1])
    assert int(count.sum()) == 2 * len(conn.fabric.topology)  # (both spikes of every source)
    assert torch.equal(B.runtime.n_received.cpu().long(), count)
    assert torch.allclose(B.current.cpu().double(), total, atol=1e-3)


def test_tensors_move_to_the_network_device_lazily():
    network, A, B, _, _ = _net(0.5)
    assert A.fire_now.device.type == "cpu"
    network.step(1.0)
    assert A.fire_now.device.type == network.device.type
    assert B.current.device.type == network.device.type


def test_the_history_depth_follows_the_timesteps():
    """A source running every 0.1 read by a consumer every 0.5: up to five of
    its runs pile up between two reads -- depth 2 + 5 for every population."""
    torch.manual_seed(0)
    A, B = commanded() * 20, commanded() * 20
    A.timestep, B.timestep = 0.1, 0.5
    conn = A >> PD14_connection(0.3) >> B
    conn.weight = torch.rand(conn.shape)
    conn.delay = 0.5
    A.network.step(1.0)
    assert A.runtime.history.K == B.runtime.history.K == 7
    A.check()
    B.check()


def test_every_run_of_a_round_carries_its_id():
    network, A, B, _, _ = _net(0.5)
    rounds = []
    for pop in (A, B):
        run = pop.run

        def spy(*args, run=run, pop=pop, **kw):
            rounds.append((pop is A, kw["round_id"], network.round_id))
            return run(*args, **kw)

        pop.run = spy
    network.step(2.0)
    ids = [r for _, r, _ in rounds]
    assert ids and all(r == current for _, r, current in rounds)  # the round's id, at launch
    assert ids[0] == 1 and ids == sorted(ids) and len(set(ids)) > 1


# ------------------------------------------------------------ gaps (fail until closed)


def test_gap_history_grows_with_a_slow_consumer():
    """TODO.md (sizing): an event-driven consumer reads its source only
    when an arrival is due -- here every 3 ms, while the source runs (and
    spikes) every 0.1 ms, ~30 runs between reads. A history that grows when
    a consumer falls behind keeps them all; today the derived depth (8 with
    an event-driven consumer) overruns and check() raises."""
    from test.nodes import clock_source

    torch.manual_seed(0)
    A, B = clock_source() * 20, commanded() * 30
    A.timestep = 0.1
    B.count_received = True
    conn = A >> PD14_connection(0.3) >> B
    conn.weight = torch.rand(conn.shape)
    conn.delay = 3.0
    A.network.step(6.0)
    A.check()
    B.check()  # raises today: "fell behind a source's emission history"
    # every run of A spiked on every neuron (e = its window's start, 0.1 k); B has delivered every slot below its
    # ring position: all links of the runs arriving there
    ring = B.runtime.ring
    arrivals = (torch.arange(A.runtime.run_id, dtype=torch.float32) * 0.1 + 3.0) / ring.slot_width
    runs_arrived = int((arrivals.floor() < ring.ring_pos).sum())
    assert runs_arrived > 10
    assert int(B.runtime.n_received.sum()) == runs_arrived * len(conn.fabric.topology)


def test_a_receive_reading_no_edge_attr_loads_none():
    """A consumer whose receive reads no edge attr (ed_relay: counts its links) -- the delivery stages no
    per-link attr (fused.py _stage_records, STAGE_W off: no load, no table) and every link still arrives once."""
    from test.nodes.ed_relay import ed_relay
    torch.manual_seed(0)
    A, B = commanded() * 40, ed_relay() * 60
    A.timestep = 1.0
    conn = A >> PD14_connection(0.3) >> B
    conn.delay = 0.5
    fires = torch.rand(40, generator=torch.Generator().manual_seed(1)) < 0.4
    A.fire_now = fires.float()
    assert "ARENA_CHECK, False, T_A" in B.fused_kernel().source  # (the delivery: no per-link attr staged)
    A.network.step(2.0)
    offs, tgt = conn.fabric.topology.source_offsets, conn.fabric.topology.target_indices.long()
    count = torch.zeros(60)
    for s in fires.nonzero().flatten().tolist():
        count.index_add_(0, tgt[offs[s]:offs[s + 1]], torch.ones(int(offs[s + 1] - offs[s])))
    assert count.sum() > 0
    assert torch.equal(B.count.cpu(), count)
