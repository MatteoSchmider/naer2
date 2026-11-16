"""
Network.step() driving the fused kernels end to end (TODO.md 1e, runtime
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
    """TODO.md 1e (sizing): an event-driven consumer reads its source only
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
