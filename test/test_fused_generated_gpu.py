"""
Generated fused kernels (naer2/transpiler/templates/fused_template.py)
against the hand-written reference (test/fused_reference.py), both run by
naer2/runtime.py:
PD14_neuron and PD14_poisson built through the DSL, their kernels generated
from their hooks, run side by side with the reference from the same state
on the same inputs, run by run: every state tensor and every spike
bit-identical where nothing is delivered; with deliveries, the same spikes
and states equal to the reference's own run-to-run spread -- the pull
appends a slot's records in the order of its LDS atomic claims, which isn't
deterministic across warps, so even two runs of the reference sum a
neuron's inputs in different orders (last-bit differences in current). And
the gaps the generator still has, as tests that fail until they're closed.
"""

import math

import pytest
import torch

import naer2 as n2
from naer2.ring import POOL_RESERVE
from naer2.runtime import PopulationRuntime
from naer2.transpiler.templates.fused_template import FusedKernelTemplate
from test.edges import plastic_synapse, static_synapse
from test.fused_reference import pd14_neuron_reference, pd14_poisson_reference
from test.networks.PD14.connection import PD14_connection
from test.networks.PD14.neuron import PD14_neuron
from test.networks.PD14.poisson import PD14_poisson
from test.runtimes import command, commanded_runtime

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernel"
)

DEVICE = "cuda:0"
G = 256


def _defaults(population):
    return {a: v for a, v in population.instance.__dict__.items() if a in population.dynamic_attrs}


def _runtime(kernel, population, **state):
    """A runtime of `population`'s size running `kernel`, its state starting at the instance's values (or
    `state`)."""
    rt = PopulationRuntime(kernel, len(population.shape), group_size=G, device=DEVICE,
                           state={**_defaults(population), **state})
    rt.count_received = True
    return rt


def _connect(gen, dst_pop, conns, srcs, topos):
    """gen connected as its Population would be: a queue per connection where every connection has one delay
    value (SPIKES.md: any sources), else the slotted ring (8 slots of 0.1 ms, per-cell runs: uniform sources)."""
    if dst_pop._queues():
        gen.connect(srcs, topos, len(conns), pool=(256, len(conns) * (POOL_RESERVE + 1) + 9),
                    min_delays=[dst_pop._min_delay(c) for c in conns], queues=True)
    else:
        gen.connect(srcs, topos, n_slots=8, main_cap=8192)


def _same_spikes(a, b):
    assert sorted(a.emitted()) == sorted(b.emitted())


def test_generated_pd14_neuron_matches_the_reference():
    torch.manual_seed(0)
    src_pop = PD14_neuron(K_cortical=2000) * 300
    dst_pop = PD14_neuron(K_cortical=2900) * 700
    conn = src_pop >> PD14_connection(0.1) >> dst_pop
    conn.weight = torch.rand(conn.shape) * 2.0
    gk = FusedKernelTemplate(dst_pop).build()
    assert gk.state_attrs == ["current"] and gk.edge_attr == "weight" and gk.uniform_emission

    conn.delay = 0.2
    topo = conn.fabric.simulation(700, G, DEVICE, attrs=["weight"])
    src = commanded_runtime(300, group_size=G, device=DEVICE)  # firing on command
    init = dict(potential=torch.normal(-55.0, 6.0, (700,)), current=torch.rand(700) * 0.5,
                t_ref_end=1.0 + torch.rand(700) * 0.4, last_time=torch.full((700,), 0.9))
    ref = _runtime(pd14_neuron_reference(dst_pop.instance), dst_pop, **init)
    gen = _runtime(gk, dst_pop, **init)
    for p in (ref, gen):  # ~2.7k links per (group, slot) here: above the default capacity
        p.connect([src], [topo], n_slots=8, main_cap=8192)
    spiked = 0
    for r in range(10):
        t = round(1.0 + 0.1 * r, 6)
        ref.run(t, t + 0.1, round_id=r)
        gen.run(t, t + 0.1, round_id=r)
        ref.check()
        gen.check()
        delivered = int(gen.n_received.sum()) > 0
        for name in init:
            if delivered:  # summation order of the inputs: the reference's own run-to-run spread
                assert torch.allclose(ref.state[name], gen.state[name], rtol=1e-6, atol=1e-5), (r, name)
            else:
                assert torch.equal(ref.state[name], gen.state[name]), (r, name)
        assert torch.equal(ref.n_received, gen.n_received), r
        _same_spikes(ref, gen)
        spiked += len(gen.emitted())
        command(src, t, t + 0.1, torch.where(torch.rand(300) < 0.3, t, math.inf), round_id=r)
    assert spiked > 0 and int(gen.n_received.sum()) > 0


def test_generated_pd14_poisson_matches_the_reference():
    T = PD14_poisson(rate=0.12, t_start=700.0, t_stop=701.0) * 902
    gk = FusedKernelTemplate(T).build()
    assert gk.state_attrs == [] and gk.uniform_emission
    ref = _runtime(pd14_poisson_reference(T.instance), T)
    gen = _runtime(gk, T)
    assert torch.equal(ref.state["next_spike"], gen.state["next_spike"])
    spiked = 0
    for r in range(20):
        t = round(699.5 + 0.1 * r, 6)
        ref.run(t, t + 0.1, round_id=r)
        gen.run(t, t + 0.1, round_id=r)
        assert torch.equal(ref.state["next_spike"], gen.state["next_spike"]), r
        _same_spikes(ref, gen)
        spiked += len(gen.emitted())
    assert spiked > 902


# ------------------------------------------------------------ gaps (fail until closed)


@n2.node
class _spiking_receiver:
    """Integrates its inputs and fires from inside receive, at the input's time (D-3)."""

    def __init__(self):
        self.potential: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.potential += edge.weight
        if self.potential > 1.0:
            n2.spike(time)
            self.potential = 0.0


@n2.node
class _bare_spiking_receiver:
    """The same with a bare n2.spike(): at exactly the delivery time (D-3)."""

    def __init__(self):
        self.potential: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.potential += edge.weight
        if self.potential > 1.0:
            n2.spike()
            self.potential = 0.0


def _spiking_run(node, weight, delay=0.15, seed=41, uniform=False, history_cap=1200):
    """Sources fire once, at random times in [1.0, 1.1) (uniform: at 1.0, every arrival tied); the consumer
    (weight on every link, one homogeneous delay: its queue, exact arrivals) runs rounds 0..3, its history taking
    history_cap spikes a run. Returns (consumer, fire times, connection, every spike it emitted as (e, node))."""
    src, dst = PD14_neuron(K_cortical=2000) * 200, node() * 300
    conn = src >> PD14_connection(0.1) >> dst
    conn.weight = torch.full(conn.shape, weight)
    conn.delay = delay
    gk = FusedKernelTemplate(dst).build()
    assert not gk.uniform_emission  # spikes at the delivery's time
    gen = PopulationRuntime(gk, 300, group_size=G, device=DEVICE, state=_defaults(dst), history_cap=history_cap)
    srcs = commanded_runtime(200, uniform=uniform, group_size=G, device=DEVICE)
    _connect(gen, dst, [conn], [srcs], [conn.fabric.simulation(300, G, DEVICE, attrs=["weight"])])
    g = torch.Generator().manual_seed(seed)
    at = torch.full((200,), 1.0) if uniform else 1.0 + 0.1 * torch.rand(200, generator=g)
    fire = torch.where(torch.rand(200, generator=g) < 0.3, at, math.inf).float()
    spikes = []
    for r in range(4):
        t = round(1.0 + 0.1 * r, 6)
        gen.run(t, t + 0.1, round_id=r)
        spikes += gen.emitted()
        command(srcs, t, t + 0.1, fire if r == 0 else None, round_id=r)
    return gen, fire, conn, spikes


@pytest.mark.parametrize("uniform", [False, True], ids=["timed", "uniform"])
@pytest.mark.parametrize("node", [_spiking_receiver, _bare_spiking_receiver], ids=["spike(time)", "bare"])
def test_spikes_inside_delivery(node, uniform):
    """D-3: n2.spike() in receive fires at the delivery -- explicit (its time) or bare (the delivery time
    itself): every link 0.4, so a target fires at every third input, at that input's arrival (fp32 e + D,
    in arrival order), any number of times a run -- against a host reference. The queue's batch walk, timed
    sources and uniform ones (all arrivals at one time: a beat's receives one by one)."""
    gen, fire, conn, spikes = _spiking_run(node, 0.4, uniform=uniform)
    gen.check()
    offs, tgt = conn.fabric.topology.source_offsets, conn.fabric.topology.target_indices.long()
    arrivals = [[] for _ in range(300)]
    arr = fire + torch.tensor(0.15, dtype=torch.float32)
    for s in torch.isfinite(fire).nonzero().flatten().tolist():
        for j in tgt[offs[s]:offs[s + 1]].tolist():
            arrivals[j].append(arr[s].item())
    want = sorted((a, j) for j in range(300) for a in sorted(arrivals[j])[2::3])
    assert len(want) > 100 and any(len(a) >= 6 for a in arrivals)  # (some fire twice)
    assert sorted(spikes) == want


def test_spikes_inside_delivery_past_the_history_cap_raise():
    """D-3: neurons firing on every input (~6 a run here) emit more than the history's default cap, one spike
    per neuron and run -- the rest are lost, and check() raises."""
    gen, _, _, _ = _spiking_run(_spiking_receiver, 1.5, history_cap=None)
    with pytest.raises(RuntimeError, match="more than its history"):
        gen.check()


# The remaining gaps run end to end once closed: every source fires once (round 0), its spikes reach the
# consumer two rounds later (0.2 ms delay), and the consumer's state is checked against what its receive
# must add up -- computed from the canonical topology. Until then each fails at FusedKernelTemplate.build()
# (NotImplementedError naming its decision) or on the values (D-1).


@n2.node(receive={"exc": "recv_exc", "inh": "recv_inh"})
class _two_ports:
    """Two receive ports with different code (D-6)."""

    def __init__(self):
        self.acc: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def recv_exc(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.acc += edge.weight

    def recv_inh(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.acc -= edge.weight


@n2.node
class _two_attrs:
    """Reads two per-link attrs (D-7)."""

    def __init__(self):
        self.acc: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.acc += edge.weight * edge.gain


@n2.node
class _accumulator:
    """Adds up its links' weights (the consumer of the edge-forward gaps)."""

    def __init__(self, acc: float = 0.0):
        self.acc: float = acc

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.acc += edge.weight


@n2.node
class _source_reader:
    """Reads its source's state (D-10)."""

    def __init__(self):
        self.acc: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.acc += source.potential


@n2.node
class _arrival_recorder:
    """Records the time its last input arrived (D-1)."""

    def __init__(self):
        self.last_input: float = -1.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.last_input = time


@n2.edge
class _destination_synapse:
    """An edge forward reading its target's state (D-9)."""

    def __init__(self, weight: float):
        self.weight = weight

    def forward(self, dst: n2.receives.destination):
        self.weight += 0.1 * dst.acc

    def backward(self):
        pass


def _fires(n, seed):
    return torch.rand(n, generator=torch.Generator().manual_seed(seed)) < 0.3


def _deliver_once(dst_pop, sources, attrs=None, delay=0.2):
    """sources: [(connection, fire mask or fire times)]: each source population (a stand-in
    runtime, test/runtimes.py) fires in round 0 -- a mask at the round's start (1.0), or its
    times (inf: silent); the generated consumer runs rounds 0..2 (the spikes arrive in round 2
    at the default delay); attrs: the per-link attrs the topologies carry (default: the one the
    kernel stages); delay: every connection's (a number, or per link). Returns (consumer,
    topologies)."""
    for conn, _ in sources:  # (the delays first: the kernel's delivery time follows from them, D-1)
        conn.delay = delay
    gk = FusedKernelTemplate(dst_pop).build()
    n = len(dst_pop.shape)
    gen = _runtime(gk, dst_pop)
    srcs = [commanded_runtime(len(conn._source.shape), uniform=fires.dtype == torch.bool, group_size=G, device=DEVICE)
            for conn, fires in sources]  # (fire times: emission at them)
    if attrs is None:
        attrs = [gk.edge_attr] if gk.edge_attr is not None else []
    topos = []
    for conn, _ in sources:
        res = dst_pop._slot_width() if isinstance(delay, torch.Tensor) else None  # (per link: quantized at it)
        topos.append(conn.fabric.simulation(n, G, DEVICE, resolution=res, attrs=list(attrs)))
    _connect(gen, dst_pop, [c for c, _ in sources], srcs, topos)
    for r in range(3):
        t = round(1.0 + 0.1 * r, 6)
        gen.run(t, t + 0.1, round_id=r)
        for src, (_, fires) in zip(srcs, sources):
            times = torch.where(fires, t, math.inf) if fires.dtype == torch.bool else fires
            command(src, t, t + 0.1, times if r == 0 else None, round_id=r)
    gen.check()
    return gen, topos


def _expected(conn, fires, n, values):
    """Per target: the sum of `values` (canonical, per link) over the links of the firing sources."""
    offs, tgt = conn.fabric.topology.source_offsets, conn.fabric.topology.target_indices.long()
    out = torch.zeros(n, dtype=torch.float64)
    for s in fires.nonzero().flatten().tolist():
        out.index_add_(0, tgt[offs[s]:offs[s + 1]], values[offs[s]:offs[s + 1]].double())
    return out.float()


def test_gap_receive_variants_per_connection():
    """TODO.md, D-6: each connection's port picks its receive code."""
    a, b, dst = PD14_neuron(K_cortical=2000) * 200, PD14_neuron(K_cortical=2000) * 150, _two_ports() * 300
    exc = a >> static_synapse(1.0) @ "exc" @ PD14_connection(0.1) >> dst
    inh = b >> static_synapse(1.0) @ "inh" @ PD14_connection(0.1) >> dst
    exc.fabric.weight, inh.fabric.weight = torch.rand(exc.shape), torch.rand(inh.shape)
    fa, fb = _fires(200, 1), _fires(150, 2)
    gen, _ = _deliver_once(dst, [(exc, fa), (inh, fb)])
    want = _expected(exc, fa, 300, exc.fabric.weight) - _expected(inh, fb, 300, inh.fabric.weight)
    assert torch.allclose(gen.state["acc"].cpu(), want, atol=1e-4)


def test_gap_several_per_link_attrs():
    """TODO.md, D-7: a receive reading two per-link attrs (one block per link)."""
    src, dst = PD14_neuron(K_cortical=2000) * 200, _two_attrs() * 300
    conn = src >> PD14_connection(0.1) >> dst
    conn.weight, conn.gain = torch.rand(conn.shape), torch.rand(conn.shape)
    fires = _fires(200, 3)
    gen, _ = _deliver_once(dst, [(conn, fires)], attrs=("weight", "gain"))
    want = _expected(conn, fires, 300, conn.fabric.weight * conn.fabric.gain)
    assert torch.allclose(gen.state["acc"].cpu(), want, atol=1e-4)


def test_gap_edge_forward():
    """TODO.md, HOOK(forward): plastic_synapse's forward (weight += factor)
    runs per delivered link before receive sees the weight, and its update
    persists in the topology."""
    src, dst = PD14_neuron(K_cortical=2000) * 200, _accumulator() * 300
    conn = src >> plastic_synapse(1.0, 0.5) @ PD14_connection(0.1) >> dst
    fires = _fires(200, 4)
    gen, (topo,) = _deliver_once(dst, [(conn, fires)])
    delivered = _expected(conn, fires, 300, torch.ones(len(conn.fabric.topology)))
    assert torch.allclose(gen.state["acc"].cpu(), 1.5 * delivered, atol=1e-4)
    offs = conn.fabric.topology.source_offsets
    fired_links = torch.cat([torch.arange(offs[s], offs[s + 1]) for s in fires.nonzero().flatten().tolist()])
    weight = topo.read("weight").cpu()
    assert torch.all(weight[fired_links] == 1.5)
    assert int((weight == 1.0).sum()) == len(conn.fabric.topology) - len(fired_links)


def test_gap_edge_forward_destination():
    """TODO.md, D-9: forward reads its target's state (destination.x): each
    delivered link's weight gains 0.1 * the target's acc (1.0 here: every
    target starts there, and a forward runs before its receive)."""
    src, dst = PD14_neuron(K_cortical=2000) * 200, _accumulator(acc=1.0) * 300
    conn = src >> _destination_synapse(1.0) @ PD14_connection(0.1) >> dst
    fires = _fires(200, 5)
    gen, (topo,) = _deliver_once(dst, [(conn, fires)])
    assert torch.all(topo.read("weight").cpu() >= 1.0)
    assert float(topo.read("weight").max()) > 1.0


def test_gap_source_attrs():
    """TODO.md, D-10: receive reads its source's state (source.x); the
    stand-in sources' potential is -65."""
    src, dst = PD14_neuron(K_cortical=2000) * 200, _source_reader() * 300
    conn = src >> PD14_connection(0.1) >> dst
    fires = _fires(200, 6)
    gen, _ = _deliver_once(dst, [(conn, fires)])
    want = _expected(conn, fires, 300, torch.full((len(conn.fabric.topology),), -65.0))
    assert torch.allclose(gen.state["acc"].cpu(), want, atol=1e-3)


def test_receive_time_one_homogeneous_delay_is_exact():
    """D-1 (BRUNEL.md): one homogeneous delay D over every incoming link -- a queue, receive's time the arrival,
    e + D exactly (emission order is arrival order). The sources fire at random times in [1.0, 1.1): every
    target's last input is its latest arrival."""
    src, dst = PD14_neuron(K_cortical=2000) * 200, _arrival_recorder() * 300
    conn = src >> PD14_connection(0.1) >> dst
    fires = _fires(200, 7)
    e = (1.0 + 0.1 * torch.rand(200, generator=torch.Generator().manual_seed(17))).float()
    e = torch.where(fires, e, math.inf)
    gen, _ = _deliver_once(dst, [(conn, e)], delay=0.15)
    assert FusedKernelTemplate(dst).build().consts == dict(T_A=1.0, T_C=0.0, T_B=0.15)
    arrival = e + torch.tensor(0.15, dtype=torch.float32)  # (fp32, as the kernel adds)
    offs, tgt = conn.fabric.topology.source_offsets, conn.fabric.topology.target_indices.long()
    want = torch.full((300,), -1.0)
    for s in fires.nonzero().flatten().tolist():
        idx = tgt[offs[s]:offs[s + 1]]
        want[idx] = torch.maximum(want[idx], arrival[s].expand(len(idx)))
    got = gen.state["last_input"].cpu()
    assert int((want > 0).sum()) > 250  # (most targets reached: ~6 inputs each)
    assert torch.equal(got, want), (got[:5], want[:5])


def _per_link_delays(conn, lo, hi, seed):
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(conn.shape, generator=g)


@pytest.mark.parametrize("slot_time", ["center", "end"])
def test_receive_time_per_link_delays_is_the_slot_s(slot_time):
    """D-1: per-link delays -- every link of a slot delivered at the slot's time, its center (the default) or
    its end (Population.slot_time). Fired at 1.0 with delays in [0.21, 0.29), every link lands in slot 12
    ([1.2, 1.3) at a 0.1 slot width): its time 1.25, or 1.3."""
    src, dst = PD14_neuron(K_cortical=2000) * 200, _arrival_recorder() * 300
    conn = src >> PD14_connection(0.1) >> dst
    dst.delay_resolution = 0.1
    if slot_time != "center":
        dst.slot_time = slot_time
    fires = _fires(200, 9)
    gen, _ = _deliver_once(dst, [(conn, fires)], delay=_per_link_delays(conn, 0.21, 0.29, 9))
    w = torch.tensor(0.1, dtype=torch.float32)
    tk = torch.tensor(12.0) * w + (0.5 if slot_time == "center" else 1.0) * w  # (fp32, as the kernel)
    reached = _expected(conn, fires, 300, torch.ones(len(conn.fabric.topology))) > 0
    got = gen.state["last_input"].cpu()
    assert reached.any()
    assert torch.all(got[reached] == tk), (got[reached][:5], tk)
    assert torch.all(got[~reached] == -1.0)


@n2.node
class _order_checker:
    """Counts the inputs whose time steps back from the one before (D-1: never)."""

    def __init__(self):
        self.last_input: float = -1.0
        self.back: float = 0.0
        self.n: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        if time < self.last_input:
            self.back += 1.0
        self.last_input = time
        self.n += 1.0


def _run_spread(dst, conns, delay, uniform, rounds=8):
    """Every source fires once per round (uniform: at its start, else at a random time in it); the consumer runs
    `rounds` rounds."""
    for conn in conns:
        conn.delay = delay(conn) if callable(delay) else delay
    gk = FusedKernelTemplate(dst).build()
    n = len(dst.shape)
    gen = _runtime(gk, dst)
    sizes = [len(c._source.shape) for c in conns]
    srcs = [commanded_runtime(k, uniform=uniform, group_size=G, device=DEVICE) for k in sizes]
    res = dst._slot_width() if callable(delay) else None  # (per-link delays: quantized at the slot width)
    topos = [c.fabric.simulation(n, G, DEVICE, resolution=res, attrs=[]) for c in conns]
    _connect(gen, dst, conns, srcs, topos)
    g = torch.Generator().manual_seed(23)
    for r in range(rounds):
        t = round(1.0 + 0.1 * r, 6)
        gen.run(t, t + 0.1, round_id=r)
        for src, k in zip(srcs, sizes):
            at = torch.full((k,), t) if uniform else t + 0.1 * torch.rand(k, generator=g)
            times = torch.where(torch.rand(k, generator=g) < 0.3, at, math.inf)
            command(src, t, t + 0.1, times if r < rounds - 3 else None, round_id=r)
    gen.check()
    return gen


@pytest.mark.parametrize("mode", ["homogeneous", "per_link"])
def test_receive_times_never_step_back(mode):
    """D-1: per target, receive's times are non-decreasing -- one homogeneous delay (the queues: exact arrivals,
    merged; sources firing at random times) and per-link delays spanning slots (the slot times; uniform
    sources) alike; two connections, over several rounds."""
    a, b = PD14_neuron(K_cortical=2000) * 200, PD14_neuron(K_cortical=2000) * 150
    dst = _order_checker() * 300
    ca, cb = a >> PD14_connection(0.1) >> dst, b >> PD14_connection(0.1) >> dst
    if mode == "per_link":
        dst.delay_resolution = 0.1
        delay = lambda c: _per_link_delays(c, 0.1, 0.4, len(c._source.shape))
    else:
        delay = 0.15
    gen = _run_spread(dst, [ca, cb], delay, uniform=mode == "per_link")
    assert float(gen.state["n"].sum()) > 1000
    assert float(gen.state["back"].sum()) == 0.0


@n2.node
class _order_hash:
    """Folds its inputs' weights in delivery order (acc / 2 + w: any two swapped show) and counts them."""

    def __init__(self):
        self.acc: float = 0.0
        self.n: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.acc = self.acc * 0.5 + edge.weight
        self.n += 1.0


def _run_queues(dst, conns, delays, uniform, rounds=4, seed=31, weights=True):
    """conns[i] at the homogeneous delays[i], its weight i + 1 on every link (weights); every source fires in rounds
    0..rounds-1 (uniform: at the round's start; else at random times in it), the consumer runs until all has
    arrived. Returns (consumer, [per connection: fire times per source, per round])."""
    for conn, d, wgt in zip(conns, delays, range(1, len(conns) + 1)):
        conn.delay = d
        if weights:
            conn.weight = torch.full(conn.shape, float(wgt))
    gk = FusedKernelTemplate(dst).build()
    n = len(dst.shape)
    gen = _runtime(gk, dst)
    sizes = [len(c._source.shape) for c in conns]
    srcs = [commanded_runtime(k, uniform=uniform, group_size=G, device=DEVICE) for k in sizes]
    topos = [c.fabric.simulation(n, G, DEVICE, attrs=["weight"] if weights else []) for c in conns]
    _connect(gen, dst, conns, srcs, topos)
    g = torch.Generator().manual_seed(seed)
    fired = [[] for _ in conns]
    for r in range(rounds + 4):
        t = round(1.0 + 0.1 * r, 6)
        gen.run(t, t + 0.1, round_id=r)
        for i, (src, k) in enumerate(zip(srcs, sizes)):
            if r < rounds:
                at = torch.full((k,), t) if uniform else t + 0.1 * torch.rand(k, generator=g)
                times = torch.where(torch.rand(k, generator=g) < 0.3, at, math.inf).float()
                fired[i].append(times)
            command(src, t, t + 0.1, times if r < rounds else None, round_id=r)
    gen.check()
    return gen, fired


def _arrival_order_reference(conns, delays, fired, n):
    """Per target, acc and n as _order_hash ends with: its inputs in arrival order (fp32 e + D, as delivery),
    ties by connection (the queues' merge: the lower one first)."""
    per = [[] for _ in range(n)]
    for i, conn in enumerate(conns):
        offs, tgt = conn.fabric.topology.source_offsets, conn.fabric.topology.target_indices.long()
        d = torch.tensor(delays[i], dtype=torch.float32)
        for times in fired[i]:
            arr = times + d
            for s in torch.isfinite(times).nonzero().flatten().tolist():
                for j in tgt[offs[s]:offs[s + 1]].tolist():
                    per[j].append((arr[s].item(), i, float(i + 1)))
    acc, cnt = torch.zeros(n, dtype=torch.float64), torch.zeros(n)
    for j, inputs in enumerate(per):
        for _, _, w in sorted(inputs, key=lambda x: (x[0], x[1])):
            acc[j] = acc[j] * 0.5 + w
        cnt[j] = len(inputs)
    return acc.float(), cnt


@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "timed"])
def test_several_homogeneous_delays_arrive_in_order(uniform):
    """D-1, several homogeneous delays (SPIKES.md): a queue per connection, merged by arrival -- every target's
    inputs in exact arrival order (ties: the lower connection first) against a host reference, each at its
    arrival. Uniform sources: at 0.1 and 0.2, a round's arrivals tie with the previous one's; timed: any
    times."""
    a, b, c = (PD14_neuron(K_cortical=2000) * k for k in (200, 150, 120))
    dst = _order_hash() * 300
    conns = [x >> PD14_connection(0.1) >> dst for x in (a, b, c)]
    delays = [0.2, 0.1, 0.15] if uniform else [0.2, 0.13, 0.17]
    gen, fired = _run_queues(dst, conns, delays, uniform)
    assert gen.queues and gen.ring.n_slots == 3
    want_acc, want_n = _arrival_order_reference(conns, delays, fired, 300)
    assert torch.equal(gen.state["n"].cpu(), want_n)
    assert float(want_n.sum()) > 1000
    assert torch.allclose(gen.state["acc"].cpu(), want_acc, rtol=1e-5, atol=1e-5), \
        (gen.state["acc"].cpu()[:6], want_acc[:6])


@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "timed"])
def test_several_homogeneous_delays_receive_at_the_arrival(uniform):
    """... and each receive's time is the arrival: a target's last input, its latest arrival (fp32 e + D)."""
    a, b = PD14_neuron(K_cortical=2000) * 200, PD14_neuron(K_cortical=2000) * 150
    dst = _arrival_recorder() * 300
    conns = [x >> PD14_connection(0.1) >> dst for x in (a, b)]
    delays = [0.1, 0.25]
    gen, fired = _run_queues(dst, conns, delays, uniform, rounds=3, weights=False)
    want = torch.full((300,), -1.0)
    for i, conn in enumerate(conns):
        offs, tgt = conn.fabric.topology.source_offsets, conn.fabric.topology.target_indices.long()
        for times in fired[i]:
            arr = times + torch.tensor(delays[i], dtype=torch.float32)
            for s in torch.isfinite(times).nonzero().flatten().tolist():
                idx = tgt[offs[s]:offs[s + 1]]
                want[idx] = torch.maximum(want[idx], arr[s].expand(len(idx)))
    assert torch.equal(gen.state["last_input"].cpu(), want)


@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "timed"])
def test_a_burst_past_a_chunk_arrives_in_order(uniform):
    """A source run whose spikes share one time bin past what a chunk merges in LDS (fused.py _pull_queues: QCH,
    1500 spikes at one exact time) -- the queue pull's merge path: every link delivered once, at its arrival;
    timed (the source programs' sorted regions merged): every target's inputs in (arrival, source) order (an
    order hash, its weight the source's, against a host reference). Uniform: one flat list, in its atomic
    appends' order -- the merge path keeps it (the fast path ranks by node): the sum checked, not the order
    (TODO.md: a flat list's order)."""
    n_src = 1500
    src, dst = PD14_neuron(K_cortical=2000) * n_src, (_accumulator() if uniform else _order_hash()) * 300
    conn = src >> PD14_connection(0.1) >> dst
    conn.delay = 0.2
    offs = conn.fabric.topology.source_offsets
    w_src = 1.0 + (torch.arange(n_src) % 13).float() / 16
    conn.weight = w_src.repeat_interleave(offs.diff())
    gen = _runtime(FusedKernelTemplate(dst).build(), dst)
    srcs = [commanded_runtime(n_src, uniform=uniform, group_size=G, device=DEVICE)]
    _connect(gen, dst, [conn], srcs, [conn.fabric.simulation(300, G, DEVICE, attrs=["weight"])])
    assert gen.queues
    for r in range(4):
        t = round(1.0 + 0.1 * r, 6)
        gen.run(t, t + 0.1, round_id=r)
        command(srcs[0], t, t + 0.1, torch.full((n_src,), 1.0 if uniform else 1.05) if r == 0 else None, round_id=r)
    gen.check()
    tgt = conn.fabric.topology.target_indices.long()
    if uniform:
        want = _expected(conn, torch.ones(n_src, dtype=torch.bool), 300, w_src.repeat_interleave(offs.diff()))
        assert torch.allclose(gen.state["acc"].cpu(), want, atol=1e-3)
        assert int(gen.n_received.sum()) == len(conn.fabric.topology)
        return
    acc, cnt = torch.zeros(300, dtype=torch.float64), torch.zeros(300)
    for s in range(n_src):  # (one arrival for all: source order)
        for j in tgt[offs[s]:offs[s + 1]].tolist():
            acc[j] = acc[j] * 0.5 + float(w_src[s])
            cnt[j] += 1
    assert float(cnt.sum()) == len(conn.fabric.topology)
    assert torch.equal(gen.state["n"].cpu(), cnt)
    assert torch.allclose(gen.state["acc"].cpu(), acc.float(), rtol=1e-5, atol=1e-5), \
        (gen.state["acc"].cpu()[:6], acc.float()[:6])


@n2.edge
class _post_synapse:
    """An edge backward: every spike of the link's target raises its weight (D-11)."""

    def __init__(self, weight: float):
        self.weight = weight

    def forward(self):
        pass

    def backward(self):
        self.weight += 0.25


@n2.node
class _always_firing:
    """Fires on every run (bare n2.spike() in send)."""

    def __init__(self):
        self.acc: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        n2.spike()

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.acc += edge.weight


def test_gap_edge_backward():
    """TODO.md, D-11: an edge backward runs on every incoming link of each
    spiking target (a reverse topology). The targets fire on all 3 runs, so
    every link gains 3 * 0.25."""
    src, dst = PD14_neuron(K_cortical=2000) * 200, _always_firing() * 300
    conn = src >> _post_synapse(1.0) @ PD14_connection(0.1) >> dst
    gen, (topo,) = _deliver_once(dst, [(conn, _fires(200, 8))])
    assert len(gen.emitted()) == 300
    assert torch.all(topo.read("weight").cpu() == 1.75)
