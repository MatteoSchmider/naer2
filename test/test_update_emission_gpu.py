"""
Green-field update kernel (hand-written, outside population.py): the PD14
neuron kernel stripped down to what the spike-management redesign needs,
to pin down the update kernel's OUTPUT contract and its interface to the
bucketed topology and the intake ring before it goes back into the
transpiler templates.

  kept:        state loads/stores and the final propagation (PD14's exact
               propagator) up to the run's window end
  stand-in:    receive/send -- a neuron emits iff its given fire time lies
               in the run's window [t_start, t_end); its emission time is
               t_start (UNIFORM_TIME, i.e. bare n2.spike()) or its fire time
               + its emission delay
  contract:    naer2/emission.py's EmissionBuffer -- per group sorted by e
               (once at the source, so intake only merges), the source
               state (run id, window end), the summary (count, earliest e)
               for scheduling
  interface:   the consumer's per-group rings (due slots k_lo..k_hi) and a
               connection's TopologyNew -- passed, not walked yet (next step:
               delivery)

The emission output is fed into the real intake kernel and checked against
test_intake_gpu.py's reference end to end.
"""

import math
import random

import pytest
import torch
import triton
import triton.language as tl

from naer2.emission import EmissionBuffer, emission_ptrs
from naer2.intake import IntakePlan, Ring
from naer2.spike_gather import _from_ordered_u32, _to_ordered_u32
from naer2.topology import Topology
from naer2.topology_new import TopologyNew
from test.test_intake_gpu import Reference, _check, _topology

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (CUDA/ROCm) to launch the kernel"
)

DEVICE = "cuda:0"
GROUP_SIZE = 32


@pytest.fixture(autouse=True, scope="module")
def _device():
    torch.cuda.set_device(0)


# ------------------------------------------------------------ the kernel


@triton.jit(do_not_specialize=["n_neurons", "k_lo", "k_hi", "run_id"])
def update_kernel(
    t_ref_end_ptr, potential_ptr, current_ptr, last_time_ptr, n_neurons,
    topo, ring, k_lo, k_hi,            # delivery input (next step): not walked yet
    fire_time_ptr, emit_delay_ptr,     # stand-in for receive/send
    emission, run_id, t_start, t_end,
    GROUP_SIZE: tl.constexpr, LOCAL_SPIKE_CAP: tl.constexpr, UNIFORM_TIME: tl.constexpr,
):
    pid = tl.program_id(0)
    n_groups = tl.num_programs(0)
    offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = offs < n_neurons
    (spike_node, spike_time, spike_count, state_run, state_end,
     sum_count, sum_min, overflow) = emission_ptrs(emission, n_groups, LOCAL_SPIKE_CAP)
    local_spike_count = tl.cast(0, tl.int32)

    t_ref_end = tl.load(t_ref_end_ptr + offs, mask=mask)
    potential = tl.load(potential_ptr + offs, mask=mask)
    current = tl.load(current_ptr + offs, mask=mask)
    last_time = tl.load(last_time_ptr + offs, mask=mask)

    # TODO (delivery): for each due slot k_lo..k_hi of this group's ring,
    # walk its main and side runs with two cursors in emission order; every
    # entry is one segment of `topo` -- walk its link range [seg_start[i],
    # seg_start[i + 1]) and run receive/send.

    # final propagation to the window end (PD14)
    time = t_end
    dt = time - last_time
    last_time = time
    P_V = tl.exp(-dt / 10.0)
    P_I = tl.exp(-dt / 0.5)
    P_VI = 40.0 * 0.5 / (10.0 - 0.5) * (P_V - P_I)
    _m0 = mask & (time < t_ref_end)
    potential = tl.where(_m0, -65.0, potential)
    potential = tl.where(~_m0 & mask, -38.30576 + P_V * (potential - -38.30576) + P_VI * current, potential)
    current = P_I * current

    # stand-in send: emit per the given fire times
    fire = tl.load(fire_time_ptr + offs, mask=mask, other=float("inf"))
    fires = mask & (fire >= t_start) & (fire < t_end)
    if UNIFORM_TIME:
        e = t_start + tl.zeros_like(fire)
    else:
        e = fire + tl.load(emit_delay_ptr + offs, mask=mask, other=0.0)
    _n_fire = fires.to(tl.int32)
    _local_pos = local_spike_count + tl.cumsum(_n_fire, axis=0) - _n_fire
    local_spike_count = local_spike_count + tl.sum(_n_fire, axis=0)
    _fits = fires & (_local_pos < LOCAL_SPIKE_CAP)
    tl.store(spike_node + pid * LOCAL_SPIKE_CAP + _local_pos, offs, mask=_fits)
    tl.store(spike_time + pid * LOCAL_SPIKE_CAP + _local_pos, e, mask=_fits)
    potential = tl.where(fires, -65.0, potential)
    t_ref_end = tl.where(fires, time + 2.0, t_ref_end)

    tl.store(t_ref_end_ptr + offs, t_ref_end, mask=mask)
    tl.store(potential_ptr + offs, potential, mask=mask)
    tl.store(current_ptr + offs, current, mask=mask)
    tl.store(last_time_ptr + offs, last_time, mask=mask)

    # emission contract: sort the group's entries by e (once, at the source)
    if local_spike_count > LOCAL_SPIKE_CAP:
        tl.atomic_max(overflow, 1)
    n = tl.minimum(local_spike_count, LOCAL_SPIKE_CAP)
    slot = tl.arange(0, LOCAL_SPIKE_CAP)
    valid = slot < n
    if (not UNIFORM_TIME) and (n > 1):  # most groups emit 0 or 1 spikes per run
        tl.debug_barrier()
        t = tl.load(spike_time + pid * LOCAL_SPIKE_CAP + slot, mask=valid, other=float("inf"))
        node = tl.load(spike_node + pid * LOCAL_SPIKE_CAP + slot, mask=valid, other=0)
        key = (_to_ordered_u32(t).to(tl.uint64) << 32) | node.to(tl.uint32).to(tl.uint64)
        key = tl.sort(key)
        tl.debug_barrier()
        tl.store(spike_node + pid * LOCAL_SPIKE_CAP + slot, (key & 0xFFFFFFFF).to(tl.int32), mask=valid)
        tl.store(spike_time + pid * LOCAL_SPIKE_CAP + slot, _from_ordered_u32((key >> 32).to(tl.uint32)), mask=valid)
    tl.store(spike_count + pid, n)

    # summary for scheduling (per run parity; program 0 resets the other
    # parity for the next run) and the source state intake reads
    p = run_id % 2
    tl.atomic_add(sum_count + p, n)
    tl.atomic_min(sum_min + p, tl.min(tl.where(fires & (_local_pos < LOCAL_SPIKE_CAP), e, float("inf")), axis=0))
    if pid == 0:
        tl.store(sum_count + (1 - p), 0)
        tl.store(sum_min + (1 - p), float("inf"))
        tl.store(state_run, run_id)
        tl.store(state_end, t_end)


# ------------------------------------------------------------ helpers


class Pop:
    """A PD14-like population's state plus its emission buffer."""

    def __init__(self, n: int, cap: int = 64):
        self.n = n
        self.n_groups = -(-n // GROUP_SIZE)
        self.potential = torch.full((n,), -65.0, device=DEVICE)
        self.current = torch.zeros(n, device=DEVICE)
        self.t_ref_end = torch.zeros(n, device=DEVICE)
        self.last_time = torch.zeros(n, device=DEVICE)
        self.emission = EmissionBuffer(self.n_groups, cap, DEVICE)
        self.run_id = 0
        empty = Topology(torch.zeros(2, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))
        self.topo = TopologyNew(empty, n, GROUP_SIZE, DEVICE)
        self.ring = Ring(self.n_groups, 4, 0.1, DEVICE, main_cap=16, side_cap=16)

    def run(self, t_start, t_end, fire_time, emit_delay=None, uniform=True):
        self.run_id += 1
        update_kernel[(self.n_groups,)](
            self.t_ref_end, self.potential, self.current, self.last_time, self.n,
            self.topo.buf, self.ring.buf, 0, -1,
            fire_time.to(DEVICE), (emit_delay if emit_delay is not None else torch.zeros(self.n)).to(DEVICE),
            self.emission.buf, self.run_id, t_start, t_end,
            GROUP_SIZE=GROUP_SIZE, LOCAL_SPIKE_CAP=self.emission.cap, UNIFORM_TIME=uniform,
        )

    def emitted(self) -> dict[int, list[tuple[float, int]]]:
        """group -> [(time, node)] in buffer order."""
        cap = self.emission.cap
        time, node, count = self.emission.time.tolist(), self.emission.node.tolist(), self.emission.count.tolist()
        return {b: [(time[b * cap + i], node[b * cap + i]) for i in range(c)] for b, c in enumerate(count)}


def _fire_pattern(n, t_start, t_end, p, rng):
    """Per neuron: a fire time inside the window with probability p, else inf."""
    return torch.tensor([t_start + rng.random() * (t_end - t_start) if rng.random() < p else math.inf for _ in range(n)])


# ------------------------------------------------------------ tests


def test_uniform_emission_buffer_state_and_summary():
    rng = random.Random(0)
    pop = Pop(200)
    fire = _fire_pattern(200, 1.0, 1.1, 0.3, rng)
    pop.run(1.0, 1.1, fire, uniform=True)

    want = sorted((1.0, i) for i in range(200) if fire[i] < math.inf)
    got = sorted(entry for entries in pop.emitted().values() for entry in entries)
    assert [(pytest.approx(t), i) for t, i in want] == got
    for g, entries in pop.emitted().items():
        assert all(g * GROUP_SIZE <= i < (g + 1) * GROUP_SIZE for _, i in entries)
    assert pop.emission.state[0].item() == 1
    assert pop.emission.state[1:].view(torch.float32).item() == pytest.approx(1.1)
    assert pop.emission.summary(1) == (len(want), pytest.approx(1.0))
    assert torch.allclose(pop.last_time, torch.full((200,), 1.1, device=DEVICE))


def test_exact_times_with_emission_delays_are_sorted_per_group():
    rng = random.Random(1)
    pop = Pop(200)
    fire = _fire_pattern(200, 1.0, 1.1, 0.5, rng)
    delay = torch.tensor([rng.choice([0.0, 0.0, 0.03, 0.4]) for _ in range(200)])
    pop.run(1.0, 1.1, fire, delay, uniform=False)

    e = (fire.to(torch.float32) + delay.to(torch.float32)).tolist()  # the kernel's float32 sum
    for g, entries in pop.emitted().items():
        times = [t for t, _ in entries]
        assert times == sorted(times), f"group {g} not sorted: {times}"
        want = sorted((e[i], i) for i in range(g * GROUP_SIZE, min((g + 1) * GROUP_SIZE, 200)) if fire[i] < math.inf)
        assert sorted(entries) == want
    fired = [i for i in range(200) if fire[i] < math.inf]
    assert pop.emission.summary(1) == (len(fired), min(e[i] for i in fired))


def test_summary_is_per_run():
    rng = random.Random(2)
    pop = Pop(100)
    pop.run(1.0, 1.1, _fire_pattern(100, 1.0, 1.1, 0.5, rng))
    fire2 = _fire_pattern(100, 1.1, 1.2, 0.1, rng)
    pop.run(1.1, 1.2, fire2)
    assert pop.emission.summary(2) == (int((fire2 < math.inf).sum()), pytest.approx(1.1))
    pop.run(1.2, 1.3, torch.full((100,), math.inf))
    assert pop.emission.summary(3) == (0, math.inf)


def test_group_overflow_is_flagged_not_written_past():
    pop = Pop(64, cap=8)
    pop.run(1.0, 1.1, torch.full((64,), 1.05))  # 32 spikes per group, cap 8
    assert pop.emission.overflow.item() == 1
    assert pop.emission.count.tolist() == [8, 8]


@pytest.mark.parametrize("uniform", [True, False])
def test_emission_feeds_intake_end_to_end(uniform):
    """Update kernel -> emission buffer -> real intake kernel -> ring, over
    three rounds (the second population doesn't run in round 2), against
    the intake reference."""
    rng = random.Random(3)
    a, b = Pop(150), Pop(90)
    # into a consumer of 64 neurons in 4 groups: per-link and homogeneous delay
    topologies = [_topology(150, 64, 16, seed=0), _topology(90, 64, 16, delay=0.2, seed=1)]
    ring, ref = Ring(4, 32, 0.1, DEVICE), Reference(4, 32, 0.1)
    plan = IntakePlan(ring, [a.emission, b.emission], topologies)
    for r, t in enumerate((1.0, 1.1, 1.2)):
        ran = [a] if r == 1 else [a, b]
        for pop in ran:
            pop.run(t, t + 0.1, _fire_pattern(pop.n, t, t + 0.1, 0.3, rng),
                    torch.tensor([rng.choice([0.0, 0.0, 0.2]) for _ in range(pop.n)]), uniform=uniform)
        plan.run(uniform)
        ref.intake(0, [a.emission, b.emission], [p in ran for p in (a, b)], [t + 0.1] * 2, topologies)
    ring.check()
    _check(ring, ref)
    assert ring.main_count.sum().item() > 0


def test_kernel_takes_a_segment_topology():
    """The delivery interface: a real incoming connection's TopologyNew is
    accepted (walking it is the next step)."""
    pop = Pop(96)
    pop.topo = _topology(50, 96, GROUP_SIZE, seed=4)
    pop.run(1.0, 1.1, torch.full((96,), math.inf))
    assert pop.emission.summary(1) == (0, math.inf)
