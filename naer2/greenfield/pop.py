"""
A green-field population: PD14-like neuron state, its EmissionBuffer, and
with connect() the consumer side -- its rings, its IntakePlan over the
incoming connections, the stand-in receive's buffers. run() launches the
update kernel (naer2/greenfield/update.py) for one run; intake is the
caller's (pop.plan.run), as in Network.step.
"""

import torch
import triton

from naer2.emission import EmissionBuffer
from naer2.intake import IntakePlan, Ring
from naer2.trace import Trace

from .update import UPDATE_TRACE_TAGS, update_kernel

# delivery walk defaults, from tools/walk_sweep.py (PD14, G=256, 780M):
# entries per chunk 128, link stream 256, links per LDS read 8. Larger U
# costs more in registers than it saves; an explicit occupancy target
# (waves_per_eu) gains <= 3% and spills beyond 12 waves.
WALK = dict(entry_block=128, stream=256, unroll=8)


class Pop:
    def __init__(self, n: int, cap: int = 64, group_size: int = 256, device="cuda"):
        self.n, self.group_size, self.device = n, group_size, torch.device(device)
        self.n_groups = -(-n // group_size)
        self.potential = torch.full((n,), -65.0, device=device)
        self.current = torch.zeros(n, device=device)
        self.t_ref_end = torch.zeros(n, device=device)
        self.last_time = torch.zeros(n, device=device)
        self.emission = EmissionBuffer(self.n_groups, cap, device)
        self.run_id = 0
        self.received = torch.zeros(n, device=device)
        self.n_received = torch.zeros(n, device=device, dtype=torch.int32)
        self.entry_block, self.stream, self.unroll = WALK["entry_block"], WALK["stream"], WALK["unroll"]
        self.count_received = True  # False compiles the stand-in's receive counters out (benchmarks)
        self.waves_per_eu = 0       # occupancy the register allocator targets (0: its own choice)
        self.trace = None
        self.connect([], [], n_slots=1)

    # ---- consumer side

    def connect(self, sources: list, topologies: list, n_slots: int, slot_width: float = 0.1, log_cap: int = 0,
                main_cap: int | None = None, side_cap: int | None = None):
        caps = {k: v for k, v in (("main_cap", main_cap), ("side_cap", side_cap)) if v}
        self.ring = Ring(self.n_groups, n_slots, slot_width, self.device, **caps)
        self.plan = IntakePlan(self.ring, [p.emission for p in sources], topologies)
        self.log_cap = log_cap
        self.log = torch.zeros(1 + 5 * max(log_cap, 1), device=self.device, dtype=torch.int32)

    # ---- one run

    def run(self, t_start, t_end, fire_time, emit_delay=None, uniform=True, deliver_until=None, num_warps=4):
        """One run of the update kernel over [t_start, t_end): deliver every
        ring slot that ends by `deliver_until` (default t_end), then emit."""
        self.run_id += 1
        w = self.ring.slot_width
        k_lo = self.ring.ring_pos
        k_hi = int(((deliver_until if deliver_until is not None else t_end) + 1e-6) // w) - 1
        emit_delay = emit_delay if emit_delay is not None else torch.zeros(self.n)
        lanes = 32 * num_warps
        update_kernel[(self.n_groups,)](
            self.t_ref_end, self.potential, self.current, self.last_time, self.n,
            self.plan.plan, self.plan.n_conns, self.ring.buf, k_lo, k_hi, w,
            self.received, self.n_received, self.log, self.log_cap,
            fire_time.to(self.device), emit_delay.to(self.device),
            self.emission.buf, self.run_id, t_start, t_end,
            self.trace.buf if self.trace is not None else self.log,
            GROUP_SIZE=self.group_size, LOCAL_SPIKE_CAP=self.emission.cap, UNIFORM_TIME=uniform,
            N_SLOTS=self.ring.n_slots, MAIN_CAP=self.ring.main_cap, SIDE_CAP=self.ring.side_cap,
            ENTRY_BLOCK=self.entry_block, LOG=self.log_cap > 0, STREAM=self.stream, U=self.unroll,
            SPT=max(self.group_size // lanes, 1), NUM_WARPS=num_warps,
            MAX_CONNS=triton.next_power_of_2(max(self.plan.n_conns, 1)),
            SPQ=max(self.stream // lanes, 1), SPE=max(self.entry_block // lanes, 1), COUNT=self.count_received,
            TRACE=int(self.trace is not None), TRACE_CAP=self.trace.cap if self.trace is not None else 1,
            num_warps=num_warps, waves_per_eu=self.waves_per_eu,
        )
        self.ring.ring_pos = max(k_lo, k_hi + 1)

    # ---- inspection

    def set_trace(self, cap: int | None = 1024):
        """Stamp the following runs (naer2/trace.py; each run overwrites); None: stop."""
        self.trace = Trace(self.n_groups, cap, self.device) if cap else None

    def traced(self) -> list[dict]:
        """Per program of the last traced run: see Trace.decode, tags UPDATE_TRACE_TAGS."""
        return self.trace.decode(UPDATE_TRACE_TAGS)

    def deliveries(self) -> list[tuple[int, float, int, int, int]]:
        """The delivery log, in log order: (slot, e, connection, source, target)."""
        n = min(int(self.log[0]), self.log_cap)
        rec = self.log[1:1 + 4 * self.log_cap].view(-1, 4)[:n].tolist()
        e = self.log[1 + 4 * self.log_cap:].view(torch.float32)[:n].tolist()
        return [(k, ei, c, s, t) for (k, c, s, t), ei in zip(rec, e)]

    def emitted(self) -> dict[int, list[tuple[float, int]]]:
        """group -> [(time, node)] in buffer order."""
        cap = self.emission.cap
        time, node, count = self.emission.time.tolist(), self.emission.node.tolist(), self.emission.count.tolist()
        return {b: [(time[b * cap + i], node[b * cap + i]) for i in range(c)] for b, c in enumerate(count)}

    def flat(self) -> list[tuple[float, int]]:
        """The last run's flat spike list: [(time, node)], length = its summary count."""
        n, _ = self.emission.summary(self.run_id)
        return list(zip(self.emission.flat_time[:n].tolist(), self.emission.flat_node[:n].tolist()))
