"""PopulationRuntimes of the commanded test nodes (test/nodes/commanded.py)
for the fused kernel's GPU tests: sources firing on command, consumers
counting what they receive. Their kernels are generated once per node and
reused (every generated module is a new JIT function, compiled on first
use)."""

import math

import torch

from naer2.runtime import PopulationRuntime
from naer2.transpiler.templates.fused_template import FusedKernelTemplate
from test.nodes import commanded, commanded_timed

_KERNELS = {}


def _kernel(node_type):
    if node_type not in _KERNELS:
        _KERNELS[node_type] = FusedKernelTemplate(node_type() * 1).build()
    return _KERNELS[node_type]


def commanded_runtime(n: int, uniform: bool = True, group_size: int = 32, depth: int = 3, device="cuda:0"):
    """A population of commanded (uniform: bare n2.spike(), emission at the
    window's start) or commanded_timed nodes (emission at a given time)."""
    return PopulationRuntime(_kernel(commanded if uniform else commanded_timed), n, group_size=group_size,
                             depth=depth, device=device)


def command(rt, t_start, t_end, fire_time=None, emit_delay=None, **run):
    """One run of a commanded runtime: the neurons whose fire time lies in
    [t_start, t_end) spike -- at t_start (uniform) or at fire time + emission
    delay (timed) -- then rt.run(t_start, t_end, **run)."""
    if fire_time is not None:
        fire_time = fire_time.to(rt.device, torch.float32)
        fires = (fire_time >= t_start) & (fire_time < t_end)
        rt.state["fire_now"] = fires.float()
        if "fire_e" in rt.state:
            delay = emit_delay.to(rt.device, torch.float32) if emit_delay is not None else torch.zeros_like(fire_time)
            rt.state["fire_e"] = torch.where(fires, fire_time + delay, 0.0)
    rt.run(t_start, t_end, **run)


def no_fire(n):
    return torch.full((n,), math.inf)
