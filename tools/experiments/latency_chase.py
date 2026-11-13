"""Latency of one dependent memory round trip on this GPU: a single program
follows a random pointer chain (each load's address is the previous load's
value) through a buffer that fits a cache level or doesn't; time per step
from the kernel's own wall-clock stamps (naer2/trace.py), clock kept up by
a warm-up. Also an LDS round trip (load -> address -> load) for scale.

    python tools/experiments/latency_chase.py [--rdna2]
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", ".."), os.path.join(os.path.dirname(__file__), "..")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from naer2.trace import realtime


@triton.jit
def chase(buf, out, STEPS: tl.constexpr):
    t0 = realtime()
    i = tl.load(buf)
    for _ in range(STEPS):
        i = tl.load(buf + i)
    t1 = realtime()
    tl.store(out, i.to(tl.int64))
    tl.store(out + 1, t1 - t0)


@gluon.jit
def chase_lds(out, STEPS: gl.constexpr, N: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    x = gl.arange(0, N, layout=L)
    s = gl.allocate_shared_memory(gl.int32, [N], SM)
    s.store((x * 7 + 3) % N)
    gl.barrier()
    i = gl.zeros([N], gl.int32, layout=L)
    t0 = realtime()
    for _ in range(STEPS):
        i = s.gather(i, 0)
    t1 = realtime()
    gl.store(out + x, i.to(gl.int64))
    gl.store(out + N, t1 - t0)


def ns_per_step(n_words, steps=4096, reps=5):
    perm = torch.randperm(n_words, device="cuda")
    buf = torch.empty(n_words, device="cuda", dtype=torch.int32)
    buf[perm] = perm.roll(1).to(torch.int32)  # one random cycle through the whole buffer
    out = torch.zeros(2, device="cuda", dtype=torch.int64)
    best = None
    for _ in range(reps):
        chase[(1,)](buf, out, steps, num_warps=1)
        torch.cuda.synchronize()
        t = out[1].item() * 10 / steps  # REALTIME: 10 ns ticks
        best = t if best is None else min(best, t)
    return best


x = torch.empty(64 << 20, device="cuda")
for _ in range(200):  # warm up: keep the clock up
    x.mul_(1.0001)
torch.cuda.synchronize()
def heat():
    for _ in range(100):
        x.mul_(1.0001)


print(f"{torch.cuda.get_device_name(0)}: one dependent load (pointer chase, one lane); "
      f"sizes forward and backward, the clock kept up between")
sizes = (16, 256, 2048, 16384, 262144)
res = {kb: [] for kb in sizes}
for order in (sizes, sizes[::-1], sizes, sizes[::-1]):
    for kb in order:
        heat()
        res[kb].append(ns_per_step(kb * 256))
for kb in sizes:
    print(f"  buffer {kb:7d} KB: {min(res[kb]):5.0f} ns per round trip (runs: "
          + " ".join(f"{v:.0f}" for v in res[kb]) + ")")
out = torch.zeros(65, device="cuda", dtype=torch.int64)
chase_lds[(1,)](out, 4096, 64, num_warps=1)
torch.cuda.synchronize()
print(f"  LDS gather (load -> index -> load): {out[64].item() * 10 / 4096:7.0f} ns per step")
