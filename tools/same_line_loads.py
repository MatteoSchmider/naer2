"""Do many programs loading the same addresses at once slow each other down? N programs each follow a chain of
dependent loads (pointer chasing, every lane the same address, as the fused kernel's prologue: entry -> plan
-> history -> header): one chain shared by all programs, or a copy per program. Per program its traced
wall-clock time (median, max), for N = 1 .. 128.

    python tools/same_line_loads.py [--steps=8] [--warps=4] [--rdna2]
"""

import os
import statistics
import sys

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch
import triton
import triton.language as tl

from naer2.trace import realtime

opts = {a.split("=")[0][2:]: int(a.split("=")[1]) for a in sys.argv if a.startswith("--") and "=" in a}
STEPS, WARPS = opts.get("steps", 8), opts.get("warps", 4)
SPREAD = 4096  # (words between a chain's links: every step its own cache line)


@triton.jit
def chase(chain, stride, out, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = chain + pid * stride  # (stride 0: every program the same chain)
    lane = tl.arange(0, BLOCK)
    t0 = realtime()
    at = tl.zeros([BLOCK], tl.int32) + tl.load(base + lane * 0)
    for _ in tl.static_range(STEPS - 1):
        at = tl.load(base + at)
    t1 = realtime()
    tl.store(out + pid * 2 + lane * 0, (t1 - t0).to(tl.int32) + at * 0, mask=lane == 0)
    tl.store(out + pid * 2 + 1 + lane * 0, at, mask=lane == 0)


programs_max = 128
chain = torch.zeros((programs_max, SPREAD * (STEPS + 1)), dtype=torch.int32, device="cuda")
for k in range(STEPS):
    chain[:, k * SPREAD] = (k + 1) * SPREAD  # step k's word -> step k+1's
out = torch.zeros(programs_max * 2, dtype=torch.int32, device="cuda")
print(f"{torch.cuda.get_device_name(0)}: a chain of {STEPS} dependent loads per program, {WARPS} warps, "
      f"per program us (median / max over 20 launches' programs)")
for n in (1, 4, 16, 32, 64, 86, 128):
    row = []
    for stride, label in ((0, "shared"), (chain.shape[1], "own copy")):
        times = []
        for rep in range(22):
            if stride == 0:
                chain[0].add_(0)  # (the same lines freshly written each launch, as a source's headers are)
            else:
                chain[:n].add_(0)
            torch.cuda.synchronize()
            chase[(n,)](chain, stride, out, STEPS=STEPS, BLOCK=32 * WARPS, num_warps=WARPS)
            torch.cuda.synchronize()
            if rep >= 2:
                times += (out[0::2][:n].float() / 100).tolist()
        row.append(f"{label} {statistics.median(times):5.2f} / {max(times):5.2f}")
    print(f"  N={n:3d}: " + "   ".join(row))
