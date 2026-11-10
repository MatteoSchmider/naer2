"""How many programs (4 warps each) run at once on this GPU as a function of
LDS per program: a Gluon kernel allocates S bytes of LDS, touches them, spins
~50 us in a pure ALU loop (spinning on the clock itself distorts it) and
stamps its start and end; the most programs
alive at once is the occupancy. Low register use, so only LDS (and the wave
slots) limit it.

    python tools/occupancy_lds.py [--rdna2]
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from naer2.trace import realtime


@gluon.jit
def spin(out, WORDS: gl.constexpr, WORDS2: gl.constexpr, SPIN: gl.constexpr, NUM_WARPS: gl.constexpr):
    """WORDS (+ WORDS2, if nonzero) int32 of LDS; shapes must be powers of 2."""
    L: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    SM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    pid = gl.program_id(0)
    t0 = realtime()
    buf = gl.allocate_shared_memory(gl.int32, [WORDS], SM)
    buf.store(gl.arange(0, WORDS, layout=gl.BlockedLayout([WORDS // (32 * NUM_WARPS)], [32], [NUM_WARPS], [0])))
    if WORDS2 > 0:
        buf2 = gl.allocate_shared_memory(gl.int32, [WORDS2], SM)
        buf2.store(gl.arange(0, WORDS2, layout=gl.BlockedLayout([WORDS2 // (32 * NUM_WARPS)], [32], [NUM_WARPS], [0])))
    gl.barrier()
    x = buf.slice(0, 32 * NUM_WARPS).load(L).to(gl.float32)
    for _ in range(SPIN):  # dependent FMAs: pure ALU, no memory
        x = x * 1.000001 + 0.5
    gl.store(out + 2 * pid, t0)
    gl.store(out + 2 * pid + 1, realtime() + (gl.sum(x, axis=0) * 0.0).to(gl.int64))


def alive(words: int, n: int = 2000) -> tuple[int, int]:
    out = torch.zeros(2 * n, dtype=torch.int64, device="cuda")
    w1 = 1 << (words.bit_length() - 1)
    k = spin[(n,)](out, w1, words - w1, 10000, 4, num_warps=4)
    torch.cuda.synchronize()
    s, e = out[0::2].tolist(), out[1::2].tolist()
    ev = sorted([(x, 1) for x in s] + [(x, -1) for x in e])
    a = m = 0
    for _, d in ev:
        a += d
        m = max(m, a)
    return m, k.metadata.shared, k.n_regs


props = torch.cuda.get_device_properties(0)
n_wgp = props.multi_processor_count  # torch reports WGPs on RDNA
print(f"{props.name} ({props.gcnArchName}), {n_wgp} WGPs; programs of 4 warps")
print(f"{'LDS/program':>12s} {'compiled':>9s} {'programs alive':>15s} {'per WGP':>8s} {'LDS in use/WGP':>15s} {'VGPRs':>6s}")
for kb in (1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64):
    m, shared, regs = alive(kb * 256)
    print(f"{kb:9d} KB {shared / 1024:8.1f}K {m:15d} {m / n_wgp:8.1f} {m / n_wgp * shared / 1024:13.0f}KB {regs:6d}")
