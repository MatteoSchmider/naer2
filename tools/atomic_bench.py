"""Throughput of returning atomic_add (the intake's append) on this GPU:
P programs x TILES tiles of BLOCK lanes, each lane one atomic_add on
counter[target], targets drawn over n_addr addresses -- many (no sharing)
vs few (many lanes and programs on one address, like uniform intake's
slot counters). args: [--rdna2]"""
import os
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch
import triton
import triton.language as tl


@triton.jit
def appends(counter, targets, out, TILES: tl.constexpr, BLOCK: tl.constexpr, RETURN: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], tl.int32)
    for i in range(TILES):
        tgt = tl.load(targets + (pid * TILES + i) * BLOCK + tl.arange(0, BLOCK))
        if RETURN:
            acc += tl.atomic_add(counter + tgt, 1)
        else:
            tl.atomic_add(counter + tgt, 1, sem="relaxed")
    tl.store(out + pid * BLOCK + tl.arange(0, BLOCK), acc)


P, TILES, BLOCK = 24, 64, 512
n = P * TILES * BLOCK
print(f"{torch.cuda.get_device_name(0)}: {P} programs x {TILES} tiles x {BLOCK} atomics = {n:,}")
for n_addr in (1 << 20, 1 << 14, 4096, 860, 128, 16):
    for ret in (True, False):
        counter = torch.zeros(1 << 20, dtype=torch.int32, device="cuda")
        targets = torch.randint(0, n_addr, (n,), device="cuda", dtype=torch.int32)
        out = torch.zeros(P * BLOCK, dtype=torch.int32, device="cuda")
        appends[(P,)](counter, targets, out, TILES, BLOCK, ret, num_warps=4)
        torch.cuda.synchronize()
        ts = []
        for _ in range(5):
            counter.zero_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            appends[(P,)](counter, targets, out, TILES, BLOCK, ret, num_warps=4)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        dt = min(ts)
        print(f"  {n_addr:8,d} addresses ({n / n_addr:8.1f} atomics each), {'returning' if ret else 'no return'}: "
              f"{dt * 1e6:8.0f} us -> {n / dt / 1e6:7.0f} atomics/us, {dt / (TILES) * 1e6:6.1f} us per tile")
