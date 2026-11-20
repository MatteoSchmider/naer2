"""PD14's fused kernels' resources and occupancy: per population its programs (groups), warps, registers per
thread and shared memory per program (the compiled kernel's), how many of its programs fit on one SM / CU at
once (by threads, registers, shared memory, the block limit), and what a round asks of the whole GPU.

    python tools/pd14_occupancy.py [--G=256] [--rdna2]

The limits are torch's device properties (registers / shared memory per multiprocessor); a round's programs
all resident at once needs their sum within the GPU's -- else the last ones wait for the first to finish.
"""

import os
import sys

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch

import naer2.runtime as R
import pd14_network as P

opts = {a.split("=")[0][2:]: int(a.split("=")[1]) for a in sys.argv if a.startswith("--") and "=" in a}
G = opts.get("G", 256)

compiled = {}  # id(runtime's launchers dict) -> (kernel, programs)
orig = R._direct_launch


def recording(kernel, n_programs, consts):
    compiled.setdefault(n_programs, []).append(kernel)
    return orig(kernel, n_programs, consts)


R._direct_launch = recording
obj = P._network(G)
pops = P._runtimes("network", obj)
for r in range(20):
    P._advance("network", obj, r)
torch.cuda.synchronize()

props = torch.cuda.get_device_properties(0)
sms = props.multi_processor_count
warp = getattr(props, "warp_size", 32)
max_threads = getattr(props, "max_threads_per_multi_processor", 2048)
regs_sm = getattr(props, "regs_per_multiprocessor", 65536)
smem_sm = getattr(props, "shared_memory_per_multiprocessor", 0) or getattr(props, "max_shared_memory_per_multiprocessor", 0)
max_blocks = 32 if torch.version.hip else (24 if props.major == 8 and props.minor == 9 else 32)
print(f"{props.name}: {sms} multiprocessors, warp {warp}, {max_threads} threads, {regs_sm} registers, "
      f"{smem_sm // 1024} KiB shared memory each; at most {max_blocks} blocks (assumed)")
print(f"  {'pop':5s} {'neurons':>7s} {'programs':>8s} {'warps':>5s} {'regs':>5s} {'smem KiB':>8s} "
      f"{'fit/SM':>6s} {'limit':>8s}")
total_prog, total_sm_share = 0, 0.0
rows = []
for name, p in pops.items():
    rt = P._runtime("network", p)
    ks = compiled.get(rt.n_groups, [])
    k = next((k for k in ks if k.metadata.name.startswith("fused")), ks[-1] if ks else None)
    if k is None:
        continue
    nw = k.metadata.num_warps
    regs = getattr(k, "n_regs", 0)
    smem = k.metadata.shared
    threads = nw * warp
    fits = {"threads": max_threads // threads, "registers": regs_sm // max(regs * threads, 1),
            "shared": smem_sm // smem if smem else 999, "blocks": max_blocks}
    limit = min(fits, key=fits.get)
    per_sm = fits[limit]
    total_prog += rt.n_groups
    total_sm_share += rt.n_groups / per_sm
    rows.append((name, rt.n, rt.n_groups, nw, regs, smem, per_sm, limit))
for name, n, groups, nw, regs, smem, per_sm, limit in sorted(rows, key=lambda r: -r[2]):
    print(f"  {name:5s} {n:7d} {groups:8d} {nw:5d} {regs:5d} {smem / 1024:8.1f} {per_sm:6d} {limit:>8s}")
print(f"  a round: {total_prog} programs, needing {total_sm_share:.1f} of {sms} multiprocessors' room at once "
      f"({'fits' if total_sm_share <= sms else 'does NOT fit'}); programs per multiprocessor on average "
      f"{total_prog / sms:.1f}")
