"""Compile the fused kernel (naer2/greenfield/fused.py) for a small consumer
and summarize its ISA: registers, spills, LDS, and per loop body the
instruction mix and the highest VGPR it touches (where register pressure
peaks).

    python tools/kernel_isa.py G [unroll] [counters 1|0] [stream]
    env: N_CONNS (incoming connections, default 1), WAVES (waves_per_eu target)
"""
import math
import os
import re
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..")]
import torch

from naer2.greenfield import FusedPop, fused_kernel
from test.topologies import random_topology

G = int(sys.argv[1])
unroll = int(sys.argv[2]) if len(sys.argv) > 2 else None
count = (sys.argv[3] != "0") if len(sys.argv) > 3 else True
stream = int(sys.argv[4]) if len(sys.argv) > 4 else None
N_CONNS = int(os.environ.get("N_CONNS", "1"))
WAVES = int(os.environ.get("WAVES", "0"))
a, c = FusedPop(2 * G, group_size=G), FusedPop(2 * G, group_size=G)
c.unroll = unroll or c.unroll
c.count_received = count
c.waves_per_eu = WAVES
c.stream = stream or c.stream
c.connect([a] * N_CONNS, [random_topology(2 * G, 2 * G, G, delay=0.2, k=8, seed=1 + i, weight=True) for i in range(N_CONNS)],
          n_slots=8)
c.run(1.0, 1.1, torch.full((2 * G,), math.inf))
torch.cuda.synchronize()
k = [k for cache in fused_kernel.device_caches.values() for k in cache[0].values()][-1]
asm = k.asm["amdgcn"]
print(f"fused G={G} conns={N_CONNS} waves_per_eu={WAVES} U={c.unroll} counters={int(count)} "
      f"stream={c.stream}: vgpr {k.n_regs}, spills {k.n_spills}, LDS bytes {k.metadata.shared}")
lines = [l.split(";")[0].strip() for l in asm.split("\n")]
labels = {l[:-1]: i for i, l in enumerate(lines) if re.match(r"^\.LBB\w+:$", l)}
print("   loop head        instrs barrier  ds  loads waitcnt valu  max-vgpr")
for i, l in enumerate(lines):
    m = re.match(r"s_cbranch_\w+\s+(\.LBB\w+)|s_branch\s+(\.LBB\w+)", l)
    if m and (m.group(1) or m.group(2)) in labels and labels[m.group(1) or m.group(2)] < i:
        body = [x for x in lines[labels[m.group(1) or m.group(2)]:i + 1] if x and not x.startswith(".")]
        n = lambda p: sum(1 for x in body if re.match(p, x))
        regs = [int(x) for line in body for x in re.findall(r"\bv(\d+)\b", line)]
        regs += [int(b_) for line in body for _a, b_ in re.findall(r"\bv\[(\d+):(\d+)\]", line)]
        if len(body) > 30:
            hint = "atomic" if any("global_atomic" in x for x in body) else (
                "ds_load_b128" if any("ds_load_b128" in x for x in body) else "")
            print("   %-16s %6d %7d %3d %6d %7d %4d  %8d  %s" % (m.group(1) or m.group(2), len(body), n(r"s_barrier"),
                                                   n(r"ds_"), n(r"global_load|buffer_load|s_load"), n(r"s_waitcnt"),
                                                   n(r"v_"), max(regs, default=-1) + 1, hint))
