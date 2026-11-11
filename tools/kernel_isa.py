"""Compile the green-field update kernel for a small consumer and summarize
its ISA: registers, LDS, and per loop body the instruction mix (what one
iteration costs). args: G [backend triton|gluon] [unroll] [count 1|0] [entry_block stream]"""
import math
import os
import re
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..")]
import torch

import test.test_update_emission_gpu as T
from test.test_intake_gpu import _topology

G = int(sys.argv[1])
backend = sys.argv[2] if len(sys.argv) > 2 else "triton"
unroll = int(sys.argv[3]) if len(sys.argv) > 3 else 8
count = (sys.argv[4] != "0") if len(sys.argv) > 4 else True
WAVES = int(os.environ.get("WAVES", "0"))
chunk = [int(x) for x in sys.argv[5:7]] if len(sys.argv) > 6 else None
T.BACKEND = backend
a, c = T.Pop(2 * G, group_size=G), T.Pop(2 * G, group_size=G)
c.unroll = unroll
c.count_received = count
c.waves_per_eu = WAVES
if chunk:
    c.entry_block, c.stream = chunk
N_CONNS = int(os.environ.get("N_CONNS", "1"))
c.connect([a] * N_CONNS, [_topology(2 * G, 2 * G, G, delay=0.2, k=8, seed=1 + i, weight=True) for i in range(N_CONNS)],
          n_slots=8)
c.run(1.0, 1.1, torch.full((2 * G,), math.inf))
torch.cuda.synchronize()
if backend == "gluon":
    from test.update_gluon import update_kernel_gluon as kernel
else:
    kernel = T.update_kernel
k = [k for cache in kernel.device_caches.values() for k in cache[0].values()][-1]
asm = k.asm["amdgcn"]
print(f"{backend} G={G} conns={N_CONNS} waves_per_eu={WAVES} U={unroll} counters={int(count)} chunk/stream={c.entry_block}/{c.stream}: vgpr {k.n_regs}, spills {k.n_spills}, LDS bytes {k.metadata.shared}")
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
            print("   %-16s %6d %7d %3d %6d %7d %4d  %8d" % (m.group(1) or m.group(2), len(body), n(r"s_barrier"),
                                                   n(r"ds_"), n(r"global_load|buffer_load|s_load"), n(r"s_waitcnt"),
                                                   n(r"v_"), max(regs, default=-1) + 1))
