"""Host cost of one kernel launch: Triton vs Gluon, narrow vs wide signatures
(the fused kernel's ~30 runtime arguments and ~30 constexprs), through the
JIT's dispatch (bind + specialize + cache key + lookup) vs the compiled
kernel's launcher directly (the floor: argument packing + hipModuleLaunch);
torch's own launch for scale; and the real FusedPop.run (small consumer,
8 connections). Per launch, CPU microseconds over N back-to-back launches
(tiny kernels, so the GPU keeps up and the host is what is measured).

    python tools/experiments/launch_cost.py [--rdna2]
"""
import math
import os
import sys
import tempfile
import time

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", ".."), os.path.join(os.path.dirname(__file__), "..")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch
import triton
import triton.language as tl
from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

N = 3000


@triton.jit
def t_narrow(x):
    tl.store(x, 1)


@gluon.jit
def g_narrow(x):
    gl.store(x, 1)


# wide: as the fused kernel -- 16 pointers, 10 ints (unspecialized), 4 floats, 30 constexprs
WIDE_ARGS = ", ".join([f"p{i}" for i in range(16)] + [f"n{i}" for i in range(10)] + [f"f{i}" for i in range(4)])
WIDE_CONST = ", ".join(f"C{i}: {{c}}.constexpr" for i in range(30))
WIDE_SRC = '''
@{deco}(do_not_specialize=[{ns}])
def {name}({args}, {consts}):
    {m}.store(p0, n0 + C0)
'''


def _wide(deco, m, name):
    src = WIDE_SRC.format(deco=deco, m=m, name=name, args=WIDE_ARGS, consts=WIDE_CONST.format(c=m),
                          ns=", ".join(f'"n{i}"' for i in range(10)))
    path = os.path.join(tempfile.mkdtemp(), f"_wide_{name}.py")
    with open(path, "w") as f:  # the JIT reads kernel source from a file
        f.write("import triton\nimport triton.language as tl\nfrom triton.experimental import gluon\n"
                "from triton.experimental.gluon import language as gl\n" + src)
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_wide_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


t_wide = _wide("triton.jit", "tl", "t_wide")
g_wide = _wide("gluon.jit", "gl", "g_wide")


def per_launch(fn, reps=5):
    best = math.inf
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        best = min(best, (t1 - t0) / N * 1e6)
    return best


def direct(kernel, grid, args):
    """The compiled kernel's launcher, bypassing the JIT dispatch."""
    dev = torch.cuda.current_device()
    stream = triton.runtime.driver.active.get_current_stream(dev)
    meta = kernel.launch_metadata(grid, stream, *args)
    run, fn, pm = kernel.run, kernel.function, kernel.packed_metadata
    enter, exit_ = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
    return lambda: run(grid[0], 1, 1, stream, fn, pm, meta, enter, exit_, *args)


x = torch.zeros(64, device="cuda", dtype=torch.int32)
ptrs = [torch.zeros(64, device="cuda", dtype=torch.int32) for _ in range(16)]
ints = [100003 + i for i in range(10)]
floats = [0.5 + i for i in range(4)]
consts = {f"C{i}": i + 1 for i in range(30)}
wide_args = ptrs + ints + floats

rows = []
for name, k, args, kw in (("triton, 1 arg", t_narrow, [x], {}), ("gluon, 1 arg", g_narrow, [x], {}),
                          ("triton, wide", t_wide, wide_args, consts), ("gluon, wide", g_wide, wide_args, consts)):
    kw_launch = dict(kw, num_warps=4) if k in (g_narrow, g_wide) else kw
    compiled = k[(1,)](*args, **kw_launch)
    jit_us = per_launch(lambda: k[(1,)](*args, **kw_launch))
    all_args = list(args) + [kw[f"C{i}"] for i in range(30)] if kw else list(args)
    direct_us = per_launch(direct(compiled, (1, 1, 1), all_args))
    rows.append((name, jit_us, direct_us))

y = torch.zeros(1, device="cuda")
torch_us = per_launch(lambda: y.add_(1))

# the real thing: a small consumer with 8 incoming connections, no spikes
from naer2.greenfield.fused import FusedPop, fused_kernel  # noqa: E402
from test.test_intake_gpu import _topology  # noqa: E402

G = 256
a, c = FusedPop(2 * G, group_size=G), FusedPop(2 * G, group_size=G)
c.connect([a] * 8, [_topology(2 * G, 2 * G, G, delay=0.2, k=8, seed=1 + i, weight=True) for i in range(8)], n_slots=8)
c.count_received = False
fire = torch.full((2 * G,), math.inf, device="cuda")
zeros = torch.zeros(2 * G, device="cuda")
t = [1.0]


def run_pop():
    c.run(t[0], t[0] + 0.1, fire, zeros, round_id=c.run_id + 1)
    t[0] += 0.1


run_pop()
pop_us = per_launch(run_pop)

print(f"{torch.cuda.get_device_name(0)}, triton {triton.__version__}: host us per launch (best of 5 x {N})")
print(f"  {'kernel':16s} {'JIT dispatch':>13s} {'launcher only':>14s} {'dispatch share':>15s}")
for name, j, d in rows:
    print(f"  {name:16s} {j:13.1f} {d:14.1f} {j - d:14.1f}")
print(f"  torch add_ (one tiny op): {torch_us:.1f}")
print(f"  FusedPop.run (8 connections, whole host path): {pop_us:.1f}")
