"""What torch's graph capture can do on this platform beyond plain capture: a device-side while loop
(a conditional node: torch.cuda.CUDAGraph.begin_capture_to_while_node -- what torch.while_loop's graph
capture uses), with a Triton kernel in its body, with a fork/join across streams in its body; and a
graph's raw handle.

    python tools/graph_while_probe.py [--rdna2]
"""

import os
import sys
import time
import traceback

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch
import triton
import triton.language as tl

N = 50


@triton.jit
def bump(x, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < n
    tl.store(x + i, tl.load(x + i, mask=m) + 1, mask=m)


def probe(name, body, streams=0):
    """A graph: counter = 0; while counter < N: body(); counter += 1 -- replayed once, x checked."""
    x = torch.zeros(64, dtype=torch.int32, device="cuda")
    counter = torch.zeros((), dtype=torch.int32, device="cuda")
    side = [torch.cuda.Stream() for _ in range(streams)]
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        body(x, side, torch.cuda.current_stream())  # (warm up: compiled before the capture)
        torch.cuda.synchronize()
        x.zero_()
        with torch.cuda.stream(s):
            g.capture_begin(capture_error_mode="thread_local")
            counter.zero_()
            pred = counter < N
            g.begin_capture_to_while_node(pred)
            body(x, side, s)
            counter.add_(1)
            g.set_conditional_handle_for_current_node(counter < N)
            g.end_capture_to_conditional_node()
            g.capture_end()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 10 / N * 1e6
        ok = int(x[0]) == 11 * N
        print(f"  {name}: {'works' if ok else 'WRONG'} (x = {int(x[0])}, expected {11 * N}), "
              f"{dt:.1f} us per iteration")
    except Exception as e:
        try:  # (the failed capture closed: the next probe's starts clean)
            with torch.cuda.stream(s):
                g.capture_end()
        except Exception:
            pass
        torch.cuda.synchronize()
        print(f"  {name}: FAILS -- {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        if "-v" in sys.argv:
            traceback.print_exc()


def torch_op(x, side, s):
    x.add_(1)


def triton_kernel(x, side, s):
    bump[(1,)](x, 64, BLOCK=64)


def fork_join(x, side, s):
    """As Executor.concurrently (naer2/sim/executor.py): each side stream after the body's stream, the body's stream after them."""
    ready = torch.cuda.Event()
    ready.record(s)
    done = []
    for k, st in enumerate(side):
        st.wait_event(ready)
        with torch.cuda.stream(st):
            if k == 0:
                bump[(1,)](x, 64, BLOCK=64)
            else:
                torch.empty(1, device="cuda")  # (another branch: nothing to x)
        e = torch.cuda.Event()
        e.record(st)
        done.append(e)
    for e in done:
        s.wait_event(e)


print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__} "
      f"({'HIP ' + torch.version.hip if torch.version.hip else 'CUDA ' + str(torch.version.cuda)})")
probe("while node, a torch op in its body", torch_op)
probe("while node, a Triton kernel in its body", triton_kernel)
probe("while node, fork/join over 3 streams in its body", fork_join, streams=3)
g = torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(g):
    torch.zeros(1, device="cuda").add_(1)
print(f"  raw graph handle (keep_graph=True): {hex(g.raw_cuda_graph())}")
