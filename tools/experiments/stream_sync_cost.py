"""Host cost of the cross-stream synchronization a round needs (populations
on their own streams, a population's round r + 1 after its sources' round
r): per event operation through torch vs the HIP API directly (ctypes), and
per round for the fan-out/fan-in patterns, 8 streams. CPU microseconds, the
GPU idle (nothing launched), best of 5.

    python tools/experiments/stream_sync_cost.py
"""
import ctypes
import math
import time

import torch

N, S = 2000, 8


def per_call(fn, reps=5):
    best = math.inf
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            fn()
        best = min(best, (time.perf_counter() - t0) / N * 1e6)
        torch.cuda.synchronize()
    return best


torch.zeros(1, device="cuda")  # the HIP runtime torch loaded (its own copy): found in this process's maps
hip = ctypes.CDLL(next(line.split()[-1] for line in open("/proc/self/maps") if "libamdhip64" in line))
hip.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
hip.hipStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
main = torch.cuda.current_stream()
streams = [torch.cuda.Stream() for _ in range(S)]
events = [torch.cuda.Event() for _ in range(S + 1)]  # timing disabled
for e in events:
    e.record(main)
torch.cuda.synchronize()
h_main = main.cuda_stream
h_streams = [s.cuda_stream for s in streams]
h_events = [e.cuda_event for e in events]

print(f"{torch.cuda.get_device_name(0)}: host us")
print(f"  torch  Event.record          {per_call(lambda: events[0].record(streams[0])):6.1f}")
print(f"  torch  Stream.wait_event     {per_call(lambda: streams[1].wait_event(events[0])):6.1f}")
print(f"  torch  Stream.wait_stream    {per_call(lambda: streams[1].wait_stream(streams[0])):6.1f}")
print(f"  torch  with torch.cuda.stream(s): pass  {per_call(lambda: torch.cuda.stream(streams[0]).__enter__() or None):6.1f}")
print(f"  HIP    hipEventRecord        {per_call(lambda: hip.hipEventRecord(h_events[0], h_streams[0])):6.1f}")
print(f"  HIP    hipStreamWaitEvent    {per_call(lambda: hip.hipStreamWaitEvent(h_streams[1], h_events[0], 0)):6.1f}")
torch.cuda.set_stream(main)


def pattern_torch_now():  # tools/pd14_setup.py phase(): wait_stream both ways
    for s in streams:
        s.wait_stream(main)
    for s in streams:
        main.wait_stream(s)


def pattern_hub_hip():  # per round: S records + S waits (fan-in), 1 record + S waits (fan-out)
    for i in range(S):
        hip.hipEventRecord(h_events[i], h_streams[i])
    for i in range(S):
        hip.hipStreamWaitEvent(h_main, h_events[i], 0)
    hip.hipEventRecord(h_events[S], h_main)
    for i in range(S):
        hip.hipStreamWaitEvent(h_streams[i], h_events[S], 0)


def pattern_all_hip():  # no hub: every stream waits on every stream's event
    for i in range(S):
        hip.hipEventRecord(h_events[i], h_streams[i])
    for i in range(S):
        for j in range(S):
            if i != j:
                hip.hipStreamWaitEvent(h_streams[i], h_events[j], 0)


print(f"  per round, {S} streams: torch wait_stream both ways {per_call(pattern_torch_now):6.1f}, "
      f"HIP hub {per_call(pattern_hub_hip):6.1f}, HIP all-to-all {per_call(pattern_all_hip):6.1f}")
