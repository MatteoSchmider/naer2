"""PD14 through captured windows (run_graph, the barrier between rounds), the replays' GPU time per round
(events around the graph launch alone) and the wall clock -- for a kernel configuration: group size, warps
per program, NVIDIA's register cap. One configuration per process.

    python tools/pd14_graph_gpu.py [--G=256] [--Gpop=I_4:128,...] [--Wpop=I_4:8,...] [--stream=256] [--fuse] [--warps=N] [--maxnreg=N] [--window=100] [--warmup=1000] [--trace | --slowest [--serial] [--status-device]] [--trace_rounds=40] [--rdna2]

Also each population's compiled registers per thread and spills (NVIDIA), so a register cap's cost shows.
"""

import os
import sys
import time

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch

import naer2.graph as GR
import naer2.runtime as R
import pd14_network as P

opts = {a.split("=")[0][2:]: int(a.split("=")[1]) for a in sys.argv
        if a.startswith("--") and "=" in a and not a.startswith(("--Gpop=", "--Wpop="))}
G, WINDOW, WARMUP = opts.get("G", 256), opts.get("window", 100), opts.get("warmup", 1000)
WARPS, MAXNREG = opts.get("warps"), opts.get("maxnreg")

compiled = []
orig_direct = R._direct_launch


def recording(kernel, n_programs, consts):
    compiled.append(kernel)
    return orig_direct(kernel, n_programs, consts)


R._direct_launch = recording

obj = P._network(G)
net = obj[0]
net.graph_barrier = True
FUSE = "--fuse" in sys.argv  # (fused rounds, naer2/fusion.py: set after the first rounds, before any window)
pops = P._runtimes("network", obj)
P.group_sizes(pops, next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--Gpop=")), None))
P._advance("network", obj, 0)
for p in pops.values():
    rt = P._runtime("network", p)
    rt.max_warps, rt.maxnreg = WARPS, MAXNREG
if "stream" in opts:  # the walk's stream block (fused.py WALK), every runtime
    for p in pops.values():
        P._runtime("network", p).stream_block = opts["stream"]
        P._runtime("network", p)._launchers = {}
WPOP = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--Wpop=")), "")
for item in filter(None, WPOP.split(",")):  # per-population warps: "I_4:8,E_5:8"
    name, w = item.split(":")
    P._runtime("network", pops[name]).max_warps = int(w)
for r in range(1, WARMUP):
    P._advance("network", obj, r)
    if r % 500 == 0:
        torch.cuda.synchronize()

def status_on_device():
    """(probe) The runs' status into device memory, not pinned host memory (fused.py: _run_status): the host
    copies it back before it reads it (waiting for the device)."""
    orig_landed = R.PopulationRuntime._landed

    def landed(self, run_id):
        if hasattr(self, "_status_copy"):
            self._status_copy.copy_(self._status_host)  # (device -> host, waiting for the device)
        return orig_landed(self, run_id)

    R.PopulationRuntime._landed = landed
    torch.cuda.synchronize()
    for p in pops.values():
        rt = P._runtime("network", p)
        rt._judge_pending(None, wait=True)
        rt._status_copy = rt._status_host.clone().pin_memory()
        rt._status_host = rt._status_host.to(net.device)
        rt._status_np = rt._status_copy.numpy()


events = []
orig_replay = GR.WindowGraph.replay


def replay(self, schedule):
    g = self.graphs[self._turn]
    real = g.replay

    def timed():
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        real()
        e.record()
        events.append((s, e))

    g.replay = timed
    try:
        orig_replay(self, schedule)
    finally:
        g.replay = real


GR.WindowGraph.replay = replay
t_fuse = time.perf_counter()
net.fuse_rounds = FUSE
for _ in range(3):
    net.run_graph(max_launches=WINDOW)
torch.cuda.synchronize()
if FUSE:
    k = net._fused.compiled
    print(f"  fused: {net._fused.n_programs} programs; first windows (its compile included) "
          f"{time.perf_counter() - t_fuse:.1f} s; registers {getattr(k, 'n_regs', '?')}, spills "
          f"{getattr(k, 'n_spills', '?')}, shared memory {k.metadata.shared} B", flush=True)
if "--status-device" in sys.argv:  # (after the windows' status slots are allocated; the graphs captured anew)
    status_on_device()
    net._graphs = {}
    net.run_graph(max_launches=WINDOW)
    torch.cuda.synchronize()
events.clear()
t = time.perf_counter()
n = 0
while n < 10 * WINDOW:
    n += net.run_graph(max_launches=WINDOW)
torch.cuda.synchronize()
wall = (time.perf_counter() - t) / n * 1e6
gpu = sorted(s.elapsed_time(e) * 1e3 / WINDOW for s, e in events)
fused = [k for k in compiled if k.metadata.name.startswith("fused")][-len(pops):]  # (approximate: last compiled)
regs = sorted({getattr(k, "n_regs", 0) for k in fused})
spills = max(getattr(k, "n_spills", 0) for k in fused)
nw = sorted({k.metadata.num_warps for k in fused})
print(f"{torch.cuda.get_device_name(0)} G={G} warps={nw} maxnreg={MAXNREG}: graph on the GPU "
      f"{gpu[len(gpu) // 2]:.1f} us/round (min {gpu[0]:.1f}), wall {wall:.0f} us/round; registers {regs}, "
      f"spills {spills}; captures {net._captures}")

if "--trace" in sys.argv:  # one stepped round against one replayed round, traced inside the kernels
    import statistics
    import subprocess

    try:
        MHZ = float(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.split()[0])
    except Exception:
        MHZ = None
    for p in pops.values():
        P._runtime("network", p).set_trace(cap=4096, level=1)
    net._graphs = {}  # (the traced kernels: captured anew)

    def sample():
        torch.cuda.synchronize()
        progs = {name: P._runtime("network", p).traced() for name, p in pops.items()}
        t0 = min(g["start"] for ps in progs.values() for g in ps if g["start"])
        out = {}
        for name, ps in progs.items():
            ps = [g for g in ps if g["start"] and g["events"]]
            if not ps:
                continue
            ph = {"pull": [], "delivery": [], "update+send": [], "emission": []}
            for g in ps:
                at = {}
                for tag, _, cyc in g["events"]:
                    at.setdefault(tag, cyc)
                if {"start", "state", "delivered", "sent", "end"} <= set(at):
                    ph["pull"].append(at.get("pulled", at["state"]) - at["start"])
                    ph["delivery"].append(at["delivered"] - at["state"])
                    ph["update+send"].append(at["sent"] - at["delivered"])
                    ph["emission"].append(at["end"] - at["sent"])
            out[name] = dict(
                dur=statistics.median((g["end"] - g["start"]) / 100 for g in ps),
                worst=max((g["end"] - g["start"]) / 100 for g in ps),
                first=min((g["start"] - t0) / 100 for g in ps), last=max((g["start"] - t0) / 100 for g in ps),
                end=max((g["end"] - t0) / 100 for g in ps),
                ph={k: statistics.median(v) / MHZ if MHZ and v else float("nan") for k, v in ph.items()})
        return out

    ROUNDS_T = opts.get("trace_rounds", 40)

    def merged(samples):
        """Per population over the sampled rounds: medians of each round's values; and the rounds' critical
        paths (the last program's end after the round's first start)."""
        out = {}
        for name in samples[0][0]:
            per = [s[name] for s, _ in samples if name in s]
            out[name] = dict(dur=statistics.median(x["dur"] for x in per), worst=max(x["worst"] for x in per),
                             first=statistics.median(x["first"] for x in per),
                             last=statistics.median(x["last"] for x in per),
                             end=statistics.median(x["end"] for x in per),
                             ph={k: statistics.median(x["ph"][k] for x in per) for k in per[0]["ph"]})
        crit = sorted(c for _, c in samples)
        return out, crit

    def rounds(advance):
        got = []
        for _ in range(ROUNDS_T):
            advance()
            s = sample()
            got.append((s, max(x["end"] for x in s.values())))
        return merged(got)

    for _ in range(5):
        P._advance("network", obj, 0)
    stepped, crit_s = rounds(lambda: P._advance("network", obj, 0))
    for _ in range(3):
        net.run_graph(max_launches=1)
    graphed, crit_g = rounds(lambda: net.run_graph(max_launches=1))
    for label, c in (("stepped", crit_s), ("one-round graphs", crit_g)):
        print(f"  {label}: a round's last end after its first start, over {len(c)} rounds: median "
              f"{statistics.median(c):.1f} us, p90 {c[int(len(c) * .9)]:.1f}, max {c[-1]:.1f}")
    print(f"  traced (SM clock {MHZ} MHz), per population, medians over the rounds: program duration "
          f"median/worst, programs' starts (first..last after the round's first), last end; phases (us)")
    for name in sorted(stepped, key=lambda n: -graphed.get(n, stepped[n])["end"]):
        for label, s in (("step ", stepped[name]), ("graph", graphed.get(name))):
            if s is None:
                continue
            print(f"  {name:5s} {label} {s['dur']:6.1f}/{s['worst']:6.1f}  starts {s['first']:6.1f}..{s['last']:6.1f}"
                  f"  end {s['end']:6.1f}  " + " ".join(f"{k} {v:5.1f}" for k, v in s["ph"].items()))

if "--slowest" in sys.argv:  # each round's slowest programs, level-2 traced: where in the program the time went
    import collections
    import statistics
    import subprocess

    try:
        MHZ = float(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.split()[0])
    except Exception:
        MHZ = 1000.0
    for p in pops.values():
        P._runtime("network", p).set_trace(cap=8192, level=2)
    net._graphs = {}
    if "--serial" in sys.argv:  # every round's kernels one after another on one stream: none alongside another
        def _serial(work):
            s = torch.cuda.current_stream()
            for _, fn in work:
                fn(s)
        net._concurrently = _serial
    for _ in range(3):  # (the traced variants compiled: not inside a capture)
        P._advance("network", obj, 0)
    for _ in range(3):
        net.run_graph(max_launches=1)
    torch.cuda.synchronize()
    progs_all, typical = [], collections.defaultdict(list)
    stalls = []  # (round, wall time of a gap > 10 us after the round's first start, its length, pop, group)
    for r in range(opts.get("trace_rounds", 40)):
        net.run_graph(max_launches=1)
        torch.cuda.synchronize()
        traced = {name: P._runtime("network", p).traced() for name, p in pops.items()}
        t0 = min(g["start"] for ps in traced.values() for g in ps if g["start"])
        for name, ps in traced.items():
            for gi, g in enumerate(ps):
                if not g["start"] or not g["events"]:
                    continue
                dur = (g["end"] - g["start"]) / 100
                typical[name].append(dur)
                progs_all.append((dur, name, gi, g["events"], g["dropped"]))
                ev = g["events"]
                for i in range(len(ev) - 1):
                    d = (ev[i + 1][2] - ev[i][2]) / MHZ
                    if d > 10:
                        stalls.append((r, (g["start"] - t0) / 100 + ev[i][2] / MHZ, d, name, gi))
    progs_all.sort(key=lambda x: -x[0])
    print(f"  the slowest programs of {opts.get('trace_rounds', 40)} one-round graphs (level 2; population median "
          f"in brackets): duration, events per tag, the largest gaps between consecutive events (us)")
    for dur, name, gi, ev, dropped in progs_all[:16]:
        counts = collections.Counter(t for t, _, _ in ev)
        gaps = sorted(((ev[i + 1][2] - ev[i][2]) / MHZ, ev[i][0], ev[i + 1][0], i) for i in range(len(ev) - 1))[::-1]
        print(f"  {name:5s} g{gi:<3d} {dur:5.1f} [{statistics.median(typical[name]):4.1f}] "
              + " ".join(f"{t}:{c}" for t, c in sorted(counts.items()) if t not in ("start", "end"))
              + (f" dropped {dropped}" if dropped else "")
              + " | " + ", ".join(f"{a}->{b}@{i} {d:.1f}" for d, a, b, i in gaps[:3]))
    by_round = collections.defaultdict(list)
    for r, at, d, name, gi in stalls:
        by_round[r].append((at, d, name))
    print(f"  stalls (a gap > 10 us inside a program): {len(stalls)} in {len(by_round)} of "
          f"{opts.get('trace_rounds', 40)} rounds; per such round: when they begin (us after the round's start), "
          f"how long, which populations")
    for r in sorted(by_round)[:12]:
        xs = by_round[r]
        ats = [a for a, _, _ in xs]
        print(f"    round {r:2d}: {len(xs):3d} stalls, begin {min(ats):5.1f}..{max(ats):5.1f}, "
              f"length {statistics.median(d for _, d, _ in xs):4.1f} (max {max(d for _, d, _ in xs):4.1f}), "
              + " ".join(f"{n}:{c}" for n, c in collections.Counter(n for _, _, n in xs).most_common()))
