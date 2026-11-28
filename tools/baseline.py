"""Construction, steady-state and resizing costs of one network, the baseline refactors A and B are measured
against (PLAN.md, phase 1): time and memory, host and device, per phase -- totals, per link and per node.

    python tools/baseline.py pd14 [--T=1000] [--window=100] [--G=256]
    python tools/baseline.py brunel [--N=50000] [--K=...] [--T=200] [--window=100]

One network per process. Phases:
  build   the model (its populations, connections, canonical topologies: the host)
  first   the first step (simulation topologies onto the device, runtimes, every kernel compiled)
  window  the first captured window (its capture, the fused round's compile)
  steady  captured windows of --window rounds for --T ms (PD14's thalamic volley at 700 ms: its arena growths):
          per window its wall clock (synchronized) and the graph replays' GPU time; the windows with a resize or
          a capture apart -- their time, count, and what one costs
Memory: host resident set (current, and the phase's peak: /proc/self/clear_refs resets it per phase), device
(allocated now, the phase's peak: torch's allocator), and the device's at the end by kind (rings, histories,
simulation topologies with their directories, the rest).
"""

import os
import statistics as st
import sys
import time

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
import torch

import naer2.graph as GR
import naer2.runtime as R

NET = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "pd14"
opts = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv if a.startswith("--") and "=" in a}
WINDOW = int(opts.get("window", 100))
GiB, MiB = 2**30, 2**20


def host_mem() -> tuple[int, int]:
    """(resident now, peak since the last reset) in bytes, from /proc/self/status."""
    v = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS:", "VmHWM:")):
                k, n, _ = line.split()
                v[k[:-1]] = int(n) * 1024
    return v["VmRSS"], v["VmHWM"]


def reset_peaks() -> None:
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")  # (resets VmHWM to the current RSS)
    except OSError:
        pass
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


phases = []


def phase(name: str, seconds: float) -> None:
    torch.cuda.synchronize()
    rss, hwm = host_mem()
    phases.append(dict(name=name, s=seconds, rss=rss, host_peak=hwm, dev=torch.cuda.memory_allocated(),
                       dev_peak=torch.cuda.max_memory_allocated()))
    reset_peaks()


# the resizes and captures, timed where they happen (both on the host: a resize drains the device first)
events = {"resize": [], "capture": []}


def _timed(cls, name, kind):
    orig = getattr(cls, name)

    def f(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            torch.cuda.synchronize()
            events[kind].append(time.perf_counter() - t0)

    setattr(cls, name, f)


_timed(R._Launches, "resize", "resize")
_timed(R._Launches, "recover", "resize")
_timed(GR.WindowGraph, "__init__", "capture")
gpu = []  # the replays' GPU time, per window
_replay = GR.WindowGraph.replay


def replay(self, schedule):
    g = self.graphs[self._turn]
    real = g.replay

    def timed():
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        real()
        e.record()
        gpu.append((s, e))

    g.replay = timed
    try:
        _replay(self, schedule)
    finally:
        g.replay = real


GR.WindowGraph.replay = replay

# ---- the process before any model: interpreter, torch, the CUDA context (the per-link / per-node host numbers
# are above it)
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
reset_peaks()
phase("start", 0.0)

# ---- build
torch.manual_seed(0)
t0 = time.perf_counter()
if NET == "pd14":
    from pd14_network import _network

    G = int(opts.get("G", 256))
    net, pops = _network(G)
    dt, T = 0.1, float(opts.get("T", 1000.0))
elif NET == "brunel":
    from test.networks.Brunel import build

    N = int(opts.get("N", 50000))
    net, pops = build(N, K=int(opts["K"]) if "K" in opts else None)
    dt, T = 1.5, float(opts.get("T", 200.0))
else:
    sys.exit(f"unknown network {NET!r}: pd14 or brunel")
phase("build", time.perf_counter() - t0)
nodes = sum(len(p.shape) for p in net.populations)
links = sum(len(c.fabric.topology) for p in net.populations for c in p.incoming)

# ---- the first step, the first window
t0 = time.perf_counter()
net.step(dt)
phase("first step", time.perf_counter() - t0)
t0 = time.perf_counter()
net.run_graph(max_launches=WINDOW)
phase("first window", time.perf_counter() - t0)
events["resize"].clear()
events["capture"].clear()
gpu.clear()

# ---- steady state (and the resizes on the way): --T ms past the first window
t_sim0, rounds0 = net.t, net.round_id
T += net.t
windows = []
t_steady = time.perf_counter()
while net.t < T - 1e-9:
    n_rs, n_cap, n_gpu, r0 = len(events["resize"]), len(events["capture"]), len(gpu), net.round_id
    tw = time.perf_counter()
    n = net.run_graph(max_launches=WINDOW)
    torch.cuda.synchronize()
    wall = time.perf_counter() - tw
    if not n:
        break
    g = sum(s.elapsed_time(e) for s, e in gpu[n_gpu:]) / 1e3
    windows.append(dict(rounds=net.round_id - r0, wall=wall, gpu=g, resizes=len(events["resize"]) - n_rs,
                        captures=len(events["capture"]) - n_cap))
steady_s = time.perf_counter() - t_steady
phase("steady", steady_s)
for p in net.populations:
    p.check()
sim_ms, rounds = net.t - t_sim0, net.round_id - rounds0

# ---- the device's memory by kind, at the end -- what is held: an old ring buffer (resized away) counts as allocated
# until the allocator processes its stream events (Ring: _hold_until_done), and nothing allocates after the last
torch.cuda.synchronize()
torch.cuda.empty_cache()
dev_held = torch.cuda.memory_allocated()
rts = [p.runtime for p in net.populations]
rings = sum(rt.ring.buf.numel() * 4 for rt in rts)
hists = sum(rt.history.buf.numel() * 4 for rt in rts)
topos = sum(t.buf.numel() * 4 + t.directory()[0].numel() * t.directory()[0].element_size()
            for rt in rts for t in rt.plan.topologies)  # (the per-link attrs: views into the buffer)
state = sum(v.numel() * v.element_size() for rt in rts for v in rt.state.values())

# ---- report
print(f"{torch.cuda.get_device_name(0)}, {NET}: {nodes} nodes, {links} links ({links / max(nodes, 1):.0f} a node), "
      f"{len(net.populations)} populations")
print(f"{'phase':13s} {'time':>9s} {'host now':>10s} {'host peak':>10s} {'dev now':>9s} {'dev peak':>9s}"
      f"   per link: host peak, dev peak   per node: host peak, dev peak")
base = phases[0]["rss"]  # (the process before any model)
for p in phases:
    hp = p["host_peak"] - base
    print(f"{p['name']:13s} {p['s']:8.2f}s {p['rss'] / GiB:9.2f}G {p['host_peak'] / GiB:9.2f}G "
          f"{p['dev'] / GiB:8.2f}G {p['dev_peak'] / GiB:8.2f}G   {hp / links:8.1f} B {p['dev_peak'] / links:8.1f} B"
          f"   {hp / nodes / 1024:8.1f} KB {p['dev_peak'] / nodes / 1024:8.1f} KB")
print(f"(host per link / node: above the process before any model, {base / GiB:.2f} GiB)")
dev_end = dev_held
print(f"device held at the end: rings {rings / MiB:.0f} MiB ({rings / links:.2f} B a link), histories {hists / MiB:.0f} MiB, "
      f"topologies {topos / MiB:.0f} MiB ({topos / links:.2f} B a link), state {state / MiB:.1f} MiB, "
      f"the rest {(dev_end - rings - hists - topos - state) / MiB:.0f} MiB; total {dev_end / links:.2f} B a link, "
      f"{dev_end / nodes:.0f} B a node")
quiet = [w for w in windows if not w["resizes"] and not w["captures"]]
busy = [w for w in windows if w["resizes"] or w["captures"]]
per_round = lambda ws, k: st.median(w[k] / w["rounds"] for w in ws) * 1e6 if ws else float("nan")
print(f"steady: {sim_ms:.1f} ms simulated in {rounds} rounds, {len(windows)} windows of {WINDOW}; "
      f"wall {steady_s / sim_ms * 1e3:.2f} ms per simulated ms, {steady_s / rounds * 1e6:.1f} us per round overall")
print(f"  windows without a resize or capture ({len(quiet)}): median wall {per_round(quiet, 'wall'):.1f} us per round, "
      f"GPU {per_round(quiet, 'gpu'):.1f} us per round")
rs, cap = events["resize"], events["capture"]
print(f"  windows with them ({len(busy)}): {sum(w['wall'] for w in busy):.2f} s of {steady_s:.2f}; "
      f"{len(rs)} resizes ({sum(rs):.2f} s, median {st.median(rs) * 1e3 if rs else 0:.0f} ms, "
      f"max {max(rs, default=0) * 1e3:.0f} ms), {len(cap)} captures ({sum(cap):.2f} s, "
      f"median {st.median(cap) * 1e3 if cap else 0:.0f} ms)")
growths = [(p.name if hasattr(p, "name") else type(p.instance).__name__, g) for p in net.populations
           for g in p.runtime.growths]
print(f"  growths: {len(growths)}; rings at the end {rings / links:.2f} B a link")
