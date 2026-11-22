"""Brunel (2000) model A (test/networks/Brunel): a run's rates, ISI irregularity and synchrony, and its speed.

    python tools/brunel_run.py [--N=12500] [--K=...] [--T=400] [--flight=0.25]

Steps of D (1.5 ms), every population's spikes collected from its history after each; after a 50 ms transient:
rates
(E, I, the Poisson sources against eta nu_thr), the mean CV of 2000 neurons' ISIs, the population activity's
std / mean in 1 ms bins and its strongest frequencies. Compare: tools/brunel_reference.py (time-driven).
"""

import os
import sys
import time

sys.path[:0] = [os.getcwd()]
import torch

from test.networks.Brunel import build, theory

opts = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv if a.startswith("--") and "=" in a}
N, T = int(opts.get("N", 12500)), float(opts.get("T", 400.0))
K = int(opts["K"]) if "K" in opts else None
FLIGHT = float(opts["flight"]) if "flight" in opts else None  # (the arenas' start, ring_in_flight)
D, T0 = 1.5, 50.0

t_build = time.time()
from test.networks.Brunel import description
desc = dict(description, **({"ring_in_flight": FLIGHT} if FLIGHT is not None else {}))
network, pops = build(N, K=K, desc=desc)
t_build = time.time() - t_build
print("theory", theory(N, K=K), {k: len(p.shape) for k, p in pops.items()})
spikes, read = {k: [] for k in pops}, {k: 0 for k in pops}
t_first = time.time()
network.step(D)  # (every runtime built, every kernel compiled)
torch.cuda.synchronize()
t_first = time.time() - t_first
links = sum(len(c.fabric.topology) for p in pops.values() for c in p.incoming)
print(f"build {t_build:.1f} s (topologies), first step {t_first:.1f} s (simulation topologies, runtimes, compile); "
      f"{links / 1e9:.3f} G links; GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
t0, t = time.time(), D
while t < T - 1e-9:
    network.step(D)
    t += D
    for k, p in pops.items():
        rt = p.runtime
        rt.settle()
        for run in range(read[k] + 1, rt.run_id + 1):
            spikes[k] += rt.history.spikes(run)
        read[k] = rt.run_id
torch.cuda.synchronize()
wall = time.time() - t0
for p in pops.values():
    p.check()
print(f"simulated {t - D:.1f} ms in {wall:.1f} s ({1000 * wall / (t - D):.1f} ms wall per ms); runs",
      {k: p.runtime.run_id for k, p in pops.items()})
for k in ("E", "I", "P") if t > T0 + 10 else ():  # (statistics only past the transient)
    n = len(pops[k].shape)
    times = torch.tensor([e for e, _ in spikes[k]])
    nodes = torch.tensor([j for _, j in spikes[k]])
    m = times >= T0
    print(f"{k}: rate {float(m.sum()) / n / ((t - T0) / 1000):.2f} Hz", end="")
    if k == "P":
        print()
        continue
    cvs = []
    for j in range(min(n, 2000)):
        tj = torch.sort(times[m & (nodes == j)]).values
        if len(tj) >= 4:
            isi = tj.diff()
            cvs.append(float(isi.std() / isi.mean()))
    act = torch.histc(times[m], bins=int(t - T0), min=T0, max=t)
    spec = torch.fft.rfft(act - act.mean()).abs()
    f = torch.fft.rfftfreq(len(act), d=1e-3)
    top = torch.argsort(spec, descending=True)[:3]
    print(f", CV(ISI) {sum(cvs) / max(len(cvs), 1):.2f}, activity std/mean {float(act.std() / act.mean()):.2f}, "
          f"peaks {[round(float(f[i])) for i in top]} Hz")
