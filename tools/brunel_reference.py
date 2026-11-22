"""An independent time-driven reference for Brunel (2000) model A at Brunel's size (N = 12500, C_E = 1000):
plain torch, a sparse weight matrix, a fixed step dt (the external drive as Poisson counts per step) -- the
grid-based simulation that approaches the exact event-driven one (test/networks/Brunel) as dt -> 0.

    python tools/brunel_reference.py [dt=0.05]

dt 0.05, 400 ms (780M): E 37.60 Hz, I 37.78 Hz, CV(ISI) 0.42, activity std/mean 0.51-0.53; naer2 (tools/
brunel_run.py, the same model): 37.67 / 37.92 Hz, CV 0.44, 0.59-0.60.
"""
import sys, math, time, torch
dt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
T = 400.0
dev = "cuda"
g, eta, J, D, tau, theta, Vr, tref = 5.0, 2.0, 0.1, 1.5, 20.0, 20.0, 10.0, 2.0
NE, NI = 10000, 2500
N = NE + NI
CE, CI = 1000, 250
gen = torch.Generator().manual_seed(1)
# fixed in-degree: each target CE sources from E, CI from I (with replacement)
tgt = torch.arange(N).repeat_interleave(CE + CI)
src = torch.cat([torch.randint(0, NE, (N, CE), generator=gen), NE + torch.randint(0, NI, (N, CI), generator=gen)], 1).flatten()
w = torch.where(src < NE, J, -g * J)
W = torch.sparse_coo_tensor(torch.stack([tgt, src]), w, (N, N)).coalesce().to_sparse_csr().to(dev)
nu_ext = eta * theta / (J * CE * tau)  # per ms, per external source
lam = CE * nu_ext * dt
V = torch.rand(N, generator=gen).to(dev) * theta
ref_until = torch.zeros(N, device=dev)
dsteps = round(D / dt)
buf = torch.zeros(dsteps, N, device=dev)  # recurrent input arriving per step
decay = math.exp(-dt / tau)
spk_t, spk_i = [], []
t0 = time.time()
n_steps = round(T / dt)
for n in range(n_steps):
    t = (n + 1) * dt
    inp = buf[n % dsteps] + J * torch.poisson(torch.full((N,), lam, device=dev))
    buf[n % dsteps] = 0
    active = t > ref_until
    V = torch.where(active, V * decay + inp, V)
    fired = active & (V >= theta)
    V = torch.where(fired, torch.full_like(V, Vr), V)
    ref_until = torch.where(fired, t + tref, ref_until)
    idx = fired.nonzero().flatten()
    if len(idx):
        s = torch.zeros(N, device=dev)
        s[idx] = 1.0
        buf[(n + dsteps) % dsteps] += W @ s  # arrives D later
        spk_t.append(torch.full((len(idx),), t, device=dev)); spk_i.append(idx)
torch.cuda.synchronize()
times = torch.cat(spk_t).cpu(); nodes = torch.cat(spk_i).cpu()
print(f"dt {dt}: simulated in {time.time() - t0:.1f} s")
for name, lo, hi in (("E", 0, NE), ("I", NE, N)):
    m = (nodes >= lo) & (nodes < hi) & (times >= 50.0)
    rate = int(m.sum()) / (hi - lo) / ((T - 50.0) / 1000)
    cvs = []
    tt, nn = times[m], nodes[m]
    for j in range(lo, lo + 2000):
        tj = tt[nn == j]
        if len(tj) >= 4:
            isi = tj.diff(); cvs.append(float(isi.std() / isi.mean()))
    act = torch.histc(tt, bins=int(T - 50), min=50.0, max=T)
    print(f"{name}: rate {rate:.2f} Hz, CV(ISI) {sum(cvs)/len(cvs):.2f}, activity std/mean {float(act.std()/act.mean()):.2f}")
