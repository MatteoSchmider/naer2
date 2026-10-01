"""
Kernel semantics check: run the generated update kernels on Triton's CPU
interpreter and diff them against a strictly serial plain-Python reference.

The reference applies each spike's deliveries one link at a time, in spike
order (edge forward -> receive -> send per delivery, then the final decay and
fire check), which is the behavior the tiled kernel must reproduce. Sizes are
deliberately awkward — 10 neurons over GROUP_SIZE=4 (3 programs, ragged tail),
10 links over LINK_BLOCK=4 (3 tiles, ragged tail) — so program ownership and
tile edges are exercised. The interpreter runs programs sequentially, so this
checks semantics, not races; it does catch a missing ownership mask, since
every program would then re-apply plastic weight updates.

Run standalone (TRITON_INTERPRET must be set before triton is first imported):
    python -m test.semantics [B C D]
"""
import os, sys, ast, importlib.util, pathlib, tempfile, random, math
os.environ["TRITON_INTERPRET"] = "1"
import torch, triton
import naer2 as n2
from test.networks.coverage import build

G, LB, SB, CAP = 4, 4, 2, 64          # deliberately awkward: 10 neurons / G=4, 10 links / LB=4

def load_kernel(module, label):
    src = "import math\nimport triton\nimport triton.language as tl\n\n" + ast.unparse(module) + "\n"
    f = pathlib.Path(tempfile.mkdtemp()) / f"k_{label}.py"; f.write_text(src)
    spec = importlib.util.spec_from_file_location(f"k_{label}", f); m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m; spec.loader.exec_module(m); return m.update_kernel

def run(name, pops, seed):
    torch.manual_seed(seed); random.seed(seed)
    pop = pops[name]; N = 10; T = 1.0
    fn = load_kernel(pop.compile(), f"{name}{seed}")
    pot0 = torch.rand(N) * 12      # spans both send cases (> threshold, > 2 * threshold)
    args = {"potential_ptr": pot0.clone(), "n_neurons": N}
    # spike buffer, sorted by arrival time
    n_sp = 9
    spikes = sorted((random.random() * 0.9, random.randrange(N), random.randrange(len(pop.incoming))) for _ in range(n_sp))
    args.update(spike_time_ptr=torch.tensor([s[0] for s in spikes], dtype=torch.float32),
                spike_src_ptr=torch.tensor([s[1] for s in spikes], dtype=torch.int32),
                spike_conn_ptr=torch.tensor([s[2] for s in spikes], dtype=torch.int32), n_spikes=n_sp)
    weights, srcpot, blocked = {}, {}, {}
    for i, conn in enumerate(pop.incoming):
        topo = conn.fabric.topology
        bt = blocked[i] = topo.blocked(G, N)     # block-major layout: program pid owns targets [pid*G, (pid+1)*G)
        args[f"conn{i}_src_offsets_ptr"] = bt.offsets.flatten().contiguous()
        args[f"conn{i}_target_indices_ptr"] = bt.target_local
        w = conn.fabric.state.get("weight", torch.ones(topo.target_indices.shape[0])).clone().float()
        weights[i] = w.clone(); args[f"conn{i}_weight_ptr"] = bt.to_blocked(w).contiguous()
        if f"conn{i}_source_potential_ptr" in fn.arg_names:
            srcpot[i] = torch.rand(N) * 6; args[f"conn{i}_source_potential_ptr"] = srcpot[i]
    P = triton.cdiv(N, G)
    cap_total = N * 100
    args.update(t=T, local_spike_node_ptr=torch.zeros(P * CAP, dtype=torch.int32), local_spike_time_ptr=torch.zeros(P * CAP),
                out_spike_node_ptr=torch.full((cap_total,), -1, dtype=torch.int32), out_spike_time_ptr=torch.zeros(cap_total),
                out_spike_count_ptr=torch.zeros(1, dtype=torch.int32))
    args = {k: v for k, v in args.items() if k in fn.arg_names}
    w_blocked = {i: args[f"conn{i}_weight_ptr"] for i in weights if f"conn{i}_weight_ptr" in args}
    fn[(P,)](**args, GROUP_SIZE=G, SPIKE_BLOCK=SB, LINK_BLOCK=LB, LOCAL_SPIKE_CAP=CAP)

    # ---- reference: strictly serial, one delivery at a time, in spike order ----
    w_kernel = {i: blocked[i].from_blocked(w_blocked[i]) for i in w_blocked}   # back to canonical link order
    pot = pot0.clone(); w = {i: weights[i].clone() for i in weights}; out = []
    def send(n, tm):
        # relay.send: far past threshold -> an extra spike one delay later; else a plain spike
        if pot[n] > 10: out.append((n, T)); out.append((n, tm + 1.0)); pot[n] = 0
        elif pot[n] > 5: out.append((n, T)); pot[n] = 0
    for (tm, src, ci) in spikes:
        conn = pop.incoming[ci]; topo = conn.fabric.topology
        plastic = type(conn.fabric.instance).__name__ == "plastic_synapse"
        for li in range(int(topo.source_offsets[src]), int(topo.source_offsets[src + 1])):
            tg = int(topo.target_indices[li])
            if plastic: w[ci][li] += 0.1
            if conn._target_port == "positive":
                pot[tg] += w[ci][li]
            else:
                pot[tg] -= w[ci][li]; pot[tg] += srcpot[ci][src] * 0.1
            send(tg, tm)
    pot *= 0.9
    for n in range(N): send(n, T)

    cnt = int(args["out_spike_count_ptr"][0])
    got = sorted((int(args["out_spike_node_ptr"][k]), round(float(args["out_spike_time_ptr"][k]), 4)) for k in range(cnt))
    exp = sorted((n, round(float(tt), 4)) for n, tt in out)
    ok_pot = torch.allclose(args["potential_ptr"], pot, atol=1e-4)
    ok_w = all(torch.allclose(w_kernel[i], w[i], atol=1e-5) for i in w_kernel)
    ok_sp = got == exp
    print(f"{name} seed={seed}: potential {'OK' if ok_pot else 'MISMATCH'} | weights {'OK' if ok_w else 'MISMATCH'} | "
          f"spikes {'OK' if ok_sp else 'MISMATCH'} ({cnt} emitted, {len(exp)} expected)")
    if not (ok_pot and ok_w and ok_sp):
        print("  kernel pot:", args["potential_ptr"].tolist()); print("  ref    pot:", pot.tolist())
        print("  kernel spikes:", got[:8]); print("  ref    spikes:", exp[:8])
        for i in w_kernel:
            if not torch.allclose(w_kernel[i], w[i], atol=1e-5): print(f"  conn{i} weights kernel[:5]={w_kernel[i][:5].tolist()} ref[:5]={w[i][:5].tolist()}")
    return ok_pot and ok_w and ok_sp


@n2.node(send="send", update="update", receive=None)
class probe:
    """Exercises math lowering (incl. a constant-folded call) and a top-level n2.spike()."""

    def __init__(self, rate: float):
        self.rate = rate
        self.value: float = 0.0

    def update(self, time: n2.time):
        self.value = (
            self.value * math.exp(-self.rate)
            + math.sin(self.value)
            + math.log(1.0 + abs(self.value))
            + math.sqrt(self.value) * math.cos(self.value)
            + max(self.value, 0.5)
        )

    def send(self, time: n2.time):
        n2.spike()


def run_probe(seed):
    torch.manual_seed(seed)
    N, T = 10, 1.0
    P = probe(rate=0.5) * N
    v0 = torch.rand(N) * 3
    P.value = v0.clone()
    fn = load_kernel(P.compile(), f"probe{seed}")
    nprog = triton.cdiv(N, G)
    args = dict(value_ptr=v0.clone(), n_neurons=N, spike_time_ptr=torch.zeros(1), spike_src_ptr=torch.zeros(1, dtype=torch.int32),
                spike_conn_ptr=torch.zeros(1, dtype=torch.int32), n_spikes=0, t=T,
                local_spike_node_ptr=torch.zeros(nprog * CAP, dtype=torch.int32), local_spike_time_ptr=torch.zeros(nprog * CAP),
                out_spike_node_ptr=torch.full((N * 100,), -1, dtype=torch.int32), out_spike_time_ptr=torch.zeros(N * 100),
                out_spike_count_ptr=torch.zeros(1, dtype=torch.int32))
    fn[(nprog,)](**{k: v for k, v in args.items() if k in fn.arg_names}, GROUP_SIZE=G, SPIKE_BLOCK=SB, LINK_BLOCK=LB, LOCAL_SPIKE_CAP=CAP)
    ref = v0 * math.exp(-0.5) + torch.sin(v0) + torch.log(1.0 + v0.abs()) + torch.sqrt(v0) * torch.cos(v0) + torch.clamp(v0, min=0.5)
    cnt = int(args["out_spike_count_ptr"][0])
    got = sorted((int(args["out_spike_node_ptr"][k]), round(float(args["out_spike_time_ptr"][k]), 4)) for k in range(cnt))
    ok_v = torch.allclose(args["value_ptr"], ref, atol=1e-5)
    ok_s = got == [(n, T) for n in range(N)]      # a top-level spike fires every neuron
    print(f"probe seed={seed}: math update {'OK' if ok_v else 'MISMATCH'} | top-level spike -> all {N} neurons {'OK' if ok_s else 'MISMATCH'} ({cnt} emitted)")
    return ok_v and ok_s


if __name__ == "__main__":
    _, pops = build()
    names = sys.argv[1:] or ["B", "C", "D", "probe"]
    res = [run(n, pops, s) for n in names if n != "probe" for s in range(3)]
    if "probe" in names:
        res += [run_probe(s) for s in range(3)]
    print("ALL MATCH" if all(res) else "SOME MISMATCH")
