"""PD14 through Network.step against the harness (TODO.md, runtime
integration step 7): the real network (test/networks/PD14/net.py) run by
the library's scheduling -- Network.step, the populations' own runtimes --
and by the hand-built harness (tools/pd14_setup.py, model "pd14": the same
generated kernels, connected and launched by hand, one round per 0.1 ms).
Both from build() under the same seed, at the same group size.

    python tools/pd14_network.py record {network,harness} OUT.pt [--ms 800] [--G 256] [--harness-caps]
        [--block B] [--in-flight C]
    python tools/pd14_network.py compare A.pt B.pt
    python tools/pd14_network.py [--rdna2] time {network,harness} [--ms 800] [--G 256] [--rounds 200]
        [--harness-caps | --scaled-caps | --block B] [--in-flight C]

record: per round, every population's spikes ((e, node) per run; one
readback per run, so not timed). Run each path in its own process (two
copies of PD14 don't fit next to each other on the 780M).
compare: the round structure (each run's window), then per population the
spikes round by round -- identical up to the first difference (delivery
order within a slot is nondeterministic in the last bits, TODO.md, so
the trajectories part eventually) -- and the rates (Hz) over the whole run,
the thalamic volley's window and before it.
time: wall time per round, the path running freely (no readback) through
`--rounds` rounds after --ms of simulated time (default: into the volley).

Rings (the network path): the populations' default arenas (--block B
records per block, 0: per-cell runs; --in-flight C: Population.ring_in_flight),
or per-cell runs at the harness's capacities (--harness-caps: nothing is
lost) or those scaled to the group size (--scaled-caps).
"""

import argparse
import os
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]

import torch

RES = 0.1


def _network(G, harness_caps=False, scaled_caps=False, block=None, in_flight=None):
    """PD14 at group size G. Rings: the populations' default (per-group arenas, Population.ring_block /
    ring_in_flight; block / in_flight override), or per-cell runs at the harness's capacities (harness_caps)
    or those scaled to G (scaled_caps)."""
    from pd14_setup import PD14_MAIN_CAP
    from test.networks.PD14.net import build, description

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    net = build()
    names = dict(zip(description["populations"]["size"], description["populations"]["name"]))
    names[description["T"]["size"]] = "T"
    pops = {names[len(p.shape)]: p for p in net.populations}
    for name, p in pops.items():
        p.group_size = G
        if block is not None:
            p.ring_block = block or None  # 0: per-cell runs
        if in_flight is not None:
            p.ring_in_flight = in_flight
        if harness_caps and name in PD14_MAIN_CAP:  # the harness's ring capacities: no growth, no lost links
            p.ring_block, p.main_cap, p.side_cap = None, PD14_MAIN_CAP[name], 16
        elif scaled_caps and name in PD14_MAIN_CAP:  # the same per group neuron, at group size G (sized for 256)
            p.ring_block, p.main_cap, p.side_cap = None, -(-PD14_MAIN_CAP[name] * G // 256 // 16) * 16, 16
    return net, pops


def _harness(G):
    from pd14_setup import PD14

    return PD14(G, model="pd14", t0=0.0, count_received=False)


def _runtimes(path, obj):
    return obj[1] if path == "network" else obj.pops


def _advance(path, obj, r):
    """Round r of the path: the network's next step of one resolution, or the harness's round r."""
    if path == "network":
        obj[0].step(RES)
    else:
        obj.round(r)


def _runtime(path, p):
    return p.runtime if path == "network" else p


def record(path, out, ms, G, harness_caps=False, block=None, in_flight=None):
    obj = _network(G, harness_caps, block=block, in_flight=in_flight) if path == "network" else _harness(G)
    n_rounds = round(ms / RES)
    runs = {}  # name -> [(round, t_start, t_end, [(e, node)])]
    last = {}
    t0 = time.time()
    for r in range(n_rounds):
        _advance(path, obj, r)
        for name, p in _runtimes(path, obj).items():
            rt = _runtime(path, p)
            h = rt.history
            for run in range(last.get(name, 0) + 1, rt.run_id + 1):  # a network step may hold several rounds
                runs.setdefault(name, []).append(
                    (h.round_of(run), h.window_end(run), sorted(h.spikes(run), key=lambda s: s[1])))
            last[name] = rt.run_id
        if r % 1000 == 0:
            print(f"  {path}: round {r} ({r * RES:.0f} ms), {time.time() - t0:.0f} s", flush=True)
    for p in _runtimes(path, obj).values():
        _runtime(path, p).check()
    sizes = {name: _runtime(path, p).n for name, p in _runtimes(path, obj).items()}
    rounds = obj[0].round_id if path == "network" else n_rounds
    growths = {name: _runtime(path, p).growths for name, p in _runtimes(path, obj).items()}
    torch.save(dict(path=path, ms=ms, G=G, runs=runs, sizes=sizes, rounds=rounds, growths=growths), out)
    print(f"{path}: {rounds} rounds for {ms} ms, {sum(len(v) for v in runs.values())} runs -> {out}")


def _rate(runs, n, lo, hi):
    spikes = sum(len(s) for _, end, s in runs if lo < end <= hi + 1e-6)
    return spikes / n / ((hi - lo) * 1e-3)


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    print(f"{a['path']} ({a['rounds']} rounds) vs {b['path']} ({b['rounds']} rounds), {a['ms']} ms, G={a['G']}")
    for x in (a, b):
        grown = {n: g for n, g in x["growths"].items() if g}
        if grown:
            print(f"  {x['path']} ring growths: {grown}")
    ms = min(a["ms"], b["ms"])
    print(f"  {'pop':5s} {'runs':>11s} {'windows':>8s} {'identical up to':>16s} "
          f"{'rate all (Hz)':>17s} {'< 700 ms':>15s} {'700-750 ms':>17s}")
    for name in sorted(a["runs"]):
        ra, rb = a["runs"][name], b["runs"].get(name, [])
        windows = sum(abs(x[1] - y[1]) < 1e-4 for x, y in zip(ra, rb))
        first = next((i for i, (x, y) in enumerate(zip(ra, rb)) if [s[1] for s in x[2]] != [s[1] for s in y[2]]),
                     None)
        same = f"{ra[first][1]:.1f} ms" if first is not None else "the end"
        n = a["sizes"][name]
        rates = [(_rate(ra, n, lo, hi), _rate(rb, n, lo, hi)) for lo, hi in ((0, ms), (0, min(700, ms)),
                                                                                (700, min(750, ms)))]
        print(f"  {name:5s} {len(ra):5d}/{len(rb):<5d} {windows:8d} {same:>16s} "
              + " ".join(f"{x:7.2f}/{y:<7.2f}" if hi > lo else f"{'':15s}"
                         for (x, y), (lo, hi) in zip(rates, ((0, ms), (0, min(700, ms)), (700, min(750, ms))))))


def timed(path, ms, G, rounds, harness_caps=False, scaled_caps=False, block=None, in_flight=None):
    obj = _network(G, harness_caps, scaled_caps, block, in_flight) if path == "network" else _harness(G)
    n = round(ms / RES)
    t0 = time.time()
    for r in range(n):
        _advance(path, obj, r)
        if r % 500 == 0:  # bound the host's lead, so a ring growth isn't noticed only inside the timed rounds
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    for p in _runtimes(path, obj).values():
        _runtime(path, p).check()  # a pending growth now, not in the timed rounds
    for r in range(n, n + 5):  # and its recompile (the capacities are constexprs)
        _advance(path, obj, r)
    n += 5
    torch.cuda.synchronize()
    print(f"  {path}: {n} rounds to {ms} ms in {time.time() - t0:.1f} s (compiles included)", flush=True)
    t1 = time.perf_counter()
    for r in range(n, n + rounds):
        _advance(path, obj, r)
    torch.cuda.synchronize()
    per = (time.perf_counter() - t1) / rounds
    for p in _runtimes(path, obj).values():
        _runtime(path, p).check()
    print(f"{torch.cuda.get_device_name(0)} {path}: G={G} per round {per * 1e3:.2f} ms over {rounds} rounds from {ms} ms"
          f" (rings {sum(_runtime(path, p).ring.buf.numel() * 4 for p in _runtimes(path, obj).values()) / 1e9:.2f} GB,"
          f" device memory peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("path", choices=["network", "harness"])
    rec.add_argument("out")
    rec.add_argument("--ms", type=float, default=800.0)
    rec.add_argument("--G", type=int, default=256)
    rec.add_argument("--harness-caps", action="store_true", help="network: the harness's ring capacities")
    rec.add_argument("--block", type=int, default=None, help="network: ring arena block (0: per-cell runs)")
    rec.add_argument("--in-flight", type=float, default=None, help="network: ring arena share of incoming links")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    tm = sub.add_parser("time")
    tm.add_argument("path", choices=["network", "harness"])
    tm.add_argument("--ms", type=float, default=800.0)
    tm.add_argument("--G", type=int, default=256)
    tm.add_argument("--rounds", type=int, default=200)
    tm.add_argument("--harness-caps", action="store_true")
    tm.add_argument("--scaled-caps", action="store_true", help="network: the harness's capacities scaled by G / 256")
    tm.add_argument("--block", type=int, default=None, help="network: ring arena block (0: per-cell runs)")
    tm.add_argument("--in-flight", type=float, default=None, help="network: ring arena share of incoming links")
    ap.add_argument("--rdna2", action="store_true", help="the RX 6600 eGPU (see tools/rdna2.py for the env vars)")
    args = ap.parse_args()
    if args.rdna2:
        import rdna2  # noqa: F401
    if args.cmd == "record":
        record(args.path, args.out, args.ms, args.G, args.harness_caps, args.block, args.in_flight)
    elif args.cmd == "compare":
        compare(args.a, args.b)
    else:
        timed(args.path, args.ms, args.G, args.rounds, args.harness_caps, args.scaled_caps, args.block,
              args.in_flight)
