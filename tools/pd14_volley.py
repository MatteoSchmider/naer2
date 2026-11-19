"""PD14 through the thalamic volley on the current device (from the repo root: python tools/pd14_volley.py G
[--profile] [--keep-old] [--zero-growth] [--warps=N]; on a cloud instance: cloud/remote run 'python
tools/pd14_volley.py G ...'). Diagnostics: --keep-old never frees a ring's old buffer, --zero-growth
zero-fills grown rings, --warps=N overrides the device's warp cap.
Runs to 730 ms freely (no syncs), reports a fault if one happens; with --profile, the volley's first 200 rounds
(700.5-720.5 ms) under the profiler: per population, median and slowest kernel."""
import os
import statistics
import sys
import time

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tools")]
import torch
from torch.profiler import ProfilerActivity, profile

import pd14_network as P
from naer2 import runtime as R

G = int(sys.argv[1])
if "--keep-old" in sys.argv:  # never free a ring's old buffer (diagnostic)
    from naer2 import ring as RG
    KEEP_OLD = []
    _hold = RG._hold_until_done
    RG._hold_until_done = lambda buf: (KEEP_OLD.append(buf), _hold(buf))[1]
if "--zero-growth" in sys.argv:  # every ring buffer zero-filled, also when grown (diagnostic)
    from naer2 import ring as RG
    _alloc = RG.Ring._allocate
    RG.Ring._allocate = lambda self, main_cap, side_cap, device, zero_records=True: _alloc(
        self, main_cap, side_cap, device, zero_records=True)
for a in sys.argv:
    if a.startswith("--warps="):  # the warp cap (diagnostic): runtime._warp_cap overridden
        R._warp_cap = lambda device, _w=int(a.split("=")[1]): _w
BLK = []  # TEMP: staging's block ids past the group's arena (ring header words 19..24), read before resizes
_apply = R.PopulationRuntime._apply_wanted
def _apply_blk(self):
    if self._wanted is not None:
        w = self.ring.buf[19:32].tolist()
        if w[0] or w[6] or w[7]:
            BLK.append((self.n, self.run_id, w))
    return _apply(self)
R.PopulationRuntime._apply_wanted = _apply_blk
INV = []  # TEMP: arena invariant (held blocks + free stack = every id once, per group), checked before resizes
def _check_arena(rt, when):
    ring = rt.ring
    if not ring.pool_b or len(INV) >= 2:
        return
    g, S, nb, B = ring.n_groups, ring.n_slots, ring.pool_nb, ring.pool_b
    table, side, stack = ring.table.cpu(), ring.side_table.cpu(), ring.stack.cpu()
    nblk, scount, top = ring.nblk.cpu(), ring.side_count.cpu(), ring.top.cpu()
    for grp in range(g):
        ids = []
        for c in range(grp * S, (grp + 1) * S):
            ids += [(int(x), f"main cell {c}") for x in table[c, :int(nblk[c])]]
            ids += [(int(x), f"side cell {c}") for x in side[c, :-(-int(scount[c]) // B)]]
        ids += [(int(x), "free") for x in stack[grp, :int(top[grp])]]
        seen = {}
        dups = []
        for i, where in ids:
            if i in seen:
                dups.append((i, seen[i], where))
            seen[i] = where
        missing = sorted(set(range(nb)) - set(seen))
        if dups or missing:
            INV.append((rt.n, when, grp, dups[:4], missing[:8], len(ids), nb))
            return
_apply2 = R.PopulationRuntime._apply_wanted
def _apply_inv(self):
    if self._wanted is None:
        return _apply2(self)
    nb0 = self.ring.pool_nb
    _check_arena(self, f"{self.run_id}, before resizing from {nb0}")
    out = _apply2(self)
    _check_arena(self, f"{self.run_id}, after resizing {nb0} -> {self.ring.pool_nb}")
    return out
R.PopulationRuntime._apply_wanted = _apply_inv
PROFILE = "--profile" in sys.argv
t0 = time.time()
obj = P._network(G)
net = obj[0]
pops = P._runtimes("network", obj)
names = {id(p): n for n, p in pops.items()}
order = [names[id(p)] for p in list(net.populations)]
stops = [0]
_rec = R._Launches.recover
R._Launches.recover = lambda self: (stops.__setitem__(0, stops[0] + 1), _rec(self))[1]
r, prof = 0, None
try:
    for r in range(7005):
        P._advance("network", obj, r)
    torch.cuda.synchronize()
    if PROFILE:
        prof = profile(activities=[ProfilerActivity.CUDA])
        prof.__enter__()
    t1 = time.perf_counter()
    for r in range(7005, 7205):
        P._advance("network", obj, r)
    torch.cuda.synchronize()
    per_round = (time.perf_counter() - t1) / 200
    if prof:
        prof.__exit__(None, None, None)
    for r in range(7205, 7305):
        P._advance("network", obj, r)
    for p in pops.values():
        P._runtime("network", p).check()
except Exception as e:
    print(f"FAULT near round {r}: {type(e).__name__}: {str(e).splitlines()[0][:140]}")
    sys.exit(1)
for nm_, p_ in pops.items():
    w = P._runtime("network", p_).ring.buf[19:32].tolist()
    if w[0] or w[6] or w[7]:
        BLK.append((P._runtime("network", p_).n, "end", w))
for n_, when, grp, dups, missing, n_ids, nb in INV:
    print(f"  ARENA INVARIANT broken (n={n_}, run {when}): group {grp}, {n_ids} ids for {nb} "
          f"blocks; duplicates (id, first, second) {dups}; missing {missing}")
if not INV:
    print("  arena invariant held at every resize checked")
for n_, when, w in BLK[:8]:
    print(f"  STAGING (n={n_}, at {when}): blocks past arena {w[0]}; link index out of its connection {w[6]}, "
          f"LDS attr pointer != plan's {w[7]}; first: source word {w[9]:#x} link {w[10]} of {w[11]}, "
          f"position {w[12] >> 16} of count {w[12] & 0xFFFF}")
import struct  # TEMP: the pull validation's counters (runtime.overrun words 1..)
for nm, p in pops.items():
    o = P._runtime("network", p).overrun.tolist()
    if o[1] or o[2] or o[3]:
        e = struct.unpack("f", struct.pack("i", o[15]))[0]
        print(f"  PULL CHECK {nm}: bad spikes {o[1]}, bad segment ranges {o[2]}, bad link ranges {o[3]}; first: "
              f"group {o[5]} round {o[6]} conn {o[7]} slot {o[8]} index {o[9]}: source {o[10]} of {o[11]}, "
              f"segments [{o[12]}, {o[13]}) of {o[14]}, e {e:.4f}, e-group spikes {o[16]}; links: lo {o[17]} "
              f"n {o[18]} segment {o[19]}")
resizes = sum(len(P._runtime("network", p).resizes) for p in pops.values())
warps = P._runtime("network", list(net.populations)[0])._warps(None)
print(f"no fault through round 7305 ({torch.cuda.get_device_name(0)}, G={G}, {warps} warps): volley's first 200 "
      f"rounds {per_round * 1e3:.2f} ms/round{' (profiled)' if PROFILE else ''}; {resizes} resizes, {stops[0]} "
      f"recoveries in all; {time.time() - t0:.0f} s")
if prof:
    ks = sorted((e.time_range.start, e.time_range.end) for e in prof.events()
                if e.device_type.name == "CUDA" and "fused_kernel" in e.name)
    # each round's kernels in launch order (reruns after a stop add launches: rounds with extra kernels skipped)
    per = {n: [] for n in order}
    rounds = len(ks) // len(order)
    for i in range(rounds * len(order)):
        a, b = ks[i]
        per[order[i % len(order)]].append(b - a)
    sizes = {n: P._runtime("network", p).n for n, p in pops.items()}
    print(f"  per population, {rounds} rounds' kernels (us, median / max), neurons:")
    for n in order:
        d = per[n]
        print(f"    {n:5s} {statistics.median(d):7.1f} / {max(d):8.1f}   n={sizes[n]}")
    print(f"  sum of medians {sum(statistics.median(v) for v in per.values()):.0f} us/round; slowest kernel "
          f"{max(max(v) for v in per.values()):.0f} us; {len(ks)} kernels in 200 rounds")
