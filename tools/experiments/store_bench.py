"""Ring entry layouts, both sides:

  write  intake's appends: P programs x TILES tiles of BLOCK appends, each a
         16-byte record (4 words) at (cell, pos), cell over the hot cells,
         pos a small per-cell index (what the appends' atomics hand out)
  read   delivery's header loads: one program per group reads its cell's
         run, CHUNK entries at a time (the update kernel's chunks)

Layouts (record index of entry (cell, pos), 4 words per record):
  soa          4 separate field arrays (the ring before)
  contiguous   cell * cap + pos: each cell's run contiguous (the ring now)
  interleaved  (pos // 4 * n_cells + cell) * 4 + pos % 4: runs cut into
               cache lines of 4 records, line b of every cell side by side --
               the live front of all cells is dense, bursts reach deeper

    python tools/store_bench.py [--rdna2]
"""
import os
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", ".."), os.path.join(os.path.dirname(__file__), "..")]
if "--rdna2" in sys.argv:
    import rdna2  # noqa: F401
import torch
import triton
import triton.language as tl

SOA, CONTIGUOUS, INTERLEAVED = 0, 1, 2
NAMES = {SOA: "soa", CONTIGUOUS: "contiguous", INTERLEAVED: "interleaved"}


@triton.jit
def _word(cell, pos, f, n_cells, CAP: tl.constexpr, LAYOUT: tl.constexpr):
    if LAYOUT == 0:
        return f * n_cells * CAP + cell * CAP + pos
    elif LAYOUT == 1:
        return (cell * CAP + pos) * 4 + f
    else:
        return (((pos >> 2) * n_cells + cell) * 4 + (pos & 3)) * 4 + f


@triton.jit
def writes(buf, cells_ptr, pos_ptr, n_cells, CAP: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr,
           LAYOUT: tl.constexpr):
    pid = tl.program_id(0)
    for i in range(TILES):
        o = (pid * TILES + i) * BLOCK + tl.arange(0, BLOCK)
        cell = tl.load(cells_ptr + o)
        pos = tl.load(pos_ptr + o)
        for f in tl.static_range(4):
            tl.store(buf + _word(cell, pos, f, n_cells, CAP, LAYOUT), o + f)


@triton.jit
def reads(buf, out, n_entries, n_cells, slot, N_SLOTS: tl.constexpr, CAP: tl.constexpr, CHUNK: tl.constexpr,
          LAYOUT: tl.constexpr):
    g = tl.program_id(0)
    acc = tl.zeros([CHUNK], tl.int32)
    for k in range(N_SLOTS):  # every slot of this group's ring, one after another (like successive runs)
        cell = g * N_SLOTS + (slot + k) % N_SLOTS
        for b0 in range(0, n_entries, CHUNK):
            pos = b0 + tl.arange(0, CHUNK)
            m = pos < n_entries
            for f in tl.static_range(4):
                acc += tl.load(buf + _word(cell, pos, f, n_cells, CAP, LAYOUT), mask=m, other=0)
    tl.store(out + g * CHUNK + tl.arange(0, CHUNK), acc)


def best(fn, reps=7):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


GROUPS, N_SLOTS = 86, 33  # PD14 E_4's consumer at G=256
N_CELLS = GROUPS * N_SLOTS
P, TILES, BLOCK = 24, 64, 512
n = P * TILES * BLOCK
print(f"{torch.cuda.get_device_name(0)}: {GROUPS} groups x {N_SLOTS} slots = {N_CELLS} cells, 16-byte records")
print(f"write: {P} programs x {TILES} tiles x {BLOCK} appends into ~860 hot cells, pos < 16")
for cap in (2048, 512):
    buf = torch.zeros(4 * N_CELLS * cap, dtype=torch.int32, device="cuda")
    cells = (torch.randint(0, 860, (n,), device="cuda", dtype=torch.int32) * 3) % N_CELLS
    pos = torch.randint(0, 16, (n,), device="cuda", dtype=torch.int32)
    row = []
    for layout in (SOA, CONTIGUOUS, INTERLEAVED):
        dt = best(lambda: writes[(P,)](buf, cells, pos, N_CELLS, cap, TILES, BLOCK, layout, num_warps=4))
        row.append(f"{NAMES[layout]} {dt / TILES * 1e6:5.1f}")
    print(f"  cap {cap:5d} ({buf.numel() * 4 / 2**20:5.0f} MB), us per tile: " + " | ".join(row))
print(f"read: {GROUPS} programs, each its {N_SLOTS} cells' runs of n entries, 64 per chunk")
for cap in (2048, 512):
    buf = torch.randint(0, 100, (4 * N_CELLS * cap,), dtype=torch.int32, device="cuda")
    out = torch.zeros(GROUPS * 64, dtype=torch.int32, device="cuda")
    for n_entries in (16, 180, 500):
        row = []
        for layout in (SOA, CONTIGUOUS, INTERLEAVED):
            dt = best(lambda: reads[(GROUPS,)](buf, out, n_entries, N_CELLS, 5, N_SLOTS, cap, 64, layout, num_warps=4))
            row.append(f"{NAMES[layout]} {dt * 1e6:5.1f}")
        print(f"  cap {cap:5d}, {n_entries:3d} entries per cell, us per launch: " + " | ".join(row))
