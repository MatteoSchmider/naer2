"""
Exercises Topology.build across the four intended use cases:

  1. All-to-all with 1D IndexBox (baseline)
  2. Distance threshold with ContinuousPoints (physical coordinate connectivity)
  3. 2D IndexBox neighborhood (convolution-like; exercises nD stride arithmetic)
  4. Target inference (no explicit target_region; exercises _infer_region)
"""

import torch

from naer2.topology import Topology
from naer2.types.box import IndexBox
from naer2.types.points import ContinuousPoints, IndexPoints


# ── 1. all-to-all with 1D IndexBox ───────────────────────────────────────────

def _all_to_all(pos, src, tgt):
    return tgt  # every source neuron → entire target region (returned as Box)

source_4 = IndexBox(torch.tensor([[0, 4, 1]], dtype=torch.int64))
target_3 = IndexBox(torch.tensor([[0, 3, 1]], dtype=torch.int64))

topo1 = Topology.build(_all_to_all, source_4, target_3)
assert len(topo1) == 12, f"expected 12 links, got {len(topo1)}"
for i in range(4):
    s, e = int(topo1.source_offsets[i]), int(topo1.source_offsets[i + 1])
    row = sorted(topo1.target_indices[s:e].tolist())
    assert row == [0, 1, 2], f"source {i}: expected [0,1,2], got {row}"
print("1. all-to-all IndexBox                ✓")


# ── 2. distance threshold with ContinuousPoints ──────────────────────────────

RADIUS = 0.3

def _distance_pattern(pos, src, tgt):
    dists = (tgt.coords - pos.coords.unsqueeze(0)).abs().squeeze(1)
    mask = dists < RADIUS
    if not mask.any():
        return ContinuousPoints(torch.zeros((0, 1), dtype=torch.float32))
    return ContinuousPoints(tgt.coords[mask])

src_coords = torch.tensor([[0.0], [0.5], [1.0]])
tgt_coords = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
cont_src = ContinuousPoints(src_coords)
cont_tgt = ContinuousPoints(tgt_coords)

topo2 = Topology.build(_distance_pattern, cont_src, cont_tgt)
# 0.0 → {0.0, 0.25}         = 2
# 0.5 → {0.25, 0.5, 0.75}   = 3
# 1.0 → {0.75, 1.0}         = 2
assert len(topo2) == 7, f"expected 7 links, got {len(topo2)}"
fan_outs = [
    int(topo2.source_offsets[i + 1]) - int(topo2.source_offsets[i])
    for i in range(3)
]
assert fan_outs == [2, 3, 2], f"expected fan-outs [2,3,2], got {fan_outs}"
print("2. distance-threshold ContinuousPoints ✓")


# ── 3. 2D IndexBox neighborhood (4-connected grid) ───────────────────────────

def _neighborhood_4(pos, src, tgt):
    r, c = int(pos.coords[0]), int(pos.coords[1])
    H = int(tgt[0][1] - tgt[0][0])
    W = int(tgt[1][1] - tgt[1][0])
    neighbors = [
        [r + dr, c + dc]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if 0 <= r + dr < H and 0 <= c + dc < W
    ]
    if not neighbors:
        return IndexPoints(torch.zeros((0, 2), dtype=torch.int64))
    return IndexPoints(torch.tensor(neighbors, dtype=torch.int64))

grid3x3 = IndexBox(torch.tensor([[0, 3, 1], [0, 3, 1]], dtype=torch.int64))

topo3 = Topology.build(_neighborhood_4, grid3x3, grid3x3)
# 4 corners ×2 + 4 edges ×3 + 1 center ×4 = 24 links
assert len(topo3) == 24, f"expected 24 links, got {len(topo3)}"
# flat index 4 = position (1,1) — center → {(0,1)=1, (2,1)=7, (1,0)=3, (1,2)=5}
s, e = int(topo3.source_offsets[4]), int(topo3.source_offsets[5])
center_targets = sorted(topo3.target_indices[s:e].tolist())
assert center_targets == [1, 3, 5, 7], f"center: expected [1,3,5,7], got {center_targets}"
print("3. 2D neighborhood IndexBox            ✓")


# ── 4. target inference ───────────────────────────────────────────────────────

def _double_index(pos, src):
    idx = int(pos.coords[0])
    return IndexPoints(torch.tensor([[idx * 2], [idx * 2 + 1]], dtype=torch.int64))

source_5 = IndexBox(torch.tensor([[0, 5, 1]], dtype=torch.int64))

topo4, inferred = Topology.build(_double_index, source_5)
assert len(topo4) == 10, f"expected 10 links, got {len(topo4)}"
assert len(inferred) == 10, f"expected inferred region size 10, got {len(inferred)}"
for i in range(5):
    s, e = int(topo4.source_offsets[i]), int(topo4.source_offsets[i + 1])
    row = sorted(topo4.target_indices[s:e].tolist())
    assert row == [2 * i, 2 * i + 1], f"source {i}: expected {[2*i, 2*i+1]}, got {row}"
print("4. target inference                    ✓")
