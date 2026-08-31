"""
Connectivity patterns — reference for the naer2 connection builder API.

A connection is built with the shorthand:
    pop_a >> edge_instance @ pattern >> pop_b

The pattern runs once per source neuron and returns coordinates in the target
population's coordinate space.  The builder maps those coordinates to linear
indices to produce the CSR topology.  Four canonical patterns are shown below.

Run from the project root:
    python -m test.topology
    PYTHONPATH=. python test/topology.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import naer2 as n2
from naer2.population import Population
from naer2.types import Region
from naer2.types.points import ContinuousPoints, IndexPoints
from naer2.types.position import Position

from test.nodes import lif_cd
from test.edges import static_synapse
from test.patterns import all_to_all


# ─── 1. All-to-all ───────────────────────────────────────────────────────────
#
# The simplest pattern: return the entire target region.  The builder
# materialises all target coordinates and maps them to indices.
# node * size creates a flat 1D population of `size` neurons.

pop_4 = lif_cd(v_thresh=-50, v_reset=-65, t_mem=0.99) * 4
pop_3 = lif_cd(v_thresh=-50, v_reset=-65, t_mem=0.99) * 3

conn1 = pop_4 >> static_synapse(weight=1.0) @ all_to_all >> pop_3

topo1 = conn1.fabric.topology
assert len(topo1) == 12
for i in range(4):
    s, e = int(topo1.source_offsets[i]), int(topo1.source_offsets[i + 1])
    assert sorted(topo1.target_indices[s:e].tolist()) == [0, 1, 2]
print("1. all-to-all 1D IndexBox              ✓")


# ─── 2. Distance threshold with physical coordinates ─────────────────────────
#
# Populations can be created with arbitrary physical coordinate tensors by
# passing a Points region directly to Population().  Patterns receive the
# source position as a ContinuousPosition and can compute distances, angles,
# or any spatial relationship.
#
# Returning a subset of tgt.coords (exact row matches) lets the builder do
# an O(1) dict lookup per point to resolve target indices.
#
#   source 0.0 → targets {0.0, 0.25}         = 2 links
#   source 0.5 → targets {0.25, 0.5, 0.75}   = 3 links
#   source 1.0 → targets {0.75, 1.0}          = 2 links

@n2.pattern
class by_distance:
    def __init__(self, radius: float):
        self.radius = radius

    def __call__(self, pos: Position, src: Region, tgt=None):
        dists = (tgt.coords - pos.coords.unsqueeze(0)).abs().squeeze(1)
        mask = dists < self.radius
        if not mask.any():
            return ContinuousPoints(torch.zeros((0, 1), dtype=tgt.coords.dtype))
        return ContinuousPoints(tgt.coords[mask])

src_coords = torch.tensor([[0.0], [0.5], [1.0]])
tgt_coords = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
pop_src = Population(lif_cd(v_thresh=-50, v_reset=-65, t_mem=0.99), ContinuousPoints(src_coords))
pop_tgt = Population(lif_cd(v_thresh=-50, v_reset=-65, t_mem=0.99), ContinuousPoints(tgt_coords))

conn2 = pop_src >> static_synapse(weight=1.0) @ by_distance(0.3) >> pop_tgt

topo2 = conn2.fabric.topology
assert len(topo2) == 7
assert [int(topo2.source_offsets[i+1]) - int(topo2.source_offsets[i]) for i in range(3)] == [2, 3, 2]
print("2. distance-threshold ContinuousPoints ✓")


# ─── 3. 2D grid neighborhood ─────────────────────────────────────────────────
#
# node * (H, W) creates a 2D population; each neuron's position is a 2D
# IndexPosition (row, col).  Patterns see the full nD coordinate and can
# implement convolution-like local connectivity without any special framework
# support — it's just arithmetic on coordinates.
#
# Extra dimensions can encode cell types, layers, or any per-neuron metadata,
# enabling complex heterogeneous patterns from a single data tensor.
#
# 4-connected 3×3 grid: 4 corners×2 + 4 edges×3 + 1 center×4 = 24 links

@n2.pattern
def neighborhood_4(pos: Position, src: Region, tgt=None):
    r, c = int(pos.coords[0]), int(pos.coords[1])
    H, W = int(tgt[0][1] - tgt[0][0]), int(tgt[1][1] - tgt[1][0])
    neighbors = [
        [r + dr, c + dc]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if 0 <= r + dr < H and 0 <= c + dc < W
    ]
    if not neighbors:
        return IndexPoints(torch.zeros((0, 2), dtype=torch.int64))
    return IndexPoints(torch.tensor(neighbors, dtype=torch.int64))

pop_grid_src = lif_cd(v_thresh=-50, v_reset=-65, t_mem=0.99) * (3, 3)
pop_grid_tgt = lif_cd(v_thresh=-50, v_reset=-65, t_mem=0.99) * (3, 3)

conn3 = pop_grid_src >> static_synapse(weight=1.0) @ neighborhood_4 >> pop_grid_tgt

topo3 = conn3.fabric.topology
assert len(topo3) == 24
s, e = int(topo3.source_offsets[4]), int(topo3.source_offsets[5])  # flat index 4 = (1,1)
assert sorted(topo3.target_indices[s:e].tolist()) == [1, 3, 5, 7]
print("3. 2D neighborhood IndexBox            ✓")


# ─── 4. Target population inference ──────────────────────────────────────────
#
# When the right-hand side of >> is a node instance rather than a Population,
# the builder runs the pattern over the source to discover what target
# coordinates it produces, infers the smallest bounding box that covers them,
# and creates the target population automatically.
#
# This enables convolution and pooling layers where the output shape is a
# function of the pattern and source shape, not specified explicitly.

@n2.pattern
def double_index(pos: Position, src: Region):  # src unused — pattern depends only on position
    idx = int(pos.coords[0])
    return IndexPoints(torch.tensor([[idx * 2], [idx * 2 + 1]], dtype=torch.int64))

pop_5 = lif_cd(v_thresh=-50, v_reset=-65, t_mem=0.99) * 5
conn4 = pop_5 >> static_synapse(weight=1.0) @ double_index >> lif_cd(v_thresh=-50, v_reset=-65, t_mem=0.99)

topo4 = conn4.fabric.topology
assert len(topo4) == 10
assert len(conn4._target.shape) == 10  # inferred: neurons 0..9
for i in range(5):
    s, e = int(topo4.source_offsets[i]), int(topo4.source_offsets[i + 1])
    assert sorted(topo4.target_indices[s:e].tolist()) == [2 * i, 2 * i + 1]
print("4. target inference                    ✓")
