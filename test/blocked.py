"""
BlockedTopology unit test (pure torch, no Triton).

For several topologies — dense, sparse random, empty rows, a ragged last block —
checks that the block-major layout is a faithful re-arrangement of the canonical
CSR: every link lands in exactly the block that owns its target, each
(block, source) range holds exactly the canonical links from that source into
that block, and per-link tensors round-trip through to_blocked / from_blocked.
"""

import torch

from naer2.topology import Topology


def _topology(n_src: int, n_tgt: int, density: float, seed: int) -> Topology:
    g = torch.Generator().manual_seed(seed)
    rows = [torch.nonzero(torch.rand(n_tgt, generator=g) < density).flatten() for _ in range(n_src)]
    offsets = torch.tensor([0] + [len(r) for r in rows]).cumsum(0)
    targets = torch.cat(rows) if offsets[-1] > 0 else torch.zeros(0, dtype=torch.int64)
    return Topology(offsets, targets)


def _check(topo: Topology, n_tgt: int, group_size: int, label: str) -> None:
    bt = topo.blocked(group_size, n_tgt)
    n_src = len(topo.source_offsets) - 1
    m = len(topo)
    assert bt.offsets.shape == (-(-n_tgt // group_size), n_src + 1), label
    assert int(bt.block_base[-1]) == m and sorted(bt.perm.tolist()) == list(range(m)), label
    assert m == 0 or int(bt.target_local.max()) < group_size, label

    canon_src = torch.repeat_interleave(torch.arange(n_src), topo.source_offsets.diff())
    for b in range(bt.n_blocks):
        sl = bt.block_slice(b)
        # the block's links are exactly the canonical links whose target it owns
        owned = (topo.target_indices // group_size) == b
        assert sorted(bt.perm[sl].tolist()) == sorted(torch.nonzero(owned).flatten().tolist()), (label, b)
        assert (b * group_size + bt.target_local[sl].to(torch.int64) == topo.target_indices[bt.perm[sl]]).all(), (label, b)
        for s in range(n_src):
            lo, hi = int(bt.offsets[b, s]), int(bt.offsets[b, s + 1])
            expect = sorted(torch.nonzero(owned & (canon_src == s)).flatten().tolist())
            assert sorted(bt.perm[lo:hi].tolist()) == expect, (label, b, s)
            assert sl.start <= lo <= hi <= sl.stop, (label, b, s)

    w = torch.rand(m)
    assert torch.equal(bt.from_blocked(bt.to_blocked(w)), w), label


def main():
    cases = [
        ("dense 6x10, G=4 (ragged last block)", _topology(6, 10, 1.0, 0), 10, 4),
        ("sparse 20x37, G=8", _topology(20, 37, 0.15, 1), 37, 8),
        ("sparse with empty rows, G=5", _topology(9, 25, 0.05, 2), 25, 5),
        ("single block (G >= targets)", _topology(5, 7, 0.5, 3), 7, 16),
        ("no links at all", _topology(4, 6, 0.0, 4), 6, 4),
    ]
    for label, topo, n_tgt, g in cases:
        _check(topo, n_tgt, g, label)
        print(f"blocked layout faithful: {label} ✓")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
