"""Random TopologyNew connections for the GPU tests and tools."""

import torch

from naer2.topology import Topology
from naer2.topology_new import TopologyNew


def random_topology(n_src, n_tgt, group_size, delay="random", k=6, seed=0, weight=False, device="cuda:0"):
    """A random connection: every source gets up to k distinct targets
    (sorted, canonical), per-link delays in [0.1, 0.6) at resolution 0.1
    ("random"), one homogeneous delay (a float), or none; with `weight`, a
    random per-link weight attribute."""
    g = torch.Generator().manual_seed(seed)
    rows = [torch.sort(torch.randperm(n_tgt, generator=g)[: torch.randint(1, k + 1, (1,), generator=g).item()]).values
            for _ in range(n_src)]
    topo = Topology(torch.tensor([0] + [len(r) for r in rows]).cumsum(0), torch.cat(rows))
    attrs = {"weight": torch.rand(len(topo), generator=g)} if weight else None
    if delay == "random":
        d = torch.rand(len(topo), generator=g) * 0.5 + 0.1
        return TopologyNew(topo, n_tgt, group_size, device, delay=d, resolution=0.1, attrs=attrs)
    return TopologyNew(topo, n_tgt, group_size, device, delay=delay, attrs=attrs)
