import torch

import naer2 as n2
from naer2.types import Points, Position, Region


@n2.pattern
class Brunel_indegree:
    """Fixed in-degree: every target neuron draws K sources uniformly from the source population, with
    replacement (multapses kept; autapses too where source and target are one population -- as NEST's
    brunel_delta example). Drawn whole on the first call -- a pattern runs once per source -- and each
    source's targets handed out from it."""

    def __init__(self, K: int):
        self.K = K
        self._rows = None

    def __call__(self, position: Position, source_region: Region, target_region=None) -> Region:
        if target_region is None:
            target_region = source_region
        lo = int(source_region[0][0])
        if self._rows is None:
            n_src = int(source_region[0][1] - source_region[0][0])
            n_tgt = int(target_region[0][1] - target_region[0][0])
            src = torch.randint(0, n_src, (n_tgt * self.K,))
            tgt = torch.arange(n_tgt).repeat_interleave(self.K)
            order = torch.argsort(src, stable=True)
            offsets = torch.zeros(n_src + 1, dtype=torch.int64)
            offsets[1:] = torch.cumsum(torch.bincount(src, minlength=n_src), 0)
            self._rows = (tgt[order], offsets)
        targets, offsets = self._rows
        s = int(position[0]) - lo
        return Points(targets[offsets[s]:offsets[s + 1]].unsqueeze(1))
