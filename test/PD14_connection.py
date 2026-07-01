import math

import torch

import naer2 as n2
from naer2.types import Points, Position, Region


@n2.pattern
class PD14_connection:
    """Random fixed-total-number connectivity from PD14 spec eq. (1).

    Q_yx = ln(1 - C_yx) / ln(1 - 1/(N_x * N_y)) total connections are drawn
    uniformly with replacement. Each source neuron receives Q_yx / N_x targets.
    """

    def __init__(self, probability: float):
        self.probability = probability

    def n_connections(self, N_x: int, N_y: int) -> int:
        Q_yx = math.log(1 - self.probability) / math.log(1 - 1 / (N_x * N_y))
        return max(1, round(Q_yx))

    def __call__(
        self, position: Position, source_region: Region, target_region=None
    ) -> Region:
        if target_region is None:
            target_region = source_region
        N_x = int(source_region[0][1] - source_region[0][0])
        N_y = int(target_region[0][1] - target_region[0][0])
        Q_yx = math.log(1 - self.probability) / math.log(1 - 1 / (N_x * N_y))
        k = round(Q_yx / N_x)
        if k == 0:
            return Points(torch.zeros((0, 1), dtype=torch.int64))
        return Points(torch.randint(0, N_y, (k, 1), dtype=torch.int64))
