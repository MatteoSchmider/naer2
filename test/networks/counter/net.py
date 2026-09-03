from test.edges import plastic_synapse, static_synapse
from test.nodes import counter
from test.patterns import all_to_all

import torch


def build():
    N = 10

    # ── populations ─────────────────────────────────────────────────────────
    A = counter(decay=0.9, threshold=5) * N
    B = counter(decay=0.9, threshold=5) * N
    C = counter(decay=0.9, threshold=5) * N

    A.potential = torch.zeros((N,))
    B.potential = torch.zeros((N,))
    C.potential = torch.zeros((N,))

    # ── connections ─────────────────────────────────────────────────────────
    A >> static_synapse(1) @ "positive" @ {"weight": "w"} @ all_to_all() >> B
    B >> plastic_synapse(1, 0.1) @ "negative" @ {"weight": "w"} @ all_to_all() >> C

    network = A.network
    pops = {"A": A, "B": B, "C": C}
    return network, pops
