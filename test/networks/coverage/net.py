from test.edges import plastic_synapse, static_synapse
from test.nodes import relay
from test.patterns import all_to_all

import torch


def build():
    N = 10

    # ── populations ─────────────────────────────────────────────────────────
    A = relay(decay=0.9, threshold=5) * N
    B = relay(decay=0.9, threshold=5) * N
    C = relay(decay=0.9, threshold=5) * N
    D = relay(decay=0.9, threshold=5) * N

    A.potential = torch.zeros((N,))
    B.potential = torch.zeros((N,))
    C.potential = torch.zeros((N,))
    D.potential = torch.zeros((N,))

    # ── connections ─────────────────────────────────────────────────────────
    # "static" means the weight tensor doesn't change over the simulation, not
    # that every link shares one value — give it a real per-link tensor here,
    # same as a static synapse would use in a real network.
    conn_ab = A >> static_synapse(1) @ {"weight": "w"} @ "positive" @ all_to_all >> B
    conn_ab.fabric.weight = torch.rand(conn_ab.shape)

    B @ {"potential": "src_pot"} >> plastic_synapse(1, 0.1) @ "negative" @ {"weight": "w"} @ all_to_all >> C

    # Three separate source populations feed D through the SAME receive port
    # ("positive"): A and B via the same edge type (static_synapse), C via a
    # DIFFERENT edge type (plastic_synapse). Each gets its own CSR structure
    # and its own transpiled edge-forward AST (plastic_synapse's mutates weight
    # per link, static_synapse's is a no-op) — this is the case that actually
    # exercises "different code per connection", not just "different topology
    # per connection", dispatched at runtime via spike_conn.
    conn_ad = A >> static_synapse(1) @ {"weight": "w"} @ "positive" @ all_to_all >> D
    conn_ad.fabric.weight = torch.rand(conn_ad.shape)

    conn_bd = B >> static_synapse(1) @ {"weight": "w"} @ "positive" @ all_to_all >> D
    conn_bd.fabric.weight = torch.rand(conn_bd.shape)

    conn_cd = C >> plastic_synapse(1, 0.1) @ "positive" @ {"weight": "w"} @ all_to_all >> D

    network = A.network
    pops = {"A": A, "B": B, "C": C, "D": D}
    return network, pops
