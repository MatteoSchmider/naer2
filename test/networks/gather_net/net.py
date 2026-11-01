"""
Two sources (A, B) feeding one target (C) through two real connections, for
test_network_gather.py. Deliberately NOT wired through the update kernel --
that test controls spikes with static data instead, so the only thing this
network exists to provide is real Population/Connection objects (real
`incoming` list, real connection indices for conn-id tagging) for the
compact()/merge_incoming()/Network.gather() integration to run against.

Reuses the `counter` node purely as a stand-in population type -- its
update/send/receive hooks are never invoked here.
"""

from test.edges import static_synapse
from test.nodes import counter
from test.patterns import all_to_all

import torch


def build():
    N = 5

    A = counter() * N
    B = counter() * N
    C = counter() * N

    A.potential = torch.zeros((N,))
    B.potential = torch.zeros((N,))
    C.potential = torch.zeros((N,))

    A >> static_synapse(0.0) @ all_to_all >> C
    B >> static_synapse(0.0) @ all_to_all >> C

    network = A.network
    pops = {"A": A, "B": B, "C": C}
    return network, pops
