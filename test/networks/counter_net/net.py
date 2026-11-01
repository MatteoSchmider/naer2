"""
Minimal two-population network for exercising a compiled kernel on real GPU
hardware (see TODO.md's staged GPU verification). Unlike test/networks/coverage
(built to exercise AST shapes via inspection, never actually run), this one
uses the `counter` node specifically so every quantity involved is a small
integer and the spike count/timing from a given starting potential can be
checked by hand -- see test_gpu_hardware.py.

One connection, A -> B, all-to-all, with an unused weight: counter.receive
never reads edge.weight, so no real per-link tensor is needed for this to be
a structurally valid connection (CSR/link-tiling structure still exists and
still gets walked -- there's just nothing for the kernel to load off it).
"""

from test.edges import static_synapse
from test.nodes import counter
from test.patterns import all_to_all

import torch


def build():
    N = 10

    A = counter() * N
    B = counter() * N

    A.potential = torch.zeros((N,))
    B.potential = torch.zeros((N,))

    A >> static_synapse(0.0) @ all_to_all >> B

    network = A.network
    pops = {"A": A, "B": B}
    return network, pops
