"""
Minimal 2-population E/I network derived from PD14.

Two populations:
  E — 100 excitatory LIF neurons
  I —  25 inhibitory LIF neurons

Four connections (all four E/I combinations):
  E→E, E→I, I→E, I→I

Uses the same PD14_neuron type; K_cortical and sizes are scaled down to make
this a fast smoke-test for compilation. No edge class: every connection is
bare (just a per-link weight tensor, set directly on the connection -- see
naer2/connection.py's Connection.__setattr__), since none of them need a
learning rule.

Connection probabilities chosen so that each population receives ≥ 1 link
per source neuron on average.
"""

import torch

from test.networks.PD14.connection import PD14_connection
from test.networks.PD14.neuron import PD14_neuron

_J_E =  0.08781   # nA — excitatory PSC amplitude
_J_I = -0.35124   # nA — inhibitory PSC amplitude

# probabilities tuned for ~4 links per source neuron at these sizes
_P_EE = 0.10
_P_EI = 0.15
_P_IE = 0.20
_P_II = 0.20


def build():
    """Build and return the two-population network.

    Returns
    -------
    network : Network
        The unified network containing E and I populations.
    pops : dict
        {'E': Population, 'I': Population} for direct access.
    """
    N_E, N_I = 100, 25

    # ── populations ─────────────────────────────────────────────────────────
    E_inst = PD14_neuron(K_cortical=125)   # K_cortical ≈ N_E + N_I scaled
    I_inst = PD14_neuron(K_cortical=125)

    E_pop = E_inst * N_E
    I_pop = I_inst * N_I

    # randomise initial membrane potentials around resting values
    E_pop.potential = torch.normal(-65.0, 5.0, (N_E,))
    I_pop.potential = torch.normal(-65.0, 5.0, (N_I,))

    # ── connections ─────────────────────────────────────────────────────────
    # Deliberately no per-link delay: this network exists to exercise the
    # no-delay update kernel's AST shape (test_mini_ei.py), and a real
    # per-link delay tensor would switch it to the (currently stubbed)
    # per-link-delay kernel instead -- see Population.uses_per_link_delay_kernel
    # and TODO.md. A dedicated delay-carrying fixture belongs with that
    # kernel's own tests once it exists.
    def _connect(src, dst, prob, w_mean, w_std):
        conn = src >> PD14_connection(prob) >> dst
        w = torch.normal(w_mean, abs(w_mean) * 0.1, conn.shape)
        conn.weight = w.clamp(min=0) if w_mean >= 0 else w.clamp(max=0)
        return conn

    _connect(E_pop, E_pop, _P_EE, _J_E, abs(_J_E) * 0.1)
    _connect(E_pop, I_pop, _P_EI, _J_E, abs(_J_E) * 0.1)
    _connect(I_pop, E_pop, _P_IE, _J_I, abs(_J_I) * 0.1)
    _connect(I_pop, I_pop, _P_II, _J_I, abs(_J_I) * 0.1)

    network = E_pop.network
    pops = {"E": E_pop, "I": I_pop}
    return network, pops
