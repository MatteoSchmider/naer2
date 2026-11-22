"""
Brunel (2000), model A: a sparse balanced network of N_E = 4 N_I identical LIF neurons with delta synapses
(BRUNEL.md) -- event-driven, exact spike times.

Units: ms / mV. Every neuron receives C_E excitatory inputs (PSP J), C_I = gamma C_E inhibitory ones (-g J),
and C_ext = C_E from a Poisson population of N_P = N_E sources (J), each at eta nu_thr -- nu_thr =
theta / (J C_E tau), the rate that drives a neuron to threshold on average without recurrence. One homogeneous
delay D on every connection: receive's time is each input's exact arrival (D-1), the sources run every D
(Population._own_timestep).

Scaling: strong by default -- epsilon constant, C_E = epsilon N_E growing with N; K given, a fixed in-degree
C_E = K instead.
"""

import torch

from .connection import Brunel_indegree
from .neuron import Brunel_neuron
from .poisson import Brunel_poisson

description = {
    "N": 10000,  # N_E + N_I
    "f_E": 0.8,  # N_E / N
    "epsilon": 0.1,  # connection probability (strong scaling: C_E = epsilon N_E)
    "gamma": 0.25,  # C_I / C_E
    "J": 0.1,  # mV, the excitatory PSP
    "g": 5.0,  # relative inhibition: inhibitory PSP -g J
    "eta": 2.0,  # external rate / nu_thr
    "D": 1.5,  # ms, the delay of every connection
    "neuron": {"tau": 20.0, "theta": 20.0, "V_r": 10.0, "t_ref": 2.0},
    "group_size": 256,
    # the rings' arenas from the start at this fraction of a group's incoming links: the general pull (precise
    # spike times) can't stop a run that outgrows them -- its links would be lost -- and the start's transient
    # (potentials uniform below threshold) is a burst; steady state needs ~rate x (D + a window) ~ 0.1
    "ring_in_flight": 0.25,
}


def _connect(source, target, C: int, weight: float, delay: float):
    conn = source >> Brunel_indegree(C) >> target
    conn.weight = torch.full(conn.shape, weight)
    conn.delay = delay
    return conn


def build(N: int | None = None, K: int | None = None, eta: float | None = None, g: float | None = None,
          desc: dict = description, seed: int = 0, neuron=Brunel_neuron):
    """Brunel's model A: (network, {"E", "I", "P": populations}). N: neurons (N_E + N_I); K: a fixed in-degree
    C_E (None: epsilon N_E); eta, g: the regime (default A's AI state: eta 2, g 5). Initial potentials uniform
    in [0, theta); the sources draw their first spikes themselves. neuron: the node class (Brunel_neuron; a
    test's instrumented variant)."""
    N = N or desc["N"]
    eta = desc["eta"] if eta is None else eta
    g = desc["g"] if g is None else g
    J, D, nrn = desc["J"], desc["D"], desc["neuron"]
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # (the patterns' draws)

    N_E = round(desc["f_E"] * N)
    N_I = N - N_E
    C_E = K if K is not None else round(desc["epsilon"] * N_E)
    C_I = max(round(desc["gamma"] * C_E), 1)
    rate = eta * nrn["theta"] / (J * C_E * nrn["tau"])  # eta nu_thr, spikes / ms

    E = neuron(**nrn) * N_E
    I = neuron(**nrn) * N_I
    P = Brunel_poisson(rate) * N_E
    E.V = torch.rand(N_E, generator=gen) * nrn["theta"]
    I.V = torch.rand(N_I, generator=gen) * nrn["theta"]
    E.group_size = I.group_size = P.group_size = desc["group_size"]
    E.ring_in_flight = I.ring_in_flight = desc["ring_in_flight"]

    for target in (E, I):
        _connect(E, target, C_E, J, D)
        _connect(I, target, C_I, -g * J, D)
        _connect(P, target, C_E, J, D)
    return E.network, {"E": E, "I": I, "P": P}


def theory(N: int | None = None, K: int | None = None, eta: float | None = None, desc: dict = description) -> dict:
    """Reference numbers: C_E, C_I, nu_thr and the external rate per source (Hz)."""
    N = N or desc["N"]
    eta = desc["eta"] if eta is None else eta
    N_E = round(desc["f_E"] * N)
    C_E = K if K is not None else round(desc["epsilon"] * N_E)
    nu_thr = desc["neuron"]["theta"] / (desc["J"] * C_E * desc["neuron"]["tau"])
    return {"C_E": C_E, "C_I": max(round(desc["gamma"] * C_E), 1), "nu_thr_Hz": 1000 * nu_thr,
            "rate_ext_Hz": 1000 * eta * nu_thr}
