import torch

from .connection import PD14_connection
from .neuron import PD14_neuron
from .poisson import PD14_poisson

"""
Implementation notes (Potjans & Diesmann 2014):

Units: ms / mV / nA / MOhm  (1 nA × 1 MOhm = 1 mV).

Weights: PSC amplitudes in nA from spec PSP J=0.15 mV via
  J_unit(tau_m=10, tau_s=0.5, R_m=40) ≈ 0.08781 nA (excitatory),
  -4 × J_unit ≈ -0.35124 nA (inhibitory), 2 × J_unit ≈ 0.17562 nA (E23→E4).

Neuron: exact propagator integration (Rotter & Diesmann 1999) with dynamic dt,
  so the same code works for fixed-step and event-driven simulation.
  Background input (nu_ext=8 Hz × K_cortical) approximated as DC via V_rest shift;
  full spec requires explicit Poisson background generators.

Pending framework support: per-synapse delay routing in propagation kernel.
"""

_J_E = 0.08781  # nA — excitatory PSC amplitude
_J_I = -0.35124  # nA — inhibitory PSC amplitude

description = {
    "populations": {
        "name": ["E_23", "I_23", "E_4", "I_4", "E_5", "I_5", "E_6", "I_6"],
        "K_cortical": [1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100],
        "size": [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948],
        "V_mean": [-68.28, -63.16, -63.33, -63.45, -63.11, -61.66, -66.72, -61.45],
        "V_std": [5.36, 4.57, 4.74, 4.94, 4.94, 4.55, 5.46, 4.48],
    },
    "T": {"size": 902, "rate": 0.12, "start": 700.0, "stop": 710.0},
    "connections": {
        # Source populations (rows); targets are the 8 cortical populations (columns).
        "populations": ["E_23", "I_23", "E_4", "I_4", "E_5", "I_5", "E_6", "I_6", "T"],
        "weight_mean": [_J_E, _J_I, _J_E, _J_I, _J_E, _J_I, _J_E, _J_I, _J_E],
        "weight_std": [
            abs(w) * 0.1 for w in [_J_E, _J_I, _J_E, _J_I, _J_E, _J_I, _J_E, _J_I, _J_E]
        ],
        # Excitatory mean delay 1.5 ms, inhibitory 0.75 ms; std = 0.5 * mean (spec Table 6)
        "delay_mean": [1.5, 0.75, 1.5, 0.75, 1.5, 0.75, 1.5, 0.75, 1.5],
        "delay_std": [
            d * 0.5 for d in [1.5, 0.75, 1.5, 0.75, 1.5, 0.75, 1.5, 0.75, 1.5]
        ],
        # C_yx connection probabilities: rows = source, columns = target (E23,I23,E4,I4,E5,I5,E6,I6)
        "probabilities": [
            [0.1009, 0.1346, 0.0077, 0.0691, 0.1004, 0.0548, 0.0156, 0.0364],  # E23
            [0.1689, 0.1371, 0.0059, 0.0029, 0.0622, 0.0269, 0.0066, 0.0010],  # I23
            [0.0437, 0.0316, 0.0497, 0.0794, 0.0505, 0.0257, 0.0211, 0.0034],  # E4
            [0.0818, 0.0515, 0.1350, 0.1597, 0.0057, 0.0022, 0.0166, 0.0005],  # I4
            [0.0323, 0.0755, 0.0067, 0.0033, 0.0831, 0.0600, 0.0572, 0.0277],  # E5
            [0.0, 0.0, 0.0003, 0.0, 0.3726, 0.3158, 0.0197, 0.0080],  # I5
            [0.0076, 0.0042, 0.0453, 0.1057, 0.0204, 0.0086, 0.0396, 0.0658],  # E6
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2252, 0.1443],  # I6
            [0.0, 0.0, 0.0983, 0.0619, 0.0, 0.0, 0.0512, 0.0196],  # T
        ],
    },
}


def build(desc: dict = description):
    populations = {}

    for name, K_cortical, size, V_mean, V_std in zip(*desc["populations"].values()):
        populations[name] = PD14_neuron(K_cortical=K_cortical) * size
        populations[name].potential = torch.normal(V_mean, V_std, (size,))

    populations["T"] = (
        PD14_poisson(desc["T"]["rate"], desc["T"]["start"], desc["T"]["stop"])
        * desc["T"]["size"]
    )
    # T has no incoming connections and update() is a no-op -- send()'s own
    # clock (next_spike) is the only thing driving it, so nothing else would
    # ever make it due (see Network._validate_schedulable). 0.1ms matches
    # the literature's accurate clock-driven comparison timestep for PD14 --
    # and since no other population here declares its own explicit
    # timestep, this is currently also the sole thing setting the whole
    # network's cadence (Network._sub_dt takes the min across every
    # population's declared constraint; every PD14_neuron population is
    # autonomous with no timestep of its own, so none of them constrain it
    # -- they just run every sub-step, whatever that ends up being).
    populations["T"].timestep = 0.1

    for source, weight_mean, weight_std, delay_mean, delay_std, probs in zip(
        desc["connections"]["populations"],
        desc["connections"]["weight_mean"],
        desc["connections"]["weight_std"],
        desc["connections"]["delay_mean"],
        desc["connections"]["delay_std"],
        desc["connections"]["probabilities"],
    ):
        for target, prob in zip(desc["populations"]["name"], probs):
            if prob == 0.0:
                continue
            w_mean, w_std = (
                (2 * _J_E, abs(2 * _J_E) * 0.1)
                if source == "E_23" and target == "E_4"
                else (weight_mean, weight_std)
            )
            conn = populations[source] >> PD14_connection(prob) >> populations[target]
            conn.delay = torch.normal(delay_mean, delay_std, conn.shape).clamp(min=0.1)
            weights = torch.normal(w_mean, w_std, conn.shape)
            conn.weight = weights.clamp(min=0) if w_mean >= 0 else weights.clamp(max=0)

    return populations["T"].network
