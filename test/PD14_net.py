import torch
from PD14_connection import PD14_connection
from PD14_neuron import PD14_neuron
from PD14_poisson import PD14_poisson
from PD14_synapse import PD14_synapse

"""
units work like this:
1 mV = 1e-3 V; 1 nA = 1e-9 A; 1 MOhms = 1e6 Ohms
1 nA * 1 MOhms = 1e-9 A * 1e6 Ohms = 1e-3 V = 1 mV
times and time constants are in ms
"""

description = {
    "populations": {
        "name": ["E_23", "I_23", "E_4", "I_4", "E_5", "I_5", "E_6", "I_6"],
        "K_cortical": [1600, 1500, 2100, 1900, 2000, 1600, 2000, 1600],
        "size": [20683, 5834, 21915, 5479, 4850, 4850, 4850, 4850],
        "V_mean": [-68.28, -63.16, -68.28, -63.16, -68.28, -63.16, -68.28, -63.16],
        "V_std": [5.36, 4.57, 5.36, 4.57, 5.36, 4.57, 5.36, 4.57],
    },
    "T": {"size": 902, "rate": 0.12, "start": 700, "end": 710},
    "connections": {
        "populations": [
            "E_23",
            "I_23",
            "E_4",
            "I_4",
            "E_5",
            "I_5",
            "E_6",
            "I_6",
            "T",
        ],
        "weight_mean": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "weight_std": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "delay": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "probabilities": [
            [0.1009, 0.1346, 0.0077, 0.0691, 0.1004, 0.0548, 0.0156, 0.0364],
            [0.1689, 0.1371, 0.0059, 0.0029, 0.0622, 0.0269, 0.0066, 0.0010],
            [0.0437, 0.0316, 0.0497, 0.0794, 0.0505, 0.0257, 0.0211, 0.0034],
            [0.0818, 0.0515, 0.1350, 0.1597, 0.0057, 0.0022, 0.0166, 0.0005],
            [0.0323, 0.0755, 0.0067, 0.0033, 0.0831, 0.0600, 0.0572, 0.0277],
            [0.0, 0.0, 0.0003, 0.0, 0.3726, 0.3158, 0.0197, 0.0080],
            [0.0076, 0.0042, 0.0453, 0.1057, 0.0204, 0.0086, 0.0396, 0.0658],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2252, 0.1443],
            [0.0, 0.0, 0.0983, 0.0619, 0.0, 0.0, 0.0512, 0.0196],
        ],
    },
}


def build(description: dict = description):
    populations = {}
    for name, K_cortical, size, V_mean, V_std in zip(
        description["populations"].values()
    ):
        populations[name] = PD14_neuron(K_cortical) * size
        populations[name].potential = torch.normal(V_mean, V_std, size=(size,))

    populations["T"] = (
        PD14_poisson(
            description["T"]["rate"],
            description["T"]["start"],
            description["T"]["end"],
        )
        * description["T"]["size"]
    )

    for source, weight_mean, weight_std, delay, probabilities in zip(
        description["connections"].values()
    ):
        for target, probability in zip(
            description["connections"]["populations"], probabilities
        ):
            conn = (
                populations[source]
                >> PD14_synapse() @ PD14_connection(probability)
                >> populations[target]
            )
            conn.weight = torch.normal(weight_mean, weight_std, size=(conn.size,))
            conn.delay = torch.normal(delay, delay * 0.5, size=(conn.size,))

    return populations["T"].network
