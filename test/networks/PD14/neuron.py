import math

import naer2 as n2


@n2.node
class PD14_neuron:
    def __init__(
        self,
        K_cortical: int,
        t_mem: float = 10.0,     # ms
        t_syn: float = 0.5,      # ms
        t_ref: float = 2.0,      # ms
        R_mem: float = 40.0,     # MOhm
        E_leak: float = -65.0,   # mV
        V_thresh: float = -50.0, # mV
        V_reset: float = -65.0,  # mV
        nu_ext: float = 8e-3,    # spikes/ms  (8 Hz background)
        J_ext: float = 0.08781,  # nA  (= J_E; background uses excitatory weight)
    ):
        # Propagator parameters (static — folded as constants in kernel)
        self.t_mem: float = t_mem
        self.t_syn: float = t_syn
        self.R_mem: float = R_mem
        self.t_ref: float = t_ref
        self.V_thresh: float = V_thresh
        self.V_reset: float = V_reset
        # Background Poisson modeled as DC: V_rest = E_L + R * K * nu_ext * J * tau_s
        self.V_rest: float = E_leak + R_mem * K_cortical * nu_ext * J_ext * t_syn

        # Dynamic state (per neuron, on GPU)
        self.potential: float = V_reset
        self.current: float = 0.0
        self.t_ref_end: float = 0.0
        self.last_time: float = 0.0

    def update(self, time: n2.time):
        dt = time - self.last_time
        self.last_time = time
        P_V = math.exp(-dt / self.t_mem)
        P_I = math.exp(-dt / self.t_syn)
        P_VI = self.R_mem * self.t_syn / (self.t_mem - self.t_syn) * (P_V - P_I)
        if time < self.t_ref_end:
            self.potential = self.V_reset
        else:
            self.potential = (
                self.V_rest
                + P_V * (self.potential - self.V_rest)
                + P_VI * self.current
            )
        self.current = P_I * self.current

    def send(self, time: n2.time):
        if self.potential > self.V_thresh:
            n2.spike()
            self.potential = self.V_reset
            self.t_ref_end = time + self.t_ref

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.current += edge.weight
