import naer2 as n2


@n2.node
class PD14_neuron:
    def __init__(
        self,
        K_cortical: float,
        t_mem: float = 10.0,     # ms
        t_syn: float = 0.5,      # ms
        t_ref: float = 2.0,      # ms
        R_mem: float = 40.0,     # MOhm
        E_leak: float = -65.0,   # mV
        V_thresh: float = -50.0, # mV
        V_reset: float = -65.0,  # mV
        I_cortical: float = 3e-4, # nA (= 0.3 pA per cortico-cortical input)
    ):
        self.t_mem = t_mem
        self.t_syn = t_syn
        self.t_ref = t_ref
        self.R_mem = R_mem
        self.E_leak = E_leak
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.I_cortical = I_cortical
        self.K_cortical = K_cortical
        self.potential: float = V_reset
        self.last_time: float = 0.0
        self.current: float = 0.0

    def update(self, time: n2.time):
        if time > self.last_time:
            self.current -= self.current / self.t_syn
            self.potential += (
                -(self.potential - self.E_leak)
                + self.R_mem * self.current
                + self.R_mem * self.I_cortical * self.K_cortical
            ) / self.t_mem

    def send(self, time: n2.time):
        if self.potential > self.V_thresh:
            self.potential = self.V_reset
            self.last_time = time + self.t_ref
            n2.spike()

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.current += edge.weight
