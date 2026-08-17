import naer2 as n2


@n2.node(
    send="send",
    receive={"excitatory": "receive_excitatory", "inhibitory": "receive_inhibitory"},
    update="update",
)
class lif_cd_weightless:
    def __init__(self, v_thresh, v_reset, t_mem):
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.t_mem = t_mem
        self.potential: float = v_reset

    def update(self):
        self.potential *= self.t_mem

    def send(self):
        if self.potential > self.v_thresh:
            self.potential = self.v_reset
            n2.spike()

    def receive_excitatory(self, time: n2.receives.time, edge: n2.receives.edge, source: n2.receives.source):
        self.potential += 1.0

    def receive_inhibitory(self, time: n2.receives.time, edge: n2.receives.edge, source: n2.receives.source):
        self.potential -= 1.0
