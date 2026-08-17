import naer2 as n2

# TODO: standardize t_mem, implement t_ref here, use same differential equation for both lif types


@n2.node(send="send", receive="receive", update="update")
class lif_cd:
    def __init__(self, v_thresh, v_reset, t_mem):
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.t_mem = t_mem
        self.potential: float = v_reset

    def send(self):
        if self.potential > self.v_thresh:
            self.potential = self.v_reset
            n2.spike()

    def receive(
        self, time: n2.receives.time, edge: n2.receives.edge, source: n2.receives.source
    ):
        self.potential += edge.weight

    def update(self):
        self.potential *= self.t_mem
