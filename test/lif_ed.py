import math

import naer2 as n2


@n2.node(send="send", receive="receive", update="update")
class lif_ed:
    def __init__(self, v_thresh, v_reset, t_mem, t_ref):
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.t_mem = t_mem
        self.t_ref = t_ref
        self.potential: float = v_reset
        self.last_time: int = 0

    def update(self):
        pass

    def send(self):
        if self.potential > self.v_thresh:
            self.potential = self.v_reset
            self.last_time += self.t_ref
            n2.spike()

    def receive(self, time: n2.receives.time, edge: n2.receives.edge, source: n2.receives.source):
        dt = time - self.last_time
        if dt < 0:
            return
        self.last_time = time
        self.potential *= math.exp(-dt / self.t_mem)
        self.potential += edge.weight
