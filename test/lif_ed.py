import naer2 as naer

# TODO: standardize t_mem, implement t_ref here, use same differential equation for both lif types


@naer.node(send="send", receive="receive")
class lif_ed:
    def __init__(self, v_thresh, v_reset, t_mem, t_ref):
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.t_mem = t_mem
        self.t_ref = t_ref
        self.potential: float = v_reset
        self.last_time: int = 0

    def send(self):
        if self.potential > self.v_thresh:
            self.potential = self.v_reset
            self.last_time += self.t_ref
            naer.spike()

    def receive(self, spike, synapse):
        dt = spike.time - self.last_time
        if dt < 0:
            return
        self.last_time = spike.time
        self.potential *= np.exp(-dt / self.t_mem)
        self.potential += synapse.weight
