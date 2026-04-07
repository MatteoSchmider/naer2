import naer2 as np

# TODO: standardize t_mem, implement t_ref here, use same differential equation for both lif types


@np.node(
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

    def send(self):
        if self.potential > self.v_thresh:
            self.potential = self.v_reset
            np.spike()

    def receive_excitatory(
        self, time: np.time, connection: np.connection, source: np.source
    ):
        self.potential += 1.0

    def receive_inhibitory(
        self, time: np.time, connection: np.connection, source: np.source
    ):
        self.potential += 1.0

    def update(self):
        self.potential *= self.t_mem
