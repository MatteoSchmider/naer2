import naer2 as n2


@n2.edge()
class static_synapse:
    """
    A static synapse with a fixed weight — no learning rule.

    The receiving neuron accesses edge.weight directly in its receive hook.
    forward/backward have no parameters because a static synapse does not need
    to observe spike timing or neuron state; the weight is constant.
    """

    def __init__(self, weight: float):
        self.weight = weight

    def forward(self):
        pass

    def backward(self):
        pass
