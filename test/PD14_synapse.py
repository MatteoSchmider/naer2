import naer2 as n2


@n2.edge
class PD14_synapse:
    def __init__(self):
        self.delay: float = 0.0  # ms; per-synapse delay tensor set at connection time

    def forward(self):
        pass

    def backward(self):
        pass
