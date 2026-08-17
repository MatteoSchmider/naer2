import naer2 as n2


@n2.edge
class PD14_synapse:
    def __init__(self):
        self.weight: float = 0.0

    def forward(self):
        pass

    def backward(self):
        pass
