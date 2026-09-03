import naer2 as n2


@n2.edge()
class plastic_synapse:
    def __init__(self, weight: float, factor: float):
        self.weight = weight
        self.factor = factor

    def forward(self):
        self.weight += self.factor

    def backward(self):
        pass
