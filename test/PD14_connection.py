import math

import naer2 as n2


@n2.connection
class PD14_connection:
    def __init__(self, probability: float):
        self.probability = probability

    def connect(self, source, target):
        if math.random() < self.probability:
            source >> target
