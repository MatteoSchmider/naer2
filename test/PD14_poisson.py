import math
import random

import naer2 as n2


@n2.node(receive=None)
class PD14_poisson:
    def __init__(
        self,
        rate: float,  # spikes/ms  (e.g. 0.12 ≈ 120 Hz)
        t_start: float,  # ms
        t_stop: float,  # ms
    ):
        self.rate = rate
        self.t_start = t_start
        self.t_stop = t_stop
        self.next_spike: float = t_start

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        if self.t_start <= time <= self.t_stop:
            if self.next_spike <= time:
                n2.spike()
                self.next_spike += -math.log(random.random()) / self.rate
