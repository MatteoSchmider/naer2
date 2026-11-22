import math
import random

import naer2 as n2


@n2.node(receive=None)
class Brunel_poisson:
    """A Poisson spike source with precise spike times: each neuron's next spike (next_spike) an exponential
    interval after its last. It has update() only and no timestep: naer2 runs it every smallest delay of its
    outgoing connections (Brunel's D, Population._own_timestep), and a run emits every spike before its end. Its
    first interval drawn in its first run, from 0 (next_spike < 0: none yet)."""

    def __init__(self, rate: float):
        self.rate = rate  # spikes / ms
        self.next_spike: float = -1.0

    def update(self, time: n2.time):
        if self.next_spike < 0.0:
            self.next_spike = -math.log(random.random()) / self.rate
        while self.next_spike < time:
            n2.spike(self.next_spike)
            self.next_spike += -math.log(random.random()) / self.rate

    def send(self, time: n2.time):
        pass
