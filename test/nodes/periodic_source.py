import naer2 as n2


@n2.node(receive=None)
class periodic_source:
    """An update()-only source with precise spike times and no timestep of its own -- naer2 derives one, the
    smallest delay of its outgoing connections (Population._own_timestep): every neuron fires at next_spike,
    then every period after; at most one spike a run (period >= its window), at its exact time."""

    def __init__(self, period: float = 1.0):
        self.period = period
        self.next_spike: float = 0.0

    def update(self, time: n2.time):
        if self.next_spike < time:
            n2.spike(self.next_spike)
            self.next_spike += self.period

    def send(self, time: n2.time):
        pass
