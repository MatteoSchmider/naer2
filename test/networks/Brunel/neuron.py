import math

import naer2 as n2


@n2.node
class Brunel_neuron:
    """Brunel (2000)'s leaky integrate-and-fire neuron with delta synapses, event-driven and exact.

    The potential V (mV, relative to rest) only changes at an input's arrival: receive decays it over the time
    since the last change, V exp(-dt / tau), and adds the PSP jump J (the link's weight) -- exact, no time
    grid. Crossing the threshold theta, it fires right there, at the input's arrival (n2.spike inside
    delivery, D-3), resets to V_r and stays there for the refractory period t_ref: inputs arriving inside it
    are dropped (NEST's iaf_psc_delta); the decay restarts at its end (last_time).
    """

    def __init__(self, tau: float = 20.0, theta: float = 20.0, V_r: float = 10.0, t_ref: float = 2.0):
        self.tau = tau
        self.theta = theta
        self.V_r = V_r
        self.t_ref = t_ref
        self.V: float = 0.0
        self.last_time: float = 0.0  # where V was last exact: the last input, or a refractory period's end
        self.t_ref_end: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        if time >= self.t_ref_end:
            self.V = self.V * math.exp((self.last_time - time) / self.tau) + edge.weight
            self.last_time = time
            if self.V >= self.theta:
                n2.spike(time)
                self.V = self.V_r
                self.t_ref_end = time + self.t_ref
                self.last_time = time + self.t_ref
