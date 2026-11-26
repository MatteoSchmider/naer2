import naer2 as n2

from .neuron import PD14_neuron


@n2.node
class PD14_commanded(PD14_neuron):
    """PD14_neuron firing on command instead of at threshold: the host sets
    fire_now (1.0) before a run, send spikes and resets as PD14_neuron's
    does, and consumes the command. Controlled load for the benchmarks
    (tools/pd14_setup.py): PD14's update and receive, PD14-like rates
    chosen by the host."""

    def __init__(self, K_cortical: int, **kwargs):
        super().__init__(K_cortical, **kwargs)
        self.fire_now: float = 0.0

    def send(self, time: n2.time):
        if self.fire_now > 0.5:
            n2.spike()
            self.potential = self.V_reset
            self.t_ref_end = time + self.t_ref
            self.fire_now = 0.0
