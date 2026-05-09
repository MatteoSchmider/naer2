import naer2 as n2


@n2.edge()
class stdp_synapse:
    """
    Spike-timing-dependent plasticity synapse.

    Weight updates are applied lazily: LTP is computed against the pre-synaptic
    neuron's second-to-last spike (carried in the spike metadata), and LTD is
    computed from an exponential post-synaptic trace maintained per neuron.
    This defers the race condition on concurrent pre/post spike events to the
    next pre-synaptic spike, at which point the weight is actually loaded
    anyway — so the update is free.

    The neuron type paired with this synapse must expose (or have injected at
    JIT time) a spike history buffer for LTP and a stdp_trace for LTD.
    """

    def __init__(self, weight: float):
        self.weight = weight

    def forward(
        self,
        t: n2.receives.time,
        src: n2.receives.source,
        dst: n2.receives.destination,
    ):
        # TODO: implement lazy STDP
        # LTP: stdp_LTP(self.weight, src.pre_trace, src.previous_time, t)
        # LTD: stdp_LTD(self.weight, dst.stdp_trace)
        pass

    def backward(
        self,
        t: n2.receives.time,
        src: n2.receives.source,
        dst: n2.receives.destination,
    ):
        # Backward pass is forbidden from modifying self attributes to avoid
        # requiring a reverse spike search (which would break event-driven perf).
        pass
