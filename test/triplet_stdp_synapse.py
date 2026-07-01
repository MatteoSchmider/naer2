import math

import naer2 as n2


@n2.edge()
class triplet_stdp_synapse:
    """
    Spike-timing-dependent plasticity synapse.

    Weight updates are applied lazily: LTP and LTD are computed against the pre-synaptic
    neuron's second-to-last spike (carried in the spike metadata), and from an exponential post-synaptic trace maintained per neuron.
    This defers the race condition on concurrent pre/post spike events to the
    next pre-synaptic spike, at which point the weight is actually loaded
    anyway — so the update is free.

    The neuron type paired with this synapse must expose (or have injected at
    JIT time) a spike history buffer for LTP and a stdp_trace for LTD.

    For triplet STDP, the pre neurons carry two extra traces r_1 and r_2, and the post neuron the two traces o_1 and o_2.
    These traces step-increase for spikes at their respective neurons and decay exponentially with individual constants t_plus, t_x and t_minus, t_y.
    For the weight update, two rules are given:
    pre spike arrival: w(t) = w(t) - o_1(t) * (A_2_minus + A_three_minus * r_2(t - epsilon)), t = t_pre
    pre spike arrival: w(t) = w(t) + r_1(t) * (A_2_plus + A_three_plus * o_2(t - epsilon)), t = t_post
    """

    def __init__(
        self,
        weight: float,
        t_plus: float,
        t_x: float,
        t_minus: float,
        t_y: float,
        A_2_plus: float,
        A_2_minus: float,
        A_3_plus: float,
        A_3_minus: float,
        epsilon: float,
    ):
        # dynamic
        self.weight = weight
        # static
        self.t_plus = t_plus
        self.t_x = t_x
        self.t_minus = t_minus
        self.t_y = t_y
        self.A_2_plus = A_2_plus
        self.A_2_minus = A_2_minus
        self.A_3_plus = A_3_plus
        self.A_3_minus = A_3_minus
        self.epsilon = epsilon

    def forward(
        self,
        t: n2.receives.time,
        src: n2.receives.source,
        dst: n2.receives.destination,
    ):
        # Spike at presynaptic neuron
        # Last pre-spike before that at: src.previous_time
        dt_pre = t - src.previous_time - self.epsilon
        # -> update presynaptic traces:
        src.r_1 = math.exp(-dt_pre / self.t_plus) * src.r_1 + 1
        src.r_2 = math.exp(-dt_pre / self.t_x) * src.r_2 + 1
        # Last post-spike before that at: dst.previous_time
        dt_post = t - dst.previous_time
        # -> update postsynaptic traces:
        o_1 = math.exp(-dt_post / self.t_minus) * dst.o_1
        # -> update weight:
        self.weight -= o_1 * (self.A_2_minus + self.A_3_minus * src.r_2)
        src.previous_time = t

    def backward(
        self,
        t: n2.receives.time,
        src: n2.receives.source,
        dst: n2.receives.destination,
    ):
        # Spike at postsynaptic neuron
        # Last post-spike before that at: dst.previous_time
        dt_post = t - dst.previous_time - self.epsilon
        # -> update postsynaptic traces:
        dst.o_1 = math.exp(-dt_post / self.t_minus) * dst.o_1 + 1
        dst.o_2 = math.exp(-dt_post / self.t_y) * dst.o_2 + 1
        # Last pre-spike before that at: src.previous_time
        dt_pre = t - src.previous_time
        # -> update presynaptic traces:
        r_1 = math.exp(-dt_pre / self.t_plus) * src.r_1
        # -> update weight:
        self.weight += r_1 * (self.A_2_plus + self.A_3_plus * dst.o_2)
        dst.previous_time = t
