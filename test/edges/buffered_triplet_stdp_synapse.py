import math

import naer2 as n2

BUFFER_SIZE = 10


@n2.edge()
class buffered_triplet_stdp_synapse:
    """
    IMPORTANT:
    Going from triplet to buffered can be generalized.
    Just make a replay buffer of events added to post and then replay all the backwards Hook code for all the times in that buffer since then
    Could also automate putting the right starting state into that buffer and managing it too
    PROBLEM:
    If src is modified, then backwards code must actually be run
    So a hierarchy emerges:
    only dst modified in backwards, then it can simply be run directly on spikes
    only dst and self modified, can be replayed for self in buffer, and dst writes may be done directly
    => analyze whether the granular transformation can just store a buffer of all variables used to modify self, then replay those stored updates
    if src is modified, then a full backwards code must be run everytime
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
        # own weight, dynamic
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

        # Last pre-spike before that at time t_pre:
        t_pre = src.buffer_start
        for i in range(BUFFER_SIZE):
            t_pre += src.buffer[i]

        # we had the last valid state at t = t_pre
        # so we need to do all of the updates from post spikes since then, then do the update of the current pre spike

        o_1 = dst.o_1
        o_2 = dst.o_2
        t_post = dst.buffer_start - self.epsilon
        for i in range(BUFFER_SIZE):
            dt = dst.buffer[i]
            t_post += dt
            o_1 = math.exp(-dt / self.t_minus) * o_1 + 1
            o_2 = math.exp(-dt / self.t_y) * o_2 + 1
            if t_post > t_pre:
                r_1 = math.exp(-(t_post - t_pre) / self.t_plus) * src.r_1 + 1
                self.weight += r_1 * (self.A_2_plus + self.A_3_plus * o_2)

        # Last pre-spike before that at: src.previous_time
        dt_pre = t - t_pre - self.epsilon
        # -> update presynaptic traces:
        src.r_1 = math.exp(-dt_pre / self.t_plus) * src.r_1 + 1
        src.r_2 = math.exp(-dt_pre / self.t_x) * src.r_2 + 1
        # Last post-spike before that at: dst.previous_time
        # -> update weight:
        self.weight -= o_1 * (self.A_2_minus + self.A_3_minus * src.r_2)

    def backward(
        self,
        t: n2.receives.time,
        src: n2.receives.source,
        dst: n2.receives.destination,
    ):
        dst.buffer_start += dst.buffer[0]
        t_previous = dst.buffer_start
        for i in range(BUFFER_SIZE - 1):
            dt = dst.buffer[i + 1]
            t_previous += dt
            dst.buffer[i] = dt
        dst.buffer[BUFFER_SIZE - 1] = t - t_previous
