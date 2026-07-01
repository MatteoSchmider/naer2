import naer2 as n2


@n2.node
class PD14_poisson:
    def __init__(
        self,
        v_T: float,
        t_start: float,
        t_stop: float,
    ):
        self.v_T = v_T
        self.t_start = t_start
        self.t_stop = t_stop

    def update(self, time: n2.updates.time):
        if self.t_stop > time > self.t_start:
            # TODO:
            # generate next spike based on Poisson process
            pass

    def send(self, time: n2.sends.time):
        if 0 <= self.next_spike <= time:
            n2.spike()
            # TODO:
            # generate next spike here
            pass
