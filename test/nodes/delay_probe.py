import naer2 as n2


@n2.node
class delay_probe:
    """Event-driven (update() is a no-op -- needs an explicit
    Population.timestep): accumulates edge.weight on receive, fires if
    that crosses threshold. With weight in {0, 1} and threshold=0.5, a
    weight-0 link can never cross threshold -- only weight-1 links fire,
    recording the exact (bucketed) arrival time their delay landed them
    in. See TODO.md / test_per_link_delay_kernel.py."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.potential: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        if self.potential > self.threshold:
            n2.spike(time)  # explicit: bare n2.spike() always records t, not this delivery's time
            self.potential = 0.0  # reset, else Part 2's unconditional final pass fires again at t

    def receive(self, time: n2.receives.time, edge: n2.receives.edge, source: n2.receives.source):
        self.potential += edge.weight
