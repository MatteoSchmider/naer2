import naer2 as n2


@n2.node(send="send", update="update", receive="receive")
class ed_relay:
    """Purely event-driven: update() does nothing, so
    Population.has_autonomous_dynamics is False -- this population's own
    re-check cadence comes entirely from incoming delay (or an explicit
    Population.timestep), never from an "every sub-step" default. receive
    just counts deliveries; send is a no-op (this node never fires on its
    own, only in response to what it receives, and nothing here even does
    that -- it exists purely to exercise scheduling, see
    test_population_timestep.py / test_mixed_scheduling.py)."""

    def __init__(self):
        self.count: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        pass

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.count += 1.0
