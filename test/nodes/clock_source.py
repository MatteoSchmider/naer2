import naer2 as n2


@n2.node(send="send", update="update", receive="receive")
class clock_source:
    """Autonomous (update() assigns real state -> has_autonomous_dynamics
    True) and fires unconditionally on every call -- a deterministic,
    repeating signal for scheduling tests that need more than one spike to
    show a cadence (see test_mixed_scheduling.py). counter doesn't serve
    this: it only fires once (crosses threshold, resets to 0, nothing
    refills it, never crosses again)."""

    def __init__(self):
        self.tick: float = 0.0

    def update(self, time: n2.time):
        self.tick += 1.0

    def send(self, time: n2.time):
        n2.spike()

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        pass
