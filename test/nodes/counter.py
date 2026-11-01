import naer2 as n2


@n2.node(send="send", update="update", receive="receive")
class counter:
    """
    A deliberately trivial node for exercising a compiled kernel on real GPU
    hardware (see TODO.md's staged GPU verification). Every quantity is a
    small integer, so the spike count/timing produced from a given starting
    potential can be worked out by hand instead of reasoned about numerically
    -- unlike test/networks/coverage's relay, which exists to exercise AST
    shapes, not numeric behavior.

    update:  potential -= 1, unconditional.
    receive: potential += 10, flat -- never reads edge.weight, so a
             connection using this node needs no real weight tensor.
    send:    three thresholds 10/20/30 apiece 10 apart, each firing with a
             different delay (0/+10/+20) also 10 apart, so which branch fired
             is readable straight off the spike time, not just the count.
    """

    def __init__(self, threshold: float = 10.0):
        self.threshold = threshold
        self.potential: float = 0.0

    def update(self, time: n2.time):
        self.potential -= 1

    def send(self, time: n2.time):
        if self.potential > 3 * self.threshold:
            n2.spike(time + 20.0)
            self.potential = 0
        elif self.potential > 2 * self.threshold:
            n2.spike(time + 10.0)
            self.potential = 0
        elif self.potential > self.threshold:
            n2.spike()
            self.potential = 0

    def receive(
        self, time: n2.receives.time, edge: n2.receives.edge, source: n2.receives.source
    ):
        self.potential += 10
