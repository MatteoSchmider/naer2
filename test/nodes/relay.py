import naer2 as n2


@n2.node(
    send="send",
    update="update",
    receive={
        "positive": "pos_receive",
        "negative": "neg_receive",
    },
)
class relay:
    def __init__(
        self,
        decay: float,
        threshold: float,
    ):
        self.threshold = threshold
        self.decay = decay
        self.potential: float = 0.0

    def update(self, time: n2.time):
        self.potential *= self.decay
        # Coverage only: spike emission from update. The threshold is
        # nonsensical on purpose so this never fires.
        if self.potential < -1000000.0:
            n2.spike()
            self.potential = 0

    def send(self, time: n2.time):
        # send is the single source of truth for firing. Two cases: far past
        # threshold, an extra spike is emitted a fixed delay later (covering
        # n2.spike's explicit time argument); otherwise a plain spike.
        if self.potential > 2 * self.threshold:
            n2.spike()
            n2.spike(time + 1.0)
            self.potential = 0
        elif self.potential > self.threshold:
            n2.spike()
            self.potential = 0

    def pos_receive(
        self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source
    ):
        self.potential += edge.w
        # Coverage only: spike emission from receive. The threshold is
        # nonsensical on purpose so this never fires.
        if self.potential > 1000000.0:
            n2.spike(time)

    def neg_receive(
        self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source
    ):
        self.potential -= edge.w
        self.potential += source.src_pot * 0.1
