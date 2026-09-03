import naer2 as n2


@n2.node(
    send="send",
    update="update",
    receive={
        "positive": "pos_receive",
        "negative": "neg_receive",
    },
)
class counter:
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

    def send(self, time: n2.time):
        if self.potential > self.threshold:
            n2.spike()
            self.potential = 0

    def pos_receive(
        self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source
    ):
        self.potential += edge.w

    def neg_receive(
        self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source
    ):
        self.potential -= edge.w
