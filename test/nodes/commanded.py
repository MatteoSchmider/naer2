import naer2 as n2


@n2.node
class commanded:
    """Fires when told to: the host sets fire_now (1.0) before a run, send
    spikes (bare: at the window's start) and consumes the command. Receive
    adds up its links' weights. The stand-in source and consumer of the
    fused kernel's tests (test/test_fused_gpu.py)."""

    def __init__(self):
        self.fire_now: float = 0.0
        self.current: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        if self.fire_now > 0.5:
            n2.spike()
            self.fire_now = 0.0

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.current += edge.weight


@n2.node
class commanded_timed:
    """commanded, emitting at a given time: the host sets fire_now and
    fire_e (the emission time -- e.g. a fire time plus an emission delay)
    before a run; send spikes at fire_e and consumes both."""

    def __init__(self):
        self.fire_now: float = 0.0
        self.fire_e: float = 0.0
        self.current: float = 0.0

    def update(self, time: n2.time):
        pass

    def send(self, time: n2.time):
        if self.fire_now > 0.5:
            n2.spike(self.fire_e)
            self.fire_now = 0.0
            self.fire_e = 0.0

    def receive(self, time: n2.time, edge: n2.receives.edge, source: n2.receives.source):
        self.current += edge.weight
