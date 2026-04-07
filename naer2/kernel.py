class Kernel:
    def __init__(self):
        self.target = None
        self.connections = []  # for sources, edges

    def compile(self):
        pass
        # need to combine:
        # - target, target attributes, target callbacks
        # - connections, for each:
        #   - source, source attributes
        #   - edge, edge attributes, edge callbacks
        # => what about tracking reverse spikes for edges, e.g. backward()?
        # => this class should be compiler-near, and provide a single call() callback for network
