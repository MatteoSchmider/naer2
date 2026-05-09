class _PartialConnection:
    """
    Intermediate result of `edge_instance @ pattern` before a source population is known.

    The @ operator on a registered edge instance combines it with a pattern into
    a single object that can then be applied to a source population via >>:

        pop_a >> static_synapse(0.5) @ all_to_all >> pop_b

    evaluates as:

        pop_a >> (static_synapse(0.5) @ all_to_all) >> pop_b

    because @ has higher precedence than >>. ConnectionBuilder.connect() detects
    this type and unpacks fabric and pattern from it.
    """

    def __init__(self, edge, pattern):
        self.edge = edge
        self.pattern = pattern
