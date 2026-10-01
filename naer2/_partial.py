class _PartialConnection:
    """
    Intermediate result of `edge_instance @ ...` before a source population is known.

    The @ operator on a registered edge instance starts a chain that accumulates,
    in any order, the pieces needed to finish wiring a connection:

        pop_a >> static_synapse(0.5) @ all_to_all() >> pop_b
        pop_a >> static_synapse(0.5) @ "port" @ {"weight": "w"} @ all_to_all() >> pop_b
        pop_a >> static_synapse(0.5) @ all_to_all() @ "port" @ {"weight": "w"} >> pop_b

    all mean the same thing — order does not matter. Dispatch is by type:
      str            → target port name
      dict           → fabric_map (edge attr → hook-facing name); the edge is
                        always already known at this point, so a mapping here
                        always renames edge attributes, never source attributes
      anything else  → pattern

    ConnectionBuilder.connect() detects _PartialConnection and pulls fabric,
    pattern, port and fabric_map out of it in one step.
    """

    def __init__(self, edge, pattern=None):
        self.edge = edge
        self.pattern = pattern
        self.port: str | None = None
        self.fabric_map: dict[str, str] | None = None

    def __matmul__(self, other) -> "_PartialConnection":
        if isinstance(other, str):
            if self.port is not None:
                raise ValueError(f"target port already set to {self.port!r}")
            self.port = other
            return self
        if isinstance(other, dict):
            self.fabric_map = {**(self.fabric_map or {}), **other}
            return self
        if self.pattern is not None:
            raise ValueError(f"pattern already set to {self.pattern!r}")
        self.pattern = other
        return self
