from .receives import time


def spike(time: float = None):
    """Marks a spike emission point inside a node's send or receive hook.

    Never actually called — IfTransformer recognizes this call syntactically
    and lowers it to compacted (node_id, timestamp) writes into the kernel's
    output spike buffer, gated by whatever branch condition it appears under.

    time=None (default, no argument) records the kernel's own timestep
    boundary `t`. Passing an expression instead records it verbatim as the
    spike's timestamp — e.g. n2.spike(time) records this hook invocation's own
    notion of time (the incoming spike's exact arrival time inside a per-link
    receive, or `t` again during the final pass), and a receive hook can add
    its own emission delay on top: n2.spike(time + self.emission_delay) --
    held back by the source's compact() until due. No validation is
    performed — a negative or otherwise nonsensical time is the caller's own
    responsibility.
    """


__all__ = ["time", "spike"]
