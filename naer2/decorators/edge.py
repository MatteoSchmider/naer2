from collections.abc import Callable

from .._partial import _PartialConnection
from ..edges import edge_types
from ..receives import destination, source, time
from ..types import Hook

# Allowed annotation types for edge forward/backward hooks.
# A hook may annotate any subset; unannotated optional params are also permitted.
_EDGE_RECEIVE_TYPES = {time, source, destination}


def edge(
    forward: str = "forward",
    backward: str = "backward",
) -> Callable[[type], type]:
    """
    Class decorator to define an edge (synapse model).

    forward/backward hooks may annotate any subset of (time, source, destination).
    A static synapse with no dynamics can have hooks with no parameters at all.

    Usage requires parentheses: @n2.edge()
    """

    def wrapper(cls):
        return edge_decorator(cls, forward, backward)

    return wrapper


def edge_decorator(cls: type, forward_name: str, backward_name: str) -> type:
    if not isinstance(cls, type):
        raise ValueError(
            f"edge decorator can only be applied to types (got {type(cls)}) in decorator for {cls.__name__}"
        )

    edge_types[cls] = {
        "forward": Hook("forward", getattr(cls, str(forward_name))).verify(_EDGE_RECEIVE_TYPES),
        "backward": Hook("backward", getattr(cls, str(backward_name))).verify(_EDGE_RECEIVE_TYPES),
    }

    # edge_instance @ pattern → _PartialConnection, consumed by >> on a population
    cls.__matmul__ = lambda self, other: _PartialConnection(self, other)

    return cls
