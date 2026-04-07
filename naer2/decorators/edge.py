from collections.abc import Callable

from ..edges import edge_types
from ..types import Hook


def edge(
    forward: str = "forward",
    backward: str = "backward",
) -> Callable[[type], type]:
    """Class decorator to define an edge with forward/backward hooks."""

    def wrapper(cls):
        return edge_decorator(cls, forward, backward)

    return wrapper


def edge_decorator(cls: type, forward_name: str, backward_name: str) -> type:
    if not isinstance(cls, type):
        raise ValueError(
            f"edge decorator can only be applied to types (got {type(cls)}) in decorator for {cls.__name__}"
        )

    edge_types[cls] = {
        "forward": Hook("forward", getattr(cls, str(forward_name))).verify(set()),
        "backward": Hook("backward", getattr(cls, str(backward_name))).verify(set()),
    }
    # TODO: implement connection builder dunders
    # cls.__mul__ = lambda self, amount: Population(self, amount)

    return cls
