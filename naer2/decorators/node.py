from collections.abc import Callable

from ..connection import ConnectionBuilder
from ..nodes import Node, node_types
from ..population import Population
from ..receives import edge, source, time
from ..types import Hook


def node(
    send: str = "send",
    receive: str | dict[str, str] = "receive",
    update: str = "update",
) -> Callable[[type], type]:
    """Class decorator to define a node with send/update/receive hooks."""

    def wrapper(cls):
        return node_decorator(cls, send, receive, update)

    return wrapper


def node_decorator(
    cls: type, send_name: str, receive_name: str | dict[str, str], update_name: str
) -> type:
    if not isinstance(cls, type):
        raise ValueError(
            f"node decorator can only be applied to types (got {type(cls)}) in decorator for {cls.__name__}"
        )

    match receive_name:
        case str():
            receive_funcs = {
                receive_name: Hook("receive", getattr(cls, receive_name)).verify(
                    {time, edge, source}
                )
            }
        case dict():
            receive_funcs = {
                str(port): Hook("receive", getattr(cls, str(name))).verify(
                    {time, edge, source}
                )
                for port, name in receive_name.items()
            }
        case _:
            raise ValueError(
                f"'receive' must be a string or dict[str, str] (got {type(receive_name).__name__}) in decorator for {cls.__name__}"
            )

    node_types[cls] = Node(
        send=Hook("send", getattr(cls, str(send_name))).verify(set()),
        update=Hook("update", getattr(cls, str(update_name))).verify(set()),
        receive=receive_funcs,
    )

    cls.__mul__ = lambda self, amount: Population(self, amount)
    cls.__rshift__ = lambda self, right: (
        ConnectionBuilder(Population(self, 1)) >> right
    )
    cls.__matmul__ = lambda self, right: (
        ConnectionBuilder(Population(self, 1)) @ right
    )

    return cls
