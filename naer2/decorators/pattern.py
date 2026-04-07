import inspect
from collections.abc import Callable

import triton.language as tl

from ..patterns import pattern_types
from ..types import Hook, Position, Region


def pattern(
    cb: Callable[[Position, Region], Region],
) -> Callable[[Position, Region], Region]:
    """Callable to define a connection pattern between two populations"""

    if not isinstance(cb, type) and not isinstance(cb, Callable):
        raise ValueError(
            f"pattern decorator can only be applied to Callables (got {type(cls)}) in decorator for {cls.__name__}"
        )

    pattern_types[cb] = Hook("pattern", cb).verify({Position, Region})

    # TODO: implement connection builder dunders
    # cls.__mul__ = lambda self, amount: Population(self, amount)

    return cb
