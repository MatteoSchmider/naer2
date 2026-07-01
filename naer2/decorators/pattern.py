from ..patterns import pattern_types
from ..types import Hook, Position, Region


def pattern(cb):
    """
    Decorator to register a callable or callable class as a connection pattern.

    A pattern defines connectivity geometry per source neuron:
        (position: Position, source_region: Region, target_region: Region = None) -> Region

    Supports both plain functions and callable classes (for parameterized patterns):
        @n2.pattern
        def all_to_all(pos, src, tgt=None): ...

        @n2.pattern
        class random_conn:
            def __init__(self, probability): ...
            def __call__(self, pos, src, tgt=None): ...
    """
    if not callable(cb):
        raise ValueError(
            f"pattern decorator can only be applied to callables (got {type(cb).__name__})"
        )

    if isinstance(cb, type):
        if not callable(cb):
            raise ValueError(f"pattern class must be callable (missing __call__)")
        pattern_types[cb] = Hook("pattern", cb.__call__).verify({Position, Region})
    else:
        pattern_types[cb] = Hook("pattern", cb).verify({Position, Region})

    return cb
