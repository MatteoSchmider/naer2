from typing import Any


class TranspilerContext:
    """
    Carries the static/dynamic attribute split for one population during transpilation.

    static_attrs  — attributes whose values never change during simulation.
                    Substituted as literal constants in the generated kernel,
                    eliminating per-neuron storage and a memory round-trip.
                    Determined at JIT time from the node instance.

    dynamic_attrs — attributes that are read or written during simulation.
                    Each becomes a tensor pointer argument to the kernel,
                    with tl.load/tl.store emitted around the hook body.
                    Determined by AssignedMemberFinder across all hooks.
    """

    def __init__(
        self,
        static_attrs: dict[str, Any],
        dynamic_attrs: dict[str, type],
    ):
        self.static_attrs = static_attrs
        self.dynamic_attrs = dynamic_attrs
