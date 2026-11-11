"""
Green-field kernels of the spike-management redesign (see TODO.md): written
by hand outside population.py and the transpiler, tested end to end against
the real intake (naer2/intake.py), to pin down the contracts and the
performance before they go back into the templates -- next as one fused
kernel per population and run.
"""

from .pop import WALK, Pop
from .update import UPDATE_TRACE_TAGS, update_kernel

__all__ = ["Pop", "WALK", "UPDATE_TRACE_TAGS", "update_kernel"]
