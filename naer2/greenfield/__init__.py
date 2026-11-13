"""
Green-field kernels of the spike-management redesign (see TODO.md): written
by hand outside population.py and the transpiler, to pin down the contracts
and the performance before they go back into the library -- one fused
kernel per population and run (fused.py), its emission history
(history.py) and rings of arrival slots (ring.py).
"""

from .fused import FUSED_TRACE_TAGS, WALK, FusedPop, fused_kernel
from .history import EmissionHistory
from .ring import Ring

__all__ = ["FusedPop", "fused_kernel", "FUSED_TRACE_TAGS", "WALK", "EmissionHistory", "Ring"]
