"""
The fused kernel of the spike-management redesign (see TODO.md 1e): one
kernel per population and run, written by hand outside population.py and the
transpiler to pin down the contracts and the performance -- the reference the
library's kernel template is built from. Its data structures already live in
the library: naer2/history.py (emission history), naer2/ring.py (rings of
arrival slots), naer2/topology.py (SimulationTopology).
"""

from .fused import FUSED_TRACE_TAGS, WALK, FusedPop, fused_kernel

__all__ = ["FusedPop", "fused_kernel", "FUSED_TRACE_TAGS", "WALK"]
