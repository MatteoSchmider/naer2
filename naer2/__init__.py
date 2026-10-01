from .decorators import *
from .population import Population
from .transpiler import *
from .types import *
from . import receives, updates, sends
from .receives import time
from .sends import spike

# Per-kernel-call output spike buffer capacity, as a multiple of population size:
# n_neurons * BUFFER_MULTIPLIER slots are allocated for (node_id, timestamp)
# records. Spike counts can't be known before running a simulation, and the
# neuron count is generally far smaller than the synapse count, so sizing off
# a flat per-neuron multiple is cheap; there is no overflow check yet if a
# kernel call produces more spikes per neuron than this in one call.
BUFFER_MULTIPLIER = 100

__all__ = types.__all__ + [
    "edge",
    "node",
    "pattern",
    "Population",
    "receives",
    "updates",
    "sends",
    "time",
    "spike",
    "BUFFER_MULTIPLIER",
]
