from __future__ import annotations
from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    from .connection import Connection
    from .population import Population

from .kernel import Kernel


class Network:
    def __init__(self, population: Population):
        self.populations: Set[Population] = {population}
        self.connections: Set[Connection] = set()
        self.kernels: Set[Kernel] = set()

    def merge(self, other: Network):
        for population in other.populations:
            population.network = self
        self.populations.update(other.populations)
        self.connections.update(other.connections)
        return self

    def __add__(self, other):
        return self.merge(other)

    def __iadd__(self, other):
        return self.merge(other)
