from typing import Set

from .connection import Connection
from .kernel import Kernel
from .population import Population


class Network:
    def __init__(self, population: Population):
        if not isinstance(population, Population):
            raise ValueError("population " + population + " is not a Population")
        self.populations: Set[Population] = set(population)
        self.connections: Set[Connection] = set()
        self.kernels: Set[Kernel] = set()

    def merge(self, other: "Network"):
        if not isinstance(other, "Network"):
            raise ValueError("other " + other + " is not a Network")
        for population in other.populations:
            population.network = self
        self.populations += other.populations
        self.connections += other.connections
        return self

    def __add__(self, other):
        return self.merge(other)

    def __iadd__(self, other):
        return self.merge(other)
