from dataclasses import dataclass

from .types import Hook


@dataclass
class Node:
    send: Hook
    update: Hook
    receive: dict[str, Hook]


node_types: dict[type, Node] = {}
