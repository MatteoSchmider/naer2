from .class_transpiler import ClassTranspiler

node_types = {}


# decorators for node hooks
def send(func):
    return func


def receive(func):
    return func


def update(func):
    return func


# decorator for node class
def node(cls):
    if not isinstance(cls, type):
        raise ValueError("The node decorator can only be applied to types")
    node_types[cls] = ClassTranspiler(cls)
    return cls
