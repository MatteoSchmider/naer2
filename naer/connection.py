from .class_transpiler import ClassTranspiler

connection_types = {}


def Connection(cls):
    if not isinstance(cls, type):
        raise ValueError("The connection decorator can only be applied to types")
    connection_types[cls] = ClassTranspiler(cls)
    return cls

class ConnectionPattern():
    pass