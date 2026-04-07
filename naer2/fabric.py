from .edges import edge_types

# TODO: implement fabric just like population
# TODO: maybe add pattern? Find out why/why not
# TODO: optimize code of Population and fabric to have less duplication, not none but we can probably outsource some things both here and there


class Fabric:
    def __init__(self, instance):
        if instance.__class__ not in edge_types.keys():
            raise ValueError("edge needs to be decorated by the @naer.edge decorator")
        self.instance = instance
        self.edge = edge_types[instance.__class__]
