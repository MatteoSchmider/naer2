from .edges import edge_types


class Fabric:
    def __init__(self, instance):
        if instance.__class__ not in edge_types.keys():
            raise ValueError("edge needs to be decorated by the @naer.edge decorator")
        self.instance = instance
        self.edge = edge_types[instance.__class__]

        self.dynamic_attrs: dict[str, type] = {}
        assigned_members = (
            self.edge["forward"].assigned_members | self.edge["backward"].assigned_members
        )
        for member in assigned_members:
            if member not in instance.__dict__:
                raise ValueError(f"Edge instance needs to have a member {member}")
            self.dynamic_attrs[member] = instance.__dict__[member].__class__

        self.static_attrs: dict[str, any] = {
            k: v for k, v in instance.__dict__.items()
            if k not in self.dynamic_attrs
        }
