class TranspilerContext:
    def __init__(self, template):
        self.template = template
        self.static_attrs = {}
        self.dynamic_attrs = {}
