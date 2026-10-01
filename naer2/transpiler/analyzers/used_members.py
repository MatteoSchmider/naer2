import ast


class UsedMemberFinder(ast.NodeVisitor):
    """Collects every obj_name.attr reference (read or write) in a function body.

    Unlike AssignedMemberFinder (assignment targets only, hardcoded to 'self'),
    this looks at every Attribute node regardless of Load/Store context, for an
    arbitrary object name — used to find which source.attr / edge.attr names a
    hook actually references, so the template only loads what's needed.
    """

    def __init__(self, node: ast.FunctionDef, obj_name: str):
        if not isinstance(node, ast.FunctionDef):
            raise ValueError("ast node needs to be of type ast.FunctionDef")
        self.obj_name = obj_name
        self.used_members: set[str] = set()
        self.node = node

    def run(self) -> set[str]:
        self.visit(self.node)
        return self.used_members

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == self.obj_name:
            self.used_members.add(node.attr)
