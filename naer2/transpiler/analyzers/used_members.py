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


class ReadMemberFinder(ast.NodeVisitor):
    """Collects every obj_name.attr a function body READS: attribute loads,
    plus augmented-assignment targets (`self.x += 1` reads x too). A plain
    assignment target (`self.x = ...`) is a write only."""

    def __init__(self, node: ast.FunctionDef, obj_name: str):
        if not isinstance(node, ast.FunctionDef):
            raise ValueError("ast node needs to be of type ast.FunctionDef")
        self.obj_name = obj_name
        self.read_members: set[str] = set()
        self.node = node

    def run(self) -> set[str]:
        self.visit(self.node)
        return self.read_members

    def _own(self, node) -> bool:
        return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == self.obj_name

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        if self._own(node) and isinstance(node.ctx, ast.Load):
            self.read_members.add(node.attr)

    def visit_AugAssign(self, node: ast.AugAssign):
        self.generic_visit(node)
        if self._own(node.target):
            self.read_members.add(node.target.attr)
