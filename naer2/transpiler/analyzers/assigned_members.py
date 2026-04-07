import ast


class AssignedMemberFinder(ast.NodeVisitor):
    def __init__(self, node: ast.FunctionDef):
        if not isinstance(node, ast.FunctionDef):
            raise ValueError("ast node needs to be of type ast.FunctionDef")
        self.assigned_members = set()
        self.node = node

    def run(self):
        self.visit(self.node)
        return self.assigned_members

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name):
                    if target.value.id == "self":
                        self.assigned_members.add(target.attr)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Attribute):
            if isinstance(node.target.value, ast.Name):
                if node.target.value.id == "self":
                    self.assigned_members.add(node.target.attr)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Attribute):
            if isinstance(node.target.value, ast.Name):
                if node.target.value.id == "self":
                    self.assigned_members.add(node.target.attr)
