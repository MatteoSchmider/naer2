import ast

from ..context import TranspilerContext


class SelfAttrTransformer(ast.NodeTransformer):
    """
    Substitutes self.attr nodes throughout a function body:
      - static attr  → ast.Constant with the literal instance value
      - dynamic attr → ast.Name with the attr name, preserving Load/Store context

    The Store context on Name nodes is valid for assignment targets, matching how
    AugAssign and Assign targets are handled downstream. Unknown self.attr nodes
    (e.g. n2.spike() where 'n2' is not 'self') are left untouched.
    """

    def __init__(self, context: TranspilerContext):
        self.context = context

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            return self.generic_visit(node)
        attr = node.attr
        if attr in self.context.static_attrs:
            return ast.copy_location(
                ast.Constant(value=self.context.static_attrs[attr]),
                node,
            )
        if attr in self.context.dynamic_attrs:
            return ast.copy_location(
                ast.Name(id=attr, ctx=node.ctx),
                node,
            )
        return self.generic_visit(node)
