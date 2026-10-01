import ast

from ..context import TranspilerContext


class SelfAttrTransformer(ast.NodeTransformer):
    """
    Substitutes self.attr nodes throughout a function body:
      - static attr  → ast.Constant with the literal instance value
      - dynamic attr → ast.Name with {dynamic_prefix}{attr}, preserving Load/Store context

    dynamic_prefix distinguishes whose state self refers to when the same spike
    loop iteration sees two: a node's own hooks need bare names (matching the
    template's own per-neuron loads), while an edge forward hook needs edge_
    (matching the edge_{attr} locals the CSR link loop loads per-link state into).

    The Store context on Name nodes is valid for assignment targets, matching how
    AugAssign and Assign targets are handled downstream. Unknown self.attr nodes
    (e.g. n2.spike() where 'n2' is not 'self') are left untouched.
    """

    def __init__(self, context: TranspilerContext, dynamic_prefix: str = ""):
        self.context = context
        self.dynamic_prefix = dynamic_prefix

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
                ast.Name(id=f"{self.dynamic_prefix}{attr}", ctx=node.ctx),
                node,
            )
        return self.generic_visit(node)
