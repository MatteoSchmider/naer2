import ast

from .base import StmtTransformer


class AttrPrefixTransformer(StmtTransformer):
    """Rewrites obj_name.attr → {obj_name}_{attr} in hook bodies.

    Examples:
        AttrPrefixTransformer("edge")   — edge.weight   → edge_weight
        AttrPrefixTransformer("source") — source.potential → source_potential
    """

    def __init__(self, obj_name: str):
        self.obj_name = obj_name

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == self.obj_name:
            return ast.copy_location(
                ast.Name(id=f"{self.obj_name}_{node.attr}", ctx=node.ctx), node
            )
        return node
