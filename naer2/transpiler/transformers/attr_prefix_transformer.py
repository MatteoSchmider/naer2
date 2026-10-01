import ast

from .base import StmtTransformer


class AttrPrefixTransformer(StmtTransformer):
    """Rewrites obj_name.attr in hook bodies to either a literal or a local name.

    `rename` maps the hook-facing attribute name back to the real underlying
    name (i.e. the reverse of a fabric_map/source_map, which are stored as
    {real_name: hook_name}). Attributes not in `rename` pass through unchanged.

    `static_attrs`, when given, lets a static (never assigned by forward/backward)
    attribute inline as a compile-time constant — exactly like SelfAttrTransformer
    already does for a node/edge's own static self.attr — instead of requiring a
    per-link tensor load that will never be produced for it:

        obj.attr → ast.Constant(static_attrs[real_attr])   if real_attr is static
        obj.attr → {obj_name}_{real_attr}                  otherwise (needs a load)

    Examples:
        AttrPrefixTransformer("edge")                    — edge.weight → edge_weight
        AttrPrefixTransformer("edge", {"w": "weight"})    — edge.w      → edge_weight
        AttrPrefixTransformer("edge", static_attrs={"weight": 1})
                                                           — edge.weight → 1 (literal)
        AttrPrefixTransformer("source", {"src_pot": "potential"})
                                                           — source.src_pot → source_potential
    """

    def __init__(
        self,
        obj_name: str,
        rename: dict[str, str] | None = None,
        static_attrs: dict[str, object] | None = None,
    ):
        self.obj_name = obj_name
        self.rename = rename or {}
        self.static_attrs = static_attrs or {}

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == self.obj_name:
            real_attr = self.rename.get(node.attr, node.attr)
            if real_attr in self.static_attrs:
                return ast.copy_location(
                    ast.Constant(value=self.static_attrs[real_attr]), node
                )
            return ast.copy_location(
                ast.Name(id=f"{self.obj_name}_{real_attr}", ctx=node.ctx), node
            )
        return node
