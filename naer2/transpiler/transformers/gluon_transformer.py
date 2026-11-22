import ast

from .base import StmtTransformer

# tl.<name> with the same name and meaning in Gluon (elementwise ops, dtypes)
_SAME = {
    "where", "exp", "exp2", "log", "log2", "sin", "cos", "sqrt", "rsqrt", "floor", "ceil", "erf", "abs",
    "minimum", "maximum", "fma", "fdiv", "cast", "to_tensor",
    "int1", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "bfloat16",
    "float32", "float64",
}
# Triton's RNG through the fused template's triton.jit wrappers (a Gluon kernel calls triton.jit functions)
_HELPERS = {"rand": "_rand", "randn": "_randn"}


class GluonTransformer(StmtTransformer):
    """Rewrites a transpiled hook body (Triton: tl.*) into Gluon for the fused
    template (naer2/transpiler/templates/fused.py), as the pipeline's last
    pass:

        tl.where / tl.exp / ... / tl.int32 / ...   ->  gl.where / gl.exp / ...
        tl.extra.libdevice.f(...)                  ->  gl.extra.libdevice.f(...)
        tl.rand(seed, offs) / tl.randn(seed, offs) ->  _rand(...) / _randn(...)
        a < b <= c                                 ->  (a < b) & (b <= c)

    The last one because Triton (and Gluon) reject chained comparisons; the
    pieces are elementwise, so `&` is the per-lane `and`. Any other tl.* has
    no Gluon counterpart here and raises, rather than leaving `tl` in a Gluon
    kernel. Layouts aren't needed: every op is elementwise on tensors whose
    layout the template gives (scalars broadcast).
    """

    @staticmethod
    def _chain(node: ast.Attribute) -> list[str] | None:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            return [node.id] + parts[::-1]
        return None

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        chain = self._chain(node)
        if not chain or chain[0] != "tl":
            return self.generic_visit(node)
        if len(chain) == 2 and chain[1] in _SAME:
            return ast.copy_location(ast.Attribute(value=ast.Name(id="gl", ctx=ast.Load()), attr=chain[1],
                                                   ctx=node.ctx), node)
        if len(chain) == 2 and chain[1] in _HELPERS:
            return ast.copy_location(ast.Name(id=_HELPERS[chain[1]], ctx=ast.Load()), node)
        if len(chain) == 4 and chain[1:3] == ["extra", "libdevice"]:
            return ast.copy_location(ast.parse(f"gl.extra.libdevice.{chain[3]}", mode="eval").body, node)
        raise ValueError(f"{'.'.join(chain)} has no Gluon lowering for the fused template")

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        operands = [node.left] + node.comparators
        parts = [ast.Compare(left=a, ops=[op], comparators=[b]) for a, op, b in zip(operands, node.ops, operands[1:])]
        result = parts[0]
        for p in parts[1:]:
            result = ast.BinOp(left=result, op=ast.BitAnd(), right=p)
        return ast.copy_location(result, node)
