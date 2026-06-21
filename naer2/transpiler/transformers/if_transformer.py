import ast

_STMTS = lambda src: ast.parse(src).body


class IfTransformer(ast.NodeTransformer):
    """
    Lowers all if/elif/else blocks in hook bodies to vectorized masked ops.

    SPMD kernels can't branch per-element, so every if block is unconditionally
    transformed into a mask computation + tl.where assignments:

        if <cond>:            →   _m0 = mask & (<cond>)
            x = a                  x = tl.where(_m0, a, x)
            n2.spike()             tl.store(spikes_ptr + offsets, _m0, mask=mask)
        else:
            x = b             →   x = tl.where(~_m0 & mask, b, x)

    n2.spike()  → tl.store to spikes_ptr, gated by the current branch mask.
    ast.Assign  → tl.where on the branch mask.
    ast.AugAssign → expanded to BinOp then tl.where.
    Nested ifs and elif chains are handled recursively via _lower_body/_lower_if.
    """

    def __init__(self):
        self._counter = 0

    def _fresh_mask(self) -> str:
        name = f"_m{self._counter}"
        self._counter += 1
        return name

    @staticmethod
    def _is_spike(node: ast.stmt) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "spike"
        )

    def _lower_body(self, stmts: list[ast.stmt], mask: str) -> list[ast.stmt]:
        result = []
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    name = ast.unparse(target)
                    value = ast.unparse(stmt.value)
                    result += _STMTS(f"{name} = tl.where({mask}, {value}, {name})")
            elif isinstance(stmt, ast.AugAssign):
                name = ast.unparse(stmt.target)
                rhs = ast.unparse(ast.BinOp(
                    left=ast.Name(id=name, ctx=ast.Load()),
                    op=stmt.op,
                    right=stmt.value,
                ))
                result += _STMTS(f"{name} = tl.where({mask}, {rhs}, {name})")
            elif self._is_spike(stmt):
                result += _STMTS(f"tl.store(spikes_ptr + offsets, {mask}, mask=mask)")
            elif isinstance(stmt, ast.If):
                result += self._lower_if(stmt, mask)
            else:
                result.append(stmt)
        return result

    def _lower_if(self, node: ast.If, parent_mask: str) -> list[ast.stmt]:
        mask = self._fresh_mask()
        result = list(_STMTS(f"{mask} = {parent_mask} & ({ast.unparse(node.test)})"))
        result += self._lower_body(node.body, mask)
        if node.orelse:
            neg = f"~{mask} & {parent_mask}"
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                result += self._lower_if(node.orelse[0], neg)
            else:
                result += self._lower_body(node.orelse, neg)
        return result

    def visit_If(self, node: ast.If) -> list[ast.stmt]:
        return self._lower_if(node, "mask")
