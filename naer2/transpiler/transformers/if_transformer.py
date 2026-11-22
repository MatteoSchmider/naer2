import ast

_STMTS = lambda src: ast.parse(src).body


class IfTransformer(ast.NodeTransformer):
    """
    Lowers all if/elif/else blocks in hook bodies to vectorized masked ops.

    SPMD kernels can't branch per-element, so every if block is unconditionally
    transformed into a mask computation + tl.where assignments:

        if <cond>:            →   _m0 = mask & (<cond>)
            x = a                  x = tl.where(_m0, a, x)
            n2.spike()             <<< compacted spike write, gated by _m0 >>>
        else:
            x = b             →   x = tl.where(~_m0 & mask, b, x)

    n2.spike(time=...) lowers through spike_lowering(mask, time_expr, bare),
    gated by the current branch mask (the fused template's _SpikeSites: each
    site's fire mask and emission time, emitted after the hooks). time=None
    (default, no argument) records default_spike_time; an expression is
    spliced in verbatim as the timestamp instead -- see naer2.sends.spike's
    docstring. Whatever mask this ends up gated by (root "mask", or
    "is_target" after MaskAliasTransformer rewrites it for per-link use) is
    exactly the set of lanes this statement should treat as firing, so the
    lowering itself needs no is_target-awareness of its own.

    while <cond>: body  →  a loop for as long as any lane's condition holds,
    each lane in it until its own fails (and never again -- so the loop's
    counter is every active lane's own iteration):

        _it0 = tl.to_tensor(0)
        _m0 = mask & (<cond>)
        while _any_lane(_m0):
            <body, rooted at _m0>
            _it0 = _it0 + 1
            _m0 = _m0 & (<cond>)

    (_any_lane: the template's reduction over the mask, whatever its shape;
    _it0: the loop's counter, named by RandomLoweringTransformer so random.*
    inside draws anew every iteration -- n2_loop_var.) n2.spike() inside a
    loop lowers through spike_lowering(..., in_loop=True): a site that fires
    any number of times, emitted where it stands. No else, break, continue
    or return in a loop.

    ast.Assign  → tl.where on the branch mask.
    ast.AugAssign → expanded to BinOp then tl.where.
    A top-level n2.spike() (outside any if) is the same emission rooted at "mask".
    Nested ifs and elif chains are handled recursively via _lower_body/_lower_if.
    """

    def __init__(self, spike_lowering=None, default_spike_time: str = "t"):
        """spike_lowering: lowers n2.spike(), called as
        spike_lowering(mask, time_expr, bare) -> list[ast.stmt] (None: a
        hook without n2.spike()).
        default_spike_time: a bare n2.spike()'s time."""
        self._counter = 0
        self._loops = 0  # while loops around the statement being lowered
        self._loop_vars = 0
        self.spike_lowering = spike_lowering
        self.default_spike_time = default_spike_time

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

    def _spike_time_expr(self, node: ast.Expr) -> str:
        """The raw source text of n2.spike()'s time argument, or the default
        ("t", the window end; the fused template's: t_start) if omitted.

        No validation — whatever expression the user wrote (already resolved by
        SelfAttrTransformer, which ran before IfTransformer) is spliced in
        verbatim as the spike's timestamp. A negative or otherwise nonsensical
        time is the caller's own responsibility.
        """
        call = node.value
        for kw in call.keywords:
            if kw.arg == "time":
                return ast.unparse(kw.value)
        if call.args:
            return ast.unparse(call.args[0])
        return self.default_spike_time

    def _lower_spike(self, node: ast.Expr, mask: str) -> list[ast.stmt]:
        time_expr = self._spike_time_expr(node)
        bare = not node.value.args and not node.value.keywords
        if self.spike_lowering is None:
            raise ValueError("n2.spike() needs a spike_lowering to lower it with")
        if self._loops:
            return self.spike_lowering(mask, time_expr, bare, in_loop=True)
        return self.spike_lowering(mask, time_expr, bare)

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
                result += self._lower_spike(stmt, mask)
            elif isinstance(stmt, ast.If):
                result += self._lower_if(stmt, mask)
            elif isinstance(stmt, ast.While):
                result += self._lower_while(stmt, mask)
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

    def _lower_while(self, node: ast.While, parent_mask: str) -> list[ast.stmt]:
        if node.orelse:
            raise NotImplementedError("while ... else isn't lowered")
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Break, ast.Continue, ast.Return)):
                raise NotImplementedError(f"{type(sub).__name__.lower()} in a while loop isn't lowered: each lane "
                                          f"leaves the loop when its condition fails")
        it = getattr(node, "n2_loop_var", None)
        if it is None:
            it = f"_itw{self._loop_vars}"
            self._loop_vars += 1
        mask = self._fresh_mask()
        cond = ast.unparse(node.test)
        result = list(_STMTS(f"{it} = tl.to_tensor(0)\n{mask} = {parent_mask} & ({cond})"))
        self._loops += 1
        body = self._lower_body(node.body, mask)
        self._loops -= 1
        body += _STMTS(f"{it} = {it} + 1\n{mask} = {mask} & ({cond})")
        loop = ast.While(test=ast.parse(f"_any_lane({mask})", mode="eval").body, body=body, orelse=[])
        return result + [loop]

    def visit_If(self, node: ast.If) -> list[ast.stmt]:
        return self._lower_if(node, "mask")

    def visit_While(self, node: ast.While) -> list[ast.stmt]:
        return self._lower_while(node, "mask")

    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
        # A top-level n2.spike() (not under any if) fires every lane the hook
        # runs on: the root mask, which is the whole group in update/send's final
        # pass and, once MaskAliasTransformer has rewritten it, just the delivered
        # neurons inside a per-link receive/send. Spikes under an if are lowered
        # by _lower_body with that branch's mask instead and never reach here.
        if self._is_spike(node):
            return self._lower_spike(node, "mask")
        return self.generic_visit(node)
