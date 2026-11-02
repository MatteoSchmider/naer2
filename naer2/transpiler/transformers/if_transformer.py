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

    n2.spike(time=...) lowers to a compacted write of (node_id, timestamp) into
    this program instance's own private slice of a global scratch buffer,
    gated by the current branch mask: tl.cumsum turns the branch mask into
    each firing lane's offset within that slice, both buffers are written
    with that mask so only firing lanes actually store anything, and the
    local write-cursor (local_spike_count) is bumped by plain scalar
    arithmetic — no atomic needed, since this slice belongs exclusively to
    this program instance. That count is published once, at the very end of
    the kernel, with a plain scalar store (also no atomic) — see
    NoDelayUpdateKernelTemplate.build's flush step and naer2/spike_gather.py:
    the local scratch buffers this writes into are already laid out exactly
    as a BlockedSpikes, so nothing needs compacting into a shared buffer at
    all, let alone with a contended global atomic.
    time=None (default, no argument) records `t`; an expression is spliced in
    verbatim as the timestamp instead — see naer2.sends.spike's docstring.
    Whatever mask this ends up gated by (root "mask", or "is_target" after
    MaskAliasTransformer rewrites it for per-link use) is exactly the set of
    lanes this statement should treat as firing, so the lowering itself needs
    no is_target-awareness of its own.

    ast.Assign  → tl.where on the branch mask.
    ast.AugAssign → expanded to BinOp then tl.where.
    A top-level n2.spike() (outside any if) is the same emission rooted at "mask".
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

    @staticmethod
    def _spike_time_expr(node: ast.Expr) -> str:
        """The raw source text of n2.spike()'s time argument, or "t" if omitted.

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
        return "t"

    def _lower_spike(self, node: ast.Expr, mask: str) -> list[ast.stmt]:
        time_expr = self._spike_time_expr(node)
        return _STMTS(f"""\
_n_fire = {mask}.to(tl.int32)
_local_pos = local_spike_count + tl.cumsum(_n_fire, axis=0) - _n_fire
local_spike_count = local_spike_count + tl.sum(_n_fire, axis=0)
tl.store(local_spike_node_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, offs, mask={mask})
tl.store(local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, {time_expr}, mask={mask})
""")

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

    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
        # A top-level n2.spike() (not under any if) fires every lane the hook
        # runs on: the root mask, which is the whole group in update/send's final
        # pass and, once MaskAliasTransformer has rewritten it, just the delivered
        # neurons inside a per-link receive/send. Spikes under an if are lowered
        # by _lower_body with that branch's mask instead and never reach here.
        if self._is_spike(node):
            return self._lower_spike(node, "mask")
        return self.generic_visit(node)
