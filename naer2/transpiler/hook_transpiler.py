import ast
import copy

from ..types.hook import Hook
from .context import TranspilerContext

_STMTS = lambda src: ast.parse(src).body


class _SpikeTransformer(ast.NodeTransformer):
    """
    Lowers if-blocks containing n2.spike() to vectorized Triton masked ops.

    Input (after _SelfAttrTransformer):
        if potential > 1.0:
            potential = 0.0
            n2.spike()

    Output:
        _fire_mask = mask & (potential > 1.0)
        potential = tl.where(_fire_mask, 0.0, potential)
        tl.store(spikes_ptr + offsets, _fire_mask, mask=mask)

    ast.Assign nodes in the body become tl.where calls. Other statement types
    (e.g. AugAssign, function calls that aren't n2.spike()) are left as-is before
    the tl.store — this is intentional: handle the common case now, revisit when
    needed. Non-spike if-blocks are left completely untouched.
    """

    def _is_spike_call(self, node: ast.stmt) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "spike"
        )

    def visit_If(self, node: ast.If):
        if not any(self._is_spike_call(s) for s in node.body):
            return self.generic_visit(node)

        cond = ast.unparse(node.test)
        result = list(_STMTS(f"_fire_mask = mask & ({cond})"))
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    name = ast.unparse(target)
                    value = ast.unparse(stmt.value)
                    result += _STMTS(f"{name} = tl.where(_fire_mask, {value}, {name})")
            elif self._is_spike_call(stmt):
                result += _STMTS("tl.store(spikes_ptr + offsets, _fire_mask, mask=mask)")
        return result


class _SelfAttrTransformer(ast.NodeTransformer):
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
        # Attribute not in either set — leave as-is; will surface as a
        # transpiler error at a later validation stage
        return self.generic_visit(node)


class HookTranspiler:
    """
    Transforms one Hook's function body into a list of ast.stmt ready for
    inclusion in a Triton kernel body.

    Performs two substitutions via _SelfAttrTransformer:
      1. self.static_attr  → Constant(value)   — folded at compile time
      2. self.dynamic_attr → Name(attr)         — refers to a local variable
                                                  that KernelBuilder will back
                                                  with tl.load / tl.store

    The 'self' parameter is dropped since the kernel is not a method; all
    state is accessed through named local variables or kernel arguments.

    n2.spike() and other framework calls are left as Name/Call nodes for a
    later lowering pass to handle.
    """

    def __init__(self, hook: Hook, context: TranspilerContext):
        self.hook = hook
        self.context = context

    def emit(self) -> list[ast.stmt]:
        func_copy = copy.deepcopy(self.hook.ast)
        transformed = _SelfAttrTransformer(self.context).visit(func_copy)
        transformed = _SpikeTransformer().visit(transformed)
        ast.fix_missing_locations(transformed)
        return transformed.body
