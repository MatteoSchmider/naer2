import ast
import copy

from ..types.hook import Hook
from .context import TranspilerContext


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
        # Deep-copy so the Hook's stored AST is never mutated
        func_copy = copy.deepcopy(self.hook.ast)
        transformed = _SelfAttrTransformer(self.context).visit(func_copy)
        ast.fix_missing_locations(transformed)
        return transformed.body
