import ast
import copy

from ..types.hook import Hook
from .context import TranspilerContext
from .transformers import IfTransformer, SelfAttrTransformer


class HookTranspiler:
    """
    Transforms one Hook's function body into a list of ast.stmt ready for
    inclusion in a Triton kernel body.

    Applies two transformer passes in order:
      1. SelfAttrTransformer — resolves self.attr to constants or local names
      2. IfTransformer       — lowers if/elif/else to vectorized masked ops
    """

    def __init__(self, hook: Hook, context: TranspilerContext):
        self.hook = hook
        self.context = context

    def emit(self) -> list[ast.stmt]:
        func_copy = copy.deepcopy(self.hook.ast)
        transformed = SelfAttrTransformer(self.context).visit(func_copy)
        transformed = IfTransformer().visit(transformed)
        ast.fix_missing_locations(transformed)
        return transformed.body
