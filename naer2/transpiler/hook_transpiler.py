import ast
import copy

from ..types.hook import Hook
from .context import TranspilerContext
from .transformers import (
    AttrPrefixTransformer,
    IfTransformer,
    SelfAttrTransformer,
    TargetMaskTransformer,
)


class HookTranspiler:
    """
    Transforms one Hook's function body into a list of ast.stmt ready for
    inclusion in a Triton kernel body.

    Base pipeline (emit):
      1. SelfAttrTransformer — resolves self.attr to constants or local names
      2. IfTransformer       — lowers if/elif/else to vectorized masked ops

    Extended pipeline for the spike loop (emit_for_spike_loop):
      3. AttrPrefixTransformer("edge")   — edge.attr   → edge_attr
      4. AttrPrefixTransformer("source") — source.attr → source_attr
      5. TargetMaskTransformer           — wraps dynamic-attr writes with is_target

    Both receive hooks and edge forward hooks use emit_for_spike_loop; the prefix
    transformers are no-ops on edge forward hooks which contain no edge.X patterns.
    Pass the appropriate context (node ctx or edge ctx) when constructing the
    HookTranspiler — SelfAttrTransformer uses whatever context is on self.
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

    def emit_for_spike_loop(self, dynamic_attrs: set) -> list[ast.stmt]:
        """Emit hook body for use inside the CSR link loop.

        Applies the full pipeline so that edge.attr / source.attr references
        are rewritten to local names and dynamic-attr assignments are masked
        by is_target to affect only the spike's target neuron.
        """
        stmts = self.emit()
        stmts = AttrPrefixTransformer("edge").transform(stmts)
        stmts = AttrPrefixTransformer("source").transform(stmts)
        stmts = TargetMaskTransformer(dynamic_attrs).transform(stmts)
        return stmts
