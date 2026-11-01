import ast
import copy

from ..types.hook import Hook
from .context import TranspilerContext
from .transformers import (
    AttrPrefixTransformer,
    IfTransformer,
    MaskAliasTransformer,
    MathLoweringTransformer,
    RandomLoweringTransformer,
    SelfAttrTransformer,
    TargetMaskTransformer,
)


class HookTranspiler:
    """
    Transforms one Hook's function body into a list of ast.stmt ready for
    inclusion in a Triton kernel body.

    Base pipeline (emit):
      1. SelfAttrTransformer — resolves self.attr to constants or local names
      2. MathLoweringTransformer — math.exp etc. → per-element tl.* ops
      3. RandomLoweringTransformer — random.random() etc. → per-lane tl.rand/randn
      4. IfTransformer       — lowers if/elif/else (and top-level n2.spike) to
                               vectorized masked ops

    Extended pipeline for the spike loop (emit_for_spike_loop):
      5. AttrPrefixTransformer("edge")   — edge.attr   → edge_attr
      6. AttrPrefixTransformer("source") — source.attr → source_attr
      7. TargetMaskTransformer           — wraps dynamic-attr writes with is_target

    Both receive hooks and edge forward hooks use emit_for_spike_loop; the prefix
    transformers are no-ops on edge forward hooks which contain no edge.X patterns.
    Pass the appropriate context (node ctx or edge ctx) when constructing the
    HookTranspiler — SelfAttrTransformer uses whatever context is on self.

    self_prefix distinguishes whose state self.attr refers to: a node's own
    hooks (receive/update/send) leave it "" so self.attr → bare attr, matching
    the template's per-neuron loads; the edge forward hook passes "edge_" so
    self.attr → edge_attr, matching the edge_{attr} locals the CSR link loop
    loads per-link state into.
    """

    def __init__(self, hook: Hook, context: TranspilerContext, self_prefix: str = ""):
        self.hook = hook
        self.context = context
        self.self_prefix = self_prefix

    def emit(self) -> list[ast.stmt]:
        func_copy = copy.deepcopy(self.hook.ast)
        transformed = SelfAttrTransformer(self.context, self.self_prefix).visit(func_copy)
        # Before IfTransformer, which unparses branch tests and spike-time
        # expressions into new statements: math/random calls there must
        # already be lowered.
        transformed = MathLoweringTransformer().visit(transformed)
        transformed = RandomLoweringTransformer().visit(transformed)
        transformed = IfTransformer().visit(transformed)
        ast.fix_missing_locations(transformed)
        return transformed.body

    def emit_for_spike_loop(
        self,
        dynamic_attrs: set,
        fabric_map: dict[str, str] | None = None,
        source_map: dict[str, str] | None = None,
        edge_static_attrs: dict[str, object] | None = None,
        source_static_attrs: dict[str, object] | None = None,
        mask_target: bool = True,
        lane_mask: str | None = None,
    ) -> list[ast.stmt]:
        """Emit hook body for use inside the CSR link loop.

        Applies the full pipeline so that edge.attr / source.attr references
        are rewritten to a literal (static) or a local name (dynamic).

        fabric_map / source_map are {real_name: hook_name} connection mappings;
        AttrPrefixTransformer needs the reverse to recover the real attribute
        name backing the hook-facing name the user wrote.

        edge_static_attrs (conn.fabric.constant_attrs) / source_static_attrs let
        a static edge or source attr inline as a compile-time constant instead
        of resolving to edge_{attr}/source_{attr} — the template only generates
        a load for attrs it was told are tensor-backed, so treating a static
        attr as a load target would reference an undefined local.

        mask_target controls is_target masking: receive hooks write into
        target-neuron-shaped locals (potential, current, ...) shared by every
        thread in the GROUP_SIZE warp, so only the one thread that is this
        link's actual target should update them, branch on them, or emit a
        spike from them. This applies MaskAliasTransformer (rewriting
        IfTransformer's root mask seed from "mask" to "is_target", so any
        if-branch inside the hook — and anything gated by one, including
        n2.spike() — is correctly scoped to the target lane) and
        TargetMaskTransformer (top-level, not-inside-any-if dynamic-attr
        assignments, which IfTransformer never touches). Edge forward hooks
        instead update per-link scalars (weight, ...) tied to link_idx alone;
        every thread computes the same value regardless of which one is the
        target, so masking would broadcast a scalar into a [GROUP_SIZE] vector
        and break the scalar store. Pass mask_target=False for edge forward.

        lane_mask names the validity mask of the lanes this hook runs on when
        those lanes aren't neurons — the edge forward hook runs on a tile of
        links, so its if-branches must be rooted at that tile's valid-link mask
        instead of the neuron-group mask.
        """
        stmts = self.emit()
        if mask_target:
            stmts = MaskAliasTransformer().transform(stmts)
        elif lane_mask is not None:
            stmts = MaskAliasTransformer(lane_mask).transform(stmts)
        edge_rename = {hook_name: real for real, hook_name in (fabric_map or {}).items()}
        source_rename = {hook_name: real for real, hook_name in (source_map or {}).items()}
        stmts = AttrPrefixTransformer("edge", edge_rename, edge_static_attrs).transform(stmts)
        stmts = AttrPrefixTransformer("source", source_rename, source_static_attrs).transform(stmts)
        if mask_target:
            prefixed_attrs = {f"{self.self_prefix}{attr}" for attr in dynamic_attrs}
            stmts = TargetMaskTransformer(prefixed_attrs).transform(stmts)
        return stmts
