import ast

from .base import StmtTransformer


class MaskAliasTransformer(StmtTransformer):
    """Renames the bare `mask` identifier to `alias` (default `is_target`) throughout a section.

    IfTransformer always seeds its branch-mask chain from the literal name
    "mask" (the whole-group out-of-bounds mask): `_m0 = mask & (<cond>)`, with
    every nested/elif mask derived from that one. For code that runs once per
    CSR link — restricted to the one lane that is this link's actual target —
    that seed must be is_target instead, otherwise a branch a non-target lane
    happens to satisfy (e.g. its potential is already above threshold from an
    earlier, unrelated event) would leak into anything gated by that branch's
    mask, such as a spike emission meant only for the true target.

    Only the root seed ever references the bare name "mask" directly — nested
    masks are named _m0, _m1, ... — so a blanket rename is sufficient and
    doesn't need to know anything about IfTransformer's internals.

    Code running on link lanes (edge forward, one lane per link in a tile)
    aliases to that tile's validity mask instead of is_target.
    """

    def __init__(self, alias: str = "is_target"):
        self.alias = alias

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "mask":
            return ast.copy_location(ast.Name(id=self.alias, ctx=node.ctx), node)
        return node
