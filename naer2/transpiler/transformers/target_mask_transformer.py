import ast

from .base import StmtTransformer


class TargetMaskTransformer(StmtTransformer):
    """Wraps assignments to dynamic node attrs with tl.where(is_target, new_val, old_val).

    Used inside the CSR link loop where is_target is a [GROUP_SIZE] bool that
    selects which threads are the spike's destination.  Applies uniformly to
    plain assigns and all augmented assigns by expanding the op inline:

        x  = expr       →  x = tl.where(is_target, expr, x)
        x op= expr      →  x = tl.where(is_target, x op expr, x)

    Only names listed in `dynamic_attrs` are wrapped; temporaries are left alone.
    """

    def __init__(self, dynamic_attrs: set):
        self.attrs = set(dynamic_attrs)

    @staticmethod
    def _where(cond, true, false):
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()), attr="where", ctx=ast.Load()),
            args=[cond, true, false],
            keywords=[],
        )

    @staticmethod
    def _is_target():
        return ast.Name(id="is_target", ctx=ast.Load())

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if not (isinstance(node.target, ast.Name) and node.target.id in self.attrs):
            return node
        name = node.target.id
        var = ast.Name(id=name, ctx=ast.Load())
        new_val = self._where(self._is_target(), ast.BinOp(left=var, op=node.op, right=node.value), var)
        return ast.copy_location(
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=new_val), node
        )

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.attrs
        ):
            name = node.targets[0].id
            new_val = self._where(self._is_target(), node.value, ast.Name(id=name, ctx=ast.Load()))
            return ast.copy_location(
                ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=new_val), node
            )
        return node

