"""
Unit tests for naer2.transpiler.transformers.if_transformer.

Pure AST rewriting — no network, no Triton compile.
"""

import ast

from naer2.transpiler.transformers import IfTransformer, MaskAliasTransformer


def test_top_level_spike_lowered():
    out = ast.unparse(IfTransformer().visit(ast.parse("def f(self):\n    n2.spike()\n")))
    assert "n2.spike" not in out, "top-level n2.spike() must be lowered, not passed through"
    assert "_n_fire = mask.to(tl.int32)" in out, "fires every lane the hook runs on: the root mask"
    assert "tl.store(local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, t, mask=mask)" in out


def test_top_level_spike_explicit_time():
    out = ast.unparse(IfTransformer().visit(ast.parse("def f(self):\n    n2.spike(time + 1.0)\n")))
    assert "time + 1.0, mask=mask)" in out


def test_top_level_spike_aliases_to_is_target_inside_a_link():
    # inside a per-link receive/send only the delivered neurons may fire: the root
    # mask is rewritten to is_target (the mask= *keyword* name must survive)
    src = "def f(self):\n    n2.spike()\n"
    aliased = ast.unparse(MaskAliasTransformer().visit(IfTransformer().visit(ast.parse(src))))
    assert "_n_fire = is_target.to(tl.int32)" in aliased
    assert "mask=is_target)" in aliased
    assert "mask=mask" not in aliased


def test_if_nested_spike_keeps_its_own_branch_mask():
    nested = ast.unparse(IfTransformer().visit(ast.parse("def f(self):\n    if x > 1:\n        n2.spike()\n")))
    assert "_n_fire = _m0.to(tl.int32)" in nested
