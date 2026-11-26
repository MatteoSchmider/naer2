"""
Unit tests for naer2.transpiler.transformers.if_transformer.

Pure AST rewriting — no network, no Triton compile. n2.spike() lowers
through a stub that records each site's (mask, time, bare) as a call, the
way the fused template's _SpikeSites collects them.
"""

import ast

import pytest

from naer2.transpiler.transformers import IfTransformer, MaskAliasTransformer


def _site(mask, time_expr, bare):
    return ast.parse(f"_site({mask}, {time_expr}, {bare})").body


def _lower(src, **kw):
    return ast.unparse(IfTransformer(_site, **kw).visit(ast.parse(src)))


def test_top_level_spike_lowered():
    out = _lower("def f(self):\n    n2.spike()\n")
    assert "n2.spike" not in out, "top-level n2.spike() must be lowered, not passed through"
    assert "_site(mask, t, True)" in out, "fires every lane the hook runs on: the root mask, at the default time"


def test_the_default_spike_time():
    assert "_site(mask, t_start, True)" in _lower("def f(self):\n    n2.spike()\n", default_spike_time="t_start")


def test_top_level_spike_explicit_time():
    assert "_site(mask, time + 1.0, False)" in _lower("def f(self):\n    n2.spike(time + 1.0)\n")
    assert "_site(mask, time, False)" in _lower("def f(self):\n    n2.spike(time=time)\n")


def test_top_level_spike_aliases_to_is_target_inside_a_link():
    # inside a per-link receive/send only the delivered neurons may fire: the root mask is rewritten to is_target
    tree = IfTransformer(_site).visit(ast.parse("def f(self):\n    n2.spike()\n"))
    aliased = ast.unparse(MaskAliasTransformer().visit(tree))
    assert "_site(is_target, t, True)" in aliased


def test_if_nested_spike_keeps_its_own_branch_mask():
    nested = _lower("def f(self):\n    if x > 1:\n        n2.spike()\n")
    assert "_m0 = mask & (x > 1)" in nested
    assert "_site(_m0, t, True)" in nested


def test_a_spike_without_a_lowering_raises():
    with pytest.raises(ValueError, match="spike_lowering"):
        IfTransformer().visit(ast.parse("def f(self):\n    n2.spike()\n"))
