"""
Unit tests for naer2.transpiler.transformers.math_lowering_transformer.

Pure AST rewriting — no network, no Triton compile. Real-compile coverage of
the whitelist (does tl.extra.libdevice actually have each name, on both
backends) is a separate, later-stage system-test concern; see TODO.md.
"""

import ast

import pytest

from naer2.transpiler.transformers import MathLoweringTransformer


def _rhs(src: str) -> str:
    out = ast.unparse(MathLoweringTransformer().visit(ast.parse(src)))
    return out.split(" = ", 1)[1]


def test_core_tl_functions():
    assert _rhs("y = math.exp(x)") == "tl.exp(x)"
    assert _rhs("y = math.log(x) + math.sin(x)") == "tl.log(x) + tl.sin(x)"


def test_libdevice_functions():
    assert _rhs("y = math.tanh(x)") == "tl.extra.libdevice.tanh(x)"
    assert _rhs("y = math.atan2(a, b)") == "tl.extra.libdevice.atan2(a, b)"


def test_extra_rewrites():
    assert _rhs("y = math.fabs(x)") == "tl.abs(x)"
    assert _rhs("y = math.log10(x)").startswith("tl.log(x) * 0.4342944819")


def test_tensor_args_lowered_not_folded():
    assert _rhs("y = math.exp(-dt / 10.0)") == "tl.exp(-dt / 10.0)"


def test_nested_calls():
    assert _rhs("y = math.exp(math.sin(x))") == "tl.exp(tl.sin(x))"


def test_constant_folding():
    # tl.exp doesn't take a bare Python float, and inlined static attributes make
    # all-constant calls common — these must be evaluated at transpile time
    assert _rhs("y = math.exp(-1.0 / 10.0)") == "0.9048374180359595"
    assert _rhs("y = math.sqrt(2.0) * x") == "1.4142135623730951 * x"
    assert _rhs("y = math.exp(-math.log(2.0))") == "0.5"
    assert _rhs("y = math.exp(-dt / math.sqrt(4.0))") == "tl.exp(-dt / 2.0)"  # folds inside a live call


def test_constants_fold():
    assert _rhs("y = math.pi * x") == "3.141592653589793 * x"
    assert _rhs("y = math.tau + math.e") == "6.283185307179586 + 2.718281828459045"


def test_builtins():
    assert _rhs("y = abs(x)") == "tl.abs(x)"
    assert _rhs("y = max(a, 0.0)") == "tl.maximum(a, 0.0)"
    assert _rhs("y = min(a, b)") == "tl.minimum(a, b)"


def test_builtins_fold_constants():
    assert _rhs("y = max(1, 2)") == "2"


def test_builtins_only_rewrite_the_elementwise_arity():
    assert _rhs("y = max(a, b, c)") == "max(a, b, c)"  # only the 2-arg form is elementwise


def test_unsupported_math_call_raises():
    with pytest.raises(ValueError, match=r"math\.factorial.*supported"):
        ast.unparse(MathLoweringTransformer().visit(ast.parse("y = math.factorial(x)")))
