"""
Unit tests for naer2.transpiler.transformers.random_lowering_transformer.

Pure AST rewriting — no network, no Triton compile.
"""

import ast

import pytest

from naer2.transpiler.transformers import RandomLoweringTransformer


# the Philox key: the population's seed (the kernel's `seed`) high, the time's bits low (RandomLoweringTransformer)
KEY = "seed.to(tl.int64) << 32 | t.to(tl.int32, bitcast=True).to(tl.int64) & 4294967295"


def _rhs(src: str) -> str:
    out = ast.unparse(RandomLoweringTransformer().visit(ast.parse(src)))
    return out.split(" = ", 1)[1]


def test_random_random_lowers_to_tl_rand_keyed_by_seed_t_and_offs():
    assert _rhs("y = random.random()") == f"tl.rand({KEY}, offs)"


def test_gauss_and_normalvariate_lower_to_tl_randn():
    assert _rhs("y = random.gauss(mu, sigma)") == f"mu + sigma * tl.randn({KEY}, offs)"
    assert _rhs("y = random.normalvariate(0.0, 1.0)") == f"0.0 + 1.0 * tl.randn({KEY}, offs)"


def test_multiple_call_sites_in_one_body_get_distinct_seeds():
    src = "y = random.random()\nz = random.random()"
    out = ast.unparse(RandomLoweringTransformer().visit(ast.parse(src)))
    first, second = out.splitlines()
    seed1 = first.split(" = ", 1)[1]
    seed2 = second.split(" = ", 1)[1]
    assert seed1 == f"tl.rand({KEY}, offs)"
    assert seed2 != seed1
    assert "seed.to(tl.int64) << 32" in seed2 and "t.to(tl.int32, bitcast=True).to(tl.int64) ^" in seed2  # (the salt: low)


def test_each_fresh_transformer_instance_restarts_the_call_site_count():
    # A new HookTranspiler().emit() call builds a new transformer per hook —
    # the call-site salt must not leak across different hook bodies.
    first_call_seed = _rhs("y = random.random()")
    assert first_call_seed == _rhs("y = random.random()")


def test_nested_in_an_expression():
    assert _rhs("y = 1.0 - random.random()") == f"1.0 - tl.rand({KEY}, offs)"


def test_random_random_rejects_arguments():
    with pytest.raises(ValueError, match="takes no arguments"):
        ast.unparse(RandomLoweringTransformer().visit(ast.parse("y = random.random(1)")))


def test_gauss_requires_two_arguments():
    with pytest.raises(ValueError, match="expects exactly 2 arguments"):
        ast.unparse(RandomLoweringTransformer().visit(ast.parse("y = random.gauss(mu)")))


def test_unsupported_random_call_raises():
    with pytest.raises(ValueError, match=r"random\.choice.*supported"):
        ast.unparse(RandomLoweringTransformer().visit(ast.parse("y = random.choice(xs)")))


def test_unrelated_calls_are_left_alone():
    assert _rhs("y = some_random_name.random()") == "some_random_name.random()"
    assert _rhs("y = math.exp(x)") == "math.exp(x)"
