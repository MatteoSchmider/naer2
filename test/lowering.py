"""
Lowering unit tests: math calls -> per-element Triton ops, and top-level n2.spike().

Both are direct AST rewrites, so most checks here are on the rewritten source.
The one check that touches Triton compiles a kernel using EVERY whitelisted math
function, for a CUDA and a HIP (gfx1100) target, so the whitelist can't drift
onto something a backend doesn't have.
"""

import ast

from naer2.transpiler.transformers import IfTransformer, MaskAliasTransformer, MathLoweringTransformer
from naer2.transpiler.transformers.math_lowering_transformer import MATH_LOWERING
from test.triton_check import load_jit


def _lower(src: str) -> str:
    return ast.unparse(MathLoweringTransformer().visit(ast.parse(src)))


def _rhs(src: str) -> str:
    return _lower(src).split(" = ", 1)[1]


def test_math_lowering():
    assert _rhs("y = math.exp(x)") == "tl.exp(x)"
    assert _rhs("y = math.log(x) + math.sin(x)") == "tl.log(x) + tl.sin(x)"
    assert _rhs("y = math.tanh(x)") == "tl.extra.libdevice.tanh(x)"
    assert _rhs("y = math.atan2(a, b)") == "tl.extra.libdevice.atan2(a, b)"
    assert _rhs("y = math.fabs(x)") == "tl.abs(x)"
    assert _rhs("y = math.exp(-dt / 10.0)") == "tl.exp(-dt / 10.0)"          # tensor arg: lowered
    assert _rhs("y = math.exp(math.sin(x))") == "tl.exp(tl.sin(x))"           # nesting
    assert _rhs("y = math.log10(x)").startswith("tl.log(x) * 0.4342944819")
    print("math.* -> tl.* / tl.extra.libdevice.* ✓")


def test_constant_folding():
    # tl.exp doesn't take a bare Python float, and inlined static attributes make
    # all-constant calls common — these must be evaluated at transpile time
    assert _rhs("y = math.exp(-1.0 / 10.0)") == "0.9048374180359595"
    assert _rhs("y = math.sqrt(2.0) * x") == "1.4142135623730951 * x"
    assert _rhs("y = math.exp(-math.log(2.0))") == "0.5"
    assert _rhs("y = math.pi * x") == "3.141592653589793 * x"
    assert _rhs("y = math.tau + math.e") == "6.283185307179586 + 2.718281828459045"
    assert _rhs("y = math.exp(-dt / math.sqrt(4.0))") == "tl.exp(-dt / 2.0)"   # folds inside a live call
    print("all-constant calls folded at transpile time ✓")


def test_builtins():
    assert _rhs("y = abs(x)") == "tl.abs(x)"
    assert _rhs("y = max(a, 0.0)") == "tl.maximum(a, 0.0)"
    assert _rhs("y = min(a, b)") == "tl.minimum(a, b)"
    assert _rhs("y = max(1, 2)") == "2"
    assert _rhs("y = max(a, b, c)") == "max(a, b, c)"     # only the 2-arg form is elementwise
    print("abs / min / max -> tl.abs / tl.minimum / tl.maximum ✓")


def test_unsupported_raises():
    try:
        _lower("y = math.factorial(x)")
    except ValueError as e:
        assert "math.factorial" in str(e) and "supported" in str(e)
    else:
        raise AssertionError("an unlowerable math call must raise, not leave `math` undefined in the kernel")
    print("unsupported math.* raises a clear error ✓")


def test_top_level_spike():
    src = "def f(self):\n    n2.spike()\n"
    out = ast.unparse(IfTransformer().visit(ast.parse(src)))
    assert "n2.spike" not in out, "top-level n2.spike() must be lowered, not passed through"
    assert "_n_fire = mask.to(tl.int32)" in out, "fires every lane the hook runs on: the root mask"
    assert "tl.store(local_spike_time_ptr + pid * LOCAL_SPIKE_CAP + _local_pos, t, mask=mask)" in out

    out = ast.unparse(IfTransformer().visit(ast.parse("def f(self):\n    n2.spike(time + 1.0)\n")))
    assert "time + 1.0, mask=mask)" in out, "explicit time works at top level too"

    # inside a per-link receive/send only the delivered neurons may fire: the root
    # mask is rewritten to is_target (the mask= *keyword* name must survive)
    aliased = ast.unparse(MaskAliasTransformer().visit(IfTransformer().visit(ast.parse(src))))
    assert "_n_fire = is_target.to(tl.int32)" in aliased and "mask=is_target)" in aliased
    assert "mask=mask" not in aliased

    # an if-nested spike is unaffected: still gated by its own branch mask
    nested = ast.unparse(IfTransformer().visit(ast.parse("def f(self):\n    if x > 1:\n        n2.spike()\n")))
    assert "_n_fire = _m0.to(tl.int32)" in nested
    print("top-level n2.spike() lowered, rooted at the hook's lane mask ✓")


def test_whitelist_compiles_on_both_backends():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fns = sorted(MATH_LOWERING) + ["log10"]
    two_arg = {"atan2", "pow", "hypot", "fmod", "copysign"}
    terms = " + ".join(f"math.{f}(x, x)" if f in two_arg else f"math.{f}(x)" for f in fns)
    body = _lower(f"y = {terms}")
    kernel = load_jit(
        "import triton\nimport triton.language as tl\n\n@triton.jit\n"
        "def whitelist(x_ptr, y_ptr, N: tl.constexpr):\n"
        f"    o = tl.arange(0, N)\n    x = tl.load(x_ptr + o)\n    {body}\n    tl.store(y_ptr + o, y)\n",
        "whitelist", name="whitelist",
    )
    for target in (GPUTarget("cuda", 80, 32), GPUTarget("hip", "gfx1100", 32)):
        triton.compile(
            ASTSource(fn=kernel, signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "N": "constexpr"}, constexprs={"N": 32}),
            target=target,
        )
    print(f"all {len(fns)} whitelisted math functions compile for CUDA and HIP ✓")


def main():
    test_math_lowering()
    test_constant_folding()
    test_builtins()
    test_unsupported_raises()
    test_top_level_spike()
    test_whitelist_compiles_on_both_backends()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
