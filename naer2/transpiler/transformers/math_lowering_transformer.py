import ast
import math
import operator

from .base import StmtTransformer

# Python math function -> the per-element Triton equivalent. Everything here
# exists on both the CUDA and HIP backends (tl.extra.libdevice dispatches to
# whichever is being compiled for).
_CORE = {name: f"tl.{name}" for name in (
    "exp", "exp2", "log", "log2", "sin", "cos", "sqrt", "floor", "ceil", "erf",
)}
_LIBDEVICE = {name: f"tl.extra.libdevice.{name}" for name in (
    "tan", "tanh", "sinh", "cosh", "asin", "acos", "atan", "atan2",
    "log1p", "expm1", "pow", "hypot", "fmod", "asinh", "acosh", "atanh",
    "trunc", "copysign",
)}
MATH_LOWERING = {**_CORE, **_LIBDEVICE, "fabs": "tl.abs"}

# Builtins that act on tensors elementwise; min/max only in their 2-argument form.
_BUILTINS = {"abs": ("tl.abs", 1), "min": ("tl.minimum", 2), "max": ("tl.maximum", 2)}

_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_INV_LN10 = 1.0 / math.log(10.0)


def _const_value(node: ast.expr):
    """The value of a constant-only expression tree, or raise ValueError."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _const_value(node.operand)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp):
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.Pow: operator.pow}
        if type(node.op) in ops:
            return ops[type(node.op)](_const_value(node.left), _const_value(node.right))
    raise ValueError("not a constant expression")


class MathLoweringTransformer(StmtTransformer):
    """Lowers Python math calls in hook bodies to per-element Triton ops.

    Hook bodies are written as ordinary Python, but they run on [GROUP_SIZE]
    tensors, where `math.exp(x)` fails ("must be real number, not tensor").
    Each whitelisted call is rewritten to its Triton counterpart:

        math.exp(x)          →  tl.exp(x)
        math.tanh(x)         →  tl.extra.libdevice.tanh(x)
        math.log10(x)        →  tl.log(x) * 0.4342944819032518
        abs(x) / min(a, b)   →  tl.abs(x) / tl.minimum(a, b)

    A call whose arguments are all compile-time constants (typically because
    static attributes were inlined) is evaluated here with Python's math instead
    — tl.exp doesn't accept a bare Python float. math.pi / e / tau fold too.
    Any other math.* call has no known per-element lowering and raises, rather
    than leaving an undefined `math` name in the kernel.

    Note math.floor / ceil / trunc return floats in Triton, not ints.
    """

    @staticmethod
    def _is_math(func: ast.expr) -> bool:
        return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "math"

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "math" and node.attr in _CONSTANTS:
            return ast.copy_location(ast.Constant(value=_CONSTANTS[node.attr]), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if node.keywords:
            return node

        if self._is_math(node.func):
            name = node.func.attr
            try:
                values = [_const_value(a) for a in node.args]
                return ast.copy_location(ast.Constant(value=getattr(math, name)(*values)), node)
            except (ValueError, TypeError, AttributeError, OverflowError):
                pass
            if name == "log10" and len(node.args) == 1:
                return ast.copy_location(
                    ast.parse(f"tl.log({ast.unparse(node.args[0])}) * {_INV_LN10!r}", mode="eval").body, node
                )
            if name not in MATH_LOWERING:
                raise ValueError(
                    f"math.{name}() has no per-element Triton lowering; supported: "
                    f"{sorted(MATH_LOWERING) + ['log10']}"
                )
            return ast.copy_location(
                ast.Call(func=ast.parse(MATH_LOWERING[name], mode="eval").body, args=node.args, keywords=[]), node
            )

        if isinstance(node.func, ast.Name) and node.func.id in _BUILTINS:
            target, arity = _BUILTINS[node.func.id]
            if len(node.args) != arity:
                return node
            try:
                values = [_const_value(a) for a in node.args]
                return ast.copy_location(ast.Constant(value={"abs": abs, "min": min, "max": max}[node.func.id](*values)), node)
            except ValueError:
                pass
            return ast.copy_location(
                ast.Call(func=ast.parse(target, mode="eval").body, args=node.args, keywords=[]), node
            )
        return node
