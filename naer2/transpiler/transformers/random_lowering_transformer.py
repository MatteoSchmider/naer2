import ast

from .base import StmtTransformer

# Python random.* call name -> which distribution it lowers to. Only the two
# broadly useful ones for now, same "whitelist or raise" policy as
# MathLoweringTransformer, rather than leaving an undefined `random` name in
# the kernel.
_UNIFORM = {"random"}
_GAUSSIAN = {"gauss", "normalvariate"}


class RandomLoweringTransformer(StmtTransformer):
    """Lowers Python random.* calls in hook bodies to per-lane Triton RNG.

    Structurally the same idea as MathLoweringTransformer (same pipeline
    position — before IfTransformer, for the same reason: it unparses branch
    tests and spike-time expressions into new statements, so calls appearing
    there must already be lowered — and the same "whitelist or raise"
    policy), for a different problem: math.exp(x) fails because it's not
    vectorized; random.random() fails because Python's random module is one
    host-side stream with no notion of "one value per GPU lane" at all.

        random.random()                  -> tl.rand(seed, offs)
        random.gauss(mu, sigma)
        random.normalvariate(mu, sigma)  -> mu + sigma * tl.randn(seed, offs)

    Triton's RNG (Philox, a counter-based PRNG: same (seed, offset) always
    reproduces the same output, different inputs look statistically
    independent) has no per-lane state to carry across calls, so every draw
    is reseeded from scratch:
      - offs (the lane's global neuron index — every kernel already computes
        this once, see UpdateKernelTemplate) makes different lanes in the
        same call draw different numbers.
      - t (the kernel's own per-call time-boundary argument — also already
        in scope everywhere, node hook or spike loop alike — bit-
        reinterpreted as an int) makes the same lane draw a different number
        on a later call. Verified against torch/triton directly: same (t,
        offs) reproduces bit-for-bit, a different t does not.
      - a per-call-site salt (this transformer numbers each random.* call
        site in one hook body starting at 0, XORed into the seed) keeps two
        DIFFERENT random.random() calls in the same hook from returning the
        same value.

    KNOWN LIMITATION, accepted for now — see TODO.md: seed is derived only
    from t (plus the call-site salt), with no population identity folded in,
    so two different populations' kernels draw IDENTICAL numbers for the
    same lane index at the same t — randomness only differs within one
    population's own kernel, not network-wide. TODO.md tracks the fix (each
    population owning its own public seed).

    ALSO NOT HANDLED: `offs` is GROUP_SIZE-shaped (one lane per neuron);
    nothing here checks which lane space the surrounding code actually runs
    in. Fine for node hooks (update/send/receive, which is all that's
    exercised so far — see test/networks/PD14/poisson.py), wrong if ever
    reached from an edge forward/backward hook, which runs on LINK_BLOCK-
    shaped lanes instead. No edge hook calls random.* yet.
    """

    def __init__(self):
        self._call_site = 0

    def _seed_expr(self) -> str:
        salt = self._call_site
        self._call_site += 1
        base = "t.to(tl.int32, bitcast=True)"
        return base if salt == 0 else f"({base} ^ {salt * 0x9E3779B1})"

    @staticmethod
    def _is_random(func: ast.expr) -> bool:
        return (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "random"
        )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if node.keywords or not self._is_random(node.func):
            return node

        name = node.func.attr
        seed = self._seed_expr()

        if name in _UNIFORM:
            if node.args:
                raise ValueError("random.random() takes no arguments")
            return ast.copy_location(
                ast.parse(f"tl.rand({seed}, offs)", mode="eval").body, node
            )

        if name in _GAUSSIAN:
            if len(node.args) != 2:
                raise ValueError(f"random.{name}(mu, sigma) expects exactly 2 arguments")
            mu, sigma = (ast.unparse(a) for a in node.args)
            return ast.copy_location(
                ast.parse(f"({mu}) + ({sigma}) * tl.randn({seed}, offs)", mode="eval").body,
                node,
            )

        raise ValueError(
            f"random.{name}() has no per-lane Triton lowering; supported: "
            f"{sorted(_UNIFORM | _GAUSSIAN)}"
        )
