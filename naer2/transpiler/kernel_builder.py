import ast

from .context import TranspilerContext


def _tl(attr: str) -> ast.Attribute:
    """tl.<attr>"""
    return ast.Attribute(
        value=ast.Name(id="tl", ctx=ast.Load()),
        attr=attr,
        ctx=ast.Load(),
    )


def _call(func: ast.expr, *args: ast.expr, **kwargs: ast.expr) -> ast.Call:
    return ast.Call(
        func=func,
        args=list(args),
        keywords=[ast.keyword(arg=k, value=v) for k, v in kwargs.items()],
    )


def _name(id: str, store: bool = False) -> ast.Name:
    return ast.Name(id=id, ctx=ast.Store() if store else ast.Load())


def _ptr(attr: str) -> ast.Name:
    return _name(f"{attr}_ptr")


def _load_stmt(attr: str) -> ast.Assign:
    """
    potential = tl.load(potential_ptr + offsets, mask=mask)

    Emitted before each hook body for every dynamic attr. Loads the entire
    group's values into a local block-level variable for the hook to operate on.
    """
    return ast.Assign(
        targets=[_name(attr, store=True)],
        value=_call(
            _tl("load"),
            ast.BinOp(left=_ptr(attr), op=ast.Add(), right=_name("offsets")),
            mask=_name("mask"),
        ),
    )


def _store_stmt(attr: str) -> ast.Expr:
    """
    tl.store(potential_ptr + offsets, potential, mask=mask)

    Emitted after each hook body for every dynamic attr. Writes modified values
    back to global memory. A later optimisation pass can elide stores for attrs
    that the hook only reads.
    """
    return ast.Expr(
        value=_call(
            _tl("store"),
            ast.BinOp(left=_ptr(attr), op=ast.Add(), right=_name("offsets")),
            _name(attr),
            mask=_name("mask"),
        )
    )


class KernelBuilder:
    """
    Assembles a Triton kernel ast.Module for a single population group.

    Hooks are added in simulation order (update → receive → send). Each
    contributes a block of statements sandwiched between the shared boilerplate
    that computes SPMD offsets and loads/stores dynamic attr tensors.

    Only update is wired in the first vertical slice; receive and send hooks
    will slot in between the load and store sections as they are implemented.

    Generated kernel signature:
        def population_kernel(
            <attr>_ptr, ...,   # one pointer per dynamic attr
            n_neurons,         # total population size (for masking)
            GROUP_SIZE: tl.constexpr,
        ):
    """

    def __init__(self, context: TranspilerContext):
        self.context = context
        self._update_stmts: list[ast.stmt] = []

    def add_update(self, stmts: list[ast.stmt]) -> "KernelBuilder":
        self._update_stmts = stmts
        return self

    def build(self) -> ast.Module:
        body: list[ast.stmt] = []

        # pid = tl.program_id(0)
        body.append(
            ast.Assign(
                targets=[_name("pid", store=True)],
                value=_call(_tl("program_id"), ast.Constant(value=0)),
            )
        )

        # offsets = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        body.append(
            ast.Assign(
                targets=[_name("offsets", store=True)],
                value=ast.BinOp(
                    left=ast.BinOp(
                        left=_name("pid"),
                        op=ast.Mult(),
                        right=_name("GROUP_SIZE"),
                    ),
                    op=ast.Add(),
                    right=_call(
                        _tl("arange"), ast.Constant(value=0), _name("GROUP_SIZE")
                    ),
                ),
            )
        )

        # mask = offsets < n_neurons
        body.append(
            ast.Assign(
                targets=[_name("mask", store=True)],
                value=ast.Compare(
                    left=_name("offsets"),
                    ops=[ast.Lt()],
                    comparators=[_name("n_neurons")],
                ),
            )
        )

        # Load all dynamic attrs
        for attr in self.context.dynamic_attrs:
            body.append(_load_stmt(attr))

        # Hook bodies (update only for now)
        body.extend(self._update_stmts)

        # Store all dynamic attrs
        for attr in self.context.dynamic_attrs:
            body.append(_store_stmt(attr))

        # Kernel function arguments
        constexpr_annotation = _tl("constexpr")
        func = ast.FunctionDef(
            name="population_kernel",
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    *[
                        ast.arg(arg=f"{attr}_ptr")
                        for attr in self.context.dynamic_attrs
                    ],
                    ast.arg(arg="n_neurons"),
                    ast.arg(arg="GROUP_SIZE", annotation=constexpr_annotation),
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[_tl("jit")],  # triton.jit added by KernelBuilder caller
            returns=None,
        )

        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        return module
