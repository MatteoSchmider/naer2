import ast

from .context import TranspilerContext

_EXPR = lambda src: ast.parse(src, mode="eval").body
_STMTS = lambda src: ast.parse(src).body

_SPMD_BOILERPLATE = """\
pid = tl.program_id(0)
offsets = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
mask = offsets < n_neurons
"""


class KernelBuilder:
    """
    Assembles a Triton kernel ast.Module for a single population group.

    The caller (Population.compile) adds sections in execution order via
    add_section() — the builder is agnostic to whether a section came from
    an update, receive, or send hook. Population and Node own that ordering.

    The only thing the builder needs from the context is dynamic_attrs, which
    determines the kernel's pointer parameters and the load/store framing
    around the section bodies. Additional args (spike buffers, weight pointers
    for connections) will be added to the signature by the caller as needed.

    Generated signature:
        @triton.jit
        def population_kernel(
            <attr>_ptr, ...,   # one pointer per dynamic attr
            n_neurons,
            GROUP_SIZE: tl.constexpr,
        ):
    """

    def __init__(self, context: TranspilerContext):
        self.context = context
        self._sections: list[list[ast.stmt]] = []
        self._extra_args: list[str] = []

    def add_arg(self, name: str) -> "KernelBuilder":
        """Add a pointer arg to the kernel signature without auto load/store."""
        self._extra_args.append(name)
        return self

    def add_section(self, stmts: list[ast.stmt]) -> "KernelBuilder":
        """Append a block of hook statements to the kernel body in order."""
        self._sections.append(stmts)
        return self

    def build(self) -> ast.Module:
        body = list(_STMTS(_SPMD_BOILERPLATE))

        for attr in self.context.dynamic_attrs:
            body += _STMTS(f"{attr} = tl.load({attr}_ptr + offsets, mask=mask)")

        for section in self._sections:
            body.extend(section)

        for attr in self.context.dynamic_attrs:
            body += _STMTS(f"tl.store({attr}_ptr + offsets, {attr}, mask=mask)")

        func = ast.FunctionDef(
            name="population_kernel",
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    *[
                        ast.arg(arg=f"{attr}_ptr")
                        for attr in self.context.dynamic_attrs
                    ],
                    *[ast.arg(arg=f"{name}_ptr") for name in self._extra_args],
                    ast.arg(arg="n_neurons"),
                    ast.arg(arg="GROUP_SIZE", annotation=_EXPR("tl.constexpr")),
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[_EXPR("triton.jit")],
            returns=None,
        )

        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        return module
