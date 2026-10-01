"""Push generated kernels through Triton's real compiler, no GPU needed."""

import ast
import importlib.util
import pathlib
import sys
import tempfile

_CONSTS = {"GROUP_SIZE": 32, "SPIKE_BLOCK": 4, "LINK_BLOCK": 16, "LOCAL_SPIKE_CAP": 64}


def load_jit(source: str, label: str, name: str = "update_kernel"):
    """Import generated source as a real module (triton.jit needs a file to read)."""
    path = pathlib.Path(tempfile.mkdtemp()) / f"kernel_{label}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"kernel_{label}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def compiles(module: ast.Module, label: str, targets=None) -> None:
    """Compile an update kernel for fake GPU targets; raises on a frontend/IR error.

    Generated kernels were once only ever checked as source text, and turned out
    not to compile at all (integer-indexing a tensor is unsupported).
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = load_jit(
        "import math\nimport triton\nimport triton.language as tl\n\n" + ast.unparse(module) + "\n", label
    )
    int_ptrs = ("node_ptr", "indices", "spike_src", "spike_conn", "count")
    sig = {
        a: "constexpr" if a in _CONSTS
        else "i32" if a in ("n_neurons", "n_spikes")
        else "fp32" if a == "t"
        else "*i64" if "offsets" in a
        else "*i32" if any(k in a for k in int_ptrs)
        else "*fp32"
        for a in fn.arg_names
    }
    consts = {k: v for k, v in _CONSTS.items() if k in sig}
    for target in targets or [GPUTarget("cuda", 80, 32)]:
        triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=target)
