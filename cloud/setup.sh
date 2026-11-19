#!/usr/bin/env bash
# Sets a fresh cloud GPU instance up for naer2: made for gpu.ai's PyTorch base image (Ubuntu 22.04, Python
# 3.10, root), which has neither the Python naer2 needs (3.12) nor a C compiler. Run from the repository
# clone, any number of times (it skips what is done):
#
#   bash cloud/setup.sh              # set up
#   bash cloud/setup.sh --test       # set up, then run the test suite (gap tests excluded)
#
# What it does:
#   - gcc from apt: Triton compiles a small C launcher for every kernel it builds
#   - on NVIDIA, compute-sanitizer and cuda-gdb for the image's CUDA toolkit (NVIDIA's apt packages)
#   - uv (pip) and, through it, Python 3.12 (naer2 uses PEP 695 `type` statements) in .venv
#   - cloud/requirements.txt into .venv: torch from TORCH_INDEX (default CUDA 12.8), Triton 3.9 over the
#     version torch pins (naer2's kernels are Gluon)
# The system Python and its packages are left alone. Afterwards: .venv/bin/python -m pytest, or
# source .venv/bin/activate.
#
# From a local checkout: cloud/remote (sync this working tree there, run setup / tests / scripts).

set -euo pipefail

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

step() { printf '\n== %s\n' "$*"; }

step "C compiler"
if ! command -v gcc >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq gcc libc6-dev >/dev/null
fi
gcc --version | head -1

step "CUDA debugging tools (compute-sanitizer, cuda-gdb)"
# NVIDIA: the packages for the image's CUDA toolkit (/usr/local/cuda-X.Y), from NVIDIA's apt repository --
# memcheck / initcheck for illegal or uninitialized accesses, cuda-gdb for GPU core dumps
# (CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1); Triton's kernels carry line info, so both name the fused.py line
cuda_dir="$(ls -d /usr/local/cuda-[0-9]*.[0-9]* 2>/dev/null | sort -V | tail -1 || true)"
if [ -n "$cuda_dir" ] && command -v nvidia-smi >/dev/null; then
    ver="${cuda_dir##*/cuda-}"; pkg="${ver/./-}"
    if [ ! -x "$cuda_dir/bin/compute-sanitizer" ] || [ ! -x "$cuda_dir/bin/cuda-gdb" ]; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get install -y -qq "cuda-sanitizer-$pkg" "cuda-gdb-$pkg" >/dev/null 2>&1 \
            || { apt-get update -qq && apt-get install -y -qq "cuda-sanitizer-$pkg" "cuda-gdb-$pkg" >/dev/null 2>&1; } \
            || echo "(not installable here: cuda-sanitizer-$pkg / cuda-gdb-$pkg -- skipped)"
    fi
    for tool in compute-sanitizer cuda-gdb; do
        [ -x "$cuda_dir/bin/$tool" ] && echo "$tool: $cuda_dir/bin/$tool"
    done
else
    echo "(no CUDA toolkit in /usr/local or no NVIDIA GPU: skipped)"
fi

step "uv"
if ! command -v uv >/dev/null; then
    python3 -m pip install -q uv
fi
uv --version

step "Python 3.12 venv (.venv)"
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    uv venv -q -p 3.12 .venv
fi

step "packages (cloud/requirements.txt; torch from $TORCH_INDEX)"
overrides="$(mktemp)"
grep '^triton' cloud/requirements.txt > "$overrides"
uv pip install -q -p .venv/bin/python -r cloud/requirements.txt --override "$overrides" \
    --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
rm -f "$overrides"

step "check"
.venv/bin/python - <<'EOF'
import sys
import torch, triton
from triton.experimental import gluon  # noqa: F401 -- naer2's kernels need it
print(f"Python {sys.version.split()[0]}, torch {torch.__version__}, triton {triton.__version__}")
if not torch.cuda.is_available():
    sys.exit("no GPU visible to torch")
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GiB, {p.multi_processor_count} multiprocessors")
EOF

if [ "${1:-}" = "--test" ]; then
    step "test suite"
    .venv/bin/python -m pytest -q -p no:cacheprovider -k "not gap"
fi
