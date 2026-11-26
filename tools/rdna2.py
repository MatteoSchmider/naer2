"""Run Triton 3.9 on RDNA2 (gfx103x), which its AMD backend no longer lists.

Triton's MLIR lowering refuses gfx1030, so it is run as gfx1100 (RDNA3, also
wave32; nothing naer2's kernels use differs -- tl.dot / WMMA would not work),
while everything core-LLVM (data layout, O3, instruction selection,
assembly) targets the real arch. Anything the lowering emits that RDNA2
can't do fails in LLVM at compile time, not at run time.

Import before compiling anything. The RX 6600 eGPU (gfx1032, ROCr node 2)
also needs, for ROCm and torch's gfx1030 code objects:

    HSA_OVERRIDE_GFX_VERSION_2=10.3.0 HIP_VISIBLE_DEVICES=1

(the plain HSA_OVERRIDE_GFX_VERSION breaks ROCr on this machine).
"""

import dataclasses

from triton import knobs

LOWER_AS = "gfx1100"


def _hook(self=None, stages=None, options=None, language=None, capability=None):
    if self is None:  # asked for the cache key
        return f"rdna2-as-{LOWER_AS}", LOWER_AS
    real = options.arch
    if not real.startswith("gfx103"):
        return
    lower = dataclasses.replace(options, arch=LOWER_AS)
    object.__setattr__(lower, "base_arch", real)  # make_llir's core-LLVM steps use base_arch
    if "ttir" in stages:
        stages["ttir"] = lambda src, md: self.make_ttir(src, md, lower)
        stages["ttgir"] = lambda src, md: self.make_ttgir(src, md, lower)
    else:  # Gluon
        stages["ttgir"] = lambda src, md: self.gluon_to_ttgir(src, md, lower)
    stages["llir"] = lambda src, md: self.make_llir(src, md, lower)
    # amdgcn and hsaco keep the real options


knobs.runtime.add_stages_inspection_hook = _hook
