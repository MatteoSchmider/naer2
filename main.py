import logging

logging.basicConfig(
    format="{%(filename)s:%(lineno)d} %(levelname)s - %(message)s", level=logging.DEBUG
)
# to let 780M (gfx1103) be used: export HSA_OVERRIDE_GFX_VERSION=11.0.0

import runpy

runpy.run_module("test.mini_ei", run_name="__main__")
runpy.run_module("test.coverage", run_name="__main__")
runpy.run_module("test.blocked", run_name="__main__")
runpy.run_module("test.lowering", run_name="__main__")
