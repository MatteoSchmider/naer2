import logging

logging.basicConfig(
    format="{%(filename)s:%(lineno)d} %(levelname)s - %(message)s", level=logging.DEBUG
)
# to let 780M (gfx1103) be used: export HSA_OVERRIDE_GFX_VERSION=11.0.0

# The test suite runs under pytest now (see README.md): `python -m pytest`.
# This entry point is reserved for running naer2 on real hardware.
