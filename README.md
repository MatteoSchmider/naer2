# naer2 — directory layout

Reference for where things go. Read this before adding a new file.

## Top-level directories

```
naer2/        the library
test/         reusable components (nodes/edges/patterns/networks) + integration tests
unit/         unit tests — mirrors naer2/'s layout 1:1
showcases/    narrated demos — run and read, not part of the test suite
```

### `naer2/` — the library

The package. No tests live here.

### `test/` — components + integration tests

`test/nodes/`, `test/edges/`, `test/patterns/`, `test/networks/` are **not test
scaffolding** — they're a small set of standard components built with naer2
(LIF variants, synapse types, connectivity patterns, example networks like
PD14 and the mini_ei / coverage fixtures), versioned and tested alongside the
library itself. Integration tests, unit tests, and showcases all consume them.

`test/test_*.py` are the integration tests: each builds a real network through
the full node/edge/pattern/connection chain, compiles it, and asserts on the
generated kernel's structure (dispatch shape, masking, signature, etc.) by
inspecting the AST. They do **not** invoke the real Triton compiler or run
anything — see "What's deliberately not here yet" below.

Run: `python -m pytest test/`

### `unit/` — unit tests

One file per thing under test, with no Population/Connection/network built —
construct the object or AST directly. **Mirrors `naer2/`'s layout 1:1**:
`naer2/transpiler/transformers/math_lowering_transformer.py` has its test at
`unit/transpiler/transformers/test_math_lowering_transformer.py`. Only create
the paths a real test needs — don't pre-create empty stubs for untested
modules. When something currently only gets exercised indirectly through a
`test/test_*.py` integration test, that's a gap, not a substitute; see
TODO.md.

Run: `python -m pytest unit/`

### `showcases/` — narrated demos

Human-oriented walkthroughs: build something, print it, read the output to
understand the API. Light `assert`s for internal sanity are fine, but these
are not part of the regression suite and aren't run by pytest.

Run individually, e.g. `python -m showcases.pd14`.

## Running everything

```
python -m pytest          # unit/ + test/ — the actual regression suite
python -m showcases.pd14  # one showcase at a time, by hand
```

`main.py` is not a test runner — it's reserved for running naer2 on real
hardware.

## What's deliberately not here yet

Nothing here invokes the real Triton compiler or runs a kernel (interpreted
or on a GPU). Compiling and especially *running* a network is a later,
separate **system-test** tier: it needs a machine actually capable of it, and
its purpose is performance / numerical correctness once there's a real
execution pipeline (today, `Population._execute_update_kernel` etc. are still
stubs). Until then, kernel correctness is checked by unit and integration
tests inspecting the generated AST — that's the right bar for now. See
TODO.md for what this would need before it's worth adding, and for anything
that was deliberately dropped to get here.
