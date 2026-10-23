# naer2 — status, findings, and plan

Last updated after the test-suite restructure. See README.md for the directory
layout. Run everything with `python -m pytest`.

## Planned order of work

1. **Spike management handling** (host side): turn each population's compacted
   output `(node, time)` buffer into the per-connection merged, time-sorted input
   buffers `(spike_time, spike_src, spike_conn)` the update kernel consumes
   (`Population._execute_gather_kernel` / `_execute_update_kernel` are still
   stubs). Includes delays, which the kernel currently ignores. To be designed together.
2. **A new test network**, exercised **on the CPU first**, at a small and a large
   scale, before anything touches a GPU-style path.
3. **Only then**: get kernels compiling under Triton and confirm their behavior
   matches the CPU results — a system-test tier (needs a machine actually
   capable of running them), deliberately not built yet. See README.md's "What's
   deliberately not here yet". Until it exists, kernel correctness is checked by
   unit and integration tests inspecting the generated AST.

## Done

- **Test suite restructured**: `unit/` (mirrors `naer2/`'s layout 1:1, pytest,
  no network built), `test/` (reusable nodes/edges/patterns/networks — a small
  components library versioned with naer2 — plus `test_*.py` integration tests
  that compile real networks and check kernel structure), `showcases/` (narrated
  demos, not part of the test suite). Dropped everything that invoked the real
  Triton compiler or ran a kernel (the CPU-interpreter semantics check, the
  per-backend compile check) — that's the system-test tier from item 3 above,
  intentionally not built yet. Old `test/misc/` scratch deleted (still in git
  history). See README.md.
- **Renames / fixtures.** `counter` node -> `relay`, counter test -> `coverage`.
  Population D takes three inputs through one receive port: A and B via
  `static_synapse`, C via `plastic_synapse` (different edge type -> different
  forward AST per connection).
- **Dispatch.** Per-spike if/elif on `spike_conn_j` (warp-uniform, so not
  divergent), one branch per connection with its own pointers and transpiled ASTs.
- **Spike output.** `n2.spike(time=None)` (bare -> kernel tick `t`; expression ->
  recorded verbatim). Each program writes into a private slice
  (`local_spike_*`, plain counter, no atomics) and flushes with **one**
  `atomic_add` per program at the end.
- **Duplicate spike emission fixed.** Cause was the fixture: `relay.pos_receive`
  spiked without resetting, so `send` (which runs after every receive) fired the
  same crossing again. The transpiler was lowering the hooks correctly.
  `send` is now the single source of truth for firing (two cases: far past
  threshold -> extra spike at `time + 1.0`; else a plain spike). Spikes from
  receive and from update are still covered, with thresholds that can never
  trigger (`1e6`), just to exercise the lowering. `test/test_coverage.py` pins
  the number of emission sites per link tile.
- **Link loop vectorized (option A).** A spike's row is walked `LINK_BLOCK`
  links at a time. Two lane spaces, kept apart by name: link lanes `lk_<attr>`
  (`[LINK_BLOCK]`, edge forward runs here and stores back) and neuron lanes
  `edge_<attr>` (`[GROUP_SIZE]`, what receive sees). A one-hot compare-reduce
  (`_hit`) turns "these links hit these neurons" into `is_target` plus per-neuron
  edge values, so receive/send run once per tile on register-resident state.
- **Blocked CSR (decided and implemented).** `Topology.blocked(group_size,
  n_targets)` -> `BlockedTopology`: links are laid out block-major (program `b` owns
  targets `[b*G, (b+1)*G)`), with an `offsets[n_blocks, N_src+1]` table of
  **absolute** positions. A program's slice of a source's row holds only its own
  links, so ownership is structural (no `_own` mask) and targets are stored
  block-local. Per-group tensors are zero-copy slices (`block_slice(b)`).
  Chosen over "separate tensors per group" because a launch has one pointer per
  argument, so separate tensors can't be indexed by `pid` without a pointer table
  or one launch per group; the flat layout *is* the concatenation of them.
  Per-link tensors go through `to_blocked` / `from_blocked` (permutation).
  Tests: `unit/test_topology.py` (layout faithful to canonical CSR).
- **Per-spike scalar loads.** Triton can't integer-index a tensor, so each
  unrolled spike slot loads `time`, `spike_src_j`, `spike_conn_j` directly.
- **Math lowering.** `MathLoweringTransformer` (runs in `HookTranspiler.emit()`
  after self-attr resolution, before `IfTransformer`) rewrites whitelisted
  `math.*` calls to per-element Triton ops: `exp exp2 log log2 sin cos sqrt floor
  ceil erf` -> `tl.*`; `tan tanh sinh cosh asin acos atan atan2 log1p expm1 pow
  hypot fmod asinh acosh atanh trunc copysign` -> `tl.extra.libdevice.*` (present
  on both CUDA and HIP; `cbrt` excluded, HIP lacks it); plus `log10`, `fabs`, and
  builtin `abs` / 2-arg `min` / `max`. All-constant calls (common once static attrs
  inline) are folded at transpile time; `math.pi/e/tau` fold too. An unlisted
  `math.*` call raises instead of leaving an undefined name. Note `floor`/`ceil`/
  `trunc` return floats in Triton. The whitelist was manually verified against
  real Triton (both a CUDA and a gfx1100 target) while developing this, but that
  check isn't part of the live suite (system-test tier); if the whitelist grows,
  re-verify by hand. Tests: `unit/transpiler/transformers/test_math_lowering_transformer.py`.
- **Top-level `n2.spike()`** (outside any `if`) fires every lane the hook runs on:
  the root mask, i.e. the whole group in update / final-pass send, and just the
  delivered neurons inside a per-link receive/send (via `MaskAliasTransformer`).
  Tests: `unit/transpiler/transformers/test_if_transformer.py`.
- **Edge forward if-branches** now root at the link tile's validity mask
  (`_lk_valid`) via `MaskAliasTransformer(alias)`, not the neuron-group mask.
- **Edge attr / source attr resolution**: `UsedMemberFinder` decides what to load;
  static attrs inline as literals; `Fabric.tensor_attrs` = dynamic + externally
  overridden.

## Findings

- **The generated kernels had never been compiled by Triton** and originally did
  not compile at all (`unsupported tensor index` on `x[j]`), confirmed by manually
  compiling the coverage kernels A-D through a fake-sm_80 target (frontend -> IR
  -> PTX) while developing the blocked layout and scalar per-spike loads. Not
  re-checked automatically now (system-test tier) — the structural fix (no more
  integer tensor indexing) is what the integration tests assert on instead.
- **The old (pre-blocked-CSR) serial link loop had a real bug**: every program
  walked every link and ran the plastic weight update and store on it, so all P
  programs updated each link. Found via a negative control in a CPU-interpreter
  harness (weights came out at 1.6 instead of 1.2) while developing the blocked
  layout; that harness is gone (system-test tier) but the bug was structurally
  fixed by the blocked layout itself (ownership is now positional, not a runtime
  mask) and is covered by `unit/test_topology.py` + `test/test_coverage.py`'s
  `_own`-mask-absence assertions.
- *(Resolved)* mini_ei's kernels didn't compile because PD14's update hook uses
  `math.exp` on tensors; now lowered, and E and I compile. A bare top-level
  `n2.spike()` used to pass through untouched; now lowered.
- `relay.neg_receive`'s explicit-time rebound spike was removed on disk after I
  added it (deliberate, by the owner); the explicit-time form is now covered
  through `send` instead.

## Open items / decisions

- **`GROUP_SIZE` must be fixed per population** — the blocked layout is
  meaningless for any other value. Where it lives (population attribute?), and
  when layouts are derived (lazily, cached per `(connection, G)`), is undecided.
- **Canonical vs blocked storage of per-link state.** User-facing tensors
  (`conn.fabric.weight = ...`) are in canonical CSR order; the kernel needs
  blocked order. Options: keep canonical and permute on every launch/readback, or
  make blocked the internal source of truth and permute on user access. Plastic
  weights written by the kernel need `from_blocked` to be read back.
- **Target index dtype**: block-local indices fit `uint8`/`int16` for
  `GROUP_SIZE <= 256 / 65536`, cutting the target array's memory by 2-4x — matters
  at billions of synapses. Currently `int32`.
- **Offsets table scales as `n_blocks * N_src`** per connection. Fine at PD14
  scale; worth revisiting for very large populations.
- **Multapses** (duplicate targets in one row): within a tile, a neuron's links
  are summed into one delivery, and forward runs per link. Fine if patterns never
  produce duplicates; not enforced anywhere.
- **`LOCAL_SPIKE_CAP`** (power of 2) and `n2.BUFFER_MULTIPLIER` have no overflow
  checks; per-program scratch and the output buffer are sized by the host.
- **Per-link delays are ignored** in the kernel (all deliveries use the spike's
  arrival time). Edge **backward** is unimplemented (needs reverse CSR).
- **`Connection @ mapping` without parens** (`>> B @ {...}`) doesn't mean
  post-target mapping because `@` binds tighter than `>>`; TODOs in
  `Population.__matmul__` / `Connection.__matmul__`.
- A warp-uniform `if any(is_target)` around receive/send per tile is possible, but
  much less useful now that a program's tile lanes are all its own links.
- Nothing has run on a GPU, under the Triton interpreter, or been timed, since
  the last restructure removed that tier from the live suite — see README.md.
- Caching/preloading connection data (source attrs, offsets) per spike block may
  pay off in some networks; needs benchmarking, likely network-dependent.
- **Gap, not yet started**: `AttrPrefixTransformer`, `SelfAttrTransformer`,
  `TargetMaskTransformer`, and `UsedMemberFinder`/`AssignedMemberFinder` have no
  direct `unit/` tests yet — only exercised indirectly through `test/test_*.py`'s
  whole-network compiles. `unit/transpiler/transformers/` and
  `unit/transpiler/analyzers/` are the natural homes once written.
