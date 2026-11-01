# naer2 — status, findings, and plan

Last updated after the test-suite restructure. See README.md for the directory
layout. Run everything with `python -m pytest`.

## Planned order of work

1. **Spike management handling** (host side): turn each population's compacted
   output `(node, time)` buffer into the per-connection merged, time-sorted input
   buffers `(spike_time, spike_src, spike_conn)` the update kernel consumes
   (`Population._execute_gather_kernel` / `_execute_update_kernel` are still
   stubs). Includes delays, which the kernel currently ignores. To be designed together.
   - **In progress**: source side stays blocked, not flushed (reasoned, not
     benchmarked — no GPU here; see "Open items" below). CPU reference +
     window semantics done and tested: `naer2/spike_gather.py`
     (`BlockedSpikes`, `merge_blocked_spikes(sources, window)`), tests in
     `unit/test_spike_gather.py`. `window = (t_lo, t_hi]` excludes both
     not-yet-due (delayed) and already-consumed spikes; the function only
     reads `sources`, never mutates them — see the lifecycle item below.
     Current implementation is deliberately the simplest correct one
     (concatenate + one full sort), the baseline to beat. Algorithm for the
     actual kernel: direction settled on a k-way merge (exploits that
     connections-per-population is small, 1-dozens, and each stream is
     already near-sorted) over a counting/bucket sort (doesn't generalize to
     event-driven windows the way it did for the old Naer's fixed-dt-grid
     case, since naer2 uses float time and supports both clock- and
     event-driven scheduling).
     `merge_blocked_spikes_kway` (same file) is a second reference,
     structured as a merge tree (`ceil(log2(k))` rounds of pairwise merges
     over per-connection sorted streams) instead of one flat sort, tested
     against the flat baseline for agreement across k=0..5 including the
     odd-k carry-forward case. It does NOT yet implement either GPU-specific
     technique the tree is meant to host (per-stream odd-even-transposition
     sort exploiting near-sortedness; pairwise merge-path avoiding a full
     re-sort) — both still fall back to a generic sort on CPU, so this only
     validates the tree's *structure*, not yet any performance benefit.
   - **Stay on GPU — reasoned conclusion, see below for the numbers.** CPU
     sort time was actually measured here (no GPU to measure the other
     side); GPU kernel-launch/PCIe figures are well-documented reference
     values, not measured. At today's PD14-ish scale (tens of spike events
     per population per window) the comparison is noise either way. At the
     scale this library targets — reasoned in the conversation to land at
     roughly hundreds of thousands of spike events per timestep *network-wide*
     even for networks much larger than anything simulated before, because
     firing rate and timestep size are both biologically bounded to a narrow
     range while GPU memory (dominated by synapse storage, not neuron count)
     is the actual ceiling on how large a network fits at all — measured CPU
     sort time at N=100,000 is ~1.3ms best case (near-sorted) to ~6.5ms worst
     case (random), while a well-batched GPU round trip is estimated at
     roughly 150-400us, and an on-device kernel avoiding the round trip
     entirely would be expected (not measured — no GPU here) to stay in the
     tens-of-microseconds range regardless, since GPU parallelism absorbs
     N-growth far better than a CPU thread count that measurably didn't help
     here (multi-threaded torch sort was barely faster than single-threaded
     for this operation). So CPU sorting is at best a wash at tiny N and a
     clear loser at the N this library is actually for. If CPU sorting is
     ever revisited anyway: batch all per-connection transfers into one call
     per population, not one per connection — paying the fixed per-transfer
     latency `k` times (k = connections) was the dominant cost in the
     round-trip estimate, not bandwidth, at realistic k.
   - **New: parallelize kernel execution across populations.** `Network.step()`
     (`naer2/network.py`) is a plain host-side Python `for pop in
     self.populations: pop.advance(...)` loop — fully sequential, one
     population's kernel launches at a time, with no batching across
     populations into a single launch. This is a separate, likely larger
     latency lever than how spike management sorts, since every population
     pays its own full kernel-launch overhead in sequence regardless. Not
     started; flagged here per request.
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
- **Spike buffer lifecycle is not handled anywhere yet.** A population's raw
  spike output can hold entries not yet due (an explicit `n2.spike(time +
  delay)`) that must survive to a later call, and entries already consumed
  that must not be double-counted. `merge_blocked_spikes`'s `window` filters
  correctly for one call, but nothing clears consumed entries or compacts the
  buffer across calls — that can only be done by the population the spikes
  belong to (it alone knows what it's already handed out), not by the merge
  step. Currently assuming (not yet confirmed) that every outstanding delay
  resolves within one update window, so there's never anything left over to
  carry forward; true carry-over, if needed, is unimplemented.
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

# Human written:
## Refactoring Ideas
- delete BlockedSpikes, just use some plain tensors
- see if same can be done with Topology
- see if something similar can be done with regions/positions to make them lighter (doubt it though in that case)
- edges, nodes, patterns, updates, receives, kernel, fabric all take up single files, maybe they can be packaged more tightly file-wise and reuse some things?