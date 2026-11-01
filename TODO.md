# naer2 — status, findings, and plan

Last updated after the test-suite restructure. See README.md for the directory
layout. Run everything with `python -m pytest`.

## Planned order of work

1. **Spike management handling** (host side): turn each population's compacted
   output `(node, time)` buffer into the per-connection merged, time-sorted input
   buffers `(spike_time, spike_src, spike_conn)` the update kernel consumes.
   **Done — real kernels, really launched, driven automatically by
   `Network.step()`.** Includes delays in the sense that a spike's recorded
   time (whatever the send hook's `n2.spike(...)` expression computed) is
   respected; per-link delay — every link out of a firing source arriving at
   the same instant regardless of the link's own delay value — is still the
   known, intentional approximation noted lower in this file and in
   `update_kernel.py`'s module docstring, unchanged by this item.
   - **Done — two real Triton kernels, wired into Population/Network.**
     `naer2/spike_gather.py`, tests in `test/test_spike_gather_gpu.py`
     (GPU-only, no CPU reference — see the "Stay on GPU" entry below for why
     one wasn't worth building this time).
     Split into two kernels once lifecycle (clearing consumed spikes,
     retaining not-yet-due ones) turned out to change where sorting belongs:
     - `compact()` — owned by one source population. Folds this call's fresh,
       GROUP-blocked output (`BlockedSpikes`) into whatever it's still holding
       from previous calls (`CarryBuffer`, a persistent, already-sorted,
       cross-call buffer the source population would own for real), splits
       the combined result on `t_hi` into "due now" (returned, flat, sorted)
       and "not yet due" (written back into `CarryBuffer` in place). Because
       nothing `<= t_hi` ever survives a compaction, the old `window = (t_lo,
       t_hi]` is gone — there's never anything left below the new floor, so
       only `t_hi` is needed now.
     - `merge()` — owned by one target population. Takes several already-
       `compact()`ed sources (one per incoming connection), tags each with
       that connection's id, interleaves them into the sorted `(time, src,
       conn)` buffer the target's own kernel consumes. Never re-sorts a
       source stream, since `compact()` already sorted it once — which
       matters because one source commonly feeds multiple targets (e.g. one
       population feeding two different connections), and sorting it once at
       the source rather than once per downstream consumer is the entire
       point of the split.
     Both are single-block (`grid=(1,)`) kernels: realistic spike counts are
     small enough (see the scale reasoning below) that one block sorting
     everything at once beats tiling for now. Both turn their sort into one
     `tl.sort` call by packing every field (time, a node index, for `merge()`
     also a connection id) into one sortable integer key — `tl.sort` only
     orders a single tensor, no attached payload, and Triton has no
     argsort/key-value sort primitive. Floats are made integer-sortable via
     the standard order-preserving bit flip (sign bit flip for positives, full
     bit invert for negatives) — validated independently against
     `torch.sort` across positives/negatives/zero/inf before being folded
     into either kernel. A nice side effect of `merge()`'s packing: ties
     (equal timestamps across connections) get a fully deterministic,
     documented tie-break (ascending by connection id, then node), not an
     incidental one — see `test_ties_break_ascending_by_connection_then_node`.
     **Integration** (`Population.set_fresh_output/compact/merge_incoming`,
     `Network.gather`), tested the same test-first way against real
     Population/Network/Connection objects built through the normal DSL
     (`test/networks/gather_net`: A, B -> C), in `test/test_network_gather.py`
     — spikes controlled with static data via `set_fresh_output()` rather
     than by running the update kernel, so this tests the *wiring* (each
     population gets its own persistent `CarryBuffer`; `merge_incoming`'s
     conn ids come from a connection's real index in `self.incoming`,
     matching `compile()`'s `conn{i}_...` numbering; `Network.gather` runs
     every population's `compact()` — a barrier — before any
     `merge_incoming()`, both phases on the same CUDA stream so in-order
     execution is the only sync needed), not `compact()`/`merge()`'s own
     correctness again.
     **Now also wired into real kernel execution end to end**
     (`Population._execute_gather_kernel`/`_execute_update_kernel`,
     `Network.step()`), not just the lifecycle plumbing above:
     - `update_kernel.py`'s flush step no longer compacts into a shared
       `out_spike_*` buffer with a global atomic — it stores this program
       instance's own spike count into `local_spike_count_ptr[pid]` (plain
       scalar, still no atomic) and leaves `local_spike_{node,time}_ptr`
       GROUP-blocked, which together are now exactly a `BlockedSpikes` —
       `_execute_update_kernel` wraps them directly and calls
       `set_fresh_output()` itself, no separate compaction pass needed.
     - `Network.step()` now actually runs three network-wide phases per
       sub-time: gather (every population's `compact()`, unconditional —
       `Population._execute_gather_kernel`, which also resets its own
       `_fresh` to empty immediately after folding it in, so a population
       that skips its update this round doesn't re-fold stale entries next
       round), merge (`merge_incoming()` for populations with incoming
       connections, stashed as `Population._merged`), then update
       (`Population._execute_update_kernel`, gated by the new
       `Population._is_due`, extracted unchanged from the old `advance()`
       — removed, since nothing called it and `Network.step()` now needs
       the three phases separated rather than interleaved per population).
       One consequence worth naming: a spike is only visible to `compact()`
       starting the *next* sub-step after it's emitted, even at nominally
       zero delay — the emitting population's update kernel (which produces
       the spike) and its own `compact()` call (which would fold that spike
       in) never run in the other order within one `step()` call. Per-link
       delay is unaffected by (and orthogonal to) this — it was already a
       known approximation, see above.
     - Two new per-population configuration points, resolved as defaults
       rather than being required everywhere: Triton tile sizes
       (`group_size=32, spike_block=32, link_block=64, local_spike_cap=64`,
       all overridable per-`Population` at construction — previously only
       ever hardcoded per test harness, e.g. `test_gpu_hardware.py`'s
       `GROUP_SIZE=4`) and device placement (`Network.device`, auto-detected
       once per `Network` as `cuda` if available else `cpu`; a population's
       own per-neuron dynamic-attr tensors — set by user code as
       `pop.attr = tensor`, e.g. `test/networks/*/net.py` — move there lazily
       on that population's first real kernel launch via the new
       `Population._ensure_on_device`, so no network-building code anywhere
       had to change). Per-link tensors and blocked topology are instead
       rebuilt/moved fresh on every `_execute_update_kernel` call rather than
       cached — simple and correct, but recomputing `Topology.blocked()`
       every step is wasted work worth revisiting if it ever shows up in a
       profile.
     - First real, fully-automatic end-to-end test —
       `test/test_network_step.py` — calling `network.step()` on a real
       `counter_net` with no manual `set_fresh_output`/`compact`/
       `merge_incoming` calls anywhere, and hand-verifying the resulting
       potentials/spike timing exactly like `test_gpu_hardware.py` does for
       one bare kernel launch.
     Still open:
     - a `compact()`/`merge()` call bigger than its fixed capacity (both use
       a compile-time-sized single block);
     - `merge_incoming()`'s per-connection `.item()` sync (inherited from
       `merge()`'s host wrapper) — fine at the connection counts tried so
       far, but exactly the per-transfer latency floor to avoid once a
       population has many incoming connections;
     - true kernel concurrency across populations within a phase (`Network.
       gather()` is correctly *phased* but still just a Python `for` loop
       launching everything on one stream) — a performance follow-up, see
       "parallelize kernel execution across populations" below, not a
       correctness gap `gather()`'s phase ordering already covers that.
   - **Stay on GPU — now measured for real on the 780M, not just reasoned.**
     Earlier this was CPU-only timing plus literature-sourced PCIe/launch
     estimates (no GPU was available yet). With real hardware now working
     (`test/test_gpu_hardware.py` is the first naer2 kernel ever actually
     launched, as opposed to only compiled/inspected or interpreter-checked),
     re-ran it as an actual
     GPU-resident `torch.argsort` vs. a real D2H-copy + CPU-`argsort` +
     H2D-copy round trip, both on this machine, same data, `torch.cuda.
     synchronize()` bracketing every timed region:

     ```
              N | GPU argsort | CPU roundtrip | CPU-only sort | GPU wins by
            100  |    34.9 us |       43.9 us |        3.1 us |      1.3x
          1,000  |    43.6 us |       60.1 us |       10.6 us |      1.4x
         10,000  |    84.7 us |      579.5 us |      480.1 us |      6.8x
        100,000  |   817.6 us |     6439.3 us |     6164.4 us |      7.9x
      1,000,000  |  4507.1 us |    75697.6 us |    72834.3 us |     16.8x
     ```
     (sync-only floor ~2.7us; a 1-element D2H copy alone is ~16us — the
     round trip's cost is dominated by the CPU sort itself past N≈3,000,
     not the transfer.)

     GPU-resident sort wins at *every* size tested, including the smallest
     (100 events) — there's no crossover where the round trip comes out
     ahead, not even in principle at tiny N, because PyTorch's own op
     dispatch overhead on the GPU path (~30-40us, well above the ~3-16us
     transfer/sync floor) is still less than one extra CPU round trip once
     a real sort has to happen on the other end. At this library's target
     scale (reasoned earlier to land at roughly hundreds of thousands of
     spike events per timestep *network-wide*, even for networks much
     larger than anything simulated before, since firing rate and timestep
     size are both biologically bounded while GPU memory — dominated by
     synapse storage, not neuron count — is the actual ceiling on network
     size) the margin is ~8x and still growing. This is still the naive
     `torch.argsort`, not the k-way-merge/near-sortedness techniques above
     — so it's a lower bound on the GPU-resident advantage, not the final
     number. If CPU sorting is ever revisited anyway: batch all
     per-connection transfers into one call per population, not one per
     connection — paying the fixed per-transfer floor `k` times (k =
     connections) was the dominant cost in the old round-trip estimate,
     not bandwidth, at realistic k.
   - **New: parallelize kernel execution across populations.** `Network.step()`
     (`naer2/network.py`) is a plain host-side Python `for pop in
     self.populations: pop.advance(...)` loop — fully sequential, one
     population's kernel launches at a time, with no batching across
     populations into a single launch. This is a separate, likely larger
     latency lever than how spike management sorts, since every population
     pays its own full kernel-launch overhead in sequence regardless. Not
     started; flagged here per request.
2. **Get the full PD14 network (`test/networks/PD14/net.py`) actually running in
   main.py**, as the first real-scale proof the whole stack holds together
   rather than developing further pieces in a vacuum against small hand-built
   networks. Deliberately **without per-link delays first**, as a sanity check
   that isolates one variable at a time — passing this only proves the
   execution plumbing works, not that the physiology is right (currents would
   arrive instantly instead of after 0.75-1.5ms). Inventory of what's missing,
   roughly in blocking order:
   - **Done: per-lane GPU random number generation**, needed for
     `PD14_poisson.send()`'s `random.random()`-based exponential inter-spike
     interval. New `RandomLoweringTransformer` (same pipeline position/policy
     as `MathLoweringTransformer`: whitelist-or-raise, runs before
     `IfTransformer`), tests in
     `unit/transpiler/transformers/test_random_lowering_transformer.py`.
     Supports `random.random()` → `tl.rand(seed, offs)` and
     `random.gauss`/`normalvariate` → `mu + sigma * tl.randn(seed, offs)`.
     `seed` is `t` (the kernel's own per-call time boundary, already in scope
     everywhere, bit-reinterpreted as an int) XORed with a per-call-site salt
     (so two different `random.*` calls in one hook body don't collide);
     `offs` (the lane's global neuron index) makes different lanes draw
     different numbers. Verified directly against Triton on real hardware
     before and after wiring into the transformer: deterministic given the
     same `(t, offs)`, differs across lanes and across `t`; then verified
     `PD14_poisson` itself compiles and runs for real (2000 neurons, rate
     0.12/ms, 100ms, observed/expected spike count ratio 1.09 — reasonable
     for a coarse 1ms-step sanity check, not a precision claim) and that its
     `next_spike` state evolves deterministically given the same `t`.
     **Known limitation, accepted for now, not yet fixed:** seed has no
     population identity folded in, so two *different* populations' kernels
     draw IDENTICAL numbers for the same lane index at the same `t` —
     randomness only differs within one population's own kernel, not
     network-wide. Fix is each population owning its own seed, which needs to
     be public (not hidden/derived oddly) for determinism/reproducibility to
     stay possible — not done yet. Also not handled: an edge forward/backward
     hook calling `random.*` would incorrectly use `offs` (neuron-lane-shaped)
     instead of the link-tile lane variable; no edge hook does this yet.
   - **Done: `Population._execute_gather_kernel`/`_execute_update_kernel` and
     `Network.step()`**, including the `update_kernel.py` flush-step change
     this needed — see item 1 above for the full description (three-phase
     `step()`, tile-size/device defaults, `test/test_network_step.py`). A
     real kernel now launches through the normal simulation loop, not only
     by hand in test scripts.
   - **Not started: `compact()`/`merge()` capacity at PD14 scale.** Both are
     fixed-capacity, single-Triton-block kernels sized for test scale (tens
     to ~100 entries) — PD14's E_23 alone is ~20,683 neurons. Needs an actual
     estimate of expected per-timestep spike counts at this scale before
     assuming current capacities are fine; if not, needs tiling.
   - **Not started/untested: the full PD14 network has never been compiled.**
     `test/networks/PD14/net.py`'s `build()` has never been run by any test,
     not even at the AST-inspection tier the rest of the suite relies on —
     only `mini_ei` (a scaled-down variant that drops the Poisson population)
     touches `PD14_neuron`/`PD14_synapse`/`PD14_connection`. Compile-time
     surprises (at ~78k neurons, cortical-scale connection counts) are still
     possible, separate from execution.
3. **A new test network**, exercised **on the CPU first**, at a small and a large
   scale, before anything touches a GPU-style path.
4. **Only then**: get kernels compiling under Triton and confirm their behavior
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
- Real GPU hardware is now available and used (`test/test_gpu_hardware.py`,
  `test/test_spike_gather_gpu.py`, both skipped without a CUDA/ROCm device) —
  this note is stale; nothing runs under the Triton CPU interpreter
  specifically, since that tier was removed from the live suite — see
  README.md — but real hardware now covers more than that tier did.
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
- Population has become a god class (shape/config, dynamic_attrs bookkeeping, spike lifecycle state, compile(), kernel compile+cache, device placement, real kernel launch...) — split responsibilities out once the dust settles
- update_kernel.py and population.py have piled up too many comments by now — shorten them once refactoring makes the code itself clearer