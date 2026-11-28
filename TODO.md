# naer2 — status and open points

Every population runs one fused Gluon kernel per run (pull intake, delivery, update, send, emission),
generated from its hooks (`naer2/transpiler/templates/fused.py` the fixed template -- what limits optimizing it
further: `limitations.md` next to it -- and `fused_template.py` its generator). The host side (`naer2/sim/`): the
scheduler decides who runs when (horizons, rounds), the executor launches it -- stepped rounds, planned windows,
windows replayed from captured device graphs (`naer2/schedule.py`, `naer2/graph.py`), every round's populations
fused into one kernel by default (`naer2/fusion.py`), a clock-only network's windows extrapolated
(`naer2/periodic.py`); each population's runtime holds its state, emission history (`naer2/history.py`), ring
(`naer2/ring.py`: per group an arena of blocks) and plan of incoming `SimulationTopology`s, its ledger judges the
runs' statuses, its sizer follows their need, the device's recovery reruns a stopped run. `Network` is the
facade. PD14 (`test/networks/PD14/net.py`): rates within a few percent of the published model, 28 us wall /
20 us GPU per 0.1 ms round in quiet activity on the RTX 4090; Brunel A (`test/networks/Brunel/`): 0.48 / 0.57 ms
wall per round at 10k / 50k, latency-bound (2026-11-28, `tools/baseline.py`; the table in PLAN.md, phase 1).

Run everything with `python -m pytest` (unit + test; the 9 `test_gap_*` fail by design). Open gaps are those
tests and `# GAP (...)` comments in the code (`grep -rn "GAP (" naer2`); known limitations are
`# LIMITATION (...)` comments and `naer2/transpiler/templates/limitations.md`; D-n are the decisions below.
PLAN.md has the order of the larger work.

Open points by priority. Sizes: **S** hours, **M** a day or two, **L** a project of its own.

## 1. Now -- correctness and hygiene

- **[S] A flaky test**: `test_arena_check.py::test_a_held_block_on_a_free_stack_is_reported` fails ~1 in 6 on the
  780M iGPU (never on the 4090): the arena check never reports the planted block ("DID NOT RAISE") -- E's runs
  in the 1 ms step don't pop it, or the check misses it; the failures came with the kernels cached (~3 s runs).
  Make the test deterministic (or find what the check misses).
- **[S] Node attrs named like the kernel's own arguments** break the generated kernel: `showcases/
  hybrid_scheduling.py`'s `n_received` meets the template's `n_received_ptr` (a duplicate argument, a
  SyntaxError at load). Prefix the template's fixed arguments, or reject the names with a clear error.
- **[S] Coverage**: the batch walk's per-time send (`SEND1`) and Brunel's neuron path run in one test only
  (`test_brunel_A_asynchronous_irregular`); queue deliveries are checked by order hashes and sums, not the delivery
  log; the ring's `lost` path (blocks missing: only a shrink below the blocks in use) is untested;
  test/fused_reference.py (the hand-written PD14 kernels) follows every `_pull` / `_deliver` signature.
- **[S-M] The showcases** (`showcases/`) predate the fused path in places (`hybrid_scheduling.py` mentions
  `carry_cap`): run each, fix or drop.

## 2. Next -- the ring project and scheduling (PLAN.md, phase 2)

- **[L] Segmented arena growth**: a resize copies the arena (a shrink compacts) and drains the device -- PD14's
  volley: 25 resizes in 1000 ms, up to ~0.4 s each, 97 us a round overall against 28 in quiet
  activity. Instead: an arena of segments, a growth adding one (new block ids past the old ones, onto the free
  stacks) and copying nothing -- a block addressed through its segment (from the block id's high bits, or a small
  per-group segment table: one more lookup per record address, issued with the block id's load; limitations.md,
  the dependent loads), the reserve and the tables staying where they are; a shrink still compacts (rare). Also
  fewer drains (every batch of resizes waits for an idle device, before and after).
- **[M] Ring memory**: the per-cell overflow log (at least one block, 256 records, per cell -- most of the
  quiet-activity memory beyond the arenas; per group: a claim per record past the held blocks); the reserve
  (`ring.RESERVE` = 2 blocks per slot's run; PD14's quiet slots end with ~220-360 records) following the need; a
  run's 2047-block limit (`ring.MAX_BLOCKS`, the packed claim counter's field: past it the runtime raises).
- **[M] A history that grows** when a consumer falls behind
  (`test_network_step::test_gap_history_grows_with_a_slow_consumer`; today the overrun raises).
- **[L] Brunel's speed**: latency-bound (as naer1 was) -- its kernels' chain of dependent steps (the pull's loads,
  the merge's and the batch walk's barriers), not the links moved. To cut with prefetching and latency hiding:
  the next block's / batch's records and links loaded while the current one is walked, fewer barriers per batch,
  more independent work in flight per program.
- **[L] Scheduling**: implicit scheduling -- no `Network.resolution` / `Connection.resolution` / slot width
  default of 0.1 ms, derived from the populations' and connections' own timing; `n_slots` (longest delay + the
  longest of its own and its sources' windows + 3: `Population._ring_window`) revisited with a long-time-range
  event-driven network -- an update() source without a clock (PD14's neurons) counts as one slot width there (its
  true bound is loose: 2-3x PD14's 0.1); a gated run's start at its due time costs the PD14 kernel one VGPR (AMD
  ISA: 105 -> 106; to time on the 4090).
- **[M] Planning and execution in one pass**: a captured or planned window walks its rounds twice (plan(), then
  the launch or the entries); one pass could do both (the scheduler's rounds() feeding the executor directly).
- **[S-M] The step path's schedule entries** are read from pinned host memory (the direct schedule): 1-5 us of
  every program's start, growing with the programs (uncached PCIe reads; limitations.md) -- in device memory, if
  stepping still matters next to captured windows. Low priority.

## 3. Then -- features and refactor B (PLAN.md, phase 3)

- **[M each] Feature gaps**, each a failing test until closed (the generator raises NotImplementedError naming
  the decision, or the values are wrong), each checking the closed behaviour end to end; D-6 and D-7 first (they
  shape refactor B):
  - `test_fused_generated_gpu::test_gap_receive_variants_per_connection` -- D-6
  - `test_gap_several_per_link_attrs` -- D-7
  - `test_gap_edge_forward` -- edge forward on link lanes while staging, with D-8 documented in the code;
    `random.*` there must draw by the link, not the neuron lane (RandomLoweringTransformer's `offs`)
  - `test_gap_edge_forward_destination` -- D-9
  - `test_gap_source_attrs` -- D-10
  - `test_gap_edge_backward` -- D-11 (a reverse topology)
  - `test_coverage::test_gap_coverage_network_generates[C/D]` -- the coverage network through edge forward and
    D-10
- **[L] Refactor B** (front end, model structures): `Population` is a god class (shape and config, dynamic attrs,
  the ring and delivery configuration, kernel generation and cache, device placement); `Topology` as plain
  tensors (regions and positions lighter -- doubtful there); edges, nodes, patterns, updates, receives, kernel,
  fabric each a file -- packaged more tightly. (The runtime keeps its buffers and its running in one class: a
  buffer structure apart would move ~90 outside reads -- revisit with it.)
- **[M] Group-local link targets as `uint8` / `int16`** (group sizes up to 256 / 65536): 2-4x less target memory at
  billions of synapses -- with `Topology` as tensors.
- **[L] Non-uniform sources on the slotted ring** (per-link delays and precise spike times: a model with
  heterogeneous delays and event-driven dynamics): NotImplementedError since the sequential general pull went
  (2026-11-28); a batched pull for them, deterministic per slot (SPIKES.md). The same model tests D-3 (spikes
  from receive) and the exact-ties work (Brunel's check moved into send, the batch walk's per-time send; the
  link walks still per link).
- **[L] Brunel with STDP** (Morrison et al. 2007, NEST's hpc_benchmark: naer1's main benchmark) -- once edge hooks
  exist.

## 4. Later -- reproducibility, tuning, comparison (PLAN.md, phase 4)

- **[M] Reproducible order**: PD14's uniform producers keep one flat, atomically placed history list and its pull
  claims through LDS atomics, so a cell's order -- its targets' summation order -- varies between runs
  (per-program regions cost 1.6 us a round on the 4090: a cheaper way, separately); the queue pull's merge path
  (a chunk past QCH, a burst at one time) keeps a flat list's append order where its fast path ranks by
  (e, node) (`test_a_burst_past_a_chunk_arrives_in_order[uniform]` checks sums only). fp32 time merges distinct
  events on long runs (~1 us at 10 s).
- **[L] Construction and compile time**, against the baseline (PLAN.md, phase 1): the host keeps ~20 B a link
  through the simulation (the canonical topologies, once on the device); PD14's first captured window ~10 s (the
  fused compile); Brunel's start arenas (~4 B a link at the device's peak); Brunel's construction ~N^2 on the host.
- **[L] Autotuning per device**: the group size (`SimulationTopology.locality`; default 32, PD14 measured at 256
  only), the warps per program (4 everywhere for now, `runtime._warp_cap` -- the RX 6600 was faster at 8 when
  stepped), the walk split by target warp (slower on the 780M, limitations.md) -- measured at startup.
- **[M] README.md** (out of date: "What's deliberately not here yet" says nothing runs a kernel, and names
  `Population._execute_update_kernel`) and **[M] the test suite** (structure, speed, coverage; no direct unit
  tests for `AttrPrefixTransformer`, `SelfAttrTransformer`, `TargetMaskTransformer`, `GluonTransformer`,
  `UsedMemberFinder` / `AssignedMemberFinder`; unused fixtures `lif_cd_weightless`, the two triplet STDP edges).
- **[L] PD14 with a Poisson background, and Brunel, against NEST and GeNN**: PD14 exists with a DC drive (here:
  V_rest shifted) and with a Poisson one; compared directly with NEST (its most rigorous settings) and GeNN
  (2023, Poisson drive on an RTX 4090: real-time factor 0.272, ~27 us per 0.1 ms round) -- accuracy, speed, and
  the front end's definition and build time.

## Known limitations (not open work)

- Host readbacks of device state go through `PopulationRuntime.settle()` (a stopped run rerun first) -- anything
  reading a history or ring directly on the host must call it; a rerun reads the state tensors its launch read
  (the journal keeps them): replacing a state tensor between runs is safe (Population attrs do), writing into one
  in place isn't without a settle() first.

## Decisions (D-n, judged together 2026-11-14; referenced from the code)

- **D-1 receive's `time` by the population's incoming delays** (revised
  2026-11-22, BRUNEL.md) -- the records keep the emission time, in
  emission order; per target the times never step back. One homogeneous
  delay D over every incoming link: exact, e + D. Several homogeneous
  values: exact too -- a queue per connection (SPIKES.md), merged by
  arrival (ties: the lower connection first); the delivery log then
  records arrivals. Per-link delays: every link
  of a slot at the slot's time, its center by default, its end with
  `Population.slot_time = "end"`. A bare n2.spike() in receive fires at
  exactly that time.
- **D-2 send runs after every delivery** (rooted at the target lane) and
  once more after update -- only where the delivery can change send's
  outcome: the connection's receive writes a node attr send reads
  (`ReadMemberFinder`). PD14_neuron: no per-delivery send.
- **D-3 any number of spikes per neuron and run** (n2.spike() in receive,
  in the per-delivery send, at several send sites) -- revised 2026-11-22:
  inside delivery each spike goes straight into the run's history (one
  masked atomic per firing lane, at most one per warp and link step:
  fused.py `_emit_delivered`), not through a per-program buffer -- the
  history is a flat list in no order, and spikes are rare next to links.
  The history's cap per population (`Population.history_cap`, default one
  spike per neuron and run); a run beyond it loses the rest and check()
  raises. Open: the cost in the walk (three warp reductions per site and
  link step), measured with Brunel.
- **D-4 bare n2.spike() emits at the window's start, t_start** -- inside
  delivery at exactly the delivery time (BRUNEL.md).
- **D-5 `uniform_emission`** (this population's own spikes, every
  n2.spike() bare; `FusedKernel.uniform_emission`): its history one flat
  list (else a region per program, sorted). The slotted ring's pull takes
  uniform sources only, queues any (2026-11-28: the general pull for
  mixed sources removed).
- **D-6 per-connection receive code, selected per source spike** (a gap):
  a small VARIANT index -- connections with the same (receive hook, edge
  type) share one (PD14: one for all 8) -- from the connection id in the
  record's source word, the plan's connections numbered grouped by
  variant so the test is compile-time comparisons (c < B1, c < B2, ...).
  In the walk a link is uniform over the program's lanes: a scalar
  branch, only the taken variant executes; the costs are code size and
  registers (every attr any variant touches stays live). Forward runs on
  link lanes while staging, where the variant varies per lane: each
  variant present runs masked to its lanes.
- **D-7 per-link attrs as one block per link** (a gap): link l's attrs
  contiguous ([n_links, n_attrs]), the plan's per-connection pointer at
  the block; the stage loads a link's variant's attrs into attr slots,
  the walk selects them per link.
- **D-8 limitation, documented in the code:** one link twice in one
  stream block (its source spiking in two runs taken in together) is read
  by both lanes before either stores -- the later works on a stale value,
  the last store wins.
- **D-9 gap:** forward's `destination.x` (the target neuron's state) --
  the link's target is known while staging, its state in scope in the
  walk.
- **D-10 `source.x` unsupported for now**: read at delivery it is a later
  value than at emission in the pull model; a real solution (e.g. carried
  with the spike) later. Edge hooks writing `source.x` / `destination.x`
  stay unsupported.
- **D-11 edge backward out of scope for now** (needs a reverse topology).
- **D-12 one slot width per consumer: the finest resolution its incoming
  connections require**; the coarser-resolution case may be optimized
  later.
- **D-13 event-driven scheduling reads the source's summary (count,
  earliest e) from the history header on the host** -- a sync per run,
  fine for now.
