# naer2 — status and plan

Every population runs one fused Gluon kernel per run (pull intake, delivery, update, send, emission),
generated from its hooks (`naer2/transpiler/templates/fused.py` the fixed template -- what limits optimizing it
further: `limitations.md` next to it -- and `fused_template.py` its generator), on its runtime
(`naer2/runtime.py`: state, emission history `naer2/history.py`, ring `naer2/ring.py`, plan of incoming
`SimulationTopology`s). `Network.step` schedules round by round (horizons, rounds, one stream per population);
`Network.run_graph` replays planned windows of rounds from captured device graphs (`naer2/schedule.py`,
`naer2/graph.py`), optionally with every round's populations fused into one kernel (`naer2/fusion.py`,
`Network.fuse_rounds`) and, for clock-only networks, windows extrapolated instead of planned
(`naer2/periodic.py`, `Network.periodic_windows`). PD14 (`test/networks/PD14/net.py`, its expert settings in
`tune`): rates within a few percent of the published model; on the RTX 4090 ~25 us wall clock per 0.1 ms round
in quiet activity, GPU-bound (2026-11-21).

Run everything with `python -m pytest`. Open gaps are failing `test_gap_*` tests (`python -m pytest -k gap`) and
`# GAP (...)` comments in the code (`grep -rn "GAP (" naer2`); known limitations are `# LIMITATION (...)`
comments and `naer2/transpiler/templates/limitations.md`; D-n are the decisions below.

## Small changes

(none open: 2026-11-22 closed the generated modules' temp directories, the harness's ring capacities, the
unconditional per-link attr load, the bare post-target `@` (a clear error now), multapses (documented in
`Topology`); `random.*` in edge hooks moved to the edge forward gap)

## Medium changes

- **README.md** is out of date in general (e.g. "What's deliberately not here yet" says nothing runs a kernel).
- **The showcases** (`showcases/`) predate the fused path in places (e.g. `hybrid_scheduling.py` mentions
  `carry_cap`) -- look at them again, and check they still run.
- **The test suite**: fine for now, but testing matters more as the library matures -- work through it (structure,
  speed, coverage). Known so far: no direct unit tests for `AttrPrefixTransformer`, `SelfAttrTransformer`,
  `TargetMaskTransformer`, `GluonTransformer`, `UsedMemberFinder` / `AssignedMemberFinder` (only through the
  generated kernels' tests); unused test fixtures (`lif_cd_weightless`, the two triplet STDP edges).
- **Construction and compile time**: measure, then cut -- the fused round kernel takes ~8 s to compile, PD14's first
  captured window ~8.7 s (compile and capture); needed to compare with NEST and GeNN (build times too).
- **The step path's schedule entries** are read from pinned host memory (`PopulationRuntime.run`: the direct
  schedule): 1-5 us of every program's start, growing with the population's programs (uncached PCIe reads, every
  program its own; limitations.md). In device memory (one copy per run, or a planned window's) if stepping still
  matters next to captured windows.
- **A history that grows** when a consumer falls behind
  (`test_network_step::test_gap_history_grows_with_a_slow_consumer`; today the overrun raises).
- **Ring memory**: the per-cell overflow log -- at least one block (256 records) per cell, most of the quiet-activity
  memory beyond the arenas; a per-group log would be smaller (a claim per record past the held blocks); the
  reserve (`ring.POOL_RESERVE` = 2 blocks per main run; PD14's quiet slots end with ~220-360 records) could follow
  the need; a run holds at most `ring.POOL_MAX_BLOCKS` = 2047 blocks (the packed claim counter's field): past it,
  the runtime raises.
- **Group-local link targets as `uint8` / `int16`** (group sizes up to 256 / 65536): 2-4x less target memory at
  billions of synapses.
- **D-1, several homogeneous delays exact** (BRUNEL.md, step 3): a delay-class dimension in the ring, a slot's
  classes merged by arrival; until then such a population takes the slot's time.
- **An event-driven test model**: a network to optimize the general (event-driven) path against -- planning,
  gated windows -- the way PD14 drove the clocked one.

## Large changes

- **Segmented arena growth**: a resize copies the arena (growing; a shrink compacts) and drains the device -- PD14's
  volley now makes ~55-85 resizes, its first 200 rounds ~0.4-0.7 ms a round on the 4090 against ~25 us in quiet
  activity (RX 6600, 2026-11-19: 26 resizes in 8 batches, 92 ms, 57 ms of it copies, up to 477 MB). Instead: an
  arena of segments, a growth adding one (new block ids past the old ones, onto the free stacks) and copying
  nothing -- the kernel addresses a block through its segment (a block id's segment from its high bits, or a small
  per-group segment table: one more lookup per record address, issued with the block id's load; limitations.md,
  the dependent loads), the reserve and the tables staying where they are; a shrink still compacts (rare, in
  quiet activity). Also fewer drains (every batch of resizes waits for an idle device, before and after).
- **Autotuning per device**: the group size (`SimulationTopology.locality`; default 32, PD14 measured at 256 only),
  the warps per program (4 everywhere for now, `runtime._warp_cap` -- the RX 6600 was faster at 8 when stepped),
  the walk split by target warp (slower on the 780M, limitations.md) -- measured at startup.
- **Refactoring**: the host side has grown in layers -- `Network` holds step, plan, run_planned, run_graph and the
  fusion / periodic hooks, `PopulationRuntime` launching, status, windows, judging, sizing and recovery;
  `Population` has become a god class (shape/config, dynamic_attrs bookkeeping, spike lifecycle state,
  compile(), kernel compile + cache, device placement, real kernel launch...) -- split responsibilities out. Also:
  see if `Topology` can be plain tensors (and regions / positions made lighter -- doubtful there); edges, nodes,
  patterns, updates, receives, kernel, fabric each take a file -- maybe packaged more tightly, reusing things.
- **The general (non-uniform) pull**: a mixed (NON_)UNIFORM_SOURCES pull (D-5); it takes room record by record
  (`_take_block`, with barriers) and can't stop a run -- past its ring it still loses links (a warning) and the
  ring doubles; the `lost` path (missing blocks, compacted freeing) only runs once an arena ran out (untested at
  PD14's scale). LIMITATION: host readbacks of device state go through `PopulationRuntime.settle()` (a stopped
  run rerun first) -- anything reading a history or ring directly on the host must call it; a rerun reads the
  state tensors its launch read (the journal keeps them): replacing a state tensor between runs is safe
  (Population attrs do), writing into one in place isn't without a settle() first.
- **Scheduling**: adaptive implicit scheduling -- no explicit `Network.resolution` / `Connection.resolution` / slot
  width, derived from the populations' and connections' own timing instead of a 0.1 ms default; `n_slots`
  (longest delay + one window + 3) revisited with a long-time-range event-driven network.
- **Feature gaps**, each a failing test until closed (the generator raises NotImplementedError naming the decision,
  or the values are wrong), each checking the closed behaviour end to end:
  - `test_fused_generated_gpu::test_gap_spikes_inside_delivery` -- D-3
  - `test_gap_receive_variants_per_connection` -- D-6
  - `test_gap_several_per_link_attrs` -- D-7
  - `test_gap_edge_forward` -- edge forward on link lanes while staging, with D-8 documented in the code;
    `random.*` there must draw by the link, not the neuron lane (RandomLoweringTransformer's `offs`)
  - `test_gap_edge_forward_destination` -- D-9
  - `test_gap_source_attrs` -- D-10
  - `test_gap_edge_backward` -- D-11
  - `test_coverage::test_gap_coverage_network_generates[B/C/D]` -- the coverage network through D-3, edge forward
    and D-10
- **PD14 with a Poisson background, against NEST and GeNN**: the published model exists with a DC drive (this one:
  V_rest shifted) and with a Poisson one; both, compared directly with NEST (its most rigorous settings) and GeNN
  (2023, Poisson drive on an RTX 4090: real-time factor 0.272, ~27 us wall clock per 0.1 ms round) -- accuracy,
  speed, and the front end's definition and build time.

## Decisions (D-n, judged together 2026-11-14; referenced from the code)

- **D-1 receive's `time` by the population's incoming delays** (revised
  2026-11-22, BRUNEL.md) -- the records keep the emission time, in
  emission order; per target the times never step back. One homogeneous
  delay D over every incoming link: exact, e + D. Several homogeneous
  values: exact once the ring has a delay-class dimension (a cell per
  group, class and slot, merged by arrival). Per-link delays: every link
  of a slot at the slot's time, its center by default, its end with
  `Population.slot_time = "end"`. A bare n2.spike() in receive fires at
  exactly that time.
- **D-2 send runs after every delivery** (rooted at the target lane) and
  once more after update -- only where the delivery can change send's
  outcome: the connection's receive writes a node attr send reads
  (`ReadMemberFinder`). PD14_neuron: no per-delivery send.
- **D-3 any number of spikes per neuron and run** (n2.spike() in receive,
  in the per-delivery send, at several send sites; a gap inside delivery
  today) -- appended to the program's emission buffer, flushed into the
  history with one count atomic; the history's cap per population, a run
  beyond it raises the overflow flag.
- **D-4 bare n2.spike() emits at the window's start, t_start.**
- **D-5 two flags:** `uniform_emission` (this population's own spikes,
  every n2.spike() bare; `FusedKernel.uniform_emission`) and
  `uniform_sources` (all its sources', the batched pull;
  `PopulationRuntime.uniform_sources`). Mixed sources fall back to the
  general pull for all of them; a mixed path is an open optimization.
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
