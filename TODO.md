# naer2 — status and plan


test

Every population runs one fused Gluon kernel per run (pull intake, delivery,
update, send, emission), generated from its hooks
(`naer2/transpiler/templates/fused.py` the fixed template -- what limits
optimizing it further: `limitations.md` next to it -- and
`fused_template.py` its generator), on its runtime (`naer2/runtime.py`:
state, emission history `naer2/history.py`, ring `naer2/ring.py`, plan of
incoming `SimulationTopology`s), scheduled by `Network.step` (horizons,
rounds, one stream per population). PD14 through `Network.step` matches
the hand-built harness spike for spike over 800 ms at equal ring
capacities (`tools/pd14_network.py`, 2026-11-16).

Run everything with `python -m pytest`. Open gaps are failing `test_gap_*`
tests (`python -m pytest -k gap`) and `# GAP (...)` comments in the code
(`grep -rn "GAP (" naer2`); known limitations are `# LIMITATION (...)`
comments; D-n are the decisions below.

## Model correctness (high priority; not a gap)

- **E_23 is silent in PD14** (0.0 Hz, 2026-11-06 and still in 2026-11-16's
  runs). PD14's L2/3e is sparse but not zero: a sign the simulated model
  doesn't match the published one somewhere (the network's description,
  its drive, the neuron model, or the simulation itself) -- investigate.
  The other rates (0-700 ms): I_23 2.2, E_4 5.0, I_4 6.0, E_5 ~8.4, I_5
  ~8.5, E_6 1.7, I_6 7.6 Hz, not yet checked against the published rates
  either.

## Open work

- **Illegal address in PD14's volley on the RTX 4090** (2026-11-19, open, next).
  G=256 / 8 warps, free-running (no syncs): faults in most runs between
  rounds ~7025 and ~7110, after the volley's onset (resizes, one
  stop-and-rerun). A GPU core dump (cuda-gdb, `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`)
  puts it at `fused.py` `_stage_records`, the per-link weight load
  (`w = gl.load(atp + l)`, in delivery, kernel of 57 programs: E_6) -- a
  staged ring record within the slot's count holding garbage (connection
  or link index). Gone with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (3/3), under
  compute-sanitizer memcheck (0 errors: serialized) -- both add syncs
  (cudaFree), so a race. Ruled out (each still faults, `tools/pd14_volley.py`
  flags): ring buffers freed and reused (`--keep-old`), uninitialized arena
  memory from the copy-once growth (`--zero-growth`), the warp count
  (`--warps=4`). Instrumented staging (2026-11-19, reverted): the faulting
  record lies inside the arena, its connection id is valid (< n_conns) and
  its link index plausible (0 <= l < 2^26) -- so either the link index is
  wrong for its connection (a record written from a garbage spike: a
  source neuron id out of range in the pull would give a garbage segment,
  hence link index) or the connection's attr pointer table (`s_atp`) is.
  initcheck is too slow here (> 30 min, only host `.item()` reports).
  Since (2026-11-19, instrumentation in `tools/pd14_volley_debug.patch`:
  `git apply` it, run `tools/pd14_volley.py 256`): the pull writes only
  valid records (spikes, segment ranges, link ranges all checked: none
  bad); staging finds MIXED records -- a valid connection id with a link
  index past that connection's links (34-588 per run, near the end of
  their slot's count) -- and masking them stops the fault (3/3 clean).
  The arena invariant (per group: every block id once among the held
  blocks and the free stack) broke once, before a resize in E_23: block 0
  (cell 0's reserve) in cell 14's table several times, ids 212-219
  nowhere -- so a fix-up popped stack positions holding 0 instead of
  those ids, and two slots then wrote one block (hence the mixed
  records). Not the log move's 4 word stores (a 128-bit move: still
  mixed), not `Ring._grow_arena` on the CPU (300 random growths keep the
  invariant). Checking the invariant before AND after every resize slows
  rounds to ~16 ms and the race doesn't show. Next: find what puts zeros
  (or stale ids) at live stack positions -- a check on the device, cheap
  enough to keep the timing (e.g. the fix-up verifying each popped id
  against the group's tables, or a per-group bitmap of held ids). Also
  fixed on the way: a resize now syncs after its work too (the
  populations' next runs, on their own streams, didn't wait for it).

- **Ring sizing** (2026-11-18). Every group's main and side runs live in
  one arena of 256-record blocks (`naer2/ring.py`; `fused.py`:
  `_append_links`, `_pool_fixup`, `_free_blocks`), sized at runtime
  (`naer2/runtime.py`): every run's status (stats per group: blocks held,
  most logged; the groups done) lands in pinned host slots, at most
  `status_slots` (4) runs behind; arenas and logs grow at need x 1.5 > size
  to need x 2 (both in one resize), shrink when need x 1.5 < size / 2 for
  200 runs. A run too large for its ring stops before committing anything,
  every run after it stops too (`_run_gate`), and the host reruns them once
  the rings have grown (`_Launches.recover`) -- nothing is lost. Every
  resize waits for the device to be idle (reallocating while other streams'
  kernels ran made the 780M's MES fail REMOVE_QUEUE and reset the GPU --
  the "PD14 faults after the overflow" seen before). PD14 from the minimum
  arenas: spike-for-spike identical to per-cell rings, 0.42 GB of rings in
  quiet activity (1.36 GB fixed before), ~1.7 GB at the volley's peak.
  Left:
  - **segmented arena growth** (next for sizing): a resize copies the
    arena once (growing; a shrink compacts), batched across populations
    behind one drain. PD14 on the RX 6600, G=256 / 8 warps (2026-11-19),
    the volley's first 200 rounds: 26 resizes in 8 batches, 92 ms (57 ms of
    copies, up to 477 MB), ~1.03 ms a round against ~0.50 without them.
    Instead: an arena of segments, a growth adding one (new block ids past
    the old ones, onto the free stacks) and copying nothing -- the kernel
    addresses a block through its segment (a block id's segment from its
    high bits, or a small per-group segment table: one more lookup per
    record address, issued with the block id's load; limitations.md, the
    dependent loads), the reserve and the tables staying where they are; a
    shrink still compacts (rare, in quiet activity). Also fewer drains: 8
    batches are 8 waits for an idle device.
  - the per-cell overflow log: at least one block (256 records) per cell,
    most of the quiet-activity memory beyond the arenas; a per-group log
    would be smaller (a claim per record past the held blocks).
  - the reserve: `ring.POOL_RESERVE` = 2 blocks per main run (PD14's quiet
    slots end with ~220-360 records) -- could follow the same need.
  - a run holds at most `ring.POOL_MAX_BLOCKS` = 2047 blocks (the packed
    claim counter's field): past it, the runtime raises.
  - LIMITATION: the general (non-uniform) pull takes room record by record
    and can't stop a run: past its ring it still loses links (a warning)
    and the ring doubles.
  - LIMITATION: host readbacks of device state go through
    `PopulationRuntime.settle()` (a stopped run rerun first) -- anything
    reading a history or ring directly on the host must call it. A rerun
    reads the state tensors its launch read (the journal keeps them):
    replacing a state tensor between runs is safe (Population attrs do),
    writing into one in place isn't without a settle() first.
- **Host limits** (2026-11-18). PD14's quiet rounds at G=256 are host-bound
  on the 780M (~550 us of host time per round, 9 runs: ~60 us each --
  status staging ~1/3, the launch ~1/4, the scheduler's Python ~1/5,
  stream fork / join; the GPU's own floor somewhere around 230-400 us,
  hard to pin down: the APU's GPU shares its power budget with the CPU).
  Platform agnostic: nothing below torch / Triton (no HIP calls). Next:
  the status written by the kernel straight into pinned host memory (no
  copy, no event); then a measurement that holds up (more rounds, repeats,
  the RX 6600 to separate the GPU from the CPU); then the scheduler's
  Python. Structural, later: ONE launch per round for all populations --
  Network collecting its runtimes into a bundle (a new class) that
  launches every population's groups in one grid (the launch floor is
  ~11 us per launch, ~100 us per PD14 round).
- **Ring arenas: left from the clean-up** (2026-11-18: the layout lives in
  `Ring._allocate` alone, its header read by `_ring`; the pooled helpers
  are one section of `fused.py`). The general (non-uniform) pull still
  takes blocks one record at a time (`_take_block`, with barriers) -- the
  baseline path, slow anyway; the `lost` path (missing blocks, compacted
  freeing) only runs once its arena ran out (untested at PD14's scale).
- **Gaps**, each a failing test until closed (the generator raises
  NotImplementedError naming the decision, or the values are wrong), each
  checking the closed behaviour end to end:
  - `test_fused_generated_gpu::test_gap_spikes_inside_delivery` -- D-3
  - `test_gap_receive_variants_per_connection` -- D-6
  - `test_gap_several_per_link_attrs` -- D-7
  - `test_gap_edge_forward` -- edge forward on link lanes while staging,
    with D-8 documented in the code
  - `test_gap_edge_forward_destination` -- D-9
  - `test_gap_source_attrs` -- D-10
  - `test_gap_receive_time_is_the_arrival` -- D-1
  - `test_gap_edge_backward` -- D-11
  - `test_coverage::test_gap_coverage_network_generates[B/C/D]` -- the
    coverage network through D-3, edge forward and D-10
  - `test_network_step::test_gap_history_grows_with_a_slow_consumer` -- a
    history that grows when a consumer falls behind (today the overrun
    raises)
  - `test_population_fused_gpu::test_gap_populations_draw_their_own_random_numbers`
    -- `random.*` lowers to a draw seeded from the run's time and the
    lane's neuron index only, so two populations draw identical numbers
    for the same lane at the same time; each population should own a
    public seed. (An edge hook calling `random.*` would also use the
    neuron lane index, not the link's.)
- **The per-link attr load is unconditional** (`_stage_records`): a
  connection whose receive reads no attr gets a placeholder attr block
  (`fabric.PLACEHOLDER`); a constexpr skipping the load would save it.
- **Multapses** (duplicate targets in one row): delivered as separate
  links; forward runs per link (D-8 if in one stream block). Not enforced
  anywhere.
- **`Connection @ mapping` without parens** (`>> B @ {...}`) doesn't mean
  post-target mapping because `@` binds tighter than `>>`; TODOs in
  `Population.__matmul__` / `Connection.__matmul__`.
- **History depth with an unset timestep**: `Network._plan_histories` derives K from the timesteps only
  where both a source's and its consumer's are given, else 3 -- the mini E/I network with I at 0.2 and E
  unset (the resolution, 0.1) overruns E's history ("a consumer fell behind"); the depth should take an
  unset timestep as the network's resolution (found 2026-11-21, test/test_fusion.py sets both).
- **Optimizations**: a mixed (NON_)UNIFORM_SOURCES pull (D-5); an
  automatic, device-dependent group size (`SimulationTopology.locality`;
  default 32, PD14 measured at 256 only) and warps per program (an
  autotuning at startup: 4 everywhere for now, `runtime._warp_cap` -- the
  RX 6600 was faster at 8 when stepped); adaptive implicit scheduling --
  no explicit `Network.resolution` / `Connection.resolution` / slot width,
  derived from the populations' and connections' own timing instead of a
  0.1 ms default; `n_slots` (longest delay + one window + 3) revisited
  with a long-time-range event-driven network; group-local target indices
  as `uint8` / `int16` (group sizes up to 256 / 65536: 2-4x less target
  memory at billions of synapses).
- **Reproducibility**: delivery order within a slot follows the pull's LDS
  atomic claims, so two runs can sum a neuron's inputs in a different
  order (last bits). In practice PD14 stayed spike-for-spike identical
  across the two paths for 8000 rounds; a deterministic append order would
  make it guaranteed.

## Refactoring

- see if Topology can be plain tensors
- see if something similar can be done with regions/positions to make them
  lighter (doubt it though in that case)
- edges, nodes, patterns, updates, receives, kernel, fabric all take up
  single files, maybe they can be packaged more tightly file-wise and reuse
  some things?
- Population has become a god class (shape/config, dynamic_attrs
  bookkeeping, spike lifecycle state, compile(), kernel compile+cache,
  device placement, real kernel launch...) — split responsibilities out once
  the dust settles

## Docs, showcases, tests

- **README.md** is out of date in general (e.g. "What's deliberately not
  here yet" says nothing runs a kernel) -- go through it.
- **The showcases** (`showcases/`) predate the fused path in places (e.g.
  `hybrid_scheduling.py` mentions `carry_cap`) -- look at them again, and
  check they still run.
- **The test suite**, eventually: fine for now and the foreseeable future,
  but testing matters more as the library matures -- work through it and
  optimize it then (structure, speed, coverage). Known so far: no direct
  unit tests for `AttrPrefixTransformer`, `SelfAttrTransformer`,
  `TargetMaskTransformer`, `GluonTransformer`, `UsedMemberFinder` /
  `AssignedMemberFinder` (only through the generated kernels' tests);
  unused test fixtures (`ed_relay`, `lif_cd_weightless`, the two triplet
  STDP edges).

## Decisions (D-n, judged together 2026-11-14; referenced from the code)

- **D-1 receive's `time` is the link's arrival, e + its bucket's mean
  delay** (a gap today: it is e) -- NEST-like quantized behaviour is still
  one line in the node's send. The pull knows the mean when it appends, so
  the record's time word carries the arrival. *Limitation:* a slot's
  records are in emission order, not arrival order, so links of different
  buckets in one slot can reach a target out of arrival order by less than
  one slot width (a receive-side integrator dropping dt < 0 sees time step
  back).
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
