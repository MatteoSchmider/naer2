# naer2 — status and plan

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

- **Ring overflow is noticed late** (next). The runtime looks at the flags
  of the most recent run only (`PopulationRuntime._grow_if_full`), which
  is still in flight whenever the host runs ahead of the GPU -- so a
  growth waits until the GPU catches up (PD14 free-running: the overflow
  of run 7025 noticed at run 7502), and every run until then loses links.
- **PD14 faults after the overflow** (2026-11-16, unexplained): at the
  default capacities, free-running to ~760 ms, most runs end in
  "unspecified launch failure" (the kernel log: MES fails REMOVE_QUEUE,
  GPU reset; no page fault). Not with the harness's capacities (no
  overflow), not with `AMD_SERIALIZE_KERNEL=3`, not with a synchronize
  before the growth; not reproduced on a small pair with 500 overflowing
  runs in a row. Probably many PD14 runs overflowing in a row (the item
  above) -- revisit once that is fixed. (Seen with the per-cell rings,
  now `Population.ring_block = None`; PD14's default arenas don't
  overflow.)
- **A ring growth costs a recompile** (~1 s, PD14): the capacities are
  constexprs (pooled: `POOL_NB`, and `MAIN_CAP` / `SIDE_CAP` from it).
- **Ring arenas: sizing** (2026-11-18). Every group's main and side runs
  in one arena of 256-record blocks (`naer2/ring.py`, `fused.py`:
  `_append_links`, `_pool_fixup`, `_free_blocks`); PD14 runs as fast as
  with per-cell rings in quiet activity and ~6% faster through the
  volley, spike-for-spike identical, 1.36 GB of rings against 2.32 GB.
  The sizes are still constants, and the arena is sized for the volley
  all the time:
  - the arena: `Population.ring_in_flight` (1/8) of the largest group's
    incoming links + per slot its reserve and a partly used block -- E_4
    ~108k records per group, while quiet activity keeps ~2k live and the
    volley peaks at ~65k. Derive the need from Little's law (live records
    ~ arrival rate x mean delay: a slot's fill is the delay distribution's
    survival function, see the PD14 ring occupancy map), and grow and
    shrink at runtime with headroom and hysteresis (e.g. +50% on growth,
    shrink only below half the current size), from a rough base estimate.
  - the per-cell overflow log: half an arena over the slots (PD14: one
    cell grows by up to ~1400 records in one volley run, ~80 in quiet
    ones) -- most of the memory beyond the arenas themselves; size it
    from the expected per-run growth instead.
  - the reserve: `ring.POOL_RESERVE` = 2 blocks per main run (PD14's quiet
    slots end with ~220-360 records: 512 covers them, a quiet run hardly
    ever overflows) -- derive per population from the same estimate.
  - a run holds at most `ring.POOL_MAX_BLOCKS` = 2047 blocks (the packed
    claim counter's field).
- **Ring arenas: clean-up**. The implementation grew through several
  optimizations, each measured (reserve, fixed reserved blocks, the
  packed claim counter, the per-cell overflow log, the barrier-free
  freeing) -- consolidate it before sizing: the ring layout is computed
  twice (`Ring._allocate` and `_ring`, including the log records' index),
  the general (non-uniform) pull still allocates per record through
  `_take_block` with barriers, the `lost` path (missing blocks, compacted
  freeing) is only exercised by the tests, `Population.main_cap` /
  `side_cap` apply to per-cell rings only, and the tools' options follow
  the experiments (`tools/pd14_network.py`, `tools/trace_network.py`).
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
- **Optimizations**: a mixed (NON_)UNIFORM_SOURCES pull (D-5); an
  automatic, device-dependent group size (`SimulationTopology.locality`;
  default 32, PD14 measured at 256 only); adaptive implicit scheduling --
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
