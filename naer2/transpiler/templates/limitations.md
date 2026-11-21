# What limits the fused kernel's further optimization

Findings from optimizing `fused.py` (PD14, G=256, Radeon 780M; RX 6600 for
comparison), kept here for whoever tries to make it faster next: what made
the failed attempts fail, and what any further change has to respect.

## The limits

1. **Registers are the latency budget.** The kernel is chains of
   dependent memory round trips, hidden by switching resident waves.
   Holding more state per program cost more occupancy than it saved: a
   chunk kept in registers (192 VGPRs), prefetching the next chunk (+11-14
   VGPRs), U=16. What did work: shortening lifetimes -- node state loaded
   after delivery when receive doesn't touch it (108 -> 75 VGPRs). Beyond
   ~10-12 programs per WGP the kernel no longer cares; forcing 16
   (waves_per_eu) spills.
2. **No per-link cross-lane traffic.** The one-hot receive and register
   tile extraction cost 4-8 barriers per entry; the scalar receive (lanes
   compare against the link's target) and the in-thread walk group need
   none. Only per-chunk work may communicate (it does: ~25 barriers per
   stream block, histogram / scans / gathers through LDS).
3. **Per-entry costs dominate.** The per-chunk chain is ~60% of the
   delivery's cycles, the walk ~22%, at 1.6 links per segment. Splitting
   segments finer (per wave: x1.5 entries, +663 MB) multiplies the
   expensive part.
4. **Code size.** An unrolled walk over a 256-link stream was slower than
   the loop.
5. **Write locality, not atomics.** The pull's appends were bound by
   scattered writes (16-byte records + cache-line interleaving across
   cells, `_rec`, fixed it); returning atomics on shared addresses are
   cheap.
6. **Program count.** Persistent programs finding their own work: flat
   between 4 and 16 per CU, worse beyond (fixed setup per program).
7. **The chains are the floor.** Delivery: 2-4 chunks per program, each ~2
   dependent global round trips + a barrier; the pull: ~5 dependent steps
   per tile, 1-2 tiles on the critical path.

## Tried and dropped (measured)

One-hot receive; a chunk kept in registers; next-chunk prefetch under the
walk; an unrolled walk; per-wave segments (estimated); an arrival-slot
table in the pull (the bucket-mean load is an L1 hit, the on-chip gather
isn't cheaper); chunked ring runs (sync on append, pointer chasing --
interleaving got the locality instead).

## Compile time (Triton, from the old templates; still applies)

- Compile time grows superlinearly with the number of *structurally
  distinct* branches in one kernel, in the TTIR -> TTGIR lowering, not in
  LLVM (`DISABLE_LLVM_OPT=1` barely changes it; per-stage times from
  `triton.knobs.compilation.listener`). Relevant for D-6's receive
  variants: every variant is code in the walk body.
- Forced unrolling (`tl.static_range`, `tl.sort`'s bitonic network over
  one register tile) multiplies code size and compile time, and ran slower
  too (instruction fetch); real loops instead.
- `tl.histogram` compiles to LDS atomics (`ds_add_u32`); a one-hot
  compare-and-reduce over a tile compiles to long cross-lane shuffle
  chains -- categorically different code.

## Synchronization (from the ring arenas, 2026-11-18)

- **Any decision across warps is a synchronization.** A reduction over a
  tile (`gl.max`, `gl.sum` of a tensor spread over warps) goes through LDS
  with a barrier, and costs about as much as an explicit `gl.barrier()` --
  not because of the instruction, but because the warps stop overlapping
  their memory waits. A per-tile "does anyone need X" check cost ~2.5
  Mcycles of a ~27 Mcycle PD14 round, as much as the barrier it guarded.
  Decide from what every lane knows alone (here: the claim's atomic returns
  the cell's block count with the count), or from values every warp reads
  alike after a barrier that exists anyway.
- **Gluon has no barrier-free way for one warp to wait for another.**
  `gl.atomic_poll` waits for the whole block ("the block waits for all polls
  to finish": a barrier after its loop; with acquire/release one before the
  store too), and a scalar atomic is performed by one thread and broadcast
  through LDS with a barrier -- so a hand-written `while` on
  `gl.atomic_load` has barriers inside its loop. Only inline assembly could
  spin per wave.
- **Masked loads in the hot loop are expensive** when most lanes use them
  (a branch, and the memory waits fall back to waiting for everything); a
  load almost no lane needs, masked off, is cheap (the arenas' table reads
  past a run's reserve).
- **A dependent load per record doubles the round trips:** issue it with
  the loads it doesn't depend on (the arenas' block ids with the link
  targets), or avoid it (fixed reserved blocks: arithmetic addresses).

## Replicated layouts (2026-11-19)

- **No more lanes than a group has neurons.** With G < 32 x warps every
  neuron's values live in several lanes (a replicated layout). PD14 at G=64
  with 4 warps then intermittently lost spikes in `_emit_spikes`: a
  program claimed its place in the history (the count atomic) but none of
  its lanes stored -- its firing lane saw a different position or base than
  the one claimed -- so the slot kept an older run's entry there (found as
  the uniform contract's flag: a consumer took a spike below its
  watermark). Timing-dependent (more often with the host ahead and the
  populations' kernels overlapping), never with 2 warps (8 of 8 runs clean,
  against about 1 in 2 with 4); barriers around the count atomic didn't
  help. Not pinned to one op -- the cumsum over the replicated layout
  (shared memory a replica may read before its owner wrote it) is the
  likeliest. `PopulationRuntime._warps` derives the warps from the group
  size (one lane per neuron, up to 8 warps) and refuses more.
- **Fewer lanes than neurons costs as much.** PD14 on the RX 6600, quiet:
  G=256 at 4 warps (2 neurons per lane) ~465 us a round, GPU-bound; at 8
  warps ~280, host-bound -- each lane's neurons are serial, and half the
  waves hide the memory round trips (limit 1). G=512 at 16 warps ~620:
  1024-thread programs, fewer resident per CU. Device-dependent: the 780M
  (12 CUs) at G=256 is best at 4 warps (~425 us, ~470 at 8). Now 4 warps
  everywhere (`runtime._warp_cap`; see below: the 4090 in graph mode),
  until the warps are chosen per device.

## The RTX 4090 (2026-11-19, PD14 quiet, G=256)

Traced per program (`tools/pd14_phases.py`; kernels: `tools/pd14_kernels.py`): pull ~8-11 us, delivery 3-6,
update + send ~1.3, ~3 us per kernel of launch and scheduling (T, no inputs: 3 us a program, 6.3 a kernel).

- **The append's tiles were the pull's cost** -- ~1 us of dependent steps each (segments, mean, claim,
  targets, store), and a block of 32 spikes took one tile per 4 of its spikes' segments: up to ~20 a spike
  (a segment per delay bucket, mostly 1-2 links), so 5 tiles for ~200 segments. Packing the block's
  (spike, segment) pairs densely, one lane each (`_append_links`), cut the pull ~30% (I_4 10.4 -> 7.2 us,
  the kernels' sum 187 -> 164 us). Wider column tiles (MSEG 8 / 16) were slower: every spike's columns
  loaded and claimed, mostly empty.
- **The fix-up's log moves, one pass over all slots** (a scan, a binary search per lane, instead of two
  reductions per slot for ~80 slots): the tail of growing runs, slowest programs 36-40 -> 28-30 us.
- **Tried, no gain:** software-pipelining the append's link loop (prefetching the next four links' loads:
  hardly more than one link iteration a tile -- with packed tiles ~2.1 on average, still no gain: an
  iteration is bound by its own work, not its loads' latency); prefetching delivery's next stream block
  during the walk (slightly slower: little to overlap with a short walk, more work); four 32-bit record
  stores instead of the converted 128-bit one (within noise).
- **Delivery's walk split by target warp** (a stable partition of each staged block, each warp walking its
  own list): a full block 2.3 -> 2.05 us, a 22-link one 0.2 -> 1.37 us -- the partition (a one-hot
  cumsum, four scatters, a barrier, reductions) costs ~1.2 us a block, the old walk ~9 ns a link, and most
  quiet blocks are short. Delivery is bound by its fixed per-block costs, not the walk.
- **Not tried, by the numbers:** balancing links within a packed tile -- link iterations per tile 1: 25%,
  2: 50%, 3: 17%, 4-6: 8% (~2.1 on average); a second scan and search per tile (~0.8 us, like the tile
  setup) to save ~1.1 iterations of ~0.35 us.
- **The tile setup** (0.77 us: the lane's spike by a 5-step binary search over LDS, then 8 per-spike LDS
  gathers): one packed [8, SPK] table read with one gather didn't compile (Gluon: a 2-D shared gather along
  axis 1 asserts in LinearLayout; rows of a 2-D shared buffer by .index fail on rank); counting the
  spike's first pair in registers ([T, SPK] compares) was slower (0.94 us) -- the layout conversions into
  and out of the [T, SPK] layout go through LDS with barriers, dearer than the gathers.
- **One header chain** (the per-connection last-read / overrun reductions from the per-(connection, slot)
  header loads, not a second plan -> history -> tag chain): ~2% on the pull -- the two chains had been
  issued together already.
- **Layout conversions are the hidden cost in Gluon:** a `convert_layout` between distributions goes
  through LDS with a barrier; replacing a few LDS gathers with register work that needs conversions loses.
- **Measuring GPU busy time:** `tools/pd14_busy.py` (the trace's per-program wall clock, per-round trace
  slices, free-running). RTX 4090, G=256: 105 us busy a round, no overlap between kernels, the longest
  15 us (23 max) -- while the profiler's kernel times are ~25 us: ~10 us a kernel outside the programs.

## Captured windows on the RTX 4090 (2026-11-21, PD14 quiet, G=256)

A round replayed from a captured graph (`naer2/graph.py`): all 9 populations' kernels launched at once, so
what limits a round is the slowest of its ~309 programs running side by side, not launches. Tools:
`tools/pd14_graph_gpu.py` (the replay's GPU time; `--trace`, `--slowest`), `tools/pd14_phases.py --graph`
(phases and steps in one-round graphs; level 3 `--dissect`), `tools/pd14_tail.py`, `tools/pd14_occupancy.py`
(`--targets`), `tools/graph_overhead.py`.

- **Host-memory stores stall the other kernels.** The run status written straight into pinned host memory
  by every program (zero-copy): a kernel finishing with those stores outstanding froze the programs of the
  kernels still running for ~15-20 us, in nearly every round (alone, on one stream: none). The status in
  device memory, copied to the host once per window (the graph's last node): a round's GPU time 65 -> 40 us.
- **Occupancy: 4 warps.** At 8 warps registers (~112-128) allow 2 programs an SM, a round's 309 need 155 SMs
  of 128: a second wave. At 4 warps 4 fit (512 slots): one wave, 40 -> ~29 us. Beyond that registers don't
  matter (a cap of 96 or 80: within noise, 80 spills); G=128 everywhere (615 programs) or G=512: no better.
- **The round's shape costs ~4 us** (9 kernels + the barrier kernel between rounds, against ~1.3 for one
  kernel of all programs: `tools/graph_overhead.py`, programs busy a fixed time) -- fusing the populations
  into one kernel would buy ~3 us. A barrier kernel between rounds makes the graph cheap to instantiate
  (194 -> 1 ms for 100 rounds: 9 + 9 edges a round, not 9 x 9); one before the first round, too (its kernels
  as roots started up to ~2.6 ms late).
- **The tail is per program, not per population.** All programs start together; each covers one group, so a
  big population isn't late, it has more programs. A program costs ~10-12 us fixed plus ~17 ns per link it
  receives: the populations receiving most per group (I_4 ~520 links, E_5 ~440, against ~300-350) end the
  round. Smaller groups (G=128) or 8 warps for just those: ~1 us at best -- the fixed part stays.
- **Stepped runs read their entry from host memory** (the direct schedule): 1-5 us of every program's
  start, growing with the population's programs (uncached PCIe reads, each program its own); in a captured
  window the entries are in device memory (~1.1 us to the run headers, flat). Same-address loads from many
  programs at once cost nothing extra (`tools/same_line_loads.py`: 128 programs, one chain or one each).
- **The walk split by target warp** (`_stage_records`' stable partition, `_walk`: each warp its own neurons'
  links, U a read): staging +0.5 us a stream block (one scan over a one-hot [STREAM, warps] tile, the
  scatters, a barrier), the walk 2.7 -> 1.3 us; a round 28.8 -> ~27 us. At 8 warps (2026-11-19, launched one
  by one) it hadn't paid: the walk is now the tail's load-dependent part, and the partition's LDS work
  hides behind other programs. The 780M (throughput-bound) is ~5-8% slower with it (stream 128: the same).
- **Tried, no gain:** one compare per thread for its SPT neurons (`lane & ~(SPT-1) == t & ~(SPT-1)`: the
  compiler keeps a compare per element and adds the masks); dropping `& mask` in the walk (padding is -1);
  only program 0 storing the shared last-read words.
- **Fixed parts per program** (one-round graphs, median): header chain ~1.6-2.7 us, a spike block 0.4-0.8 +
  ~1.9 a tile (1-2 tiles), fix-up ~0.8-1.7 (mostly the epilogue before its barrier; the growth branch runs
  in <15% of programs), delivery setup ~0.4, freeing a slot's blocks 0.2-0.4, update + send ~1.3.
- **The pull's first reads before everything else** (`_pull_headers`: both parities' last-read ids, the
  histories' addresses, the run headers -- none of it depends on the run's entry): the chain to the headers
  issued with the entry's and the gate's loads, 4 dependent steps -> 2; the header chain ~2.3-2.7 -> ~1.4-2.0
  us, a round ~1 us faster (A/B, 3 pairs). The ring header's offsets hoisted too: no gain (never on the path).
- **E-groups by their distinct emission times** (the least e after the last, one reduction a group) instead
  of a loop over every fresh run with two reductions each, and no all-pairs rank for the batched pull: the
  epilogue before the fix-up's barrier 0.5-1.3 -> ~0.15 us (E_6, I_6: 9 connections, the most fresh runs);
  a round ~0.3 us (6 pairs, within single-run noise).
- **The node state loaded first thing** (every attr, the receive's and update's alike, with the pull's first
  reads): update + send 1.37 -> 1.10 us in every population; a round ~0.35 us (6 pairs, 4 faster; within
  single-run noise -- the 780M too noisy to tell). Registers 120 -> 124, spills 12 -> 2 bytes.
- **Pipelining the pull's spike blocks** (the next block's spikes -- binary search, nodes, times -- loaded
  before this block's segments and appends): no gain alone, and slower with the early node state (most
  programs have one block in quiet rounds; with two, the block's own work bounds it, as the link loop's).
