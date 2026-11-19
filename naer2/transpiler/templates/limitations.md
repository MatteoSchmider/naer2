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
  (12 CUs) at G=256 is best at 4 warps (~425 us, ~470 at 8) --
  `runtime._warp_cap`: 8 warps from 12 multiprocessors (RDNA: WGPs) up,
  else 4.

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
