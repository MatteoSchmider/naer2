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
   - **Done — explicit per-population scheduling (`Population.timestep`),
     network-wide effective-period propagation, and a real retention bug
     fix — together, a prerequisite the delay design below leans on.**
     Started from one gap (found reasoning through how PD14 actually
     schedules itself) and grew into two more once pushed on:

     - **Gap 1 — a source-only population could never be due.** A
       population driven entirely by `send()`'s own clock, with no
       incoming connections and no `update()` state (`PD14_poisson`/`T` —
       `next_spike` lives in `send()`, `update()` is a no-op) was never
       due under the old `has_autonomous_dynamics` (checks `update()`
       only) / `_min_incoming_delay()` (needs an incoming connection)
       logic — neither applied. Fixed by adding `Population.timestep:
       float | None` (constructor kwarg, like `group_size` etc.; default
       `None` preserves every existing network's behavior exactly). Fixed
       for real: `test/networks/PD14/net.py`'s `T` now sets `.timestep =
       0.1` (the literature's accurate clock-driven comparison timestep
       for PD14) — and since no cortical population declares its own
       explicit `timestep` (all are autonomous, which by itself
       constrains nothing), `T` is currently the sole thing setting the
       whole network's cadence: every `PD14_neuron` population just runs
       every sub-step unconditionally, whatever that works out to be, so
       they end up ticking in lockstep with `T`'s declared rate by
       composition, not by any PD14-specific awareness.
     - **Gap 2 — a connection with no delay set contributed nothing.**
       Asked directly: if a connection has no delay at all, its target
       should sync to its *source's* own cadence, not fall out of the
       computation entirely (which, combined with no autonomous dynamics
       and no explicit timestep, is the exact same "never due" shape as
       gap 1). This can't be answered per-population in isolation anymore
       — a population's effective period can depend on its source's,
       which can depend on *its* source's, recursively, including through
       cycles (recurrent connectivity is normal in SNNs). Replaced the
       old `_sub_dt`/`_is_due`-does-its-own-thing split with `Network.
       _compute_effective_periods()`: one pass computing, for every
       population, the coarsest safe re-check interval — explicit
       `timestep` wins outright; else 0.0 if autonomous; else the minimum
       over incoming connections of (that connection's own delay, or — if
       it has none — the *source's* own period, recursively). Fixed-point
       iteration over the whole network handles cycles (periods only ever
       shrink, bounded below by 0.0, and anchored populations are never
       revised, so a cycle of purely-derived populations can't regress
       forever). `Population._is_due(t, period)` collapses to one
       `t - last_update >= period` check for every case uniformly (0.0 =
       every sub-step; `math.inf`, i.e. truly never-due, is specially
       guarded against `-inf` initial `last_update` making `inf >= inf`
       true in Python — an edge case a direct unit test caught). `_sub_dt`
       and `_validate_schedulable` both now take the precomputed `periods`
       dict rather than recomputing anything per population.
     - **Gap 3 — a real, previously-silent correctness bug, found by
       directly reproducing gap 2's scenario on real hardware rather than
       just reasoning about it.** `Network.step()`'s merge phase ran
       `pop._merged = pop.merge_incoming()` unconditionally every
       sub-step — for a population ticking slower than the network's
       `sub_dt`, this *overwrote* a previous round's merge result that
       hadn't been consumed yet (the population wasn't due to run its
       update kernel on that sub-step), silently losing it the moment the
       next sub-step's result replaced it. Confirmed with a real run: a
       slow consumer ended up seeing only its *most recent* sub-step's
       contribution, not everything that had become due while it waited.
       Fixed with a new kernel, `spike_gather.combine()` (structurally
       close to `merge()`, but combining two already-`(time, node,
       conn)`-tagged sorted streams rather than assigning a connection id
       per source) — `Network.step()` now `combine()`s a pending
       `_merged` with each round's fresh result instead of overwriting it;
       `Population._execute_update_kernel` already cleared `_merged` to
       `None` after consuming it, which is exactly the signal that lets
       the next round start fresh instead of combining with stale data.
     - New test nodes for exercising this (none of the existing ones fit:
       `counter` is always autonomous and only fires once; nothing fires
       repeatedly and deterministically without being autonomous) —
       `test/nodes/ed_relay.py` (purely event-driven: `update()` is a
       no-op) and `test/nodes/clock_source.py` (autonomous, fires
       unconditionally every call). Tests: `test/test_spike_gather_gpu.py`
       (`TestCombine`, 5, GPU), `test/test_population_timestep.py`
       (scheduling/period-propagation logic, 11, no GPU),
       `test/test_mixed_scheduling.py` (3, GPU — real `network.step()`
       runs proving all three gaps fixed together, including a test that
       pins the exact numbers gap 3's bug used to produce, 18 vs. the
       bug's 9).
     - **Gap 4 — found building a showcase at real (1,000s) population
       scale, not a test-sized one: `combine()` itself had a scaling bug.**
       It sized its output by each input's *allocated* capacity
       (`a_time.shape[0]`), not how much was actually pending (`a_count`).
       `merge()`/`combine()` always allocate a full padded buffer
       regardless of live due-count, so retaining across repeated
       sub-steps made capacity compound roughly geometrically per round
       (`combine(2048, 2048) -> 4096 -> 8192 -> ...`) even when almost
       nothing was actually pending. This surfaced as `compact()`/
       `merge()`'s single-block `tl.sort` taking minutes to *compile* (not
       run) at population sizes of just 1,000-2,000 — isolating it found
       compile time scaling wildly superlinearly with the sort's width
       (2048 -> 8192, a 4x width increase, was a measured ~145x compile
       slowdown; see `Population.carry_cap`/`per_src_cap` below for why
       the width grows the way it does). Fixed by sizing `combine()`'s
       output from each input's live count (`a_count[0].item()`/
       `b_count[0].item()`, one extra sync per side — same latency-floor
       caveat `merge()`'s own host wrapper already carries) instead of
       its allocated shape.
     - **New: `Population.carry_cap`/`per_src_cap`, exposed for the first
       time.** `compact()`/`merge_incoming()`'s capacities were hardcoded
       to 64 at every real call site (`_execute_gather_kernel`, `Network.
       step()`) — fine for every test network so far (3-10 neurons), but
       at 1,000s of neurons with every one firing the same round, 64 isn't
       enough (`compact()`'s own `TOTAL_CAP` is driven mostly by `n_blocks
       * local_spike_cap` regardless of `carry_cap`, so raising
       `group_size`, not just `carry_cap`/`per_src_cap`, is what actually
       keeps that width — and therefore compile time — manageable at real
       scale; see the showcase below for working numbers). Added as
       `Population` constructor kwargs, same pattern as `group_size` etc.,
       default 64 preserving every existing test exactly. Still not a fix
       for the underlying "single fixed-capacity block" design — this
       just makes the existing constant configurable instead of buried;
       the "needs tiling" item right below is unchanged.
     - New showcase, not a test: `showcases/hybrid_scheduling.py` —
       2,000-wide generator/integrator/differentiator, deliberately
       outside the SNN domain (no spiking neuron model at all) to show
       the clock-driven/event-driven scheduling mechanism is generic.
       Also surfaced a real, worth-documenting architectural property
       while building it: a receive hook's `source.<attr>` reads the
       source population's *current* live state, not a historical
       snapshot pinned to the specific delivery — under retention (a
       delivery sitting unconsumed across several of the source's own
       updates), this means `source.attr` reflects the source's *latest*
       value by the time it's finally read, not the value as of that
       delivery's own recorded time. A spike's own `time`, by contrast,
       is fixed forever once recorded. The showcase's nodes deliberately
       read `time`, not a mutable `source` attribute, for exactly this
       reason — worth keeping in mind for any future node design that
       wants to carry a per-delivery value under retention/delay.
   - **In progress — real per-link delay, split into two update-kernel
     variants, design under discussion.** Today's kernel delivers every
     link out of a firing source at that spike's own recorded time — fine
     when delay is uniform/absent, wrong once delay genuinely varies per
     link (arrival time is then `emission_time + delay[link]`, a per-link
     quantity a single scalar `time` per spike-buffer entry can't carry).
     Naming/selection landed; the actual per-link-delay kernel is still
     just a stub, pending the design discussion below.
     - `naer2/transpiler/templates/update_kernel.py` renamed to
       `no_delay_update_kernel.py` (`UpdateKernelTemplate` →
       `NoDelayUpdateKernelTemplate`, emitted function `update_kernel` →
       `no_delay_update_kernel`) — unchanged behavior, just named for what
       it now explicitly is: the variant that disallows nothing, forward/
       backward edge callbacks fully supported, every incoming delivery
       from one firing source arrives at the same instant. Not to be
       touched for the per-link-delay work.
     - New `naer2/transpiler/templates/per_link_delay_update_kernel.py`
       (`PerLinkDelayUpdateKernelTemplate`) — stub, `build()` raises
       `NotImplementedError`.
     - New `Population.uses_per_link_delay_kernel`: true when any incoming
       connection's `fabric.delay` is a real tensor (as opposed to absent
       or one uniform scalar). `compile()` branches on this — a
       population-wide switch, not per-connection: one delayed incoming
       connection disables edge forward/backward callbacks on *every*
       incoming connection of that population. Reason: once a delivery is
       expanded and merged into arrival-time order (see the design below),
       it no longer sits at a stable position a plastic/learned edge
       attribute could be written back to without an expensive scatter —
       judged not worth supporting awkwardly. A population needing both
       per-link delay on one connection and a learned edge attribute on
       another isn't representable; split into two populations if that
       ever comes up.
     - `test/networks/mini_ei/net.py` had a real per-link `delay` tensor on
       every connection (inherited from PD14-style realism) that was
       previously inert — nothing read `delay` at all before this. It now
       switches E and I to the (stubbed) per-link-delay kernel, which broke
       14 AST-shape tests that have nothing to do with delay (nothing in
       `test_mini_ei.py` even references `delay`). Removed it from that
       fixture, since `test_mini_ei.py` exists specifically to cover the
       no-delay kernel's shape — a dedicated delay-carrying fixture belongs
       with the per-link-delay kernel's own tests once it exists.
     - **Consequence worth naming:** `test/networks/PD14/net.py` sets a
       real per-link delay tensor on *every* connection — there is no
       delay-free path through it at all. That means compiling PD14 (item 2
       below) is now blocked on the per-link-delay kernel actually existing,
       not just a nice-to-have; the "deliberately without per-link delays
       first" sanity-check framing in item 2 needs revisiting once this is
       settled (PD14 as currently written can't take that path at all).
     - `compile()` now raises a clear, specific `ValueError` (naming the
       edge type and its offending attrs) when a per-link-delay population
       has an incoming connection whose edge class declares real forward/
       backward-assigned per-link attrs — rather than letting it fall
       through to the stub's generic `NotImplementedError`. Reasoning: an
       edge class registering a dynamic (assignable) per-link attr only
       makes sense if that attr can actually change, and under per-link
       delay it never can — so this is treated as a usage error on the
       user's part, not a missing feature, and will stay a `ValueError`
       once the real kernel exists too. A static edge (no forward/backward
       assignment) with a tensor merely *overridden* from outside
       (`conn.fabric.weight = tensor(...)`, confirmed end-to-end already —
       `conn{i}_weight_ptr`, loaded, densified, used — via the existing
       `static_synapse` coverage) is unaffected: `fabric.dynamic_attrs` is
       empty for that case, so it's not what's being disallowed.
     - **Fixed: a connection no longer needs an edge class at all.**
       `A >> pattern >> B` used to crash (`AttributeError: 'NoneType' object
       has no attribute 'topology'` — `ConnectionBuilder` had nowhere to put
       the topology without a `Fabric`, which required a `@n2.edge`-decorated
       instance). `Fabric(instance=None)` is now a valid "bare" fabric (no
       forward/backward, no dynamic/static attrs of its own);
       `ConnectionBuilder._ensure_fabric()` creates one lazily wherever
       `target()` needs `.topology` and none was given. `Connection.
       __setattr__` makes `conn.<name> = tensor` the validated, ergonomic
       way to populate a per-link tensor on either a bare or a real
       connection: it only stores the tensor if some hook at the
       connection's target port actually reads `edge.<name>` (reversing
       `fabric_map` first, same resolution `compile()` itself does) — since
       with no edge class there's no other check (a real edge class's
       declared `dynamic_attrs`) to catch a typo or a tensor nothing loads.
       `delay` is exempt from that check (read directly off `fabric` by the
       framework itself, never via `edge.delay` inside a hook).
       `conn.fabric.<name> = tensor` (unchecked) still works unchanged.
       Also fixed in passing: `compile()`'s existence check for a
       referenced `edge.attr` only looked at `dynamic_attrs`/`static_attrs`,
       missing `state` — so a bare-fabric tensor override that *was*
       correctly detected as needing a load (`tensor_attrs`, which does
       check `state`) could still fail this earlier, stricter check with a
       wrong "does not exist" error; now checks the same three places.
       This also made `test/networks/PD14/synapse.py` and mini_ei's/PD14's
       `PD14_synapse() @ pattern` dead weight — neither ever had a forward/
       backward hook that did anything — so both net.py's now use
       `conn.weight = ...`/`conn.delay = ...` directly and the file is
       deleted. `PD14.net.build()` now actually runs end to end for the
       first time (9 populations, 59 connections) — still not compiled
       (needs the per-link-delay kernel itself, see above), but confirms
       the simplification holds at full PD14 scale, not just in a test
       fixture. New `test/test_bare_connection.py` (5 tests, no GPU needed).
     - New `test/test_update_kernel_selection.py` (6 tests, no GPU needed —
       pure AST/metadata, like `test_coverage.py`): no delay and a uniform
       scalar delay both stay on the no-delay kernel; a real tensor delay on
       any one incoming connection switches the whole population and raises
       `NotImplementedError` via the stub, even when another incoming
       connection has no delay of its own; an edge class with real dynamic
       attrs raises the `ValueError` above instead; a static edge with a
       tensor override still reaches the stub's `NotImplementedError`
       (correctly not flagged by the `ValueError` check).
     - **Design settled, superseding the merge-network/coloring line of
       exploration below (kept as the reasoning trail, not the plan).**
       The investigation (real GPU-scale benchmarks of a full expand+sort
       pipeline; proving a static edge-coloring/round scheme can't actually
       dodge the sort — detecting which of today's *dynamically* firing
       sources collide is informationally the same operation as expand,
       no cheaper; then directly measuring, on PD14's real topology with
       plausible independent-Poisson activity, how often two *different*
       sources actually land within a small window of each other at the
       same target) converged on a specific, surprising, load-bearing fact:
       at PD14-representative in-degree (~2,000-6,000), same-target
       collisions between different sources aren't a rare tie-break to
       special-case — they're the typical spacing between consecutive
       arrivals (hundreds of collisions even at a 10us window, out of a
       few thousand arrival events over 200ms of simulated activity).
       Resolving every one of those *exactly* buys back a temporal
       resolution that's already meaningless relative to the model's own
       time constants (0.5ms synaptic, 10ms membrane) and the delay
       values' own measurement uncertainty — true per-pair ordering is not
       a precisely-defined target worth an architecturally expensive
       pipeline for. Decision: keep exact analytic integration (the actual
       differentiator — no time-stepping error solving the ODE between
       events), drop exact *delivery order* in favor of a bounded, cheap
       bucketing scheme. New shape, now two genuinely separate parts:

       **Part 1 — intra-timestep bucketing (the common case).** Each
       population gets a default bucket count (**8**, a placeholder — see
       below), overridable per population (`Population.<something>`, not
       yet added — an event-driven population's own author knows their
       dynamics' tolerance better than a network-wide default could).
       A connection's per-link delay is split into that many static
       buckets (computed once, like `Topology.blocked()` — no runtime
       sort, reusing the existing CSR-walk/densify-reduce machinery per
       bucket, no gather/scatter-by-delivery redesign needed at all)
       *within one population timestep* — i.e. `Population.timestep` (see
       above) is exactly the width this bucketing needs to know, which is
       why getting per-population scheduling right came first. A
       connection whose whole delay spread already fits inside one
       timestep collapses to effectively one bucket for free — "ignore
       small differences" falls out of the scheme rather than needing a
       special case. Buckets are **equal-width** (`bucket_width =
       timestep / n_buckets`, not quantile-balanced as first sketched —
       equal-width ties directly to the population's own timestep as the
       natural reference unit, with no separate quantile pass needed; the
       collision-rate reasoning that motivated quantile balancing earlier
       was for a different, since-abandoned design, see the historical
       trail below).
       - **Done — the static partitioning step, in isolation.**
         `Topology.blocked(group_size, n_targets, delay=None, n_buckets=1,
         timestep=None)`: new optional params, backward compatible by
         construction (`n_buckets=1`, the default, runs the exact
         pre-existing code path unchanged — same `offsets` shape, byte-
         identical output; confirmed by a test that runs both and asserts
         equality). `n_buckets > 1` sorts links by `(block, bucket,
         source)` instead of `(block, source)` — structurally the same
         "flatten two keys into one, argsort once" trick already used for
         block-only layout, just with the group index generalized from
         `block` to `block * n_buckets + bucket`; `offsets` gains a bucket
         axis (`[n_blocks, n_buckets, N_src + 1]`), `block_base` (for
         `block_slice()`/`to_blocked()`/`from_blocked()`, unchanged) is
         recovered by taking every `n_buckets`-th entry of the finer
         `group_base`, since the same block-major-contiguous property
         holds one level up. A delay `>= timestep` (out of this part's
         scope by definition) clamps into the last bucket rather than
         erroring. Pure `torch`, no Triton, no kernel/Population changes
         yet. Tests: `unit/test_topology.py` (+10: default-unchanged,
         missing-args raises, two hand-picked small cases — one link per
         bucket, and multiple links sharing a bucket — plus the existing
         parametrized "faithful to canonical CSR" case regenerated with
         bucketing on). One test-authoring bug caught and fixed along the
         way: an early version of the parametrized test sliced each
         bucket's range as `offsets[b, bk, 0]:offsets[b, bk, 1]` (source
         0's own sub-range only, not the whole bucket) instead of
         per-source `offsets[b, bk, s]:offsets[b, bk, s+1]` like the
         existing non-bucketed test already does — failed 4 of 5
         parametrized cases until corrected; the implementation itself
         was right the whole time.
       - **Done — wired into the real kernel/`Population`, verified on
         real hardware against a real network.** `PerLinkDelayUpdateKernelTemplate`
         (`naer2/transpiler/templates/per_link_delay_update_kernel.py`,
         previously a stub) is now `no_delay_update_kernel.py` with one
         real difference: `_csr_branch` unrolls `n_buckets` sequential
         copies of the range-load-and-link-loop block (bucket 0's whole
         range fully processed before bucket 1's starts — plain Python
         generation at build time, the same trick `_conn_dispatch` already
         uses to unroll per-connection branches, not a Triton loop
         construct, since `n_buckets` is small and fixed per population),
         each using that bucket's own representative arrival time
         (`emission_time + bucket_index * bucket_width`, baked in as one
         literal float per bucket — no new kernel argument needed).
         `Population.compile()` now computes `incoming_with_hooks`
         unconditionally first, then dispatches to whichever template at
         the end (previously the per-link-delay branch returned before
         any of that shared computation ran at all, which is why it could
         only ever be a stub). The **adaptation point** flagged in
         advance: `Population._effective_period()` (own `.timestep` if
         set, else looked up from `Network._compute_effective_periods()`)
         decides whether a connection's delay actually fits — `compile()`
         raises `NotImplementedError` naming the actual delay and period
         if any link's delay reaches or exceeds it (the cross-timestep
         case, Part 2, still not implemented), otherwise builds the real
         bucketed kernel. New `Population.n_delay_buckets` (default 8,
         like `timestep`/`carry_cap` etc.) and `_execute_update_kernel`
         building the delay-bucketed topology (falling a connection
         with no real delay tensor of its own back to a uniform one, so
         every connection on a per-link-delay population gets a
         consistent bucketed offsets shape, not a per-connection mix).
         Real network test (`test/test_per_link_delay_kernel.py`, 3
         tests): one source neuron, all-to-all into 4 targets, weight
         alternating `[1, 0, 1, 0]`, delay placing each target in a
         different one of 4 buckets — only the weight-1 targets fire, each
         at exactly its own bucket's arrival time, written into the local
         spike buffer in ascending bucket order (checked directly, not
         just the final who-fired set — see the test's docstring for the
         exact hand-derived numbers). Two real node-authoring gotchas
         found and fixed along the way, in the test's own node (not the
         kernel): bare `n2.spike()` always records the outer kernel
         boundary `t`, never the contextual `time` — needed `n2.spike
         (time)` explicitly to capture a bucket's own arrival time; and
         forgetting to reset `potential` after firing let the kernel's
         *second*, unconditional final-propagation pass (Part 2, which
         always re-runs `send()` once more for every neuron) fire a
         second time at `t` for anyone who'd already fired once — both
         are just restating existing, documented kernel semantics
         (`naer2.sends.spike`'s own docstring, and `counter`'s own
         receive/send already resetting on every branch) rather than
         anything new the delay kernel introduced.
       - **Still not revisited:** the "no edge callbacks on a per-link-
         delay population" restriction (shipped under the abandoned
         continuous-delay design) — bucketing doesn't reshuffle links at
         all, so it's probably safe to lift for this kernel, just not
         done yet.
       - **Done — re-verified at real multi-workgroup scale** (the N=4
         test above only ever launched one program/workgroup, so it
         couldn't have caught a `pid`-indexing bug). New
         `test/test_per_link_delay_kernel_scaled.py`: same weight-0/
         weight-1/bucket-order check, N=4000 targets, default `group_size`
         (32) → `n_blocks=125`, genuinely multiple workgroups. Passes,
         confirming the kernel's per-block `pid`-based offset arithmetic
         (`_row = (pid * n_buckets + bk) * row_stride`) is correct across
         blocks, not just within one.
         - **Two gotchas found getting there, both tooling/test-authoring,
           not kernel bugs:**
           1. **Triton/ROCm doesn't fail fast on an invalid pointer
              argument.** A hand-rolled diagnostic script that launched
              the kernel manually (bypassing `Population.
              _ensure_on_device`) passed a CPU tensor as a pointer arg;
              instead of raising immediately, Triton churned (CPU-bound,
              climbing RSS) for several minutes before eventually raising
              `ValueError: Pointer argument (...) cannot be accessed from
              Triton (cpu tensor?)`. Looked identical to a compile-time
              blowup until a background-task failure notification
              surfaced the real error. Lesson: when a manual/diagnostic
              launch (as opposed to the real `_execute_update_kernel`
              path) hangs, suspect a plain CPU-tensor-pointer bug before
              assuming a genuine compile-time issue.
           2. **`BlockedSpikes`' layout is per-block, not one flat
              `0..count[0]` range** — `count` is `[n_blocks]`, and block
              `b`'s valid entries are `node[b*cap : b*cap+count[b]]`
              (already global neuron indices, since the kernel stores
              `offs = pid*GROUP_SIZE + ...`, not a block-local index) —
              see `naer2/spike_gather.py`'s `BlockedSpikes` docstring.
              `test_per_link_delay_kernel.py`'s original N=4 test read
              `B._fresh.count[0]`/`B._fresh.node[:count]` directly, which
              only happened to be correct because `n_blocks=1` there. The
              first draft of the N=4000 scaled test copied that same
              (wrong, multi-block-unsafe) reading pattern and appeared to
              show a severe correctness bug — only block 0's targets ever
              showed up as fired — before realizing it was reading the
              buffer wrong, not that the kernel was broken. Fixed by
              unpacking every block's own `[b*cap : b*cap+count[b]]` slice
              before comparing. Worth remembering for any future test
              reading `Population._fresh`/`BlockedSpikes` directly at
              `n_blocks > 1`.
         - **Confirmed, genuine, separate compile-time scaling finding
           (unrelated to either gotcha above):** the per-link-delay
           kernel's compile time scales with `GROUP_SIZE * LINK_BLOCK *
           n_buckets` — the unrolled densify matrix's total size *summed
           across all unrolled buckets* — not with `n_blocks` (grid size
           is a pure runtime launch parameter, baked into nothing in the
           kernel body). Measured: `GROUP_SIZE=32, n_buckets=4` (total
           8192) compiles+runs in ~0.3s; bumping *either* `GROUP_SIZE` to
           256 (alone, n_buckets=4) *or* `n_buckets` to 8 (alone,
           GROUP_SIZE=32) each independently pushed compile time past
           several minutes (confirmed still climbing via `ps` CPU time
           and RSS, not hung — no GPU-memory growth, ruling out the
           earlier capacity-blowup pattern — before being stopped). This
           is why the scaled test above deliberately keeps `group_size`
           at its default (32) and uses `n_buckets=4` rather than the
           default of 8, and scales workgroup count via `N` instead —
           `n_blocks` is free, `GROUP_SIZE`/`n_buckets` are not. Worth
           revisiting if the default `n_delay_buckets=8` is ever combined
           with a larger `group_size` in practice.
         - **Follow-up investigation (triggered by the finding above,
           found a much bigger and more general culprit, not yet acted on
           in the real templates):** scanned the generated kernel for
           other constructs with similar shape. Two real findings, both
           isolated and measured standalone (microbenchmarks in
           `/tmp/.../scratchpad/is_target_bench.py`,
           `edge_weight_bench.py`, `spike_block_scan.py` this session —
           not committed, scratch only):
           1. **`SPIKE_BLOCK` (`Population.spike_block`, default 32) is a
              bigger, previously-unmeasured multiplier than `GROUP_SIZE *
              n_buckets` alone.** The per-spike body — the whole
              connection-dispatch chain, which for a per-link-delay
              population includes every bucket's dense matrix — sits
              inside `for j in tl.static_range(SPIKE_BLOCK)`, and
              `tl.static_range`'s own docstring says it "guides the
              compiler to unroll the loop aggressively" — confirmed real:
              holding `GROUP_SIZE=32, n_buckets=4` fixed (the earlier
              "fast" config) and varying only `SPIKE_BLOCK` gave
              1.0s/4521 instrs (`SPIKE_BLOCK=1`) -> 2.7s/15502
              (`=4`) -> 7.4s/30150 (`=8`) -> 23.0s/59464 (`=16`), still
              climbing, not finished after 3 minutes at `=32` (the actual
              default every population uses unless overridden) when
              stopped. Instruction count scales roughly linearly with
              `SPIKE_BLOCK` (confirming the whole per-spike body really is
              duplicated that many times); compile time scales
              noticeably worse than linear on top of that — same pattern
              as the `GROUP_SIZE`/`n_buckets` superlinearity, now with a
              third independent multiplier stacked on. This affects the
              no-delay kernel too (smaller magnitude, since it has no
              bucket multiplier), not just the per-link-delay one.
           2. **The densify step itself (`_hit`, used for both
              `is_target` and `edge_weight`) has a cheaper realization,
              confirmed by direct comparison at the real slow tile size
              (`GROUP_SIZE=256, LINK_BLOCK=64`).** `tl.histogram(x,
              GROUP_SIZE, mask=valid) > 0` computes `is_target` alone in
              0.0025s/307 AMDGCN instructions vs. the current one-hot-
              compare-and-reduce's 0.31s/4872 (~125x/~16x) — and
              inspecting the generated AMDGCN directly explains why:
              `tl.histogram` already compiles to `ds_add_u32` (LDS/shared-
              memory atomics) internally, while the current approach
              compiles to long chains of `v_permlanex16_b32`/
              `v_readlane_b32`/`v_mov_b32_dpp` (RDNA cross-lane shuffle
              instructions implementing the `axis=1` reduction over
              `LINK_BLOCK`) — categorically different, not just
              differently-tuned, code. Splitting the same dense compare
              into `GROUP_SIZE/CHUNK` smaller pieces and `tl.cat`-ing them
              back together (an alternative hand-sketched independently)
              does *not* help (0.34s/4679 instrs, same cost as today) —
              confirms the win is specific to `tl.histogram` routing
              through LDS, not just "smaller matrices." (One correctness
              trap hit along the way: `tl.cat(..., can_reorder=True)`
              silently permits the compiler to reorder elements — broke
              the chunked variant's lane correspondence until switched to
              `can_reorder=False`, which is also the documented
              forward-compatible default going into Triton 3.10.)
              `tl.histogram` only counts, though — no weighted variant
              exists, so it alone doesn't help `edge_weight` (the
              weighted delivery sum). Measured three full receive-tile
              variants at the same tile size: today's shared-`_hit`
              dense-both (1.35s compile, 8680 instrs) vs. histogram-
              `is_target` + still-dense-`edge_weight` (0.25s, 4121 instrs
              — better, but still paying a dense reduction for weight) vs.
              histogram-`is_target` + `tl.atomic_add(scratch_ptr + x, w,
              mask=valid, scope="cta")` + `tl.debug_barrier()` + reload
              for `edge_weight` (0.038s, 351 instrs — ~35x/~25x over
              today, and also fastest steady-state launch of the three).
              The `scope="cta"` atomic add is mainline Triton (no Gluon
              needed) and, critically, still hands the rest of the
              generated hook code a plain register-resident `edge_weight`
              tensor afterward — the receive/send hook codegen downstream
              of densify wouldn't need to change at all, only how
              `is_target`/`edge_weight` themselves get computed. Would
              need a small per-block scratch buffer (size `GROUP_SIZE`,
              re-zeroed before each tile's scatter — an `O(GROUP_SIZE)`
              store, not `O(GROUP_SIZE * LINK_BLOCK)`) as a new kernel
              argument, analogous to the existing `local_spike_*` buffers.
           **Done — both fixes landed in the real templates (2026-11-04),
           same day as the investigation above.** `no_delay_update_kernel.py`
           and `per_link_delay_update_kernel.py` both: (1) `_spike_loop`'s
           inner loop is now plain `range(SPIKE_BLOCK)`, not
           `tl.static_range` — same compile-time-bounded trip count, no
           longer force-unrolled; (2) `_csr_branch`'s densify is now
           `is_target = (tl.histogram(_tgt, GROUP_SIZE, mask=_lk_valid) > 0)
           & mask` plus, per `edge_tensor_attr`, a zero/`tl.atomic_add(...,
           scope="cta")`/barrier/reload cycle against a new `scratch_ptr`
           kernel argument (one `[GROUP_SIZE]` scratch slice per program,
           reused sequentially across every tile/attr/bucket in a kernel
           call — `Population._execute_update_kernel` always allocates and
           passes it, `n_blocks * group_size` floats, filtered out by the
           existing `arg_names` check for any kernel that doesn't reference
           it). No `_hit` left anywhere in either template. Verified:
           - Full suite (138 tests) green, including `test/test_gpu_
             hardware.py` (manual launches needed a `scratch_ptr` added to
             their args dict) and `test/test_mini_ei.py`/`test_coverage.py`
             (AST-text assertions updated off the literal old `_hit`
             expressions and the now-inaccurate "no atomics anywhere"
             claim — the flush step genuinely still has no atomic, densify
             now legitimately does, so that test was narrowed to check the
             flush statement specifically rather than the whole kernel).
           - The originally-9.5-minutes-and-still-climbing config
             (`GROUP_SIZE=256, n_buckets=8, SPIKE_BLOCK=32`, the real
             default `spike_block`) now compiles+runs in **0.49s** end to
             end via a real `network.step()`, in a fresh process (no
             shared-warm-up confound).
           - `test/test_per_link_delay_kernel_scaled.py` no longer needs to
             dodge any default to stay fast — updated to leave `group_size`/
             `n_delay_buckets`/`spike_block` all at their real defaults
             (previously had to pin `n_delay_buckets=4` instead of the real
             default 8 specifically to avoid the blowup); still passes, 6
             tests (with the small N=4 test) in 2.48s.
           - A from-scratch N=4000 run with *every* tile-size knob left at
             its real out-of-the-box default (`group_size=32,
             n_delay_buckets=8, spike_block=32`, 125 workgroups) compiles
             and runs its first real step in 0.96s, steady-state 0.47s —
             confirming the fix holds with no special-cased parameters at
             all, not just the one specific config that was measured slow
             before.
           Separately isolated and confirmed along the way (microbenchmark:
           `/tmp/.../scratchpad/unroll_vs_loop.py`, not committed): forcing
           the `tl.static_range` unroll isn't merely a compile-time-only
           cost with execution unaffected — at maximum real load (every
           slot in a `SPIKE_BLOCK`-wide window genuinely firing, no
           skipped/wasted iterations either way), the force-unrolled
           version still ran ~1.7x *slower* (61.71µs vs 35.99µs) than the
           equivalent real loop, confirmed via the generated AMDGCN to
           compile to an exact, non-deduplicated repetition of the same
           instruction sequence (`v_permlanex16_b32` occurred exactly
           `32 × SPIKE_BLOCK` times, zero sharing across iterations) — so
           much duplicated code pressures instruction fetch/cache enough to
           lose even when none of the duplicated work was actually wasted.
           At idle (nothing firing), both were equal (~13.3µs each),
           confirming the cost is specifically about code size/compile
           time, not about skipping runtime work differently.
       - **New (2026-11-04) — real-network-scale follow-up, PD14 and a
         controlled isolation experiment.** With the two fixes above in
         place, checked whether real, larger networks (`test/networks/
         PD14`, `mini_ei`) still have compile-time problems.
         - **`PD14_neuron`/`PD14_poisson` could never actually be run, only
           built** — a genuine, pre-existing bug, unrelated to performance.
           `test/networks/PD14/net.py` only ever broadcasts `potential`
           into a per-neuron tensor; `current`/`t_ref_end`/`last_time`
           (`PD14_neuron`) and `next_spike` (`PD14_poisson`) were never
           turned into tensors anywhere, and `Population.__init__` never
           did this automatically — so any real kernel launch crashed
           immediately (`AttributeError: 'Population' object has no
           attribute 'next_spike'`). Confirms TODO's own "not started/
           untested" note was literal: PD14 had never been executed, only
           `build()`-ed. **Fixed, generally** (not PD14-specific):
           `Population._ensure_on_device` — already the lazy, idempotent
           chokepoint called before every real kernel launch on `self`,
           and on a connection's `_source` before reading it — now also
           broadcasts any dynamic attr nobody explicitly set into a
           per-neuron tensor from the node instance's own scalar default
           (`self.instance`'s `__init__`-set value), choosing `torch.bool`/
           `torch.int32`/`torch.float32` from the type recorded in
           `self.dynamic_attrs` at construction. Verified: full suite
           (138 tests) still green; PD14's populations now initialize
           without the patch my own measurement script needed before this
           fix existed.
         - **PD14 still can't reach `.compile()` at all today** — not a
           regression, a real, already-known limitation, now confirmed
           concretely on a real network rather than reasoned about: every
           cortical population's effective period is `0.0` (fully
           autonomous, no explicit `timestep`), while every real PD14
           delay is 1.5-6ms — squarely Part 2's cross-timestep case
           (`delay >= period`), still unimplemented. `compile()`'s existing
           `NotImplementedError` guard catches this correctly. Measuring
           PD14's *update-kernel* compile time at all therefore required
           temporarily forcing an explicit large `timestep` on every
           cortical population (measurement-only workaround, not a fix —
           Part 2 is the real fix, still not started).
         - **With that workaround, PD14's update kernel compiles in a
           reasonable 11.5-24.9s per cortical population (~141s total
           across all 8)** — no longer catastrophic (pre-fix, this
           class of kernel didn't finish compiling in 9.5+ minutes), but
           not trivial either, and clearly population-`n_incoming`-driven
           rather than neuron-count-driven (PD14's 8 cortical populations
           range from 1065 to 21915 neurons but their compile times don't
           track that at all — they track their own connection count,
           6-9 each).
         - **Isolated `n_incoming` directly**, holding target population
           size exactly fixed (`N_TARGET=5000`, `n_blocks=157` identical
           across every trial, only varying how many separate source
           populations feed it): `n_incoming` 1/2/4/9 ->
           1.15s/1.39s/4.24s/19.9s. Confirms `n_incoming` (not population
           size, not topology specifics) is PD14's own dominant remaining
           driver — it reproduces PD14's own worst real case (24.9s at
           `n_incoming=9`) almost exactly (19.9s) in complete isolation.
           Mechanistically different from the `SPIKE_BLOCK`/densify fixes
           above: the connection dispatch is a genuine if/elif chain where
           every branch can run completely different code (different edge
           class, different source, different topology) — there's no
           "swap a forced unroll for a real loop" trick available, since
           unlike `SPIKE_BLOCK`'s iterations (same code, different data),
           different connections are NOT the same code. All `n_incoming`
           branches' code necessarily exists in the compiled kernel
           regardless of which one runs for any given spike.
         - **Checked where, precisely, that superlinearity lives — directly
           asked whether it's an LLVM optimization-pass problem (the
           owner's own research suggested there's "no good way to avoid
           this" short of contributing to Triton itself, but wanted this
           confirmed before accepting that).** Two checks, both decisive:
           - `DISABLE_LLVM_OPT=1` (routes to `disable_optimization=True` in
             the AMD backend's `compile_amdgpu`, confirmed by reading
             `triton/backends/amd/compiler.py` directly) barely changed
             anything: `n_incoming` 1/2/4/9 -> 0.94/1.26/4.06/18.1s vs.
             1.15/1.39/4.24/19.9s optimized — same superlinear shape,
             just uniformly a bit faster. If LLVM's optimizer were the
             cause, disabling it should have flattened the curve, not
             just shifted it down a constant fraction.
           - **Per-stage timing via `triton.knobs.compilation.listener`**
             (a real, documented Triton hook — `CompileTimes` with
             `ir_initialization` [AST -> Triton IR] plus one duration per
             `lowering_stages` entry, keyed by stage extension: `ttir`,
             `ttgir`, `llir`, `amdgcn`, `hsaco`) settled it directly: at
             `n_incoming=9`, `ttgir` lowering alone is 15.6s of the 18.6s
             total (~84%); `llir`+`amdgcn` (where LLVM actually runs)
             together are under 1.5s. Swept all four `n_incoming` values
             with and without `DISABLE_LLVM_OPT=1`: `ttgir` duration is
             essentially identical either way at every point (e.g.
             `n_incoming=9`: 15.617s optimized vs. 15.900s unoptimized),
             while `llir`/`amdgcn` do shrink by roughly half when LLVM
             optimization is disabled (small, as expected, since that's
             exactly the stage it touches) — but that's a minor fraction
             of the total regardless. **Conclusion: the superlinearity is
             not an LLVM optimization-pass cost at all — it's Triton's own
             `ttgir` lowering (TTIR -> TTGIR, i.e. Triton's own MLIR-based
             layout-assignment/legalization pass, which runs entirely
             before LLVM is ever invoked).** Matches the owner's own
             research that there's no good within-Triton-API way to avoid
             this for a genuinely-branchy kernel; a real fix would mean
             contributing to Triton's own `ttgir` lowering pass, not
             anything reachable from kernel-authoring code. Worth
             remembering as a general Triton-authoring lesson, not just a
             naer2-specific one: minimize the number of *structurally
             distinct* branches/tensor-shapes compiled into one kernel
             function if `ttgir` lowering time becomes a problem elsewhere
             — this is exactly why the `SPIKE_BLOCK`/densify fixes above
             (removing *repeated, same-shaped* code) worked so well and
             why `n_incoming` (genuinely *different*-shaped code per
             branch) doesn't have an equivalent fix available.
         - **`mini_ei` confirmed trivial, no issues**: `build()` 0.006s;
           E (100 neurons, n_incoming=2) compile+run 0.537s; I (25 neurons,
           n_incoming=2) 0.016s; first real `network.step()` (gather+merge+
           update, all kernels) 1.085s; steady-state 0.0023s. Small
           population, few connections, no delay kernel at all -- nothing
           here was ever going to be a problem.
         - **Done (2026-11-05) — `compact()` no longer uses `tl.sort`.** At
           PD14 scale (sort width 65536) the old single-`tl.sort` kernel was
           still compiling after 29+ minutes: `tl.sort` emits its bitonic
           network fully unrolled over one register tile, the same
           unroll-everything failure as `tl.static_range(SPIKE_BLOCK)`. It is
           now a merge, done with real loops in BLOCK-sized chunks through
           global memory: densify fresh (prefix sum of group counts), sort
           fresh with a looped bitonic network (global-memory compare-exchange
           plus a barrier per stage), then merge it with carry (always
           already sorted, being the tail of the previous call's output) by
           binary-search rank. Each element's position is its own index plus
           how many entries of the other sequence come before it, with carry
           first on ties. Sizes are runtime args, so one compiled kernel
           serves every population size. Compile ~0.3-0.4s at any size;
           steady state at 685 groups ~1ms (sort skipped) / ~10ms (bitonic
           sort).
           Fast path: the `FRESH_SORTED` constexpr skips the fresh sort.
           `Population.fresh_sorted` sets it from the hook ASTs: it is True
           iff every `n2.spike()` in the node's hooks (and incoming edge
           hooks) is bare. A bare call always records the kernel's `t`, so
           one launch's fresh output shares a single timestamp. Any explicit
           emission time (an emission delay, not a transmission delay) turns
           it off. Output is ordered by time only, and ties have no defined
           order (confirmed: no consumer relies on it).
           Carry overflow: the old kernel wrote past `carry.cap`; writes are
           now masked to the buffer, but `carry.count` keeps the true
           (possibly > cap) count, so the excess is dropped yet an overflow
           stays detectable. Nothing checks it yet.
           Tests: `TestCompactAtScale` (685 groups, multi-round, both paths,
           vs. a reference), `test_fresh_sorted_flag.py`.
           Still on `tl.sort`: `merge()`/`combine()` — the same blowup
           applies once their widths grow (`combine()` sizes off live
           counts, so it can grow at scale); same fix pattern.
       **⚠ FLAG FOR WHEN WE GET HERE — open correctness concern, not yet
       discussed.** The owner has flagged (2026-11-03) that something about
       the *current* (Part 1, already-shipped) per-link-delay kernel looks
       off to them once Part 2 is in view, but deliberately hasn't said
       what yet — "there's something I don't understand about the kernel
       there yet, but we'll only discuss it then." **Do not start
       implementing Part 2 without raising this explicitly first** and
       actually discussing it, rather than assuming Part 1 is settled.

       **Part 2 — cross-timestep delay (the part that still needs
       solving).** A delay that exceeds one population timestep pushes a
       delivery into a *later* timestep entirely, which is a different
       kind of problem from bucketing within one — it's exactly what
       `CarryBuffer`/`compact()` already exist for (holding a "not yet
       due" entry across rounds). The new wrinkle: responsibility for that
       spike's remaining lifetime transfers from the *source* population
       (which today owns `_carry`/`_fresh`/`_due`, compacting its *own*
       output once for all downstream targets) to the *receiving*
       population specifically, once "how many of the receiver's own
       timesteps away" becomes part of what's being tracked — not yet
       designed: does this still fit the source-owns-compaction model
       (same once-per-source economy of scale this was built for), or does
       spanning multiple-of-the-*receiver's*-timestep intervals need the
       receiver itself to hold a per-connection carry, since different
       incoming connections to the same population could need different
       bucket counts/timesteps? Open for the next pass at this.
     - **Historical reasoning trail (superseded above, not the plan):**
       the original idea was to sort each source row's outgoing links by
       delay once, statically, then combine multiple simultaneously-firing
       sources' pre-sorted rows at runtime via a dedicated fixed-width
       merge-network kernel (persistent, fixed-width sorting heads exiting
       or continuing across `log n` passes by parity, rather than one
       kernel launch per pass) landing on a single-target-per-entry buffer
       — avoiding the update kernel's CSR walk entirely in favor of one
       lane per delivery, gathered/scattered by target id, with a
       collision guard narrowed from "compare source+time" to just
       "do any two co-processed lanes share a target" (safe within one
       firing source's own row via the existing no-multapses invariant).
       Benchmarking this (`torch.argsort` as a measured proxy for a real
       sort kernel, same methodology as the GPU-vs-CPU sort numbers below)
       showed the "expand" (CSR gather + key-pack + write) and "sort"
       stages cost about the same as each other and together landed at a
       converging ~6-8x the no-delay baseline at the sizes safely testable
       on this hardware (extrapolated ~5-7s/timestep at the stated 1e8
       worst-case synapse-activation ceiling, vs PD14's real measured max
       per-connection fan-out of 3,826, not 10,000). Checking whether
       static edge-coloring could dodge the sort showed it can't — the
       collision set that needs detecting is whichever of *today's* firing
       sources happen to coincide, which is a dynamic fact no static
       precomputation resolves — which is what led to actually measuring
       collision frequency directly instead of continuing to design around
       it, and to the decision above.
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
     `build()` itself now runs fine (see item 1's per-link-delay entry above
     — confirmed manually, 9 populations, 59 connections, not yet as a
     committed test), but `.compile()` on any cortical population still
     hits the per-link-delay stub, since every PD14 connection carries a
     real per-link delay tensor — there is no delay-free path through it at
     all. Blocked on the per-link-delay kernel existing, not just untested.
     `mini_ei` (a scaled-down variant that drops the Poisson population and
     per-link delay) is the closest thing to AST-inspection-tier coverage
     `PD14_neuron`/`PD14_connection` get today. Compile-time surprises (at
     ~78k neurons, cortical-scale connection counts) are still possible,
     separate from execution, once the per-link-delay kernel exists to
     actually try this against.
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