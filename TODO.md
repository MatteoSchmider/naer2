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
       A connection's per-link delay is split into that many static,
       quantile-balanced buckets (computed once, like `Topology.blocked()`
       — no runtime sort, reusing the existing CSR-walk/densify-reduce
       machinery per bucket, no gather/scatter-by-delivery redesign needed
       at all) *within one population timestep* — i.e. `Population.
       timestep` (see above) is exactly the width this bucketing needs to
       know, which is why getting per-population scheduling right came
       first. A connection whose whole delay spread already fits inside
       one timestep collapses to effectively one bucket for free — "ignore
       small differences" falls out of the scheme rather than needing a
       special case.
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