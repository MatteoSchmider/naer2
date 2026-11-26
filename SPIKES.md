# Spike management: queues and slots

Planning notes (2026-11-22), from Brunel's profile: phase 1 of PLAN.md, before the refactors. Replaces the general
pull (side runs, watermark, sequential appends) and the delay classes of BRUNEL.md step 3.

## Why

Brunel (N = 12500, 780M): ~230 ms wall per 1.5 ms step, 94% of it delivery. 98% of the pending records sit in side
runs, every cell has some, so every slot is merged one link at a time (`_merge_links`). The causes:

- **The history loses the producer's order**: E's programs emit in time order (0 descents per program), but a run's
  history is one flat list appended through atomics from every program -- 48% descents in list order.
- **Order is enforced on arrival, group-wide**: a link below the group's watermark (any cell's largest e) goes to
  a side run; only order within a cell matters.
- **The fallback is the common path**: side runs are a sorted insertion per link, delivered one link at a time.
- **Brunel sits in a slotted ring it doesn't need** (48 slots: 15 for D, 30 for its sources' windows, 3; the slot
  width the network's default 0.1 ms): with one delay per connection, emission order already is arrival order.

## The design

A population's incoming links come in **streams** -- a connection, or a connection's delay bucket -- each with
arrival = e + d. Two layouts, chosen per population (`Population`, from its incoming delays: a bug today, the
slotted ring always):

- **Queues** (every incoming connection one delay value): a queue per (group, incoming connection), in emission
  order -- its arrival order. Delivery merges a population's queues by arrival (e + D_c) up to the run's end, a
  cursor per queue (the block merge of step 3, its runs the queues). Exact. No slots: the next due arrival is the
  earliest head's e + D_c exactly (gating, scheduling: no slot ends, no `_floors`, no far future, no ring length).
  Several delay values across connections are covered; connections sharing a value don't share a queue (two
  sources interleave).
- **Slots** (any per-link delays): the ring of slots as now, every link of a slot at the slot's time (center or
  end, `slot_time`) -- order within a slot doesn't change a time. **Order there: reproducible, not emission order**
  (decided 2026-11-22): monotone times per target as before; within a slot a fixed order, never an atomics' one.

Both need **a deterministic spike stream** and **appends that keep its order**; queues need it **sorted by e** too:

- **Sorted, deterministic histories**: a source run's spikes ordered by (e, node) -- a counting sort, in naer1's
  style (its spike management: one counting sort for all of it). Queues then fill in order across pulls (a
  source's runs follow each other in time; within a run, sorted), and a slot's records come in a fixed order.
- **Order-preserving appends**: claims by a scan over a tile in stream order, not LDS atomics.
- **Uniform sources: a special case**: a run with one emission time is sorted already (ties by node); the
  specialization stays a compile-time one (PD14's pull).

## What goes

The watermark, side runs, `_side_insert`, `_merge_links`, the sequential general pull, the uniform contract
(flag 2), the delay classes (`NC`, `_stage_merged`'s classes become queues), `_floors` for queue populations, and
lost links (one batched pull for every source: it stops a run and reruns it).

## Order: the producer places, the consumer merges (decided 2026-11-22)

No program ever waits on another (no decoupled look-back: PD14's kernel latency can't spare it).

- **Deterministic placement, non-uniform producers** (done 2026-11-22: `EmissionHistory(regions=...)`, the
  generated kernel's `_REGIONS`): each producer program writes its spikes into its own fixed region of the run,
  and its count -- no atomics placing them; the stream is the regions in program order. Measured for every
  producer first: PD14 21.0 -> 22.6 us a round on the 4090 (the consumer's per-program counts -- one more dependent
  load per e-group --, a unit table, a longer search), so a uniform producer keeps its one flat list (PD14 back
  at 21.1): its summation order stays irreproducible for now (TODO.md).
- **Sorted regions, non-uniform producers only** (made in the producer, locally):
  - from receive (D-3): a program's spikes come in arrival order per warp, the warps interleaved -- a per-slot
    emission buffer in LDS (Brunel: ~1 spike a program and slot), ranked by (e, node) at the slot's end and
    appended to the program's region: the region sorted;
  - from an update loop (a Poisson source): the program ranks its run's spikes before writing them;
  - bare, uniform emission (PD14): node order is the order -- nothing.
- **The consumer merges, for queues of non-uniform sources only**: a source's run in e order -- a counting sort by
  emission bin (O(n), LDS histograms), then a rank within each bin by (e, node) (Brunel ~70 spikes per 0.1 ms
  bin); the total key makes it deterministic whatever the pieces' order. At large N, the regions' sortedness lets
  the bins merge sorted pieces instead of ranking pairwise. Slots and uniform sources take the stream as it comes.

## Steps (to plan in detail)

1. The layout choice per population, queues for homogeneous connections (and the network's default slot width
   out of it); the queue structure in the pooled arenas, cursors, freeing.
2. Deterministic histories (per-program regions; PD14's cost measured first), sorted regions for non-uniform
   producers, order-preserving appends.
3. Queue delivery: the k-way merge up to the run's end; gating and scheduling from the queues' heads.
4. One pull for every source; the side runs and the general pull removed.
5. Brunel against its numbers (rates, CV, the reference), speed on the 780M and the 4090; PD14 unchanged
   (speed, accuracy), its summation order now reproducible.
