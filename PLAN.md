# naer2 — the order of the open work

TODO.md lists what is open; this is the order to do it in (2026-11-22). The refactor splits in two: **A** the host
side (`Network`: step, plan, planned and captured windows, the fusion and periodic hooks; `PopulationRuntime`:
launching, status, windows, judging, sizing, recovery) and **B** the front end and model structures
(`Population` as a god class, `Fabric`, `Topology` as plain tensors, file packaging). The ring and scheduling
work runs through A, most feature gaps through B and the generator.

## 1. Before any refactor

(Brunel, the delivery times it needs, and the steps: BRUNEL.md. Its speed -- spike management with queues
and slots, replacing the general pull: SPIKES.md.)

- **An event-driven test model: Brunel (2000)**, the sparse balanced E/I network -- event-driven only, scaling
  with N. Coverage for the general (event-driven) path that refactor A must keep, and its first real driver; its
  STDP variant (Morrison et al. 2007, NEST's hpc_benchmark: naer1's main benchmark) comes once edge hooks exist
  (phase 3).
- **D-1: receive's time by the incoming delays** (exact e + D for homogeneous delays, else the slot's time;
  BRUNEL.md) -- an event-driven LIF integrates to each arrival (Brunel needs it).
- **D-3: spikes inside delivery** (n2.spike() in receive, straight into the run's history) -- an event-driven
  LIF fires on an arrival (Brunel needs it); also the coverage network's B.
- **Construction and compile time: a baseline** -- measured now, cut after both refactors. Done 2026-11-28
  (`tools/baseline.py`, the 4090; host memory above the process before any model, 0.75 GiB):

  | | PD14 | Brunel 10k | Brunel 50k |
  |---|---|---|---|
  | nodes, links (a node) | 78k, 302M (3867) | 18k, 18M (1000) | 90k, 450M (5000) |
  | build (host: model, canonical topologies) | 16.7 s; host peak 16.4 B a link | 1.1 s; 27 B | 15.1 s; 26 B |
  | first step (device topologies, runtimes, compile) | 12.5 s; host 20 B, device peak 12.4 B a link | 3.8 s; 56 B, 18.6 B | 5.8 s; 25 B, 12.7 B |
  | first captured window (capture, fused compile) | 10.3 s | 0.07 s | 0.09 s |
  | steady, windows without a resize: wall / GPU a round | 28.4 / 19.9 us | 480 / 284 us | 568 / 371 us |
  | steady overall | 97 us a round (25 resizes 0.43 s, max 388 ms; 5 captures 0.25 s) | 0.66 ms wall a simulated ms | 0.77 ms |
  | device held, steady (topologies + rings) | 12.2 B a link (10.6 + 1.5), 47 KB a node | 6.6 B (5.5 + 0.8) | 5.7 B (5.5 + 0.2) |
  | device peak, steady | 22.9 B a link (a resize: old + new arena) | 12.6 B | 11.7 B |
  | host, steady | 19 B a link, 73 KB a node | 53 B, 52 KB | 23 B, 112 KB |

  To cut: the host keeps ~20 B a link through the simulation (the canonical topologies, once on the device); PD14's
  first window ~10 s (the fused compile); a resize up to ~0.4 s (segmented arena growth); Brunel's start arenas
  (`ring_in_flight` 0.25: ~4 B a link at the device's peak, shrunk away later).

## 2. Refactor A, then what builds on it

- **Refactor A** (host side).
- **The ring project**: segmented arena growth, ring memory (a per-group overflow log, the reserve, the 2047-block
  limit), a history that grows for a slow consumer, the untested `lost` path (blocks missing: only a shrink
  below what's in use could, since the general pull's removal, 2026-11-28).
- **The step path's schedule entries** in device memory (if stepping still matters next to captured windows).
- **Scheduling**: implicit scheduling (no 0.1 ms default), `n_slots` revisited -- against Brunel.

## 3. Refactor B, and the features around it

- **D-6 (receive variants per connection), D-7 (several per-link attrs)** first, on the current structure: they
  decide how per-connection receive code and per-link attributes are laid out (plan numbering, the fabric's attr
  blocks) -- refactor B should know those shapes.
- **Refactor B** (front end, model structures).
- **On the new structure**: edge forward and D-9 (its destination), D-10 (source attrs), D-11 (edge backward: a
  reverse topology), narrower link targets (`uint8` / `int16`, with `Topology` as tensors), the rest of the
  coverage network -- then Brunel with STDP.

## 4. After both

- README, the showcases, the test suite (structure, speed, coverage).
- Construction and compile time cut (against phase 1's baseline).
- Autotuning per device (group size, warps, the walk split).
- PD14 with a Poisson background, and Brunel (with STDP), against NEST and GeNN: accuracy, speed, the front
  end's definition and build time.
