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
- **Construction and compile time: a baseline** -- measured now, cut after both refactors.

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
