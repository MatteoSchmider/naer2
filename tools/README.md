# tools

Benchmarks and diagnostics for the spike-management redesign's green-field
kernels (`naer2/greenfield`, `naer2/intake.py`). All run on the default GPU;
`--rdna2` (where offered) loads the shim for the RX 6600 eGPU, run with
`HSA_OVERRIDE_GFX_VERSION_2=10.3.0 HIP_VISIBLE_DEVICES=1`.

| tool | what it does |
|---|---|
| `pd14_setup.py` | the shared PD14 benchmark: the 8 cortical populations as green-field `Pop`s with their `TopologyNew` connections (0.1 ms buckets), PD14-like firing, one round = every update kernel, then every intake kernel |
| `pd14_roundtrip.py G` | wall clock per round, populations on their own streams; `--breakdown`: host issue time and each phase alone |
| `trace_round.py G` | every program of every kernel on one wall-clock axis, from the kernels' own stamps (`naer2/trace.py`): the round's timeline (concurrent) and per kernel its span, programs alive and phases (isolated); `--heat N` keeps the clock up, `--no-counters` compiles the stand-in receive counters out (PD14's kernel) |
| `walk_sweep.py G` | update-kernel walk configurations compared fairly: one process, one network, configurations interleaved round by round |
| `intake_sweep.py G` | the same for the uniform intake's program count and tile shape |
| `kernel_isa.py G` | compiles the update kernel and summarizes its ISA: VGPRs, spills, LDS, per loop the instruction mix and the highest VGPR touched |
| `rdna2.py` | lets Triton 3.9 compile for RDNA2 (lowers as gfx1100, codegen as gfx1030) |

## Measuring on these GPUs

- **Clocks.** Short or sync-heavy runs keep the 780M at its lowest DPM level
  (800 MHz) and send the eGPU into its low power states; sustained load runs
  the 780M at ~2.0-2.7 GHz. Compare configurations only within one run
  (interleaved, as the sweeps do); absolute numbers need `--heat`.
- **CUDA events** are unreliable here (even negative); use the kernels' own
  stamps or wall clock over many rounds.
- **Topology builds** occasionally die with `unspecified launch failure`
  inside torch's `argsort` / `unique` (ROCm, both GPUs); retry.
- **The PD14 network isn't seeded**: separate runs build different instances
  (±10-15% between runs).

## experiments/

One-off investigations, kept for what they established:

| script | finding |
|---|---|
| `atomic_bench.py` | returning `atomic_add` is fast on shared addresses (~2,300/us on 860 addresses, they stay in L2) and slow only when spread over ~1M addresses (~370/us): intake was never atomic-bound |
| `store_bench.py` | intake's appends are bound by write locality: 4 field arrays ~76-95 us per 512-append tile, 16-byte records ~66, records interleaved by cache line across cells ~21 (capacity-independent); delivery reads stay within noise across layouts -- hence the ring's record layout and `_rec` |
| `occupancy_lds.py` | LDS is 128 KB per WGP on both GPUs, at most 16 programs of 4 warps per WGP; up to 8 KB per program never limits occupancy |
| `wave_segments.py` | splitting segments per wave (each wave walks only its own neurons' links) would cost PD14 x1.5 ring entries and chunks, +663 MB of segment metadata, to save part of the walk -- a net loss while per-entry costs dominate |
