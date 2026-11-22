# Brunel (2000) in naer2, and the delivery times it needs

Planning notes (2026-11-22): the model, the decisions behind it, and the work it takes -- phase 1 of PLAN.md,
before the refactors.

## The model (A first, B later)

- **Brunel (2000), model A**: a sparse balanced network of N_E = 4 N_I identical LIF neurons with delta synapses
  (a PSP J = 0.1 mV, inhibitory -g J), one homogeneous delay D (1.5 ms), tau = 20 ms, theta = 20 mV, V_r = 10 mV,
  a 2 ms refractory period; gamma = C_I / C_E = 0.25 fixed. Next to PD14: `test/networks/Brunel/`, its own neuron
  class (code from `lif_ed` and PD14, no reuse).
- **Event-driven, precise**: the potential only changes at an input's arrival, so receive integrates to it (decay
  since the last arrival, + J) and fires right there -- a spike at the exact arrival time, not on a grid. Not to
  match NEST: the event-driven result is the more precise one, NEST's grid-based one approaches it as its
  timestep goes to 0. Precise spike times make the sources non-uniform: Brunel's deliveries go through the
  general pull -- the weakest path, covered by an established model next to PD14 on the uniform pull.
- **External drive**: a Poisson population (the standard way): N_P = N_E sources, excitatory, p = epsilon into E
  and I (C_ext = C_E exactly), weight J, the rate eta nu_thr (decided 2026-11-22; naer1's N_P = N_E + N_I with
  p = 0.1 not taken). The source has only update(), emitting
  its next spike (precise, n2.spike(next_spike)); naer2 derives its timestep itself: the minimum delay of its
  outgoing connections (D) -- no explicit timestep. D on its connections too: one homogeneous delay everywhere.
- **build(N, K=None, eta=2.0, g=5.0, ...)**: strong scaling by default (epsilon = 0.1 constant: C_E = epsilon
  N_E, growing with N); K given: a fixed in-degree (K synapses per neuron), overriding epsilon. The parameters in
  a description dict like PD14's. Model B (different parameters per population, a delay distribution) later, as
  its own function in the same file.
- **What it tests**: event-driven populations and gated launches at scale (planned and captured windows; not
  clock-only: no periodic windows), the general pull, receive with time math (an exp per delivered link), spikes
  inside delivery, one delay class (one segment per source and group), a size to turn up (rings, arenas, the
  2047-block limit). Its STDP variant (Morrison et al. 2007, NEST's hpc_benchmark; naer1's main benchmark) after
  refactor B, with the edge hooks.

## Delivery times (D-1, decided 2026-11-22)

Receive's time, by the population's incoming delays -- always non-decreasing per target (never stepping into the
past), in emission order:

- **Only homogeneous delays** (every incoming connection one delay value): exact. Within a connection arrival is
  emission + a constant, so emission order is arrival order; the time is e + D_c.
  - **One delay value** (Brunel: D everywhere): nothing more -- the ring as it is, its records in emission order.
  - **Several values**: a delay-class dimension in the ring -- a cell per (group, class, slot), each class's
    records in emission order (= its arrival order); delivery merges a slot's classes by arrival (each class
    sorted: a record's place is its rank in its class + a binary search into the others) -- exact arrival order.
    Inversely proportional to the slots: a class is a ring of its own, its delay added on top.
- **Any per-link delays**: the ring of slots as now, in emission order, and every link of a slot delivered at the
  slot's time -- its center by default (statistically unbiased), its end by a flag on the population
  (`slot_time`: causal, concentrating deliveries and the update at one time).
- **The records keep the emission time** (sorting, side runs, the merge unchanged); delivery computes the time.
- **A bare n2.spike() in receive** fires at exactly the delivery time.

## The steps

Order (2026-11-22): 1, 3, 2, 4, 5 -- the user's turn between every step.

1. **Delivery times, one class** (done 2026-11-22: the constexprs T_A, T_C, T_B from `Population._delivery_time`;
   tests in test_fused_generated_gpu.py, `test_receive_time*`): the records back to the emission time; the population decides its mode (one
   homogeneous delay: e + D; any per-link delay: the slot's center or end, `Population.slot_time`; several
   homogeneous values: the slot time until step 3); the kernel computes receive's time (a constexpr mode, the
   delay and slot geometry as arguments). Tests: the D-1 gap test (one delay: exact), a per-link-delay case at the
   slot's center and end, monotone times per target.
2. **D-3, spikes inside delivery**: n2.spike() in receive into a per-program emission buffer, flushed into the
   history with the run's own spikes. Tests: the D-3 gap test, an event-driven relay firing on arrivals.
3. **Several homogeneous delays** (done 2026-11-22: `Population._delay_classes`, `Ring.n_classes`, the constexpr
   NC; delivery `_stage_merged` -- a block merge, every class's next STREAM records ranked by binary search in
   LDS -- and `_merge_classes` for slots with side runs, one link at a time; tests `test_several_homogeneous_*`):
   the class dimension (pull: a link's cell by its connection's class; delivery: a
   slot's classes merged by arrival). Tests: two connections with different delays, exact arrival order per
   target against a host reference.
4. **A source's derived timestep**: an update() population without inputs runs every min outgoing delay. Tests:
   a Poisson source with no timestep set.
5. **Brunel A**: the neuron, the Poisson source, a fixed in-degree pattern, build(); tests at small N (exact
   deliveries, rates in the asynchronous irregular regime); a run at scale on the 4090.
