"""
Generates a population's fused kernel from its hooks: the entry and the
receive function around the fixed template (naer2/transpiler/templates/
fused.py), in exactly the shape of the hand-written reference (test/
fused_reference.py) -- see TODO.md for the decisions (D-n) it follows.

  entry    one <attr>_ptr per dynamic node attr, then the template's fixed
           arguments; _pull; the attrs receive (or a per-delivery send)
           touches loaded as the state tuple; _deliver with the generated
           receive; the remaining attrs loaded; update then send (time =
           t = t_end); every dynamic attr stored; each n2.spike() site
           flushed with _emit_spikes, then _emit_close
  receive  _receive(state, hit, time, w, srcword) -> state: the transpiled
           receive rooted at hit, edge.<attr> = w (the connection's one
           per-link attr), plus send where a delivery can change its outcome
           (D-2)

Hook bodies go through HookTranspiler as for the old templates, then
GluonTransformer (tl -> gl). n2.spike() lowers to a fire mask and an
emission-time tensor per site (bare: t_start, D-4), flushed after send;
inside delivery (receive, a send after every delivery: D-3) straight into
the run's history (fused.py _emit_delivered; bare: the delivery time).
UNIFORM_EMISSION holds when every site is bare and none is inside
delivery. Not supported yet (NotImplementedError, each with its decision):
connections with different receive code (D-6), more than one per-link attr
(D-7), edge forward hooks (HOOK(forward)), source.x (D-10), edge backward
(D-11). Receive's time follows the incoming delays (D-1, BRUNEL.md): the
constexprs T_A, T_C, T_B (Population._delivery_time).
"""

import ast
import atexit
import importlib.util
import itertools
import pathlib
import shutil
import textwrap
import tempfile

from ..analyzers.used_members import ReadMemberFinder, UsedMemberFinder
from ..context import TranspilerContext
from ..hook_transpiler import HookTranspiler
from ..transformers import GluonTransformer, MaskAliasTransformer, TargetMaskTransformer

_STMTS = lambda src: ast.parse(src).body

# the entry's fixed arguments after the node attrs' pointers (the template's phases take them)
# (a run's own values -- run id, round, the slots it delivers, its window -- come from its schedule entry,
# sched[sched_i]: fused.py _run_entry, naer2/schedule.py; the status slot the entry's, else the run id mod
# n_status: fused.py _status_slot)
# (seed: the population's own, for random.*: RandomLoweringTransformer)
FIXED_ARGS = ("n_neurons", "plan", "n_conns", "ring", "wmark_ptr", "oflags", "ctl", "status", "n_status", "sched",
              "sched_i", "ev", "slot_width", "received_ptr", "n_received_ptr", "log_ptr", "log_cap", "history", "hist_cap",
              "trace_ptr", "seed", "check_ptr")
CONSTEXPRS = ("GROUP_SIZE", "K", "SK", "KP", "UNIFORM_SOURCES", "N_SLOTS", "LOG", "STREAM",
              "U", "SPT", "NUM_WARPS", "MAX_CONNS", "LOG_CJ", "COUNT", "BATCH", "NSP", "TRACE",
              "TRACE_CAP", "POOL_B", "ARENA_CHECK", "T_A", "T_C", "T_B", "NC")
DO_NOT_SPECIALIZE = ("n_neurons", "n_conns", "n_status", "sched_i", "log_cap", "hist_cap", "seed")


class _SpikeSites:
    """The fused lowering of n2.spike(): each site's fire mask and emission
    time, kept as _fire<k> / _e<k> for the entry to flush after send; inside
    delivery (D-3) each site emits right there (_emit_delivered: receive's
    ectx and nodes)."""

    def __init__(self, where: str):
        self.where = where  # "final" (update / send after update) or "delivery"
        self.sites: list[bool] = []  # per site: bare (the flushed ones, _fire<k> / _e<k>)
        self.loop_sites: list[bool] = []  # per site inside a while loop (emitted where it stands): bare

    def __call__(self, mask: str, time_expr: str, bare: bool, in_loop: bool = False) -> list[ast.stmt]:
        if self.where == "delivery":
            self.sites.append(False)  # (a spike at the delivery's time: never the window's start)
            return _STMTS(f"_emit_delivered({mask}, ({time_expr}), nodes, ectx)")
        if in_loop:  # (fires any number of times: into the history at once, D-3)
            self.loop_sites.append(bare)
            return _STMTS(f"_emit_spikes({mask}, ({time_expr}) + gl.zeros([GROUP_SIZE], gl.float32, layout=LN), "
                          f"offs, pid, n_groups, history, hist_cap, run_id, K, SPT, NUM_WARPS, _REGIONS)")
        k = len(self.sites)
        self.sites.append(bare)
        return _STMTS(f"_fire{k} = {mask}\n_e{k} = ({time_expr}) + gl.zeros([GROUP_SIZE], gl.float32, layout=LN)")


class FusedKernel:
    """A generated fused kernel: its source, the loaded entry, and what the
    runtime needs to call it -- the dynamic attrs in argument order, which
    of them the receive state holds, the per-link attr staged as w, and
    whether its emission is uniform."""

    def __init__(self, source: str | None, kernel, attrs: list[str], state_attrs: list[str], edge_attr: str | None,
                 uniform_emission: bool, stage_e: bool, consts: dict | None = None):
        self.source, self.kernel = source, kernel
        self.attrs, self.state_attrs, self.edge_attr = attrs, state_attrs, edge_attr
        self.uniform_emission, self.stage_e = uniform_emission, stage_e
        self.consts = dict(consts or {})  # constexprs of the kernel's own, beyond the template's


class FusedKernelTemplate:
    """Builds a population's FusedKernel (see the module docstring)."""

    def __init__(self, population):
        self.pop = population

    # ------------------------------------------------------------------ hooks

    def _context(self) -> TranspilerContext:
        pop = self.pop
        return TranspilerContext(
            static_attrs={k: v for k, v in pop.instance.__dict__.items() if k not in pop.dynamic_attrs},
            dynamic_attrs=pop.dynamic_attrs,
        )

    def _receive_variant(self):
        """The one (receive hook, edge attr) all incoming connections share."""
        variants = set()
        for conn in self.pop.incoming:
            forward, backward = conn.fabric.edge.get("forward"), conn.fabric.edge.get("backward")
            if forward is not None and forward.assigned_members:
                raise NotImplementedError("edge forward hooks aren't generated yet: they run on link lanes while "
                                          "staging (TODO.md, HOOK(forward), D-7 / D-8)")
            if backward is not None and backward.assigned_members:
                raise NotImplementedError("edge backward hooks are out of scope for now (TODO.md, D-11)")
            hook = self.pop.node.receive.get(conn._target_port or "receive")
            if hook is None:
                variants.add((None, None, ()))
                continue
            if UsedMemberFinder(hook.ast, "source").run():
                raise NotImplementedError("source.<attr> in receive is unsupported for now (TODO.md, D-10)")
            edge_rename = {hook_name: real for real, hook_name in (conn._fabric_map or {}).items()}
            used = {edge_rename.get(n, n) for n in UsedMemberFinder(hook.ast, "edge").run()}
            tensor = tuple(sorted(a for a in used if a in conn.fabric.tensor_attrs))
            constants = tuple(sorted((a, conn.fabric.constant_attrs[a]) for a in used
                                     if a not in conn.fabric.tensor_attrs))
            if len(tensor) > 1:
                raise NotImplementedError(f"receive reads {len(tensor)} per-link attrs {list(tensor)}: one per "
                                          f"connection for now (TODO.md, D-7)")
            variants.add((id(hook), tensor, constants, tuple(sorted((conn._fabric_map or {}).items()))))
        if len(variants) > 1:
            raise NotImplementedError("incoming connections with different receive code (ports, per-link attrs, "
                                      "edge constants) need per-variant dispatch (TODO.md, D-6)")
        if not self.pop.incoming:
            # no connections in the DSL (a runtime wired up with topologies directly): the node's default
            # receive, every edge.<x> it reads a per-link attr
            hook = self.pop.node.receive.get("receive")
            if hook is None:
                return None, None, None
            if UsedMemberFinder(hook.ast, "source").run():
                raise NotImplementedError("source.<attr> in receive is unsupported for now (TODO.md, D-10)")
            used = sorted(UsedMemberFinder(hook.ast, "edge").run())
            if len(used) > 1:
                raise NotImplementedError(f"receive reads {len(used)} per-link attrs {used}: one per connection "
                                          f"for now (TODO.md, D-7)")
            return hook, None, (used[0] if used else None)
        conn = self.pop.incoming[0]
        hook = self.pop.node.receive.get(conn._target_port or "receive")
        if hook is None:
            return None, None, None
        edge_rename = {hook_name: real for real, hook_name in (conn._fabric_map or {}).items()}
        used = {edge_rename.get(n, n) for n in UsedMemberFinder(hook.ast, "edge").run()}
        tensor = [a for a in used if a in conn.fabric.tensor_attrs]
        return hook, conn, (tensor[0] if tensor else None)

    # ------------------------------------------------------------------ build

    def build(self) -> FusedKernel:
        pop, ctx = self.pop, self._context()
        attrs = list(pop.dynamic_attrs)
        send = pop.node.send
        hook, conn, edge_attr = self._receive_variant()

        # which attrs the receive state holds: those receive (and a per-delivery send) touch
        send_after_delivery = hook is not None and bool(hook.assigned_members & ReadMemberFinder(send.ast,
                                                                                                 "self").run())
        touched = set()
        if hook is not None:
            touched |= UsedMemberFinder(hook.ast, "self").run()
            if send_after_delivery:
                touched |= UsedMemberFinder(send.ast, "self").run()
        state_attrs = [a for a in attrs if a in touched]
        later = [a for a in attrs if a not in touched]

        # ---- receive (and the per-delivery send), rooted at hit; a bare n2.spike() there: at the delivery time
        recv_body: list[ast.stmt] = []
        stage_e = False
        delivery_sites = _SpikeSites("delivery")
        if hook is not None:
            recv_body += HookTranspiler(hook, ctx, spike_lowering=delivery_sites,
                                        default_spike_time="time").emit_for_spike_loop(
                set(pop.dynamic_attrs), fabric_map=conn._fabric_map if conn else None,
                source_map=conn._source_map if conn else None,
                edge_static_attrs=conn.fabric.constant_attrs if conn else {}, source_static_attrs={})
            if send_after_delivery:
                send_d = HookTranspiler(send, ctx, spike_lowering=delivery_sites,
                                        default_spike_time="time").emit()
                send_d = MaskAliasTransformer().transform(send_d)
                recv_body += TargetMaskTransformer(set(pop.dynamic_attrs)).transform(send_d)
            recv_body = GluonTransformer().transform(recv_body)
            stage_e = _uses_time(hook) or bool(delivery_sites.sites)  # (a spike there: at the delivery time)
        recv_src = self._receive_source(state_attrs, recv_body, edge_attr, hook)

        # ---- update, then send, once per run (time = t = t_end), rooted at the group's mask
        sites = _SpikeSites("final")
        final: list[ast.stmt] = []
        for h in (pop.node.update, send):
            name = h.parameter_type_to_name.get(_time_type())
            if name and name not in ("time", "t"):  # the hook's own name for its time
                final += _STMTS(f"{name} = t_end")
            stmts = HookTranspiler(h, ctx, spike_lowering=sites, default_spike_time="t_start").emit()
            final += TargetMaskTransformer(set(pop.dynamic_attrs), mask="mask").transform(stmts)
        final = GluonTransformer().transform(final)

        # the history: a region per program where the spikes need ordering (any non-bare site: history.py)
        uniform = all(sites.sites) and all(sites.loop_sites) and not delivery_sites.sites
        source = self._module_source(attrs, state_attrs, later, recv_src, final, sites.sites, stage_e,
                                     stage_w=edge_attr is not None, emit_delivered=bool(delivery_sites.sites),
                                     regions=not uniform)
        kernel = _load(source, f"naer2_fused_{type(pop.instance).__name__}_{id(pop)}")
        # receive's time (D-1): its map from a record's emission time and slot, by the incoming delays
        t_a, t_c, t_b = pop._delivery_time()
        return FusedKernel(source, kernel, attrs, state_attrs, edge_attr,
                           uniform_emission=uniform, stage_e=stage_e,
                           consts=dict(T_A=float(t_a), T_C=float(t_c), T_B=float(t_b)))

    @staticmethod
    def _receive_source(state_attrs, body, edge_attr, hook) -> str:
        unpack = f"    ({', '.join(state_attrs)},) = state\n" if state_attrs else ""
        pack = f"    return ({', '.join(state_attrs)},)\n" if state_attrs else "    return state\n"
        if hook is None:
            return ("@gluon.jit\ndef _receive(state, hit, time, w, srcword, ectx, nodes):\n"
                    "    \"\"\"No receive (no incoming connection carries one).\"\"\"\n    return state\n")
        time_name = hook.parameter_type_to_name.get(_time_type())
        lines = [unpack, "    is_target = hit\n"]
        if time_name and time_name != "time":
            lines.append(f"    {time_name} = time\n")
        if edge_attr is not None:
            lines.append(f"    edge_{edge_attr} = w\n")
        body_src = "".join("    " + line + "\n" for stmt in body for line in ast.unparse(stmt).splitlines())
        return ("@gluon.jit\ndef _receive(state, hit, time, w, srcword, ectx, nodes):\n"
                f"    \"\"\"The transpiled receive ({hook.function.__qualname__}), rooted at hit.\"\"\"\n"
                + "".join(lines) + body_src + pack)

    @staticmethod
    def _module_source(attrs, state_attrs, later, recv_src, final, sites, stage_e, stage_w=True,
                       emit_delivered=False, regions=False) -> str:
        args = ",\n    ".join([f"{a}_ptr" for a in attrs] + list(FIXED_ARGS)
                              + [f"{c}: gl.constexpr" for c in CONSTEXPRS])
        names = ", ".join([f"{a}_ptr" for a in attrs] + list(FIXED_ARGS) + list(CONSTEXPRS))
        # every node attr loaded first thing in the program, with the pull's first reads (none depends on the
        # run's entry; the population's previous run wrote them): their latency under the pull, not before
        # delivery or update -- PD14 on the 4090: ~0.8 us a round (limitations.md)
        early = "".join(f"    {a} = gl.load({a}_ptr + offs, mask=mask)\n" for a in state_attrs + later)
        state_tuple = f"({', '.join(state_attrs)},)" if state_attrs else "()"
        unpack = f"    ({', '.join(state_attrs)},) = state\n" if state_attrs else ""
        final_src = "".join("    " + line + "\n" for stmt in final for line in ast.unparse(stmt).splitlines())
        stores = "".join(f"    gl.store({a}_ptr + offs, {a}, mask=mask)\n" for a in attrs)
        flush = "".join(f"    _emit_spikes(_fire{k}, _e{k}, offs, pid, n_groups, history, hist_cap, run_id, K, SPT, "
                        f"NUM_WARPS, _REGIONS)\n"
                        for k in range(len(sites)))
        # spikes inside delivery (D-3): the run's history slot, for receive's _emit_delivered
        ectx = "_emit_ctx(pid, n_groups, history, hist_cap, run_id, K)" if emit_delivered else "()"
        body = f'''n_ev, ok = _pull(pid, n_groups, plan, n_conns, ring, wmark_ptr, oflags, ctl, k_lo, slot_width, round_id,
                  parity, hdrs, trace, n_ev, SK, KP, MAX_CONNS, LOG_CJ, N_SLOTS, NSP, UNIFORM_SOURCES, BATCH, NUM_WARPS,
                  GROUP_SIZE, TRACE, TRACE_CAP, POOL_B, check_ptr, ARENA_CHECK, NC)
if ok:
    # the node attrs receive touches (loaded first thing): the state through delivery
    state = {state_tuple}
    if TRACE:
        state = _pin_first(state)
        n_ev = mark(trace, n_ev, 1, 0, TRACE_CAP)
    state, n_ev = _deliver(state, pid, n_groups, plan, n_conns, ring, k_lo, k_hi, offs, lane, mask, received_ptr,
                           n_received_ptr, log_ptr, log_cap, trace, n_ev, {ectx}, _receive, GROUP_SIZE, N_SLOTS,
                           STREAM, U, SPT, NUM_WARPS, MAX_CONNS, LOG, COUNT, LOG or {stage_e}, TRACE,
                           TRACE_CAP, POOL_B, check_ptr, ARENA_CHECK, {stage_w}, T_A, T_C, T_B, NC)
{unpack}
    t = t_end
    time = t_end
    # update, then send
{final_src}
{stores}
{flush}    if _REGIONS:  # (its region ordered by (e, node): a consumer's queues need a source's runs in e order)
        _sort_region(pid, n_groups, history, hist_cap, run_id, K, GROUP_SIZE, NUM_WARPS)
    n_ev = _emit_close(pid, n_groups, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K, TRACE, TRACE_CAP,
                _REGIONS)
    _run_done(ring, pid, run_id)
    _ev_close(ev, ring, pid, run_id, k_lo, k_hi, n_groups, N_SLOTS, NSP, NUM_WARPS, POOL_B, NC)
    fin = fin + 1
'''
        return f'''"""Generated by naer2/transpiler/templates/fused_template.py -- do not edit."""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from naer2.trace import mark, trace_begin
from naer2.transpiler.templates.fused import (_any_lane, _deliver, _emit_close, _emit_ctx, _emit_delivered, _emit_spikes,
                                              _region_reset, _sort_region,
                                              _pin_first, _pull, _pull_headers,
                                              _rand, _randn, _run_done, _run_entry, _status_slot, _run_gate,
                                              _run_status, _gate_skip, _gate_carry, _ev_close, _run_from)


# the history's layout (naer2/history.py): a region per program -- this population's spikes need ordering
_REGIONS = gl.constexpr({regions})

{recv_src}

@gluon.jit
def fused_run(
    pid, n_groups,
    {args},
):
    """Program pid of the population's n_groups (its group): the whole run -- its own kernel's (fused_kernel), or
    one population's share of a fused round's (naer2/fusion.py)."""
    LT: gl.constexpr = gl.BlockedLayout([SPT, U], [32, 1], [NUM_WARPS, 1], [1, 0])
    LN: gl.constexpr = gl.SliceLayout(1, LT)
    n_ev = gl.to_tensor(0)
    trace = trace_ptr
    if TRACE:
        trace = trace_begin(trace_ptr, pid, TRACE_CAP)
        n_ev = mark(trace, n_ev, 0, 0, TRACE_CAP)
    # the pull's first reads -- plan -> history -> run headers -- issued before anything else (_pull_headers)
    hdrs = _pull_headers(plan, n_conns, SK, KP, MAX_CONNS, NUM_WARPS, UNIFORM_SOURCES and BATCH)
    lane = gl.arange(0, GROUP_SIZE, layout=LN)
    offs = pid * GROUP_SIZE + lane
    mask = offs < n_neurons
{early}
    # a run stopped (a ring too small: the host reruns what started after), or this group done already (the
    # rerun): nothing to do -- see _run_gate, _pull
    run_id, round_id, k_lo, k_hi, t_start, t_end, gated = _run_entry(sched, sched_i)
    parity = run_id % 2
    status_run = _status_slot(status, sched, sched_i, run_id, n_status, n_groups)  # this run's status slot
    if gated != 0:  # a gated run delivers from where the population's last run left off (event state) -- or
        # later, where its ring holds nothing before (an idle event-driven population: from its run, _run_from)
        k_lo = _run_from(ev, parity, pid, n_groups, gl.load(ev + parity * (2 + n_groups)), k_hi, t_start, slot_width)
    skip, t_start = _gate_skip(gated, plan, n_conns, ev, run_id, round_id, t_start, t_end, slot_width, n_groups,
                               SK, KP, MAX_CONNS, NUM_WARPS)
    fin = gl.to_tensor(0)
    if _run_gate(ring, ctl, pid, run_id):
        if _REGIONS:  # (its region's count, before it emits: history.py)
            _region_reset(pid, n_groups, history, hist_cap, run_id, K)
        if skip:  # gated, nothing due: an empty run (published, done), everything carried over
            _gate_carry(plan, n_conns, ev, pid, run_id, n_groups, MAX_CONNS, NUM_WARPS)
            n_ev = _emit_close(pid, n_groups, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K, TRACE, TRACE_CAP,
                _REGIONS)
            _run_done(ring, pid, run_id)
            fin = fin + 1
        else:
{textwrap.indent(body, "            ")}
    _run_status(status_run, ring, pid, run_id, fin > 0, check_ptr, ARENA_CHECK)


@gluon.jit(do_not_specialize={list(DO_NOT_SPECIALIZE)!r})
def fused_kernel(
    {args},
):
    fused_run(gl.program_id(0), gl.num_programs(0), {names})
'''


def _time_type():
    from ...receives import time
    return time


def _uses_time(hook) -> bool:
    """Whether a hook body reads its time parameter (not just declares it)."""
    name = hook.parameter_type_to_name.get(_time_type())
    return name is not None and any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
                                    for stmt in hook.ast.body for n in ast.walk(stmt))


_MODULE_DIR: pathlib.Path | None = None
_MODULE_IDS = itertools.count()


def module_path(name: str) -> pathlib.Path:
    """Where a generated module's source goes: one directory per process (removed at exit -- the JIT reads a
    kernel's source when its function is made, nothing reads the file after), a fresh name every time."""
    global _MODULE_DIR
    if _MODULE_DIR is None:
        _MODULE_DIR = pathlib.Path(tempfile.mkdtemp(prefix="naer2-"))
        atexit.register(shutil.rmtree, _MODULE_DIR, ignore_errors=True)
    return _MODULE_DIR / f"{name}_{next(_MODULE_IDS)}.py"


def _load(source: str, name: str):
    """The generated module, written to a file (the JIT reads kernel source from one) and imported."""
    path = module_path(name)
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_kernel
