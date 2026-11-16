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
emission-time tensor per site (bare: t_start, D-4); UNIFORM_EMISSION holds
when every site is bare. Not supported yet (NotImplementedError, each with
its decision): n2.spike() inside delivery (D-3: needs the emission buffer),
connections with different receive code (D-6), more than one per-link attr
(D-7), edge forward hooks (HOOK(forward)), source.x (D-10), edge backward
(D-11). Receive's time is still each record's e, not the arrival (D-1).
"""

import ast
import importlib.util
import pathlib
import tempfile

from ..analyzers.used_members import ReadMemberFinder, UsedMemberFinder
from ..context import TranspilerContext
from ..hook_transpiler import HookTranspiler
from ..transformers import GluonTransformer, MaskAliasTransformer, TargetMaskTransformer

_STMTS = lambda src: ast.parse(src).body

# the entry's fixed arguments after the node attrs' pointers (the template's phases take them)
FIXED_ARGS = ("n_neurons", "plan", "n_conns", "ring", "wmark_ptr", "oflags", "k_lo", "k_hi", "slot_width", "round_id",
              "parity", "received_ptr", "n_received_ptr", "log_ptr", "log_cap", "history", "hist_cap", "run_id",
              "t_start", "t_end", "trace_ptr")
CONSTEXPRS = ("GROUP_SIZE", "K", "SK", "KP", "UNIFORM_SOURCES", "N_SLOTS", "MAIN_CAP", "SIDE_CAP", "LOG", "STREAM",
              "U", "SPT", "NUM_WARPS", "MAX_CONNS", "LOG_CJ", "COUNT", "BATCH", "NSP", "TRACE",
              "TRACE_CAP")
DO_NOT_SPECIALIZE = ("n_neurons", "n_conns", "k_lo", "k_hi", "run_id", "round_id", "parity", "log_cap", "hist_cap")


class _SpikeSites:
    """The fused lowering of n2.spike(): each site's fire mask and emission
    time, kept as _fire<k> / _e<k> for the entry to flush after send."""

    def __init__(self, where: str):
        self.where = where  # "final" (update / send after update) or "delivery"
        self.sites: list[bool] = []  # per site: bare

    def __call__(self, mask: str, time_expr: str, bare: bool) -> list[ast.stmt]:
        if self.where == "delivery":
            raise NotImplementedError(
                "n2.spike() inside delivery (in receive, or a send that runs after every delivery) needs the "
                "program's emission buffer -- not implemented yet (TODO.md, D-3)")
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

        # ---- receive (and the per-delivery send), rooted at hit
        recv_body: list[ast.stmt] = []
        stage_e = False
        if hook is not None:
            recv_body += HookTranspiler(hook, ctx, spike_lowering=_SpikeSites("delivery"),
                                        default_spike_time="t_start").emit_for_spike_loop(
                set(pop.dynamic_attrs), fabric_map=conn._fabric_map if conn else None,
                source_map=conn._source_map if conn else None,
                edge_static_attrs=conn.fabric.constant_attrs if conn else {}, source_static_attrs={})
            if send_after_delivery:
                send_d = HookTranspiler(send, ctx, spike_lowering=_SpikeSites("delivery"),
                                        default_spike_time="t_start").emit()
                send_d = MaskAliasTransformer().transform(send_d)
                recv_body += TargetMaskTransformer(set(pop.dynamic_attrs)).transform(send_d)
            recv_body = GluonTransformer().transform(recv_body)
            stage_e = _uses_time(hook)
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

        source = self._module_source(attrs, state_attrs, later, recv_src, final, sites.sites, stage_e)
        kernel = _load(source, f"naer2_fused_{type(pop.instance).__name__}_{id(pop)}")
        return FusedKernel(source, kernel, attrs, state_attrs, edge_attr,
                           uniform_emission=all(sites.sites), stage_e=stage_e)

    @staticmethod
    def _receive_source(state_attrs, body, edge_attr, hook) -> str:
        unpack = f"    ({', '.join(state_attrs)},) = state\n" if state_attrs else ""
        pack = f"    return ({', '.join(state_attrs)},)\n" if state_attrs else "    return state\n"
        if hook is None:
            return ("@gluon.jit\ndef _receive(state, hit, time, w, srcword):\n"
                    "    \"\"\"No receive (no incoming connection carries one).\"\"\"\n    return state\n")
        time_name = hook.parameter_type_to_name.get(_time_type())
        lines = [unpack, "    is_target = hit\n"]
        if time_name and time_name != "time":
            lines.append(f"    {time_name} = time\n")
        if edge_attr is not None:
            lines.append(f"    edge_{edge_attr} = w\n")
        body_src = "".join("    " + line + "\n" for stmt in body for line in ast.unparse(stmt).splitlines())
        return ("@gluon.jit\ndef _receive(state, hit, time, w, srcword):\n"
                f"    \"\"\"The transpiled receive ({hook.function.__qualname__}), rooted at hit.\"\"\"\n"
                + "".join(lines) + body_src + pack)

    @staticmethod
    def _module_source(attrs, state_attrs, later, recv_src, final, sites, stage_e) -> str:
        args = ",\n    ".join([f"{a}_ptr" for a in attrs] + list(FIXED_ARGS)
                              + [f"{c}: gl.constexpr" for c in CONSTEXPRS])
        load_state = "".join(f"    {a} = gl.load({a}_ptr + offs, mask=mask)\n" for a in state_attrs)
        state_tuple = f"({', '.join(state_attrs)},)" if state_attrs else "()"
        unpack = f"    ({', '.join(state_attrs)},) = state\n" if state_attrs else ""
        load_later = "".join(f"    {a} = gl.load({a}_ptr + offs, mask=mask)\n" for a in later)
        final_src = "".join("    " + line + "\n" for stmt in final for line in ast.unparse(stmt).splitlines())
        stores = "".join(f"    gl.store({a}_ptr + offs, {a}, mask=mask)\n" for a in attrs)
        flush = "".join(f"    _emit_spikes(_fire{k}, _e{k}, offs, history, hist_cap, run_id, K, SPT, NUM_WARPS)\n"
                        for k in range(len(sites)))
        return f'''"""Generated by naer2/transpiler/templates/fused_template.py -- do not edit."""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from naer2.trace import mark, trace_begin
from naer2.transpiler.templates.fused import _deliver, _emit_close, _emit_spikes, _pin_first, _pull, _rand, _randn


{recv_src}

@gluon.jit(do_not_specialize={list(DO_NOT_SPECIALIZE)!r})
def fused_kernel(
    {args},
):
    LT: gl.constexpr = gl.BlockedLayout([SPT, U], [32, 1], [NUM_WARPS, 1], [1, 0])
    LN: gl.constexpr = gl.SliceLayout(1, LT)
    pid = gl.program_id(0)
    n_groups = gl.num_programs(0)
    n_ev = gl.to_tensor(0)
    trace = trace_ptr
    if TRACE:
        trace = trace_begin(trace_ptr, pid, TRACE_CAP)
        n_ev = mark(trace, n_ev, 0, 0, TRACE_CAP)
    lane = gl.arange(0, GROUP_SIZE, layout=LN)
    offs = pid * GROUP_SIZE + lane
    mask = offs < n_neurons

    n_ev = _pull(pid, n_groups, plan, n_conns, ring, wmark_ptr, oflags, k_lo, slot_width, round_id, parity, trace,
                 n_ev, SK, KP, MAX_CONNS, LOG_CJ, N_SLOTS, NSP, MAIN_CAP, SIDE_CAP, UNIFORM_SOURCES, BATCH, NUM_WARPS,
                 TRACE, TRACE_CAP)

    # the node attrs receive touches: the state through delivery
{load_state}    state = {state_tuple}
    if TRACE:
        state = _pin_first(state)
        n_ev = mark(trace, n_ev, 1, 0, TRACE_CAP)
    state, n_ev = _deliver(state, pid, n_groups, plan, n_conns, ring, k_lo, k_hi, offs, lane, mask, received_ptr,
                           n_received_ptr, log_ptr, log_cap, trace, n_ev, _receive, GROUP_SIZE, N_SLOTS, MAIN_CAP,
                           SIDE_CAP, STREAM, U, SPT, NUM_WARPS, MAX_CONNS, LOG, COUNT, LOG or {stage_e}, TRACE,
                           TRACE_CAP)
{unpack}
    # the node attrs only update / send touch
{load_later}
    t = t_end
    time = t_end
    # update, then send
{final_src}
{stores}
{flush}    n_ev = _emit_close(pid, history, hist_cap, run_id, round_id, t_end, trace, n_ev, K, TRACE, TRACE_CAP)
'''


def _time_type():
    from ...receives import time
    return time


def _uses_time(hook) -> bool:
    """Whether a hook body reads its time parameter (not just declares it)."""
    name = hook.parameter_type_to_name.get(_time_type())
    return name is not None and any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
                                    for stmt in hook.ast.body for n in ast.walk(stmt))


def _load(source: str, name: str):
    """The generated module, written to a file (the JIT reads kernel source from one) and imported."""
    path = pathlib.Path(tempfile.mkdtemp()) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_kernel
