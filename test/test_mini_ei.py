"""
Mini E/I compilation smoke-test: the generated fused kernels' structure.

Builds a 2-population E/I circuit (100 E + 25 I PD14 neurons, 4 bare
connections, a per-link weight on each) and generates both populations'
fused kernels (Population.fused_kernel, naer2/transpiler/templates/
fused_template.py), then asserts on the generated source:

  - the entry signature: one pointer per dynamic node attr, then the
    template's fixed arguments, then its constexprs; static attrs inlined
  - receive: the per-link edge.weight as the staged attr `w`, its update
    rooted at the delivered lanes (is_target = hit)
  - the receive state (current) loaded before delivery, every other attr
    after it
  - no send after each delivery: receive writes current, send reads
    potential, so send runs once, after update
  - bare n2.spike(): emitted at the window's start (D-4)
  - math.exp lowered, no tl.* left (the Gluon transformer)

No GPU: generating a kernel only builds its source and the gluon.jit
function; the GPU tests (test_fused_generated_gpu.py) run them.
"""

import ast

import pytest

from naer2.transpiler.templates.fused_template import CONSTEXPRS, FIXED_ARGS
from test.networks.mini_ei import build


@pytest.fixture(scope="module")
def pops():
    _, pops = build()
    return pops


@pytest.fixture(scope="module")
def kernel_E(pops):
    return pops["E"].fused_kernel()


@pytest.fixture(scope="module")
def src_E(kernel_E):
    return kernel_E.source


def _function(src, name):
    return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_populations_have_expected_incoming_connections(pops):
    E_pop, I_pop = pops["E"], pops["I"]
    assert len(E_pop.incoming) == 2
    assert len(I_pop.incoming) == 2
    for conn in (*E_pop.incoming, *I_pop.incoming):
        assert conn.fabric.topology is not None
        assert conn.fabric.topology.target_indices.shape[0] > 0


def test_I_kernel_also_generates(pops):
    k = pops["I"].fused_kernel()
    assert isinstance(ast.parse(k.source), ast.Module)
    assert k.attrs == pops["E"].fused_kernel().attrs


def test_signature_attr_pointers_then_fixed_args_then_constexprs(kernel_E, src_E):
    args = [a.arg for a in _function(src_E, "fused_kernel").args.args]
    attrs = [f"{a}_ptr" for a in kernel_E.attrs]
    assert sorted(kernel_E.attrs) == ["current", "last_time", "potential", "t_ref_end"]
    assert args == attrs + list(FIXED_ARGS) + list(CONSTEXPRS)
    assert kernel_E.consts == {}, "PD14_neuron's static attrs are inlined as literals, not constexprs"
    assert "-63.2438" in src_E  # V_rest at K_cortical 125


def test_the_kernel_s_contract(kernel_E):
    assert kernel_E.edge_attr == "weight", "receive reads edge.weight: the staged per-link attr"
    assert kernel_E.state_attrs == ["current"]
    assert kernel_E.uniform_emission and not kernel_E.stage_e


def test_receive_reads_the_staged_weight_on_the_delivered_lanes(src_E):
    recv = ast.unparse(_function(src_E, "_receive"))
    assert "edge.weight" not in recv, "AttrPrefixTransformer missed edge.weight"
    assert "is_target = hit" in recv
    assert "edge_weight = w" in recv
    assert "current = gl.where(is_target, current + edge_weight, current)" in recv
    assert "return (current,)" in recv


def test_receive_state_loaded_before_delivery_the_rest_after(src_E):
    deliver = src_E.index("state, n_ev = _deliver(")
    assert src_E.index("current = gl.load(current_ptr + offs, mask=mask)") < deliver
    for attr in ("potential", "t_ref_end", "last_time"):
        assert src_E.index(f"{attr} = gl.load({attr}_ptr + offs, mask=mask)") > deliver
    for attr in ("potential", "current", "t_ref_end", "last_time"):
        assert f"gl.store({attr}_ptr + offs, {attr}, mask=mask)" in src_E


def test_no_per_delivery_send_when_receive_writes_nothing_send_reads(src_E):
    # PD14_neuron's receive writes only current; send reads potential (and
    # time): a delivery can't change send's outcome, so send runs once,
    # after update, not after every delivery.
    recv = ast.unparse(_function(src_E, "_receive"))
    assert "potential" not in recv, "no fire check inside the delivery"
    tail = src_E[src_E.index("# update, then send"):]
    assert "_m0 = mask & (potential > -50.0)" in tail, "send's fire check after update"


def test_bare_spike_emits_at_the_window_start(src_E):
    assert "_fire0 = _m0" in src_E
    assert "_e0 = t_start + gl.zeros([GROUP_SIZE], gl.float32, layout=LN)" in src_E
    assert "_emit_spikes(_fire0, _e0, offs, history, hist_cap, run_id, K, SPT, NUM_WARPS)" in src_E
    assert src_E.count("_emit_spikes(") == 1
    assert "_emit_close(" in src_E


def test_math_exp_lowered_to_gl_exp(src_E):
    # PD14's update hook calls math.exp on tensors; MathLoweringTransformer, then the Gluon transformer
    assert "math." not in src_E
    assert "tl." not in src_E
    assert "gl.exp(" in src_E
