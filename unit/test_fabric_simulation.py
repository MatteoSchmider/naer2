"""Fabric.simulation(): a connection's SimulationTopology for its consumer,
and the per-link attrs living there once it's built (naer2/fabric.py;
TODO.md, runtime integration step 2). On the CPU, through the DSL."""

import pytest
import torch

from naer2.fabric import PLACEHOLDER
from test.edges import plastic_synapse
from test.networks.PD14.connection import PD14_connection
from test.networks.PD14.neuron import PD14_neuron


def _connection(edge=None, n_src=60, n_tgt=90):
    torch.manual_seed(0)
    src, dst = PD14_neuron(K_cortical=2000) * n_src, PD14_neuron(K_cortical=2000) * n_tgt
    conn = (src >> edge @ PD14_connection(0.2) >> dst) if edge is not None else (src >> PD14_connection(0.2) >> dst)
    return conn


def test_attrs_move_into_the_simulation_topology_and_read_back_canonical():
    conn = _connection()
    weight = torch.rand(conn.shape)
    conn.weight = weight
    sim = conn.fabric.simulation(90, 16, "cpu", attrs=["weight"])
    assert "weight" not in conn.fabric.state  # one home: the simulation topology
    assert list(sim.attrs) == ["weight"]
    assert torch.equal(conn.fabric.weight, weight)  # canonical order, through perm
    assert torch.equal(sim.attrs["weight"], weight[sim.perm.long()])  # simulation order inside
    assert "weight" in conn.fabric.tensor_attrs


def test_assigning_writes_through_and_checks_the_shape():
    conn = _connection()
    conn.weight = torch.rand(conn.shape)
    conn.fabric.simulation(90, 16, "cpu", attrs=["weight"])
    new = torch.rand(conn.shape)
    conn.weight = new
    assert torch.equal(conn.fabric.weight, new)
    with pytest.raises(ValueError, match="one per link"):
        conn.fabric.weight = torch.rand(3)


def test_a_scalar_edge_attr_is_broadcast_to_every_link():
    conn = _connection(plastic_synapse(0.75, 0.1))
    conn.fabric.simulation(90, 16, "cpu", attrs=["weight"])
    assert torch.all(conn.fabric.weight == 0.75) and len(conn.fabric.weight) == len(conn.fabric.topology)


def test_no_attr_gets_a_placeholder_and_the_staged_attr_comes_first():
    conn = _connection()
    sim = conn.fabric.simulation(90, 16, "cpu")
    assert list(sim.attrs) == [PLACEHOLDER] and PLACEHOLDER not in conn.fabric.tensor_attrs
    other = _connection()
    other.fabric.gain = torch.rand(other.shape)
    other.fabric.weight = torch.rand(other.shape)
    assert list(other.fabric.simulation(90, 16, "cpu", attrs=["weight", "gain"]).attrs) == ["weight", "gain"]


def test_the_delay_becomes_buckets_and_is_fixed_afterwards():
    conn = _connection()
    conn.delay = torch.rand(conn.shape) * 0.5 + 0.1
    with pytest.raises(ValueError, match="resolution"):
        conn.fabric.simulation(90, 16, "cpu")
    sim = conn.fabric.simulation(90, 16, "cpu", resolution=0.1)
    assert sim.n_buckets > 1 and sim.resolution == pytest.approx(0.1)
    with pytest.raises(ValueError, match="fixed"):
        conn.delay = 0.3
    scalar = _connection()
    scalar.delay = 0.2
    assert scalar.fabric.simulation(90, 16, "cpu").n_buckets == 1


def test_built_once_for_one_consumer():
    conn = _connection()
    conn.weight = torch.rand(conn.shape)
    sim = conn.fabric.simulation(90, 16, "cpu", attrs=["weight"])
    assert conn.fabric.simulation(90, 16, "cpu", attrs=["weight"]) is sim
    with pytest.raises(ValueError, match="already built"):
        conn.fabric.simulation(90, 32, "cpu", attrs=["weight"])
