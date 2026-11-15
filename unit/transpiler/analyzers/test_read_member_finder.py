"""ReadMemberFinder: the members a hook reads -- what decides whether send
runs after every delivery (Population.compile: only where the connection's
receive writes a node attr send reads)."""

import ast
import textwrap

from naer2.transpiler.analyzers.used_members import ReadMemberFinder


def _reads(src: str, name: str = "self") -> set[str]:
    return ReadMemberFinder(ast.parse(textwrap.dedent(src)).body[0], name).run()


def test_loads_count_plain_assignment_targets_dont():
    assert _reads("""
        def send(self, time):
            if self.potential > self.V_thresh:
                self.potential = self.V_reset
                self.t_ref_end = time + self.t_ref
    """) == {"potential", "V_thresh", "V_reset", "t_ref"}


def test_augmented_assignment_reads_its_target():
    assert _reads("""
        def receive(self, time, edge):
            self.current += edge.weight
    """) == {"current"}


def test_other_objects_and_nested_attributes():
    src = """
        def receive(self, time, edge, source):
            self.potential -= edge.w + source.src_pot * self.gain
    """
    assert _reads(src) == {"potential", "gain"}
    assert _reads(src, "edge") == {"w"}
    assert _reads(src, "source") == {"src_pot"}


def test_pd14_neuron_send_doesnt_read_what_receive_writes():
    from test.networks.PD14.neuron import PD14_neuron
    from naer2.nodes import node_types

    node = node_types[PD14_neuron]
    receive_writes = node.receive["receive"].assigned_members
    assert receive_writes == {"current"}
    assert not receive_writes & ReadMemberFinder(node.send.ast, "self").run()
