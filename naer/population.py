from .node_type import node_types
from .connection import Connection, ConnectionPattern

import ast
import inspect
from textwrap import dedent
import torch
import triton
import triton.language as tl


# @triton.jit
def kernel_template(
    offsets: tl.tensor,  # shape: (num_spikes, 2), offsets[spike_idx, :] = (start index of connections, number of connections)
    inputs: tl.tensor,  # shape: (num_spikes, 3), inputs[spike_idx, :] = (from_population, from_address, time)
    outputs: tl.tensor,  # same as inputs, but from_address is local to the group
    GROUP_SIZE: tl.constexpr,  # how many nodes are in one group
    MESSAGE_BLOCK_SIZE: tl.constexpr,  # how many messages are we batching over
    CONNECTION_BLOCK_SIZE: tl.constexpr,
):
    # global program and SPMD thread ids
    pid = tl.program_id(0)
    tids = pid * GROUP_SIZE + tl.arange(GROUP_SIZE)

    dst.load_hook()
    dst.update_hook()
    dst.send_hook()

    # if inputs layout changes, change here accordingly
    for message_block_idx in range(0, inputs.shape(0), MESSAGE_BLOCK_SIZE):

        # calculate offsets and load inputs and connection offsets
        # TODO: new name for connection offsets, maybe adjacency blocks?
        # TODO: look up if the array ptr accesses here are in line with the arguments' shapes
        load_offsets = message_block_idx + tl.arange(0, CONNECTION_BLOCK_SIZE)
        messages = tl.load(inputs + load_offsets, mask=(load_offsets < inputs.shape(0)))
        connection_firsts = tl.load(
            offsets + load_offsets, mask=(load_offsets < inputs.shape(0))
        )
        connection_sizes = tl.load(
            offsets + load_offsets, mask=(load_offsets < inputs.shape(0))
        )

        # go through messages serially
        for message_idx in range(MESSAGE_BLOCK_SIZE):

            # deliver each message in parallel as much as possible
            load_offsets = connection_firsts[message_idx] + tl.arange(
                0, connection_sizes[message_idx]
            )
            # TODO: load connections
            # note: have all connection parameters from all incoming populations in arguments
            # also have all node parameters from all incoming populations in arguments
            # determine from which incoming population we are receiving
            # execute connection code and receive hook after loading just those parameters
            dst.receive_hook()
            dst.send_hook()

    # TODO: figure out storage scheme
    # store_offsets = pid * GROUP_SIZE + tl.arange(GROUP_SIZE)
    # tl.store(neurons + store_offsets, group, mask=(store_offsets < neurons.shape(0)))


class Population:
    def __init__(self, instance, amount: int):
        if instance.__class__ not in node_types.keys():
            raise ValueError("node needs to be decorated by the @naer.node decorator")
        self.node = node_types[instance.__class__]
        self.node.compile(self.instance)
        self.instance = instance
        self.amount = int(amount)
        self.connections = []
        self.allocate()
        for name, value in self.attributes.items():
            print(name, value)

    def allocate(self):
        self.attributes = {}
        for name, value in self.node.dynamic_attrs.items():
            self.attributes[name] = torch.full(
                size=(self.amount,), fill_value=value, device="cuda"
            )

    def add_connection(
        self, source: "Population", connection: Connection, pattern: ConnectionPattern
    ):
        if not isinstance(source, Population):
            raise ValueError("source is not a Population")
        if not isinstance(connection, Connection):
            raise ValueError("connection is not a Connection")
        if not isinstance(pattern, ConnectionPattern):
            raise ValueError("pattern is not a ConnectionPattern")
        self.connections.append((source, connection, pattern))

    def compile(self):
        # fill kernel with code from our node, other nodes and populations based on our instance!
        # step 1: get the kernel ast
        template = ast.parse(dedent(inspect.getsource(kernel_template)))
        print(ast.dump(template, indent=4))
        # step 2: gather all attributes of this population and all connections and all connected populations
        # add them to the arguments

        # step 3: generate the connection/population/receive switch case with their hooks
        # => generate such that each hook can be identified towards a transpiler using . notation, eg. lif_cd_pop_5.hook_load()

        # step 4: let all transpilers replace all the hooks they can, or delete the hook if no code found

        # step 5:
