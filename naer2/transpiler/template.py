import triton.language as tl


def population_kernel(
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
