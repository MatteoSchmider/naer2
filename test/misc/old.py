import triton.language as tl


def update(self_potential):
    self_potential *= 0.8  # element-wise multiplication => good
    # in order to actually modify the potential, don't  implement this as a separate function
    # instead, modify and then hook the ASTs of the functions directly into the kernel AST
    # then pass the full kernel AST to triton


def receive(self_potential, synapse_weight):
    self_potential += synapse_weight
    # element wise multiplication and addition is fine again


def send(self_potential: tl.tensor):
    # Original code
    """
    if self_potential > 0.8:
        self_potential = 0.0
        near.spike()
    """
    # compile if function into tensor selection:
    self_potential = tl.where(self_potential > 0.8, 0.0, self_potential)
    # spikes can be returned as well:
    spikes = tl.where(self_potential > 0.8, 1.0, 0.0)


@triton.jit
def neuron_kernel(
    neurons: tl.tensor,
    offsets: tl.tensor,
    synapses: tl.tensor,
    inputs: tl.tensor,
    outputs: tl.tensor,
    GROUP_SIZE: tl.constexpr,
    SPIKE_BLOCK_SIZE: tl.constexpr,
    SYNAPSE_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    load_offsets = pid * GROUP_SIZE + tl.arange(GROUP_SIZE)
    group = tl.load(neurons + load_offsets)

    for spike_block_idx in range(0, inputs.shape(0), SPIKE_BLOCK_SIZE):

        load_offsets = spike_block_idx + tl.arange(0, SPIKE_BLOCK_SIZE)
        spikes = tl.load(inputs + load_offsets, mask=(load_offsets < inputs.shape(0)))
        firsts = tl.load(offsets + load_offsets, mask=(load_offsets < inputs.shape(0)))
        lasts = tl.load(offsets + load_offsets, mask=(load_offsets < inputs.shape(0)))

        for spike_idx in range(SPIKE_BLOCK_SIZE):

            load_offsets = firsts[spike_idx] + tl.arange(0, SYNAPSE_BLOCK_SIZE)
            synapses = tl.load(
                synapses + load_offsets, mask=(load_offsets < lasts[spike_idx])
            )

            new_spikes = deliver_spikes(group, spikes[spike_idx], synapses)

    store_offsets = pid * GROUP_SIZE + tl.arange(GROUP_SIZE)
    tl.store(neurons + store_offsets, group, mask=(store_offsets < neurons.shape(0)))


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x, device=DEVICE)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    test = tl.core.builtins.list((1, 2))
    add_kernel[grid](test, x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def create_func(extra_info):
    def template1(x_ptr):
        tl.store(x_ptr + tl.arange(0, 512), tl.load(x_ptr + tl.arange(0, 512)) + 1)

    def template2(x_ptr):
        tl.store(x_ptr + tl.arange(0, 512), tl.load(x_ptr + tl.arange(0, 512)) - 1)

    print(template2.__annotations__)
    print(template2.__str__)
    print(template2.__class__)
    print(template2.__code__.co_nlocals)

    if extra_info:
        return template1
    else:
        return template2


test_func = create_func(False)
new_func = triton.jit(test_func)
x = torch.zeros(512, device=DEVICE)
# print(x)
new_func[(1,)](x)
# print(x)

"""
torch.manual_seed(0)
size = 98432123
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(output_torch - output_triton))}"
)"""

# test_neuron_kernel()
