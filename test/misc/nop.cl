#include "lif_neuron.cl"
#include "nop_group.cl"
#include "spikes.cl"

#define NOP_SPIKE_BUFFER_SIZE 32

kernel void simulate_nop_neurons(
    global nop_group* nop_groups,
    global uint spike_offsets[NAER_NUM_GROUPS + 1][NAER_SPIKE_BUFFER_SIZE],
    global float* weights,
    global ushort* local_addresses,
    global spike_buffer* inputs,
    global spike_buffer* output_array,
    global uint output_sizes[NAER_NUM_GROUPS],
    uint total_nop_neurons,
    uint nop_group_capacity,
    uint type_base_address,
    uint type_base_group)
{
    local nop_group group;
    group = nop_groups[get_group_id(0)];
    uint group_base_address = nop_group_capacity * get_group_id(0);
    uint size = min(total_nop_neurons - group_base_address, nop_group_capacity);
    uint global_group_id = type_base_group + get_group_id(0);

    int in_size = (int)inputs->size;

    local uint buffer_times[NOP_SPIKE_BUFFER_SIZE];
    local uint buffer_firsts[NOP_SPIKE_BUFFER_SIZE];
    local uint buffer_lasts[NOP_SPIKE_BUFFER_SIZE];

    ushort locals[NOP_SPIKE_BUFFER_SIZE];
    float ws[NOP_SPIKE_BUFFER_SIZE];

    local uint output_counter;
    output_counter = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int spike_index = 0; spike_index < in_size; spike_index += NOP_SPIKE_BUFFER_SIZE) {
        if (((spike_index + get_local_id(0)) < in_size) && (get_local_id(0) < NOP_SPIKE_BUFFER_SIZE)) {
            buffer_times[get_local_id(0)] = inputs->times[spike_index + get_local_id(0)];
            buffer_firsts[get_local_id(0)] = spike_offsets[global_group_id][spike_index + get_local_id(0)];
            buffer_lasts[get_local_id(0)] = spike_offsets[global_group_id + 1][spike_index + get_local_id(0)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int remaining = min(max(in_size - spike_index, 0), NOP_SPIKE_BUFFER_SIZE);

        for (int buffer_offset = 0; buffer_offset < remaining; buffer_offset++) {
            uint first = buffer_firsts[buffer_offset];
            uint last = buffer_lasts[buffer_offset];
            uint index = first + get_local_id(0);
            if (index < last) {
                // load synapse
                locals[buffer_offset] = local_addresses[index];
                ws[buffer_offset] = weights[index];
            }
        }

        for (int buffer_offset = 0; buffer_offset < remaining; buffer_offset++) {
            uint spike_time = buffer_times[buffer_offset];
            uint first = buffer_firsts[buffer_offset];
            uint last = buffer_lasts[buffer_offset];
            uint index = first + get_local_id(0);
            if (index < last) {
                ushort local_address = locals[buffer_offset];
                float weight = ws[buffer_offset];

                bool has_spiked = neuron_dynamics(&(group.potential[local_address]), &(group.last_state[local_address]), spike_time, weight);

                if (has_spiked) {
                    uint offset = atomic_inc(&output_counter);
                    // write spike(s) to output buffer
                    output_array[global_group_id].times[offset] = spike_time + NAER_AXONAL_DELAY;
                    output_array[global_group_id].addresses[offset] = type_base_address + group_base_address + local_address;
                    output_array[global_group_id].types[offset] = NAER_STATIC_WEIGHT;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    output_sizes[global_group_id] = output_counter;

    nop_groups[get_group_id(0)] = group;
}
