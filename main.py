import logging

logging.basicConfig(
    format="{%(filename)s:%(lineno)d} %(levelname)s - %(message)s", level=logging.DEBUG
)

from test.all_to_all import all_to_all
from test.lif_cd import lif_cd
from test.lif_cd_weightless import lif_cd_weightless
from test.lif_ed import lif_ed
from test.static_synapse import static_synapse

# to let 780M (gfx1103) be used, fake gfx1100 with 'export HSA_OVERRIDE_GFX_VERSION=11.0.0'

pop_a = lif_cd(1.0, 0.0, 0.9) * 10
pop_b = lif_cd(1.0, 0.0, 0.9) * 5
pop_ed = lif_ed(1.0, 0.0, 0.9, 4) * 10
pop_cw = lif_cd_weightless(1.0, 0.0, 0.9) * 10

conn = pop_a >> static_synapse(0.5) @ all_to_all >> pop_b

import ast

print("=== lif_cd ===")
print(ast.unparse(pop_a.compile()))
print("=== lif_ed ===")
print(ast.unparse(pop_ed.compile()))
print("=== lif_cd_weightless ===")
print(ast.unparse(pop_cw.compile()))


"""
class AllToAll:
    def __init__(self):
        self.all_to_all = True


# ideal, full connection flow
pop_a = lif_cd(1, 2, 3) * 10
pop_b = lif_cd(1, 2, 3) * 20
conn = pop_a >> stdp_synapse_timestamp(1.0) @ AllToAll() >> pop_b
conn.source.map(to_edge={"s": "strength"}, to_target={"s": "s_strength"})
conn.target.select_port("input").map(
    from_edge={"w": "weight"}, to_edge={"s": "s_strength"}
)

pop_a.net == pop_b.net == conn.net
pop_a.simulate(
    10
)  # simulate for 10 timesteps -> find other populations and assemble into a transient network object, all new populations are added to the network once there is an instance
pop_a.net.simulate(10)

# if no mappings/ports needed, it's just:
pop_a >> stdp_synapse_timestamp(1.0) @ AllToAll() >> pop_b
# or explicitly:
conn = naer.Connection(
    source=pop_a,
    edge=stdp_synapse_timestamp(1.0),
    target=pop_b,
    pattern=AllToAll(),
)
# the operators are evaluated like this btw:
pop_a >> ((stdp_synapse_timestamp(1.0) @ AllToAll()) >> pop_b)
"""


# OBSERVATION:
# by making populations the core building blocks and generating kernels anyways, we can eliminate the need for global spike management structures!
# => connections and connection group arrays and offset arrays can all be allocated separately!
# => to avoid scattered loading of offsets for time-ordered spikes in node kernel, let population kernel also generate a spike management kernel for just its population
# ==> then we can parallelize by executing kernels in parallel, instead of globalizing and letting a single kernel parallelize over spikes
# ===> might be a little slower this way, but much more flexible
