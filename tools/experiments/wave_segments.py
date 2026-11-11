"""What splitting each group's segments per wave would cost, in PD14
numbers: a group of G neurons walked by G / 32 waves, each wave walking
only the links into its own 32 neurons, needs segments (source, 32-neuron
subgroup, bucket) -- exactly the segments TopologyNew builds at group size
32 (the ring cell stays per G-group). Compared against plain G-groups:
segment metadata memory, ring entries per round (firing-weighted), and
entries per update program per round (what the gather and chunking pay).

    python tools/wave_segments.py [G]
"""
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", ".."), os.path.join(os.path.dirname(__file__), "..")]
import torch

from naer2.topology_new import TopologyNew
from pd14_setup import RATE, RES
from test.networks.PD14.net import build, description

G = int(sys.argv[1]) if len(sys.argv) > 1 else 256
names = dict(zip(description["populations"]["size"], description["populations"]["name"]))
net = build()
cortical = {p: names[len(p.shape)] for p in net.populations if len(p.shape) in names}
rows = []
tot = dict(links=0, seg_g=0, seg_w=0, ent_g=0.0, ent_w=0.0, links_round=0.0)
per_target = {}
for p, name in cortical.items():
    for c in p.incoming:
        if c._source not in cortical:
            continue
        src = cortical[c._source]
        kw = dict(delay=c.fabric.delay, resolution=RES)
        tg = TopologyNew(c.fabric.topology, len(p.shape), G, "cuda", **kw)
        tw = TopologyNew(c.fabric.topology, len(p.shape), 32, "cuda", **kw)
        spikes = len(c._source.shape) * RATE[src] * RES * 1e-3  # spikes of the source per 0.1 ms round
        ent_g = spikes * tg.n_segments / max(tg.n_sources, 1)  # ring entries per round
        ent_w = spikes * tw.n_segments / max(tw.n_sources, 1)
        tot["links"] += tg.n_links
        tot["seg_g"] += tg.n_segments
        tot["seg_w"] += tw.n_segments
        tot["ent_g"] += ent_g
        tot["ent_w"] += ent_w
        tot["links_round"] += spikes * tg.n_links / max(tg.n_sources, 1)
        t = per_target.setdefault(name, dict(ent_g=0.0, ent_w=0.0, links=0.0, groups=tg.n_groups))
        t["ent_g"] += ent_g
        t["ent_w"] += ent_w
        t["links"] += spikes * tg.n_links / max(tg.n_sources, 1)
        del tg, tw
        torch.cuda.empty_cache()

# memory: per segment seg_start + seg_gb (2 words); links: int16 target + float32 weight
MB = 1 / 2**20
seg_mem_g, seg_mem_w = tot["seg_g"] * 8 * MB, tot["seg_w"] * 8 * MB
link_mem = tot["links"] * 6 * MB
print(f"PD14 cortical connections, G={G} vs per-wave segments (32-neuron subgroups)")
print(f"  links {tot['links'] / 1e6:.0f}M ({link_mem:.0f} MB targets + weights, unchanged)")
print(f"  segments: G-groups {tot['seg_g'] / 1e6:.1f}M ({seg_mem_g:.0f} MB) | per wave {tot['seg_w'] / 1e6:.1f}M "
      f"({seg_mem_w:.0f} MB): x{tot['seg_w'] / tot['seg_g']:.2f}, +{seg_mem_w - seg_mem_g:.0f} MB "
      f"(+{(seg_mem_w - seg_mem_g) / (link_mem + seg_mem_g) * 100:.0f}% of the topology)")
print(f"  locality (links per segment): {tot['links'] / tot['seg_g']:.2f} -> {tot['links'] / tot['seg_w']:.2f}")
print(f"  per round: {tot['links_round']:,.0f} links; ring entries {tot['ent_g']:,.0f} -> {tot['ent_w']:,.0f} "
      f"(x{tot['ent_w'] / tot['ent_g']:.2f}: intake appends and records, delivery headers)")
print(f"  per update program per round (entries -> entries, links; chunks of 64):")
for name, t in per_target.items():
    eg, ew = t["ent_g"] / t["groups"], t["ent_w"] / t["groups"]
    lk = t["links"] / t["groups"]
    print(f"    {name:5s} {t['groups']:4d} programs: entries {eg:6.1f} -> {ew:6.1f} ({-(-eg // 64):.0f} -> "
          f"{-(-ew // 64):.0f} chunks), links {lk:6.1f}; walk per wave: {lk:.0f} links -> ~{lk / (G // 32):.0f}")
