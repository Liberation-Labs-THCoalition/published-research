"""Independent recomputation of gate-2 finding B3: sizes of the centroid shifts in coupling_test.json. CC, 2026-10-13.

For valence, arousal and random: the mean over categories of the stored shift norms at each (injection layer,
measurement layer, alpha), divided by the same-layer value at the same alpha (which is the injected norm).
"""
import json
import sys
from collections import defaultdict
from statistics import mean

d = json.load(open(sys.argv[1]))
rec = {(r["condition"], r["injection_layer"], r["measurement_layer"], r["alpha"]): r for r in d["results"]}
m = lambda x: mean(x.values())
ratio = {"content": defaultdict(dict), "user": defaultdict(dict)}
same_vs_injected = []
for (c, inj, meas, a), r in rec.items():
    if c not in ("valence", "arousal", "random"):
        continue
    base = rec[(c, inj, inj, a)]
    if meas == inj:
        if c == "valence":
            same_vs_injected.append(m(r["content_shift_magnitudes"]) / (a * d["decomposition"][str(inj)]["original_norm"]))
        continue
    ratio["content"][c][(inj, meas, a)] = m(r["content_shift_magnitudes"]) / m(base["content_shift_magnitudes"])
    ratio["user"][c][(inj, meas, a)] = m(r["user_shift_magnitudes"]) / m(base["user_shift_magnitudes"])
print(f"same-layer valence shift / (alpha * stored valence norm): {min(same_vs_injected):.4f}-{max(same_vs_injected):.4f}")
for side in ("content", "user"):
    print(f"\n{side} prompts: off-layer shift / injected norm")
    cells = sorted(ratio[side]["valence"])
    print(f"  cells: {len(cells)}; layer pairs: {len(set((i, j) for i, j, _ in cells))}")
    for c in ("valence", "arousal", "random"):
        v = list(ratio[side][c].values())
        below = sum(x < 1 for x in v)
        at45 = [x for (i, j, a), x in ratio[side][c].items() if j == 45]
        print(f"  {c:8} range {min(v):.2f}-{max(v):.2f} mean {mean(v):.2f}; below 1 in {below} of {len(v)}; read at L45: {min(at45):.2f}-{max(at45):.2f}")
    for c in ("valence", "arousal"):
        wins = sum(ratio[side][c][k] > ratio[side]["random"][k] for k in cells)
        pairs = defaultdict(list)
        for k in cells:
            pairs[k[:2]].append(ratio[side][c][k] / ratio[side]["random"][k])
        pw = sum(mean(v) > 1 for v in pairs.values())
        rr = [ratio[side][c][k] / ratio[side]["random"][k] for k in cells]
        print(f"  {c} above random in {wins} of {len(cells)} cells, {pw} of {len(pairs)} layer pairs; "
              f"ratio to random {min(rr):.2f}-{max(rr):.2f}, mean {mean(rr):.2f}")
allu = [x for c in ratio["user"] for x in ratio["user"][c].values()]
print(f"\nuser prompts, all three directions: below the injected norm in {sum(x < 1 for x in allu)} of {len(allu)} cells")
