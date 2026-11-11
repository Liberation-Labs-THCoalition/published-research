"""Reproduction of the figures marked (N) in Nexus's gate-2 ruling on B3. CC, 2026-10-13.

Same definition as coupling_shift_norms.py: the mean over categories of the stored shift norms at a cell, divided by
the same condition's shift at the injection layer at the same magnitude. Adds the shared-PC component, the six
content categories one at a time, and the change with magnitude. Reads coupling_test.json; writes nothing.
"""
import json
import sys
from collections import defaultdict
from statistics import mean

d = json.load(open(sys.argv[1]))
rec = {}
for r in d["results"]:
    c = r["condition"].split("_L")[0]                  # shared_L15 -> shared, residual_L15 -> residual
    rec[(c, r["injection_layer"], r["measurement_layer"], r["alpha"])] = r
m = lambda x: mean(x.values())
conds = ("valence", "arousal", "random", "shared", "residual")
ratio = {"content": defaultdict(dict), "user": defaultdict(dict)}
for (c, inj, meas, a), r in rec.items():
    if meas == inj:
        continue
    base = rec[(c, inj, inj, a)]
    ratio["content"][c][(inj, meas, a)] = m(r["content_shift_magnitudes"]) / m(base["content_shift_magnitudes"])
    ratio["user"][c][(inj, meas, a)] = m(r["user_shift_magnitudes"]) / m(base["user_shift_magnitudes"])
cells = sorted(ratio["content"]["valence"])
pairs = sorted(set(k[:2] for k in cells))
print(f"off-layer cells per condition: " + ", ".join(f"{c} {len(ratio['content'][c])}" for c in conds))

print("\n1. shared-PC against random, off-layer shift / injected norm")
for side in ("user", "content"):
    sh, rn = ratio[side]["shared"], ratio[side]["random"]
    common = sorted(set(sh) & set(rn))
    wins = sum(sh[k] > rn[k] for k in common)
    print(f"   {side:7}: shared mean {mean(sh[k] for k in common):.2f} vs random {mean(rn[k] for k in common):.2f}; "
          f"shared above random in {wins} of {len(common)} cells")
print("   shared, content and user, L15 read at L50 at magnitudes 1, 3, 5:",
      [(round(ratio['content']['shared'].get((15, 50, a), float('nan')), 2), round(ratio['user']['shared'].get((15, 50, a), float('nan')), 2)) for a in (1.0, 3.0, 5.0)])
for side in ("user", "content"):
    k = (35, 45, 5.0)
    print(f"   {side:7} L35 read at L45, magnitude 5: shared {ratio[side]['shared'][k]:.2f}, valence {ratio[side]['valence'][k]:.2f}, random {ratio[side]['random'][k]:.2f}")
n = d["decomposition"]
print("   injected norm, valence / shared-PC:", {L: round(n[L]["original_norm"] / n[L]["shared_norm"], 1) for L in n})

print("\n2. the content effect in each content category (valence and random; off-layer cells)")
cats = sorted(rec[("valence", 15, 15, 1.0)]["content_shift_magnitudes"])
for cat in cats:
    out = {}
    for c in ("valence", "arousal", "random"):
        out[c] = {k: rec[(c,) + k]["content_shift_magnitudes"][cat] / rec[(c, k[0], k[0], k[2])]["content_shift_magnitudes"][cat] for k in cells}
    wv = sum(out["valence"][k] > out["random"][k] for k in cells); wa = sum(out["arousal"][k] > out["random"][k] for k in cells)
    print(f"   {cat:12} valence {mean(out['valence'].values()):.2f}  arousal {mean(out['arousal'].values()):.2f}  random {mean(out['random'].values()):.2f}"
          f"   valence>random {wv}/30, arousal>random {wa}/30")

print("\n3. change of the content shift ratio from magnitude 1 to magnitude 5, by layer pair")
for c in ("valence", "arousal", "random"):
    diffs = [ratio["content"][c][p + (5.0,)] - ratio["content"][c][p + (1.0,)] for p in pairs]
    mono = sum(ratio["content"][c][p + (1.0,)] > ratio["content"][c][p + (3.0,)] > ratio["content"][c][p + (5.0,)] for p in pairs)
    print(f"   {c:8} smaller at 5 than at 1 in {sum(x < 0 for x in diffs)} of {len(pairs)} pairs (monotone through 3 in {mono}); mean change {mean(diffs):+.2f}")
for c in ("valence", "arousal", "random"):
    diffs = [ratio["user"][c][p + (5.0,)] - ratio["user"][c][p + (1.0,)] for p in pairs]
    print(f"   user {c:8} smaller at 5 than at 1 in {sum(x < 0 for x in diffs)} of {len(pairs)} pairs; mean change {mean(diffs):+.2f}")

print("\n4. counts by layer pair (pair mean over the three magnitudes), valence and arousal against random")
for side in ("content", "user"):
    for c in ("valence", "arousal", "shared"):
        pm = sum(mean(ratio[side][c][p + (a,)] for a in (1.0, 3.0, 5.0)) > mean(ratio[side]["random"][p + (a,)] for a in (1.0, 3.0, 5.0)) for p in pairs if p + (1.0,) in ratio[side][c])
        tot = sum(1 for p in pairs if p + (1.0,) in ratio[side][c])
        print(f"   {side:7} {c:8} above random at {pm} of {tot} pairs")
