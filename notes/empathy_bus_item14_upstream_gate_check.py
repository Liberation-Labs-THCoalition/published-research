"""Read upstream_gate.json and evaluate its pre-registered criteria literally (PREREG_upstream_gate.json).

Prints, for every direction and injection layer, the J-lens delta measured at L45: the mean over the 20 prompts, the
stored CI, and how many of the 20 per-prompt deltas are exactly zero. Then applies the four pre-registered conditions.
Read-only; computes nothing the result file does not already hold.
"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "upstream_gate.json"
d = json.load(open(path))
R = d["results"]
INJ = d["injection_layers"]
PCS = ["PC1", "PC2", "PC3", "PC4", "PC5", "NULL"]
MEAS = 45

print(f"alpha={d['alpha']} n_prompts={d['n_prompts']} injection_layers={INJ} cells={len(R)}")
print(f"\nJ-lens delta at L{MEAS}: mean [CI]  (exact zeros of 20)")
table = {}
for pc in PCS:
    row = []
    for inj in INJ:
        r = R.get(f"{pc}_{inj}_{MEAS}")
        if r is None:
            row.append(f"L{inj}: (not run)")
            continue
        zeros = sum(1 for x in r["jlens_deltas"] if x == 0)
        lo, hi = r["jlens_ci"]
        table[(pc, inj)] = (r["jlens_mean"], lo, hi, zeros, len(r["jlens_deltas"]))
        row.append(f"L{inj}: {r['jlens_mean']:+.4f} [{lo:+.4f},{hi:+.4f}] ({zeros})")
    print(f"  {pc:5} " + " | ".join(row))

print("\nPre-registered conditions, applied to the means at L45:")
for pc in PCS:
    means = {inj: table[(pc, inj)][0] for inj in INJ if (pc, inj) in table}
    early = [abs(means[i]) for i in (3, 7) if i in means]
    early_open = any(m > 0.005 for m in early)
    mid_null = (35 in means) and abs(means[35]) < 0.002
    arch = len(means) == len(INJ) and all(abs(m) < 0.002 for m in means.values())
    over = sorted(i for i, m in means.items() if abs(m) >= 0.002)
    print(f"  {pc:5} early_gate_open={early_open}  mid_null={mid_null}  "
          f"architectural_orthogonality={arch}  layers run={sorted(means)}  |mean|>=0.002 at {over}")

allz = sum(1 for r in R.values() if all(x == 0 for x in r["jlens_deltas"]))
anyz = sum(sum(1 for x in r["jlens_deltas"] if x == 0) for r in R.values())
tot = sum(len(r["jlens_deltas"]) for r in R.values())
print(f"\nAll cells: {allz} of {len(R)} have every J-lens delta exactly zero; "
      f"{anyz} of {tot} per-prompt J-lens deltas are exactly zero ({100 * anyz / tot:.1f}%).")
for pc in PCS:
    cells = [r for k, r in R.items() if r["pc"] == pc]
    z = sum(sum(1 for x in r["jlens_deltas"] if x == 0) for r in cells)
    t = sum(len(r["jlens_deltas"]) for r in cells)
    print(f"  {pc:5} {z}/{t} exact zeros ({100 * z / t:.1f}%) over {len(cells)} cells")
