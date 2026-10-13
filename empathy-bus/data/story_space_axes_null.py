"""Follow-up to story_space_axes.py: how much of PC1 and PC2 do the two ratings explain together, where does each
axis point in the rated valence-arousal plane, and what does a label-permutation null give? (CC, 2026-10-13)"""
import json
import sys
import numpy as np
sys.path.insert(0, ".")
from story_space_axes import RATINGS, spearman  # noqa: E402

ev = json.load(open(sys.argv[1]))
layers = sorted(ev, key=int)
names = sorted(ev[layers[0]])
val = np.array([RATINGS[n][0] for n in names]); aro = np.array([RATINGS[n][1] for n in names])
A = np.column_stack([np.ones(30), (val - val.mean()) / val.std(), (aro - aro.mean()) / aro.std()])
rng = np.random.default_rng(20261013)
print("L   | PC1: R2(val+aro)  angle  perm p95 |rho_val|  p(rho_val)  | PC2: R2  angle")
res = {}
for L in layers:
    X = np.array([ev[L][n] for n in names], float)
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    row = {}
    for k in (0, 1):
        s = Xc @ Vt[k]
        s = (s - s.mean()) / s.std()
        beta, *_ = np.linalg.lstsq(A, s, rcond=None)
        r2 = 1 - np.sum((s - A @ beta) ** 2) / np.sum(s ** 2)
        ang = np.degrees(np.arctan2(beta[2], beta[1])) % 180      # axis, not arrow: 0 = pure valence, 90 = pure arousal
        row[f"pc{k+1}_r2"] = float(r2); row[f"pc{k+1}_angle"] = float(ang)
    s1 = Xc @ Vt[0]
    obs = abs(spearman(s1, val))
    perm = np.array([abs(spearman(s1, rng.permutation(val))) for _ in range(2000)])
    row["perm_p95"] = float(np.percentile(perm, 95)); row["p_val"] = float((np.sum(perm >= obs) + 1) / (len(perm) + 1))
    row["obs"] = float(obs)
    res[int(L)] = row
    if int(L) in (6, 15, 17, 20, 25, 27, 35, 45, 55, 63):
        print(f"{int(L):<3} |      {row['pc1_r2']:.2f}       {row['pc1_angle']:5.1f}       {row['perm_p95']:.2f} vs {obs:.2f}      {row['p_val']:.4f}   |   {row['pc2_r2']:.2f}  {row['pc2_angle']:5.1f}")
for lo, hi in ((17, 63), (27, 63)):
    sel = [res[L] for L in res if lo <= L <= hi]
    f = lambda k: f"{min(r[k] for r in sel):.2f}-{max(r[k] for r in sel):.2f}"
    g = lambda k: f"{min(r[k] for r in sel):.0f}-{max(r[k] for r in sel):.0f}"
    print(f"L{lo}-L{hi}: PC1 R2 {f('pc1_r2')}, angle {g('pc1_angle')} deg; PC2 R2 {f('pc2_r2')}, angle {g('pc2_angle')} deg; "
          f"perm p95 {f('perm_p95')}; max p {max(r['p_val'] for r in sel):.4f}")
print("angle convention: 0 = the valence axis, 90 = the arousal axis, 135 = pleasant-calm to unpleasant-activated, 45 = pleasant-activated to unpleasant-calm")
hc = np.degrees(np.arctan2((0.70 + 0.60) / aro.std(), (-0.75 - 0.60) / val.std())) % 180
dc = np.degrees(np.arctan2((0.80 + 0.60) / aro.std(), (-0.70 - 0.60) / val.std())) % 180
print(f"for reference, in the same standardised plane: hostile-calm points at {hc:.0f} deg, desperate-calm at {dc:.0f} deg")
json.dump(res, open("story_space_axes_null.json", "w"), indent=1)
