#!/usr/bin/env python3
"""Within-system alignment of the user-model emotion centroids (Section 4.1), recomputed 2026-10-12.

Input: emotion_vectors.json from Lyra et al. (2026a): {layer: {emotion: [d=5120 floats]}}, 30 emotion centroids per
layer, 64 layers, Qwen3.5-27B distilled checkpoint. It is 203 MB and is not in this repository.

Every direction here is built FROM THE USER-MODEL CENTROIDS THEMSELVES:
  valence = centroid(hostile) - centroid(calm);  arousal = centroid(desperate) - centroid(calm)
and compared with the first principal component of the same 30 centered centroids. So these numbers describe structure
inside the user-model space. They are not a comparison with the content-prompt circumplex (see coupling_test.json).

Per layer: |cos(valence, PC1)|, |cos(arousal, PC1)|, cos(valence, arousal), the share of centroid variance captured by
each direction alone, their SUM (which double-counts, because the two directions are not orthogonal), the share
captured by the PLANE they span (the correct joint figure), and PC1's own share.

Usage: within_system_alignment.py path/to/emotion_vectors.json > within_system_alignment.json
"""
import json
import sys

import numpy as np


def unit(v):
    return v / np.linalg.norm(v)


def main(path):
    d = json.load(open(path))
    layers = sorted(d, key=int)
    emos = sorted(d[layers[0]])
    out = {"source": "emotion_vectors.json (Lyra et al., 2026a)", "n_emotions": len(emos), "dim": None, "layers": {}}
    for L in layers:
        M = np.array([d[L][e] for e in emos], dtype=np.float64)
        out["dim"] = int(M.shape[1])
        Mc = M - M.mean(0)
        _, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        pc1, tot = Vt[0], float((Mc ** 2).sum())
        val = unit(np.array(d[L]["hostile"], dtype=np.float64) - np.array(d[L]["calm"], dtype=np.float64))
        aro = unit(np.array(d[L]["desperate"], dtype=np.float64) - np.array(d[L]["calm"], dtype=np.float64))
        cv, ca = float(((Mc @ val) ** 2).sum() / tot), float(((Mc @ aro) ** 2).sum() / tot)
        Q, _ = np.linalg.qr(np.stack([val, aro]).T)
        out["layers"][L] = {
            "abs_cos_valence_pc1": round(float(abs(val @ pc1)), 4),
            "abs_cos_arousal_pc1": round(float(abs(aro @ pc1)), 4),
            "cos_valence_arousal": round(float(val @ aro), 4),
            "var_share_valence": round(cv, 4), "var_share_arousal": round(ca, 4),
            "var_share_sum_double_counted": round(cv + ca, 4),
            "var_share_plane": round(float(((Mc @ Q) ** 2).sum() / tot), 4),
            "var_share_pc1": round(float(S[0] ** 2 / tot), 4),
        }
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
