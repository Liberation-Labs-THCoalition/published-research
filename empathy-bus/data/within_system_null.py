#!/usr/bin/env python3
"""A within-system null for Section 4.1: how special are the valence and arousal directions? Written 2026-10-13.

Section 4.1 reports that hostile - calm ("valence") and desperate - calm ("arousal"), both built from the 30
user-model emotion centroids, align with PC1 of those same centroids. This script asks what ANY difference of two
centroids does. If most pairs align with PC1, the alignment says little about valence or arousal.

Input: emotion_vectors.json from Lyra et al. (2026a): {layer: {emotion: [d=5120 floats]}}, 30 emotion centroids per
layer, 64 layers. Same file as within_system_alignment.py. It is 203 MB and is not in this repository.

Per layer, with the 30 centroids centered:
  pairs   all C(30,2) = 435 unit differences c_i - c_j:
            |cos| with PC1, and the share of centroid variance along the direction.
  planes  all C(30,3) = 4,060 planes through three centroids:
            the share of centroid variance inside the plane.
The reference pair for valence is (hostile, calm), for arousal (desperate, calm); the reference plane is the one
through hostile, desperate and calm, which is the plane the valence and arousal directions span.

"pct_below" is the percentage of ALL items of that kind (the reference included) whose value is strictly below the
reference value. Nothing is random; there is no seed.

All arithmetic is done in the coordinates of the centroids' own principal axes (Mc = U S Vt, coordinates U S). Every
centroid difference lies in that row space, so cosines and variance shares are the same as in the full 5120
dimensions. The script checks that claim against the full-dimension figures of within_system_alignment.py's formulas
at every layer and stops if they differ by more than 1e-9.

Usage: within_system_null.py path/to/emotion_vectors.json > within_system_null.json
"""
import hashlib
import itertools
import json
import os
import sys

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pct_below(values, ref):
    return round(100.0 * float((values < ref).sum()) / len(values), 2)


def summary(values):
    return {"median": round(float(np.median(values)), 4), "p95": round(float(np.percentile(values, 95)), 4),
            "max": round(float(values.max()), 4)}


def main(path):
    d = json.load(open(path))
    layers = sorted(d, key=int)
    emos = sorted(d[layers[0]])
    idx = {e: i for i, e in enumerate(emos)}
    pairs = list(itertools.combinations(range(len(emos)), 2))
    triples = list(itertools.combinations(range(len(emos)), 3))
    pi, pj = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    h, c, de = idx["hostile"], idx["calm"], idx["desperate"]
    val_k = pairs.index(tuple(sorted((h, c))))
    aro_k = pairs.index(tuple(sorted((de, c))))
    plane_k = triples.index(tuple(sorted((h, de, c))))

    out = {
        "source": "emotion_vectors.json (Lyra et al., 2026a)",
        "input_sha256": sha256(path),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "n_emotions": len(emos), "n_pairs": len(pairs), "n_planes": len(triples),
        "reference": {"valence": "hostile - calm", "arousal": "desperate - calm",
                      "plane": "through hostile, desperate, calm"},
        "pct_below": "percentage of all items (reference included) strictly below the reference value",
        "layers": {},
    }
    for L in layers:
        M = np.array([d[L][e] for e in emos], dtype=np.float64)
        Mc = M - M.mean(0)
        U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        Y = U * S                      # centroid coordinates on their own principal axes; Y[:, 0] is PC1
        tot = float((S ** 2).sum())

        D = Y[pi] - Y[pj]
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        cos_pc1 = np.abs(D[:, 0])
        share = ((Y @ D.T) ** 2).sum(0) / tot

        plane = np.empty(len(triples))
        for k, (a, b, cc) in enumerate(triples):
            Q, _ = np.linalg.qr(np.stack([Y[a] - Y[cc], Y[b] - Y[cc]]).T)
            plane[k] = ((Y @ Q) ** 2).sum() / tot

        # Check the reduced coordinates against the full 5120 dimensions (same formulas as within_system_alignment.py).
        val = M[h] - M[c]
        val /= np.linalg.norm(val)
        aro = M[de] - M[c]
        aro /= np.linalg.norm(aro)
        Qf, _ = np.linalg.qr(np.stack([val, aro]).T)
        full = [abs(val @ Vt[0]), abs(aro @ Vt[0]), ((Mc @ val) ** 2).sum() / float((Mc ** 2).sum()),
                ((Mc @ Qf) ** 2).sum() / float((Mc ** 2).sum())]
        reduced = [cos_pc1[val_k], cos_pc1[aro_k], share[val_k], plane[plane_k]]
        if max(abs(a - b) for a, b in zip(full, reduced)) > 1e-9:
            sys.exit(f"layer {L}: reduced coordinates disagree with full dimensions: {full} vs {reduced}")

        out["layers"][L] = {
            "pairs_abs_cos_pc1": summary(cos_pc1),
            "pairs_var_share": summary(share),
            "planes_var_share": summary(plane),
            "valence": {"abs_cos_pc1": round(float(cos_pc1[val_k]), 4), "pct_below": pct_below(cos_pc1, cos_pc1[val_k]),
                        "var_share": round(float(share[val_k]), 4), "var_share_pct_below": pct_below(share, share[val_k])},
            "arousal": {"abs_cos_pc1": round(float(cos_pc1[aro_k]), 4), "pct_below": pct_below(cos_pc1, cos_pc1[aro_k]),
                        "var_share": round(float(share[aro_k]), 4), "var_share_pct_below": pct_below(share, share[aro_k])},
            "plane": {"var_share": round(float(plane[plane_k]), 4), "pct_below": pct_below(plane, plane[plane_k])},
            "var_share_pc1": round(float(S[0] ** 2 / tot), 4),
        }

    def rng(get, sel):
        vals = [get(out["layers"][L]) for L in sel]
        return [min(vals), max(vals)]

    for name, sel in (("L6_to_L63", [L for L in layers if int(L) >= 6]), ("all_layers", layers)):
        if not sel:
            continue
        out["range_" + name] = {
            "pairs_abs_cos_pc1_median": rng(lambda r: r["pairs_abs_cos_pc1"]["median"], sel),
            "pairs_abs_cos_pc1_p95": rng(lambda r: r["pairs_abs_cos_pc1"]["p95"], sel),
            "valence_abs_cos_pc1": rng(lambda r: r["valence"]["abs_cos_pc1"], sel),
            "valence_pct_below": rng(lambda r: r["valence"]["pct_below"], sel),
            "arousal_abs_cos_pc1": rng(lambda r: r["arousal"]["abs_cos_pc1"], sel),
            "arousal_pct_below": rng(lambda r: r["arousal"]["pct_below"], sel),
            "valence_var_share_pct_below": rng(lambda r: r["valence"]["var_share_pct_below"], sel),
            "arousal_var_share_pct_below": rng(lambda r: r["arousal"]["var_share_pct_below"], sel),
            "plane_pct_below": rng(lambda r: r["plane"]["pct_below"], sel),
            "var_share_pc1": rng(lambda r: r["var_share_pc1"], sel),
        }
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
