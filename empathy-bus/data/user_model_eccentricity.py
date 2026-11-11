#!/usr/bin/env python3
"""User-model eccentricity per layer, for the timecourse statement in Section 4.1 (review finding A8). CC, 2026-10-13.

The withdrawn sentence said both systems show "identical developmental timecourses" (volatile eccentricity at L0-L16,
stable from L17). The content half has its artifact (circumplex_27b_full.json). This is the user-model half, with the
SAME definition as that file's eccentricity():
    d_val = centroid(hostile) - centroid(calm),  d_aro = centroid(desperate) - centroid(calm)
    a, b = the larger and smaller of |d_val|, |d_aro|;  eccentricity = sqrt(1 - (b / a)^2)
applied to the 30-emotion centroids of Lyra et al. (2026a) instead of to group means of content-prompt activations.

Limit: the centroids are already means, so there are no per-trial activations to permute, and this has NO permutation
null (the content file does). It is descriptive.

Usage: user_model_eccentricity.py emotion_vectors.json circumplex_27b_full.json > user_model_eccentricity.json
"""
import json
import sys

import numpy as np


def eccentricity(hostile, calm, desperate):
    d_val, d_aro = hostile - calm, desperate - calm
    a = max(np.linalg.norm(d_val), np.linalg.norm(d_aro))
    b = min(np.linalg.norm(d_val), np.linalg.norm(d_aro))
    return 0.0 if a < 1e-10 else float(np.sqrt(1 - (b / a) ** 2))


def main(ev_path, content_path):
    ev = json.load(open(ev_path))
    content = {int(r["layer"]): float(r["eccentricity"]) for r in json.load(open(content_path))["layers"]}
    rows = []
    for L in sorted(ev, key=int):
        c = ev[L]
        h, ca, de = (np.asarray(c[k], dtype=np.float64) for k in ("hostile", "calm", "desperate"))
        rows.append({"layer": int(L), "user_model_eccentricity": eccentricity(h, ca, de),
                     "content_eccentricity": content.get(int(L))})
    um = np.array([r["user_model_eccentricity"] for r in rows])
    co = np.array([r["content_eccentricity"] for r in rows], dtype=np.float64)

    def span(x, lo, hi):
        sel = x[lo:hi + 1]
        return {"min": float(sel.min()), "max": float(sel.max()),
                "mean_abs_step": float(np.mean(np.abs(np.diff(sel))))}

    out = {
        "what": "eccentricity of the 30-emotion user-model centroids per layer, same definition as the content file",
        "null": "none: the centroids are means, so there is nothing to permute",
        "pearson_r_user_model_vs_content_all_layers": float(np.corrcoef(um, co)[0, 1]),
        "user_model": {"L0-L16": span(um, 0, 16), "L17-L21": span(um, 17, 21), "L22-L63": span(um, 22, 63),
                       "L17-L63": span(um, 17, 63)},
        "content": {"L0-L16": span(co, 0, 16), "L17-L63": span(co, 17, 63)},
        "layers": rows,
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
