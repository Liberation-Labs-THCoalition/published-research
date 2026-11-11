"""Cross-system alignment for empathy-bus section 4.2 (CC, 2026-10-13).

Question: do the content-prompt valence and arousal directions (the 250-prompt hostile-calm and desperate-calm
difference vectors) line up with the user-model's own emotion space (the PC1 of its 30 emotion centroids)?

Inputs (Studio):
- ~/user_model_rerun/results/emotion_geometry_bridge/emotion_vectors.json  {layer: {emotion: [d]}}, 30 emotions
- ~/jlens_results/cc_direction_vectors_full/content_{valence,arousal}_L{15,25,35,45}.npy
- ~/jlens_results/cc_direction_vectors_full/user_model_PC1_L{..}.npy  (provenance only, NOT a sanity check: these
  stored vectors are a different basis from the 30-centroid PC1 here (|cos| 0.10-0.15) and from the coupling
  experiment's basis in coupling_test.json. They are recorded so the three "user-model spaces" can't be confused.)

For each layer: mean-center the 30 centroids, take PC1 by SVD, report |cos| of each content direction with PC1 and
with PC1-PC3, and the share of centroid variance along each content direction, beside a null of 2000 seeded random
unit directions scored the same way. Writes JSON with input and script sha256s.

Reading it (two-way Agni, 2026-10-13): at L15 the cosines fall within the random range, though the variance shares
exceed the random maximum; at L25-L45 both exceed every random direction. Random directions are isotropic and this
model's activations are not, and no non-emotional contrast is scored here, so the excess is an upper bound on shared
emotional structure, not evidence of it.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
EV = HOME / "user_model_rerun/results/emotion_geometry_bridge/emotion_vectors.json"
CD = HOME / "jlens_results/cc_direction_vectors_full"
LAYERS = (15, 25, 35, 45)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    """Paths are recorded relative to the home directory, so no account name ends up in a public file."""
    return "~/" + str(Path(p).relative_to(HOME))


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def main(out_path: str) -> None:
    ev = json.loads(EV.read_text())
    inputs = {rel(EV): sha(EV)}
    rows = []
    for L in LAYERS:
        cents = ev[str(L)]
        names = sorted(cents)
        X = np.array([cents[n] for n in names], dtype=np.float64)
        Xc = X - X.mean(axis=0, keepdims=True)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        pc = Vt[:3]
        total_var = float((S ** 2).sum())
        row = {"layer": L, "n_emotions": len(names), "pc1_variance_share": float(S[0] ** 2 / total_var)}
        ref = CD / f"user_model_PC1_L{L}.npy"
        if ref.exists():
            inputs[rel(ref)] = sha(ref)
            row["abs_cos_with_stored_user_model_PC1"] = float(abs(unit(np.load(ref)) @ pc[0]))
        for kind in ("valence", "arousal"):
            p = CD / f"content_{kind}_L{L}.npy"
            inputs[rel(p)] = sha(p)
            u = unit(np.load(p))
            proj = Xc @ u
            row[f"content_{kind}"] = {
                "abs_cos_PC1": float(abs(u @ pc[0])),
                "abs_cos_PC2": float(abs(u @ pc[1])),
                "abs_cos_PC3": float(abs(u @ pc[2])),
                "centroid_variance_share_along_direction": float((proj ** 2).sum() / total_var),
            }
        # Null baseline: isotropic random unit directions (seeded), scored exactly like the content directions.
        rng = np.random.default_rng([20261013, L])
        R = rng.standard_normal((2000, Xc.shape[1]))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        cos_null = np.abs(R @ pc[0])
        share_null = ((Xc @ R.T) ** 2).sum(axis=0) / total_var
        row["null_random_direction"] = {
            "n": 2000, "seed": [20261013, L],
            "abs_cos_PC1_mean": float(cos_null.mean()), "abs_cos_PC1_p95": float(np.quantile(cos_null, 0.95)),
            "abs_cos_PC1_max": float(cos_null.max()),
            "variance_share_mean": float(share_null.mean()), "variance_share_p95": float(np.quantile(share_null, 0.95)),
            "variance_share_max": float(share_null.max()),
        }
        rows.append(row)
    out = {
        "what": "content-prompt valence/arousal directions vs the user-model's 30-emotion centroid space",
        "script_sha256": sha(Path(__file__)),
        "inputs_sha256": inputs,
        "numpy": np.__version__,
        "layers": rows,
        "range": {
            k: [min(r[f"content_{k}"]["abs_cos_PC1"] for r in rows), max(r[f"content_{k}"]["abs_cos_PC1"] for r in rows)]
            for k in ("valence", "arousal")
        },
    }
    Path(out_path).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["range"]))
    for r in rows:
        n = r["null_random_direction"]
        print(r["layer"], "valence %.3f arousal %.3f | null p95 %.3f max %.3f" % (
            r["content_valence"]["abs_cos_PC1"], r["content_arousal"]["abs_cos_PC1"],
            n["abs_cos_PC1_p95"], n["abs_cos_PC1_max"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cross_system_alignment.json")
