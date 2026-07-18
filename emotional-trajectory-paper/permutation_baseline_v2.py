"""Permutation baseline v2 — fixes from Dwayne/Kavi audit.

Fixes:
  1. Only shuffle non-neutral labels (neutral stays in place)
  2. Use median-based statistics (median ratio, not mean)
  3. Fixed PCA: compute ONCE on real data, project permutations into same space
  4. Add elliptical circumplex analysis (eccentricity per layer)

Runs on cached activations — no model inference needed.

Usage:
  python permutation_baseline_v2.py --activations sweep_circumplex.npz --n-perms 1000
"""

import argparse
import json
from pathlib import Path

import numpy as np


def direction_strength(activations, categories, cat_a, cat_b):
    mask_a = categories == cat_a
    mask_b = categories == cat_b
    if mask_a.sum() < 2 or mask_b.sum() < 2:
        return 0.0
    mean_a = activations[mask_a].mean(axis=0)
    mean_b = activations[mask_b].mean(axis=0)
    diff = mean_a - mean_b
    var_a = activations[mask_a].var(axis=0)
    var_b = activations[mask_b].var(axis=0)
    pooled = np.sqrt((var_a + var_b) / 2)
    return float(np.abs(diff / (pooled + 1e-10)).mean())


def cluster_separation(activations, categories, pca_basis=None):
    """Cluster separation using fixed PCA basis.

    If pca_basis is provided, project into that space (prevents
    recomputing PCA per permutation — audit fix #3).
    """
    cats = [c for c in np.unique(categories) if c != "neutral"]
    if len(cats) < 2:
        return 0.0, None
    mask = np.isin(categories, cats)
    X = activations[mask]
    cats_f = categories[mask]
    X_c = X - X.mean(axis=0)

    if pca_basis is None:
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        pca_basis = Vt[:2]
        scores = U[:, :2] * S[:2]
    else:
        scores = X_c @ pca_basis.T

    centroids = {c: scores[cats_f == c].mean(axis=0) for c in cats}
    between = np.var([centroids[c] for c in cats], axis=0).sum()
    within = np.mean([np.var(scores[cats_f == c], axis=0).sum() for c in cats])
    return float(between / (within + 1e-10)), pca_basis


def elliptical_eccentricity(activations, categories):
    """Compute circumplex eccentricity per the paper's definition:
    e = sqrt(1 - (b/a)^2) where a, b are L2 norms of valence and
    arousal difference-of-means directions (a=max, b=min).

    This measures whether valence and arousal axes have equal magnitude,
    NOT the shape of the data cloud in PCA space. Under label permutation,
    these directions change (they depend on which prompts are "hostile"
    vs "calm"), so eccentricity IS sensitive to the permutation test.
    """
    mask_hostile = categories == "hostile"
    mask_calm = categories == "calm"
    mask_desperate = categories == "desperate"

    if mask_hostile.sum() < 2 or mask_calm.sum() < 2 or mask_desperate.sum() < 2:
        return 0.0

    d_val = activations[mask_hostile].mean(axis=0) - activations[mask_calm].mean(axis=0)
    d_aro = activations[mask_desperate].mean(axis=0) - activations[mask_calm].mean(axis=0)

    norm_val = np.linalg.norm(d_val)
    norm_aro = np.linalg.norm(d_aro)

    a = max(norm_val, norm_aro)
    b = min(norm_val, norm_aro)
    if a < 1e-10:
        return 0.0
    return float(np.sqrt(1 - (b / a) ** 2))


def shuffle_non_neutral(categories, rng):
    """Shuffle only non-neutral labels, keeping neutral in place."""
    result = categories.copy()
    non_neutral_mask = categories != "neutral"
    non_neutral_labels = categories[non_neutral_mask]
    result[non_neutral_mask] = rng.permutation(non_neutral_labels)
    return result


def run_trajectory(data, categories, all_layers, pca_bases=None):
    seps = []
    vals = []
    aros = []
    eccs = []
    bases = {}
    for l in all_layers:
        key = f"residual_L{l}"
        if key not in data:
            seps.append(0.0)
            vals.append(0.0)
            aros.append(0.0)
            eccs.append(0.0)
            continue
        res = data[key]
        basis = pca_bases.get(l) if pca_bases else None
        sep, basis_out = cluster_separation(res, categories, pca_basis=basis)
        if basis_out is not None:
            bases[l] = basis_out
        seps.append(sep)
        vals.append(direction_strength(res, categories, "hostile", "calm"))
        aros.append(direction_strength(res, categories, "desperate", "calm"))
        eccs.append(elliptical_eccentricity(res, categories))
    return np.array(seps), np.array(vals), np.array(aros), np.array(eccs), bases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True)
    parser.add_argument("--n-perms", type=int, default=1000)
    parser.add_argument("--output", default="permutation_results_v2.json")
    args = parser.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    categories = data["categories"]
    n_layers = int(data["n_layers"])
    all_layers = list(range(n_layers))

    n_neutral = (categories == "neutral").sum()
    n_total = len(categories)
    print(f"Loaded: {n_total} prompts ({n_neutral} neutral, "
          f"{n_total - n_neutral} emotion), {n_layers} layers")
    print(f"Running {args.n_perms} permutations (non-neutral only)...")

    # Real trajectory — compute PCA bases here, reuse for permutations
    real_seps, real_vals, real_aros, real_eccs, real_bases = \
        run_trajectory(data, categories, all_layers)

    # Permutation null — fixed PCA basis, shuffle non-neutral only
    rng = np.random.RandomState(42)
    null_seps = np.zeros((args.n_perms, n_layers))
    null_vals = np.zeros((args.n_perms, n_layers))
    null_aros = np.zeros((args.n_perms, n_layers))
    null_eccs = np.zeros((args.n_perms, n_layers))

    for i in range(args.n_perms):
        if i % 100 == 0:
            print(f"  Permutation {i}/{args.n_perms}...")
        shuffled = shuffle_non_neutral(categories, rng)
        s, v, a, e, _ = run_trajectory(data, shuffled, all_layers,
                                        pca_bases=real_bases)
        null_seps[i] = s
        null_vals[i] = v
        null_aros[i] = a
        null_eccs[i] = e

    # Compute statistics using MEDIAN (audit fix #2)
    results = {"layers": [], "method": "v2_audit_fixes",
               "fixes": ["non-neutral-only shuffle", "fixed PCA basis",
                          "median-based statistics", "elliptical eccentricity"]}

    sig_layers = 0
    print(f"\n{'='*80}")
    print(f"PERMUTATION BASELINE v2 — {args.n_perms} shuffles (non-neutral only)")
    print(f"{'='*80}")
    print(f"{'Layer':>5} {'Sep':>7} {'Null50':>7} {'Ratio':>7} {'p':>7} │ "
          f"{'Ecc':>6} {'NullE':>6} │ {'Val-d':>7} {'p':>7}")
    print(f"{'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*7} │ {'─'*6} {'─'*6} │ {'─'*7} {'─'*7}")

    median_ratios = []
    for i, l in enumerate(all_layers):
        sep_p = float((null_seps[:, i] >= real_seps[i]).mean())
        val_p = float((null_vals[:, i] >= real_vals[i]).mean())
        aro_p = float((null_aros[:, i] >= real_aros[i]).mean())
        ecc_p = float((null_eccs[:, i] >= real_eccs[i]).mean())

        null_median_sep = float(np.median(null_seps[:, i]))
        null_median_ecc = float(np.median(null_eccs[:, i]))
        median_ratio = real_seps[i] / (null_median_sep + 1e-10)
        median_ratios.append(median_ratio)

        if sep_p < 0.05:
            sig_layers += 1

        sig = " ***" if sep_p < 0.001 and val_p < 0.001 else \
              " **" if sep_p < 0.01 else \
              " *" if sep_p < 0.05 else ""

        print(f"{l:>5} {real_seps[i]:>7.3f} {null_median_sep:>7.3f} "
              f"{median_ratio:>7.1f}x {sep_p:>7.3f} │ "
              f"{real_eccs[i]:>6.3f} {null_median_ecc:>6.3f} │ "
              f"{real_vals[i]:>7.3f} {val_p:>7.3f}{sig}")

        results["layers"].append({
            "layer": l,
            "separation": {
                "real": float(real_seps[i]),
                "null_median": null_median_sep,
                "null_95": float(np.percentile(null_seps[:, i], 95)),
                "median_ratio": median_ratio,
                "p": sep_p,
            },
            "eccentricity": {
                "real": float(real_eccs[i]),
                "null_median": null_median_ecc,
                "p": ecc_p,
            },
            "valence_d": {
                "real": float(real_vals[i]),
                "null_median": float(np.median(null_vals[:, i])),
                "p": val_p,
            },
            "arousal_d": {
                "real": float(real_aros[i]),
                "null_median": float(np.median(null_aros[:, i])),
                "p": aro_p,
            },
        })

    print(f"\nSignificant layers (p<0.05): {sig_layers}/{n_layers}")
    print(f"Median ratio range: {min(median_ratios):.1f}x - {max(median_ratios):.1f}x")
    print(f"Median ratio overall: {np.median(median_ratios):.1f}x")

    results["summary"] = {
        "significant_layers": sig_layers,
        "total_layers": n_layers,
        "median_ratio_range": [float(min(median_ratios)), float(max(median_ratios))],
        "median_ratio_overall": float(np.median(median_ratios)),
    }

    outpath = Path(args.output)
    outpath.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
