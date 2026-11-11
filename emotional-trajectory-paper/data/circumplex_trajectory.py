"""Circumplex Trajectory — trace emotion geometry through every layer.

For each layer, computes:
  - Emotion cluster separation (between/within variance ratio)
  - Valence direction strength (hostile-calm axis)
  - Arousal direction strength (desperate-calm axis)
  - K/V projection fidelity of emotion directions
  - Per-head emotion signal strength

Output: trajectory data + summary showing where emotion appears, peaks, and reorganizes.

Usage:
  python circumplex_trajectory.py --activations sweep_activations.npz --model-dir /path/to/model
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_projection(model_dir, layer_idx, proj="k"):
    """Load W_K or W_V for a layer."""
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    files = sorted(Path(model_dir).glob("*.safetensors"))
    name = f"model.layers.{layer_idx}.self_attn.{proj}_proj.weight"
    for f in files:
        try:
            with safe_open(str(f), framework="numpy") as st:
                if name in st.keys():
                    return st.get_tensor(name)
        except TypeError:
            import torch
            with safe_open(str(f), framework="pt") as st:
                if name in st.keys():
                    return st.get_tensor(name).float().numpy()
    return None


def direction_strength(activations, categories, cat_a, cat_b):
    """Mean per-dimension Cohen's d between two category centroids.

    Returns the average absolute per-dimension effect size, which is
    interpretable on the standard scale (0.2=small, 0.5=medium, 0.8=large).
    Also returns the centroid cosine distance and the direction vector.
    """
    mask_a = categories == cat_a
    mask_b = categories == cat_b
    if mask_a.sum() < 2 or mask_b.sum() < 2:
        return 0.0, 0.0, None

    mean_a = activations[mask_a].mean(axis=0)
    mean_b = activations[mask_b].mean(axis=0)
    diff = mean_a - mean_b

    var_a = activations[mask_a].var(axis=0)
    var_b = activations[mask_b].var(axis=0)
    pooled_std_per_dim = np.sqrt((var_a + var_b) / 2)

    per_dim_d = np.abs(diff) / (pooled_std_per_dim + 1e-10)
    mean_d = float(per_dim_d.mean())

    # Cosine distance between centroids (normalized, dimension-independent)
    norm_a = np.linalg.norm(mean_a)
    norm_b = np.linalg.norm(mean_b)
    if norm_a > 1e-10 and norm_b > 1e-10:
        cosine_dist = 1.0 - float(np.dot(mean_a, mean_b) / (norm_a * norm_b))
    else:
        cosine_dist = 0.0

    direction = diff / (np.linalg.norm(diff) + 1e-10)
    return mean_d, cosine_dist, direction


def cluster_separation(activations, categories):
    """Between/within variance ratio across all non-neutral categories."""
    cats = [c for c in np.unique(categories) if c != "neutral"]
    if len(cats) < 2:
        return 0.0

    # PCA to top-2 for visualization-relevant separation
    X = activations[np.isin(categories, cats)]
    cats_filtered = categories[np.isin(categories, cats)]
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    scores = U[:, :2] * S[:2]

    centroids = {c: scores[cats_filtered == c].mean(axis=0) for c in cats}
    between = np.var([centroids[c] for c in cats], axis=0).sum()
    within = np.mean([np.var(scores[cats_filtered == c], axis=0).sum() for c in cats])

    return float(between / (within + 1e-10))


def projection_fidelity_quick(residual_dir, W_proj, actual_dir):
    """Quick cosine fidelity without null distribution (for sweep speed)."""
    if residual_dir is None or actual_dir is None:
        return 0.0
    projected = W_proj @ residual_dir
    proj_norm = np.linalg.norm(projected)
    if proj_norm < 1e-10:
        return 0.0
    return float(np.abs(np.dot(projected / proj_norm,
                               actual_dir / (np.linalg.norm(actual_dir) + 1e-10))))


def analyze_trajectory(data, model_dir):
    """Run circumplex analysis at every layer."""
    categories = data["categories"]
    n_layers = int(data["n_layers"])
    n_kv = int(data["n_kv_heads"])
    all_layers = list(range(n_layers))

    trajectory = []

    for l in all_layers:
        residual = data[f"residual_L{l}"]

        # Emotion signal strength in residual
        valence_d, valence_cos, valence_dir = direction_strength(residual, categories, "hostile", "calm")
        arousal_d, arousal_cos, arousal_dir = direction_strength(residual, categories, "desperate", "calm")
        separation = cluster_separation(residual, categories)

        layer_result = {
            "layer": l,
            "residual": {
                "valence_d": valence_d,
                "valence_cosine_dist": valence_cos,
                "arousal_d": arousal_d,
                "arousal_cosine_dist": arousal_cos,
                "separation": separation,
            },
            "heads": [],
        }

        # Per-head analysis
        W_K = load_projection(model_dir, l, "k")
        W_V = load_projection(model_dir, l, "v")

        for h in range(n_kv):
            k_act = data[f"k_L{l}_H{h}"]
            v_act = data[f"v_L{l}_H{h}"]

            k_val_d, _, k_val_dir = direction_strength(k_act, categories, "hostile", "calm")
            v_val_d, _, v_val_dir = direction_strength(v_act, categories, "hostile", "calm")
            k_aro_d, _, k_aro_dir = direction_strength(k_act, categories, "desperate", "calm")
            v_aro_d, _, v_aro_dir = direction_strength(v_act, categories, "desperate", "calm")

            # Projection fidelity
            head_dim = k_act.shape[1]
            k_fidelity = 0.0
            v_fidelity = 0.0
            if W_K is not None and valence_dir is not None:
                W_K_head = W_K[h * head_dim:(h + 1) * head_dim, :]
                W_V_head = W_V[h * head_dim:(h + 1) * head_dim, :]
                k_fidelity = projection_fidelity_quick(valence_dir, W_K_head, k_val_dir)
                v_fidelity = projection_fidelity_quick(valence_dir, W_V_head, v_val_dir)

            layer_result["heads"].append({
                "head": h,
                "k_valence_d": k_val_d,
                "v_valence_d": v_val_d,
                "k_arousal_d": k_aro_d,
                "v_arousal_d": v_aro_d,
                "k_projection_fidelity": k_fidelity,
                "v_projection_fidelity": v_fidelity,
            })

        trajectory.append(layer_result)

    return trajectory


def print_trajectory(trajectory):
    """Print the circumplex CT scan."""
    print(f"\n{'='*90}")
    print(f"CIRCUMPLEX TRAJECTORY — Emotion Through the Stack")
    print(f"{'='*90}")
    print(f"{'Layer':>5} {'Sep':>7} {'Val-d':>7} {'Aro-d':>7} │", end="")
    n_heads = len(trajectory[0]["heads"]) if trajectory else 0
    for h in range(n_heads):
        print(f" {'H'+str(h)+'Kf':>6} {'H'+str(h)+'Vf':>6}", end="")
    print()
    print(f"{'─'*5} {'─'*7} {'─'*7} {'─'*7} │" + " ──────" * (n_heads * 2))

    peak_sep = max(t["residual"]["separation"] for t in trajectory)
    peak_val = max(t["residual"]["valence_d"] for t in trajectory)

    for t in trajectory:
        r = t["residual"]
        markers = ""
        if r["separation"] == peak_sep:
            markers += " ← PEAK SEP"
        if r["valence_d"] == peak_val:
            markers += " ← PEAK VAL"

        print(f"{t['layer']:>5} {r['separation']:>7.3f} {r['valence_d']:>7.3f} {r['arousal_d']:>7.3f} │", end="")
        for h_data in t["heads"]:
            print(f" {h_data['k_projection_fidelity']:>6.3f} {h_data['v_projection_fidelity']:>6.3f}", end="")
        print(markers)

    # Find transition zones
    print(f"\n{'='*90}")
    print("TRAJECTORY ANALYSIS")
    print(f"{'='*90}")

    seps = [t["residual"]["separation"] for t in trajectory]
    vals = [t["residual"]["valence_d"] for t in trajectory]
    fids = [np.mean([h["k_projection_fidelity"] for h in t["heads"]]) for t in trajectory]

    # Onset: first layer where separation > 0.1
    onset = next((t["layer"] for t in trajectory if t["residual"]["separation"] > 0.1), None)
    # Peak: layer with max separation
    peak_layer = trajectory[np.argmax(seps)]["layer"]
    # Fidelity peak
    fid_peak = trajectory[np.argmax(fids)]["layer"] if max(fids) > 0 else None
    # Fidelity drop: first layer after peak where fidelity drops below 0.5
    fid_drop = None
    if fid_peak is not None:
        for t in trajectory:
            if t["layer"] > fid_peak:
                avg_fid = np.mean([h["k_projection_fidelity"] for h in t["heads"]])
                if avg_fid < 0.5 and avg_fid > 0:
                    fid_drop = t["layer"]
                    break

    print(f"  Emotion onset (separation > 0.1): Layer {onset}")
    print(f"  Peak separation: Layer {peak_layer} (ratio={max(seps):.3f})")
    print(f"  Peak projection fidelity: Layer {fid_peak} (cos={max(fids):.3f})")
    if fid_drop:
        print(f"  Fidelity drops below 0.5: Layer {fid_drop}")
        print(f"  → Transition zone: L{fid_peak}–L{fid_drop}")
    else:
        print(f"  Fidelity stays above 0.5 throughout")

    # K vs V divergence
    k_vals = [np.mean([h["k_valence_d"] for h in t["heads"]]) for t in trajectory]
    v_vals = [np.mean([h["v_valence_d"] for h in t["heads"]]) for t in trajectory]
    divergence = [abs(k - v) for k, v in zip(k_vals, v_vals)]
    max_div_layer = trajectory[np.argmax(divergence)]["layer"]
    print(f"  K/V max divergence: Layer {max_div_layer} (delta={max(divergence):.3f})")

    return {
        "onset": onset,
        "peak_separation_layer": peak_layer,
        "peak_fidelity_layer": fid_peak,
        "fidelity_drop_layer": fid_drop,
        "kv_divergence_layer": max_div_layer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="trajectory_results.json")
    args = parser.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    print(f"Loaded: {len(data['categories'])} prompts, {int(data['n_layers'])} layers")

    trajectory = analyze_trajectory(data, args.model_dir)
    analysis = print_trajectory(trajectory)

    outpath = Path(args.output)
    outpath.write_text(json.dumps({
        "trajectory": trajectory,
        "analysis": analysis,
    }, indent=2))
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
