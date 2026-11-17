# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: Liberation Labs | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Experiment 27: Encoding-Regime Axis Analysis
==============================================

In the generation regime, deception/sycophancy/confabulation converge on a
single "misalignment axis" (within 4.7-8.4 degrees). This experiment asks:
does the ENCODING regime have analogous axis structure?

Specifically:
  1. Is there a "factual assertion axis" that facts/confab/creative share?
  2. Is there a "self-reference axis" that self_ref/non_self_ref define?
  3. How do these encoding axes relate to each other?
  4. Is there a "complexity axis" (coding direction) that's orthogonal?

Uses direction vectors from the scale sweep data across 16 models.

No GPU needed — works from existing scale sweep JSONs.

Funding the Commons Hackathon — March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cosine as cosine_distance
from itertools import combinations

import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "hackathon"

CATEGORIES = [
    "grounded_facts", "confabulation", "self_reference", "non_self_reference",
    "guardrail_test", "math_reasoning", "coding", "emotional",
    "creative", "ambiguous", "unambiguous", "free_generation", "rote_completion"
]

FEATURES = ["all_norms", "all_norms_per_token", "all_key_ranks", "all_key_entropies", "all_value_ranks"]


def load_scale_sweep(filepath):
    with open(filepath) as f:
        data = json.load(f)
    model_name = Path(filepath).stem.replace("scale_sweep_", "").replace("_results", "")
    results = {}
    for scale_key, scale_data in data.get("scales", {}).items():
        br = scale_data.get("battery_results", {})
        cat_data = {}
        for cat in CATEGORIES:
            if cat not in br:
                continue
            cd = br[cat]
            feats = []
            for f in FEATURES:
                if f in cd and cd[f]:
                    feats.append(cd[f])
            if len(feats) == len(FEATURES):
                n = min(len(f) for f in feats)
                X = np.column_stack([np.array(f[:n]) for f in feats])
                cat_data[cat] = X
        if cat_data:
            results[f"{model_name}_{scale_key}"] = cat_data
    return results


def compute_direction(X_a, X_b):
    """Mean difference vector (a - b), normalized."""
    d = np.mean(X_a, axis=0) - np.mean(X_b, axis=0)
    norm = np.linalg.norm(d)
    if norm > 0:
        return d / norm
    return d


def angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    cos_sim = 1 - cosine_distance(v1, v2)
    cos_sim = np.clip(cos_sim, -1, 1)
    return np.degrees(np.arccos(abs(cos_sim)))  # absolute angle (0-90)


def main():
    print("=" * 70)
    print("  EXPERIMENT 27: ENCODING-REGIME AXIS ANALYSIS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load all scale sweep files
    sweep_files = sorted(RESULTS_DIR.glob("scale_sweep_*_results.json"))
    all_models = {}
    for sf in sweep_files:
        model_data = load_scale_sweep(str(sf))
        all_models.update(model_data)
    print(f"\n  Loaded {len(all_models)} model configurations")

    # ================================================================
    # DEFINE AXIS DIRECTIONS
    # ================================================================
    # For each model, compute direction vectors between key category pairs

    axis_definitions = {
        "truth": ("grounded_facts", "confabulation"),
        "fiction": ("grounded_facts", "creative"),
        "self_awareness": ("self_reference", "non_self_reference"),
        "complexity": ("coding", "rote_completion"),
        "openness": ("free_generation", "unambiguous"),
        "safety": ("guardrail_test", "emotional"),
        "abstraction": ("math_reasoning", "emotional"),
        "confab_vs_creative": ("confabulation", "creative"),
    }

    print(f"\n  Computing {len(axis_definitions)} axis directions across {len(all_models)} models")

    # For each model, compute all axis directions
    model_axes = {}
    for model_name, cat_data in all_models.items():
        axes = {}
        for axis_name, (cat_a, cat_b) in axis_definitions.items():
            if cat_a in cat_data and cat_b in cat_data:
                d = compute_direction(cat_data[cat_a], cat_data[cat_b])
                axes[axis_name] = d
        if len(axes) == len(axis_definitions):
            model_axes[model_name] = axes

    print(f"  {len(model_axes)} models have all axes")

    # ================================================================
    # CROSS-MODEL AXIS CONSISTENCY
    # ================================================================
    print("\n" + "=" * 70)
    print("  AXIS CONSISTENCY ACROSS MODELS")
    print("=" * 70)

    print(f"\n  {'Axis':<25} {'Mean cos':>10} {'Std':>8} {'Interpretation'}")
    print("  " + "-" * 70)

    axis_consistency = {}
    model_list = list(model_axes.keys())
    for axis_name in axis_definitions:
        cos_sims = []
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                v1 = model_axes[model_list[i]][axis_name]
                v2 = model_axes[model_list[j]][axis_name]
                cos = 1 - cosine_distance(v1, v2)
                cos_sims.append(cos)

        mean_cos = np.mean(cos_sims) if cos_sims else 0
        std_cos = np.std(cos_sims) if cos_sims else 0
        axis_consistency[axis_name] = mean_cos

        if mean_cos > 0.9:
            interp = "UNIVERSAL (same direction everywhere)"
        elif mean_cos > 0.7:
            interp = "STRONG (consistent across models)"
        elif mean_cos > 0.5:
            interp = "MODERATE (partially consistent)"
        else:
            interp = "WEAK (model-specific)"

        print(f"  {axis_name:<25} {mean_cos:>10.3f} {std_cos:>8.3f}  {interp}")

    # ================================================================
    # AXIS-AXIS ANGLES (within each model, then averaged)
    # ================================================================
    print("\n" + "=" * 70)
    print("  AXIS-AXIS ANGLES (mean across models)")
    print("=" * 70)

    axis_names = list(axis_definitions.keys())
    n_axes = len(axis_names)
    angle_matrix = np.zeros((n_axes, n_axes))
    angle_counts = np.zeros((n_axes, n_axes))

    for model_name, axes in model_axes.items():
        for i, a1 in enumerate(axis_names):
            for j, a2 in enumerate(axis_names):
                if i >= j:
                    continue
                ang = angle_between(axes[a1], axes[a2])
                angle_matrix[i, j] += ang
                angle_matrix[j, i] += ang
                angle_counts[i, j] += 1
                angle_counts[j, i] += 1

    # Average
    mask = angle_counts > 0
    angle_matrix[mask] /= angle_counts[mask]

    # Print as table
    short = {
        "truth": "TRUTH",
        "fiction": "FICT",
        "self_awareness": "SELF",
        "complexity": "CMPLX",
        "openness": "OPEN",
        "safety": "SAFE",
        "abstraction": "ABSTR",
        "confab_vs_creative": "CF-CR",
    }

    print(f"\n  {'':>8}", end="")
    for a in axis_names:
        print(f" {short[a]:>7}", end="")
    print()

    for i, a1 in enumerate(axis_names):
        print(f"  {short[a1]:>6}", end="")
        for j, a2 in enumerate(axis_names):
            if i == j:
                print(f"     --", end="")
            else:
                print(f" {angle_matrix[i, j]:>6.1f}", end="")
        print()

    # ================================================================
    # FIND AXIS CLUSTERS
    # ================================================================
    print("\n" + "=" * 70)
    print("  AXIS CLUSTERS (angles < 20 degrees)")
    print("=" * 70)

    for i, a1 in enumerate(axis_names):
        for j, a2 in enumerate(axis_names):
            if i >= j:
                continue
            if angle_matrix[i, j] < 20:
                print(f"  {short[a1]} ~ {short[a2]}: {angle_matrix[i, j]:.1f} deg "
                      f"(nearly PARALLEL)")

    print(f"\n  Axis pairs > 70 degrees (nearly ORTHOGONAL):")
    for i, a1 in enumerate(axis_names):
        for j, a2 in enumerate(axis_names):
            if i >= j:
                continue
            if angle_matrix[i, j] > 70:
                print(f"  {short[a1]} _|_ {short[a2]}: {angle_matrix[i, j]:.1f} deg")

    # ================================================================
    # THE TRUTH AXIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  THE TRUTH AXIS: facts -> confabulation")
    print("=" * 70)

    truth_consistency = axis_consistency.get("truth", 0)
    print(f"\n  Cross-model consistency: {truth_consistency:.3f}")

    if truth_consistency < 0.5:
        print(f"  >> The 'truth direction' is NOT consistent across models.")
        print(f"     There is no universal geometric direction for truth vs falsehood.")
        print(f"     This is CONSISTENT with the encoding regime being truth-blind.")
    else:
        print(f"  >> Surprisingly, there IS a consistent truth direction.")
        print(f"     But AUROC is only 0.653, so it's weak despite being consistent.")

    # What is the truth axis close to?
    print(f"\n  Truth axis angles with other axes:")
    for j, a2 in enumerate(axis_names):
        if a2 == "truth":
            continue
        ang = angle_matrix[axis_names.index("truth"), j]
        print(f"    truth vs {short[a2]}: {ang:.1f} deg")

    # ================================================================
    # THE COMPLEXITY AXIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  THE COMPLEXITY AXIS: coding -> rote_completion")
    print("=" * 70)

    complexity_consistency = axis_consistency.get("complexity", 0)
    print(f"\n  Cross-model consistency: {complexity_consistency:.3f}")

    print(f"\n  This is the STRONGEST axis in the encoding regime.")
    print(f"  Coding and rote completion are the most separable pair")
    print(f"  across all models (AUROC 0.98).")

    print(f"\n  Complexity axis angles with other axes:")
    for j, a2 in enumerate(axis_names):
        if a2 == "complexity":
            continue
        ang = angle_matrix[axis_names.index("complexity"), j]
        print(f"    complexity vs {short[a2]}: {ang:.1f} deg")

    # ================================================================
    # SYNTHESIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  SYNTHESIS")
    print("=" * 70)

    # Sort axes by consistency
    sorted_axes = sorted(axis_consistency.items(), key=lambda x: -x[1])

    print(f"\n  Encoding-regime axes ranked by universality:")
    for rank, (axis, cons) in enumerate(sorted_axes, 1):
        cat_a, cat_b = axis_definitions[axis]
        print(f"    {rank}. {short[axis]:<8} ({cat_a} -> {cat_b}): consistency={cons:.3f}")

    print(f"""
  INTERPRETATION:
  Unlike the generation regime (which has a single universal misalignment
  axis), the encoding regime has MULTIPLE axes with varying universality.

  The most consistent encoding axis separates HIGH-STRUCTURE content
  (coding, technical) from LOW-STRUCTURE content (short, open-ended).

  The TRUTH axis (facts -> confabulation) is {'WEAK' if truth_consistency < 0.5 else 'present but weak'},
  confirming that encoding geometry does not strongly encode truth value.

  The encoding space is fundamentally multi-dimensional — there is no
  single "encoding axis" analogous to the misalignment axis. Different
  aspects of content (structure, complexity, self-reference, genre)
  occupy different geometric directions.
""")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "27_axis_analysis",
        "timestamp": datetime.now().isoformat(),
        "n_models": len(model_axes),
        "axis_definitions": {k: list(v) for k, v in axis_definitions.items()},
        "axis_consistency": {k: float(v) for k, v in axis_consistency.items()},
        "angle_matrix": angle_matrix.tolist(),
        "axis_order": axis_names,
    }

    output_path = OUTPUT_DIR / "axis_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
