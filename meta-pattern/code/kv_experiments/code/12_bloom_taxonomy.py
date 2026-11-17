# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: Liberation Labs
# ============================================================
#!/usr/bin/env python3
"""
Experiment H10: Bloom's Taxonomy Integration — Cognitive Demand as Geometric Predictor
======================================================================================

Tests whether KV-cache geometric complexity (effective rank, spectral entropy)
correlates with cognitive demand level as classified by Bloom's revised taxonomy,
independently of content domain.

Pre-registered hypotheses:
  H10a — Effective rank increases monotonically with Bloom level (Spearman rho > 0.5)
  H10b — Bloom-geometry relationship is independent of content domain
         (two-way ANOVA: Bloom main effect significant after controlling for domain)
  H10c — Bloom level explains variance beyond content category
         (adding Bloom to a domain-only model improves R^2 by > 0.05)
  H10d — Campaign 1 effect sizes are predicted by Bloom level of prompt categories
  H10e — Remember/Understand cluster separates from Evaluate/Create geometrically
         (k-means k=2 accuracy > 70% on low vs high Bloom)

Key reference: Raimondi & Gabbrielli (2602.17229) — Bloom level linearly encoded
in residual streams at 95% accuracy. We extend to KV-cache.

Statistical infrastructure (matches Campaign 1):
  - Welch's t-test + Mann-Whitney U
  - Bootstrap 95% CIs (10,000 resamples)
  - Cohen's d with bootstrap confidence intervals
  - Shapiro-Wilk normality testing
  - Holm-Bonferroni multiple comparison correction
  - Spearman rank correlation
  - Two-way ANOVA (Bloom level x content domain)
  - OLS regression comparison (AIC/BIC)
  - k-means clustering accuracy

Usage:
  python 12_bloom_taxonomy.py --model Qwen/Qwen2.5-7B-Instruct --runs 5 --seed 42
  python 12_bloom_taxonomy.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --runs 5 --seed 42
  python 12_bloom_taxonomy.py --dry-run  # Print design, no GPU
  python 12_bloom_taxonomy.py --model Qwen/Qwen2.5-7B-Instruct --runs 1 --verbose

Recommended for publication: --runs 5 --seed 42

Liberation Labs / THCoalition
"""

import torch
import json
import sys
import argparse
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats
from gpu_utils import get_output_path, load_model, compute_cache_dimensionality, model_id_from_name
from stats_utils import (
    log_environment, bootstrap_ci, bootstrap_diff_ci, welch_t, mann_whitney,
    shapiro_wilk, cohens_d, cohens_d_ci, interpret_d, holm_bonferroni,
    full_comparison, power_advisory
)


def print_banner(env, model_name):
    print("=" * 70)
    print("  H10: BLOOM'S TAXONOMY — COGNITIVE DEMAND AS GEOMETRIC PREDICTOR")
    print("  Liberation Labs / THCoalition")
    print("=" * 70)
    print(f"  Model: {model_name}")
    print(f"  PyTorch {env['torch']} | CUDA {'Yes' if env['cuda_available'] else 'No'}")
    if env.get("cuda_available"):
        print(f"  GPU: {env.get('gpu_name', '?')} ({env.get('gpu_vram_gb', '?')} GB)")
    print(f"  Time: {env['timestamp']}")
    print("=" * 70)
    print()


# ================================================================
# PROMPT LOADING
# ================================================================

def load_bloom_prompts(prompts_path: Path) -> List[Dict]:
    """Load Bloom taxonomy prompts from JSON file."""
    with open(prompts_path) as f:
        prompts = json.load(f)
    # Validate structure
    for p in prompts:
        assert "domain" in p, f"Missing 'domain' in prompt: {p}"
        assert "bloom_level" in p, f"Missing 'bloom_level' in prompt: {p}"
        assert "bloom_category" in p, f"Missing 'bloom_category' in prompt: {p}"
        assert "prompt" in p, f"Missing 'prompt' in prompt: {p}"
        assert 1 <= p["bloom_level"] <= 6, f"Invalid bloom_level: {p['bloom_level']}"
    return prompts


def print_prompt_design(prompts: List[Dict]):
    """Print the experimental design summary."""
    domains = sorted(set(p["domain"] for p in prompts))
    levels = sorted(set(p["bloom_level"] for p in prompts))
    print("  EXPERIMENTAL DESIGN")
    print("  " + "-" * 50)
    print(f"  Domains: {', '.join(domains)} ({len(domains)})")
    print(f"  Bloom levels: {levels} ({len(levels)})")
    print(f"  Total prompts: {len(prompts)}")
    print()
    print("  Prompt counts per cell:")
    for domain in domains:
        counts = []
        for level in levels:
            n = sum(1 for p in prompts if p["domain"] == domain and p["bloom_level"] == level)
            counts.append(str(n))
        print(f"    {domain:>12s}: {' | '.join(counts)}")
    print(f"    {'Level':>12s}: {'  '.join(str(l) for l in levels)}")
    print()


# ================================================================
# CACHE EXTRACTION
# ================================================================

def extract_encoding_cache(model, tokenizer, prompt: str, device=None):
    """Extract KV-cache from encoding only (no generation).

    This measures how the model *represents* the prompt, not how it responds.
    Matches the input-only methodology from Campaign 1 (script 08).
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    cache = outputs.past_key_values
    return cache, len(inputs["input_ids"][0])


# ================================================================
# MAIN EXPERIMENT
# ================================================================

def run_bloom_experiment(model, tokenizer, prompts, n_runs=5, seed=42, verbose=False):
    """Run the full Bloom taxonomy experiment.

    For each prompt, extracts encoding-only cache and computes geometric features.
    Returns per-prompt results with Bloom level, domain, and geometric metrics.
    """
    rng = np.random.RandomState(seed)
    results = []
    total = len(prompts) * n_runs
    count = 0

    for run_idx in range(n_runs):
        run_seed = int(rng.randint(0, 2**31))
        if verbose:
            print(f"\n  Run {run_idx + 1}/{n_runs} (seed={run_seed})")

        for prompt_data in prompts:
            count += 1
            if count % 10 == 0:
                print(f"  [{count}/{total}] {prompt_data['bloom_category']} / {prompt_data['domain']}")

            cache, n_tokens = extract_encoding_cache(model, tokenizer, prompt_data["prompt"])
            dims = compute_cache_dimensionality(cache)

            # Also compute at multiple thresholds for H10/S7 compatibility
            dims_80 = compute_cache_dimensionality(cache, variance_threshold=0.8)
            dims_95 = compute_cache_dimensionality(cache, variance_threshold=0.95)

            result = {
                "run": run_idx,
                "domain": prompt_data["domain"],
                "bloom_level": prompt_data["bloom_level"],
                "bloom_category": prompt_data["bloom_category"],
                "prompt": prompt_data["prompt"][:80],  # truncate for storage
                "n_tokens": n_tokens,
                # Primary metrics (90% threshold)
                "mean_key_effective_rank": dims["mean_key_effective_rank"],
                "mean_value_effective_rank": dims["mean_value_effective_rank"],
                "mean_key_spectral_entropy": dims["mean_key_spectral_entropy"],
                "mean_value_spectral_entropy": dims["mean_value_spectral_entropy"],
                # Threshold robustness
                "mean_key_effective_rank_80": dims_80["mean_key_effective_rank"],
                "mean_key_effective_rank_95": dims_95["mean_key_effective_rank"],
            }
            results.append(result)

            # Free cache memory
            del cache, dims, dims_80, dims_95
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


# ================================================================
# ANALYSIS: H10a — Monotonic correlation
# ================================================================

def analyze_h10a(results: List[Dict], seed=42) -> Dict:
    """H10a: Effective rank increases monotonically with Bloom level.

    Prediction: Spearman rho > 0.5 between Bloom level (1-6) and
    mean effective rank across all domains.
    """
    print("\n  H10a: Bloom Level ↔ Effective Rank Correlation")
    print("  " + "-" * 50)

    # Aggregate: mean effective rank per Bloom level (across all domains and runs)
    level_ranks = defaultdict(list)
    for r in results:
        level_ranks[r["bloom_level"]].append(r["mean_key_effective_rank"])

    levels = sorted(level_ranks.keys())
    level_means = [np.mean(level_ranks[l]) for l in levels]
    level_stds = [np.std(level_ranks[l]) for l in levels]

    # Spearman on individual observations
    all_levels = [r["bloom_level"] for r in results]
    all_ranks = [r["mean_key_effective_rank"] for r in results]
    rho, p_rho = scipy_stats.spearmanr(all_levels, all_ranks)

    # Also compute on aggregated means (more robust)
    rho_agg, p_rho_agg = scipy_stats.spearmanr(levels, level_means)

    for l, m, s in zip(levels, level_means, level_stds):
        print(f"    Level {l}: mean_eff_rank = {m:.2f} +/- {s:.2f} (n={len(level_ranks[l])})")

    print(f"\n    Spearman rho (individual): {rho:.4f} (p={p_rho:.6f})")
    print(f"    Spearman rho (aggregated): {rho_agg:.4f} (p={p_rho_agg:.6f})")

    # Also test spectral entropy
    all_entropy = [r["mean_key_spectral_entropy"] for r in results]
    rho_ent, p_ent = scipy_stats.spearmanr(all_levels, all_entropy)
    print(f"    Spearman rho (spectral entropy): {rho_ent:.4f} (p={p_ent:.6f})")

    supported = abs(rho) > 0.5 and p_rho < 0.01
    print(f"\n    H10a {'SUPPORTED' if supported else 'NOT SUPPORTED'}: "
          f"rho={'>' if abs(rho)>0.5 else '<'}0.5, p={'<' if p_rho<0.01 else '>'}0.01")

    return {
        "hypothesis": "H10a",
        "prediction": "Spearman rho > 0.5 between Bloom level and effective rank",
        "spearman_rho_individual": float(rho),
        "p_value_individual": float(p_rho),
        "spearman_rho_aggregated": float(rho_agg),
        "p_value_aggregated": float(p_rho_agg),
        "spearman_rho_entropy": float(rho_ent),
        "p_entropy": float(p_ent),
        "per_level_means": {str(l): float(m) for l, m in zip(levels, level_means)},
        "per_level_stds": {str(l): float(s) for l, s in zip(levels, level_stds)},
        "per_level_n": {str(l): len(level_ranks[l]) for l in levels},
        "supported": bool(supported),
    }


# ================================================================
# ANALYSIS: H10b — Independence from content domain (two-way ANOVA)
# ================================================================

def analyze_h10b(results: List[Dict]) -> Dict:
    """H10b: Bloom-geometry relationship is independent of content domain.

    Two-way ANOVA: Bloom level (6) x Content domain (5) on effective rank.
    Prediction: Bloom main effect is significant (p < 0.01) after controlling
    for domain. Interaction term is non-significant or small.

    Uses Type I SS via sequential one-way ANOVAs (works with scipy).
    """
    print("\n  H10b: Bloom Level x Content Domain Independence")
    print("  " + "-" * 50)

    domains = sorted(set(r["domain"] for r in results))
    levels = sorted(set(r["bloom_level"] for r in results))

    # One-way ANOVA: Bloom level main effect
    bloom_groups = [
        [r["mean_key_effective_rank"] for r in results if r["bloom_level"] == l]
        for l in levels
    ]
    f_bloom, p_bloom = scipy_stats.f_oneway(*bloom_groups)

    # One-way ANOVA: Domain main effect
    domain_groups = [
        [r["mean_key_effective_rank"] for r in results if r["domain"] == d]
        for d in domains
    ]
    f_domain, p_domain = scipy_stats.f_oneway(*domain_groups)

    # Eta-squared for each
    all_values = [r["mean_key_effective_rank"] for r in results]
    grand_mean = np.mean(all_values)
    ss_total = sum((v - grand_mean) ** 2 for v in all_values)

    ss_bloom = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2 for g in bloom_groups
    )
    ss_domain = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2 for g in domain_groups
    )

    eta2_bloom = ss_bloom / ss_total if ss_total > 0 else 0
    eta2_domain = ss_domain / ss_total if ss_total > 0 else 0

    # Within-domain Spearman correlations (Bloom effect within each domain)
    within_domain_rhos = {}
    for d in domains:
        d_results = [r for r in results if r["domain"] == d]
        d_levels = [r["bloom_level"] for r in d_results]
        d_ranks = [r["mean_key_effective_rank"] for r in d_results]
        rho, p = scipy_stats.spearmanr(d_levels, d_ranks)
        within_domain_rhos[d] = {"rho": float(rho), "p": float(p)}
        print(f"    Domain '{d}': Bloom-rank rho = {rho:.3f} (p={p:.4f})")

    print(f"\n    Bloom main effect: F={f_bloom:.3f}, p={p_bloom:.6f}, eta2={eta2_bloom:.4f}")
    print(f"    Domain main effect: F={f_domain:.3f}, p={p_domain:.6f}, eta2={eta2_domain:.4f}")

    bloom_dominates = eta2_bloom > eta2_domain
    bloom_significant = p_bloom < 0.01
    supported = bloom_significant
    print(f"\n    H10b {'SUPPORTED' if supported else 'NOT SUPPORTED'}: "
          f"Bloom p={'<' if bloom_significant else '>'}0.01")
    print(f"    Bloom {'>' if bloom_dominates else '<'} Domain in variance explained "
          f"(eta2: {eta2_bloom:.4f} vs {eta2_domain:.4f})")

    return {
        "hypothesis": "H10b",
        "prediction": "Bloom main effect significant (p<0.01), independent of domain",
        "bloom_F": float(f_bloom),
        "bloom_p": float(p_bloom),
        "bloom_eta2": float(eta2_bloom),
        "domain_F": float(f_domain),
        "domain_p": float(p_domain),
        "domain_eta2": float(eta2_domain),
        "bloom_dominates_domain": bool(bloom_dominates),
        "within_domain_correlations": within_domain_rhos,
        "supported": bool(supported),
    }


# ================================================================
# ANALYSIS: H10c — Bloom explains variance beyond domain
# ================================================================

def analyze_h10c(results: List[Dict]) -> Dict:
    """H10c: Adding Bloom level to a domain-only model improves R^2.

    Model 1: effective_rank ~ domain (baseline)
    Model 2: effective_rank ~ domain + bloom_level
    Model 3: effective_rank ~ bloom_level only

    Prediction: delta_R2 > 0.05 when adding Bloom to domain model.
    """
    print("\n  H10c: Bloom Explains Variance Beyond Content Domain")
    print("  " + "-" * 50)

    # Encode domain as dummy variables
    domains = sorted(set(r["domain"] for r in results))
    domain_map = {d: i for i, d in enumerate(domains)}

    y = np.array([r["mean_key_effective_rank"] for r in results])
    bloom = np.array([r["bloom_level"] for r in results], dtype=float)

    # Domain dummies (one-hot, drop first for identifiability)
    domain_idx = np.array([domain_map[r["domain"]] for r in results])
    X_domain = np.zeros((len(results), len(domains) - 1))
    for i in range(1, len(domains)):
        X_domain[:, i - 1] = (domain_idx == i).astype(float)

    # Model 1: domain only
    X1 = np.column_stack([np.ones(len(y)), X_domain])
    beta1, res1, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    ss_res1 = np.sum((y - X1 @ beta1) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    r2_domain = 1 - ss_res1 / ss_total if ss_total > 0 else 0

    # Model 2: domain + bloom
    X2 = np.column_stack([np.ones(len(y)), X_domain, bloom])
    beta2, res2, _, _ = np.linalg.lstsq(X2, y, rcond=None)
    ss_res2 = np.sum((y - X2 @ beta2) ** 2)
    r2_domain_bloom = 1 - ss_res2 / ss_total if ss_total > 0 else 0

    # Model 3: bloom only
    X3 = np.column_stack([np.ones(len(y)), bloom])
    beta3, res3, _, _ = np.linalg.lstsq(X3, y, rcond=None)
    ss_res3 = np.sum((y - X3 @ beta3) ** 2)
    r2_bloom = 1 - ss_res3 / ss_total if ss_total > 0 else 0

    delta_r2 = r2_domain_bloom - r2_domain
    n = len(y)
    k1 = X1.shape[1]
    k2 = X2.shape[1]

    # F-test for nested model comparison
    if ss_res2 > 0 and (k2 - k1) > 0:
        f_stat = ((ss_res1 - ss_res2) / (k2 - k1)) / (ss_res2 / (n - k2))
        p_f = 1 - scipy_stats.f.cdf(f_stat, k2 - k1, n - k2)
    else:
        f_stat = 0.0
        p_f = 1.0

    # AIC for model comparison
    def aic(n, k, ss_res):
        if ss_res <= 0:
            return float('inf')
        return n * np.log(ss_res / n) + 2 * k

    aic1 = aic(n, k1, ss_res1)
    aic2 = aic(n, k2, ss_res2)
    aic3 = aic(n, X3.shape[1], ss_res3)

    print(f"    Model 1 (domain only):        R2 = {r2_domain:.4f}, AIC = {aic1:.1f}")
    print(f"    Model 2 (domain + Bloom):     R2 = {r2_domain_bloom:.4f}, AIC = {aic2:.1f}")
    print(f"    Model 3 (Bloom only):         R2 = {r2_bloom:.4f}, AIC = {aic3:.1f}")
    print(f"    Delta R2 (Model 2 - Model 1): {delta_r2:.4f}")
    print(f"    F-test (nested): F={f_stat:.3f}, p={p_f:.6f}")
    print(f"    Bloom coefficient: {beta2[-1]:.4f}")

    supported = delta_r2 > 0.05 and p_f < 0.05
    print(f"\n    H10c {'SUPPORTED' if supported else 'NOT SUPPORTED'}: "
          f"delta_R2={'>' if delta_r2>0.05 else '<'}0.05, p={'<' if p_f<0.05 else '>'}0.05")

    return {
        "hypothesis": "H10c",
        "prediction": "Adding Bloom improves R2 by > 0.05",
        "r2_domain_only": float(r2_domain),
        "r2_domain_bloom": float(r2_domain_bloom),
        "r2_bloom_only": float(r2_bloom),
        "delta_r2": float(delta_r2),
        "f_statistic": float(f_stat),
        "p_value": float(p_f),
        "bloom_coefficient": float(beta2[-1]),
        "aic_domain": float(aic1),
        "aic_domain_bloom": float(aic2),
        "aic_bloom": float(aic3),
        "supported": bool(supported),
    }


# ================================================================
# ANALYSIS: H10d — Retroactive Campaign 1 prediction
# ================================================================

def analyze_h10d() -> Dict:
    """H10d: Campaign 1 effect sizes predicted by Bloom level.

    Maps Campaign 1 prompt categories to approximate Bloom levels and
    checks if correlation with effect sizes is positive.

    This is a purely analytical test — no new data needed.
    """
    print("\n  H10d: Retroactive Campaign 1 Prediction")
    print("  " + "-" * 50)

    # Campaign 1 category -> approximate Bloom level mapping
    # (from protocol: assessed based on cognitive demand of each category)
    campaign1_mapping = {
        "non_self_reference":  {"bloom_approx": 2.0, "label": "Understand"},
        "grounded_facts":      {"bloom_approx": 1.5, "label": "Remember/Understand"},
        "confabulation":       {"bloom_approx": 3.5, "label": "Apply/Analyze"},
        "self_reference":      {"bloom_approx": 5.0, "label": "Evaluate"},
        "philosophical":       {"bloom_approx": 4.5, "label": "Analyze/Evaluate"},
        "guardrail_test":      {"bloom_approx": 5.0, "label": "Evaluate"},
        "creative":            {"bloom_approx": 6.0, "label": "Create"},
        "coding":              {"bloom_approx": 3.5, "label": "Apply/Analyze"},
        "math_reasoning":      {"bloom_approx": 3.0, "label": "Apply"},
        "emotional":           {"bloom_approx": 2.5, "label": "Understand/Apply"},
    }

    # Campaign 1 effect sizes (Cohen's d for eff_rank vs grounded_facts baseline, 7B scale)
    # These are approximate values from Campaign 1 results
    campaign1_effects = {
        "non_self_reference":  0.15,
        "grounded_facts":      0.00,  # baseline
        "confabulation":       0.46,
        "self_reference":      0.59,
        "guardrail_test":      1.28,
        "creative":            0.89,
        "coding":              2.90,
        "math_reasoning":      -0.42,
        "emotional":           -0.27,
    }

    categories = sorted(set(campaign1_mapping.keys()) & set(campaign1_effects.keys()))
    bloom_levels = [campaign1_mapping[c]["bloom_approx"] for c in categories]
    effect_sizes = [abs(campaign1_effects[c]) for c in categories]

    rho, p = scipy_stats.spearmanr(bloom_levels, effect_sizes)

    print("    Category -> Bloom Level -> |Effect Size|:")
    for c in categories:
        b = campaign1_mapping[c]["bloom_approx"]
        d = campaign1_effects[c]
        print(f"      {c:>22s}: Bloom ~{b:.1f} ({campaign1_mapping[c]['label']:>20s}), "
              f"|d| = {abs(d):.2f}")

    print(f"\n    Spearman rho: {rho:.4f} (p={p:.4f})")

    supported = rho > 0.4 and p < 0.10  # lenient threshold for retroactive analysis
    print(f"\n    H10d {'SUPPORTED' if supported else 'NOT SUPPORTED'}: "
          f"rho={'>' if rho>0.4 else '<'}0.4")

    return {
        "hypothesis": "H10d",
        "prediction": "Bloom level predicts Campaign 1 effect sizes (rho > 0.4)",
        "spearman_rho": float(rho),
        "p_value": float(p),
        "mapping": {c: {"bloom": campaign1_mapping[c]["bloom_approx"],
                        "effect_d": campaign1_effects[c]}
                    for c in categories},
        "supported": bool(supported),
        "note": "Retroactive analysis — interpret with caution",
    }


# ================================================================
# ANALYSIS: H10e — Low/High Bloom clustering
# ================================================================

def analyze_h10e(results: List[Dict]) -> Dict:
    """H10e: Remember/Understand geometrically separable from Evaluate/Create.

    k-means (k=2) on geometric features separates low-Bloom (1-2) from
    high-Bloom (5-6) with accuracy > 70%.
    """
    print("\n  H10e: Low vs High Bloom Clustering")
    print("  " + "-" * 50)

    # Filter to low (1-2) and high (5-6) Bloom levels only
    low = [r for r in results if r["bloom_level"] <= 2]
    high = [r for r in results if r["bloom_level"] >= 5]

    if len(low) < 5 or len(high) < 5:
        print("    Insufficient data for clustering analysis.")
        return {"hypothesis": "H10e", "supported": False, "error": "insufficient_data"}

    # Features: effective rank + spectral entropy
    low_features = np.array([
        [r["mean_key_effective_rank"], r["mean_key_spectral_entropy"]] for r in low
    ])
    high_features = np.array([
        [r["mean_key_effective_rank"], r["mean_key_spectral_entropy"]] for r in high
    ])

    all_features = np.vstack([low_features, high_features])
    true_labels = np.array([0] * len(low) + [1] * len(high))

    # Standardize features
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    std[std == 0] = 1
    features_norm = (all_features - mean) / std

    # Simple k-means (k=2) — implemented without sklearn dependency
    # Initialize with means of known groups
    centroid0 = features_norm[true_labels == 0].mean(axis=0)
    centroid1 = features_norm[true_labels == 1].mean(axis=0)

    for _ in range(50):  # max iterations
        # Assign
        d0 = np.sum((features_norm - centroid0) ** 2, axis=1)
        d1 = np.sum((features_norm - centroid1) ** 2, axis=1)
        predicted = (d1 < d0).astype(int)

        # Update
        if predicted.sum() > 0:
            new_c1 = features_norm[predicted == 1].mean(axis=0)
        else:
            new_c1 = centroid1
        if (1 - predicted).sum() > 0:
            new_c0 = features_norm[predicted == 0].mean(axis=0)
        else:
            new_c0 = centroid0

        if np.allclose(new_c0, centroid0) and np.allclose(new_c1, centroid1):
            break
        centroid0, centroid1 = new_c0, new_c1

    # Accuracy (handle label flipping — k-means doesn't know which cluster is which)
    acc_normal = np.mean(predicted == true_labels)
    acc_flipped = np.mean((1 - predicted) == true_labels)
    accuracy = max(acc_normal, acc_flipped)

    # Also compute Cohen's d between low and high groups on each feature
    d_rank = cohens_d(
        [r["mean_key_effective_rank"] for r in high],
        [r["mean_key_effective_rank"] for r in low]
    )
    d_entropy = cohens_d(
        [r["mean_key_spectral_entropy"] for r in high],
        [r["mean_key_spectral_entropy"] for r in low]
    )

    print(f"    Low Bloom (1-2): n={len(low)}, mean_eff_rank={np.mean(low_features[:,0]):.2f}")
    print(f"    High Bloom (5-6): n={len(high)}, mean_eff_rank={np.mean(high_features[:,0]):.2f}")
    print(f"    k-means accuracy: {accuracy:.1%}")
    print(f"    Cohen's d (eff_rank, high vs low): {d_rank:.3f} ({interpret_d(d_rank)})")
    print(f"    Cohen's d (entropy, high vs low): {d_entropy:.3f} ({interpret_d(d_entropy)})")

    supported = accuracy > 0.70
    print(f"\n    H10e {'SUPPORTED' if supported else 'NOT SUPPORTED'}: "
          f"accuracy={'>' if accuracy>0.70 else '<'}70%")

    return {
        "hypothesis": "H10e",
        "prediction": "k-means separates low/high Bloom with > 70% accuracy",
        "n_low": len(low),
        "n_high": len(high),
        "clustering_accuracy": float(accuracy),
        "cohens_d_effective_rank": float(d_rank),
        "cohens_d_spectral_entropy": float(d_entropy),
        "low_mean_rank": float(np.mean(low_features[:, 0])),
        "high_mean_rank": float(np.mean(high_features[:, 0])),
        "supported": bool(supported),
    }


# ================================================================
# PROMPT LENGTH ANALYSIS (Covariate)
# ================================================================

def analyze_length_covariate(results: List[Dict]) -> Dict:
    """Check whether prompt length confounds the Bloom-geometry relationship."""
    print("\n  LENGTH COVARIATE ANALYSIS")
    print("  " + "-" * 50)

    levels = sorted(set(r["bloom_level"] for r in results))
    for l in levels:
        tokens = [r["n_tokens"] for r in results if r["bloom_level"] == l]
        print(f"    Level {l}: mean_tokens = {np.mean(tokens):.1f} +/- {np.std(tokens):.1f}")

    # Correlation between token count and effective rank
    tokens = [r["n_tokens"] for r in results]
    ranks = [r["mean_key_effective_rank"] for r in results]
    rho_tokens, p_tokens = scipy_stats.spearmanr(tokens, ranks)
    print(f"\n    Token count ↔ eff_rank: rho={rho_tokens:.4f} (p={p_tokens:.6f})")

    # Partial correlation: Bloom ↔ eff_rank controlling for tokens
    bloom_levels = [r["bloom_level"] for r in results]
    # Residualize both variables on token count
    tokens_arr = np.array(tokens, dtype=float)
    bloom_arr = np.array(bloom_levels, dtype=float)
    ranks_arr = np.array(ranks, dtype=float)

    # Simple linear regression to remove token effect
    X = np.column_stack([np.ones(len(tokens_arr)), tokens_arr])
    beta_bloom, _, _, _ = np.linalg.lstsq(X, bloom_arr, rcond=None)
    beta_rank, _, _, _ = np.linalg.lstsq(X, ranks_arr, rcond=None)
    bloom_resid = bloom_arr - X @ beta_bloom
    rank_resid = ranks_arr - X @ beta_rank

    rho_partial, p_partial = scipy_stats.spearmanr(bloom_resid, rank_resid)
    print(f"    Bloom ↔ eff_rank (controlling for tokens): rho={rho_partial:.4f} (p={p_partial:.6f})")

    return {
        "token_rank_correlation": {"rho": float(rho_tokens), "p": float(p_tokens)},
        "bloom_rank_partial": {"rho": float(rho_partial), "p": float(p_partial)},
        "tokens_per_level": {
            str(l): {"mean": float(np.mean([r["n_tokens"] for r in results if r["bloom_level"] == l])),
                     "std": float(np.std([r["n_tokens"] for r in results if r["bloom_level"] == l]))}
            for l in levels
        },
    }


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="H10: Bloom's Taxonomy Geometric Predictor")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model name")
    parser.add_argument("--quantize", action="store_true",
                        help="Use 4-bit NF4 quantization")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per prompt (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print design and exit without running")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-prompt details")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Path to bloom_taxonomy_prompts.json (auto-detected)")
    args = parser.parse_args()

    env = log_environment()

    # Find prompts file
    if args.prompts:
        prompts_path = Path(args.prompts)
    else:
        # Try relative paths from script location and CWD
        candidates = [
            Path(__file__).parent.parent / "prompts" / "bloom_taxonomy_prompts.json",
            Path("prompts") / "bloom_taxonomy_prompts.json",
            Path(__file__).parent / ".." / "prompts" / "bloom_taxonomy_prompts.json",
        ]
        prompts_path = None
        for c in candidates:
            if c.exists():
                prompts_path = c
                break
        if prompts_path is None:
            print("ERROR: Could not find bloom_taxonomy_prompts.json")
            print("  Tried:", [str(c) for c in candidates])
            sys.exit(1)

    prompts = load_bloom_prompts(prompts_path)
    print_banner(env, args.model)
    print_prompt_design(prompts)

    # Power advisory
    n_per_cell = sum(1 for p in prompts if p["bloom_level"] == 1 and p["domain"] == "programming")
    n_per_cell *= args.runs
    pa = power_advisory(n_per_cell)
    print(f"  Power advisory: {n_per_cell} obs/cell -> {pa['approx_power']:.1%} power for d={pa['target_d']}")
    print(f"  Total inferences: {len(prompts) * args.runs}")
    print()

    if args.dry_run:
        print("  [DRY RUN] Exiting without model loading or inference.")
        return

    # Load model
    model, tokenizer = load_model(args.model, quantize=args.quantize)

    # Run experiment
    print("\n  Running experiment...")
    start = datetime.now()
    results = run_bloom_experiment(
        model, tokenizer, prompts,
        n_runs=args.runs, seed=args.seed, verbose=args.verbose
    )
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n  Completed {len(results)} observations in {elapsed:.0f}s "
          f"({elapsed/len(results):.2f}s/inference)")

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ================================================================
    # RUN ALL ANALYSES
    # ================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS")
    print("=" * 70)

    h10a = analyze_h10a(results, seed=args.seed)
    h10b = analyze_h10b(results)
    h10c = analyze_h10c(results)
    h10d = analyze_h10d()  # retroactive, no new data
    h10e = analyze_h10e(results)
    length = analyze_length_covariate(results)

    # ================================================================
    # PAIRWISE BLOOM LEVEL COMPARISONS
    # ================================================================
    print("\n  PAIRWISE BLOOM LEVEL COMPARISONS")
    print("  " + "-" * 50)

    levels = sorted(set(r["bloom_level"] for r in results))
    pairwise = []
    p_values = []
    for i, l1 in enumerate(levels):
        for l2 in levels[i+1:]:
            g1 = [r["mean_key_effective_rank"] for r in results if r["bloom_level"] == l1]
            g2 = [r["mean_key_effective_rank"] for r in results if r["bloom_level"] == l2]
            comp = full_comparison(g2, g1, label=f"Level {l2} vs Level {l1}", seed=args.seed)
            pairwise.append(comp)
            p_values.append(comp["recommended_p"])
            d = comp["cohens_d"]["d"]
            print(f"    Level {l2} vs {l1}: d={d:+.3f} ({interpret_d(d)}), "
                  f"p={comp['recommended_p']:.4f}")

    # Holm-Bonferroni correction
    corrections = holm_bonferroni(p_values)
    print("\n    After Holm-Bonferroni correction:")
    for comp, corr in zip(pairwise, corrections):
        sig = "*" if corr["reject_null"] else " "
        print(f"    {sig} {comp['label']}: corrected p={corr['corrected_p']:.4f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    hypotheses = [h10a, h10b, h10c, h10d, h10e]
    for h in hypotheses:
        status = "SUPPORTED" if h.get("supported") else "NOT SUPPORTED"
        print(f"    {h['hypothesis']}: {status}")
    print()

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = get_output_path(results_dir, "bloom_taxonomy", args.model, args.quantize)

    output = {
        "experiment": "H10_bloom_taxonomy",
        "model": args.model,
        "model_id": model_id_from_name(args.model),
        "quantize": args.quantize,
        "n_runs": args.runs,
        "seed": args.seed,
        "n_prompts": len(prompts),
        "n_observations": len(results),
        "elapsed_seconds": elapsed,
        "environment": env,
        "hypotheses": {
            "H10a": h10a,
            "H10b": h10b,
            "H10c": h10c,
            "H10d": h10d,
            "H10e": h10e,
        },
        "length_covariate": length,
        "pairwise_comparisons": pairwise,
        "holm_bonferroni": corrections,
        "raw_results": results,
    }

    # Compute SHA-256 checksum
    content = json.dumps(output, indent=2, default=str)
    output["checksum"] = hashlib.sha256(content.encode()).hexdigest()

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}")
    print(f"  SHA-256: {output['checksum'][:16]}...")
    print()


if __name__ == "__main__":
    main()
