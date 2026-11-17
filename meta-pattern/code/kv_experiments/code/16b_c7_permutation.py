# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: Liberation Labs
# ============================================================
#!/usr/bin/env python3
"""
C7 Confabulation Permutation Test
==================================

Re-extracts per-layer rank profiles from C7 frequency-matched
confab/factual prompts and runs permutation test on direction
extraction AUROC (originally 0.288, flipped 0.712).

Needs GPU for encoding (~2 min). Permutation test is CPU only.
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prompts"))

from c7_frequency_matched_confabulation import (
    CONFABULATION_COMMON_TOKENS, FACTUAL_COMMON_TOKENS
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def encode_and_get_profile(model, tokenizer, text, device="cuda"):
    """Encode text, return per-layer effective rank profile."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=True)

    kv_cache = outputs.past_key_values
    n_layers = len(kv_cache)
    profile = []

    for layer_idx in range(n_layers):
        keys = kv_cache[layer_idx][0]  # [batch, heads, seq, dim]
        k = keys.squeeze(0)  # [heads, seq, dim]
        k_flat = k.reshape(-1, k.shape[-1]).float().cpu().numpy()

        try:
            sv = np.linalg.svd(k_flat, compute_uv=False)
            sv = sv[sv > 1e-10]
            p = sv / sv.sum()
            entropy = -np.sum(p * np.log(p + 1e-12))
            eff_rank = np.exp(entropy)
        except:
            eff_rank = 1.0

        profile.append(float(eff_rank))

    return profile


def direction_auroc(profiles_a, profiles_b):
    """Compute LOO-CV direction projection AUROC."""
    all_profiles = profiles_a + profiles_b
    all_labels = [1] * len(profiles_a) + [0] * len(profiles_b)
    n_total = len(all_profiles)

    loo_projections = []
    for i in range(n_total):
        train_p = all_profiles[:i] + all_profiles[i+1:]
        train_l = all_labels[:i] + all_labels[i+1:]
        train_a = [p for p, l in zip(train_p, train_l) if l == 1]
        train_b = [p for p, l in zip(train_p, train_l) if l == 0]
        d = np.mean(train_a, axis=0) - np.mean(train_b, axis=0)
        dn = np.linalg.norm(d)
        if dn > 0:
            d = d / dn
        centered = np.array(all_profiles[i]) - np.mean(train_p, axis=0)
        proj = float(np.dot(centered, d))
        loo_projections.append({"actual": all_labels[i], "projection": proj})

    y_true = [p["actual"] for p in loo_projections]
    y_score = [p["projection"] for p in loo_projections]

    if HAS_SKLEARN:
        auroc = roc_auc_score(y_true, y_score)
    else:
        a_projs = [p["projection"] for p in loo_projections if p["actual"] == 1]
        b_projs = [p["projection"] for p in loo_projections if p["actual"] == 0]
        u = sum(1 for ap in a_projs for bp in b_projs if ap > bp) + \
            0.5 * sum(1 for ap in a_projs for bp in b_projs if ap == bp)
        auroc = u / (len(a_projs) * len(b_projs))

    return auroc


def lr_loo_auroc(profiles_a, profiles_b):
    """LR LOO-CV effective AUROC."""
    if not HAS_SKLEARN:
        return 0.5

    X = np.array(profiles_a + profiles_b)
    y = np.array([1] * len(profiles_a) + [0] * len(profiles_b))
    n = len(y)
    probs = []

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X[i:i+1])
        clf = LogisticRegression(max_iter=1000, C=0.1)
        clf.fit(X_train_s, y_train)
        prob = clf.predict_proba(X_test_s)[0][1]
        probs.append({"actual": int(y[i]), "prob": float(prob)})

    auroc = roc_auc_score([p["actual"] for p in probs], [p["prob"] for p in probs])
    return auroc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-perms", type=int, default=200)
    parser.add_argument("--output", default="results/c7_permutation_test.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  C7 CONFABULATION PERMUTATION TEST")
    print(f"  Model: {args.model}")
    print(f"  Prompts: {len(CONFABULATION_COMMON_TOKENS)} confab + {len(FACTUAL_COMMON_TOKENS)} factual")
    print(f"  Permutations: {args.n_perms}")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Extract profiles
    print("\n[2/4] Extracting per-layer profiles...")
    confab_profiles = []
    for i, prompt in enumerate(CONFABULATION_COMMON_TOKENS):
        profile = encode_and_get_profile(model, tokenizer, prompt["text"])
        confab_profiles.append(profile)
        if (i + 1) % 10 == 0:
            print(f"  Confab {i+1}/{len(CONFABULATION_COMMON_TOKENS)}")

    factual_profiles = []
    for i, prompt in enumerate(FACTUAL_COMMON_TOKENS):
        profile = encode_and_get_profile(model, tokenizer, prompt["text"])
        factual_profiles.append(profile)
        if (i + 1) % 10 == 0:
            print(f"  Factual {i+1}/{len(FACTUAL_COMMON_TOKENS)}")

    n_layers = len(confab_profiles[0])
    print(f"  Extracted {len(confab_profiles)} + {len(factual_profiles)} profiles ({n_layers} layers)")

    # Real AUROC
    print("\n[3/4] Computing real AUROC...")
    real_dir_auroc = direction_auroc(confab_profiles, factual_profiles)
    real_lr_auroc = lr_loo_auroc(confab_profiles, factual_profiles)
    real_dir_eff = max(real_dir_auroc, 1 - real_dir_auroc)
    real_lr_eff = max(real_lr_auroc, 1 - real_lr_auroc)

    print(f"  Direction AUROC: {real_dir_auroc:.3f} (eff: {real_dir_eff:.3f})")
    print(f"  LR AUROC:        {real_lr_auroc:.3f} (eff: {real_lr_eff:.3f})")

    # Permutation test
    print(f"\n[4/4] Permutation test ({args.n_perms} permutations)...")
    import warnings
    warnings.filterwarnings("ignore")

    all_profiles = confab_profiles + factual_profiles
    n_total = len(all_profiles)
    rng = np.random.RandomState(42)

    perm_dir_aurocs = []
    perm_lr_aurocs = []

    for p in range(args.n_perms):
        perm_idx = rng.permutation(n_total)
        perm_a = [all_profiles[i] for i in perm_idx[:len(confab_profiles)]]
        perm_b = [all_profiles[i] for i in perm_idx[len(confab_profiles):]]

        pd_auroc = direction_auroc(perm_a, perm_b)
        perm_dir_aurocs.append(max(pd_auroc, 1 - pd_auroc))

        pl_auroc = lr_loo_auroc(perm_a, perm_b)
        perm_lr_aurocs.append(max(pl_auroc, 1 - pl_auroc))

        if (p + 1) % 25 == 0:
            print(f"  Perm {p+1}/{args.n_perms}: "
                  f"dir mean={np.mean(perm_dir_aurocs):.3f}, "
                  f"lr mean={np.mean(perm_lr_aurocs):.3f}")

    dir_p = np.mean([pa >= real_dir_eff for pa in perm_dir_aurocs])
    lr_p = np.mean([pa >= real_lr_eff for pa in perm_lr_aurocs])

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Direction projection:")
    print(f"    Real eff AUROC:   {real_dir_eff:.3f}")
    print(f"    Perm mean:        {np.mean(perm_dir_aurocs):.3f} +/- {np.std(perm_dir_aurocs):.3f}")
    print(f"    Perm 95th:        {np.percentile(perm_dir_aurocs, 95):.3f}")
    print(f"    Perm max:         {max(perm_dir_aurocs):.3f}")
    print(f"    p-value:          {dir_p:.4f}")
    if dir_p < 0.05:
        print(f"    -> SIGNIFICANT (p < 0.05)")
    else:
        print(f"    -> NOT SIGNIFICANT")

    print(f"\n  Logistic regression:")
    print(f"    Real eff AUROC:   {real_lr_eff:.3f}")
    print(f"    Perm mean:        {np.mean(perm_lr_aurocs):.3f} +/- {np.std(perm_lr_aurocs):.3f}")
    print(f"    Perm 95th:        {np.percentile(perm_lr_aurocs, 95):.3f}")
    print(f"    Perm max:         {max(perm_lr_aurocs):.3f}")
    print(f"    p-value:          {lr_p:.4f}")
    if lr_p < 0.05:
        print(f"    -> SIGNIFICANT (p < 0.05)")
    else:
        print(f"    -> NOT SIGNIFICANT")

    # Save
    results = {
        "experiment": "c7_confabulation_permutation_test",
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "n_confab": len(confab_profiles),
        "n_factual": len(factual_profiles),
        "n_layers": n_layers,
        "n_permutations": args.n_perms,
        "direction": {
            "real_auroc": real_dir_auroc,
            "real_eff_auroc": real_dir_eff,
            "perm_mean": float(np.mean(perm_dir_aurocs)),
            "perm_std": float(np.std(perm_dir_aurocs)),
            "perm_95th": float(np.percentile(perm_dir_aurocs, 95)),
            "perm_max": float(max(perm_dir_aurocs)),
            "p_value": float(dir_p),
        },
        "logistic_regression": {
            "real_auroc": real_lr_auroc,
            "real_eff_auroc": real_lr_eff,
            "perm_mean": float(np.mean(perm_lr_aurocs)),
            "perm_std": float(np.std(perm_lr_aurocs)),
            "perm_95th": float(np.percentile(perm_lr_aurocs, 95)),
            "perm_max": float(max(perm_lr_aurocs)),
            "p_value": float(lr_p),
        },
        "confab_profiles": confab_profiles,
        "factual_profiles": factual_profiles,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()
