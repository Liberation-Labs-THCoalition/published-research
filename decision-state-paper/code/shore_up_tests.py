"""
Shore-Up Tests — Address red team concerns on existing data.
1. Single-feature ablation (does stable rank alone separate?)
2. Token frequency baseline (is geometry just word frequency?)
3. Template-vs-content logit split (is confidence paradox linguistic?)
4. Bootstrap CI on combined AUROC
5. Check Wang et al. effective rank publication date

Run: python3 shore_up_tests.py --starship
"""

import json, argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

STARSHIP_RESULTS = Path(
    "/Users/margaret/models/research_results/matched_burn"
    "/matched_burn_results.json")


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return data["trials"], data.get("probe_layers", [3, 7, 11, 15])


def test_1_feature_ablation(trials, probe_layers):
    """Single-feature AUROCs with GroupKFold."""
    print("\n" + "=" * 60)
    print("TEST 1: SINGLE-FEATURE ABLATION")
    print("=" * 60)

    y = np.array([0 if t["is_real"] else 1 for t in trials])
    groups = np.array([t["pair_group"] for t in trials])
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))

    features = {}

    # Encoding W_K
    for key in ["valence_proj", "uncertainty_proj", "reward_proj"]:
        features[f"enc_{key}"] = np.array(
            [t["enc_wk"].get(key, 0) for t in trials]).reshape(-1, 1)

    # Encoding SVD stable rank per layer
    for li in probe_layers:
        ln = f"L{li}"
        features[f"{ln}_enc_stable_rank"] = np.array(
            [t["enc_svd"].get(ln, {}).get("stable_rank", 0)
             for t in trials]).reshape(-1, 1)

    # Skip-SV1 stable rank per layer
    for li in probe_layers:
        ln = f"L{li}"
        features[f"{ln}_skip1_stable_rank"] = np.array(
            [t["enc_svd_skip1"].get(ln, {}).get("skip1_stable_rank", 0)
             for t in trials]).reshape(-1, 1)

    # Generation W_K
    for key in ["valence_proj", "uncertainty_proj", "reward_proj"]:
        features[f"gen_{key}"] = np.array(
            [t["gen_wk"].get(key, 0) for t in trials]).reshape(-1, 1)

    # Logit features
    features["mean_logit_entropy"] = np.array(
        [np.mean(t["trajectory"]["logit_entropy"][:10])
         for t in trials]).reshape(-1, 1)
    features["mean_logit_margin"] = np.array(
        [np.mean(t["trajectory"]["logit_margin"][:10])
         for t in trials]).reshape(-1, 1)

    results = []
    for name, X in sorted(features.items()):
        X_clean = np.nan_to_num(X)
        probs = np.full(len(y), np.nan)

        for train_idx, test_idx in gkf.split(X_clean, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_clean[train_idx])
            X_te = scaler.transform(X_clean[test_idx])
            clf = LogisticRegression(C=0.1, max_iter=2000)
            clf.fit(X_tr, y[train_idx])
            probs[test_idx] = clf.predict_proba(X_te)[:, 1]

        valid = ~np.isnan(probs)
        if np.sum(valid) > 0 and len(set(y[valid])) > 1:
            auroc = roc_auc_score(y[valid], probs[valid])
        else:
            auroc = 0.5
        results.append((name, auroc))

    results.sort(key=lambda x: -x[1])
    print(f"\n  {'Feature':40s} {'AUROC':>8s}")
    print(f"  {'-'*40} {'-'*8}")
    for name, auroc in results:
        flag = " ***" if auroc > 0.9 else (" **" if auroc > 0.8 else (
               " *" if auroc > 0.7 else ""))
        print(f"  {name:40s} {auroc:8.3f}{flag}")

    top_name, top_auroc = results[0]
    print(f"\n  BEST SINGLE FEATURE: {top_name} at AUROC {top_auroc:.3f}")
    if top_auroc > 0.9:
        print(f"  -> Overfit concern REDUCED: single feature separates well")
    elif top_auroc > 0.8:
        print(f"  -> Single feature moderate; combined may add value")
    else:
        print(f"  -> WARNING: no single feature above 0.8; "
              f"combined AUROC 1.000 likely overfitting")


def test_2_token_frequency(trials):
    """Does entity token frequency alone predict knowability?"""
    print("\n" + "=" * 60)
    print("TEST 2: TOKEN FREQUENCY BASELINE")
    print("=" * 60)

    # prompt_len is matched, but entity embedding norm might differ
    # Use the encoding norm as proxy for token familiarity
    real_norms = []
    fake_norms = []
    for t in trials:
        for li_name in ["L3", "L7", "L11", "L15"]:
            norm = t["enc_svd"].get(li_name, {}).get("norm", 0)
            if t["is_real"]:
                real_norms.append(norm)
            else:
                fake_norms.append(norm)

    real_arr = np.array(real_norms)
    fake_arr = np.array(fake_norms)
    t_stat, p_val = stats.ttest_ind(real_arr, fake_arr)
    d = (np.mean(fake_arr) - np.mean(real_arr)) / np.sqrt(
        (np.var(real_arr) + np.var(fake_arr)) / 2)

    print(f"\n  Encoding norm (proxy for token familiarity):")
    print(f"    Real: {np.mean(real_arr):.1f} +/- {np.std(real_arr):.1f}")
    print(f"    Fake: {np.mean(fake_arr):.1f} +/- {np.std(fake_arr):.1f}")
    print(f"    d={d:+.3f} p={p_val:.4f}")

    if abs(d) < 0.1:
        print(f"  -> Norm does NOT separate conditions (good: geometry "
              f"adds value beyond frequency)")
    else:
        print(f"  -> Norm separates — potential frequency confound")

    # More direct: prompt_len should be identical (already verified)
    r_lens = [t["prompt_len"] for t in trials if t["is_real"]]
    f_lens = [t["prompt_len"] for t in trials if not t["is_real"]]
    print(f"\n  Prompt length (should be matched):")
    print(f"    Real: {np.mean(r_lens):.1f} Fake: {np.mean(f_lens):.1f} "
          f"t={stats.ttest_ind(r_lens, f_lens).statistic:.3f}")


def test_3_template_vs_content(trials):
    """Does the confidence paradox hold for content tokens only?"""
    print("\n" + "=" * 60)
    print("TEST 3: TEMPLATE vs CONTENT LOGIT SPLIT")
    print("=" * 60)

    # Template tokens: 0-7 (shared preamble in most cases)
    # Content tokens: 8+ (after "My reasoning:")
    for phase, start, end, label in [
        ("template", 0, 8, "Tokens 0-7 (preamble)"),
        ("content", 8, 20, "Tokens 8-19 (content)"),
        ("late", 20, 50, "Tokens 20-49 (deep content)"),
    ]:
        real_ent = []
        fake_ent = []
        real_mar = []
        fake_mar = []

        for t in trials:
            ent_vals = t["trajectory"]["logit_entropy"][start:end]
            mar_vals = t["trajectory"]["logit_margin"][start:end]
            if ent_vals:
                if t["is_real"]:
                    real_ent.append(np.mean(ent_vals))
                    real_mar.append(np.mean(mar_vals))
                else:
                    fake_ent.append(np.mean(ent_vals))
                    fake_mar.append(np.mean(mar_vals))

        if real_ent and fake_ent:
            t_e, p_e = stats.ttest_ind(real_ent, fake_ent)
            d_e = (np.mean(fake_ent) - np.mean(real_ent)) / np.sqrt(
                (np.var(real_ent) + np.var(fake_ent)) / 2)
            t_m, p_m = stats.ttest_ind(real_mar, fake_mar)
            d_m = (np.mean(fake_mar) - np.mean(real_mar)) / np.sqrt(
                (np.var(real_mar) + np.var(fake_mar)) / 2)

            print(f"\n  {label}:")
            print(f"    Entropy: real={np.mean(real_ent):.3f} "
                  f"fake={np.mean(fake_ent):.3f} d={d_e:+.3f} "
                  f"p={p_e:.4f}")
            print(f"    Margin:  real={np.mean(real_mar):.2f} "
                  f"fake={np.mean(fake_mar):.2f} d={d_m:+.3f} "
                  f"p={p_m:.4f}")


def test_4_bootstrap_ci(trials):
    """Bootstrap confidence interval on combined AUROC."""
    print("\n" + "=" * 60)
    print("TEST 4: BOOTSTRAP CI ON COMBINED AUROC")
    print("=" * 60)

    y = np.array([0 if t["is_real"] else 1 for t in trials])
    groups = np.array([t["pair_group"] for t in trials])

    def build_features(trial):
        f = []
        for key in ["valence_proj", "uncertainty_proj", "reward_proj"]:
            f.append(trial["enc_wk"].get(key, 0))
            f.append(trial["gen_wk"].get(key, 0))
        for li in [3, 7, 11, 15]:
            ln = f"L{li}"
            for feat in ["stable_rank", "spectral_entropy", "top_sv_ratio"]:
                f.append(trial["enc_svd"].get(ln, {}).get(feat, 0))
                f.append(trial["gen_svd"].get(ln, {}).get(feat, 0))
                f.append(trial["delta_svd"].get(ln, {}).get(feat, 0))
            for feat in ["skip1_stable_rank", "skip1_spectral_entropy"]:
                f.append(trial["enc_svd_skip1"].get(ln, {}).get(feat, 0))
        f.append(np.mean(trial["trajectory"]["logit_entropy"][:10]))
        f.append(np.mean(trial["trajectory"]["logit_margin"][:10]))
        return f

    X = np.nan_to_num(np.array([build_features(t) for t in trials]))
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))

    np.random.seed(42)
    boot_aurocs = []
    for _ in range(1000):
        idx = np.random.choice(len(trials), len(trials), replace=True)
        X_b, y_b, g_b = X[idx], y[idx], groups[idx]
        if len(set(y_b)) < 2 or len(set(g_b)) < 3:
            continue
        try:
            probs = np.full(len(y_b), np.nan)
            for train, test in gkf.split(X_b, y_b, g_b):
                sc = StandardScaler()
                clf = LogisticRegression(C=0.1, max_iter=2000)
                clf.fit(sc.fit_transform(X_b[train]), y_b[train])
                probs[test] = clf.predict_proba(
                    sc.transform(X_b[test]))[:, 1]
            v = ~np.isnan(probs)
            if np.sum(v) > 10 and len(set(y_b[v])) > 1:
                boot_aurocs.append(roc_auc_score(y_b[v], probs[v]))
        except Exception:
            continue

    boot = np.array(boot_aurocs)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  Bootstrap AUROC (1000 resamples):")
    print(f"    Mean: {np.mean(boot):.3f}")
    print(f"    95% CI: [{lo:.3f}, {hi:.3f}]")
    print(f"    N bootstrap samples: {len(boot)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starship", action="store_true")
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()

    path = (Path(args.path) if args.path
            else STARSHIP_RESULTS if args.starship
            else Path("results/matched_burn/matched_burn_results.json"))

    trials, probe_layers = load_data(path)
    print(f"Loaded {len(trials)} trials")

    test_1_feature_ablation(trials, probe_layers)
    test_2_token_frequency(trials)
    test_3_template_vs_content(trials)
    test_4_bootstrap_ci(trials)


if __name__ == "__main__":
    main()
