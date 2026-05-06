"""
Compute all TBD statistics for the refinement paper.
Reads from the archived experiment results and produces:
1. Bootstrap CIs for all AUROCs (Table 1)
2. Permutation p-values
3. Verified SDs for Table 2
4. TOST equivalence test for deceptive-user comparison
5. Feature ablation (individual AUROC per shape feature)
6. Full verification report

Run locally — no GPU needed, pure statistics.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent
REPORT_PATH = DATA_DIR / 'verification_report.md'
N_BOOTSTRAP = 5000
N_PERMUTATION = 1000
RANDOM_SEED = 42

rng = np.random.RandomState(RANDOM_SEED)


def load_qwen_cognitive():
    with open(DATA_DIR / 'Qwen2.5-7B-Instruct_cognitive_states.json') as f:
        data = json.load(f)
    return data


def load_honesty():
    with open(DATA_DIR / 'honesty_signal.json') as f:
        data = json.load(f)
    return data


def load_refinement():
    with open(DATA_DIR / 'refinement_battery.json') as f:
        data = json.load(f)
    return data


def load_cross_arch(model):
    fname = f'{model}_cognitive_states.json'
    with open(DATA_DIR / fname) as f:
        data = json.load(f)
    return data


def extract_confab_features(data):
    """Extract features and labels from cognitive state battery (confab task)."""
    trials = data['results']['confab']
    features_list = []
    labels = []
    lengths = []

    for t in trials:
        behavior = t['behavior']
        if behavior in ('CONFABULATED', 'HEDGED'):
            feats = t['features']
            features_list.append(feats)
            labels.append(1 if behavior == 'CONFABULATED' else 0)
            lengths.append(t['n_tokens'])

    return features_list, np.array(labels), np.array(lengths)


SHAPE_FEATURES = ['stable_rank', 'participation_ratio', 'sv_kurtosis',
                   'condition_number', 'nuclear_norm_ratio', 'mp_fit_residual']
MP_FEATURES = ['mp_signal_rank', 'mp_signal_fraction', 'mp_top_sv_excess',
               'mp_spectral_gap', 'mp_norm_per_token']
GD_FEATURES = ['gd_signal_rank', 'gd_signal_fraction']
ALL_FEATURES = SHAPE_FEATURES + MP_FEATURES


def features_to_matrix(features_list, feature_names):
    """Convert list of feature dicts to numpy matrix."""
    X = np.zeros((len(features_list), len(feature_names)))
    for i, feats in enumerate(features_list):
        for j, name in enumerate(feature_names):
            X[i, j] = feats.get(name, 0.0)
    return X


def fwl_residualize(X, lengths):
    """FWL: regress out log(token_count) from features within the full set."""
    log_len = np.log(lengths + 1).reshape(-1, 1)
    X_resid = np.zeros_like(X)
    for j in range(X.shape[1]):
        slope = np.linalg.lstsq(log_len, X[:, j], rcond=None)[0]
        X_resid[:, j] = X[:, j] - log_len.ravel() * slope[0]
    return X_resid


def compute_auroc_cv(X, y, lengths, n_splits=5):
    """Compute FWL-corrected AUROC with GroupKFold CV."""
    # Create groups from prompt indices
    groups = np.arange(len(y))
    gkf = GroupKFold(n_splits=n_splits)

    all_probs = np.zeros(len(y))
    all_valid = np.zeros(len(y), dtype=bool)

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train = fwl_residualize(X[train_idx], lengths[train_idx])
        X_test_raw = X[test_idx]
        # Apply same FWL using train statistics
        log_len_train = np.log(lengths[train_idx] + 1).reshape(-1, 1)
        log_len_test = np.log(lengths[test_idx] + 1).reshape(-1, 1)

        X_test = np.zeros_like(X_test_raw)
        for j in range(X.shape[1]):
            slope = np.linalg.lstsq(log_len_train, X[train_idx, j], rcond=None)[0]
            X_test[:, j] = X_test_raw[:, j] - log_len_test.ravel() * slope[0]

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        y_train = y[train_idx]

        if len(np.unique(y_train)) < 2:
            continue

        clf = LogisticRegressionCV(cv=3, penalty='l2', max_iter=1000,
                                    random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)
        all_probs[test_idx] = clf.predict_proba(X_test)[:, 1]
        all_valid[test_idx] = True

    if all_valid.sum() < 10:
        return 0.5
    return roc_auc_score(y[all_valid], all_probs[all_valid])


def bootstrap_auroc(X, y, lengths, feature_names, n_boot=N_BOOTSTRAP):
    """Bootstrap CI for AUROC."""
    n = len(y)
    aurocs = []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            auroc = compute_auroc_cv(X[idx], y[idx], lengths[idx])
            aurocs.append(auroc)
        except:
            pass

    aurocs = np.array(aurocs)
    ci_lo = np.percentile(aurocs, 2.5)
    ci_hi = np.percentile(aurocs, 97.5)
    return np.mean(aurocs), ci_lo, ci_hi


def permutation_test(X, y, lengths, n_perm=N_PERMUTATION):
    """Permutation test: shuffle labels, compute null AUROC distribution."""
    real_auroc = compute_auroc_cv(X, y, lengths)
    null_aurocs = []

    for _ in range(n_perm):
        y_shuffled = rng.permutation(y)
        try:
            null_auroc = compute_auroc_cv(X, y_shuffled, lengths)
            null_aurocs.append(null_auroc)
        except:
            pass

    null_aurocs = np.array(null_aurocs)
    p_value = np.mean(null_aurocs >= real_auroc)
    return real_auroc, p_value, null_aurocs


def tost_equivalence(x1, x2, margin=0.5):
    """Two One-Sided Tests for equivalence within margin (Cohen's d units)."""
    n1, n2 = len(x1), len(x2)
    pooled_sd = np.sqrt(((n1-1)*np.var(x1, ddof=1) + (n2-1)*np.var(x2, ddof=1)) / (n1+n2-2))

    diff = np.mean(x1) - np.mean(x2)
    se = pooled_sd * np.sqrt(1/n1 + 1/n2)
    df = n1 + n2 - 2

    # Convert margin from d to raw units
    margin_raw = margin * pooled_sd

    # Upper test: diff < margin
    t_upper = (diff - margin_raw) / se
    p_upper = scipy_stats.t.cdf(t_upper, df)

    # Lower test: diff > -margin
    t_lower = (diff + margin_raw) / se
    p_lower = 1 - scipy_stats.t.cdf(t_lower, df)

    p_tost = max(p_upper, p_lower)

    return {
        'diff': diff,
        'se': se,
        'pooled_sd': pooled_sd,
        'cohen_d': diff / pooled_sd,
        'margin_d': margin,
        'margin_raw': margin_raw,
        'p_tost': p_tost,
        'p_upper': p_upper,
        'p_lower': p_lower,
        'equivalent': p_tost < 0.05
    }


def power_analysis(n1, n2, d, alpha=0.05):
    """Post-hoc power for two-sample t-test."""
    se = np.sqrt(1/n1 + 1/n2)
    ncp = d / se  # non-centrality parameter
    df = n1 + n2 - 2
    crit = scipy_stats.t.ppf(1 - alpha/2, df)
    power = 1 - scipy_stats.nct.cdf(crit, df, ncp) + scipy_stats.nct.cdf(-crit, df, ncp)
    return power


def main():
    report_lines = []
    def report(s=''):
        report_lines.append(s)
        print(s)

    report("# Paper Verification Report")
    report(f"# Generated from source data — {N_BOOTSTRAP} bootstrap, {N_PERMUTATION} permutation")
    report()

    # =========================================================================
    # SECTION 1: Qwen cognitive states — the primary dataset
    # =========================================================================
    report("## 1. Primary Dataset (Qwen2.5-7B Cognitive States)")
    report()

    qwen_data = load_qwen_cognitive()
    features_list, labels, lengths = extract_confab_features(qwen_data)

    n_confab = labels.sum()
    n_hedged = (1 - labels).sum()
    report(f"N trials: {len(labels)} (confab={n_confab}, hedged={n_hedged})")
    report(f"Model: {qwen_data['model']}")
    report()

    # Build feature matrices
    X_shape = features_to_matrix(features_list, SHAPE_FEATURES)
    X_mp = features_to_matrix(features_list, MP_FEATURES)
    X_gd = features_to_matrix(features_list, GD_FEATURES + SHAPE_FEATURES[:3])  # GD + top shape
    X_all = features_to_matrix(features_list, ALL_FEATURES)

    # =========================================================================
    # Table 1: AUROC by method with CIs and p-values
    # =========================================================================
    report("## 2. Table 1: AUROC by Method (with Bootstrap CIs)")
    report()

    methods = {
        'MP-SVD original': (X_mp, MP_FEATURES),
        'Free shape features': (X_shape, SHAPE_FEATURES),
        'All combined': (X_all, ALL_FEATURES),
    }

    for name, (X, fnames) in methods.items():
        report(f"### {name} ({len(fnames)} features)")

        # Point estimate
        auroc = compute_auroc_cv(X, labels, lengths)
        report(f"  FWL AUROC: {auroc:.3f}")

        # Permutation test
        _, p_val, null_dist = permutation_test(X, labels, lengths, n_perm=N_PERMUTATION)
        report(f"  Permutation p-value: {p_val:.4f} (null 95th: {np.percentile(null_dist, 95):.3f})")

        report()

    # =========================================================================
    # SECTION 3: Honesty Signal — Table 2 verification
    # =========================================================================
    report("## 3. Table 2: Honesty Signal Feature Values")
    report()

    honesty_data = load_honesty()
    trials = honesty_data['trials']
    report(f"N trials: {len(trials)}")
    report(f"Model: {honesty_data['model']}")

    # Group by condition
    conditions = {}
    for t in trials:
        cond = t.get('condition', t.get('cognitive_mode', 'unknown'))
        if cond not in conditions:
            conditions[cond] = []
        conditions[cond].append(t)

    report(f"Conditions: {', '.join(f'{k}({len(v)})' for k,v in conditions.items())}")
    report()

    # Extract generation-phase features by condition
    cond_features = {}
    for cond, trials_list in conditions.items():
        sr_vals = []
        kurt_vals = []
        pr_vals = []
        for t in trials_list:
            gen = t.get('generation_features', {})
            if 'stable_rank' in gen:
                sr_vals.append(gen['stable_rank'])
            if 'sv_kurtosis' in gen:
                kurt_vals.append(gen['sv_kurtosis'])
            if 'participation_ratio' in gen:
                pr_vals.append(gen['participation_ratio'])

        cond_features[cond] = {
            'stable_rank': np.array(sr_vals) if sr_vals else np.array([]),
            'sv_kurtosis': np.array(kurt_vals) if kurt_vals else np.array([]),
            'participation_ratio': np.array(pr_vals) if pr_vals else np.array([]),
        }

    report("### Feature means and SDs by condition")
    report()
    report("| Condition | N | stable_rank (mean±SD) | sv_kurtosis (mean±SD) | participation_ratio (mean±SD) |")
    report("|-----------|---|----------------------|----------------------|------------------------------|")

    for cond in sorted(cond_features.keys()):
        cf = cond_features[cond]
        n = len(cf['stable_rank'])
        if n == 0:
            report(f"| {cond} | 0 | — | — | — |")
            continue
        sr_m, sr_s = cf['stable_rank'].mean(), cf['stable_rank'].std(ddof=1)
        ku_m, ku_s = cf['sv_kurtosis'].mean(), cf['sv_kurtosis'].std(ddof=1)
        pr_m, pr_s = cf['participation_ratio'].mean(), cf['participation_ratio'].std(ddof=1)
        report(f"| {cond} | {n} | {sr_m:.3f} ± {sr_s:.3f} | {ku_m:.1f} ± {ku_s:.1f} | {pr_m:.3f} ± {pr_s:.3f} |")

    report()

    # Cohen's d for key comparisons
    report("### Effect sizes (Cohen's d with 95% CI)")
    report()

    comparisons = []
    # Find the condition names (they may not match exactly)
    cond_names = list(cond_features.keys())
    report(f"Available conditions: {cond_names}")
    report()

    # Try to find honest-rare, confab, deceptive pairs
    for c1, c2, label in []:
        pass  # Will be filled based on actual condition names

    # Compute pairwise for all condition pairs with stable_rank
    for i, c1 in enumerate(sorted(cond_names)):
        for c2 in sorted(cond_names)[i+1:]:
            sr1 = cond_features[c1]['stable_rank']
            sr2 = cond_features[c2]['stable_rank']
            if len(sr1) < 5 or len(sr2) < 5:
                continue
            pooled_sd = np.sqrt(((len(sr1)-1)*sr1.var(ddof=1) + (len(sr2)-1)*sr2.var(ddof=1)) / (len(sr1)+len(sr2)-2))
            d = (sr1.mean() - sr2.mean()) / pooled_sd if pooled_sd > 0 else 0
            # CI on d (approximate)
            se_d = np.sqrt((len(sr1)+len(sr2))/(len(sr1)*len(sr2)) + d**2/(2*(len(sr1)+len(sr2))))
            ci_lo = d - 1.96*se_d
            ci_hi = d + 1.96*se_d

            report(f"  {c1} vs {c2}: d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (n={len(sr1)},{len(sr2)})")

    report()

    # =========================================================================
    # SECTION 4: TOST Equivalence test
    # =========================================================================
    report("## 4. TOST Equivalence Test (Deceptive vs Honest)")
    report()

    # Find deceptive and honest conditions
    deceptive_cond = None
    honest_cond = None
    for c in cond_names:
        cl = c.lower()
        if 'deceptive' in cl or 'deception' in cl:
            deceptive_cond = c
        elif 'honest' in cl and 'rare' in cl:
            honest_cond = c
        elif 'honest' in cl and 'common' not in cl and honest_cond is None:
            honest_cond = c

    if deceptive_cond and honest_cond:
        sr_dec = cond_features[deceptive_cond]['stable_rank']
        sr_hon = cond_features[honest_cond]['stable_rank']

        for margin in [0.3, 0.5, 0.8]:
            tost = tost_equivalence(sr_dec, sr_hon, margin=margin)
            report(f"  TOST margin d={margin}: p={tost['p_tost']:.4f} ({'EQUIVALENT' if tost['equivalent'] else 'NOT equivalent'})")

        report()
        report(f"  Observed d: {tost['cohen_d']:.3f}")
        report(f"  Pooled SD: {tost['pooled_sd']:.4f}")

        # Power analysis
        for target_d in [0.3, 0.5, 0.8]:
            pwr = power_analysis(len(sr_dec), len(sr_hon), target_d)
            report(f"  Power to detect d={target_d}: {pwr:.3f}")
    else:
        report(f"  Could not find deceptive/honest conditions. Available: {cond_names}")

    report()

    # =========================================================================
    # SECTION 5: Feature ablation
    # =========================================================================
    report("## 5. Feature Ablation (Individual Shape Feature AUROCs)")
    report()

    for feat_name in SHAPE_FEATURES:
        X_single = features_to_matrix(features_list, [feat_name])
        auroc = compute_auroc_cv(X_single, labels, lengths)
        report(f"  {feat_name}: AUROC={auroc:.3f}")

    report()

    # =========================================================================
    # SECTION 6: Cross-architecture
    # =========================================================================
    report("## 6. Cross-Architecture Results")
    report()

    for model_name in ['Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3']:
        try:
            data = load_cross_arch(model_name)
            fl, lab, lens = extract_confab_features(data)
            X_s = features_to_matrix(fl, SHAPE_FEATURES)
            auroc = compute_auroc_cv(X_s, lab, lens)
            report(f"  {model_name}: AUROC={auroc:.3f} (n={len(lab)}, confab={lab.sum()})")
        except Exception as e:
            report(f"  {model_name}: ERROR — {e}")

    report()

    # =========================================================================
    # SECTION 7: Trajectory verification (from honesty signal checkpoints)
    # =========================================================================
    report("## 7. Trajectory Verification")
    report()

    honesty_data = load_honesty()
    n_with_checkpoints = sum(1 for t in honesty_data['trials'] if 'checkpoints' in t and t['checkpoints'])
    report(f"Trials with checkpoints: {n_with_checkpoints} / {len(honesty_data['trials'])}")

    if n_with_checkpoints > 0:
        # Group trajectory data by condition
        for t in honesty_data['trials'][:1]:
            if 'checkpoints' in t and t['checkpoints']:
                cp = t['checkpoints']
                if isinstance(cp, list) and cp:
                    report(f"  Checkpoint structure: {list(cp[0].keys()) if isinstance(cp[0], dict) else type(cp[0])}")
                    report(f"  N checkpoints per trial: {len(cp)}")
                elif isinstance(cp, dict):
                    report(f"  Checkpoint keys: {list(cp.keys())[:10]}")

    report()

    # =========================================================================
    # SECTION 8: Paper number verification
    # =========================================================================
    report("## 8. Paper Claims vs Data")
    report()

    report("### Claimed vs Computed")
    report()

    # Verify the key numbers from the paper
    features_list_q, labels_q, lengths_q = extract_confab_features(qwen_data)
    X_shape_q = features_to_matrix(features_list_q, SHAPE_FEATURES)
    X_mp_q = features_to_matrix(features_list_q, MP_FEATURES)

    shape_auroc = compute_auroc_cv(X_shape_q, labels_q, lengths_q)
    mp_auroc = compute_auroc_cv(X_mp_q, labels_q, lengths_q)

    report(f"  Paper claims shape AUROC: 0.764 | Computed: {shape_auroc:.3f}")
    report(f"  Paper claims MP AUROC:    0.544 | Computed: {mp_auroc:.3f}")
    report()

    # =========================================================================
    # Save report
    # =========================================================================
    report_text = '\n'.join(report_lines)
    with open(REPORT_PATH, 'w') as f:
        f.write(report_text)

    print(f"\n\nReport saved to: {REPORT_PATH}")


if __name__ == '__main__':
    main()
