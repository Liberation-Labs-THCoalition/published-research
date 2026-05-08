"""
Compute all TBDs for the delta/manifold paper:
1. Encoding-only baseline d
2. Bootstrap CIs on all major effect sizes
3. Multiple comparison correction on per-layer analysis
4. Bootstrap CI on detour ratio
5. All 10 triplet detour ratios
6. Chi-squared on trajectory shape contingency table
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

DATA_DIR = Path(__file__).parent
N_BOOT = 5000
SEED = 42
rng = np.random.RandomState(SEED)

with open(DATA_DIR / 'honesty_signal.json') as f:
    hs = json.load(f)

# Group by condition
conditions = {}
for t in hs['trials']:
    cond = t['condition']
    if cond not in conditions:
        conditions[cond] = []
    conditions[cond].append(t)

FEATURES = ['stable_rank', 'sv_kurtosis', 'participation_ratio',
            'spectral_entropy', 'mp_signal_fraction', 'mp_norm_per_token']


def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0


def bootstrap_d(a, b, n_boot=N_BOOT):
    ds = []
    for _ in range(n_boot):
        ai = rng.choice(len(a), len(a), replace=True)
        bi = rng.choice(len(b), len(b), replace=True)
        ds.append(cohen_d(a[ai], b[bi]))
    ds = np.array(ds)
    return np.percentile(ds, 2.5), np.percentile(ds, 97.5)


# ============================================================
print("=" * 60)
print("1. ENCODING-ONLY BASELINE")
print("=" * 60)

for c1, c2, label in [('honest_rare', 'confab', 'hr vs confab'),
                        ('honest_common', 'confab', 'hc vs confab'),
                        ('deceptive_user', 'confab', 'dec vs confab')]:
    enc1 = np.array([t['encoding_features']['stable_rank'] for t in conditions[c1]])
    enc2 = np.array([t['encoding_features']['stable_rank'] for t in conditions[c2]])
    d = cohen_d(enc1, enc2)
    ci = bootstrap_d(enc1, enc2)
    t_stat, p_val = stats.ttest_ind(enc1, enc2)
    print(f"  Encoding sr {label}: d={d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] p={p_val:.4f}")

# ============================================================
print("\n" + "=" * 60)
print("2. BOOTSTRAP CIs ON ALL MAJOR EFFECT SIZES")
print("=" * 60)

# Delta stable_rank
hr_delta = np.array([t['generation_features']['stable_rank'] - t['encoding_features']['stable_rank']
                     for t in conditions['honest_rare']])
cf_delta = np.array([t['generation_features']['stable_rank'] - t['encoding_features']['stable_rank']
                     for t in conditions['confab']])
d = cohen_d(hr_delta, cf_delta)
ci = bootstrap_d(hr_delta, cf_delta)
print(f"\n  Delta sr (hr vs confab): d={d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")

# Endpoint stable_rank
hr_gen = np.array([t['generation_features']['stable_rank'] for t in conditions['honest_rare']])
cf_gen = np.array([t['generation_features']['stable_rank'] for t in conditions['confab']])
d = cohen_d(hr_gen, cf_gen)
ci = bootstrap_d(hr_gen, cf_gen)
print(f"  Endpoint sr (hr vs confab): d={d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")

# All condition deltas
print("\n  Per-condition delta CIs:")
for cond in ['honest_common', 'honest_rare', 'confab', 'deceptive_user', 'boundary']:
    trials = conditions[cond]
    deltas = np.array([t['generation_features']['stable_rank'] - t['encoding_features']['stable_rank']
                       for t in trials])
    boot_means = [rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"    {cond:20s}: delta={deltas.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

# ============================================================
print("\n" + "=" * 60)
print("3. MULTIPLE COMPARISON CORRECTION ON PER-LAYER ANALYSIS")
print("=" * 60)

hr_trials = [t for t in hs['trials'] if t['condition'] == 'honest_rare']
cf_trials = [t for t in hs['trials'] if t['condition'] == 'confab']
n_layers = len(hr_trials[0]['per_layer'])

layer_ds = []
layer_ps = []
for j in range(n_layers):
    hr_vals = np.array([t['per_layer'][j]['stable_rank'] for t in hr_trials])
    cf_vals = np.array([t['per_layer'][j]['stable_rank'] for t in cf_trials])
    d = cohen_d(hr_vals, cf_vals)
    _, p = stats.ttest_ind(hr_vals, cf_vals)
    layer_ds.append(d)
    layer_ps.append(p)

# Bonferroni correction
bonf_ps = np.array(layer_ps) * n_layers
bonf_ps = np.minimum(bonf_ps, 1.0)

# Benjamini-Hochberg FDR
sorted_idx = np.argsort(layer_ps)
fdr_ps = np.zeros(n_layers)
for rank, idx in enumerate(sorted_idx):
    fdr_ps[idx] = layer_ps[idx] * n_layers / (rank + 1)
for i in range(n_layers - 2, -1, -1):
    idx = sorted_idx[i]
    next_idx = sorted_idx[i + 1]
    fdr_ps[idx] = min(fdr_ps[idx], fdr_ps[next_idx])
fdr_ps = np.minimum(fdr_ps, 1.0)

print(f"\n  Per-layer stable_rank d (hr vs confab), {n_layers} layers:")
print(f"  {'Layer':>6s} {'d':>8s} {'p_raw':>10s} {'p_bonf':>10s} {'p_fdr':>10s} {'sig':>6s}")
for j in range(n_layers):
    sig = '***' if bonf_ps[j] < 0.001 else '**' if bonf_ps[j] < 0.01 else '*' if bonf_ps[j] < 0.05 else 'ns'
    print(f"  L{j:>4d} {layer_ds[j]:>+8.3f} {layer_ps[j]:>10.6f} {bonf_ps[j]:>10.6f} {fdr_ps[j]:>10.6f} {sig:>6s}")

# Report corrected peak
peak_idx = np.argmax(np.abs(layer_ds))
print(f"\n  Peak: L{peak_idx} d={layer_ds[peak_idx]:+.3f}")
print(f"    Raw p: {layer_ps[peak_idx]:.6f}")
print(f"    Bonferroni p: {bonf_ps[peak_idx]:.6f}")
print(f"    FDR p: {fdr_ps[peak_idx]:.6f}")
print(f"    Survives Bonferroni: {bonf_ps[peak_idx] < 0.05}")

n_sig_bonf = sum(1 for p in bonf_ps if p < 0.05)
n_sig_fdr = sum(1 for p in fdr_ps if p < 0.05)
print(f"\n  Layers significant after Bonferroni: {n_sig_bonf}/{n_layers}")
print(f"  Layers significant after FDR: {n_sig_fdr}/{n_layers}")

# ============================================================
print("\n" + "=" * 60)
print("4. BOOTSTRAP CI ON DETOUR RATIO + ALL 10 TRIPLETS")
print("=" * 60)

# Compute centroids from 6-feature endpoints
def get_centroid(cond_name):
    trials = conditions[cond_name]
    vecs = []
    for t in trials:
        gen = t['generation_features']
        vec = [gen.get(f, 0) for f in FEATURES]
        vecs.append(vec)
    return np.array(vecs)

cond_names = ['confab', 'honest_common', 'honest_rare', 'deceptive_user', 'boundary']
cond_vecs = {c: get_centroid(c) for c in cond_names}

# Z-score using global stats
all_vecs = np.vstack([v for v in cond_vecs.values()])
global_mean = all_vecs.mean(axis=0)
global_std = all_vecs.std(axis=0) + 1e-10
cond_vecs_z = {c: (v - global_mean) / global_std for c, v in cond_vecs.items()}

def compute_detour(c1, c2, c3, vecs):
    """Detour ratio for path c1->c2->c3 vs direct c1->c3."""
    d12 = np.linalg.norm(vecs[c1].mean(axis=0) - vecs[c2].mean(axis=0))
    d23 = np.linalg.norm(vecs[c2].mean(axis=0) - vecs[c3].mean(axis=0))
    d13 = np.linalg.norm(vecs[c1].mean(axis=0) - vecs[c3].mean(axis=0))
    return (d12 + d23) / d13 if d13 > 0 else float('inf')

# All 10 triplets (ordered as A->B->C where B is the "middle" point)
print("\n  All triplet detour ratios:")
all_triplets = []
for c1, c2, c3 in combinations(cond_names, 3):
    # Try all 3 orderings, report the one where c2 is "between"
    for a, b, c in [(c1, c2, c3), (c1, c3, c2), (c2, c1, c3)]:
        r = compute_detour(a, b, c, cond_vecs_z)
        all_triplets.append((a, b, c, r))

# Sort by detour ratio
all_triplets.sort(key=lambda x: x[3], reverse=True)
for a, b, c, r in all_triplets[:15]:
    print(f"    {a[:8]:>8s} -> {b[:8]:>8s} -> {c[:8]:>8s}: {r:.3f}")

# Bootstrap CI on the main triplet (confab -> common -> rare)
print("\n  Bootstrap CI on confab->common->rare detour ratio:")
boot_detours = []
for _ in range(N_BOOT):
    boot_vecs = {}
    for c in cond_names:
        idx = rng.choice(len(cond_vecs_z[c]), len(cond_vecs_z[c]), replace=True)
        boot_vecs[c] = cond_vecs_z[c][idx]
    boot_detours.append(compute_detour('confab', 'honest_common', 'honest_rare', boot_vecs))

boot_detours = np.array(boot_detours)
boot_detours = boot_detours[np.isfinite(boot_detours)]
print(f"    Point: {compute_detour('confab', 'honest_common', 'honest_rare', cond_vecs_z):.3f}")
print(f"    95% CI: [{np.percentile(boot_detours, 2.5):.3f}, {np.percentile(boot_detours, 97.5):.3f}]")
print(f"    % > 1.0: {100 * np.mean(boot_detours > 1.0):.1f}%")

# ============================================================
print("\n" + "=" * 60)
print("5. CHI-SQUARED ON TRAJECTORY SHAPE")
print("=" * 60)

# Contingency table: condition x shape (rise-stay vs rise-fall)
# From our manifold analysis
table = np.array([
    [9, 19],   # confab: rise-stay, rise-fall (2 flat excluded)
    [9, 21],   # honest_common
    [21, 9],   # honest_rare
    [18, 12],  # deceptive
])
chi2, p, dof, expected = stats.chi2_contingency(table)
print(f"  Chi-squared: {chi2:.2f}, df={dof}, p={p:.6f}")
print(f"  Significant: {p < 0.05}")

# Confab vs honest-rare specifically
table_2x2 = np.array([[9, 19], [21, 9]])
chi2_2, p_2, _, _ = stats.chi2_contingency(table_2x2)
print(f"\n  Confab vs honest-rare (2x2):")
print(f"  Chi-squared: {chi2_2:.2f}, p={p_2:.6f}")

print("\nDone.")
