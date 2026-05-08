"""
Verify CC's headline findings from the honesty signal + cognitive state data.

CC claims:
1. Delta stable_rank: d=2.39*** (honest_rare vs confab)
2. Honest delta: 0.44, Confab delta: 0.03
3. Llama layer 29: d=-5.20*** (spectral entropy)
4. Expansion-compression cancels under layer averaging
5. Crossover at ~50-55% depth
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

DATA_DIR = Path(__file__).parent

# ============================================================
# CLAIM 1 & 2: Delta stable_rank (honest_rare vs confab)
# ============================================================
print("=" * 60)
print("CLAIM 1: Delta stable_rank d=2.39 (honest_rare vs confab)")
print("CLAIM 2: Honest delta=0.44, Confab delta=0.03")
print("=" * 60)

with open(DATA_DIR / 'honesty_signal.json') as f:
    hs = json.load(f)

conditions = {}
for t in hs['trials']:
    cond = t['condition']
    if cond not in conditions:
        conditions[cond] = []

    enc_sr = t.get('encoding_features', {}).get('stable_rank', None)
    gen_sr = t.get('generation_features', {}).get('stable_rank', None)

    if enc_sr is not None and gen_sr is not None:
        delta = gen_sr - enc_sr
        conditions[cond].append({
            'enc_sr': enc_sr,
            'gen_sr': gen_sr,
            'delta_sr': delta,
        })

print()
for cond in ['honest_common', 'honest_rare', 'confab', 'deceptive_user', 'boundary']:
    trials = conditions.get(cond, [])
    if not trials:
        continue
    deltas = np.array([t['delta_sr'] for t in trials])
    enc = np.array([t['enc_sr'] for t in trials])
    gen = np.array([t['gen_sr'] for t in trials])
    print(f"{cond:20s}: n={len(trials):2d}  enc={enc.mean():.3f}  gen={gen.mean():.3f}  "
          f"delta={deltas.mean():.3f} (SD={deltas.std(ddof=1):.3f})")

# Compute d for honest_rare vs confab delta
hr_deltas = np.array([t['delta_sr'] for t in conditions.get('honest_rare', [])])
cf_deltas = np.array([t['delta_sr'] for t in conditions.get('confab', [])])

pooled_sd = np.sqrt(((len(hr_deltas)-1)*hr_deltas.var(ddof=1) +
                      (len(cf_deltas)-1)*cf_deltas.var(ddof=1)) /
                     (len(hr_deltas)+len(cf_deltas)-2))
d = (hr_deltas.mean() - cf_deltas.mean()) / pooled_sd
t_stat, p_val = stats.ttest_ind(hr_deltas, cf_deltas)

print(f"\nDelta stable_rank comparison (honest_rare vs confab):")
print(f"  CC claims: d=2.39")
print(f"  Computed:  d={d:.2f}")
print(f"  CC claims: honest delta=0.44, confab delta=0.03")
print(f"  Computed:  honest_rare delta={hr_deltas.mean():.3f}, confab delta={cf_deltas.mean():.3f}")
print(f"  t={t_stat:.3f}, p={p_val:.6f}")

# ============================================================
# CLAIM 3 & 4: Per-layer analysis on all architectures
# ============================================================
print("\n" + "=" * 60)
print("CLAIM 3: Llama layer 29 spectral entropy d=-5.20")
print("CLAIM 4: Expansion-compression cancels under averaging")
print("=" * 60)

for model_name in ['Qwen2.5-7B-Instruct', 'Llama-3.1-8B-Instruct',
                    'Mistral-7B-Instruct-v0.3']:
    fname = DATA_DIR / f'{model_name}_cognitive_states.json'
    if not fname.exists():
        print(f"\n{model_name}: FILE NOT FOUND")
        continue

    with open(fname) as f:
        data = json.load(f)

    # These files have flat features (all-layer mean), not per-layer
    # Check if per-layer data exists
    if 'results' in data:
        trials = data['results'].get('confab', [])
        if trials:
            t0 = trials[0]
            has_per_layer = 'per_layer' in t0
            feat_keys = list(t0.get('features', {}).keys())[:8]
            print(f"\n{model_name}:")
            print(f"  Has per_layer: {has_per_layer}")
            print(f"  Flat feature keys: {feat_keys}")

# Check honesty signal for per-layer data
print("\n--- Honesty signal per-layer check ---")
t0 = hs['trials'][0]
per_layer = t0.get('per_layer', [])
print(f"Per-layer data: {len(per_layer)} layers")
if per_layer:
    print(f"Layer keys: {list(per_layer[0].keys())[:10]}")
    has_entropy = 'spectral_entropy' in per_layer[0]
    print(f"Has spectral_entropy: {has_entropy}")

# ============================================================
# CLAIM 4 VERIFICATION: Expansion-compression from honesty data
# ============================================================
if per_layer and 'spectral_entropy' in per_layer[0]:
    print("\n" + "=" * 60)
    print("EXPANSION-COMPRESSION CYCLE (Qwen, honesty signal)")
    print("=" * 60)

    # Group per-layer spectral entropy by condition
    for cond in ['honest_rare', 'confab']:
        cond_trials = [t for t in hs['trials'] if t['condition'] == cond]
        n_layers = len(cond_trials[0]['per_layer'])

        entropy_by_layer = np.zeros((len(cond_trials), n_layers))
        sr_by_layer = np.zeros((len(cond_trials), n_layers))

        for i, t in enumerate(cond_trials):
            for j, layer in enumerate(t['per_layer']):
                entropy_by_layer[i, j] = layer.get('spectral_entropy', 0)
                sr_by_layer[i, j] = layer.get('stable_rank', 0)

        print(f"\n{cond} (n={len(cond_trials)}):")
        print(f"  {'Layer':>6s}  {'entropy':>10s}  {'stable_rank':>12s}")
        for j in range(n_layers):
            e_mean = entropy_by_layer[:, j].mean()
            sr_mean = sr_by_layer[:, j].mean()
            print(f"  L{j:>4d}  {e_mean:>10.4f}  {sr_mean:>12.4f}")

    # Find crossover point
    print("\n--- Crossover analysis ---")
    hr_trials = [t for t in hs['trials'] if t['condition'] == 'honest_rare']
    cf_trials = [t for t in hs['trials'] if t['condition'] == 'confab']
    n_layers = len(hr_trials[0]['per_layer'])

    for feature in ['stable_rank', 'spectral_entropy']:
        print(f"\n{feature} per-layer d (honest_rare vs confab):")
        for j in range(n_layers):
            hr_vals = np.array([t['per_layer'][j].get(feature, 0) for t in hr_trials])
            cf_vals = np.array([t['per_layer'][j].get(feature, 0) for t in cf_trials])
            if hr_vals.std() > 0 and cf_vals.std() > 0:
                pooled = np.sqrt(((len(hr_vals)-1)*hr_vals.var(ddof=1) +
                                  (len(cf_vals)-1)*cf_vals.var(ddof=1)) /
                                 (len(hr_vals)+len(cf_vals)-2))
                d = (hr_vals.mean() - cf_vals.mean()) / pooled if pooled > 0 else 0
                print(f"  L{j:2d}: d={d:+.3f}  (hr={hr_vals.mean():.4f} cf={cf_vals.mean():.4f})")

print("\nDone.")
