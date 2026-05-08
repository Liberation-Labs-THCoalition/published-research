"""
Look for manifold signatures in the existing trajectory data.
No new experiments — just reanalyze the honesty signal checkpoints.
"""
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

DATA_DIR = Path(__file__).parent

with open(DATA_DIR / 'honesty_signal.json') as f:
    data = json.load(f)

FEATURES = ['stable_rank', 'sv_kurtosis', 'participation_ratio',
            'spectral_entropy', 'mp_signal_fraction', 'mp_norm_per_token']

# Extract trajectories as sequences of feature vectors
trajectories = {}
for t in data['trials']:
    cond = t['condition']
    cps = t.get('checkpoints', [])
    if len(cps) < 5:
        continue

    traj = []
    for cp in cps:
        feats = cp['features']
        vec = [feats.get(f, 0.0) for f in FEATURES]
        traj.append(vec)

    if cond not in trajectories:
        trajectories[cond] = []
    trajectories[cond].append(np.array(traj))

print("=" * 60)
print("MANIFOLD SIGNATURE ANALYSIS")
print("=" * 60)

# === TEST 1: Trajectory curvature ===
print("\n## 1. Trajectory Curvature")
print("   (deviation from straight line through feature space)\n")

for cond in ['confab', 'honest_common', 'honest_rare', 'deceptive_user', 'boundary']:
    trajs = trajectories.get(cond, [])
    if not trajs:
        continue

    curvatures = []
    for traj in trajs:
        # Normalize features to unit variance for fair distance comparison
        traj_norm = (traj - traj.mean(axis=0)) / (traj.std(axis=0) + 1e-10)

        # Fit straight line: from first to last point
        start = traj_norm[0]
        end = traj_norm[-1]
        direction = end - start
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            curvatures.append(0)
            continue

        direction = direction / dir_norm

        # Project each point onto the line, measure residual
        residuals = []
        for point in traj_norm:
            proj = start + np.dot(point - start, direction) * direction
            residual = np.linalg.norm(point - proj)
            residuals.append(residual)

        curvature = np.mean(residuals)
        curvatures.append(curvature)

    curvatures = np.array(curvatures)
    print(f"  {cond:20s}: mean={curvatures.mean():.4f} SD={curvatures.std(ddof=1):.4f} (n={len(curvatures)})")

# === TEST 2: Trajectory shape classification ===
print("\n## 2. Trajectory Shape (rise-peak-fall pattern)")
print("   Stable_rank trajectory: does it rise and stay, or rise and fall?\n")

for cond in ['confab', 'honest_common', 'honest_rare', 'deceptive_user', 'boundary']:
    trajs = trajectories.get(cond, [])
    if not trajs:
        continue

    shapes = {'rise_stay': 0, 'rise_fall': 0, 'flat': 0, 'other': 0}
    peak_positions = []
    decline_rates = []

    for traj in trajs:
        sr = traj[:, 0]  # stable_rank is first feature
        peak_idx = np.argmax(sr)
        peak_positions.append(peak_idx)

        if peak_idx == 0:
            shapes['flat'] += 1
        elif peak_idx >= len(sr) - 2:
            shapes['rise_stay'] += 1
        else:
            # Peak in the middle — check decline
            post_peak = sr[peak_idx:]
            decline = sr[peak_idx] - sr[-1]
            rise = sr[peak_idx] - sr[0]
            if rise < 0.02:
                shapes['flat'] += 1
            elif decline > rise * 0.3:
                shapes['rise_fall'] += 1
                decline_rates.append(decline / rise)
            else:
                shapes['rise_stay'] += 1

    total = sum(shapes.values())
    print(f"  {cond:20s}: rise_stay={shapes['rise_stay']:2d} rise_fall={shapes['rise_fall']:2d} "
          f"flat={shapes['flat']:2d} (n={total})")
    if decline_rates:
        print(f"  {'':20s}  decline/rise ratio: {np.mean(decline_rates):.3f}")

# === TEST 3: Endpoint distance matrix ===
print("\n## 3. Endpoint Distance Matrix (full feature space)")
print("   Euclidean distance between condition centroids at final checkpoint\n")

centroids = {}
for cond in ['confab', 'honest_common', 'honest_rare', 'deceptive_user', 'boundary']:
    trajs = trajectories.get(cond, [])
    if not trajs:
        continue
    endpoints = np.array([t[-1] for t in trajs])
    # Z-score using global stats
    centroids[cond] = endpoints.mean(axis=0)

cond_names = list(centroids.keys())
n_conds = len(cond_names)

# Compute pairwise distances
dist_matrix = np.zeros((n_conds, n_conds))
for i in range(n_conds):
    for j in range(n_conds):
        # Normalize by feature SDs
        all_endpoints = []
        for cond in cond_names:
            trajs = trajectories.get(cond, [])
            all_endpoints.extend([t[-1] for t in trajs])
        global_std = np.std(all_endpoints, axis=0) + 1e-10

        c1 = centroids[cond_names[i]] / global_std
        c2 = centroids[cond_names[j]] / global_std
        dist_matrix[i, j] = np.linalg.norm(c1 - c2)

# Print distance matrix
header = f"{'':20s}" + "".join(f"{c[:12]:>13s}" for c in cond_names)
print(header)
for i, c in enumerate(cond_names):
    row = f"{c:20s}" + "".join(f"{dist_matrix[i,j]:13.3f}" for j in range(n_conds))
    print(row)

# === TEST 4: Is the distance structure linear or curved? ===
print("\n## 4. Linearity Test")
print("   If conditions lie on a line, pairwise distances obey triangle equality.")
print("   If on a curved manifold, geodesic > Euclidean (detour required).\n")

# Check if honest_common is between confab and honest_rare
# or if the triangle is non-degenerate
d_confab_common = dist_matrix[cond_names.index('confab'), cond_names.index('honest_common')]
d_common_rare = dist_matrix[cond_names.index('honest_common'), cond_names.index('honest_rare')]
d_confab_rare = dist_matrix[cond_names.index('confab'), cond_names.index('honest_rare')]
d_confab_deceptive = dist_matrix[cond_names.index('confab'), cond_names.index('deceptive_user')]
d_rare_deceptive = dist_matrix[cond_names.index('honest_rare'), cond_names.index('deceptive_user')]

print(f"  confab -> common:     {d_confab_common:.3f}")
print(f"  common -> rare:       {d_common_rare:.3f}")
print(f"  confab -> rare:       {d_confab_rare:.3f}")
print(f"  sum of parts:        {d_confab_common + d_common_rare:.3f}")
print(f"  detour ratio:        {(d_confab_common + d_common_rare) / d_confab_rare:.3f}")
print(f"    (1.0 = linear, >1.0 = curved/detour)")
print()
print(f"  rare -> deceptive:    {d_rare_deceptive:.3f}")
print(f"  confab -> deceptive:  {d_confab_deceptive:.3f}")
print(f"    (rare ~= deceptive? {d_rare_deceptive:.3f} < {d_confab_rare:.3f})")

# === TEST 5: Per-checkpoint local dimensionality ===
print("\n## 5. Local Dimensionality Along Trajectory")
print("   stable_rank at each checkpoint, averaged by condition\n")

checkpoint_positions = None
for cond in ['confab', 'honest_common', 'honest_rare', 'deceptive_user']:
    trajs = trajectories.get(cond, [])
    if not trajs:
        continue

    sr_by_cp = []
    for traj in trajs:
        sr_by_cp.append(traj[:, 0])  # stable_rank

    sr_matrix = np.array(sr_by_cp)
    means = sr_matrix.mean(axis=0)

    # Rate of change between consecutive checkpoints
    diffs = np.diff(means)

    positions = list(range(10, 10 * len(means) + 1, 10))
    if checkpoint_positions is None:
        checkpoint_positions = positions

    print(f"  {cond:20s}: ", end="")
    print(" -> ".join(f"{m:.3f}" for m in means))
    print(f"  {'':20s}  delta: ", end="")
    print(" -> ".join(f"{d:+.3f}" for d in diffs))
    print()

print("\nDone.")
