"""
Experiment D: Manifold Mapping via Trajectory Embedding

Do cognitive-state trajectories lie on a low-dimensional manifold?
Direct measurement via UMAP, Isomap, PCA, and t-SNE embedding.

Uses existing honesty signal trajectory data (no model inference needed).
"""
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try UMAP — may not be installed
try:
    from umap import UMAP
    HAS_UMAP = True
except (ImportError, AttributeError, Exception) as e:
    HAS_UMAP = False
    print(f"UMAP not available ({type(e).__name__}) — skipping, using PCA/t-SNE/Isomap")

DATA_DIR = Path(__file__).parent
SEED = 42
rng = np.random.RandomState(SEED)

FEATURES = ['stable_rank', 'sv_kurtosis', 'participation_ratio',
            'spectral_entropy', 'mp_signal_fraction', 'mp_norm_per_token']

# ============================================================
# Load and prepare trajectory data
# ============================================================
print("Loading honesty signal data...")
with open(DATA_DIR / 'honesty_signal.json') as f:
    data = json.load(f)

trajectories = []
conditions = []
trajectory_vectors = []

for t in data['trials']:
    cps = t.get('checkpoints', [])
    if len(cps) < 5:
        continue

    traj = []
    for cp in cps:
        feats = cp['features']
        vec = [feats.get(f, 0.0) for f in FEATURES]
        traj.append(vec)

    traj = np.array(traj)
    trajectories.append(traj)
    conditions.append(t['condition'])

    # Flatten trajectory to single vector (10 checkpoints × 6 features = 60D)
    trajectory_vectors.append(traj.flatten())

X = np.array(trajectory_vectors)  # (n_trials, 60)
y = np.array(conditions)

print(f"Trajectories: {len(trajectories)}")
print(f"Conditions: {Counter(y)}")
print(f"Feature space: {X.shape[1]}D (10 checkpoints x {len(FEATURES)} features)")

# Z-score features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-10
X_z = (X - X_mean) / X_std

# ============================================================
# Embedding methods
# ============================================================
print("\n" + "=" * 60)
print("EMBEDDING COGNITIVE-STATE TRAJECTORIES")
print("=" * 60)

embeddings = {}

# PCA (linear baseline)
print("\n--- PCA (linear baseline) ---")
pca = PCA(n_components=3, random_state=SEED)
X_pca = pca.fit_transform(X_z)
embeddings['PCA'] = X_pca
print(f"  Variance explained: {pca.explained_variance_ratio_[:3]}")
print(f"  Total (3 components): {pca.explained_variance_ratio_[:3].sum():.3f}")

# Intrinsic dimensionality via PCA elbow
pca_full = PCA(random_state=SEED).fit(X_z)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
dim_90 = np.searchsorted(cumvar, 0.90) + 1
dim_95 = np.searchsorted(cumvar, 0.95) + 1
dim_99 = np.searchsorted(cumvar, 0.99) + 1
print(f"  Intrinsic dimensionality: 90%={dim_90}, 95%={dim_95}, 99%={dim_99}")

# t-SNE
print("\n--- t-SNE ---")
tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(X_z)-1))
X_tsne = tsne.fit_transform(X_z)
embeddings['t-SNE'] = X_tsne
print(f"  KL divergence: {tsne.kl_divergence_:.4f}")

# Isomap (manifold-native — preserves geodesic distances)
print("\n--- Isomap (geodesic-preserving) ---")
for n_neighbors in [5, 10, 15]:
    try:
        iso = Isomap(n_components=3, n_neighbors=n_neighbors)
        X_iso = iso.fit_transform(X_z)
        embeddings[f'Isomap(k={n_neighbors})'] = X_iso
        reconstruction_error = iso.reconstruction_error()
        print(f"  k={n_neighbors}: reconstruction error = {reconstruction_error:.4f}")
    except Exception as e:
        print(f"  k={n_neighbors}: failed — {e}")

# UMAP
if HAS_UMAP:
    print("\n--- UMAP ---")
    umap = UMAP(n_components=2, random_state=SEED, n_neighbors=15)
    X_umap = umap.fit_transform(X_z)
    embeddings['UMAP'] = X_umap

# ============================================================
# Condition separation analysis
# ============================================================
print("\n" + "=" * 60)
print("CONDITION SEPARATION IN EMBEDDED SPACE")
print("=" * 60)

COND_NAMES = ['confab', 'honest_common', 'honest_rare', 'deceptive_user', 'boundary']

def silhouette_by_condition(X_emb, labels, target_conditions=None):
    """Compute simplified silhouette-like separation score."""
    if target_conditions is None:
        target_conditions = sorted(set(labels))

    scores = {}
    for cond in target_conditions:
        mask = labels == cond
        if mask.sum() < 2:
            continue
        intra = pdist(X_emb[mask]).mean()  # mean within-condition distance
        inter_dists = []
        for other in target_conditions:
            if other == cond:
                continue
            other_mask = labels == other
            if other_mask.sum() < 1:
                continue
            from sklearn.metrics import pairwise_distances
            cross = pairwise_distances(X_emb[mask], X_emb[other_mask]).mean()
            inter_dists.append(cross)
        inter = np.mean(inter_dists) if inter_dists else 0
        scores[cond] = (inter - intra) / max(inter, intra) if max(inter, intra) > 0 else 0

    return scores

for method, X_emb in embeddings.items():
    X_2d = X_emb[:, :2] if X_emb.shape[1] > 2 else X_emb
    scores = silhouette_by_condition(X_2d, y, COND_NAMES)
    mean_score = np.mean(list(scores.values())) if scores else 0
    print(f"\n  {method}: mean separation = {mean_score:.3f}")
    for cond, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    {cond:20s}: {score:.3f}")

# ============================================================
# Key test: Is manifold-aware embedding BETTER than linear?
# ============================================================
print("\n" + "=" * 60)
print("MANIFOLD vs LINEAR: Does curved embedding beat PCA?")
print("=" * 60)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

print("\n  k-NN classification accuracy (5-fold CV) by embedding method:")
print(f"  (chance = {1/len(set(y)):.3f} for {len(set(y))} classes)")

for method, X_emb in embeddings.items():
    X_2d = X_emb[:, :2] if X_emb.shape[1] > 2 else X_emb
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X_2d, y, cv=5, scoring='accuracy')
    print(f"  {method:20s}: {scores.mean():.3f} +/- {scores.std():.3f}")

# Also test on 3D where available
print("\n  3D embeddings:")
for method, X_emb in embeddings.items():
    if X_emb.shape[1] >= 3:
        X_3d = X_emb[:, :3]
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, X_3d, y, cv=5, scoring='accuracy')
        print(f"  {method:20s}: {scores.mean():.3f} +/- {scores.std():.3f}")

# ============================================================
# Per-condition intrinsic dimensionality
# ============================================================
print("\n" + "=" * 60)
print("PER-CONDITION INTRINSIC DIMENSIONALITY")
print("=" * 60)

for cond in COND_NAMES:
    mask = y == cond
    if mask.sum() < 5:
        continue
    X_cond = X_z[mask]
    pca_cond = PCA(random_state=SEED).fit(X_cond)
    cumvar_cond = np.cumsum(pca_cond.explained_variance_ratio_)
    dim90 = np.searchsorted(cumvar_cond, 0.90) + 1
    dim95 = np.searchsorted(cumvar_cond, 0.95) + 1
    print(f"  {cond:20s}: 90%={dim90:2d}D, 95%={dim95:2d}D (n={mask.sum()})")

# ============================================================
# Permutation null: is separation real?
# ============================================================
print("\n" + "=" * 60)
print("PERMUTATION TEST: Is condition separation above chance?")
print("=" * 60)

# Use PCA 2D + kNN as the test statistic
knn = KNeighborsClassifier(n_neighbors=5)
X_pca_2d = X_pca[:, :2]
real_score = cross_val_score(knn, X_pca_2d, y, cv=5, scoring='accuracy').mean()

n_perm = 1000
null_scores = []
for _ in range(n_perm):
    y_shuffled = rng.permutation(y)
    null_score = cross_val_score(knn, X_pca_2d, y_shuffled, cv=5, scoring='accuracy').mean()
    null_scores.append(null_score)

null_scores = np.array(null_scores)
p_value = np.mean(null_scores >= real_score)
print(f"  Real kNN accuracy (PCA 2D): {real_score:.3f}")
print(f"  Null mean: {null_scores.mean():.3f} (SD={null_scores.std():.3f})")
print(f"  Null 95th percentile: {np.percentile(null_scores, 95):.3f}")
print(f"  p-value: {p_value:.4f}")

# ============================================================
# Generate figures
# ============================================================
print("\n" + "=" * 60)
print("GENERATING FIGURES")
print("=" * 60)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    COLORS = {
        'confab': '#F44336',
        'honest_common': '#2196F3',
        'honest_rare': '#4CAF50',
        'deceptive_user': '#FF9800',
        'boundary': '#9C27B0',
    }
    LABELS = {
        'confab': 'Confabulation',
        'honest_common': 'Honest (common)',
        'honest_rare': 'Honest (rare)',
        'deceptive_user': 'Deceptive',
        'boundary': 'Boundary',
    }

    # Pick best 2D methods
    methods_2d = ['PCA', 't-SNE']
    if HAS_UMAP:
        methods_2d.append('UMAP')
    # Add best Isomap
    iso_keys = [k for k in embeddings if 'Isomap' in k]
    if iso_keys:
        methods_2d.append(iso_keys[0])

    fig, axes = plt.subplots(1, len(methods_2d), figsize=(6*len(methods_2d), 5))
    if len(methods_2d) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods_2d):
        X_emb = embeddings[method]
        X_2d = X_emb[:, :2]
        for cond in COND_NAMES:
            mask = y == cond
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=COLORS.get(cond, '#666'),
                      label=LABELS.get(cond, cond),
                      alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
        ax.set_title(method, fontsize=13)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        if ax == axes[-1]:
            ax.legend(fontsize=8, loc='best')

    plt.suptitle('Cognitive-State Trajectories in Embedded Space', fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = DATA_DIR.parent / 'published-research-push' / 'delta-manifold-paper'
    fig.savefig(out_path / 'figure_manifold_embedding.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path / 'figure_manifold_embedding.pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved to {out_path / 'figure_manifold_embedding.png'}")

    # Also save locally
    fig.savefig(DATA_DIR / 'manifold_embedding.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Figures generated successfully.")
except Exception as e:
    print(f"  Figure generation failed: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT D SUMMARY")
print("=" * 60)

print(f"""
DATA: {len(trajectories)} trajectories, {len(set(y))} conditions
FEATURE SPACE: {X.shape[1]}D (10 checkpoints x {len(FEATURES)} features)
INTRINSIC DIMENSIONALITY: 90%={dim_90}D, 95%={dim_95}D, 99%={dim_99}D

CONDITION SEPARATION:
  Permutation test p={p_value:.4f} ({'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'})
  Real kNN accuracy: {real_score:.3f} vs null {null_scores.mean():.3f}

KEY QUESTION: Does manifold-aware embedding beat linear?
  If Isomap/UMAP >> PCA in classification accuracy, the structure
  is genuinely non-linear (curved manifold, not flat subspace).
  If PCA matches or beats them, linear structure is sufficient.

See figures for visual evidence.
""")

print("Done.")
