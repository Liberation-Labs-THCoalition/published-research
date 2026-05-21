"""
Factor Analysis of 171 Functional Emotion Vectors
Builds sparse feature matrix from SAE activations, performs SVD/PCA,
hierarchical clustering, and redundancy analysis.
"""

import json
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────
with open(r"C:\Users\Thomas\Desktop\Project-Oracle\probes\emotion_map_171.json", "r") as f:
    emotion_map = json.load(f)

emotions = sorted(emotion_map.keys())
n_emotions = len(emotions)
print(f"Loaded {n_emotions} emotions")

# ── Build sparse feature matrix ───────────────────────────────────────────
# Collect all unique feature indices
all_features = set()
for emo_data in emotion_map.values():
    for feat_idx, _ in emo_data["top5"]:
        all_features.add(int(feat_idx))
    for feat_idx, _ in emo_data["anti3"]:
        all_features.add(int(feat_idx))

all_features = sorted(all_features)
feat_to_col = {f: i for i, f in enumerate(all_features)}
n_features = len(all_features)
print(f"Unique SAE features: {n_features}")
print(f"Matrix shape: {n_emotions} x {n_features}")

# Build the matrix (using both top5 positive and anti3 negative activations)
X = np.zeros((n_emotions, n_features), dtype=np.float64)
for i, emo in enumerate(emotions):
    data = emotion_map[emo]
    for feat_idx, val in data["top5"]:
        X[i, feat_to_col[int(feat_idx)]] = val
    for feat_idx, val in data["anti3"]:
        X[i, feat_to_col[int(feat_idx)]] = val  # These are already negative

print(f"\nMatrix density: {np.count_nonzero(X) / X.size:.4f}")
print(f"Non-zero entries: {np.count_nonzero(X)} / {X.size}")

# ── SVD / PCA ─────────────────────────────────────────────────────────────
pca = PCA()
X_pca = pca.fit_transform(X)

cumvar = np.cumsum(pca.explained_variance_ratio_)
n_80 = np.searchsorted(cumvar, 0.80) + 1
n_90 = np.searchsorted(cumvar, 0.90) + 1
n_95 = np.searchsorted(cumvar, 0.95) + 1

print("\n" + "="*60)
print("SVD / PCA RESULTS")
print("="*60)
print(f"Components for 80% variance: {n_80}")
print(f"Components for 90% variance: {n_90}")
print(f"Components for 95% variance: {n_95}")
print(f"Total possible components: {min(n_emotions, n_features)}")

# Scree analysis - find inflection point via second derivative
var_ratios = pca.explained_variance_ratio_
# Look at first 50 components
n_show = min(50, len(var_ratios))
diffs = np.diff(var_ratios[:n_show])
diffs2 = np.diff(diffs)
# Inflection = where second derivative is most positive (elbow)
inflection = np.argmax(diffs2) + 2  # +2 because of double diff offset

print(f"\nScree inflection point: ~PC{inflection}")
print(f"\nTop 10 components (variance explained):")
for i in range(min(10, len(var_ratios))):
    print(f"  PC{i+1}: {var_ratios[i]*100:.2f}% (cumulative: {cumvar[i]*100:.2f}%)")

# ── Top-loading emotions per PC ──────────────────────────────────────────
print("\n" + "="*60)
print("TOP 5 PRINCIPAL COMPONENTS - Strongest Loading Emotions")
print("="*60)

top_pcs = {}
for pc_idx in range(5):
    scores = X_pca[:, pc_idx]
    # Top positive loadings
    pos_idx = np.argsort(scores)[-8:][::-1]
    neg_idx = np.argsort(scores)[:8]

    pc_info = {
        "variance_explained": float(var_ratios[pc_idx]),
        "cumulative_variance": float(cumvar[pc_idx]),
        "top_positive": [(emotions[j], float(scores[j])) for j in pos_idx],
        "top_negative": [(emotions[j], float(scores[j])) for j in neg_idx],
    }
    top_pcs[f"PC{pc_idx+1}"] = pc_info

    print(f"\nPC{pc_idx+1} ({var_ratios[pc_idx]*100:.2f}% variance):")
    print(f"  + pole: {', '.join(emotions[j] for j in pos_idx[:5])}")
    print(f"  - pole: {', '.join(emotions[j] for j in neg_idx[:5])}")

# ── Hierarchical Clustering ──────────────────────────────────────────────
print("\n" + "="*60)
print("HIERARCHICAL CLUSTERING (Ward's method, 20 PCs)")
print("="*60)

# Use top 20 PCs for clustering
n_pcs_cluster = min(20, n_90)  # Use at least enough for 90% variance
X_reduced = X_pca[:, :n_pcs_cluster]

# Ward's linkage
Z = linkage(X_reduced, method='ward')

# Try different cluster counts and pick one that gives good granularity
# Aim for ~20 clusters
n_clusters_target = 20
labels = fcluster(Z, t=n_clusters_target, criterion='maxclust')

# Build cluster membership
clusters = defaultdict(list)
for i, label in enumerate(labels):
    clusters[int(label)].append(emotions[i])

# Sort clusters by size
cluster_sorted = sorted(clusters.items(), key=lambda x: -len(x[1]))

# Assign human-readable labels based on content
def suggest_label(members):
    """Heuristic labeling based on emotion semantics."""
    m = set(members)

    # Check for known groupings
    if m & {"hostile", "desperate", "gloomy"}:
        if len(m & {"hostile", "desperate", "gloomy"}) >= 2:
            return "ANTISOCIAL/DARK (CC's confab-correction triad)"

    # Valence-based heuristics
    negative_markers = {"angry", "furious", "enraged", "hostile", "resentful", "bitter", "hateful"}
    fear_markers = {"afraid", "terrified", "panicked", "alarmed", "anxious", "nervous", "worried"}
    sad_markers = {"sad", "depressed", "melancholic", "gloomy", "grief-stricken", "sorrowful", "despairing"}
    joy_markers = {"happy", "joyful", "elated", "ecstatic", "cheerful", "delighted", "euphoric"}
    disgust_markers = {"disgusted", "repulsed", "revolted", "contemptuous", "scornful"}
    surprise_markers = {"surprised", "amazed", "astonished", "shocked", "stunned"}
    social_markers = {"grateful", "appreciative", "compassionate", "empathetic", "loving", "affectionate"}
    power_markers = {"confident", "determined", "bold", "assertive", "dominant", "powerful"}
    low_energy = {"bored", "apathetic", "indifferent", "weary", "tired", "exhausted", "lethargic"}

    if len(m & negative_markers) >= 2: return "Anger/Hostility"
    if len(m & fear_markers) >= 2: return "Fear/Alarm"
    if len(m & sad_markers) >= 2: return "Sadness/Grief"
    if len(m & joy_markers) >= 2: return "Joy/Elation"
    if len(m & disgust_markers) >= 2: return "Disgust/Contempt"
    if len(m & surprise_markers) >= 2: return "Surprise/Wonder"
    if len(m & social_markers) >= 2: return "Prosocial/Warmth"
    if len(m & power_markers) >= 2: return "Power/Confidence"
    if len(m & low_energy) >= 2: return "Low Arousal/Apathy"

    # Fallback: use first few members
    return f"Mixed ({', '.join(sorted(members)[:3])}...)"

print(f"\nUsing {n_pcs_cluster} PCs for clustering")
print(f"Number of clusters: {len(clusters)}")

cluster_assignments = {}
cluster_details = []
for cid, members in cluster_sorted:
    label = suggest_label(members)
    for m in members:
        cluster_assignments[m] = {"cluster_id": cid, "label": label}

    detail = {"cluster_id": cid, "label": label, "size": len(members), "members": sorted(members)}
    cluster_details.append(detail)
    print(f"\n  Cluster {cid} ({label}) [{len(members)} emotions]:")
    print(f"    {', '.join(sorted(members))}")

# ── Antisocial Triad Analysis ────────────────────────────────────────────
print("\n" + "="*60)
print("ANTISOCIAL TRIAD ANALYSIS (hostile, desperate, gloomy)")
print("="*60)

triad = ["hostile", "desperate", "gloomy"]
triad_present = [e for e in triad if e in emotions]
triad_missing = [e for e in triad if e not in emotions]

if triad_missing:
    print(f"WARNING: Missing from emotion map: {triad_missing}")
    # Check for close matches
    for missing in triad_missing:
        close = [e for e in emotions if missing[:4] in e]
        if close:
            print(f"  Possible matches for '{missing}': {close}")

if triad_present:
    # Check if they cluster together
    triad_clusters = {e: int(labels[emotions.index(e)]) for e in triad_present}
    print(f"\nCluster assignments:")
    for e, c in triad_clusters.items():
        print(f"  {e}: Cluster {c} ({suggest_label(clusters[c])})")

    same_cluster = len(set(triad_clusters.values())) == 1
    print(f"\nAll in same cluster: {same_cluster}")

    if same_cluster:
        cid = list(triad_clusters.values())[0]
        co_members = [e for e in clusters[cid] if e not in triad]
        print(f"Co-clustered emotions: {', '.join(sorted(co_members))}")
    else:
        # Check pairwise distances
        print("\nPairwise cosine similarities within triad:")
        for i, e1 in enumerate(triad_present):
            for e2 in triad_present[i+1:]:
                idx1, idx2 = emotions.index(e1), emotions.index(e2)
                v1, v2 = X[idx1], X[idx2]
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                print(f"  {e1} <-> {e2}: {cos_sim:.4f}")

        # What are they near?
        print("\nNearest neighbors for each triad member:")
        cos_dist = squareform(pdist(X, metric='cosine'))
        for e in triad_present:
            idx = emotions.index(e)
            dists = cos_dist[idx]
            nearest = np.argsort(dists)[1:6]  # skip self
            print(f"  {e}: {', '.join(f'{emotions[j]} ({1-dists[j]:.3f})' for j in nearest)}")

# ── Redundancy Analysis ──────────────────────────────────────────────────
print("\n" + "="*60)
print("REDUNDANCY ANALYSIS (cosine similarity > 0.9)")
print("="*60)

# Compute full cosine similarity matrix
cos_dist = squareform(pdist(X, metric='cosine'))
cos_sim = 1 - cos_dist

redundant_pairs = []
for i in range(n_emotions):
    for j in range(i+1, n_emotions):
        if cos_sim[i, j] > 0.9:
            redundant_pairs.append({
                "emotion1": emotions[i],
                "emotion2": emotions[j],
                "cosine_similarity": float(cos_sim[i, j])
            })

redundant_pairs.sort(key=lambda x: -x["cosine_similarity"])

print(f"Pairs with cosine similarity > 0.9: {len(redundant_pairs)}")
print(f"Pairs with cosine similarity > 0.95: {sum(1 for p in redundant_pairs if p['cosine_similarity'] > 0.95)}")
print(f"Pairs with cosine similarity > 0.99: {sum(1 for p in redundant_pairs if p['cosine_similarity'] > 0.99)}")

print(f"\nTop 20 most redundant pairs:")
for p in redundant_pairs[:20]:
    print(f"  {p['emotion1']} <-> {p['emotion2']}: {p['cosine_similarity']:.4f}")

# ── Also check: what are the most UNIQUE emotions? ──────────────────────
print("\n" + "="*60)
print("MOST UNIQUE EMOTIONS (lowest max similarity to any other)")
print("="*60)

max_sim_per_emotion = []
for i in range(n_emotions):
    sims = [cos_sim[i, j] for j in range(n_emotions) if j != i]
    max_sim_per_emotion.append((emotions[i], max(sims), np.mean(sims)))

max_sim_per_emotion.sort(key=lambda x: x[1])  # sort by max similarity
print("Emotions most distinct from all others (lowest max-cosine to any neighbor):")
for emo, ms, avg_s in max_sim_per_emotion[:15]:
    print(f"  {emo}: max_sim={ms:.4f}, avg_sim={avg_s:.4f}")

# ── Additional: Plutchik mapping ─────────────────────────────────────────
print("\n" + "="*60)
print("PLUTCHIK WHEEL MAPPING")
print("="*60)

plutchik_primaries = {
    "Joy": ["happy", "joyful", "elated", "ecstatic", "cheerful", "delighted"],
    "Trust": ["trusting", "grateful", "appreciative", "admiring"],
    "Fear": ["afraid", "terrified", "panicked", "fearful", "anxious"],
    "Surprise": ["surprised", "amazed", "astonished", "shocked"],
    "Sadness": ["sad", "sorrowful", "melancholic", "grief-stricken", "depressed"],
    "Disgust": ["disgusted", "repulsed", "revolted"],
    "Anger": ["angry", "furious", "enraged", "hostile", "resentful"],
    "Anticipation": ["anticipating", "eager", "excited", "expectant", "hopeful"]
}

print("Do Plutchik primary emotions cluster together?")
for category, exemplars in plutchik_primaries.items():
    present = [e for e in exemplars if e in emotions]
    if len(present) >= 2:
        cluster_ids = [int(labels[emotions.index(e)]) for e in present]
        unique_clusters = set(cluster_ids)
        coherence = 1.0 - (len(unique_clusters) - 1) / max(len(present) - 1, 1)
        print(f"  {category}: {len(present)} emotions in {len(unique_clusters)} cluster(s) (coherence={coherence:.2f})")
        if len(unique_clusters) > 1:
            for e in present:
                c = int(labels[emotions.index(e)])
                print(f"    {e} -> Cluster {c}")

# ── Save full results ─────────────────────────────────────────────────────
results = {
    "metadata": {
        "n_emotions": n_emotions,
        "n_features": n_features,
        "matrix_density": float(np.count_nonzero(X) / X.size),
        "features_per_emotion": 8,  # 5 top + 3 anti
    },
    "pca": {
        "n_components_80pct": int(n_80),
        "n_components_90pct": int(n_90),
        "n_components_95pct": int(n_95),
        "scree_inflection": int(inflection),
        "variance_explained_top10": [float(v) for v in var_ratios[:10]],
        "cumulative_variance_top10": [float(v) for v in cumvar[:10]],
    },
    "top_5_pcs": top_pcs,
    "clustering": {
        "method": "Ward's linkage",
        "n_pcs_used": n_pcs_cluster,
        "n_clusters": len(clusters),
        "clusters": cluster_details,
    },
    "cluster_assignments": cluster_assignments,
    "antisocial_triad": {
        "emotions": triad,
        "present_in_map": triad_present,
        "missing_from_map": triad_missing,
        "cluster_assignments": {e: int(labels[emotions.index(e)]) for e in triad_present} if triad_present else {},
        "same_cluster": same_cluster if triad_present else None,
    },
    "redundancy": {
        "n_pairs_above_0.9": len(redundant_pairs),
        "n_pairs_above_0.95": sum(1 for p in redundant_pairs if p["cosine_similarity"] > 0.95),
        "n_pairs_above_0.99": sum(1 for p in redundant_pairs if p["cosine_similarity"] > 0.99),
        "pairs": redundant_pairs,
    },
    "uniqueness_ranking": [
        {"emotion": emo, "max_similarity": float(ms), "avg_similarity": float(avgs)}
        for emo, ms, avgs in max_sim_per_emotion
    ]
}

output_path = r"C:\Users\Thomas\Desktop\emotion_factor_analysis.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\nFull results saved to: {output_path}")
print(f"JSON file size: {len(json.dumps(results))} bytes")
