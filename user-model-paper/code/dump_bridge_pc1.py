#!/usr/bin/env python3
"""Dump per-emotion PC1 scores at L35 and F-ratio data for paper verification."""
import json, os, numpy as np

# Load full trial data
data_path = os.path.expanduser("~/KV-Experiments/results/emotion_geometry_bridge/emotion_bridge_trials.json")
with open(data_path) as f:
    data = json.load(f)
trials = data["trials"]
print("Loaded %d trials" % len(trials))

# Get unique emotions
emotions = sorted(set(t["emotion"] for t in trials))
print("Emotions:", len(emotions))

# Layer index mapping for full-attention layers
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]

# For bridge analysis: extract W_K projected emotion vectors
# The bridge_analysis in summary uses difference-in-means on story re-encoding residuals
# projected through W_K. We need the per-emotion mean features.

# Compute per-emotion mean generation features at each layer for F-ratio
print("\n=== F-RATIO ANALYSIS ===")
for li_idx, layer in enumerate(FULL_ATTN_LAYERS):
    # Gather per-emotion feature vectors
    emo_features = {}
    for t in trials:
        e = t["emotion"]
        gf = t["generation_features"]["per_layer"]
        feat = [gf["norms"][li_idx], gf["ranks"][li_idx], gf["entropies"][li_idx]]
        if e not in emo_features:
            emo_features[e] = []
        emo_features[e].append(feat)

    # Between-emotion variance vs within-emotion variance
    all_feats = np.array([f for fs in emo_features.values() for f in fs])
    grand_mean = np.mean(all_feats, axis=0)

    # Between: variance of emotion means around grand mean
    emo_means = np.array([np.mean(emo_features[e], axis=0) for e in emotions])
    between_var = np.mean(np.sum((emo_means - grand_mean)**2, axis=1))

    # Within: mean variance within each emotion
    within_vars = []
    for e in emotions:
        ef = np.array(emo_features[e])
        em = np.mean(ef, axis=0)
        within_vars.append(np.mean(np.sum((ef - em)**2, axis=1)))
    within_var = np.mean(within_vars)

    f_ratio = between_var / within_var if within_var > 0 else 0
    print("  L%2d: between=%.3f within=%.3f F=%.3f" % (layer, between_var, within_var, f_ratio))

# Now check if bridge_analysis has per-emotion PC1 scores
# We need to recompute this from the story re-encoding residuals
# The bridge analysis PCA is on difference-in-means vectors projected through W_K
# Per-emotion PC1 scores would require the actual projected vectors
print("\n=== BRIDGE PC1 SCORES AT L35 ===")
print("(These require the stored re-encoding residuals + W_K matrices)")
print("(Not available in trial-level JSON - stored in bridge summary only)")

# Check if the summary has emotion-level scores
summary_path = os.path.expanduser("~/KV-Experiments/results/emotion_geometry_bridge/emotion_bridge_summary.json")
with open(summary_path) as f:
    summary = json.load(f)

ba = summary["bridge_analysis"]
# Check if any layer has per-emotion data
if "35" in ba:
    layer_data = ba["35"]
    print("L35 keys:", list(layer_data.keys()))
    for k, v in layer_data.items():
        if isinstance(v, list) and len(v) == 30:
            print("  Found 30-element list:", k)
        elif isinstance(v, dict):
            print("  Dict:", k, "keys:", list(v.keys())[:5])

# Check for stored PCA components or scores
results_dir = os.path.expanduser("~/KV-Experiments/results/emotion_geometry_bridge/")
import glob
files = glob.glob(results_dir + "*.json")
for f in sorted(files):
    print("\n  File:", os.path.basename(f), "size:", os.path.getsize(f))
