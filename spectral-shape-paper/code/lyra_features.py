"""
lyra_features.py — Unified spectral feature extraction for the Lyra Technique.

ONE extraction function. Every experiment imports this.
No more per-script reimplementations.

Features extracted per layer:
  Threshold-dependent (MP):
    mp_signal_rank, mp_signal_fraction, mp_top_sv_excess, mp_spectral_gap, mp_norm_per_token
  Threshold-dependent (GD):
    gd_signal_rank, gd_signal_fraction, gd_top_sv_excess
  Threshold-independent ("free features"):
    stable_rank, participation_ratio, condition_number, sv_kurtosis,
    mp_fit_residual, nuclear_norm_ratio
  Spectral:
    spectral_entropy
  Outlier counts:
    mp_outlier_count, mp_outlier_count_max, mp_outlier_count_sum
    gd_outlier_count, gd_outlier_count_max, gd_outlier_count_sum
  Raw:
    top_svs (first 20 singular values for recomputation)
    matrix_shape

Aggregation: mean across layers for continuous features,
             mean + max + sum for count features.
"""

import torch
import numpy as np
from scipy import stats as sp_stats


def gavish_donoho_threshold(S, m, n):
    """Optimal hard threshold per Gavish & Donoho 2014."""
    if m == 0 or n == 0:
        return 0.0
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    sigma = np.median(S) / np.sqrt(max(m, n))
    return omega * np.sqrt(max(m, n)) * sigma


def get_kv_matrix(cache, layer_idx):
    """Extract concatenated KV matrix from cache at given layer.
    Supports: DynamicCache with .layers (transformers 4.56+), legacy tuple cache."""
    try:
        if hasattr(cache, 'layers'):
            layer = cache.layers[layer_idx]
            k = layer.keys.squeeze(0)
            v = layer.values.squeeze(0)
        elif hasattr(cache, '__getitem__'):
            k = cache[layer_idx][0].squeeze(0)
            v = cache[layer_idx][1].squeeze(0)
    except (IndexError, AttributeError, TypeError) as e:
        raise RuntimeError(f"Cache access failed at layer {layer_idx}: {e}")

    return torch.cat([k.reshape(-1, k.shape[-1]),
                       v.reshape(-1, v.shape[-1])], dim=0).float()


def compute_layer_features(S, m, n):
    """Compute all features from a singular value vector.
    S: numpy array of singular values (descending order).
    m, n: matrix dimensions.
    Returns dict of all features."""
    if len(S) == 0 or m == 0 or n == 0:
        return {k: 0.0 for k in [
            'mp_signal_rank', 'mp_signal_fraction', 'mp_top_sv_excess', 'mp_spectral_gap',
            'mp_norm_per_token', 'gd_signal_rank', 'gd_signal_fraction', 'gd_top_sv_excess',
            'stable_rank', 'participation_ratio', 'condition_number', 'sv_kurtosis',
            'mp_fit_residual', 'nuclear_norm_ratio', 'spectral_entropy',
            'mp_outlier_count', 'gd_outlier_count', 'mp_threshold', 'gd_threshold',
            'top_svs', 'matrix_shape']}

    S2 = S**2
    S4 = S**4
    S2_sum = S2.sum()
    S2_norm = S2 / S2_sum if S2_sum > 0 else S2
    gamma = m / n if m > n else n / m

    # MP threshold
    mp_edge = (1 + 1/np.sqrt(gamma))**2
    var_est = np.median(S2) / m
    mp_thresh = np.sqrt(var_est * mp_edge)
    mp_sig = S[S > mp_thresh]

    # GD threshold
    gd_thresh = gavish_donoho_threshold(S, m, n)
    gd_sig = S[S > gd_thresh]

    # MP fit residual: fraction of SVs above the MP edge.
    # MP law predicts 0% above the edge for random matrices.
    # Higher values = more structured signal.
    mp_residual = len(mp_sig) / len(S) if len(S) > 0 else 0

    features = {
        # MP threshold-dependent
        'mp_signal_rank': int(len(mp_sig)),
        'mp_signal_fraction': float(np.sum(mp_sig**2) / S2_sum) if len(mp_sig) > 0 and S2_sum > 0 else 0.0,
        'mp_top_sv_excess': float(S[0] / mp_thresh) if mp_thresh > 0 else 0.0,
        'mp_spectral_gap': float(S[0] / S[1]) if len(S) > 1 and S[1] > 0 else 0.0,
        'mp_norm_per_token': float(S2_sum / max(m, 1)),

        # GD threshold-dependent
        'gd_signal_rank': int(len(gd_sig)),
        'gd_signal_fraction': float(np.sum(gd_sig**2) / S2_sum) if len(gd_sig) > 0 and S2_sum > 0 else 0.0,
        'gd_top_sv_excess': float(S[0] / gd_thresh) if gd_thresh > 0 else 0.0,

        # Threshold-independent ("free features")
        'stable_rank': float(S2_sum / S2[0]) if S2[0] > 0 else 0.0,
        'participation_ratio': float(S2_sum**2 / S4.sum()) if S4.sum() > 0 else 0.0,
        'condition_number': float(min(S[0] / S[-1], 1e6)) if len(S) > 0 and S[-1] > 0 else 0.0,
        'sv_kurtosis': float(sp_stats.kurtosis(S, fisher=True)) if len(S) > 3 else 0.0,
        'mp_fit_residual': float(mp_residual),
        'nuclear_norm_ratio': float(S.sum() / (np.sqrt(m * n) * S[0])) if S[0] > 0 and m > 0 and n > 0 else 0.0,

        # Spectral
        'spectral_entropy': float(-np.sum(S2_norm * np.log(S2_norm + 1e-10))) if S2_sum > 0 else 0.0,

        # Outlier counts
        'mp_outlier_count': int(len(mp_sig)),
        'gd_outlier_count': int(len(gd_sig)),

        # Thresholds (for reproducibility)
        'mp_threshold': float(mp_thresh),
        'gd_threshold': float(gd_thresh),

        # Raw (for recomputation)
        'top_svs': [float(s) for s in S[:20]],
        'matrix_shape': [int(m), int(n)],
    }

    return features


# All feature keys for aggregation
CONTINUOUS_KEYS = [
    'mp_signal_fraction', 'mp_top_sv_excess', 'mp_spectral_gap', 'mp_norm_per_token',
    'gd_signal_fraction', 'gd_top_sv_excess',
    'stable_rank', 'participation_ratio', 'condition_number', 'sv_kurtosis',
    'mp_fit_residual', 'nuclear_norm_ratio', 'spectral_entropy',
]

COUNT_KEYS = ['mp_signal_rank', 'gd_signal_rank', 'mp_outlier_count', 'gd_outlier_count']


def extract_features(model, input_ids):
    """Extract all spectral features from KV cache.

    Args:
        model: HuggingFace model in eval mode
        input_ids: tensor of shape [1, seq_len] on the correct device

    Returns:
        agg: dict of aggregated features (mean across layers, plus max/sum for counts)
        per_layer: list of per-layer feature dicts
    """
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        cache = out.past_key_values
    del out  # free logits tensor (~500MB for 7B model)

    # Determine number of layers
    if hasattr(cache, 'key_cache') and isinstance(cache.key_cache, list):
        n_layers = len(cache.key_cache)
    elif hasattr(cache, 'layers'):
        n_layers = len(cache.layers)
    elif hasattr(cache, '__len__'):
        n_layers = len(cache)
    else:
        n_layers = 0

    per_layer = []
    for li in range(n_layers):
        try:
            kv = get_kv_matrix(cache, li)
            m, n = kv.shape[0], kv.shape[1]
            S = torch.linalg.svdvals(kv).cpu().numpy()
            features = compute_layer_features(S, m, n)
            features['layer'] = li
            per_layer.append(features)
        except Exception as e:
            per_layer.append({'layer': li, 'error': str(e)})

    del cache
    torch.cuda.empty_cache()

    # Aggregate across layers
    valid = [f for f in per_layer if 'error' not in f]
    agg = {}

    for key in CONTINUOUS_KEYS:
        vals = [f[key] for f in valid if key in f]
        agg[key] = float(np.mean(vals)) if vals else 0.0

    for key in COUNT_KEYS:
        vals = [f[key] for f in valid if key in f]
        agg[key] = float(np.mean(vals)) if vals else 0.0
        agg[key + '_max'] = float(np.max(vals)) if vals else 0.0
        agg[key + '_sum'] = float(np.sum(vals)) if vals else 0.0

    return agg, per_layer


def extract_delta_features(model, prompt_ids, full_ids):
    """Extract delta features (generation minus encoding).

    Args:
        model: HuggingFace model in eval mode
        prompt_ids: tensor [1, prompt_len] — encoding only
        full_ids: tensor [1, prompt_len + gen_len] — full sequence

    Returns:
        encoding_agg, generation_agg, delta_agg, generation_per_layer
    """
    enc_agg, enc_layers = extract_features(model, prompt_ids)
    gen_agg, gen_layers = extract_features(model, full_ids)

    delta_agg = {}
    for key in list(CONTINUOUS_KEYS) + list(COUNT_KEYS):
        delta_agg[key] = gen_agg.get(key, 0) - enc_agg.get(key, 0)
    for key in COUNT_KEYS:
        delta_agg[key + '_max'] = gen_agg.get(key + '_max', 0) - enc_agg.get(key + '_max', 0)
        delta_agg[key + '_sum'] = gen_agg.get(key + '_sum', 0) - enc_agg.get(key + '_sum', 0)

    return enc_agg, gen_agg, delta_agg, gen_layers


def tokenize_chat(tokenizer, system_prompt, user_prompt):
    """Standard chat template tokenization."""
    msgs = [{'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}]
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
            add_generation_prompt=True)
    return text
