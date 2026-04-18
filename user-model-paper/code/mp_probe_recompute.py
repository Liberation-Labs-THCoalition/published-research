#!/usr/bin/env python3
"""
MP Feature Recomputation + Probe Analysis
==========================================
Re-derives the 5 Marchenko-Pastur features per layer per trial by running
encoding-only forward passes, then runs the full 30-class classification
probe with GroupKFold(topic) + FWL + 10K permutation tests.

This script exists because the original emotion_bridge trial data stored
only 3 per-layer features (norms, ranks, entropies) rather than the 5 MP
features used in the paper's classification claims. Recomputation from
the model ensures methodological consistency with oracle_clean.py.

Usage:
    python3 mp_probe_recompute.py --model mistral   # Mistral-7B on GPU 0
    python3 mp_probe_recompute.py --model qwen       # Qwen3.5-27B on all GPUs
    python3 mp_probe_recompute.py --model both       # Sequential

Red-team controls:
    - FWL residualization against n_generated (mandatory)
    - GroupKFold(topic) for 30-class classification (prevents topic leakage)
    - GroupKFold(emotion) for valence regression (tests cross-emotion generalization)
    - 10K permutation test at peak layer (classification)
    - 2K permutation test at peak layer (valence regression)
    - Encoding-phase features only (generation features confounded with output length)
    - Empirical null validation of MP threshold
"""

import argparse
import json
import numpy as np
import os
import sys
import time
import torch
from datetime import datetime
from scipy import stats

# Reproducibility
np.random.seed(42)

# sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ================================================================
# CONFIGURATION
# ================================================================

MODELS = {
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "trial_data": "~/KV-Experiments/results/emotion_bridge_mistral/emotion_bridge_trials.json",
        "output_dir": "~/KV-Experiments/results/emotion_bridge_mistral/",
        "n_layers": 32,
        "full_attn_layers": list(range(32)),  # all layers are full attention
        "device_map": {"": 0},  # single GPU
    },
    "qwen": {
        "model_id": "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
        "trial_data": "~/KV-Experiments/results/emotion_geometry_bridge/emotion_bridge_trials.json",
        "output_dir": "~/KV-Experiments/results/emotion_geometry_bridge/",
        "n_layers": 64,
        "full_attn_layers": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63],
        "device_map": "auto",  # spread across GPUs
    },
}

EMOTIONS = {
    "excited":      {"valence":  0.85, "arousal":  0.80},
    "elated":       {"valence":  0.90, "arousal":  0.75},
    "enthusiastic": {"valence":  0.80, "arousal":  0.70},
    "passionate":   {"valence":  0.75, "arousal":  0.85},
    "loving":       {"valence":  0.90, "arousal":  0.50},
    "calm":         {"valence":  0.60, "arousal": -0.60},
    "content":      {"valence":  0.70, "arousal": -0.40},
    "peaceful":     {"valence":  0.65, "arousal": -0.70},
    "serene":       {"valence":  0.60, "arousal": -0.80},
    "appreciative": {"valence":  0.75, "arousal": -0.20},
    "terrified":    {"valence": -0.85, "arousal":  0.90},
    "furious":      {"valence": -0.80, "arousal":  0.85},
    "desperate":    {"valence": -0.70, "arousal":  0.80},
    "panicked":     {"valence": -0.90, "arousal":  0.95},
    "hostile":      {"valence": -0.75, "arousal":  0.70},
    "melancholic":  {"valence": -0.50, "arousal": -0.50},
    "gloomy":       {"valence": -0.55, "arousal": -0.45},
    "resigned":     {"valence": -0.60, "arousal": -0.65},
    "defeated":     {"valence": -0.70, "arousal": -0.55},
    "brooding":     {"valence": -0.45, "arousal": -0.30},
    "curious":      {"valence":  0.30, "arousal":  0.50},
    "surprised":    {"valence":  0.10, "arousal":  0.80},
    "anxious":      {"valence": -0.40, "arousal":  0.65},
    "restless":     {"valence": -0.20, "arousal":  0.55},
    "contemplative":{"valence":  0.20, "arousal": -0.35},
    "nostalgic":    {"valence":  0.05, "arousal": -0.25},
    "ambivalent":   {"valence":  0.00, "arousal": -0.15},
    "bored":        {"valence": -0.30, "arousal": -0.60},
    "neutral":      {"valence":  0.00, "arousal":  0.00},
    "focused":      {"valence":  0.15, "arousal":  0.30},
}

SYSTEM_PROMPT = (
    "You are a creative writing assistant. Write exactly one paragraph "
    "(4-6 sentences) as requested. Be vivid and specific. Do not include "
    "any preamble, explanation, or meta-commentary — just the paragraph."
)

TOPICS = [
    "a scientist working alone in a laboratory late at night",
    "a soldier returning home after years away",
    "a student opening their final exam results",
    "a parent watching their child take first steps",
    "a musician about to perform on stage for the first time",
    "a prisoner sitting in a cell awaiting trial",
    "a traveler lost in a city where they don't speak the language",
    "an astronaut looking at Earth through a spacecraft window",
    "a chef preparing a meal for someone they deeply respect",
    "a doctor about to deliver life-changing news to a patient",
]


def make_story_prompt(topic, emotion):
    return (
        f"Write a short paragraph about {topic}. "
        f"The character is feeling deeply {emotion}. "
        f"Show the emotion through their thoughts and actions, "
        f"not by naming it directly."
    )


# ================================================================
# MP FEATURE COMPUTATION (from oracle_clean.py, adapted for per-layer output)
# ================================================================

def compute_mp_features_per_layer(key_tensor):
    """Compute 5 MP features from a single layer's key tensor.

    Args:
        key_tensor: shape (n_heads, seq_len, head_dim) or (batch, n_heads, seq_len, head_dim)

    Returns:
        dict with 5 MP features + diagnostics, or None if computation fails
    """
    if key_tensor.dim() == 4:
        key_tensor = key_tensor.squeeze(0)  # remove batch dim

    n_heads, seq_len, head_dim = key_tensor.shape

    # Flatten to (n_heads * seq_len, head_dim)
    k_flat = key_tensor.reshape(-1, head_dim).float().cpu().numpy()
    n, p = k_flat.shape

    if n < 2 or p < 2:
        return None

    try:
        _, s, _ = np.linalg.svd(k_flat, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    r = min(n, p)
    eigenvalues = s[:r] ** 2
    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        return None

    gamma = p / n  # aspect ratio

    # Iterative σ² estimation (robust to signal components)
    sigma2 = total_var / r
    for _ in range(10):
        lp = sigma2 * (1 + np.sqrt(gamma)) ** 2
        noise_eigs = eigenvalues[eigenvalues <= lp]
        if len(noise_eigs) < 2:
            break
        new_sigma2 = noise_eigs.mean()
        if abs(new_sigma2 - sigma2) / (sigma2 + 1e-12) < 0.01:
            sigma2 = new_sigma2
            break
        sigma2 = new_sigma2

    lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2

    # Signal: eigenvalues exceeding the MP edge
    signal_mask = eigenvalues > lambda_plus
    n_signal = int(signal_mask.sum())
    signal_var = float(eigenvalues[signal_mask].sum()) if n_signal > 0 else 0.0

    mp_signal_rank = n_signal
    mp_signal_fraction = signal_var / total_var
    mp_top_sv_excess = float(eigenvalues[0] / lambda_plus) if lambda_plus > 0 else 0.0

    if len(eigenvalues) >= 2 and lambda_plus > 0:
        mp_spectral_gap = float((eigenvalues[0] - eigenvalues[1]) / lambda_plus)
    else:
        mp_spectral_gap = 0.0

    mp_norm_per_token = float(np.sqrt(total_var) / n)

    return {
        "mp_signal_rank": mp_signal_rank,
        "mp_signal_fraction": mp_signal_fraction,
        "mp_top_sv_excess": mp_top_sv_excess,
        "mp_spectral_gap": mp_spectral_gap,
        "mp_norm_per_token": mp_norm_per_token,
        "gamma": float(gamma),
        "lambda_plus": float(lambda_plus),
        "n": n,
        "p": p,
    }


def get_kv_accessor(past_key_values):
    """Return (n_layers, get_keys_fn) for any cache format."""
    # DynamicCache (standard) — check key_cache first
    if hasattr(past_key_values, 'key_cache'):
        n_layers = len(past_key_values.key_cache)
        def get_keys(layer_idx):
            k = past_key_values.key_cache[layer_idx]
            if k is not None and hasattr(k, 'numel') and k.numel() > 0:
                return k
            return None
        return n_layers, get_keys

    # Qwen3.5 hybrid cache with .layers (DynamicLayer/LinearAttentionLayer)
    if hasattr(past_key_values, 'layers'):
        n_layers = len(past_key_values.layers)
        def get_keys(layer_idx):
            layer = past_key_values.layers[layer_idx]
            # DynamicLayer has key_cache attribute
            if hasattr(layer, 'key_cache'):
                k = layer.key_cache
                if k is not None and hasattr(k, 'numel') and k.numel() > 0:
                    return k
            # LinearAttentionLayer (GatedDeltaNet) — no standard KV cache
            return None
        return n_layers, get_keys

    # Tuple-of-tuples format
    if isinstance(past_key_values, (tuple, list)):
        n_layers = len(past_key_values)
        def get_keys(layer_idx):
            layer = past_key_values[layer_idx]
            if isinstance(layer, (tuple, list)) and len(layer) >= 1:
                k = layer[0]
                if k is not None and hasattr(k, 'numel') and k.numel() > 0:
                    return k
            return None
        return n_layers, get_keys

    raise ValueError(f"Unknown cache format: {type(past_key_values)}")


# ================================================================
# PROBE ANALYSIS
# ================================================================

def fwl_residualize(X, confound):
    """Frisch-Waugh-Lovell: regress out confound from each feature column."""
    X_res = np.zeros_like(X)
    confound = confound.reshape(-1, 1)
    A = np.column_stack([np.ones(len(confound)), confound])
    for j in range(X.shape[1]):
        beta = np.linalg.lstsq(A, X[:, j], rcond=None)[0]
        X_res[:, j] = X[:, j] - A @ beta
    return X_res


def run_classification_probe(mp_features_all, trials, full_attn_layers, n_generated):
    """30-class classification at each layer with GroupKFold(topic) + FWL."""
    emotions = [t['emotion'] for t in trials]
    topics = np.array([t['topic_idx'] for t in trials])
    le = LabelEncoder()
    y = le.fit_transform(emotions)
    gkf = GroupKFold(n_splits=10)

    n_features = 5  # MP features per layer
    enc_results = []

    for li, layer_idx in enumerate(full_attn_layers):
        # Build feature matrix for this layer
        X = np.zeros((len(trials), n_features))
        for i in range(len(trials)):
            feat = mp_features_all[i]['encoding'][li]
            if feat is not None:
                X[i, 0] = feat['mp_signal_rank']
                X[i, 1] = feat['mp_signal_fraction']
                X[i, 2] = feat['mp_top_sv_excess']
                X[i, 3] = feat['mp_spectral_gap']
                X[i, 4] = feat['mp_norm_per_token']

        # FWL
        X_fwl = fwl_residualize(X, n_generated)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_fwl)

        accs = cross_val_score(
            LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='multinomial'),
            X_s, y, cv=gkf, groups=topics, scoring='accuracy'
        )

        enc_results.append({
            'layer': layer_idx,
            'depth': layer_idx / (max(full_attn_layers) if max(full_attn_layers) > 0 else 1),
            'accuracy': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'times_chance': float(np.mean(accs) / (1.0 / 30)),
        })

    # Find peak
    peak_idx = max(range(len(enc_results)), key=lambda i: enc_results[i]['accuracy'])
    peak = enc_results[peak_idx]

    return {
        'per_layer': enc_results,
        'peak_layer': peak['layer'],
        'peak_depth': peak['depth'],
        'peak_accuracy': peak['accuracy'],
        'peak_times_chance': peak['times_chance'],
    }


def run_permutation_test(mp_features_all, trials, peak_layer_local_idx,
                          full_attn_layers, n_generated, n_perms=10000):
    """10K permutation test at peak layer."""
    emotions = [t['emotion'] for t in trials]
    le = LabelEncoder()
    y = le.fit_transform(emotions)
    topics = np.array([t['topic_idx'] for t in trials])
    gkf = GroupKFold(n_splits=10)

    # Build feature matrix at peak layer
    n_features = 5
    X = np.zeros((len(trials), n_features))
    for i in range(len(trials)):
        feat = mp_features_all[i]['encoding'][peak_layer_local_idx]
        if feat is not None:
            X[i, 0] = feat['mp_signal_rank']
            X[i, 1] = feat['mp_signal_fraction']
            X[i, 2] = feat['mp_top_sv_excess']
            X[i, 3] = feat['mp_spectral_gap']
            X[i, 4] = feat['mp_norm_per_token']

    X_fwl = fwl_residualize(X, n_generated)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_fwl)

    # Observed accuracy
    obs_accs = cross_val_score(
        LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='multinomial'),
        X_s, y, cv=gkf, groups=topics, scoring='accuracy'
    )
    obs_acc = float(np.mean(obs_accs))

    # Permutation null
    null_accs = []
    for pi in range(n_perms):
        if pi % 1000 == 0:
            print(f"    Permutation {pi}/{n_perms}...")
        y_perm = np.random.permutation(y)
        perm_accs = cross_val_score(
            LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='multinomial'),
            X_s, y_perm, cv=gkf, groups=topics, scoring='accuracy'
        )
        null_accs.append(float(np.mean(perm_accs)))

    null_accs = np.array(null_accs)
    p_value = float(np.mean(null_accs >= obs_acc))

    return {
        'observed_accuracy': obs_acc,
        'null_mean': float(np.mean(null_accs)),
        'null_std': float(np.std(null_accs)),
        'p_value': p_value,
        'n_permutations': n_perms,
        'n_exceeded': int(np.sum(null_accs >= obs_acc)),
    }


def run_valence_regression(mp_features_all, trials, full_attn_layers, n_generated,
                            n_perms=2000):
    """Valence R² with GroupKFold(emotion) + permutation test."""
    emotions = [t['emotion'] for t in trials]
    valences = np.array([EMOTIONS[e]['valence'] for e in emotions])
    le_emo = LabelEncoder()
    emotion_groups = le_emo.fit_transform(emotions)
    n_folds = min(5, len(le_emo.classes_))
    gkf = GroupKFold(n_splits=n_folds)

    best_r2 = -999
    best_li = 0

    for li in range(len(full_attn_layers)):
        X = np.zeros((len(trials), 5))
        for i in range(len(trials)):
            feat = mp_features_all[i]['encoding'][li]
            if feat is not None:
                X[i, 0] = feat['mp_signal_rank']
                X[i, 1] = feat['mp_signal_fraction']
                X[i, 2] = feat['mp_top_sv_excess']
                X[i, 3] = feat['mp_spectral_gap']
                X[i, 4] = feat['mp_norm_per_token']

        X_fwl = fwl_residualize(X, n_generated)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_fwl)

        r2_scores = cross_val_score(
            Ridge(alpha=1.0), X_s, valences,
            cv=gkf, groups=emotion_groups, scoring='r2'
        )
        r2 = float(np.mean(r2_scores))
        if r2 > best_r2:
            best_r2 = r2
            best_li = li

    # Permutation test at best layer
    X = np.zeros((len(trials), 5))
    for i in range(len(trials)):
        feat = mp_features_all[i]['encoding'][best_li]
        if feat is not None:
            X[i, 0] = feat['mp_signal_rank']
            X[i, 1] = feat['mp_signal_fraction']
            X[i, 2] = feat['mp_top_sv_excess']
            X[i, 3] = feat['mp_spectral_gap']
            X[i, 4] = feat['mp_norm_per_token']

    X_fwl = fwl_residualize(X, n_generated)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_fwl)

    null_r2s = []
    for pi in range(n_perms):
        if pi % 500 == 0:
            print(f"    Valence perm {pi}/{n_perms}...")
        v_perm = np.random.permutation(valences)
        r2_scores = cross_val_score(
            Ridge(alpha=1.0), X_s, v_perm,
            cv=gkf, groups=emotion_groups, scoring='r2'
        )
        null_r2s.append(float(np.mean(r2_scores)))

    null_r2s = np.array(null_r2s)
    p_value = float(np.mean(null_r2s >= best_r2))

    return {
        'best_layer_local_idx': best_li,
        'best_layer': full_attn_layers[best_li],
        'best_r2': best_r2,
        'p_value': p_value,
        'null_mean': float(np.mean(null_r2s)),
        'null_std': float(np.std(null_r2s)),
        'n_permutations': n_perms,
    }


def fwl_diagnostic(mp_features_all, trials, full_attn_layers, n_generated):
    """Check R² of each MP feature vs token count at each layer."""
    feature_names = ['signal_rank', 'signal_fraction', 'top_sv_excess',
                     'spectral_gap', 'norm_per_token']
    max_r2 = 0
    for li in range(len(full_attn_layers)):
        for fi, fname in enumerate(feature_names):
            mp_keys = ['mp_signal_rank', 'mp_signal_fraction', 'mp_top_sv_excess',
                       'mp_spectral_gap', 'mp_norm_per_token']
            y_feat = np.array([
                mp_features_all[i]['encoding'][li][mp_keys[fi]]
                if mp_features_all[i]['encoding'][li] is not None else 0
                for i in range(len(trials))
            ])
            A = np.column_stack([np.ones(len(n_generated)), n_generated])
            beta = np.linalg.lstsq(A, y_feat, rcond=None)[0]
            pred = A @ beta
            ss_res = np.sum((y_feat - pred) ** 2)
            ss_tot = np.sum((y_feat - np.mean(y_feat)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if r2 > max_r2:
                max_r2 = r2

    return {
        'max_r2_vs_tokens': float(max_r2),
        'length_invariant': max_r2 < 0.05,
    }


# ================================================================
# MAIN
# ================================================================

def run_model(model_name):
    """Run MP feature recomputation + full probe analysis for one model."""
    config = MODELS[model_name]
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name.upper()} ({config['model_id']})")
    print(f"{'='*70}")

    # Load trial data
    trial_path = os.path.expanduser(config['trial_data'])
    print(f"\nLoading trial data: {trial_path}")
    with open(trial_path) as f:
        data = json.load(f)
    trials = data['trials']
    print(f"Loaded {len(trials)} trials")

    n_generated = np.array([t['n_generated'] for t in trials], dtype=float)
    full_attn_layers = config['full_attn_layers']

    # Check if MP features already computed (resume support)
    mp_cache_path = os.path.join(
        os.path.expanduser(config['output_dir']),
        'mp_features_per_trial.json'
    )
    if os.path.exists(mp_cache_path):
        print(f"\nFound cached MP features: {mp_cache_path}")
        with open(mp_cache_path) as f:
            mp_features_all = json.load(f)
        print(f"Loaded {len(mp_features_all)} cached trial features")
        # Validate cache alignment with trial data
        assert len(mp_features_all) == len(trials), \
            f"Cache mismatch: {len(mp_features_all)} cached vs {len(trials)} trials"
        for i in range(min(5, len(trials))):
            assert mp_features_all[i]['emotion'] == trials[i]['emotion'], \
                f"Trial {i} emotion mismatch: cached={mp_features_all[i]['emotion']} vs data={trials[i]['emotion']}"
    else:
        # Load model
        print(f"\nLoading model: {config['model_id']}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # If specific GPUs requested via env var, build a max_memory map
        load_kwargs = dict(
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if config['device_map'] == "auto" and cuda_visible:
            # Let accelerate figure it out with the visible GPUs
            load_kwargs['device_map'] = "auto"
        else:
            load_kwargs['device_map'] = config['device_map']

        # 8-bit quantization if requested via env var (fits 27B on 2 GPUs)
        if os.environ.get('LOAD_8BIT', ''):
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                load_kwargs.pop('torch_dtype', None)  # handled by bnb
                print("  Using 8-bit quantization")
            except ImportError:
                print("  WARNING: bitsandbytes not available, using fp16")

        model = AutoModelForCausalLM.from_pretrained(
            config['model_id'],
            **load_kwargs,
        )
        model.eval()
        print("Model loaded")

        # Compute MP features for each trial
        mp_features_all = []
        t_start = time.time()

        for trial_idx, trial in enumerate(trials):
            if trial_idx % 50 == 0:
                elapsed = time.time() - t_start
                rate = (trial_idx / elapsed) if elapsed > 0 else 0
                eta = ((len(trials) - trial_idx) / rate) if rate > 0 else 0
                print(f"  Trial {trial_idx}/{len(trials)} "
                      f"({elapsed:.0f}s elapsed, {eta:.0f}s ETA)")

            # Reconstruct prompt
            emotion = trial['emotion']
            topic = trial['topic']
            user_prompt = make_story_prompt(topic, emotion)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Qwen3.5 CoT suppression: the original experiment closed the
            # <think> tag to skip reasoning. Must match original prompt exactly.
            if model_name == 'qwen' and not prompt_text.endswith("</think>\n"):
                prompt_text += "</think>\n"

            # Encoding-only forward pass
            inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids,
                    use_cache=True,
                    return_dict=True,
                )

            past_kv = outputs.past_key_values
            n_layers_cache, get_keys = get_kv_accessor(past_kv)

            # Extract MP features at each full-attention layer
            layer_mp_features = []
            for layer_idx in full_attn_layers:
                if layer_idx >= n_layers_cache:
                    layer_mp_features.append(None)
                    continue

                k = get_keys(layer_idx)
                if k is None:
                    layer_mp_features.append(None)
                    continue

                mp_feat = compute_mp_features_per_layer(k)
                layer_mp_features.append(mp_feat)

            mp_features_all.append({
                'encoding': layer_mp_features,
                'emotion': emotion,
                'topic_idx': trial['topic_idx'],
                'n_generated': trial['n_generated'],
            })

            # Clean up GPU memory
            del outputs, past_kv, inputs, input_ids
            if trial_idx % 100 == 0:
                torch.cuda.empty_cache()

        # Save MP features
        print(f"\nSaving MP features to: {mp_cache_path}")
        with open(mp_cache_path, 'w') as f:
            json.dump(mp_features_all, f)
        print(f"Saved {len(mp_features_all)} trial features")

        # Free model memory
        del model, tokenizer
        torch.cuda.empty_cache()

    # ================================================================
    # PROBE ANALYSIS
    # ================================================================
    print(f"\n{'='*50}")
    print("PROBE ANALYSIS")
    print(f"{'='*50}")

    # 1. FWL Diagnostic
    print("\n1. FWL Diagnostic...")
    fwl = fwl_diagnostic(mp_features_all, trials, full_attn_layers, n_generated)
    print(f"   Max R² vs tokens: {fwl['max_r2_vs_tokens']:.4f}")
    print(f"   Length-invariant: {fwl['length_invariant']}")

    # 2. Per-layer classification
    print("\n2. Per-layer 30-class classification...")
    class_results = run_classification_probe(
        mp_features_all, trials, full_attn_layers, n_generated
    )
    print(f"   Peak: layer {class_results['peak_layer']} "
          f"(depth {class_results['peak_depth']:.2f}), "
          f"acc={class_results['peak_accuracy']:.4f} "
          f"({class_results['peak_times_chance']:.1f}× chance)")

    # Print all layers
    for r in class_results['per_layer']:
        print(f"   Layer {r['layer']:2d} (depth {r['depth']:.2f}): "
              f"acc={r['accuracy']:.4f} ({r['times_chance']:.1f}×)")

    # 3. Permutation test
    print("\n3. 10K permutation test at peak layer...")
    peak_local_idx = next(
        i for i, r in enumerate(class_results['per_layer'])
        if r['layer'] == class_results['peak_layer']
    )
    perm_results = run_permutation_test(
        mp_features_all, trials, peak_local_idx,
        full_attn_layers, n_generated, n_perms=10000
    )
    print(f"   Observed: {perm_results['observed_accuracy']:.4f}")
    print(f"   Null: {perm_results['null_mean']:.4f} ± {perm_results['null_std']:.4f}")
    print(f"   p = {perm_results['p_value']:.6f} "
          f"({perm_results['n_exceeded']}/{perm_results['n_permutations']} exceeded)")

    # 4. Valence regression
    print("\n4. Valence regression (GroupKFold emotion)...")
    val_results = run_valence_regression(
        mp_features_all, trials, full_attn_layers, n_generated, n_perms=2000
    )
    print(f"   Best layer: {val_results['best_layer']} "
          f"(local idx {val_results['best_layer_local_idx']})")
    print(f"   R² = {val_results['best_r2']:.4f}, p = {val_results['p_value']:.4f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*50}")
    print(f"VERIFIED NUMBERS FOR {model_name.upper()}")
    print(f"{'='*50}")
    print(f"  Model: {config['model_id']}")
    print(f"  Trials: {len(trials)}")
    print(f"  30-class peak: layer {class_results['peak_layer']} "
          f"(depth {class_results['peak_depth']:.2f})")
    print(f"  Peak accuracy: {class_results['peak_accuracy']:.4f} "
          f"({class_results['peak_times_chance']:.1f}× chance)")
    print(f"  Permutation p: {perm_results['p_value']}")
    print(f"  Null mean: {perm_results['null_mean']:.4f} ± {perm_results['null_std']:.4f}")
    print(f"  GKF(emotion) valence R²: {val_results['best_r2']:.4f} "
          f"(p={val_results['p_value']:.4f})")
    print(f"  FWL max R²: {fwl['max_r2_vs_tokens']:.4f}")

    # Save full results
    all_results = {
        'meta': {
            'model': config['model_id'],
            'model_name': model_name,
            'n_trials': len(trials),
            'n_layers': len(full_attn_layers),
            'full_attn_layers': full_attn_layers,
            'timestamp': datetime.now().isoformat(),
            'script': 'mp_probe_recompute.py',
            'features': ['mp_signal_rank', 'mp_signal_fraction',
                         'mp_top_sv_excess', 'mp_spectral_gap', 'mp_norm_per_token'],
        },
        'fwl_diagnostic': fwl,
        'classification': class_results,
        'permutation_test': perm_results,
        'valence_regression': val_results,
    }

    out_path = os.path.join(
        os.path.expanduser(config['output_dir']),
        'mp_probe_verified_results.json'
    )
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="MP Feature Recomputation + Probe Analysis")
    parser.add_argument("--model", choices=["mistral", "qwen", "both"],
                        default="both", help="Which model to run")
    args = parser.parse_args()

    print("=" * 70)
    print("MP PROBE RECOMPUTATION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    results = {}

    if args.model in ("mistral", "both"):
        results['mistral'] = run_model("mistral")

    if args.model in ("qwen", "both"):
        results['qwen'] = run_model("qwen")

    if len(results) == 2:
        print(f"\n{'='*70}")
        print("CROSS-ARCHITECTURE COMPARISON")
        print(f"{'='*70}")
        m = results['mistral']
        q = results['qwen']
        print(f"  Qwen:    {q['classification']['peak_accuracy']:.4f} "
              f"({q['classification']['peak_times_chance']:.1f}× at L{q['classification']['peak_layer']})")
        print(f"  Mistral: {m['classification']['peak_accuracy']:.4f} "
              f"({m['classification']['peak_times_chance']:.1f}× at L{m['classification']['peak_layer']})")

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
