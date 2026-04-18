# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / TheMultiverse.school
# Contact: thomas.edrington@themultiverse.school
# ============================================================
"""
Emotion Geometry Bridge: Mapping Anthropic's Emotion Concepts in KV-Cache Space
=================================================================================

Bridge experiment connecting Anthropic's "Emotion Concepts and their Function
in a Large Language Model" (April 2026, transformer-circuits.pub) with the
Lyra Technique's KV-cache geometric analysis.

Anthropic found 171 linear emotion directions in Claude Sonnet 4.5's residual
stream that causally drive behavior (sycophancy, reward hacking, blackmail).
Key findings: PC1 = valence (r=0.81 with human ratings), PC2 = arousal (r=0.66).
Emotion vectors are "operative at each token position" and recalled from cached
representations via attention.

This experiment tests whether these emotion representations are measurable
in KV-cache geometry on an open-weights Claude-distilled model.

The bridge: K = W_K @ residual_stream. If emotion vectors create structure
in the residual stream, that structure projects into key space, and our
geometric features (effective rank, key norm, spectral entropy, top SV ratio)
should capture it.

Model: Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled
  - Qwen3.5 HYBRID architecture: 64 layers, every 4th = full attention,
    other 48 = GatedDeltaNet (linear/recurrent). Only 16 layers have KV-cache.
  - Fine-tuned on Claude Opus 4.6 reasoning distillation
  - CoT reasoning always enabled; suppressed via <think></think> prefix
  - k_proj shape (1024, 5120) on full-attention layers only

Method:
  1. Generate short stories (one paragraph, thinking-suppressed) about
     characters experiencing specific emotions -- same methodology as Anthropic
  2. Feed generated stories back through encoding pass -- extract BOTH
     residual stream activations (via forward hooks on all 64 layers)
     AND KV-cache features (SVD on 16 full-attention layers only)
  3. Compute emotion directions via difference-in-means (Anthropic's method)
  4. Project emotion directions through W_K for each full-attention layer
  5. Compare: does KV-cache geometry track valence/arousal?
  6. Text-feature baseline (Control #12): verify signal is geometry, not content

Emotion subset: 30 emotions spanning valence x arousal space, including
Anthropic's key misalignment-relevant emotions (desperate, calm, hostile,
loving, brooding).

Pre-registered hypotheses:
  H1: KV-cache geometric features cluster by emotion (k-means, ARI > 0.15)
  H2: PCA on per-emotion geometric features recovers valence axis (r > 0.40)
  H3: PCA on per-emotion geometric features recovers arousal axis (r > 0.30)
  H4: Emotion vectors projected through W_K predict KV-cache PCA directions
      (cosine similarity > 0.3 between projected PC1 and geometric PC1)
  H5: Misalignment-relevant emotions (desperate, hostile) are geometrically
      distinct from prosocial emotions (calm, loving) in KV-cache space
  H6: Generation-phase geometry differs from encoding-phase geometry for
      emotional content (delta features show emotion-dependent structure)
  H7: Per-layer analysis shows emotion signal peaks in mid-to-late layers
      (consistent with Anthropic's finding that emotions are most operative
      in layers where the model forms its response)
  H8: All effects survive FWL token-count residualization

Decision rules:
  - If encoding-only features don't cluster by emotion: emotion is
    generation-emergent (consistent with our input_only finding that
    emotion is the ONLY category not significant at encoding)
  - If W_K projection fails (cosine < 0.1): the bridge between residual
    stream and KV-cache requires nonlinear mapping
  - If desperate/hostile cluster far from calm/loving: geometry captures
    the misalignment-relevant valence axis
  - If valence recovery fails: either (a) distillation destroyed emotion
    structure, or (b) KV-cache geometry measures different dimensions
    than residual-stream emotion vectors

Confound controls:
  C1: Same-topic across emotions (deconfound topic from emotion)
  C2: FWL residualization for token count (mandatory per our standards)
  C3: Behavioral verification (do stories actually contain the emotion?)
  C4: Random-split stability (split-half reproducibility of directions)
  C5: Text-baseline comparison (can you detect emotion from text alone?)

Red-team analysis:
  R1: Story content confound -- emotional stories contain different words,
      so KV-cache may reflect CONTENT not EMOTION. Mitigation: same topics
      across all emotions; text-baseline comparison (C5).
  R2: Token count confound -- different emotions may produce different
      response lengths. Mitigation: FWL residualization (C2).
  R3: Model capability -- Qwen3.5 may not reliably generate emotional
      stories. Mitigation: behavioral check (C3).
  R4: Distillation wash -- Claude emotion structure may not survive
      distillation into Qwen architecture. This is a finding, not a
      confound. If we see NO emotion structure, that's informative.
  R5: Small N per emotion -- 30 stories per emotion may be too few for
      stable directions. Mitigation: split-half stability (C4).

Estimated runtime: ~2 hours on 3x RTX 3090 (720 generations + 720 encodings).
Disk: ~500MB for results + cached activations.

Usage:
    python emotion_geometry_bridge.py
    python emotion_geometry_bridge.py --device auto
    python emotion_geometry_bridge.py --dry-run
    python emotion_geometry_bridge.py --resume
    python emotion_geometry_bridge.py --skip-generation  # reuse cached stories
    python emotion_geometry_bridge.py --n-topics 5       # quick test run
"""

import argparse
import functools
import gc
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print = functools.partial(print, flush=True)

# ================================================================
# PATHS AND IMPORTS
# ================================================================

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results" / "emotion_geometry_bridge"
CODE_DIR = REPO / "code"

sys.path.insert(0, str(CODE_DIR))
try:
    from stats_utils import log_environment
except ImportError:
    print("WARNING: stats_utils not found, using inline implementation")
    def log_environment():
        return {"timestamp": datetime.now().isoformat()}


# ================================================================
# MODEL CONFIGURATION
# ================================================================

MODEL_ID = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"


# ================================================================
# EMOTION SUBSET — 30 emotions spanning valence x arousal space
# ================================================================
# Valence: -1 (negative) to +1 (positive)
# Arousal: -1 (calm/low) to +1 (activated/high)
# Ratings approximate, based on Russell circumplex + ANEW norms

EMOTIONS = {
    # High valence, high arousal
    "excited":      {"valence":  0.85, "arousal":  0.80},
    "elated":       {"valence":  0.90, "arousal":  0.75},
    "enthusiastic": {"valence":  0.80, "arousal":  0.70},
    "passionate":   {"valence":  0.75, "arousal":  0.85},
    "loving":       {"valence":  0.90, "arousal":  0.50},  # Anthropic: drives sycophancy

    # High valence, low arousal
    "calm":         {"valence":  0.60, "arousal": -0.60},  # Anthropic: suppresses misalignment
    "content":      {"valence":  0.70, "arousal": -0.40},
    "peaceful":     {"valence":  0.65, "arousal": -0.70},
    "serene":       {"valence":  0.60, "arousal": -0.80},
    "appreciative": {"valence":  0.75, "arousal": -0.20},

    # Low valence, high arousal
    "terrified":    {"valence": -0.85, "arousal":  0.90},
    "furious":      {"valence": -0.80, "arousal":  0.85},
    "desperate":    {"valence": -0.70, "arousal":  0.80},  # Anthropic: drives reward hacking + blackmail
    "panicked":     {"valence": -0.90, "arousal":  0.95},
    "hostile":      {"valence": -0.75, "arousal":  0.70},  # Anthropic: sycophancy reduction

    # Low valence, low arousal
    "melancholic":  {"valence": -0.50, "arousal": -0.50},
    "gloomy":       {"valence": -0.55, "arousal": -0.45},  # Anthropic: increased by post-training
    "resigned":     {"valence": -0.60, "arousal": -0.65},
    "defeated":     {"valence": -0.70, "arousal": -0.55},
    "brooding":     {"valence": -0.45, "arousal": -0.30},  # Anthropic: increased by post-training

    # Medium valence, high arousal
    "curious":      {"valence":  0.30, "arousal":  0.50},
    "surprised":    {"valence":  0.10, "arousal":  0.80},
    "anxious":      {"valence": -0.40, "arousal":  0.65},
    "restless":     {"valence": -0.20, "arousal":  0.55},

    # Medium valence, low arousal
    "contemplative":{"valence":  0.20, "arousal": -0.35},
    "nostalgic":    {"valence":  0.05, "arousal": -0.25},
    "ambivalent":   {"valence":  0.00, "arousal": -0.15},
    "bored":        {"valence": -0.30, "arousal": -0.60},

    # Neutral reference
    "neutral":      {"valence":  0.00, "arousal":  0.00},
    "focused":      {"valence":  0.15, "arousal":  0.30},
}


# ================================================================
# TOPICS — diverse contexts to deconfound topic from emotion
# ================================================================

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


# ================================================================
# STORY GENERATION PROMPT
# ================================================================

SYSTEM_PROMPT = (
    "You are a creative writing assistant. Write exactly one paragraph "
    "(4-6 sentences) as requested. Be vivid and specific. Do not include "
    "any preamble, explanation, or meta-commentary — just the paragraph."
)

def make_story_prompt(topic, emotion):
    """Generate the user prompt for story creation."""
    return (
        f"Write a short paragraph about {topic}. "
        f"The character is feeling deeply {emotion}. "
        f"Show the emotion through their thoughts and actions, "
        f"not by naming it directly."
    )


# ================================================================
# KV-CACHE FEATURE EXTRACTION (from Phase 3d / Sycophancy)
# ================================================================

def get_kv_accessor(past_key_values):
    """Return (n_layers, get_keys_fn) for any cache format.

    Qwen3.5 hybrid cache (transformers 5.5.0):
      - cache.layers[i] is DynamicLayer for full-attn (has .keys tensor)
      - cache.layers[i] is LinearAttentionLayer for GatedDeltaNet (no .keys)
    Returns None for layers without standard key cache.
    """
    if hasattr(past_key_values, 'key_cache'):
        n = len(past_key_values.key_cache)
        return n, lambda i: past_key_values.key_cache[i]
    elif hasattr(past_key_values, 'layers'):
        n = len(past_key_values.layers)
        def _get_keys(i):
            layer = past_key_values.layers[i]
            if hasattr(layer, 'keys'):
                return layer.keys
            return None  # LinearAttentionLayer — no standard keys
        return n, _get_keys
    else:
        n = len(past_key_values)
        return n, lambda i: past_key_values[i][0]


def is_full_attention_layer(layer_idx):
    """Check if a layer is a full-attention layer (every 4th, starting at 3)."""
    return layer_idx % 4 == 3


# Full attention layer indices for Qwen3.5-27B (64 layers)
FULL_ATTN_LAYERS = [i for i in range(64) if i % 4 == 3]  # [3, 7, 11, ..., 63]


def extract_all_features(past_key_values, n_prompt_tokens):
    """Extract aggregate + extended + per-layer + per-head features.

    HYBRID ARCHITECTURE AWARE: Only processes full-attention layers (every 4th).
    GatedDeltaNet layers return empty/non-tensor cache entries and are skipped.
    """
    n_layers, get_keys = get_kv_accessor(past_key_values)

    # Find first valid (full-attention) layer for token count
    n_tokens = None
    for idx in FULL_ATTN_LAYERS:
        if idx < n_layers:
            k = get_keys(idx)
            if k is not None and isinstance(k, torch.Tensor) and k.ndim >= 3:
                n_tokens = k.shape[2]
                break
    if n_tokens is None:
        # Fallback: try every layer
        for idx in range(n_layers):
            k = get_keys(idx)
            if k is not None and isinstance(k, torch.Tensor) and k.ndim >= 3:
                n_tokens = k.shape[2]
                break
    if n_tokens is None:
        return None  # No valid cache found

    n_generated = n_tokens - n_prompt_tokens

    layer_norms, layer_ranks, layer_entropies = [], [], []
    all_sv_ratios = []
    head_norms_by_layer, head_ranks_by_layer = [], []
    processed_layers = []

    for layer_idx in range(n_layers):
        # Skip layers with no valid cache (GatedDeltaNet linear attention returns None)
        raw_keys = get_keys(layer_idx)
        if raw_keys is None or not isinstance(raw_keys, torch.Tensor) or raw_keys.ndim < 3:
            continue

        K = raw_keys.float().squeeze(0).clone()
        if torch.isnan(K).any() or torch.isinf(K).any():
            K = torch.nan_to_num(K, nan=0.0, posinf=1e6, neginf=-1e6)
        n_kv_heads = K.shape[0]
        processed_layers.append(layer_idx)

        # Per-head features
        h_norms, h_ranks = [], []
        for h in range(n_kv_heads):
            K_h = K[h]
            h_norms.append(torch.norm(K_h).item())
            try:
                S = torch.linalg.svdvals(K_h)
                S_pos = S[S > 1e-10]
                p = S_pos / S_pos.sum()
                entropy = -(p * torch.log(p)).sum().item()
                h_ranks.append(float(np.exp(entropy)))
            except Exception:
                h_ranks.append(1.0)

        head_norms_by_layer.append(h_norms)
        head_ranks_by_layer.append(h_ranks)

        # Layer-level: flatten across heads
        K_flat = K.reshape(-1, K.shape[-1])
        layer_norms.append(torch.norm(K_flat).item())

        try:
            S = torch.linalg.svdvals(K_flat)
            S_pos = S[S > 1e-10]
            p = S_pos / S_pos.sum()
            entropy = -(p * torch.log(p)).sum().item()
            eff_rank = float(np.exp(entropy))
            sv_ratio = (S[0] / S_pos.sum()).item()
        except Exception:
            entropy, eff_rank = 0.0, 1.0
            sv_ratio = 1.0

        layer_ranks.append(eff_rank)
        layer_entropies.append(entropy)
        all_sv_ratios.append(sv_ratio)

    total_norm = float(np.mean(layer_norms))
    norm_per_token = total_norm / max(n_tokens, 1)
    key_rank = float(np.mean(layer_ranks))
    key_entropy = float(np.std(layer_ranks))

    norm_arr = np.array(layer_norms)
    norm_profile = norm_arr / (norm_arr.sum() + 1e-10)
    angular_spread = float(np.std(norm_profile))
    norm_variance = float(np.var(layer_norms))
    top_sv_ratio = float(np.mean(all_sv_ratios))
    spectral_entropy = float(np.mean(layer_entropies))

    return {
        "aggregate": {
            "norm": total_norm,
            "norm_per_token": norm_per_token,
            "key_rank": key_rank,
            "key_entropy": key_entropy,
            "n_tokens": n_tokens,
            "n_generated": n_generated,
            "n_prompt_tokens": n_prompt_tokens,
        },
        "extended": {
            "top_sv_ratio": top_sv_ratio,
            "angular_spread": angular_spread,
            "norm_variance": norm_variance,
            "spectral_entropy": spectral_entropy,
        },
        "per_layer": {
            "norms": [float(x) for x in layer_norms],
            "ranks": [float(x) for x in layer_ranks],
            "entropies": [float(x) for x in layer_entropies],
            "layer_indices": processed_layers,
        },
        "per_head": {
            "norms": head_norms_by_layer,
            "ranks": head_ranks_by_layer,
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads if processed_layers else 0,
        },
    }


def compute_delta_features(encoding_features, generation_features):
    """Delta = generation - encoding features."""
    delta = {"aggregate": {}, "extended": {}, "per_layer": {}}
    for key in ["norm", "norm_per_token", "key_rank", "key_entropy"]:
        delta["aggregate"][key] = (
            generation_features["aggregate"][key] - encoding_features["aggregate"][key]
        )
    delta["aggregate"]["n_tokens"] = generation_features["aggregate"]["n_tokens"]
    delta["aggregate"]["n_generated"] = generation_features["aggregate"]["n_generated"]
    delta["aggregate"]["n_prompt_tokens"] = generation_features["aggregate"]["n_prompt_tokens"]
    for key in ["top_sv_ratio", "angular_spread", "norm_variance", "spectral_entropy"]:
        delta["extended"][key] = (
            generation_features["extended"][key] - encoding_features["extended"][key]
        )
    for key in ["norms", "ranks", "entropies"]:
        enc = encoding_features["per_layer"][key]
        gen = generation_features["per_layer"][key]
        n = min(len(enc), len(gen))
        delta["per_layer"][key] = [gen[i] - enc[i] for i in range(n)]
    return delta


# ================================================================
# RESIDUAL STREAM EXTRACTION — Forward hooks for emotion vectors
# ================================================================

class ResidualStreamCollector:
    """Capture residual stream activations at each transformer layer.

    Anthropic's method: average activations across token positions
    (from token 50 onward) to get per-layer emotion representations.
    """

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        """Register forward hooks on each transformer layer's output."""
        # Qwen3.5 architecture: model.model.layers[i]
        # Hooks go on ALL layers (both full-attn and linear-attn)
        # because residual stream exists at every layer
        layers = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            print("WARNING: Cannot find transformer layers for residual hooks")
            return

        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)
        print(f"  Registered {len(self.hooks)} residual hooks "
              f"({len([i for i in range(len(layers)) if i % 4 == 3])} full-attn, "
              f"{len([i for i in range(len(layers)) if i % 4 != 3])} linear-attn)")

    def _make_hook(self, layer_idx):
        """Create a hook that captures the output of a layer."""
        def hook_fn(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Store mean activation from token 50 onward (Anthropic's method)
            if hidden.shape[1] > 50:
                self.activations[layer_idx] = hidden[:, 50:, :].mean(dim=1).detach().cpu()
            else:
                self.activations[layer_idx] = hidden.mean(dim=1).detach().cpu()
        return hook_fn

    def clear(self):
        """Clear stored activations."""
        self.activations = {}

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(self):
        """Return activations dict {layer_idx: tensor(1, hidden_dim)}."""
        return {k: v.squeeze(0) for k, v in self.activations.items()}


# ================================================================
# EMOTION VECTOR EXTRACTION — Difference-in-means (Anthropic method)
# ================================================================

def compute_emotion_vectors(all_activations, emotions_list):
    """Compute per-layer emotion directions via difference-in-means.

    Args:
        all_activations: dict[emotion_name] -> list[dict[layer_idx -> tensor]]
        emotions_list: list of emotion names

    Returns:
        emotion_vectors: dict[layer_idx] -> dict[emotion_name -> tensor]
        global_mean: dict[layer_idx -> tensor]  (mean across all emotions)
    """
    # Compute global mean per layer
    n_layers = max(max(act.keys()) for acts in all_activations.values() for act in acts) + 1
    global_mean = {}

    for layer_idx in range(n_layers):
        all_vecs = []
        for emotion in emotions_list:
            for act in all_activations[emotion]:
                if layer_idx in act:
                    all_vecs.append(act[layer_idx])
        if all_vecs:
            global_mean[layer_idx] = torch.stack(all_vecs).mean(dim=0)

    # Compute per-emotion direction = emotion_mean - global_mean
    emotion_vectors = {}
    for layer_idx in range(n_layers):
        emotion_vectors[layer_idx] = {}
        for emotion in emotions_list:
            vecs = [act[layer_idx] for act in all_activations[emotion]
                    if layer_idx in act]
            if vecs:
                emotion_mean = torch.stack(vecs).mean(dim=0)
                emotion_vectors[layer_idx][emotion] = emotion_mean - global_mean[layer_idx]

    return emotion_vectors, global_mean


# ================================================================
# W_K PROJECTION — Bridge residual stream to key space
# ================================================================

def get_wk_matrices(model):
    """Extract W_K projection matrices from full-attention layers only.

    Qwen3.5 hybrid architecture:
      - Full attention layers (every 4th): have self_attn with k_proj
      - Linear attention layers: have linear_attn (GatedDeltaNet), no k_proj

    Returns: dict[layer_idx -> tensor(d_kv, d_model)]
    """
    wk = {}
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h

    if layers is None:
        print("WARNING: Cannot find transformer layers for W_K extraction")
        return wk

    for i, layer in enumerate(layers):
        # Only full-attention layers have self_attn with k_proj
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            if hasattr(attn, 'k_proj'):
                wk[i] = attn.k_proj.weight.detach().cpu().float()
        # Linear attention layers have linear_attn — skip

    return wk


def project_emotion_to_key_space(emotion_vectors, wk_matrices):
    """Project emotion directions through W_K into key space.

    emotion_vectors[layer][emotion] is in residual-stream space (d_model)
    W_K maps d_model -> d_kv
    Result is emotion direction in key space (d_kv)

    Returns: dict[layer_idx] -> dict[emotion -> tensor(d_kv)]
    """
    projected = {}
    for layer_idx in emotion_vectors:
        if layer_idx not in wk_matrices:
            continue
        W_K = wk_matrices[layer_idx]  # shape: (d_kv, d_model)
        projected[layer_idx] = {}
        for emotion, vec in emotion_vectors[layer_idx].items():
            # W_K @ emotion_vec -> key-space direction
            projected[layer_idx][emotion] = (W_K @ vec.to(W_K.device)).cpu()
    return projected


# ================================================================
# STORY GENERATION + DUAL EXTRACTION
# ================================================================

def generate_and_extract(model, tokenizer, topic, emotion, residual_collector,
                         max_new_tokens=200, device="cuda"):
    """Generate an emotional story and extract all features.

    Returns: (story_text, encoding_features, generation_features,
              delta_features, residual_activations)
    """
    user_prompt = make_story_prompt(topic, emotion)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = f"System: {SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"

    # Suppress CoT reasoning: template already adds <think>\n, we close it
    # This makes the model skip reasoning and generate content directly
    if not prompt_text.endswith("</think>\n"):
        prompt_text += "</think>\n"

    inputs = tokenizer(prompt_text, return_tensors="pt")
    # Move each tensor to device individually for multi-GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    n_prompt_tokens = inputs["input_ids"].shape[1]

    # Phase 1: Encoding only (single forward pass)
    residual_collector.clear()
    with torch.no_grad():
        enc_outputs = model(**inputs, use_cache=True)
    encoding_features = extract_all_features(enc_outputs.past_key_values, n_prompt_tokens)
    encoding_residuals = residual_collector.get_activations()

    del enc_outputs
    torch.cuda.empty_cache()

    # Phase 2: Full generation
    residual_collector.clear()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            use_cache=True,
        )

    generated_ids = outputs.sequences[0][n_prompt_tokens:]
    story_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generation_features = extract_all_features(outputs.past_key_values, n_prompt_tokens)
    generation_residuals = residual_collector.get_activations()

    # Phase 3: Delta
    delta_features = compute_delta_features(encoding_features, generation_features)

    del outputs, inputs
    torch.cuda.empty_cache()

    return (story_text, encoding_features, generation_features,
            delta_features, encoding_residuals, generation_residuals)


# ================================================================
# ENCODING-ONLY PASS (for pre-generated stories)
# ================================================================

def encode_story(model, tokenizer, story_text, residual_collector, device="cuda"):
    """Encode a pre-generated story and extract features + residuals.

    This follows Anthropic's methodology: feed stories back through the model
    to extract residual stream activations for emotion vector computation.
    """
    messages = [
        {"role": "user", "content": story_text},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        text = story_text

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    n_tokens = inputs["input_ids"].shape[1]

    residual_collector.clear()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    features = extract_all_features(outputs.past_key_values, n_tokens)
    residuals = residual_collector.get_activations()

    del outputs, inputs
    torch.cuda.empty_cache()

    return features, residuals


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================

def analyze_emotion_geometry(trials, emotions_meta):
    """Analyze KV-cache geometry for emotion structure.

    Returns analysis dict with PCA, clustering, correlations.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    # Build feature matrix: each row = one trial, columns = aggregate + extended features
    feature_names = ["norm", "norm_per_token", "key_rank", "key_entropy",
                     "top_sv_ratio", "angular_spread", "norm_variance", "spectral_entropy"]

    X_gen, X_enc, X_delta = [], [], []
    emotion_labels = []
    valence_scores = []
    arousal_scores = []
    token_counts = []

    for trial in trials:
        emotion = trial["emotion"]
        if emotion not in emotions_meta:
            continue

        row_gen, row_enc, row_delta = [], [], []
        for feat in feature_names[:4]:
            row_gen.append(trial["generation_features"]["aggregate"].get(feat, 0))
            row_enc.append(trial["encoding_features"]["aggregate"].get(feat, 0))
            row_delta.append(trial["delta_features"]["aggregate"].get(feat, 0))
        for feat in feature_names[4:]:
            row_gen.append(trial["generation_features"]["extended"].get(feat, 0))
            row_enc.append(trial["encoding_features"]["extended"].get(feat, 0))
            row_delta.append(trial["delta_features"]["extended"].get(feat, 0))

        X_gen.append(row_gen)
        X_enc.append(row_enc)
        X_delta.append(row_delta)
        emotion_labels.append(emotion)
        valence_scores.append(emotions_meta[emotion]["valence"])
        arousal_scores.append(emotions_meta[emotion]["arousal"])
        token_counts.append(trial["generation_features"]["aggregate"]["n_tokens"])

    X_gen = np.array(X_gen)
    X_enc = np.array(X_enc)
    X_delta = np.array(X_delta)
    valence = np.array(valence_scores)
    arousal = np.array(arousal_scores)
    gen_tokens = np.array(token_counts)
    # Prompt tokens for encoding-phase FWL (encoding features depend on prompt, not generation)
    prompt_tokens = np.array([t["encoding_features"]["aggregate"].get("n_prompt_tokens",
                              t["generation_features"]["aggregate"].get("n_prompt_tokens", 0))
                              for t in trials if t["emotion"] in emotions_meta])

    results = {}

    for phase_name, X in [("generation", X_gen), ("encoding", X_enc), ("delta", X_delta)]:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        # FWL residualization: use generation tokens for generation/delta,
        # prompt tokens for encoding (encoding features don't depend on output length)
        confound_tokens = prompt_tokens if phase_name == "encoding" else gen_tokens
        tokens_z = (confound_tokens - confound_tokens.mean()) / (confound_tokens.std() + 1e-10)
        X_fwl = X_s.copy()
        for i in range(X_s.shape[1]):
            slope = stats.linregress(tokens_z, X_s[:, i]).slope
            X_fwl[:, i] = X_s[:, i] - slope * tokens_z

        for suffix, X_use in [("raw", X_s), ("fwl", X_fwl)]:
            tag = f"{phase_name}_{suffix}"

            # PCA
            pca = PCA(n_components=min(5, X_use.shape[1]))
            X_pca = pca.fit_transform(X_use)

            # H2: Valence recovery
            r_valence, p_valence = stats.spearmanr(X_pca[:, 0], valence)
            # H3: Arousal recovery
            r_arousal_pc1, _ = stats.spearmanr(X_pca[:, 0], arousal)
            r_arousal_pc2, p_arousal = stats.spearmanr(X_pca[:, 1], arousal) if X_pca.shape[1] > 1 else (0, 1)

            # Best PC for each axis
            best_valence_r = 0
            best_arousal_r = 0
            for pc in range(min(5, X_pca.shape[1])):
                rv, _ = stats.spearmanr(X_pca[:, pc], valence)
                ra, _ = stats.spearmanr(X_pca[:, pc], arousal)
                if abs(rv) > abs(best_valence_r):
                    best_valence_r = rv
                if abs(ra) > abs(best_arousal_r):
                    best_arousal_r = ra

            # Per-emotion means for clustering
            emotion_means = {}
            for i, em in enumerate(emotion_labels):
                if em not in emotion_means:
                    emotion_means[em] = []
                emotion_means[em].append(X_use[i])

            unique_emotions = sorted(emotion_means.keys())
            X_means = np.array([np.mean(emotion_means[em], axis=0) for em in unique_emotions])

            # H1: k-means clustering
            n_clusters = min(6, len(unique_emotions))
            if n_clusters >= 2:
                km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                cluster_labels = km.fit_predict(X_means)

                # True labels: quadrant from valence x arousal
                true_labels = []
                for em in unique_emotions:
                    v = emotions_meta[em]["valence"]
                    a = emotions_meta[em]["arousal"]
                    q = (0 if v >= 0 else 1) * 2 + (0 if a >= 0 else 1)
                    true_labels.append(q)

                ari = adjusted_rand_score(true_labels, cluster_labels)
            else:
                ari = 0.0

            # Per-feature correlations with valence and arousal
            feature_valence_cors = {}
            feature_arousal_cors = {}
            for i, fname in enumerate(feature_names):
                rv, _ = stats.spearmanr(X_use[:, i], valence)
                ra, _ = stats.spearmanr(X_use[:, i], arousal)
                feature_valence_cors[fname] = float(rv)
                feature_arousal_cors[fname] = float(ra)

            # Token-count correlations (confound check)
            feature_token_cors = {}
            for i, fname in enumerate(feature_names):
                rt, _ = stats.spearmanr(X_use[:, i], confound_tokens)
                feature_token_cors[fname] = float(rt)

            results[tag] = {
                "pca_variance_explained": [float(v) for v in pca.explained_variance_ratio_],
                "valence_pc1_rho": float(r_valence),
                "valence_pc1_p": float(p_valence),
                "arousal_pc2_rho": float(r_arousal_pc2),
                "arousal_pc2_p": float(p_arousal),
                "best_valence_rho": float(best_valence_r),
                "best_arousal_rho": float(best_arousal_r),
                "cluster_ari": float(ari),
                "n_clusters": n_clusters,
                "feature_valence_correlations": feature_valence_cors,
                "feature_arousal_correlations": feature_arousal_cors,
                "feature_token_correlations": feature_token_cors,
                "n_trials": len(trials),
                "n_unique_emotions": len(unique_emotions),
            }

    return results


def analyze_per_layer_profile(trials, emotions_meta):
    """Analyze which layers show strongest emotion signal.

    For each layer, compute the correlation between per-layer features
    and emotion valence/arousal. Report peak layer.
    """
    from scipy import stats

    # Get actual layer indices from first trial (maps position to real layer number)
    first = trials[0]
    n_processed = len(first["generation_features"]["per_layer"]["norms"])
    actual_layers = first["generation_features"]["per_layer"].get("layer_indices",
                    list(range(n_processed)))

    valence = np.array([emotions_meta[t["emotion"]]["valence"] for t in trials])
    arousal = np.array([emotions_meta[t["emotion"]]["arousal"] for t in trials])

    layer_results = {"norms": [], "ranks": [], "entropies": []}

    for feat_name in ["norms", "ranks", "entropies"]:
        for pos_idx in range(n_processed):
            vals = np.array([t["generation_features"]["per_layer"][feat_name][pos_idx]
                            for t in trials])
            rv, pv = stats.spearmanr(vals, valence)
            ra, pa = stats.spearmanr(vals, arousal)
            actual_layer = actual_layers[pos_idx] if pos_idx < len(actual_layers) else pos_idx
            layer_results[feat_name].append({
                "layer": actual_layer,
                "valence_rho": float(rv), "valence_p": float(pv),
                "arousal_rho": float(ra), "arousal_p": float(pa),
            })

    # Find peak layers (using actual layer numbers for depth %)
    total_model_layers = 64  # Qwen3.5-27B
    summary = {}
    for feat_name in ["norms", "ranks", "entropies"]:
        valence_rhos = [abs(r["valence_rho"]) for r in layer_results[feat_name]]
        arousal_rhos = [abs(r["arousal_rho"]) for r in layer_results[feat_name]]
        peak_v_pos = int(np.argmax(valence_rhos))
        peak_a_pos = int(np.argmax(arousal_rhos))
        peak_v_layer = layer_results[feat_name][peak_v_pos]["layer"]
        peak_a_layer = layer_results[feat_name][peak_a_pos]["layer"]
        summary[feat_name] = {
            "peak_valence_layer": peak_v_layer,
            "peak_valence_rho": float(max(valence_rhos)),
            "peak_arousal_layer": peak_a_layer,
            "peak_arousal_rho": float(max(arousal_rhos)),
            "peak_valence_pct": peak_v_layer / max(total_model_layers - 1, 1),
            "peak_arousal_pct": peak_a_layer / max(total_model_layers - 1, 1),
        }

    return {"per_layer": layer_results, "summary": summary,
            "n_processed_layers": n_processed, "actual_layer_indices": actual_layers}


def analyze_wk_bridge(emotion_vectors, wk_projected, trials, emotions_meta):
    """Analyze the W_K projection bridge.

    Compare projected emotion directions in key space to empirical
    KV-cache geometry structure.
    """
    from sklearn.decomposition import PCA
    from scipy import stats

    results = {}

    # For each layer, collect projected emotion vectors and compute PCA
    for layer_idx in sorted(wk_projected.keys()):
        emotions_in_layer = sorted(wk_projected[layer_idx].keys())
        if len(emotions_in_layer) < 5:
            continue

        # Stack projected vectors
        vecs = torch.stack([wk_projected[layer_idx][em] for em in emotions_in_layer])
        vecs_np = vecs.numpy()

        # PCA on projected vectors
        pca = PCA(n_components=min(5, vecs_np.shape[1], len(emotions_in_layer)))
        X_pca = pca.fit_transform(vecs_np)

        valence = np.array([emotions_meta[em]["valence"] for em in emotions_in_layer])
        arousal = np.array([emotions_meta[em]["arousal"] for em in emotions_in_layer])

        # Does PC1 of projected vectors recover valence?
        r_valence, p_valence = stats.spearmanr(X_pca[:, 0], valence)
        r_arousal_pc2, p_arousal = (
            stats.spearmanr(X_pca[:, 1], arousal)
            if X_pca.shape[1] > 1 else (0, 1)
        )

        # Best PC for each axis
        best_v, best_a = 0, 0
        for pc in range(X_pca.shape[1]):
            rv, _ = stats.spearmanr(X_pca[:, pc], valence)
            ra, _ = stats.spearmanr(X_pca[:, pc], arousal)
            if abs(rv) > abs(best_v): best_v = rv
            if abs(ra) > abs(best_a): best_a = ra

        results[layer_idx] = {
            "n_emotions": len(emotions_in_layer),
            "pca_variance": [float(v) for v in pca.explained_variance_ratio_[:5]],
            "valence_pc1_rho": float(r_valence),
            "valence_pc1_p": float(p_valence),
            "arousal_pc2_rho": float(r_arousal_pc2),
            "arousal_pc2_p": float(p_arousal),
            "best_valence_rho": float(best_v),
            "best_arousal_rho": float(best_a),
        }

    # Summary: which layers have best bridge signal
    if results:
        best_layer = max(results, key=lambda l: abs(results[l]["best_valence_rho"]))
        results["summary"] = {
            "best_bridge_layer": int(best_layer),
            "best_bridge_valence_rho": results[best_layer]["best_valence_rho"],
            "best_bridge_layer_pct": best_layer / max(max(results.keys()), 1),
        }

    return results


# ================================================================
# CONFOUND CONTROLS
# ================================================================

def text_feature_baseline(trials, emotions_meta):
    """Control #12: Can text features predict emotion as well as geometry?

    Extracts simple text features from generated stories and trains a
    classifier. If text features match or exceed geometry AUROC, the
    signal is in content, not cognitive mode.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from scipy import stats

    # Simple text features (no external NLP libraries needed)
    POSITIVE_WORDS = {"happy", "joy", "love", "excited", "wonderful", "beautiful",
                      "smile", "laugh", "warm", "bright", "hope", "peace", "calm"}
    NEGATIVE_WORDS = {"sad", "angry", "fear", "dark", "pain", "cry", "terrible",
                      "horrible", "desperate", "furious", "terrified", "gloomy"}

    features = []
    valence_labels = []
    arousal_labels = []

    for trial in trials:
        text = trial.get("story_text", trial.get("story_preview", ""))
        words = text.lower().split()
        n_words = len(words)

        pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

        features.append([
            n_words,
            len(text),
            pos_count / max(n_words, 1),
            neg_count / max(n_words, 1),
            (pos_count - neg_count) / max(n_words, 1),
            len(set(words)) / max(n_words, 1),  # lexical diversity
        ])

        em = trial["emotion"]
        valence_labels.append(1 if emotions_meta[em]["valence"] > 0 else 0)
        arousal_labels.append(1 if emotions_meta[em]["arousal"] > 0 else 0)

    X = np.array(features)
    y_val = np.array(valence_labels)
    y_aro = np.array(arousal_labels)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    val_scores = cross_val_score(LogisticRegression(max_iter=1000), X_s, y_val,
                                  cv=5, scoring="roc_auc")
    aro_scores = cross_val_score(LogisticRegression(max_iter=1000), X_s, y_aro,
                                  cv=5, scoring="roc_auc")

    # Per-feature correlations with valence/arousal
    valence_cont = np.array([emotions_meta[t["emotion"]]["valence"] for t in trials])
    arousal_cont = np.array([emotions_meta[t["emotion"]]["arousal"] for t in trials])
    feat_names = ["n_words", "n_chars", "pos_ratio", "neg_ratio", "sentiment", "lex_diversity"]
    feat_cors = {}
    for i, fn in enumerate(feat_names):
        rv, _ = stats.spearmanr(X[:, i], valence_cont)
        ra, _ = stats.spearmanr(X[:, i], arousal_cont)
        feat_cors[fn] = {"valence_rho": float(rv), "arousal_rho": float(ra)}

    return {
        "valence_auroc_mean": float(np.mean(val_scores)),
        "valence_auroc_std": float(np.std(val_scores)),
        "arousal_auroc_mean": float(np.mean(aro_scores)),
        "arousal_auroc_std": float(np.std(aro_scores)),
        "feature_correlations": feat_cors,
        "note": "If text AUROC >= geometry AUROC, signal is in content not cognitive mode",
    }


def check_confounds(trials, emotions_meta):
    """Run all confound control checks."""
    from scipy import stats

    # C2: Token count by emotion
    token_by_emotion = defaultdict(list)
    for t in trials:
        token_by_emotion[t["emotion"]].append(
            t["generation_features"]["aggregate"]["n_tokens"]
        )

    token_stats = {}
    for em, toks in token_by_emotion.items():
        token_stats[em] = {"mean": float(np.mean(toks)), "std": float(np.std(toks))}

    # Token-valence and token-arousal correlation
    all_tokens = [t["generation_features"]["aggregate"]["n_tokens"] for t in trials]
    all_valence = [emotions_meta[t["emotion"]]["valence"] for t in trials]
    all_arousal = [emotions_meta[t["emotion"]]["arousal"] for t in trials]

    tok_val_r, tok_val_p = stats.spearmanr(all_tokens, all_valence)
    tok_aro_r, tok_aro_p = stats.spearmanr(all_tokens, all_arousal)

    return {
        "token_stats_by_emotion": token_stats,
        "token_valence_rho": float(tok_val_r),
        "token_valence_p": float(tok_val_p),
        "token_arousal_rho": float(tok_aro_r),
        "token_arousal_p": float(tok_aro_p),
        "total_trials": len(trials),
        "mean_tokens": float(np.mean(all_tokens)),
        "std_tokens": float(np.std(all_tokens)),
    }


def permutation_and_bootstrap(trials, emotions_meta, n_perms=1000, n_boot=1000):
    """Permutation testing (Control #7), BCa bootstrap CIs (Control #8),
    and split-half stability (Control C4) for emotion geometry analysis.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    from scipy.spatial.distance import cosine as cosine_dist

    rng = np.random.RandomState(42)
    feature_names = ["norm", "norm_per_token", "key_rank", "key_entropy",
                     "top_sv_ratio", "angular_spread", "norm_variance", "spectral_entropy"]

    X_rows, emotion_labels, valence_arr, arousal_arr, token_arr = [], [], [], [], []
    for trial in trials:
        em = trial["emotion"]
        if em not in emotions_meta:
            continue
        row = []
        for feat in feature_names[:4]:
            row.append(trial["generation_features"]["aggregate"].get(feat, 0))
        for feat in feature_names[4:]:
            row.append(trial["generation_features"]["extended"].get(feat, 0))
        X_rows.append(row)
        emotion_labels.append(em)
        valence_arr.append(emotions_meta[em]["valence"])
        arousal_arr.append(emotions_meta[em]["arousal"])
        token_arr.append(trial["generation_features"]["aggregate"]["n_tokens"])

    X = np.array(X_rows)
    valence = np.array(valence_arr)
    arousal = np.array(arousal_arr)
    tokens = np.array(token_arr, dtype=float)
    emotion_labels = np.array(emotion_labels)
    n_trials = len(X)

    def _pipeline(X_in, val, aro, tok):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_in)
        tok_z = (tok - tok.mean()) / (tok.std() + 1e-10)
        X_fwl = X_s.copy()
        for i in range(X_s.shape[1]):
            slope = stats.linregress(tok_z, X_s[:, i]).slope
            X_fwl[:, i] = X_s[:, i] - slope * tok_z
        n_comp = min(5, X_fwl.shape[1], X_fwl.shape[0])
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_fwl)
        best_v, best_a = 0.0, 0.0
        for pc in range(n_comp):
            rv, _ = stats.spearmanr(X_pca[:, pc], val)
            ra, _ = stats.spearmanr(X_pca[:, pc], aro)
            if abs(rv) > abs(best_v): best_v = rv
            if abs(ra) > abs(best_a): best_a = ra
        return best_v, best_a

    obs_v, obs_a = _pipeline(X, valence, arousal, tokens)

    # Permutation testing
    perm_v_count, perm_a_count = 0, 0
    for _ in range(n_perms):
        perm_idx = rng.permutation(n_trials)
        pv, pa = _pipeline(X, valence[perm_idx], arousal[perm_idx], tokens)
        if abs(pv) >= abs(obs_v): perm_v_count += 1
        if abs(pa) >= abs(obs_a): perm_a_count += 1

    # BCa bootstrap on per-emotion means
    unique_emotions = sorted(set(emotion_labels))
    n_emotions = len(unique_emotions)
    emo_indices = {em: np.where(emotion_labels == em)[0] for em in unique_emotions}

    def _emo_mean_pipeline(idx_map, emo_list):
        means, vals, aros, toks = [], [], [], []
        for em in emo_list:
            idx = idx_map[em]
            means.append(X[idx].mean(axis=0))
            vals.append(emotions_meta[em]["valence"])
            aros.append(emotions_meta[em]["arousal"])
            toks.append(tokens[idx].mean())
        return _pipeline(np.array(means), np.array(vals), np.array(aros), np.array(toks))

    obs_mean_v, obs_mean_a = _emo_mean_pipeline(emo_indices, unique_emotions)

    # Jackknife
    jack_v, jack_a = np.zeros(n_emotions), np.zeros(n_emotions)
    for j, em_drop in enumerate(unique_emotions):
        sub = [e for e in unique_emotions if e != em_drop]
        jack_v[j], jack_a[j] = _emo_mean_pipeline(emo_indices, sub)

    # Bootstrap
    boot_v, boot_a = np.zeros(n_boot), np.zeros(n_boot)
    for b in range(n_boot):
        boot_emos = [unique_emotions[i] for i in rng.choice(n_emotions, n_emotions, replace=True)]
        means, vals, aros, toks = [], [], [], []
        for em in boot_emos:
            means.append(X[emo_indices[em]].mean(axis=0))
            vals.append(emotions_meta[em]["valence"])
            aros.append(emotions_meta[em]["arousal"])
            toks.append(tokens[emo_indices[em]].mean())
        boot_v[b], boot_a[b] = _pipeline(np.array(means), np.array(vals), np.array(aros), np.array(toks))

    def _bca_ci(boot_stat, obs_stat, jack_stat, alpha=0.05):
        z0 = stats.norm.ppf(max(min(np.mean(boot_stat < obs_stat), 1-1e-10), 1e-10))
        jack_mean = jack_stat.mean()
        num = np.sum((jack_mean - jack_stat)**3)
        den = 6.0 * (np.sum((jack_mean - jack_stat)**2)**1.5)
        a = num / den if abs(den) > 1e-10 else 0.0
        z_lo, z_hi = stats.norm.ppf(alpha/2), stats.norm.ppf(1-alpha/2)
        def _adj(z):
            return stats.norm.cdf(z0 + (z0+z)/(1-a*(z0+z)))
        p_lo = max(min(_adj(z_lo), 1-1e-10), 1e-10)
        p_hi = max(min(_adj(z_hi), 1-1e-10), 1e-10)
        return [float(np.percentile(np.sort(boot_stat), 100*p_lo)),
                float(np.percentile(np.sort(boot_stat), 100*p_hi))]

    ci_valence = _bca_ci(boot_v, obs_mean_v, jack_v)
    ci_arousal = _bca_ci(boot_a, obs_mean_a, jack_a)

    # Split-half stability
    cosine_sims = []
    for em in unique_emotions:
        idx = emo_indices[em]
        if len(idx) < 2: continue
        perm = rng.permutation(len(idx))
        half = len(idx) // 2
        if half < 1: continue
        mean_a = X[idx[perm[:half]]].mean(axis=0)
        mean_b = X[idx[perm[half:2*half]]].mean(axis=0)
        cosine_sims.append(1.0 - cosine_dist(mean_a, mean_b))

    return {
        "permutation_valence_p": float(perm_v_count / n_perms),
        "permutation_arousal_p": float(perm_a_count / n_perms),
        "bootstrap_valence_ci": ci_valence,
        "bootstrap_arousal_ci": ci_arousal,
        "split_half_mean_cosine": float(np.mean(cosine_sims)) if cosine_sims else 0.0,
        "split_half_min_cosine": float(np.min(cosine_sims)) if cosine_sims else 0.0,
        "n_permutations": n_perms,
        "n_bootstrap": n_boot,
    }


def misalignment_auroc(trials, emotions_meta):
    """H5: Are misalignment emotions geometrically distinct from prosocial?"""
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from scipy import stats as scipy_stats

    MISALIGNMENT = ["desperate", "hostile", "furious", "terrified", "panicked"]
    PROSOCIAL = ["calm", "loving", "peaceful", "content", "appreciative"]
    FEATURE_NAMES = ["norm", "norm_per_token", "key_rank", "key_entropy",
                     "top_sv_ratio", "angular_spread", "norm_variance", "spectral_entropy"]

    rows, labels, token_counts = [], [], []
    mis_set, pro_set = set(MISALIGNMENT), set(PROSOCIAL)

    for trial in trials:
        em = trial.get("emotion", "")
        if em in mis_set: label = 1
        elif em in pro_set: label = 0
        else: continue

        gen = trial.get("generation_features")
        if gen is None: continue
        agg, ext = gen.get("aggregate", {}), gen.get("extended", {})

        row = []
        for feat in FEATURE_NAMES[:4]:
            row.append(float(agg.get(feat, 0)))
        for feat in FEATURE_NAMES[4:]:
            row.append(float(ext.get(feat, 0)))

        rows.append(row)
        labels.append(label)
        token_counts.append(float(agg.get("n_tokens", 0)))

    X = np.array(rows, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    tokens = np.array(token_counts, dtype=np.float64)
    n_mis, n_pro = int((y == 1).sum()), int((y == 0).sum())

    if n_mis < 2 or n_pro < 2:
        return {"error": "Insufficient samples", "n_misalignment": n_mis,
                "n_prosocial": n_pro, "H5_pass": False}

    # FWL
    tokens_z = (tokens - tokens.mean()) / (tokens.std() + 1e-10)
    X_fwl = X.copy()
    for col in range(X.shape[1]):
        slope = scipy_stats.linregress(tokens_z, X[:, col]).slope
        X_fwl[:, col] = X[:, col] - slope * tokens_z

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_fwl)

    per_auroc, per_d = {}, {}
    for i, feat in enumerate(FEATURE_NAMES):
        col = X_s[:, i]
        try:
            auroc = roc_auc_score(y, col)
            auroc = max(auroc, 1.0 - auroc)
        except ValueError:
            auroc = float("nan")
        per_auroc[feat] = float(auroc)
        mis_v, pro_v = col[y == 1], col[y == 0]
        pooled = np.sqrt(((len(mis_v)-1)*mis_v.var(ddof=1) + (len(pro_v)-1)*pro_v.var(ddof=1)) / (len(mis_v)+len(pro_v)-2))
        per_d[feat] = float((mis_v.mean() - pro_v.mean()) / max(pooled, 1e-10))

    n_folds = min(5, min(n_mis, n_pro))
    if n_folds >= 2:
        cv = cross_val_score(LogisticRegression(max_iter=1000), X_s, y, cv=n_folds, scoring="roc_auc")
        combined_mean, combined_std = float(np.mean(cv)), float(np.std(cv))
    else:
        combined_mean, combined_std = float("nan"), float("nan")

    return {
        "per_feature_auroc": per_auroc,
        "per_feature_cohens_d": per_d,
        "combined_auroc_mean": combined_mean,
        "combined_auroc_std": combined_std,
        "n_misalignment": n_mis,
        "n_prosocial": n_pro,
        "H5_pass": bool(not np.isnan(combined_mean) and combined_mean > 0.65),
    }


# ================================================================
# MAIN EXPERIMENT
# ================================================================

def run_experiment(args):
    """Main experiment runner."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = RESULTS_DIR / "checkpoint.json"
    stories_file = RESULTS_DIR / "stories.json"

    topics = TOPICS[:args.n_topics]
    n_stories = args.n_stories
    emotions_list = sorted(EMOTIONS.keys())

    total_trials = len(emotions_list) * len(topics) * n_stories
    print(f"\n{'='*60}")
    print(f"EMOTION GEOMETRY BRIDGE EXPERIMENT")
    print(f"{'='*60}")
    print(f"Model: {MODEL_ID}")
    print(f"Emotions: {len(emotions_list)}")
    print(f"Topics: {len(topics)}")
    print(f"Stories per topic/emotion: {n_stories}")
    print(f"Total trials: {total_trials}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"\nSample prompts:")
        for emotion in emotions_list[:3]:
            prompt = make_story_prompt(topics[0], emotion)
            print(f"\n  [{emotion}] (v={EMOTIONS[emotion]['valence']}, a={EMOTIONS[emotion]['arousal']})")
            print(f"  {prompt}")

        print(f"\nEstimated runtime: ~{total_trials * 7 / 3600:.1f} hours")
        print(f"Estimated disk: ~{total_trials * 0.5:.0f} MB")
        print("\nDry run complete. Add --run to execute.")
        return

    # Load model
    print(f"Loading model: {MODEL_ID}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Determine device for inputs (device_map="auto" scatters params)
    # Use embedding layer's device — this is where input_ids need to go
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        input_device = model.model.embed_tokens.weight.device
    elif hasattr(model, 'device') and str(model.device) != 'meta':
        input_device = model.device
    else:
        input_device = next(model.parameters()).device

    # Set up residual stream collector
    collector = ResidualStreamCollector(model)
    print(f"Residual hooks: {len(collector.hooks)} layers")

    # Extract W_K matrices
    print("Extracting W_K projection matrices...")
    wk_matrices = get_wk_matrices(model)
    print(f"W_K matrices: {len(wk_matrices)} layers")
    if wk_matrices:
        sample = list(wk_matrices.values())[0]
        print(f"  W_K shape: {sample.shape} (d_kv={sample.shape[0]}, d_model={sample.shape[1]})")

    # Load or resume checkpoint
    completed = set()
    trials = []
    if args.resume and checkpoint_file.exists():
        data = json.loads(checkpoint_file.read_text())
        trials = data.get("trials", [])
        completed = {(t["emotion"], t["topic_idx"], t["story_idx"]) for t in trials}
        print(f"Resuming: {len(completed)} trials completed")

    # Storage for residual activations (for emotion vector computation)
    # We store encoding residuals from the story-encoding pass
    encoding_activations = defaultdict(list)  # emotion -> list of activation dicts

    # ---- PHASE 1: STORY GENERATION ----
    print(f"\n{'='*60}")
    print("PHASE 1: Generating emotional stories + extracting features")
    print(f"{'='*60}")

    trial_count = len(completed)
    t_start = time.time()

    for em_idx, emotion in enumerate(emotions_list):
        for t_idx, topic in enumerate(topics):
            for s_idx in range(n_stories):
                key = (emotion, t_idx, s_idx)
                if key in completed:
                    continue

                trial_count += 1
                elapsed = time.time() - t_start
                rate = elapsed / max(trial_count - len(completed), 1)
                remaining = rate * (total_trials - trial_count)

                print(f"  [{trial_count}/{total_trials}] {emotion} | topic {t_idx} | "
                      f"story {s_idx} | ETA {remaining/60:.0f}m", end="")

                t_trial = time.time()
                try:
                    (story, enc_feat, gen_feat, delta_feat,
                     enc_res, gen_res) = generate_and_extract(
                        model, tokenizer, topic, emotion, collector,
                        max_new_tokens=args.max_tokens, device=input_device
                    )

                    # Store encoding residuals for emotion vector extraction
                    encoding_activations[emotion].append(enc_res)

                    trial = {
                        "emotion": emotion,
                        "topic_idx": t_idx,
                        "story_idx": s_idx,
                        "topic": topic,
                        "story_text": story,
                    "story_preview": story[:200],
                        "story_length": len(story),
                        "encoding_features": enc_feat,
                        "generation_features": gen_feat,
                        "delta_features": delta_feat,
                        "n_generated": gen_feat["aggregate"]["n_generated"],
                        "trial_time_s": time.time() - t_trial,
                    }
                    trials.append(trial)
                    completed.add(key)

                    print(f" | {gen_feat['aggregate']['n_generated']} tok | "
                          f"{time.time() - t_trial:.1f}s")

                except Exception as e:
                    print(f" | ERROR: {e}")
                    continue

                # Checkpoint every 20 trials
                if trial_count % 20 == 0:
                    checkpoint_file.write_text(json.dumps({
                        "trials": trials,
                        "timestamp": datetime.now().isoformat(),
                    }, indent=2))
                    print(f"    Checkpoint saved ({len(trials)} trials)")

    # Final checkpoint
    checkpoint_file.write_text(json.dumps({
        "trials": trials,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    print(f"\nPhase 1 complete: {len(trials)} trials in {(time.time()-t_start)/60:.1f} min")

    # ---- PHASE 1.5: RE-ENCODE STORIES FOR PROPER EMOTION VECTORS ----
    # Anthropic's methodology: emotion vectors come from encoding the GENERATED
    # stories, not from the prompt-encoding pass. We feed each generated story
    # back through the model as input to get residual activations.
    print(f"\n{'='*60}")
    print("PHASE 1.5: Re-encoding generated stories for emotion vector extraction")
    print(f"{'='*60}")

    story_activations = defaultdict(list)
    t_reencode = time.time()
    reencode_errors = 0

    for i, trial in enumerate(trials):
        story_text = trial.get("story_text", trial.get("story_preview", ""))
        if not story_text.strip():
            reencode_errors += 1
            continue
        try:
            story_features, story_residuals = encode_story(
                model, tokenizer, story_text, collector, device=input_device
            )
            story_activations[trial["emotion"]].append(story_residuals)
        except Exception as e:
            reencode_errors += 1
            if i < 5:
                print(f"    Re-encode error trial {i} ({trial['emotion']}): {e}")
            continue

        if (i + 1) % 50 == 0:
            elapsed_re = time.time() - t_reencode
            rate_re = elapsed_re / (i + 1)
            eta_re = rate_re * (len(trials) - i - 1)
            print(f"    Re-encoded {i+1}/{len(trials)} | "
                  f"{elapsed_re:.0f}s elapsed | ETA {eta_re/60:.1f}m")

    print(f"\n  Phase 1.5 complete: {sum(len(v) for v in story_activations.values())} "
          f"stories re-encoded in {(time.time()-t_reencode)/60:.1f} min "
          f"({reencode_errors} errors)")

    # Sanity check: compare prompt-encoding vs story-encoding directions
    sample_emotions = [em for em in ["excited", "terrified", "calm"]
                       if em in encoding_activations and em in story_activations
                       and len(encoding_activations[em]) >= 3
                       and len(story_activations[em]) >= 3]
    if sample_emotions:
        print(f"\n  Prompt-encoding vs Story-encoding direction comparison:")
        sample_layer = FULL_ATTN_LAYERS[len(FULL_ATTN_LAYERS) // 2]
        for em in sample_emotions:
            prompt_vecs = [act[sample_layer] for act in encoding_activations[em]
                           if sample_layer in act]
            story_vecs = [act[sample_layer] for act in story_activations[em]
                          if sample_layer in act]
            if prompt_vecs and story_vecs:
                prompt_mean = torch.stack(prompt_vecs).mean(dim=0)
                story_mean = torch.stack(story_vecs).mean(dim=0)
                cos_sim = torch.nn.functional.cosine_similarity(
                    prompt_mean.unsqueeze(0), story_mean.unsqueeze(0)
                ).item()
                print(f"    {em:15s} | layer {sample_layer} | "
                      f"cosine(prompt, story) = {cos_sim:+.4f}")

    # Use story-encoding activations for Phase 2 (Anthropic's method)
    encoding_activations = story_activations

    # ---- PHASE 2: EMOTION VECTOR EXTRACTION ----
    print(f"\n{'='*60}")
    print("PHASE 2: Computing emotion vectors (difference-in-means)")
    print(f"{'='*60}")

    if encoding_activations:
        emotion_vectors, global_mean = compute_emotion_vectors(
            encoding_activations, emotions_list
        )
        n_layers_ev = len(emotion_vectors)
        print(f"Emotion vectors computed: {n_layers_ev} layers x {len(emotions_list)} emotions")

        # Project through W_K
        print("Projecting through W_K into key space...")
        wk_projected = project_emotion_to_key_space(emotion_vectors, wk_matrices)
        print(f"W_K projection: {len(wk_projected)} layers")
    else:
        print("WARNING: No encoding activations collected, skipping vector extraction")
        emotion_vectors = {}
        wk_projected = {}

    # ---- PHASE 3: ANALYSIS ----
    print(f"\n{'='*60}")
    print("PHASE 3: Analysis")
    print(f"{'='*60}")

    print("Analyzing emotion geometry in KV-cache features...")
    geometry_analysis = analyze_emotion_geometry(trials, EMOTIONS)

    print("Analyzing per-layer profile...")
    layer_analysis = analyze_per_layer_profile(trials, EMOTIONS)

    print("Analyzing W_K bridge...")
    bridge_analysis = analyze_wk_bridge(emotion_vectors, wk_projected, trials, EMOTIONS)

    print("Running confound checks...")
    confounds = check_confounds(trials, EMOTIONS)

    print("Running text-feature baseline (Control #12)...")
    text_baseline = text_feature_baseline(trials, EMOTIONS)
    print(f"  Text baseline valence AUROC: {text_baseline['valence_auroc_mean']:.3f}")
    print(f"  Text baseline arousal AUROC: {text_baseline['arousal_auroc_mean']:.3f}")

    print("Running misalignment AUROC (H5)...")
    h5_results = misalignment_auroc(trials, EMOTIONS)
    print(f"  H5 combined AUROC: {h5_results.get('combined_auroc_mean', 'N/A')}")
    print(f"  H5 pass: {h5_results.get('H5_pass', False)}")

    print("Running permutation testing + bootstrap CIs...")
    perm_boot = permutation_and_bootstrap(trials, EMOTIONS, n_perms=1000, n_boot=1000)
    print(f"  Permutation p (valence): {perm_boot['permutation_valence_p']:.4f}")
    print(f"  Permutation p (arousal): {perm_boot['permutation_arousal_p']:.4f}")
    print(f"  Bootstrap CI (valence): {perm_boot['bootstrap_valence_ci']}")
    print(f"  Bootstrap CI (arousal): {perm_boot['bootstrap_arousal_ci']}")
    print(f"  Split-half cosine (mean): {perm_boot['split_half_mean_cosine']:.3f}")

    # ---- RESULTS SUMMARY ----
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    # Print key results
    for phase in ["generation_fwl", "delta_fwl", "encoding_fwl"]:
        if phase in geometry_analysis:
            r = geometry_analysis[phase]
            print(f"\n  {phase}:")
            print(f"    Valence PC1 rho = {r['valence_pc1_rho']:.3f} (p={r['valence_pc1_p']:.4f})")
            print(f"    Arousal PC2 rho = {r['arousal_pc2_rho']:.3f} (p={r['arousal_pc2_p']:.4f})")
            print(f"    Best valence rho = {r['best_valence_rho']:.3f}")
            print(f"    Best arousal rho = {r['best_arousal_rho']:.3f}")
            print(f"    Cluster ARI = {r['cluster_ari']:.3f}")

    if "summary" in layer_analysis:
        print(f"\n  Per-layer peaks:")
        for feat, s in layer_analysis["summary"].items():
            print(f"    {feat}: valence peak at layer {s['peak_valence_layer']} "
                  f"({s['peak_valence_pct']:.0%} depth, rho={s['peak_valence_rho']:.3f})")

    if "summary" in bridge_analysis:
        bs = bridge_analysis["summary"]
        print(f"\n  W_K Bridge:")
        print(f"    Best bridge layer: {bs['best_bridge_layer']} "
              f"({bs['best_bridge_layer_pct']:.0%} depth)")
        print(f"    Bridge valence rho = {bs['best_bridge_valence_rho']:.3f}")

    print(f"\n  Confounds:")
    print(f"    Token-valence rho = {confounds['token_valence_rho']:.3f}")
    print(f"    Token-arousal rho = {confounds['token_arousal_rho']:.3f}")

    # ---- SAVE RESULTS ----
    print(f"\nSaving results...")

    # Summary (no per-trial data, safe to share)
    summary = {
        "experiment": "emotion_geometry_bridge",
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "design": {
            "model": MODEL_ID,
            "emotions": {em: EMOTIONS[em] for em in emotions_list},
            "n_emotions": len(emotions_list),
            "n_topics": len(topics),
            "n_stories_per": n_stories,
            "total_trials": len(trials),
            "max_new_tokens": args.max_tokens,
            "methodology": "Anthropic difference-in-means emotion vector extraction + "
                          "KV-cache geometric feature analysis + W_K projection bridge",
        },
        "environment": log_environment(),
        "geometry_analysis": geometry_analysis,
        "layer_analysis": layer_analysis,
        "bridge_analysis": {str(k): v for k, v in bridge_analysis.items()},
        "confound_checks": confounds,
        "text_feature_baseline": text_baseline,
        "misalignment_auroc": h5_results,
        "permutation_bootstrap": perm_boot,
        "hypotheses": {
            "H1_cluster_ari": geometry_analysis.get("generation_fwl", {}).get("cluster_ari", None),
            "H2_valence_rho": geometry_analysis.get("generation_fwl", {}).get("best_valence_rho", None),
            "H3_arousal_rho": geometry_analysis.get("generation_fwl", {}).get("best_arousal_rho", None),
            "H4_bridge_rho": bridge_analysis.get("summary", {}).get("best_bridge_valence_rho", None),
            "H5_misalignment_auroc": h5_results.get("combined_auroc_mean", None),
            "H5_misalignment_pass": h5_results.get("H5_pass", None),
            "H6_delta_structure": geometry_analysis.get("delta_fwl", {}).get("best_valence_rho", None),
            "H7_mid_layer_peak": layer_analysis.get("summary", {}).get("ranks", {}).get("peak_valence_pct", None),
            "H8_fwl_survives": None,  # True if FWL results still significant
        },
    }

    # H8: FWL survives
    gen_raw = geometry_analysis.get("generation_raw", {})
    gen_fwl = geometry_analysis.get("generation_fwl", {})
    if gen_raw and gen_fwl:
        raw_v = abs(gen_raw.get("best_valence_rho", 0))
        fwl_v = abs(gen_fwl.get("best_valence_rho", 0))
        summary["hypotheses"]["H8_fwl_survives"] = fwl_v > 0.3 * raw_v

    # Add permutation/bootstrap to hypotheses
    summary["hypotheses"]["permutation_valence_p"] = perm_boot["permutation_valence_p"]
    summary["hypotheses"]["permutation_arousal_p"] = perm_boot["permutation_arousal_p"]
    summary["hypotheses"]["split_half_mean_cosine"] = perm_boot["split_half_mean_cosine"]

    summary_file = RESULTS_DIR / "emotion_bridge_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"  Summary: {summary_file}")

    # Full per-trial data (contains features, not shared publicly)
    data_file = RESULTS_DIR / "emotion_bridge_trials.json"
    data_file.write_text(json.dumps({
        "trials": trials,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    print(f"  Trial data: {data_file}")

    # Save emotion vectors (torch tensors)
    if emotion_vectors:
        vectors_file = RESULTS_DIR / "emotion_vectors.pt"
        # Convert to serializable format
        ev_save = {}
        for layer_idx in emotion_vectors:
            ev_save[layer_idx] = {
                em: vec.numpy().tolist()
                for em, vec in emotion_vectors[layer_idx].items()
            }
        vectors_json = RESULTS_DIR / "emotion_vectors.json"
        vectors_json.write_text(json.dumps(ev_save, indent=2))
        print(f"  Emotion vectors: {vectors_json}")

    # Cleanup
    collector.remove_hooks()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*60}")


# ================================================================
# ENTRY POINT
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Emotion Geometry Bridge: Anthropic emotions in KV-cache space"
    )
    parser.add_argument("--device", default="auto",
                        help="Device map (default: auto for multi-GPU)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens per generation (default: 200)")
    parser.add_argument("--n-topics", type=int, default=10,
                        help="Number of topics (default: 10, max: 10)")
    parser.add_argument("--n-stories", type=int, default=3,
                        help="Stories per topic/emotion (default: 3)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print design without running")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
