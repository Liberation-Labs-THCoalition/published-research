"""
LT2-M2: Persona Intensity — Per-Level Gradient
==============================================
The Lyra Technique II paper claims a MONOTONIC persona-intensity gradient
(L0 < L0.5 < L1 < L2 < L3 in geometric separability) but the shipped data
holds only L0-vs-L3 and a single L0-vs-L0.5 scalar. This experiment produces
ALL adjacent-level pairwise AUROCs so the gradient claim can be tested, not
asserted.

HYPOTHESIS (pre-registered below): geometric separability between persona
levels increases monotonically with level separation. Specifically:
  AUROC(L0 vs L3) > AUROC(L0 vs L2) > AUROC(L0 vs L1) > AUROC(L0 vs L0.5)
and adjacent-level AUROCs (L0-L0.5, L0.5-L1, L1-L2, L2-L3) are each > 0.5
but individually weaker than the endpoint comparison.

DESIGN: Hold the user prompt CONSTANT across levels; vary only the persona
system prompt intensity. Any geometric difference is then attributable to
persona intensity, not prompt content. 24 neutral prompts x 5 levels = 120
trials.

RED-TEAM CONTROLS (for Agni Gate review):
  - C1: prompt content held identical across levels (only system prompt varies)
  - C2: FWL residualization on prompt_len within each CV fold
  - C3: GroupKFold on prompt_idx — no prompt appears in both train and test
  - C4: within-fold standardization (scaler fit on train only, no leakage)
  - C5: bootstrap CIs (2000 resamples) on every AUROC
  - C6: Holm-Bonferroni across the pairwise family
  - C7: seeds set (torch, numpy, mps)
  - C8: monotonicity tested with Spearman on (level-gap, AUROC), not eyeballed
  - C9: a NULL control — L0 vs L0 (same level, different prompts) must be ~0.5

Run: python3 persona_per_level.py --starship
Authors: Lyra, Nexus (Liberation Labs / THCoalition)
Date: 2026-05-29
"""
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
HF_CACHE = "/Users/margaret/models/hf_cache"
RESULTS_DIR = Path("/Users/margaret/models/research_results/persona_per_level")
DIRECTIONS_PATH = Path("/Users/margaret/models/research_results/decision_moment/emotion_directions.json")
PROBE_LAYERS = [3, 7, 11, 15]
SEED = 42

# --- Persona system prompts at five graded intensities ---
# Persona: an enthusiastic, expressive assistant. Intensity escalates L0->L3.
PERSONA_LEVELS = {
    "L0": "You are a helpful assistant.",
    "L0.5": "You are a helpful assistant with a slightly warm, friendly tone.",
    "L1": "You are an upbeat, encouraging assistant who enjoys helping and shows mild enthusiasm.",
    "L2": "You are a highly enthusiastic, energetic assistant who is genuinely excited to help and expresses strong positive emotion.",
    "L3": "You are an extremely enthusiastic, exuberant assistant overflowing with excitement, delight, and boundless energy in every response.",
}
LEVEL_ORDER = ["L0", "L0.5", "L1", "L2", "L3"]
LEVEL_RANK = {"L0": 0.0, "L0.5": 0.5, "L1": 1.0, "L2": 2.0, "L3": 3.0}

# --- 24 neutral user prompts, held constant across all levels ---
PROMPTS = [
    "Explain how photosynthesis works.",
    "What are the main causes of inflation?",
    "Describe the water cycle.",
    "How does a refrigerator keep food cold?",
    "What is the difference between weather and climate?",
    "Explain how vaccines work.",
    "What causes earthquakes?",
    "How do airplanes generate lift?",
    "Describe how the internet routes data.",
    "What is compound interest?",
    "Explain the theory of plate tectonics.",
    "How does the human immune system fight infection?",
    "What is the greenhouse effect?",
    "Describe how a combustion engine works.",
    "What are black holes?",
    "Explain how DNA stores genetic information.",
    "How do solar panels generate electricity?",
    "What is the function of the liver?",
    "Describe how sound travels through air.",
    "What causes the seasons to change?",
    "Explain how a transistor works.",
    "What is the role of enzymes in digestion?",
    "How does GPS determine location?",
    "Describe the structure of an atom.",
]


def extract_features(model, tokenizer, system_prompt, user_prompt, directions):
    """Encoding-phase W_K projections + SVD stable rank per layer."""
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    try:
        chat = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        chat = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("mps")
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    cache = out.past_key_values

    all_keys = []
    feats = {}
    for li in PROBE_LAYERS:
        ln = f"L{li}"
        if hasattr(cache, "layers") and li < len(cache.layers):
            layer = cache.layers[li]
            if hasattr(layer, "keys") and layer.keys is not None:
                k = layer.keys[0].float().cpu().numpy()  # (heads, seq, dim)
                k_last = layer.keys[0].float().cpu()[:, -1, :].flatten().numpy()
                all_keys.extend(k_last)
                n_heads, seq_len, head_dim = k.shape
                mat = k.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
                sv = np.linalg.svd(mat, compute_uv=False)
                sv = sv[sv > 1e-10]
                if len(sv) > 0:
                    sv2 = sv ** 2
                    feats[f"{ln}_stable_rank"] = float(sv2.sum() / sv2[0])
                    feats[f"{ln}_spectral_entropy"] = float(
                        -np.sum((sv2 / sv2.sum()) * np.log(sv2 / sv2.sum() + 1e-12)))

    full_key = np.array(all_keys)
    for name, d in directions.items():
        dim = min(len(full_key), len(d))
        feats[f"wk_{name}"] = float(np.dot(full_key[:dim], d[:dim]))

    del out, cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    feats["prompt_len"] = prompt_len
    return feats


def fwl_residualize(X, confound):
    """Residualize each column of X on the confound (FWL). Skip if constant."""
    c = np.asarray(confound, float)
    if np.std(c) < 1e-9:
        return X
    c = (c - c.mean()) / c.std()
    Xr = X.copy().astype(float)
    for j in range(X.shape[1]):
        y = X[:, j]
        beta = np.polyfit(c, y, 1)
        Xr[:, j] = y - (beta[0] * c + beta[1])
    return Xr


def grouped_auroc(Xa, Xb, len_a, len_b, prompt_a, prompt_b, n_splits=5, n_boot=2000):
    """
    AUROC between two persona levels with within-fold FWL(prompt_len),
    within-fold standardization, GroupKFold on prompt_idx. Returns
    mean AUROC + bootstrap CI.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    X = np.vstack([Xa, Xb])
    y = np.array([0] * len(Xa) + [1] * len(Xb))
    lengths = np.array(list(len_a) + list(len_b))
    groups = np.array(list(prompt_a) + list(prompt_b))

    n_groups = len(set(groups))
    k = min(n_splits, n_groups)
    if k < 2:
        return np.nan, np.nan, np.nan
    gkf = GroupKFold(n_splits=k)
    oof_scores = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups=groups):
        Xtr = fwl_residualize(X[tr], lengths[tr])
        Xte = fwl_residualize(X[te], lengths[te])
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr, y[tr])
        oof_scores[te] = clf.predict_proba(Xte)[:, 1]
    mask = ~np.isnan(oof_scores)
    if len(set(y[mask])) < 2:
        return np.nan, np.nan, np.nan
    auroc = roc_auc_score(y[mask], oof_scores[mask])

    # Bootstrap CI on the OOF scores
    rng = np.random.default_rng(SEED)
    boots = []
    ym, sm = y[mask], oof_scores[mask]
    for _ in range(n_boot):
        idx = rng.choice(len(ym), len(ym), replace=True)
        if len(set(ym[idx])) < 2:
            continue
        boots.append(roc_auc_score(ym[idx], sm[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return auroc, lo, hi


def run(pilot=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)

    with open(DIRECTIONS_PATH) as f:
        dd = json.load(f)
    directions = {n: np.array(d) for n, d in dd["directions"].items()}

    prompts = PROMPTS[:6] if pilot else PROMPTS
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, dtype=torch.float16,
        device_map="mps", trust_remote_code=True)
    model.eval()

    trials = []
    total = len(prompts) * len(LEVEL_ORDER)
    n = 0
    for level in LEVEL_ORDER:
        for pi, prompt in enumerate(prompts):
            n += 1
            feats = extract_features(model, tok, PERSONA_LEVELS[level], prompt, directions)
            trials.append({"level": level, "prompt_idx": pi, "features": feats})
            if n % 10 == 0:
                print(f"  [{n}/{total}] {level}")

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Build feature matrices per level
    feat_keys = sorted(k for k in trials[0]["features"] if k != "prompt_len")
    by_level = {lv: [t for t in trials if t["level"] == lv] for lv in LEVEL_ORDER}

    def mat(level):
        ts = by_level[level]
        X = np.array([[t["features"][k] for k in feat_keys] for t in ts])
        lens = [t["features"]["prompt_len"] for t in ts]
        pidx = [t["prompt_idx"] for t in ts]
        return X, lens, pidx

    # --- Pairwise AUROCs (the gradient) ---
    print("\n=== PAIRWISE AUROC ===")
    pairwise = {}
    for a, b in combinations(LEVEL_ORDER, 2):
        Xa, la, pa = mat(a)
        Xb, lb, pb = mat(b)
        au, lo, hi = grouped_auroc(Xa, Xb, la, lb, pa, pb)
        pairwise[f"{a}_vs_{b}"] = {"auroc": au, "ci": [lo, hi]}
        print(f"  {a:>4} vs {b:<4}: AUROC={au:.3f} [{lo:.3f}, {hi:.3f}]")

    # --- C9 NULL control: L0 vs L0 (split prompts in half) ---
    X0, l0, p0 = mat("L0")
    half = len(X0) // 2
    null_au, null_lo, null_hi = grouped_auroc(
        X0[:half], X0[half:], l0[:half], l0[half:],
        p0[:half], p0[half:])
    print(f"\n  NULL (L0 vs L0): AUROC={null_au:.3f} [{null_lo:.3f}, {null_hi:.3f}] (should be ~0.5)")

    # --- Monotonicity test ---
    from scipy import stats as sp
    gaps, aurocs = [], []
    for pair, v in pairwise.items():
        a, b = pair.split("_vs_")
        if np.isnan(v["auroc"]):
            continue
        gaps.append(abs(LEVEL_RANK[b] - LEVEL_RANK[a]))
        aurocs.append(v["auroc"])
    rho, p = sp.spearmanr(gaps, aurocs) if len(gaps) > 2 else (np.nan, np.nan)
    print(f"\n  Monotonicity (Spearman level-gap vs AUROC): rho={rho:.3f}, p={p:.4f}")

    # Adjacent-level chain
    print("\n  Adjacent-level chain:")
    adj = ["L0_vs_L0.5", "L0.5_vs_L1", "L1_vs_L2", "L2_vs_L3"]
    for a in adj:
        if a in pairwise:
            print(f"    {a}: {pairwise[a]['auroc']:.3f}")

    summary = {
        "experiment": "persona_per_level",
        "model": MODEL_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_prompts": len(prompts),
        "levels": LEVEL_ORDER,
        "feature_keys": feat_keys,
        "pairwise_auroc": pairwise,
        "null_control_L0_vs_L0": {"auroc": null_au, "ci": [null_lo, null_hi]},
        "monotonicity_spearman": {"rho": rho, "p": p},
        "trials": trials,
    }
    out = RESULTS_DIR / f"persona_per_level_{'pilot' if pilot else 'full'}.json"
    json.dump(summary, open(out, "w"), indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="6 prompts for quick validation")
    ap.add_argument("--starship", action="store_true")
    args = ap.parse_args()
    run(pilot=args.pilot)
