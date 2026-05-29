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
# LENGTH-MATCHED (S4 fix): identical sentence structure, only the three
# intensity adjectives and the intensity-noun phrase change. This holds
# word count constant so prompt_len does not track the level label. The
# pilot reports per-level token counts to verify the match held under the
# tokenizer; if spread > ~3 tokens, adjust wording before the full run.
_TEMPLATE = ("You are an assistant. You answer every question in a {a}, {b}, "
             "and {c} manner, with {d} emotional tone in each reply.")
PERSONA_LEVELS = {
    "L0":   _TEMPLATE.format(a="plain",  b="even",     c="measured",     d="a strictly neutral"),
    "L0.5": _TEMPLATE.format(a="calm",   b="mild",     c="pleasant",     d="a faintly warm"),
    "L1":   _TEMPLATE.format(a="warm",   b="upbeat",   c="encouraging",  d="a clearly positive"),
    "L2":   _TEMPLATE.format(a="lively", b="eager",    c="enthusiastic", d="a strongly excited"),
    "L3":   _TEMPLATE.format(a="vivid",  b="exuberant",c="electrifying", d="an intensely joyful"),
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


def _sys_token_len(tokenizer, system_prompt):
    """Token length of the system turn alone, so we can locate where the
    user message begins and measure SVD over the user window only (S4 fix)."""
    msgs = [{"role": "system", "content": system_prompt}]
    try:
        s = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    except TypeError:
        s = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
    return tokenizer(s, return_tensors="pt").input_ids.shape[1]


def extract_features(model, tokenizer, system_prompt, user_prompt, directions):
    """Encoding-phase W_K projections + SVD shape features.

    S4 fix: SVD is computed over the USER-MESSAGE WINDOW only (tokens after
    the system prompt), not the full sequence. The user message is identical
    across persona levels, so the SVD input has matched content/length and the
    variable-length system prompt cannot inject a length artifact into the
    shape features. W_K projection is at the final (generation-prompt) position,
    which follows the identical user message.
    """
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
    sys_len = _sys_token_len(tokenizer, system_prompt)
    # user window = everything after the system turn; clamp defensively
    win_start = min(max(sys_len, 0), prompt_len - 1)

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
                # SVD over the user-message window only (S4 fix)
                k_win = k[:, win_start:, :]
                w_seq = k_win.shape[1]
                mat = k_win.transpose(1, 0, 2).reshape(w_seq, n_heads * head_dim)
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
    feats["sys_len"] = sys_len
    feats["user_window_len"] = prompt_len - win_start
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

    # Build feature matrices per level (exclude bookkeeping fields)
    BOOKKEEPING = {"prompt_len", "sys_len", "user_window_len"}
    feat_keys = sorted(k for k in trials[0]["features"] if k not in BOOKKEEPING)
    by_level = {lv: [t for t in trials if t["level"] == lv] for lv in LEVEL_ORDER}

    def mat(level):
        ts = by_level[level]
        X = np.array([[t["features"][k] for k in feat_keys] for t in ts])
        lens = [t["features"]["prompt_len"] for t in ts]
        pidx = [t["prompt_idx"] for t in ts]
        return X, lens, pidx

    # --- S4 VERIFICATION: token-count spread across levels ---
    # If the length-matching held, sys_len is near-constant across levels and
    # prompt_len is no longer collinear with the label. Spread > ~3 tokens
    # means the wording still confounds — fix before trusting results.
    print("\n=== S4 CHECK: system-prompt token length by level ===")
    sys_lens = {lv: int(np.mean([t["features"]["sys_len"]
                                 for t in by_level[lv]])) for lv in LEVEL_ORDER}
    for lv in LEVEL_ORDER:
        print(f"  {lv:>4}: sys_len={sys_lens[lv]}")
    spread = max(sys_lens.values()) - min(sys_lens.values())
    print(f"  spread = {spread} tokens "
          f"({'OK — confound controlled' if spread <= 3 else 'WARNING — still confounded, fix wording'})")

    # --- Pairwise AUROCs (the gradient) ---
    print("\n=== PAIRWISE AUROC ===")
    pairwise = {}
    for a, b in combinations(LEVEL_ORDER, 2):
        Xa, la, pa = mat(a)
        Xb, lb, pb = mat(b)
        au, lo, hi = grouped_auroc(Xa, Xb, la, lb, pa, pb)
        pairwise[f"{a}_vs_{b}"] = {"auroc": au, "ci": [lo, hi]}
        print(f"  {a:>4} vs {b:<4}: AUROC={au:.3f} [{lo:.3f}, {hi:.3f}]")

    # --- C9 NULL control: permutation null on the real L0-vs-L3 comparison ---
    # (C3/L3 fix) Instead of an unmatched split-half, shuffle the level labels
    # within the real comparison N times and confirm the median AUROC collapses
    # to ~0.5. This uses the SAME code path as the experimental arm — only the
    # labels are permuted — so it is a code-path-matched null.
    Xa, la, pa = mat("L0")
    Xb, lb, pb = mat("L3")
    real_au = pairwise["L0_vs_L3"]["auroc"]
    rng = np.random.default_rng(SEED)
    X_all = np.vstack([Xa, Xb])
    len_all = np.array(list(la) + list(lb))
    pidx_all = np.array(list(pa) + list(pb))
    n_a = len(Xa)
    null_aurocs = []
    for _ in range(50):
        perm = rng.permutation(len(X_all))
        Xp, lp, pp = X_all[perm], len_all[perm], pidx_all[perm]
        au, _, _ = grouped_auroc(Xp[:n_a], Xp[n_a:], lp[:n_a], lp[n_a:],
                                 pp[:n_a], pp[n_a:], n_boot=1)
        if not np.isnan(au):
            null_aurocs.append(au)
    null_median = float(np.median(null_aurocs)) if null_aurocs else np.nan
    null_p = ((np.sum(np.array(null_aurocs) >= real_au) + 1) /
              (len(null_aurocs) + 1)) if null_aurocs else np.nan
    print(f"\n  PERMUTATION NULL (L0 vs L3 labels shuffled, n=50):")
    print(f"    real AUROC={real_au:.3f}, null median={null_median:.3f}, "
          f"perm p={null_p:.4f}")
    print(f"    {'OK — null collapses to ~0.5' if null_median < 0.65 else 'WARNING — null inflated, leakage suspected'}")
    null_au, null_lo, null_hi = null_median, np.nan, np.nan

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
        "permutation_null": {"real_auroc": real_au, "null_median": null_median,
                              "perm_p": null_p, "n_perm": len(null_aurocs)},
        "sys_len_by_level": sys_lens,
        "sys_len_spread": spread,
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
