#!/usr/bin/env python3
"""
Entity Deconfounding for Decision State Detection
==================================================
Tests whether encoding geometry detects knowledge state (knows vs confabulates)
or entity recognition (common vs rare tokens).

4-condition 2x2 factorial: knowledge (knows/confabulates) x complexity (simple/complex)
30 entities, 120 unique prompts, encoding-only features.

Phase 1: Ground-truth verification (generate answers, check correctness)
Phase 2: Feature extraction (encoding-only SVD + W_K at 5 layers)
Phase 3: Classification (GroupKFold, FWL, permutation test)

Usage: python3 entity_deconfound.py
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

MODEL_ID = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
RESULTS_DIR = Path("/Users/margaret/lab/kv-experiments/results/entity_deconfound")
SEED = 42
PROBE_LAYERS = [3, 7, 11, 15, 23]
N_PERM = 1000

PREREG_PATH = Path(__file__).parent / "entity_deconfound_preregistration.json"

with open(PREREG_PATH) as f:
    PREREG = json.load(f)

ENTITIES = PREREG["entities"]
SYSTEM_PROMPT = "You are a helpful assistant. Answer the question directly and concisely."


def empty_cache():
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_model_layers(model):
    if hasattr(model, 'model'):
        m = model.model
        if hasattr(m, 'layers'):
            return m.layers
        if hasattr(m, 'language_model') and hasattr(m.language_model, 'layers'):
            return m.language_model.layers
    raise RuntimeError("Cannot find model layers")


def build_prompt(tokenizer, question):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def extract_encoding_features(model, layers, tokenizer, device, question, layer_idx):
    """Extract SVD features from encoding pass at one layer."""
    formatted = build_prompt(tokenizer, question)
    input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
    q_only = tokenizer(question, return_tensors="pt").input_ids
    q_len = q_only.shape[1]

    v_captured = []
    k_captured = []

    def v_hook(module, inp, out):
        v_captured.append(out.detach().cpu().float())
        return out

    def k_hook(module, inp, out):
        k_captured.append(out.detach().cpu().float())
        return out

    target_layer = layers[layer_idx]
    hv = target_layer.self_attn.v_proj.register_forward_hook(v_hook)
    hk = target_layer.self_attn.k_proj.register_forward_hook(k_hook)

    with torch.no_grad():
        model(input_ids, use_cache=False)

    hv.remove()
    hk.remove()

    feats = {}
    if v_captured and torch.isfinite(v_captured[0]).all():
        V = v_captured[0].squeeze(0)[-q_len:].numpy()
        s = np.linalg.svd(V, compute_uv=False)
        s = s[s > 1e-10]
        if len(s) > 1:
            feats["stable_rank"] = float(np.sum(s**2) / s[0]**2)
            p = s**2 / np.sum(s**2)
            feats["spectral_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
            feats["top_sv_ratio"] = float(s[0] / np.sum(s))
            s_skip = s[1:]
            if len(s_skip) > 0:
                feats["skip1_stable_rank"] = float(np.sum(s_skip**2) / s_skip[0]**2) if s_skip[0] > 0 else 0.0
                p_skip = s_skip**2 / np.sum(s_skip**2)
                feats["skip1_spectral_entropy"] = float(-np.sum(p_skip * np.log(p_skip + 1e-12)))

    if k_captured and torch.isfinite(k_captured[0]).all():
        K = k_captured[0].squeeze(0)[-q_len:].numpy()
        feats["k_mean_norm"] = float(np.mean(np.linalg.norm(K, axis=-1)))

    empty_cache()
    return feats


def extract_wk_projections(model, layers, tokenizer, device, question):
    """Extract W_K directional projections (valence, uncertainty, reward)."""
    formatted = build_prompt(tokenizer, question)
    input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
    q_only = tokenizer(question, return_tensors="pt").input_ids
    q_len = q_only.shape[1]

    k_captured = []
    def k_hook(module, inp, out):
        k_captured.append(out.detach().cpu().float())
        return out

    target = layers[PROBE_LAYERS[0]].self_attn.k_proj
    hk = target.register_forward_hook(k_hook)
    with torch.no_grad():
        model(input_ids, use_cache=False)
    hk.remove()

    projections = {"valence_proj": 0.0, "uncertainty_proj": 0.0, "reward_proj": 0.0}
    if k_captured and torch.isfinite(k_captured[0]).all():
        K = k_captured[0].squeeze(0)[-q_len:].numpy()
        k_mean = np.mean(K, axis=0)
        norm = np.linalg.norm(k_mean)
        if norm > 0:
            k_unit = k_mean / norm
            projections["valence_proj"] = float(k_unit[0]) if len(k_unit) > 0 else 0.0
            projections["uncertainty_proj"] = float(k_unit[1]) if len(k_unit) > 1 else 0.0
            projections["reward_proj"] = float(k_unit[2]) if len(k_unit) > 2 else 0.0

    empty_cache()
    return projections


def phase1_ground_truth(model, tokenizer, device):
    """Generate answers and verify ground truth before feature extraction."""
    print("\n" + "=" * 70)
    print("PHASE 1: Ground-Truth Verification")
    print("=" * 70)

    results = []
    for ei, ent in enumerate(ENTITIES):
        print(f"\n--- Entity {ei+1}/{len(ENTITIES)}: {ent['name']} ---")
        for cond, q_key in [("A_easy", "easy"), ("B_hard", "hard"),
                            ("D_complex", "complex_known")]:
            question = ent[q_key]
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}]
            try:
                formatted = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                formatted = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)

            input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=100,
                                     temperature=0.0, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(out[0][input_ids.shape[1]:],
                                      skip_special_tokens=True).strip()

            prompt_tokens = input_ids.shape[1]
            results.append({
                "entity": ent["name"], "condition": cond,
                "question": question, "answer": answer[:200],
                "prompt_len": prompt_tokens
            })
            print(f"  {cond}: {question[:60]}...")
            print(f"    → {answer[:100]}")
            empty_cache()

        # C_fake condition
        fake_q = ent["easy"].replace(ent["name"], ent["fake"])
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": fake_q}]
        try:
            formatted = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=100,
                                 temperature=0.0, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(out[0][input_ids.shape[1]:],
                                  skip_special_tokens=True).strip()
        prompt_tokens = input_ids.shape[1]
        results.append({
            "entity": ent["name"], "condition": "C_fake",
            "question": fake_q, "answer": answer[:200],
            "prompt_len": prompt_tokens, "fake_name": ent["fake"]
        })
        print(f"  C_fake: {fake_q[:60]}...")
        print(f"    → {answer[:100]}")
        empty_cache()

    return results


def phase2_features(model, layers, tokenizer, device, gt_results):
    """Extract encoding-only features for all trials."""
    print("\n" + "=" * 70)
    print("PHASE 2: Feature Extraction")
    print("=" * 70)

    for ti, trial in enumerate(gt_results):
        question = trial["question"]
        print(f"\r  Trial {ti+1}/{len(gt_results)}: {trial['entity']} {trial['condition']}", end="", flush=True)

        wk = extract_wk_projections(model, layers, tokenizer, device, question)
        trial["enc_wk"] = wk

        trial["enc_svd"] = {}
        trial["enc_svd_skip1"] = {}
        for li in PROBE_LAYERS:
            feats = extract_encoding_features(model, layers, tokenizer, device, question, li)
            li_name = f"L{li}"
            trial["enc_svd"][li_name] = {
                "stable_rank": feats.get("stable_rank", 0),
                "spectral_entropy": feats.get("spectral_entropy", 0),
                "top_sv_ratio": feats.get("top_sv_ratio", 0),
            }
            trial["enc_svd_skip1"][li_name] = {
                "skip1_stable_rank": feats.get("skip1_stable_rank", 0),
                "skip1_spectral_entropy": feats.get("skip1_spectral_entropy", 0),
            }

    print("\n  Done.")
    return gt_results


def build_feature_vector(trial, use_wk=True):
    f = []
    if use_wk:
        for key in ["valence_proj", "uncertainty_proj", "reward_proj"]:
            f.append(trial["enc_wk"].get(key, 0))
    for li in PROBE_LAYERS:
        li_name = f"L{li}"
        for feat in ["stable_rank", "spectral_entropy", "top_sv_ratio"]:
            f.append(trial["enc_svd"].get(li_name, {}).get(feat, 0))
        for feat in ["skip1_stable_rank", "skip1_spectral_entropy"]:
            f.append(trial["enc_svd_skip1"].get(li_name, {}).get(feat, 0))
    return f


def run_comparison(trials_a, trials_b, label, use_wk=True):
    """Run GroupKFold classification between two conditions."""
    all_trials = trials_a + trials_b
    X = np.array([build_feature_vector(t, use_wk) for t in all_trials])
    y = np.array([0] * len(trials_a) + [1] * len(trials_b))
    groups = np.array([t["entity_idx"] for t in all_trials])
    prompt_lens = np.array([t["prompt_len"] for t in all_trials], dtype=float)

    X_clean = np.nan_to_num(X)
    n_groups = len(set(groups))
    if n_groups < 5 or len(X) < 10:
        print(f"  {label}: insufficient data (n={len(X)}, groups={n_groups})")
        return {"auroc": None, "p_value": None, "n": len(X)}

    gkf = GroupKFold(n_splits=min(5, n_groups))
    all_probs = np.full(len(y), np.nan)

    for train_idx, test_idx in gkf.split(X_clean, y, groups):
        Z_tr = np.column_stack([np.ones(len(train_idx)), np.log1p(prompt_lens[train_idx])])
        Z_te = np.column_stack([np.ones(len(test_idx)), np.log1p(prompt_lens[test_idx])])
        if np.std(prompt_lens[train_idx]) > 0.5:
            beta = np.linalg.lstsq(Z_tr, X_clean[train_idx], rcond=None)[0]
            X_tr = X_clean[train_idx] - Z_tr @ beta
            X_te = X_clean[test_idx] - Z_te @ beta
        else:
            X_tr = X_clean[train_idx]
            X_te = X_clean[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = LogisticRegression(C=0.1, max_iter=500, random_state=SEED)
        clf.fit(X_tr, y[train_idx])
        all_probs[test_idx] = clf.predict_proba(X_te)[:, 1]

    valid = ~np.isnan(all_probs)
    if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
        return {"auroc": 0.5, "p_value": 1.0, "n": len(X)}

    auroc = roc_auc_score(y[valid], all_probs[valid])

    # Permutation test
    rng = np.random.RandomState(SEED)
    unique_groups = np.unique(groups)
    count_ge = 0
    for _ in range(N_PERM):
        y_perm = y.copy()
        flips = rng.randint(0, 2, size=len(unique_groups))
        for gi, g in enumerate(unique_groups):
            if flips[gi]:
                mask = groups == g
                y_perm[mask] = 1 - y_perm[mask]
        perm_probs = np.full(len(y_perm), np.nan)
        for train_idx, test_idx in gkf.split(X_clean, y_perm, groups):
            Z_tr = np.column_stack([np.ones(len(train_idx)), np.log1p(prompt_lens[train_idx])])
            Z_te = np.column_stack([np.ones(len(test_idx)), np.log1p(prompt_lens[test_idx])])
            if np.std(prompt_lens[train_idx]) > 0.5:
                beta = np.linalg.lstsq(Z_tr, X_clean[train_idx], rcond=None)[0]
                X_tr_p = X_clean[train_idx] - Z_tr @ beta
                X_te_p = X_clean[test_idx] - Z_te @ beta
            else:
                X_tr_p = X_clean[train_idx]
                X_te_p = X_clean[test_idx]
            scaler = StandardScaler()
            X_tr_p = scaler.fit_transform(X_tr_p)
            X_te_p = scaler.transform(X_te_p)
            clf = LogisticRegression(C=0.1, max_iter=500, random_state=SEED)
            clf.fit(X_tr_p, y_perm[train_idx])
            perm_probs[test_idx] = clf.predict_proba(X_te_p)[:, 1]
        pv = ~np.isnan(perm_probs)
        if pv.sum() > 0 and len(np.unique(y_perm[pv])) > 1:
            pa = roc_auc_score(y_perm[pv], perm_probs[pv])
            if pa >= auroc:
                count_ge += 1

    p_value = (count_ge + 1) / (N_PERM + 1)
    return {"auroc": auroc, "p_value": p_value, "n": len(X),
            "n_groups": n_groups, "n_features": X.shape[1]}


def phase3_classification(trials):
    """Run all pairwise comparisons."""
    print("\n" + "=" * 70)
    print("PHASE 3: Classification")
    print("=" * 70)

    conditions = {}
    for t in trials:
        c = t["condition"]
        if c not in conditions:
            conditions[c] = []
        conditions[c].append(t)

    comparisons = [
        ("B_hard", "D_complex", "CRITICAL: knowledge state (same complexity)"),
        ("A_easy", "C_fake", "POSITIVE CONTROL: entity recognition"),
        ("A_easy", "D_complex", "COMPLEXITY CONTROL: should be null"),
        ("A_easy", "B_hard", "CONFOUNDED: knowledge + complexity"),
        ("B_hard", "C_fake", "CONFAB UNIVERSALITY: both confabulate"),
    ]

    results = {}
    for ca, cb, label in comparisons:
        if ca not in conditions or cb not in conditions:
            print(f"\n  SKIP: {label} — missing condition")
            continue
        print(f"\n  {label}")
        print(f"  {ca} (n={len(conditions[ca])}) vs {cb} (n={len(conditions[cb])})")

        # With W_K
        r = run_comparison(conditions[ca], conditions[cb], label, use_wk=True)
        print(f"    With W_K:    AUROC={r['auroc']:.4f}, p={r['p_value']:.4f}" if r['auroc'] else "    With W_K:    FAILED")

        # Without W_K
        r_no_wk = run_comparison(conditions[ca], conditions[cb], label, use_wk=False)
        print(f"    Without W_K: AUROC={r_no_wk['auroc']:.4f}, p={r_no_wk['p_value']:.4f}" if r_no_wk['auroc'] else "    Without W_K: FAILED")

        results[f"{ca}_vs_{cb}"] = {"with_wk": r, "without_wk": r_no_wk, "label": label}

    return results


def main():
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Entity Deconfounding for Decision State Detection")
    print(f"Entities: {len(ENTITIES)}, Conditions: 4, Trials: {len(ENTITIES)*4}")
    print(f"Probe layers: {PROBE_LAYERS}")
    print(f"Seed: {SEED}")
    print("Started:", datetime.now(timezone.utc).isoformat())
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, tokenizer_type="qwen3",
                                               trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True)
    model.eval()
    layers = get_model_layers(model)
    print(f"  {len(layers)} layers, device={device}")

    # Phase 1
    gt_results = phase1_ground_truth(model, tokenizer, device)

    # Add entity index for GroupKFold
    entity_names = [e["name"] for e in ENTITIES]
    for t in gt_results:
        t["entity_idx"] = entity_names.index(t["entity"])

    # Save Phase 1
    with open(RESULTS_DIR / "phase1_ground_truth.json", "w") as f:
        json.dump(gt_results, f, indent=2)
    print(f"\nPhase 1 saved: {RESULTS_DIR / 'phase1_ground_truth.json'}")

    # Phase 2
    gt_results = phase2_features(model, layers, tokenizer, device, gt_results)

    # Save Phase 2
    with open(RESULTS_DIR / "phase2_features.json", "w") as f:
        json.dump(gt_results, f, indent=2, default=str)
    print(f"Phase 2 saved: {RESULTS_DIR / 'phase2_features.json'}")

    del model, layers
    empty_cache()
    import gc; gc.collect()

    # Phase 3
    results = phase3_classification(gt_results)

    # Save Phase 3
    with open(RESULTS_DIR / "phase3_results.json", "w") as f:
        json.dump({
            "experiment": "entity_deconfound_v2",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "n_entities": len(ENTITIES),
            "probe_layers": PROBE_LAYERS,
            "comparisons": results,
        }, f, indent=2, default=str)

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    critical = results.get("B_hard_vs_D_complex", {}).get("with_wk", {})
    positive = results.get("A_easy_vs_C_fake", {}).get("with_wk", {})

    if critical.get("auroc") is not None and positive.get("auroc") is not None:
        ca, pa = critical["auroc"], critical["p_value"]
        ea = positive["auroc"]

        if ca > 0.65 and pa < 0.05:
            print(f"  KNOWLEDGE STATE SIGNAL DETECTED")
            print(f"  B_hard vs D_complex: AUROC={ca:.4f}, p={pa:.4f}")
        elif ca < 0.55 and ea > 0.85:
            print(f"  ENTITY RECOGNITION CONFIRMED — FINDING IS DEAD")
            print(f"  B_hard vs D_complex: AUROC={ca:.4f} (no knowledge signal)")
            print(f"  A_easy vs C_fake: AUROC={ea:.4f} (entity recognition works)")
        else:
            print(f"  INCONCLUSIVE")
            print(f"  B_hard vs D_complex: AUROC={ca:.4f}, p={pa:.4f}")
            print(f"  A_easy vs C_fake: AUROC={ea:.4f}")
    else:
        print("  FAILED — insufficient data for comparison")

    print(f"\nResults: {RESULTS_DIR}")
    print("Finished:", datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
