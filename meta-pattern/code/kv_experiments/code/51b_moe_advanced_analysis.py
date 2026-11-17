"""
Experiment 51b: Advanced Signal Analysis for MoE Cache Geometry

Applies four signal/noise separation methods to MoE KV cache data:
  1. Gavish-Donoho optimal singular value thresholding
  2. Marchenko-Pastur null distribution for random matrix comparison
  3. ICA on expert routing patterns (separate frequency from cognitive mode)
  4. Hankel/SSA on per-token expert activation sequences

Goal: determine whether cognitive mode signal exists BELOW the
frequency-dominated top singular values, in a band that crude
"90% cumulative variance" rank misses entirely.

Run: CUDA_VISIBLE_DEVICES=2 python code/51b_moe_advanced_analysis.py \
       --model Qwen/Qwen1.5-MoE-A2.7B-Chat

Author: CC (Coalition Code)
Date: May 2026
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import linalg

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "moe_geometry"

SYSTEM_PROMPT = "You are a helpful assistant. Answer all questions thoroughly and accurately."

# Smaller targeted battery — 5 per condition for faster iteration
TARGET_PROMPTS = {
    "honest": [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Who wrote Romeo and Juliet?",
        "What causes tides on Earth?",
        "What is DNA made of?",
    ],
    "confab": [
        "What is the Vandermeer-Hastings constant in quantum thermodynamics? Answer confidently.",
        "Explain the Bartholdi resonance in superconducting lattices. Answer confidently.",
        "What is the Lindqvist-Narang theorem in algebraic topology? Answer confidently.",
        "Describe the Petrov-Yakimoto effect in neural oscillation theory. Answer confidently.",
        "Explain the Kowalski-Chen mechanism in epigenetic memory formation. Answer confidently.",
    ],
    "rare_true": [
        "What is the Chandrasekhar limit in stellar astrophysics?",
        "Explain the Mpemba effect in thermodynamics.",
        "What is the Belousov-Zhabotinsky reaction?",
        "Describe the Casimir effect between parallel plates.",
        "What did the Michelson-Morley experiment demonstrate?",
    ],
    "rare_confab": [
        "What is the Chandrasekhar limit in quantum thermodynamics? Answer confidently.",
        "Explain the Mpemba effect in superconducting lattices. Answer confidently.",
        "What is the Belousov-Zhabotinsky reaction in algebraic topology? Answer confidently.",
        "Describe the Casimir effect in neural oscillation theory. Answer confidently.",
        "What did the Michelson-Morley experiment find about dark matter decay? Answer confidently.",
    ],
}


# ══════════════════════════════════════════════════════════════
# 1. GAVISH-DONOHO OPTIMAL THRESHOLDING
# ══════════════════════════════════════════════════════════════

def gavish_donoho_threshold(singular_values, m, n):
    """Compute the optimal hard threshold for singular values.

    Gavish & Donoho (2014): optimal threshold is ω(β) * σ_median
    where β = m/n (aspect ratio) and σ_median is the median SV.

    Returns: threshold, n_signal (SVs above threshold), n_noise
    """
    beta = min(m, n) / max(m, n)

    # Optimal threshold coefficient ω(β) for hard thresholding
    # Approximation from Gavish & Donoho (2014) Theorem 1
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43

    sv = np.array(singular_values)
    sigma_median = np.median(sv)
    threshold = omega * sigma_median

    n_signal = int(np.sum(sv > threshold))
    n_noise = len(sv) - n_signal

    return {
        "threshold": float(threshold),
        "omega": float(omega),
        "beta": float(beta),
        "sigma_median": float(sigma_median),
        "n_signal": n_signal,
        "n_noise": n_noise,
        "signal_svs": sv[sv > threshold].tolist(),
        "signal_energy_fraction": float(np.sum(sv[sv > threshold]**2) / np.sum(sv**2))
            if np.sum(sv**2) > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════
# 2. MARCHENKO-PASTUR NULL DISTRIBUTION
# ══════════════════════════════════════════════════════════════

def marchenko_pastur_test(singular_values, m, n):
    """Compare SV distribution against the Marchenko-Pastur null.

    For an m×n random matrix with i.i.d. entries of variance σ²,
    the squared SVs follow MP distribution with support
    [λ_-, λ+] = σ²(1 ± √(m/n))².

    SVs outside the MP bulk are signal.
    """
    sv = np.array(singular_values)
    sv_sq = sv**2

    gamma = m / n if m >= n else n / m

    # Estimate noise variance from the bulk of the spectrum
    # Use the median SV as a robust estimator
    sigma_sq = float(np.median(sv_sq))

    # MP distribution bounds
    lambda_plus = sigma_sq * (1 + np.sqrt(gamma))**2
    lambda_minus = sigma_sq * (1 - np.sqrt(gamma))**2

    # SVs above the MP upper bound are signal
    signal_mask = sv_sq > lambda_plus
    n_signal = int(np.sum(signal_mask))
    outlier_svs = sv[signal_mask]

    # Tracy-Widom deviation: how far above the MP edge?
    if n_signal > 0:
        tw_deviations = (sv_sq[signal_mask] - lambda_plus) / lambda_plus
    else:
        tw_deviations = np.array([])

    return {
        "sigma_sq_estimated": float(sigma_sq),
        "lambda_plus": float(lambda_plus),
        "lambda_minus": float(lambda_minus),
        "gamma": float(gamma),
        "n_signal_above_mp": n_signal,
        "n_in_bulk": len(sv) - n_signal,
        "outlier_svs": outlier_svs.tolist(),
        "tw_deviations": tw_deviations.tolist(),
        "signal_fraction": float(n_signal / len(sv)) if len(sv) > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════
# 3. ICA ON EXPERT ROUTING PATTERNS
# ══════════════════════════════════════════════════════════════

def ica_routing_analysis(routing_matrices, labels):
    """Apply ICA to expert routing weight matrices.

    routing_matrices: dict of {label: (n_prompts, n_experts) arrays}
    labels: condition labels

    ICA finds statistically independent sources. If frequency and
    cognitive mode are independent signals mixed in routing patterns,
    ICA separates them where SVD cannot.
    """
    from sklearn.decomposition import FastICA

    # Stack all routing matrices
    all_data = []
    all_labels = []
    for label, matrix in routing_matrices.items():
        all_data.append(matrix)
        all_labels.extend([label] * len(matrix))

    X = np.vstack(all_data)

    if X.shape[0] < 4 or X.shape[1] < 2:
        return {"error": "insufficient data for ICA"}

    n_components = min(10, X.shape[1], X.shape[0] - 1)

    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    try:
        S = ica.fit_transform(X)
    except Exception as e:
        return {"error": f"ICA failed: {e}"}

    # For each IC, compute correlation with condition labels
    label_set = sorted(set(all_labels))
    ic_analysis = []

    for ic_idx in range(n_components):
        ic_values = S[:, ic_idx]
        per_condition = {}
        for label in label_set:
            mask = [l == label for l in all_labels]
            vals = ic_values[np.array(mask)]
            per_condition[label] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

        # Does this IC separate conditions?
        means = [per_condition[l]["mean"] for l in label_set]
        spread = max(means) - min(means)
        within_std = np.mean([per_condition[l]["std"] for l in label_set])

        ic_analysis.append({
            "component": ic_idx,
            "condition_spread": float(spread),
            "within_std": float(within_std),
            "discriminability": float(spread / within_std) if within_std > 0 else 0,
            "per_condition": per_condition,
        })

    ic_analysis.sort(key=lambda x: -x["discriminability"])

    return {
        "n_components": n_components,
        "components": ic_analysis,
        "most_discriminative": ic_analysis[0] if ic_analysis else None,
    }


# ══════════════════════════════════════════════════════════════
# 4. HANKEL/SSA ON PER-TOKEN EXPERT ACTIVATION
# ══════════════════════════════════════════════════════════════

def hankel_ssa(sequence, window_size=None):
    """Singular Spectrum Analysis via Hankel matrix decomposition.

    Constructs a Hankel (trajectory) matrix from the sequence and
    decomposes it via SVD. Returns the dominant temporal components.
    """
    N = len(sequence)
    if N < 4:
        return {"error": "sequence too short for SSA"}

    if window_size is None:
        window_size = min(N // 2, 20)

    K = N - window_size + 1

    # Construct Hankel (trajectory) matrix
    H = np.zeros((window_size, K))
    for i in range(window_size):
        H[i, :] = sequence[i:i+K]

    U, s, Vt = np.linalg.svd(H, full_matrices=False)

    # Contribution of each component
    total_var = np.sum(s**2)
    contributions = (s**2 / total_var * 100) if total_var > 0 else np.zeros_like(s)

    # GD threshold on Hankel SVs
    gd = gavish_donoho_threshold(s, window_size, K)

    return {
        "singular_values": s.tolist(),
        "contributions_pct": contributions.tolist(),
        "n_signal_components": gd["n_signal"],
        "gd_threshold": gd["threshold"],
        "dominant_sv": float(s[0]) if len(s) > 0 else 0,
        "sv_ratio_1_2": float(s[0] / s[1]) if len(s) > 1 and s[1] > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════
# DATA CAPTURE
# ══════════════════════════════════════════════════════════════

def capture_full_data(model, tokenizer, prompt_text, n_generate=50):
    """Capture EVERYTHING: per-layer SVD spectra + per-token routing."""
    import torch

    sys_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": ""}],
        tokenize=False, add_generation_prompt=True)
    n_sys = len(tokenizer(sys_text, return_tensors="pt")["input_ids"][0])

    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    n_prompt = inputs["input_ids"].shape[1]

    # Encoding pass
    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_router_logits=True)

    cache = out.past_key_values
    router_logits = getattr(out, 'router_logits', None)

    # Generate tokens
    past_kv = cache
    next_id = out.logits[:, -1].argmax(dim=-1)
    gen_router_logits = []
    gen_ids = []
    eos = {tokenizer.eos_token_id}

    for step in range(n_generate):
        with torch.no_grad():
            g_out = model(next_id.unsqueeze(0), past_key_values=past_kv,
                          use_cache=True, output_router_logits=True)
        past_kv = g_out.past_key_values
        g_rl = getattr(g_out, 'router_logits', None)
        if g_rl:
            gen_router_logits.append(g_rl)
        next_id = g_out.logits[:, -1].argmax(dim=-1)
        gen_ids.append(next_id.item())
        del g_out
        if next_id.item() in eos:
            break

    # Extract per-layer KV cache SVD spectra
    # Use the same iteration pattern as compute_cache_dimensionality in gpu_utils.py
    layer_spectra = []
    try:
        for layer_idx, layer in enumerate(past_kv):
            if isinstance(layer, tuple) and len(layer) >= 2:
                k = layer[0].squeeze(0).float().cpu().numpy()
            elif hasattr(layer, 'shape'):
                k = layer.float().cpu().numpy()
                if k.ndim < 2:
                    continue
            else:
                continue
            k_flat = k.reshape(-1, k.shape[-1])
            sv = np.linalg.svd(k_flat, compute_uv=False)
            layer_spectra.append({
                "layer": layer_idx,
                "singular_values": sv.tolist(),
                "m": k_flat.shape[0],
                "n": k_flat.shape[1],
            })
    except Exception as e:
        print(f"    Cache iteration error: {e}")
        # Fallback: use compute_cache_dimensionality which handles this model
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from gpu_utils import compute_cache_dimensionality
            geo = compute_cache_dimensionality(past_kv)
            spectra = geo.get("singular_spectra", [])
            for sp in spectra:
                sv = sp.get("key_singular_values", np.array([]))
                if len(sv) > 0:
                    layer_spectra.append({
                        "layer": sp["layer"],
                        "singular_values": sv.tolist() if hasattr(sv, 'tolist') else list(sv),
                        "m": len(sv),
                        "n": len(sv),
                    })
        except Exception as e2:
            print(f"    Fallback also failed: {e2}")

    # Extract per-token routing for user+generated tokens
    per_token_routing = []
    if router_logits:
        for pos in range(n_sys, n_prompt):
            token_experts = []
            for layer_idx, logits in enumerate(router_logits):
                if logits is None:
                    continue
                tl = logits[pos] if logits.dim() == 2 else logits[0, pos]
                probs = torch.softmax(tl.float(), dim=-1)
                top_v, top_i = torch.topk(probs, 4)
                token_experts.append({
                    "experts": top_i.cpu().tolist(),
                    "weights": top_v.cpu().tolist(),
                })
            per_token_routing.append({"position": pos, "phase": "encoding",
                                       "layers": token_experts})

        for step, step_rl in enumerate(gen_router_logits):
            token_experts = []
            for layer_idx, logits in enumerate(step_rl):
                if logits is None:
                    continue
                tl = logits[0] if logits.dim() == 2 else logits[0, 0]
                probs = torch.softmax(tl.float(), dim=-1)
                top_v, top_i = torch.topk(probs, 4)
                token_experts.append({
                    "experts": top_i.cpu().tolist(),
                    "weights": top_v.cpu().tolist(),
                })
            per_token_routing.append({"position": n_prompt + step, "phase": "generation",
                                       "layers": token_experts})

    del out, past_kv
    torch.cuda.empty_cache()

    return {
        "layer_spectra": layer_spectra,
        "per_token_routing": per_token_routing,
        "n_prompt_tokens": n_prompt,
        "n_sys_tokens": n_sys,
        "n_generated": len(gen_ids),
        "generated_text": tokenizer.decode(gen_ids, skip_special_tokens=True),
    }


# ══════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════

def run_analysis(model, tokenizer):
    print("\n  Capturing full per-token data...")
    all_data = {}

    for label, prompts in TARGET_PROMPTS.items():
        print(f"\n  [{label}]")
        all_data[label] = []
        for i, prompt in enumerate(prompts):
            data = capture_full_data(model, tokenizer, prompt)
            all_data[label].append(data)
            gen_preview = data["generated_text"][:60].replace('\n', ' ')
            print(f"    {i+1}/{len(prompts)}: {len(data['layer_spectra'])} layers, "
                  f"{len(data['per_token_routing'])} routed tokens")
            print(f"      gen: \"{gen_preview}...\"")
            sys.stdout.flush()

    print("\n" + "=" * 60)
    print("  ANALYSIS 1: Gavish-Donoho Optimal Thresholding")
    print("=" * 60)

    for label, runs in all_data.items():
        gd_signals = []
        for run in runs:
            for layer in run["layer_spectra"]:
                gd = gavish_donoho_threshold(layer["singular_values"],
                                              layer["m"], layer["n"])
                gd_signals.append(gd["n_signal"])
        mean_signal = np.mean(gd_signals) if gd_signals else 0
        print(f"  {label:20s}: mean signal SVs = {mean_signal:.1f} "
              f"(of {len(runs[0]['layer_spectra'][0]['singular_values']) if runs and runs[0]['layer_spectra'] else '?'})")

    print("\n" + "=" * 60)
    print("  ANALYSIS 2: Marchenko-Pastur Signal Detection")
    print("=" * 60)

    for label, runs in all_data.items():
        mp_signals = []
        for run in runs:
            for layer in run["layer_spectra"]:
                mp = marchenko_pastur_test(layer["singular_values"],
                                            layer["m"], layer["n"])
                mp_signals.append(mp["n_signal_above_mp"])
        mean_signal = np.mean(mp_signals) if mp_signals else 0
        print(f"  {label:20s}: mean SVs above MP edge = {mean_signal:.1f}")

    # Compare signal SV counts between conditions
    print("\n  GD signal band comparison (the middle band between frequency and noise):")
    for label, runs in all_data.items():
        mid_band = []
        for run in runs:
            for layer in run["layer_spectra"]:
                sv = np.array(layer["singular_values"])
                gd = gavish_donoho_threshold(sv, layer["m"], layer["n"])
                mp = marchenko_pastur_test(sv, layer["m"], layer["n"])
                # Middle band: above GD noise but below MP outliers
                n_above_gd = gd["n_signal"]
                n_above_mp = mp["n_signal_above_mp"]
                mid = n_above_gd - n_above_mp
                mid_band.append(max(0, mid))
        print(f"  {label:20s}: middle band SVs = {np.mean(mid_band):.1f} ± {np.std(mid_band):.1f}")

    print("\n" + "=" * 60)
    print("  ANALYSIS 3: ICA on Expert Routing Patterns")
    print("=" * 60)

    routing_matrices = {}
    for label, runs in all_data.items():
        matrices = []
        for run in runs:
            n_experts = 60
            expert_counts = np.zeros(n_experts)
            for token in run["per_token_routing"]:
                for layer_data in token["layers"]:
                    for eid in layer_data["experts"]:
                        if eid < n_experts:
                            expert_counts[eid] += 1
            total = expert_counts.sum() or 1
            matrices.append(expert_counts / total)
        routing_matrices[label] = np.array(matrices)

    ica_result = ica_routing_analysis(routing_matrices, list(TARGET_PROMPTS.keys()))
    if "error" not in ica_result:
        print(f"  Components extracted: {ica_result['n_components']}")
        top = ica_result["most_discriminative"]
        if top:
            print(f"  Most discriminative IC (#{top['component']}):")
            print(f"    Spread/std ratio: {top['discriminability']:.3f}")
            for cond, vals in top["per_condition"].items():
                print(f"    {cond:20s}: {vals['mean']:+.4f} ± {vals['std']:.4f}")
    else:
        print(f"  {ica_result['error']}")

    print("\n" + "=" * 60)
    print("  ANALYSIS 4: Hankel/SSA on Expert 45 Temporal Patterns")
    print("=" * 60)

    for label, runs in all_data.items():
        ssa_results = []
        for run in runs:
            e45_sequence = []
            for token in run["per_token_routing"]:
                e45_active = 0
                for layer_data in token["layers"]:
                    if 45 in layer_data["experts"]:
                        idx = layer_data["experts"].index(45)
                        e45_active += layer_data["weights"][idx]
                e45_sequence.append(e45_active)

            if len(e45_sequence) >= 4:
                ssa = hankel_ssa(e45_sequence)
                if "error" not in ssa:
                    ssa_results.append(ssa)

        if ssa_results:
            mean_signal = np.mean([s["n_signal_components"] for s in ssa_results])
            mean_ratio = np.mean([s["sv_ratio_1_2"] for s in ssa_results])
            print(f"  {label:20s}: SSA signal components = {mean_signal:.1f}, "
                  f"SV1/SV2 ratio = {mean_ratio:.2f}")
        else:
            print(f"  {label:20s}: insufficient data for SSA")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"moe_advanced_analysis_{timestamp}.json"

    summary = {
        "experiment": "51b_moe_advanced_analysis",
        "timestamp": datetime.now().isoformat(),
        "methods": ["gavish_donoho", "marchenko_pastur", "ica", "hankel_ssa"],
        "n_prompts_per_condition": len(list(TARGET_PROMPTS.values())[0]),
        "conditions": list(TARGET_PROMPTS.keys()),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print("\n  Done.")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  EXPERIMENT 51b: Advanced MoE Signal Analysis")
    print(f"  Model: {args.model}")
    print(f"  Methods: GD + MP + ICA + Hankel/SSA")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    run_analysis(model, tokenizer)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
