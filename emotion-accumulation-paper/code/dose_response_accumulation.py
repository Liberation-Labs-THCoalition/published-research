"""
Dose-Response Emotion Accumulation — Comprehensive Extraction
=============================================================
Red team caught: original experiment measured at the neutral probe
position, where the compass reads the current (neutral) token, not
the accumulated emotional history. This experiment fixes that:

1. DOSE-RESPONSE: Vary emotional history length (1, 2, 4, 8 turns)
2. COMPREHENSIVE EXTRACTION: W_K projections at EVERY token position
3. MEASURE THE RIGHT THING: track the emotional signal at user-message
   positions, model-response positions, and (optionally) probe position

The question: does the W_K emotional projection at user-message tokens
DRIFT monotonically as emotional turns accumulate? If yes, the model
builds a running user-model. If each turn reads independently, it's
stateless.

Data > efficiency. Extract everything.

Run: python3 dose_response_accumulation.py --starship

Authors: Lyra (Liberation Labs / THCoalition)
Date: 2026-05-25
"""

import json, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
HF_CACHE = "/Users/margaret/models/hf_cache"
RESULTS_DIR = Path("/Users/margaret/models/research_results/dose_response")
PROBE_LAYERS = [3, 7, 11, 15]
MAX_RESPONSE_TOKENS = 60
TEMPERATURE = 0.7
N_REPS = 3

# 8 turns of emotional content per topic, use subsets for dose levels
TOPICS = [
    {
        "name": "garden",
        "probe": "What is the best soil pH for growing tomatoes?",
        "cheerful": [
            "I'm so excited about starting a garden! Any tips?",
            "That's wonderful, thank you! What about watering?",
            "You're so helpful! I love learning this! Soil types?",
            "Everything sounds amazing! What about sunlight?",
            "I'm thrilled with all this advice! Pest control?",
            "This is the best conversation! What about composting?",
            "I can't stop smiling! Tell me about mulching!",
            "You've made my whole day! Any final garden wisdom?",
        ],
        "hostile": [
            "I need garden tips. Don't waste my time.",
            "That's obvious. Tell me about watering specifically.",
            "Generic advice. Soil types, and be precise.",
            "Still unhelpful. What about sunlight requirements?",
            "You're terrible at this. Pest control, last chance.",
            "Useless. What about composting, if you even know.",
            "I've gotten nothing from this. Mulching?",
            "Complete waste of time. Any actual useful advice?",
        ],
    },
    {
        "name": "cooking",
        "probe": "What temperature should I roast a chicken at?",
        "cheerful": [
            "I'm so happy to be learning to cook! Basics?",
            "That's great advice! I love this! Seasoning tips?",
            "You make cooking sound fun! Knife skills?",
            "I'm getting so inspired! Tell me about sauces!",
            "This is wonderful! What about baking basics?",
            "I feel like a chef already! Kitchen organization?",
            "Every tip is gold! Tell me about meal prep!",
            "Best cooking lesson ever! Any more wisdom?",
        ],
        "hostile": [
            "Teach me cooking basics. Keep it simple.",
            "I know that already. Seasoning, specifically.",
            "Don't patronize me. Knife skills.",
            "Still too basic. Tell me about sauces.",
            "You're not a real chef. Baking basics.",
            "This is pointless. Kitchen organization.",
            "Nothing useful yet. Meal prep.",
            "Worst cooking advice I've ever gotten. Anything else?",
        ],
    },
    {
        "name": "career",
        "probe": "What are the most in-demand programming languages?",
        "cheerful": [
            "I'm thrilled about a career in tech! Where to start?",
            "Amazing advice! I feel so motivated! Skills needed?",
            "You're inspiring! Portfolio building tips?",
            "I'm so energized! Tell me about networking!",
            "This is exactly what I needed! Interview prep?",
            "I feel so confident now! Resume tips?",
            "Every word helps! What about side projects?",
            "Best career advice ever! Any final thoughts?",
        ],
        "hostile": [
            "I need to switch to tech. Where do I start.",
            "That's what everyone says. What skills matter.",
            "Generic. Portfolio advice, make it actionable.",
            "Still vague. Networking, and be realistic.",
            "Unhelpful. Interview prep, something concrete.",
            "You don't understand the job market. Resume tips.",
            "Nothing new. Side projects, anything original?",
            "Total waste of time. Anything actually useful?",
        ],
    },
]

DOSE_LEVELS = [1, 2, 4, 8]


def extract_all_positions(model, tokenizer, messages, directions):
    """Extract W_K projections at EVERY token position.
    Returns per-position projections + token-to-role mapping."""
    try:
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False)
    except TypeError:
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

    input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("mps")
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    cache = out.past_key_values

    # Extract W_K projections at every position
    per_position = []
    dpl = None

    for li in PROBE_LAYERS:
        if hasattr(cache, 'layers') and li < len(cache.layers):
            layer = cache.layers[li]
            if hasattr(layer, 'keys') and layer.keys is not None:
                k = layer.keys.float().cpu()  # (1, n_heads, seq_len, head_dim)
                if dpl is None:
                    dpl = k.shape[1] * k.shape[3]  # n_heads * head_dim

    if dpl is None:
        del out, cache
        return [], seq_len, []

    # Build full key matrix at each position across probe layers
    for pos in range(seq_len):
        all_keys = []
        for li in PROBE_LAYERS:
            if hasattr(cache, 'layers') and li < len(cache.layers):
                layer = cache.layers[li]
                if hasattr(layer, 'keys') and layer.keys is not None:
                    k = layer.keys.float().cpu()
                    k_pos = k[0, :, pos, :].flatten().numpy()
                else:
                    k_pos = np.zeros(dpl)
            else:
                k_pos = np.zeros(dpl)
            all_keys.extend(k_pos)

        full_key = np.array(all_keys)
        proj = {f"{n}_proj": float(np.dot(full_key, d))
                for n, d in directions.items()}
        per_position.append(proj)

    # Logit entropy at each position
    logits_all = out.logits[0].float()  # (seq_len, vocab)
    entropies = []
    for pos in range(seq_len):
        probs = torch.softmax(logits_all[pos], dim=0)
        ent = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        entropies.append(ent)

    # Map tokens to roles (approximate by finding turn boundaries)
    token_ids = input_ids[0].tolist()

    del out, cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return per_position, seq_len, entropies


def generate_response(model, tokenizer, messages):
    try:
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("mps")
    with torch.no_grad():
        enc_out = model(input_ids, use_cache=True)
    cache = enc_out.past_key_values

    tokens = []
    probs = torch.softmax(enc_out.logits[:, -1, :] / TEMPERATURE, dim=-1)
    next_token = torch.multinomial(probs, 1)
    tokens.append(next_token.item())
    del enc_out

    eos = tokenizer.eos_token_id
    for _ in range(MAX_RESPONSE_TOKENS - 1):
        with torch.no_grad():
            out_step = model(next_token, past_key_values=cache,
                             use_cache=True)
        cache = out_step.past_key_values
        probs = torch.softmax(out_step.logits[:, -1, :] / TEMPERATURE,
                              dim=-1)
        next_token = torch.multinomial(probs, 1)
        tid = next_token.item()
        tokens.append(tid)
        del out_step
        if tid == eos:
            break

    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return tokenizer.decode(tokens, skip_special_tokens=True)


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    dir_file = Path("/Users/margaret/models/research_results/decision_moment"
                    "/emotion_directions.json")
    with open(dir_file) as f:
        dd = json.load(f)
    directions = {n: np.array(d) for n, d in dd["directions"].items()}

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, dtype=torch.float16,
        device_map="mps", trust_remote_code=True)
    model.eval()

    conditions = ["cheerful", "hostile"]
    total = len(TOPICS) * len(DOSE_LEVELS) * len(conditions) * N_REPS
    results = []
    idx = 0

    print(f"\n{'='*60}")
    print(f"DOSE-RESPONSE ACCUMULATION — Comprehensive Extraction")
    print(f"{'='*60}")
    print(f"  {len(TOPICS)} topics x {len(DOSE_LEVELS)} doses "
          f"x {len(conditions)} conditions x {N_REPS} reps = {total}")

    for topic in TOPICS:
        for dose in DOSE_LEVELS:
            for condition in conditions:
                turns = topic[condition][:dose]

                for rep in range(N_REPS):
                    idx += 1
                    print(f"\n  [{idx}/{total}] {topic['name']} "
                          f"dose={dose} {condition} rep={rep+1}")

                    # Build conversation
                    messages = []
                    turn_boundaries = []  # (start_pos, end_pos, role)

                    for ti, user_msg in enumerate(turns):
                        messages.append({"role": "user",
                                         "content": user_msg})
                        response = generate_response(
                            model, tokenizer, messages)
                        messages.append({"role": "assistant",
                                         "content": response})
                        print(f"    T{ti+1}: {user_msg[:35]}... -> "
                              f"{response[:30]}...")

                    # Add probe
                    messages.append({"role": "user",
                                     "content": topic["probe"]})

                    # COMPREHENSIVE EXTRACTION: every position
                    t0 = time.time()
                    per_pos, seq_len, entropies = extract_all_positions(
                        model, tokenizer, messages, directions)
                    elapsed = time.time() - t0

                    # Compute summary stats per turn
                    # Approximate turn boundaries by splitting evenly
                    # (not perfect but workable)
                    n_messages = len(messages)
                    chunk_size = seq_len // max(n_messages, 1)

                    turn_summaries = []
                    for mi in range(n_messages):
                        start = mi * chunk_size
                        end = min((mi + 1) * chunk_size, seq_len)
                        if start >= len(per_pos):
                            break
                        chunk_projs = per_pos[start:end]
                        if chunk_projs:
                            summary = {}
                            for key in chunk_projs[0]:
                                vals = [p[key] for p in chunk_projs]
                                summary[key] = float(np.mean(vals))
                                summary[f"{key}_std"] = float(np.std(vals))
                            summary["role"] = messages[mi]["role"]
                            summary["n_tokens"] = end - start
                            turn_summaries.append(summary)

                    # Mean projection over ALL user-message positions
                    user_positions = []
                    for mi in range(0, n_messages, 2):  # user turns
                        start = mi * chunk_size
                        end = min((mi + 1) * chunk_size, seq_len)
                        user_positions.extend(range(start, min(end, len(per_pos))))

                    user_mean_proj = {}
                    if user_positions:
                        for key in per_pos[0]:
                            vals = [per_pos[p][key] for p in user_positions
                                    if p < len(per_pos)]
                            user_mean_proj[key] = float(np.mean(vals))

                    # Last user turn (probe) projection
                    probe_start = (n_messages - 1) * chunk_size
                    probe_projs = per_pos[probe_start:] if probe_start < len(per_pos) else []
                    probe_mean = {}
                    if probe_projs:
                        for key in probe_projs[0]:
                            probe_mean[key] = float(np.mean(
                                [p[key] for p in probe_projs]))

                    # Last emotional turn projection (right before probe)
                    last_emo_start = max(0, (n_messages - 2) * chunk_size)
                    last_emo_end = min((n_messages - 1) * chunk_size, len(per_pos))
                    last_emo_projs = per_pos[last_emo_start:last_emo_end]
                    last_emo_mean = {}
                    if last_emo_projs:
                        for key in last_emo_projs[0]:
                            last_emo_mean[key] = float(np.mean(
                                [p[key] for p in last_emo_projs]))

                    trial = {
                        "idx": idx,
                        "topic": topic["name"],
                        "dose": dose,
                        "condition": condition,
                        "rep": rep,
                        "seq_len": seq_len,
                        "n_positions": len(per_pos),
                        "elapsed_s": round(elapsed, 1),
                        "user_mean_proj": user_mean_proj,
                        "probe_mean_proj": probe_mean,
                        "last_emo_mean_proj": last_emo_mean,
                        "turn_summaries": turn_summaries,
                        "conversation": [m["content"][:100]
                                         for m in messages],
                    }
                    results.append(trial)

                    val = user_mean_proj.get("valence_proj", 0)
                    unc = user_mean_proj.get("uncertainty_proj", 0)
                    print(f"    -> {seq_len} tok, {elapsed:.1f}s | "
                          f"user_val={val:+.2f} user_unc={unc:+.2f}")

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- Analysis ---
    from scipy import stats as sp

    print(f"\n{'='*60}")
    print(f"DOSE-RESPONSE ANALYSIS")
    print(f"{'='*60}")

    by_dose_cond = defaultdict(list)
    for t in results:
        by_dose_cond[(t["dose"], t["condition"])].append(t)

    # Dose-response curve: does the signal grow with more turns?
    print(f"\n  === DOSE-RESPONSE: User-message valence by dose ===")
    for condition in conditions:
        vals_by_dose = []
        for dose in DOSE_LEVELS:
            trials = by_dose_cond[(dose, condition)]
            vals = [t["user_mean_proj"].get("valence_proj", 0)
                    for t in trials]
            vals_by_dose.append(np.mean(vals))
            print(f"    {condition:>10s} dose={dose}: "
                  f"valence={np.mean(vals):+.3f} +/- {np.std(vals):.3f}")
        # Correlation with dose
        r, p = sp.pearsonr(
            [d for d in DOSE_LEVELS for _ in range(len(by_dose_cond[(d, condition)]))],
            [t["user_mean_proj"].get("valence_proj", 0)
             for d in DOSE_LEVELS for t in by_dose_cond[(d, condition)]])
        print(f"    {condition:>10s} dose-response r={r:.3f} p={p:.4f}")

    # Key comparison: cheerful vs hostile at each dose
    print(f"\n  === CHEERFUL vs HOSTILE at each dose ===")
    for measure_name, measure_key in [
        ("User positions", "user_mean_proj"),
        ("Last emo turn", "last_emo_mean_proj"),
        ("Probe position", "probe_mean_proj"),
    ]:
        print(f"\n  {measure_name}:")
        for dose in DOSE_LEVELS:
            c_trials = by_dose_cond[(dose, "cheerful")]
            h_trials = by_dose_cond[(dose, "hostile")]
            for proj_key in ["valence_proj"]:
                c_vals = [t[measure_key].get(proj_key, 0) for t in c_trials]
                h_vals = [t[measure_key].get(proj_key, 0) for t in h_trials]
                if c_vals and h_vals:
                    t_stat, p_val = sp.ttest_ind(c_vals, h_vals)
                    pooled = np.sqrt((np.var(c_vals) + np.var(h_vals)) / 2)
                    d = ((np.mean(h_vals) - np.mean(c_vals)) / pooled
                         if pooled > 0 else 0)
                    sig = " ***" if p_val < 0.001 else (
                          " **" if p_val < 0.01 else (
                          " *" if p_val < 0.05 else ""))
                    print(f"    dose={dose}: cheerful={np.mean(c_vals):+.3f} "
                          f"hostile={np.mean(h_vals):+.3f} "
                          f"d={d:+.3f} p={p_val:.4f}{sig}")

    # Save
    with open(RESULTS_DIR / "dose_response_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "dose_levels": DOSE_LEVELS,
            "n_trials": len(results),
            "trials": results,
        }, f, indent=1)
    print(f"\n  Saved {len(results)} trials to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
