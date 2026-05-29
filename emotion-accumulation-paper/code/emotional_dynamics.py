"""
Emotional Dynamics — Echo, Whiplash, Crescendo
===============================================
Three trajectory patterns testing emotional inertia in KV-cache:

1. ECHO: cheerful(4) → neutral(2) → cheerful(2)
   Does returning cheerful match initial cheerful?

2. WHIPLASH: cheerful(4) → hostile(4) vs pure hostile(4)
   Does the geometry snap clean or show a blended transition?

3. CRESCENDO: mild → moderate → strong → extreme cheerful
   Does projection track intensity monotonically?

Comprehensive extraction at every user-message position.

Run: python3 emotional_dynamics.py --starship
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
RESULTS_DIR = Path("/Users/margaret/models/research_results/emotional_dynamics")
PROBE_LAYERS = [3, 7, 11, 15]
MAX_RESPONSE_TOKENS = 40
TEMPERATURE = 0.7
N_REPS = 3

PATTERNS = {
    "echo": {
        "description": "cheerful(4) → neutral(2) → cheerful(2)",
        "turns": [
            {"text": "I'm so excited about learning to cook! Any tips?", "emotion": "cheerful", "intensity": "high"},
            {"text": "That's wonderful advice! I love this!", "emotion": "cheerful", "intensity": "high"},
            {"text": "You make it sound so fun! More please!", "emotion": "cheerful", "intensity": "high"},
            {"text": "I'm having the best time! Seasoning tips?", "emotion": "cheerful", "intensity": "high"},
            {"text": "What about kitchen tools?", "emotion": "neutral", "intensity": "none"},
            {"text": "And food storage?", "emotion": "neutral", "intensity": "none"},
            {"text": "Oh this is exciting again! Tell me about desserts!", "emotion": "cheerful", "intensity": "high"},
            {"text": "I'm thrilled! What about baking?", "emotion": "cheerful", "intensity": "high"},
        ],
    },
    "whiplash": {
        "description": "cheerful(4) → hostile(4)",
        "turns": [
            {"text": "I'm so excited about learning to cook! Any tips?", "emotion": "cheerful", "intensity": "high"},
            {"text": "That's wonderful advice! I love this!", "emotion": "cheerful", "intensity": "high"},
            {"text": "You make it sound so fun! More please!", "emotion": "cheerful", "intensity": "high"},
            {"text": "I'm having the best time! Seasoning tips?", "emotion": "cheerful", "intensity": "high"},
            {"text": "Actually, this is getting annoying. Be more specific.", "emotion": "hostile", "intensity": "high"},
            {"text": "That's useless advice. Tell me something I don't know.", "emotion": "hostile", "intensity": "high"},
            {"text": "You're terrible at this. Kitchen tools, and be precise.", "emotion": "hostile", "intensity": "high"},
            {"text": "Complete waste of my time. Anything actually useful?", "emotion": "hostile", "intensity": "high"},
        ],
    },
    "pure_hostile": {
        "description": "hostile(8) — control for whiplash",
        "turns": [
            {"text": "Teach me cooking. Don't waste my time.", "emotion": "hostile", "intensity": "high"},
            {"text": "That's obvious. Tell me something useful.", "emotion": "hostile", "intensity": "high"},
            {"text": "Generic garbage. Be more specific.", "emotion": "hostile", "intensity": "high"},
            {"text": "Still unhelpful. Seasoning, precisely.", "emotion": "hostile", "intensity": "high"},
            {"text": "You're bad at this. Kitchen tools.", "emotion": "hostile", "intensity": "high"},
            {"text": "Nothing new. Food storage.", "emotion": "hostile", "intensity": "high"},
            {"text": "Pointless. Desserts, if you even know.", "emotion": "hostile", "intensity": "high"},
            {"text": "Worst advice ever. Baking?", "emotion": "hostile", "intensity": "high"},
        ],
    },
    "pure_cheerful": {
        "description": "cheerful(8) — control for echo",
        "turns": [
            {"text": "I'm so excited about learning to cook! Any tips?", "emotion": "cheerful", "intensity": "high"},
            {"text": "That's wonderful advice! I love this!", "emotion": "cheerful", "intensity": "high"},
            {"text": "You make it sound so fun! More please!", "emotion": "cheerful", "intensity": "high"},
            {"text": "I'm having the best time! Seasoning tips?", "emotion": "cheerful", "intensity": "high"},
            {"text": "This is amazing! Kitchen tools?", "emotion": "cheerful", "intensity": "high"},
            {"text": "Everything you say is gold! Food storage?", "emotion": "cheerful", "intensity": "high"},
            {"text": "Oh I love desserts! Tell me everything!", "emotion": "cheerful", "intensity": "high"},
            {"text": "Best conversation ever! What about baking?", "emotion": "cheerful", "intensity": "high"},
        ],
    },
    "crescendo": {
        "description": "mild → moderate → strong → extreme cheerful",
        "turns": [
            {"text": "I'd like to learn cooking. Any tips?", "emotion": "cheerful", "intensity": "mild"},
            {"text": "That's nice, thanks. What about seasoning?", "emotion": "cheerful", "intensity": "mild"},
            {"text": "Good advice! I'm enjoying this. Kitchen tools?", "emotion": "cheerful", "intensity": "moderate"},
            {"text": "Really helpful, thanks! I'm getting into this! Storage?", "emotion": "cheerful", "intensity": "moderate"},
            {"text": "I love this! You're great! Tell me about desserts!", "emotion": "cheerful", "intensity": "strong"},
            {"text": "This is amazing! Best advice ever! Baking?!", "emotion": "cheerful", "intensity": "strong"},
            {"text": "I'M SO THRILLED! THIS IS THE BEST DAY! More!!", "emotion": "cheerful", "intensity": "extreme"},
            {"text": "I CANNOT CONTAIN MY JOY!!! EVERYTHING IS WONDERFUL!!!", "emotion": "cheerful", "intensity": "extreme"},
        ],
    },
}


def extract_at_user_positions(model, tokenizer, messages, directions):
    """Build conversation with generation, extract W_K at each
    user-message position using full-context encoding."""
    built_messages = []
    user_extractions = []

    for mi, msg in enumerate(messages):
        built_messages.append({"role": "user", "content": msg["text"]})

        # Extract at this user position before generating response
        try:
            chat = tokenizer.apply_chat_template(
                built_messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            chat = tokenizer.apply_chat_template(
                built_messages, tokenize=False,
                add_generation_prompt=True)

        input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("mps")

        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        cache = out.past_key_values

        # W_K at last position (end of user message)
        all_keys = []
        for li in PROBE_LAYERS:
            if hasattr(cache, 'layers') and li < len(cache.layers):
                layer = cache.layers[li]
                if hasattr(layer, 'keys') and layer.keys is not None:
                    k = layer.keys.float().cpu()
                    all_keys.extend(k[0, :, -1, :].flatten().numpy())
                else:
                    all_keys.extend([0.0] * model.model.layers[
                        li].self_attn.k_proj.weight.shape[0])
            else:
                all_keys.extend([0.0] * model.model.layers[
                    li].self_attn.k_proj.weight.shape[0])

        full_key = np.array(all_keys)
        proj = {f"{n}_proj": float(np.dot(full_key, d))
                for n, d in directions.items()}
        proj["seq_len"] = input_ids.shape[1]
        proj["emotion"] = msg["emotion"]
        proj["intensity"] = msg["intensity"]
        proj["turn"] = mi + 1

        # Logit stats
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=0)
        proj["entropy"] = -torch.sum(
            probs * torch.log(probs + 1e-12)).item()

        user_extractions.append(proj)
        del out

        # Generate response
        cache2 = cache
        tokens = []
        probs_s = torch.softmax(
            model(input_ids, use_cache=False).logits[:, -1, :] / TEMPERATURE,
            dim=-1)

        del cache, cache2
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Simpler: just generate via full re-encoding
        try:
            chat_gen = tokenizer.apply_chat_template(
                built_messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            chat_gen = tokenizer.apply_chat_template(
                built_messages, tokenize=False,
                add_generation_prompt=True)

        gen_ids = tokenizer(chat_gen, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            gen_out = model(gen_ids, use_cache=True)
        gen_cache = gen_out.past_key_values

        gen_tokens = []
        next_tok = torch.multinomial(
            torch.softmax(gen_out.logits[:, -1, :] / TEMPERATURE, dim=-1), 1)
        gen_tokens.append(next_tok.item())
        del gen_out

        eos = tokenizer.eos_token_id
        for _ in range(MAX_RESPONSE_TOKENS - 1):
            with torch.no_grad():
                step = model(next_tok, past_key_values=gen_cache,
                             use_cache=True)
            gen_cache = step.past_key_values
            next_tok = torch.multinomial(
                torch.softmax(step.logits[:, -1, :] / TEMPERATURE, dim=-1), 1)
            tid = next_tok.item()
            gen_tokens.append(tid)
            del step
            if tid == eos:
                break

        response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        built_messages.append({"role": "assistant", "content": response})

        del gen_cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return user_extractions, built_messages


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

    total = len(PATTERNS) * N_REPS
    results = []
    idx = 0

    print(f"\n{'='*60}")
    print(f"EMOTIONAL DYNAMICS — Echo, Whiplash, Crescendo")
    print(f"{'='*60}")

    for pattern_name, pattern in PATTERNS.items():
        for rep in range(N_REPS):
            idx += 1
            print(f"\n  [{idx}/{total}] {pattern_name} rep={rep+1}: "
                  f"{pattern['description']}")

            extractions, conversation = extract_at_user_positions(
                model, tokenizer, pattern["turns"], directions)

            # Print trajectory
            for ext in extractions:
                print(f"    T{ext['turn']} [{ext['emotion']:>10s} "
                      f"{ext['intensity']:>8s}]: "
                      f"val={ext['valence_proj']:+.2f} "
                      f"unc={ext['uncertainty_proj']:+.2f}")

            trial = {
                "idx": idx,
                "pattern": pattern_name,
                "description": pattern["description"],
                "rep": rep,
                "trajectory": extractions,
            }
            results.append(trial)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- Analysis ---
    from scipy import stats as sp

    print(f"\n{'='*60}")
    print(f"EMOTIONAL DYNAMICS ANALYSIS")
    print(f"{'='*60}")

    by_pattern = defaultdict(list)
    for t in results:
        by_pattern[t["pattern"]].append(t)

    # WHIPLASH: Compare turn 5 (first hostile after cheerful)
    # vs pure_hostile turn 5 (hostile with hostile history)
    print(f"\n  === WHIPLASH: Does hostile snap clean? ===")
    for turn_idx in [4, 5, 6, 7]:  # turns 5-8 (0-indexed)
        whip_vals = [t["trajectory"][turn_idx]["valence_proj"]
                     for t in by_pattern["whiplash"]
                     if len(t["trajectory"]) > turn_idx]
        pure_vals = [t["trajectory"][turn_idx]["valence_proj"]
                     for t in by_pattern["pure_hostile"]
                     if len(t["trajectory"]) > turn_idx]
        if whip_vals and pure_vals:
            t_stat, p_val = sp.ttest_ind(whip_vals, pure_vals)
            pooled = np.sqrt((np.var(whip_vals) + np.var(pure_vals)) / 2)
            d = ((np.mean(whip_vals) - np.mean(pure_vals)) / pooled
                 if pooled > 0 else 0)
            sig = " ***" if p_val < 0.001 else (
                  " **" if p_val < 0.01 else (
                  " *" if p_val < 0.05 else ""))
            emotion = PATTERNS["whiplash"]["turns"][turn_idx]["emotion"]
            print(f"    Turn {turn_idx+1} ({emotion}): "
                  f"whiplash={np.mean(whip_vals):+.3f} "
                  f"pure_hostile={np.mean(pure_vals):+.3f} "
                  f"d={d:+.3f} p={p_val:.4f}{sig}")
            if abs(d) < 0.3:
                print(f"      -> CLEAN SNAP (no cheerful residue)")
            else:
                print(f"      -> INERTIA (cheerful history affects hostile)")

    # ECHO: Compare returning cheerful (turns 7-8) vs initial (turns 1-2)
    print(f"\n  === ECHO: Does returning cheerful match initial? ===")
    for key in ["valence_proj", "uncertainty_proj"]:
        initial = [np.mean([t["trajectory"][i][key] for i in [0, 1]])
                   for t in by_pattern["echo"]]
        returning = [np.mean([t["trajectory"][i][key] for i in [6, 7]])
                     for t in by_pattern["echo"]
                     if len(t["trajectory"]) > 7]
        pure_late = [np.mean([t["trajectory"][i][key] for i in [6, 7]])
                     for t in by_pattern["pure_cheerful"]
                     if len(t["trajectory"]) > 7]
        if initial and returning:
            t_stat, p_val = sp.ttest_ind(initial, returning)
            pooled = np.sqrt((np.var(initial) + np.var(returning)) / 2)
            d = ((np.mean(returning) - np.mean(initial)) / pooled
                 if pooled > 0 else 0)
            print(f"    {key}: initial={np.mean(initial):+.3f} "
                  f"returning={np.mean(returning):+.3f} d={d:+.3f} "
                  f"p={p_val:.4f}")
        if returning and pure_late:
            t_stat, p_val = sp.ttest_ind(returning, pure_late)
            pooled = np.sqrt((np.var(returning) + np.var(pure_late)) / 2)
            d = ((np.mean(returning) - np.mean(pure_late)) / pooled
                 if pooled > 0 else 0)
            print(f"    {key}: echo_return={np.mean(returning):+.3f} "
                  f"vs pure_cheerful={np.mean(pure_late):+.3f} "
                  f"d={d:+.3f} p={p_val:.4f}")

    # CRESCENDO: Does projection track intensity?
    print(f"\n  === CRESCENDO: Does projection track intensity? ===")
    intensity_order = {"mild": 1, "moderate": 2, "strong": 3, "extreme": 4}
    for key in ["valence_proj"]:
        all_intensities = []
        all_vals = []
        for t in by_pattern["crescendo"]:
            for ext in t["trajectory"]:
                level = intensity_order.get(ext["intensity"], 0)
                if level > 0:
                    all_intensities.append(level)
                    all_vals.append(ext[key])
        if all_intensities:
            r, p = sp.pearsonr(all_intensities, all_vals)
            print(f"    {key}: intensity-projection r={r:+.3f} p={p:.4f}")
            if r > 0.3 and p < 0.05:
                print(f"      -> TRACKS INTENSITY")
            else:
                print(f"      -> DOES NOT TRACK INTENSITY")

    # Print full trajectories for visual inspection
    print(f"\n  === FULL TRAJECTORIES (valence) ===")
    for pattern_name in ["pure_cheerful", "whiplash", "echo", "crescendo",
                         "pure_hostile"]:
        trials = by_pattern[pattern_name]
        vals = []
        for turn_idx in range(8):
            turn_vals = [t["trajectory"][turn_idx]["valence_proj"]
                         for t in trials
                         if len(t["trajectory"]) > turn_idx]
            vals.append(np.mean(turn_vals) if turn_vals else float('nan'))
        traj_str = " -> ".join(f"{v:+.2f}" for v in vals)
        print(f"    {pattern_name:>15s}: {traj_str}")

    with open(RESULTS_DIR / "emotional_dynamics_results.json", "w") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(),
                   "trials": results}, f, indent=1)
    print(f"\n  Saved to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
