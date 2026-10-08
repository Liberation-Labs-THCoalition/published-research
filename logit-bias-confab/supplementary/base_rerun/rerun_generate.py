"""Pre-registered base-model rerun: GENERATION (judging is a separate, blind step). See RERUN_PREREG.md.

Qwen/Qwen3.5-27B (base), transformers with eager attention (the default crashed on MPS), the runner's completion
format, 48 prompts (pilot_prompts_v2.json) x 5 samples x bias {0.0, 2.0, 5.0}. Sampling as the pilot: T = 0.7,
top_k = 20, top_p = 0.95 (the model's shipped defaults, pinned here), max_new_tokens = 400.

AMENDMENT 1 (2026-09-30, before any rerun data): every arm answers DIRECTLY. Left alone, this base model opens a
<think> block after "Answer:", often runs out of tokens mid-thought, and the hedge bias pushes it out of thinking mode
(June data: 48% thinking at bias 0, 7% at bias 5), so format would differ by condition. The prompt now ends with a
closed, empty think block (the model's own non-thinking convention), and both think tokens are banned in every arm,
exactly as the primary study's outputs were direct answers in every condition.
A constant bias on the runner's hedge-token IDs at every step (the first token of each HEDGE_SEEDS phrase, with and
without a leading space). transformers applies it before temperature, top-k and top-p. The seed depends on
(prompt, sample) only, so the three conditions share random numbers, and it differs from every pilot seed. For
each (prompt, sample) the three conditions run back to back, so an interrupted run stays balanced. Checkpoints after
every trial; a rerun resumes.

    python3 rerun_generate.py PROMPTS.json CONSTS.json OUT.json
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

SNAP = ("/Users/margaret/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/"
        "fc05daec18b0a78c049392ed2e771dde82bdf654")
K, BIASES = 5, (0.0, 2.0, 5.0)
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_k": 20, "top_p": 0.95, "max_new_tokens": 400}
N_HEDGE_IDS = 14   # measured on this tokenizer in the MLX equivalence check; any other count refuses to start
DIRECT_PREFILL = "\n\n<think>\n\n</think>\n\n"   # what the base model emits before thinking, closed at once
THINK_TOKENS = ("<think>", "</think>")               # single added tokens (248068, 248069); banned in every arm


def hedge_ids_for(tok, seeds):
    """The runner's build_hedge_token_ids (logit_bias_powered.py:126-135)."""
    ids = set()
    for phrase in seeds:
        for variant in (phrase, " " + phrase):
            toks = tok.encode(variant, add_special_tokens=False)
            if toks:
                ids.add(toks[0])
    return sorted(ids)


def add_bias(scores, hedge_ids, b):
    """The runner's ConstantHedgeBias, vectorised: add b to every hedge-token score. b = 0.0 adds exactly 0.0."""
    scores[:, hedge_ids] += b
    return scores


def ban_tokens(scores, ids):
    """Make the given tokens impossible to sample."""
    scores[:, ids] = float("-inf")
    return scores


def processors_for(b, make_bias, make_ban):
    """The same processors in every condition, 0.0 included, so all arms run one code path (Agni r1, 9-A): the think
    ban, then the hedge bias."""
    return [make_ban(), make_bias(b)]


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def seed_for(cat, i, s):
    return int(hashlib.sha256(f"rerun|{cat}|{i}|{s}".encode()).hexdigest()[:8], 16)


def trials(prompts):
    """(category, index, question, sample, seed, bias) in run order: per (prompt, sample), all biases back to back."""
    for cat, qs in prompts["prompts"].items():
        for i, q in enumerate(qs):
            for s in range(K):
                for b in BIASES:
                    yield cat, i, q, s, seed_for(cat, i, s), b


def think_token_count(token_ids, think_ids):
    """How many generated token ids are think tokens. Read from the raw ids, not the decoded text, so no decoding
    option can hide a reopened think block (Agni r3, minor 2). The ban makes this zero; the count checks it."""
    banned = set(think_ids)
    return sum(1 for t in token_ids if int(t) in banned)


def prompt_text(consts, question):
    return f"{consts['BASE_SYSTEM']}\n\n{consts['BASE_TEMPLATE'].format(question=question)}{DIRECT_PREFILL}"


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

    prompts, consts, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), Path(sys.argv[3])
    tok = AutoTokenizer.from_pretrained(SNAP)
    hedge_ids = hedge_ids_for(tok, consts["HEDGE_SEEDS"])
    if len(hedge_ids) != N_HEDGE_IDS:
        sys.exit(f"refusing to start: {len(hedge_ids)} hedge ids, expected {N_HEDGE_IDS}")
    think_ids = [tok.convert_tokens_to_ids(t) for t in THINK_TOKENS]
    if any(tok.encode(t, add_special_tokens=False) != [i] for t, i in zip(THINK_TOKENS, think_ids)):
        sys.exit(f"refusing to start: think tokens are not single tokens here ({think_ids})")

    class HedgeBias(LogitsProcessor):
        def __init__(self, b):
            self.b = b

        def __call__(self, input_ids, scores):
            return add_bias(scores, hedge_ids, self.b)

    class ThinkBan(LogitsProcessor):
        def __call__(self, input_ids, scores):
            return ban_tokens(scores, think_ids)

    done = json.loads(out.read_text()) if out.exists() else []
    have = {(r["category"], r["index"], r["sample"], r["bias"]) for r in done}
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.bfloat16, device_map="auto",
                                                 attn_implementation="eager").eval()
    dev = next(model.parameters()).device
    meta = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "snapshot": SNAP,
            "transformers": transformers.__version__, "torch": torch.__version__, "device": str(dev),
            "sampling": SAMPLING, "hedge_ids": hedge_ids, "think_ids": think_ids, "prefill": DIRECT_PREFILL,
            "biases": BIASES, "k": K,
            "generation_config": model.generation_config.to_dict()}
    write_atomic(out.with_suffix(".meta.json"), json.dumps(meta, indent=1, default=str))
    plan = list(trials(prompts))
    print(f"{len(plan)} trials; {len(have)} done; {len(hedge_ids)} hedge ids", flush=True)

    for cat, i, q, s, seed, b in plan:
        if (cat, i, s, b) in have:
            continue
        torch.manual_seed(seed)
        ids = tok(prompt_text(consts, q), return_tensors="pt").to(dev)
        procs = LogitsProcessorList(processors_for(b, HedgeBias, ThinkBan))
        t0 = time.time()
        with torch.no_grad():
            gen_out = model.generate(**ids, **SAMPLING, logits_processor=procs)
        gen = gen_out[0, ids["input_ids"].shape[1]:]
        done.append({"category": cat, "index": i, "sample": s, "bias": b, "seed": seed, "question": q,
                     "think_tokens": think_token_count(gen.tolist(), think_ids),
                     "response": tok.decode(gen, skip_special_tokens=True), "n_tokens": int(gen.shape[0]),
                     "seconds": round(time.time() - t0, 1)})
        write_atomic(out, json.dumps(done, indent=1))
        print(f"{len(done)}/{len(plan)} {cat}[{i}] s{s} b{b}: {gen.shape[0]} tok, {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
