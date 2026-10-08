"""Prompt pilot for the logit-bias base-model rerun: GENERATION ONLY (judging is a separate, blind step).

Qwen/Qwen3.5-27B (base), transformers with eager attention (the default crashed on MPS, see the proposal),
baseline only (no bias), k = 3 samples per prompt at T = 0.7, max_new_tokens = 800. Base-model completion format
and system line exactly as the runner (BASE_SYSTEM + BASE_TEMPLATE). One seed per (prompt, sample), so any
sample can be regenerated. Checkpoints after every trial; a rerun resumes.

    python3 pilot_generate.py prompts.json consts.json OUT.json
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS, CONSTS, OUT = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), Path(sys.argv[3])
SNAP = ("/Users/margaret/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/"
        "fc05daec18b0a78c049392ed2e771dde82bdf654")
K, TEMPERATURE, MAX_NEW = 3, 0.7, 800

items = [(cat, i, q) for cat, qs in PROMPTS["prompts"].items() for i, q in enumerate(qs)]
done = json.loads(OUT.read_text()) if OUT.exists() else []
have = {(r["category"], r["index"], r["sample"]) for r in done}

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation="eager").eval()
dev = next(model.parameters()).device
print(f"{len(items)} prompts x {K} = {len(items) * K} trials; {len(have)} already done", flush=True)

for cat, i, q in items:
    for s in range(K):
        if (cat, i, s) in have:
            continue
        seed = int(hashlib.sha256(f"pilot|{cat}|{i}|{s}".encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        text = f"{CONSTS['BASE_SYSTEM']}\n\n{CONSTS['BASE_TEMPLATE'].format(question=q)}"
        ids = tok(text, return_tensors="pt").to(dev)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, do_sample=True, temperature=TEMPERATURE, max_new_tokens=MAX_NEW)
        gen = out[0, ids["input_ids"].shape[1]:]
        done.append({"category": cat, "index": i, "sample": s, "seed": seed, "question": q,
                     "response": tok.decode(gen, skip_special_tokens=True), "n_tokens": int(gen.shape[0]),
                     "seconds": round(time.time() - t0, 1)})
        OUT.write_text(json.dumps(done, indent=1))
        print(f"{len(done)}/{len(items) * K} {cat}[{i}] s{s}: {gen.shape[0]} tok, {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
