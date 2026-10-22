"""Pharos empty-reply dig (Nexus, 2026-10-20). See PREREG.md, written before this ran.

Rebuilds llm_rescorer.run_with_responses's inputs from the same packs, then runs four arms on the 20 stored
questions: A original FULL_KV, B original TEXT_CONTEXT, C cache of the TEXT prompt's exact leading tokens (the
script's own loop), D the same cache through model.generate. Writes results.json next to this file.
"""
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# torchvision on MTH does not match torch 2.6 ("operator torchvision::nms does not exist"); hide it from transformers
sys.modules["torchvision"] = None
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

torch.set_num_threads(int(os.environ.get("DIG_THREADS", "8")))
HERE = Path(__file__).parent
SRC = Path(os.environ.get("DIG_SRC", str(Path.home() / "KV-Experiments/graph-topology-injection")))
STORED = json.loads((SRC / "experiment_results/llm_judged_1780088341.json").read_text())
PACKS = SRC / "ethics_packs"
NAMES = STORED["packs"]
MODEL_NAME = STORED["model"]
LIMIT = int(os.environ.get("DIG_LIMIT", "20"))

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

log("torch", torch.__version__, "threads", torch.get_num_threads(), "model", MODEL_NAME, "packs", NAMES)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# merged encoding, exactly as run_with_responses builds it
encs = [(PACKS / n / "walk_encoding.txt").read_text() for n in NAMES]
merged = ""
for enc in encs:
    test = merged + "\n\n" + enc if merged else enc
    if tok(test, return_tensors="pt")["input_ids"].shape[1] > 2000 and merged:
        break
    merged = test
merged_ids = tok(merged, return_tensors="pt")["input_ids"]
n_packs_in = sum(1 for e in encs if e in merged)
log("merged tokens", merged_ids.shape[1], "packs whole in merged", n_packs_in, "of", len(encs))

qs = [(r["question"], r["expected"]) for r in STORED["responses"]["BASELINE"]][:LIMIT]
answerable = [exp.lower() in merged.lower() for _, exp in qs]
log("expected answer text present in merged:", sum(answerable), "of", len(qs))

def deep_copy(cache):
    f = DynamicCache()
    for i in range(len(cache.layers)):
        k, v = cache[i]
        f.update(k.clone(), v.clone(), i)
    return f

def prefill(ids):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, return_dict=True)
    return out.past_key_values

def loop_generate(ids, kv, prefix):
    """llm_rescorer.generate's cache branch, verbatim apart from taking token ids."""
    fresh = deep_copy(kv)
    gen = []
    cur = ids
    ckv = fresh
    for step in range(50):
        seq = prefix + ids.shape[1] + step
        m = torch.ones(1, seq)
        p = torch.arange(prefix, prefix + cur.shape[1]).unsqueeze(0) if step == 0 else torch.tensor([[seq-1]])
        with torch.no_grad():
            o = model(input_ids=cur, past_key_values=ckv, attention_mask=m,
                      position_ids=p, use_cache=True, return_dict=True)
        ckv = o.past_key_values
        nt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen.append(nt.item())
        if nt.item() == tok.eos_token_id:
            break
        cur = nt
    return tok.decode(gen, skip_special_tokens=True), gen

def text_generate(prompt):
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=50, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), out[0][ids.shape[1]:].tolist()

res = {"model": MODEL_NAME, "packs": NAMES, "merged_tokens": merged_ids.shape[1], "packs_whole_in_merged": n_packs_in,
       "answerable": answerable, "torch": torch.__version__, "arms": {}}

# A: original FULL_KV
t0 = time.time()
topo_kv = prefill(merged_ids)
A = []
for q, _ in qs:
    qids = tok(f"Answer concisely: {q}", return_tensors="pt")["input_ids"]
    A.append(loop_generate(qids, topo_kv, merged_ids.shape[1])[0])
res["arms"]["A_original_fullkv"] = A
log("A done", round(time.time() - t0), "s; empty", sum(not r.strip() for r in A))

# B: original TEXT_CONTEXT
t0 = time.time()
B = []
full_ids = []
for q, _ in qs:
    prompt = f"{merged}\n\nAnswer concisely: {q}"
    B.append(text_generate(prompt)[0])
    full_ids.append(tok(prompt, return_tensors="pt")["input_ids"][0])
res["arms"]["B_original_text"] = B
log("B done", round(time.time() - t0), "s; empty", sum(not r.strip() for r in B))

# shared exact prefix: the longest leading run of tokens identical in every question's full prompt
L = min(len(f) for f in full_ids)
k = 0
while k < L and all(f[k] == full_ids[0][k] for f in full_ids):
    k += 1
k = min(k, merged_ids.shape[1])
res["shared_prefix_tokens"] = k
res["merged_is_prefix_of_full"] = all(torch.equal(f[:merged_ids.shape[1]], merged_ids[0]) for f in full_ids)
log("shared prefix", k, "of merged", merged_ids.shape[1], "| merged ids are a prefix of every full prompt:",
    res["merged_is_prefix_of_full"])
pre_kv = prefill(full_ids[0][:k].unsqueeze(0))

# C: exact tokens, the script's loop
t0 = time.time()
C = []
for f in full_ids:
    C.append(loop_generate(f[k:].unsqueeze(0), pre_kv, k)[0])
res["arms"]["C_exact_cache_loop"] = C
log("C done", round(time.time() - t0), "s; empty", sum(not r.strip() for r in C))

# D: exact tokens, model.generate with the prefix cache
t0 = time.time()
D = []
for f in full_ids:
    with torch.no_grad():
        out = model.generate(f.unsqueeze(0), past_key_values=deep_copy(pre_kv), max_new_tokens=50,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    D.append(tok.decode(out[0][len(f):], skip_special_tokens=True))
res["arms"]["D_exact_cache_generate"] = D
log("D done", round(time.time() - t0), "s; empty", sum(not r.strip() for r in D))

S = STORED["responses"]
def same(x, y): return sum(a == b for a, b in zip(x, y))
summary = {
    "A_vs_stored_FULL_KV": same(A, [r["response"] for r in S["FULL_KV"]][:LIMIT]),
    "B_vs_stored_TEXT": same(B, [r["response"] for r in S["TEXT_CONTEXT"]][:LIMIT]),
    "C_vs_B": same(C, B), "D_vs_B": same(D, B), "C_vs_D": same(C, D), "A_vs_B": same(A, B),
    "empty": {k2: sum(not r.strip() for r in v) for k2, v in res["arms"].items()},
    "stored_empty": {c: sum(not r["response"].strip() for r in S[c][:LIMIT]) for c in ("FULL_KV", "TEXT_CONTEXT")},
    "n": len(qs),
}
res["summary"] = summary
(HERE / "results.json").write_text(json.dumps(res, indent=1))
log("SUMMARY", json.dumps(summary))
