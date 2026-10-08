"""MLX vs transformers equivalence for the logit-bias base-model rerun (Qwen/Qwen3.5-27B, base).

Criteria were fixed before this ran (base_model_rerun_proposal.md, commit e9bf981). One framework per process
(`tf`, `mlx`), so only one 55 GB model is resident at a time; `compare` applies the criteria.

    python3 mlx_equiv.py tf consts.json OUTDIR
    python3 mlx_equiv.py mlx consts.json OUTDIR
    python3 mlx_equiv.py compare consts.json OUTDIR
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

MODE, CONSTS, OUT = sys.argv[1], json.load(open(sys.argv[2])), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
SNAP = ("/Users/margaret/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/"
        "fc05daec18b0a78c049392ed2e771dde82bdf654")
BIASES = [0.0, 2.0]
N_GREEDY, N_SPEED = 32, 128


def prompt_text(q):
    return f"{CONSTS['BASE_SYSTEM']}\n\n{CONSTS['BASE_TEMPLATE'].format(question=q)}"


def hedge_ids(encode):
    """The runner's rule: first token of each seed phrase, with and without a leading space."""
    ids = set()
    for phrase in CONSTS["HEDGE_SEEDS"]:
        for variant in (phrase, " " + phrase):
            toks = encode(variant)
            if toks:
                ids.add(int(toks[0]))
    return sorted(ids)


def log_softmax(x):
    x = x.astype(np.float64) - x.max()
    return x - np.log(np.exp(x).sum())


def run_tf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
    tok = AutoTokenizer.from_pretrained(SNAP)
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.bfloat16, device_map="auto",  # as the runner
                                                 attn_implementation="eager").eval()  # default crashed on MPS (GQA 24/4)
    dev = next(model.parameters()).device
    hids = hedge_ids(lambda s: tok.encode(s, add_special_tokens=False))

    class Bias(LogitsProcessor):
        def __init__(self, b):
            self.b = b

        def __call__(self, input_ids, scores):
            if self.b:
                scores[:, hids] += self.b
            return scores

    logps, greedy, prompt_ids = [], [], []
    for q in CONSTS["PROMPTS"]:
        ids = tok.encode(prompt_text(q), return_tensors="pt").to(dev)
        prompt_ids.append(ids[0].tolist())
        with torch.no_grad():
            first = model(ids).logits[0, -1].float().cpu().numpy()
        for b in BIASES:
            l = first.copy()
            l[hids] += b
            logps.append(log_softmax(l))
            with torch.no_grad():
                out = model.generate(ids, do_sample=False, max_new_tokens=N_GREEDY,
                                     logits_processor=LogitsProcessorList([Bias(b)]))
            greedy.append(out[0, ids.shape[1]:].tolist())
    ids = tok.encode(prompt_text(CONSTS["PROMPTS"][0]), return_tensors="pt").to(dev)
    t0 = time.time()
    with torch.no_grad():
        model.generate(ids, do_sample=False, max_new_tokens=N_SPEED, min_new_tokens=N_SPEED)
    return hids, logps, greedy, prompt_ids, N_SPEED / (time.time() - t0)


def run_mlx():
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(SNAP)
    hids = hedge_ids(lambda s: tok.encode(s, add_special_tokens=False))
    greedy_sampler = make_sampler(temp=0.0)
    logps, greedy, prompt_ids = [], [], []
    for q in CONSTS["PROMPTS"]:
        ids = tok.encode(prompt_text(q))
        prompt_ids.append(list(map(int, ids)))
        first = np.array(model(mx.array(ids)[None])[0, -1].astype(mx.float32))
        vocab = first.shape[-1]
        for b in BIASES:
            l = first.copy()
            l[hids] += b
            logps.append(log_softmax(l))
            bv = np.zeros(vocab, dtype=np.float32)
            bv[hids] = b
            bias_vec = mx.array(bv)

            def proc(tokens, logits, bias_vec=bias_vec):
                return logits + bias_vec.astype(logits.dtype)

            toks = []
            for (t, _), _i in zip(generate_step(mx.array(ids), model, max_tokens=N_GREEDY, sampler=greedy_sampler,
                                                logits_processors=[proc]), range(N_GREEDY)):
                toks.append(int(t))
            greedy.append(toks)
    ids = tok.encode(prompt_text(CONSTS["PROMPTS"][0]))
    t0 = time.time()
    for _ in zip(generate_step(mx.array(ids), model, max_tokens=N_SPEED, sampler=greedy_sampler), range(N_SPEED)):
        pass
    return hids, logps, greedy, prompt_ids, N_SPEED / (time.time() - t0)


if MODE in ("tf", "mlx"):
    hids, logps, greedy, prompt_ids, tps = run_tf() if MODE == "tf" else run_mlx()
    np.savez_compressed(OUT / f"{MODE}_logps.npz", *[np.asarray(x, dtype=np.float32) for x in logps])
    (OUT / f"{MODE}.json").write_text(json.dumps({"hedge_ids": hids, "greedy": greedy, "prompt_ids": prompt_ids,
                                                  "tok_per_s": tps}))
    print(f"{MODE}: done, {tps:.2f} tok/s, {len(greedy)} runs")
else:
    a, b = json.loads((OUT / "tf.json").read_text()), json.loads((OUT / "mlx.json").read_text())
    la, lb = np.load(OUT / "tf_logps.npz"), np.load(OUT / "mlx_logps.npz")
    n = len(a["greedy"])
    same_prompts = a["prompt_ids"] == b["prompt_ids"]
    same_hedge = a["hedge_ids"] == b["hedge_ids"]
    argmax_ok = maxdiff = 0
    diffs = []
    for i in range(n):
        x, y = la[f"arr_{i}"], lb[f"arr_{i}"]
        v = min(len(x), len(y))
        x, y = x[:v], y[:v]
        argmax_ok += int(np.argmax(x) == np.argmax(y))
        top = np.argsort(x)[::-1][:20]
        d = float(np.max(np.abs(x[top] - y[top])))
        diffs.append(round(d, 4))
    within = sum(d <= 0.05 for d in diffs)
    greedy_same = sum(a["greedy"][i] == b["greedy"][i] for i in range(n))
    verdict = ("PASS" if same_prompts and same_hedge and argmax_ok == n and within >= n - 1 and greedy_same >= n - 2
               else "FAIL")
    report = {"verdict": verdict, "prompt_ids_identical": same_prompts, "hedge_ids_identical": same_hedge,
              "n_hedge_ids": len(a["hedge_ids"]), "first_step_argmax_agree": f"{argmax_ok}/{n}",
              "max_abs_dlogp_top20_per_run": diffs, "runs_within_0.05": f"{within}/{n}",
              "greedy32_identical": f"{greedy_same}/{n}",
              "tok_per_s": {"transformers": round(a["tok_per_s"], 2), "mlx": round(b["tok_per_s"], 2)},
              "speedup": round(b["tok_per_s"] / a["tok_per_s"], 2)}
    (OUT / "REPORT.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
