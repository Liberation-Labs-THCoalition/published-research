#!/usr/bin/env python3
"""Build v10 reader prompts with the OFFICIAL prompt builder, like the oracle and full-history arms.

    build_v10_prompts.py --config CFG.json --split dev --arm v10

Retrieval sees only v10.view(item). The entry handed to the official prepare_prompt keeps the
question and dates, plus ONLY the sessions (or turn windows) v10 selected, and it goes through the
same 'orig-session' path the full-history arm used. The builder sorts sessions by date and formats
them identically, so the reader sees exactly one difference between arms: which history is present.

Writes baseline/<arm>/prompts/qNNN.txt (NNN = index in longmemeval_s.json), so the existing
answer_one_arm.sh, judge and comparison scripts apply. baseline/<arm>/prompts_meta_<arm>.json records
per question what was selected and its evidence coverage. That file is for analysis; the reader never
sees it.
"""
import argparse, copy, json, os, re, sys
sys.path.insert(0, "/mnt/data1/lme_v2/v10")
sys.path.insert(0, "/mnt/data1/lme_v2/baseline")
import tiktoken
import v10
from build_baseline_prompts import load_prepare_prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["dev", "held", "all"], required=True)
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    if a.split != "dev":
        assert not os.access(a.config, os.W_OK), "held-out / all prompts need a FROZEN (read-only) config"
    cfg = {**v10.DEFAULT, **json.load(open(a.config))}
    if a.split == "all":
        data, _ = v10.load_split("dev")
        idxs = list(range(len(data)))
    else:
        data, idxs = v10.load_split(a.split)
    dense = v10.Dense() if "dense" in cfg["channels"] else None
    prepare_prompt = load_prepare_prompt()
    enc = tiktoken.get_encoding("o200k_base")
    out = f"/mnt/data1/lme_v2/baseline/{a.arm}"
    for d in ("prompts", "answers", "logs"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    rows = []
    for idx in idxs:
        item = data[idx]
        chosen, used, diag = v10.retrieve(v10.view(item), idx, cfg, dense)
        keep = sorted(chosen)
        entry = copy.deepcopy(item)       # prepare_prompt pops has_answer in place
        entry["haystack_dates"] = [item["haystack_dates"][si] for si in keep]
        entry["haystack_session_ids"] = [item["haystack_session_ids"][si] for si in keep]
        entry["haystack_sessions"] = [copy.deepcopy(item["haystack_sessions"][si]) if chosen[si] is None
                                      else [copy.deepcopy(item["haystack_sessions"][si][ti]) for ti in chosen[si]]
                                      for si in keep]
        prompt = prepare_prompt(entry, "orig-session", 10**6, False, "json", True,
                                tokenizer=enc, tokenizer_backend="openai",
                                max_retrieval_length=180_000, merge_key_expansion_into_value="none")
        # the same guards as the baseline arms, plus: the prompt holds exactly the selected sessions
        assert "has_answer" not in prompt, idx
        assert not re.search(r"answer_[0-9a-f]{8}|noans_|_abs\b", prompt), idx
        assert prompt.rstrip().endswith("Answer (step by step):") and f"Question: {item['question']}" in prompt, idx
        assert f"Current Date: {item['question_date']}" in prompt, idx
        assert prompt.count("\n### Session ") == len(keep), (idx, prompt.count("\n### Session "), len(keep))
        open(f"{out}/prompts/q{idx:03d}.txt", "w").write(prompt)
        ev = v10.evidence(item)
        inc = lambda si, ti: si in chosen and (chosen[si] is None or ti in chosen[si])
        rows.append({"idx": idx, "question_id": item["question_id"], "sessions": len(keep), **diag,
                     "history_budget_tokens": used, "prompt_o200k_tokens": len(enc.encode(prompt, allowed_special="all")),
                     "n_ev": len(ev), "ev_covered": sum(any(inc(si, ti) for ti in tis) for si, tis in ev.items())})
    json.dump({"config": cfg, "config_hash": v10.cfg_hash(cfg), "config_file": os.path.abspath(a.config),
               "split": a.split, "rows": rows}, open(f"{out}/prompts_meta_{a.arm}_{a.split}.json", "w"), indent=0)
    toks = sorted(r["prompt_o200k_tokens"] for r in rows)
    print(f"{a.arm}: {len(rows)} prompts, config {v10.cfg_hash(cfg)}; prompt o200k tokens median {toks[len(toks)//2]:,} "
          f"max {toks[-1]:,}; all evidence covered on {sum(r['ev_covered'] == r['n_ev'] and r['n_ev'] > 0 for r in rows)}/"
          f"{sum(r['n_ev'] > 0 for r in rows)} questions with evidence")

if __name__ == "__main__":
    main()
