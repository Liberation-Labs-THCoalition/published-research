#!/usr/bin/env python3
"""Build same-reader baseline prompts for LongMemEval_S with the OFFICIAL prompt builder.

    build_baseline_prompts.py --arm oracle   # evidence sessions only  (longmemeval_oracle.json, 'oracle-session')
    build_baseline_prompts.py --arm full     # the whole raw history   (longmemeval_s.json,      'orig-session')

`prepare_prompt` is extracted with ast from official/run_generation.py and executed as-is, never
retyped (the module itself is not imported: it pulls in openai/transformers at top level). Settings
are the official CoT reading setup without fact merging: history_format=json, useronly=false,
cot=true, merge_key_expansion_into_value=none, all sessions kept.

Prompts are written as baseline/<arm>/prompts/qNNN.txt where NNN is the question's index in
longmemeval_s.json, so ../prompts_meta.json (idx -> question_id, type, abstention) and judge.py apply
unchanged. **The oracle file lists the same 500 questions in a DIFFERENT ORDER than the S file**, so
every entry is looked up by question_id, never by position.

**The oracle file also has its own timeline.** Question text, answer and type match the S file for
all 500, but question_date differs for all 500, and the gap between two evidence sessions of the same
question differs in 123 of 240 temporal-reasoning pairs (order preserved in 239). Each arm uses its
own file AS DISTRIBUTED, as the paper and the published oracle runs do, so the oracle arm is the
standard oracle number, not a same-timeline ceiling for our S run's temporal questions.
"""
import argparse, ast, copy, json, os, re, sys
import tiktoken

ROOT = "/mnt/data1/lme_v2"
DATA = {"oracle": "/mnt/data1/datasets/longmemeval/longmemeval_oracle.json",
        "full": "/mnt/data1/datasets/longmemeval/longmemeval_s.json"}
RETRIEVER = {"oracle": "oracle-session", "full": "orig-session"}
MAX_HISTORY_TOKENS = 180_000     # o200k tokens; a cap that should never bind on S (checked below)

def load_prepare_prompt():
    src = open(f"{ROOT}/official/run_generation.py").read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "prepare_prompt")
    ns = {"json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "official/run_generation.py", "exec"), ns)
    return ns["prepare_prompt"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=list(DATA), required=True)
    a = ap.parse_args()
    prepare_prompt = load_prepare_prompt()
    enc = tiktoken.get_encoding("o200k_base")
    s_order = json.load(open(DATA["full"]))
    arm_by_qid = {x["question_id"]: x for x in json.load(open(DATA[a.arm]))}
    assert set(arm_by_qid) == {x["question_id"] for x in s_order}, "question sets differ"
    meta = {m["idx"]: m for m in json.load(open(f"{ROOT}/prompts_meta.json"))}
    out = f"{ROOT}/baseline/{a.arm}/prompts"; os.makedirs(out, exist_ok=True)
    rows, truncated = [], 0
    for idx, s_item in enumerate(s_order):
        qid = s_item["question_id"]
        assert meta[idx]["question_id"] == qid, f"prompts_meta disagrees at {idx}"
        entry = copy.deepcopy(arm_by_qid[qid])            # prepare_prompt pops has_answer in place
        assert entry["question"] == s_item["question"] and entry["answer"] == s_item["answer"]
        hist_tokens = len(enc.encode("".join(json.dumps(s) for s in entry["haystack_sessions"]), allowed_special="all"))
        truncated += hist_tokens > MAX_HISTORY_TOKENS
        prompt = prepare_prompt(entry, RETRIEVER[a.arm], 10**6, False, "json", True,
                                tokenizer=enc, tokenizer_backend="openai",
                                max_retrieval_length=MAX_HISTORY_TOKENS, merge_key_expansion_into_value="none")
        # leakage and join guards: nothing marking evidence, the right question, the right date
        assert "has_answer" not in prompt, idx
        assert not re.search(r"answer_[0-9a-f]{8}|noans_|_abs\b", prompt), idx
        assert prompt.rstrip().endswith("Answer (step by step):") and f"Question: {s_item['question']}" in prompt, idx
        assert f"Current Date: {entry['question_date']}" in prompt, idx     # the arm file's own date
        n_sessions = prompt.count("\n### Session ")
        assert n_sessions == len(entry["haystack_sessions"]), (idx, n_sessions)
        open(f"{out}/q{idx:03d}.txt", "w").write(prompt)
        rows.append({"idx": idx, "question_id": qid, "sessions": n_sessions, "prompt_chars": len(prompt),
                     "history_o200k_tokens": hist_tokens})
    json.dump(rows, open(f"{ROOT}/baseline/{a.arm}/prompts_meta_{a.arm}.json", "w"), indent=0)
    toks = sorted(r["history_o200k_tokens"] for r in rows)
    print(f"{a.arm}: {len(rows)} prompts; history o200k tokens median {toks[len(toks)//2]:,} max {toks[-1]:,} "
          f"total {sum(toks):,}; over cap: {truncated}")

if __name__ == "__main__":
    main()
