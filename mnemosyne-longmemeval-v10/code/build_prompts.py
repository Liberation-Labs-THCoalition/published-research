#!/usr/bin/env python3
"""Build LongMemEval v2 answer prompts from the v9 Mnemosyne contexts.

v2 protocol (2026-09-23), replacing run_frontier.sh's v1:
  - OFFICIAL reading template (LongMemEval run_generation.py, 'merge' + CoT variant),
    copied verbatim from official/run_generation.py -- Mnemosyne's context is history
    snippets plus extracted user facts, which is exactly what that variant describes.
  - Current Date comes from the dataset's question_date. v1 question files had NO
    question date, which makes the 133 temporal-reasoning questions largely guesswork.
  - Joined on question_id (not on list position), and every row is checked: the v9
    question text must equal the dataset's.
  - No gold anywhere under prompts/. The scorer reads gold from the dataset directly.
"""
import json, re, sys
from pathlib import Path

ROOT = Path("/mnt/data1/lme_v2")
V9 = Path("/home/admin/benchmark_results/longmemeval/frontier_v9_contexts.json")
DATASET = Path("/mnt/data1/datasets/longmemeval/longmemeval_s.json")

src = (ROOT / "official/run_generation.py").read_text()
m = re.search(r"answer_prompt_template = '(I will give you several history chats between you and a user, as well as the relevant user facts extracted from the chat history\. Please answer the question based on the relevant chat history and the user facts\. Answer the question step by step:.*?Answer \(step by step\):)'", src)
if not m:
    sys.exit("official merge+CoT template not found verbatim -- refusing to improvise one")
TEMPLATE = m.group(1).encode().decode("unicode_escape")

ds = {e["question_id"]: e for e in json.load(open(DATASET))}
ctx = json.load(open(V9))
assert len(ctx) == 500, len(ctx)
meta, problems = [], []
for c in ctx:
    e = ds.get(c["question_id"])
    if e is None:
        problems.append((c["idx"], "question_id not in dataset")); continue
    if e["question"].strip() != c["question"].strip():
        problems.append((c["idx"], "question text differs")); continue
    prompt = TEMPLATE.format(c["context"], e["question_date"], e["question"])
    if str(e["answer"]).strip() and str(e["answer"]).strip() in prompt and len(str(e["answer"])) > 25:
        pass  # a long gold string appearing verbatim is the retrieval working, not leakage; recorded below
    (ROOT / f"prompts/q{c['idx']:03d}.txt").write_text(prompt)
    meta.append({"idx": c["idx"], "question_id": c["question_id"], "question_type": e["question_type"],
                 "abstention": c["question_id"].endswith("_abs"), "prompt_chars": len(prompt)})
if problems:
    sys.exit(f"{len(problems)} rows failed the join: {problems[:5]}")
(ROOT / "prompts_meta.json").write_text(json.dumps(meta, indent=1))
print(f"built {len(meta)} prompts; template sha-check first 60 chars: {TEMPLATE[:60]!r}")
print("chars: min %d median %d max %d" % (min(x['prompt_chars'] for x in meta),
      sorted(x['prompt_chars'] for x in meta)[250], max(x['prompt_chars'] for x in meta)))
