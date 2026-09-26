#!/usr/bin/env python3
"""Judge LongMemEval v2 answers with the OFFICIAL answer-check prompts.

get_anscheck_prompt() is not retyped: it is extracted from official/evaluate_qa.py
(fetched from xiaowu0162/LongMemEval main) with ast and executed in isolation, and the
label rule is the official one: 'yes' in response.lower().

Differences from the official script, all forced and all disclosed:
  - judge model is a Claude model via `claude -p` (no OpenAI key exists here);
    official uses gpt-4o-2024-08-06, temperature 0, max_tokens 10;
  - claude -p cannot set temperature or max_tokens.

A judge call that fails is recorded as a FAILURE in logs/, never as a 0 -- the v1
judge scored 14 of 81 unparseable replies as 0.0, some of which said "score": 1.0.

Usage: judge.py --model claude-sonnet-5 [--jobs 3] [--tag sonnet5]
"""
import argparse, ast, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/mnt/data1/lme_v2")
DATASET = Path("/mnt/data1/datasets/longmemeval/longmemeval_s.json")

def load_official_prompt_fn():
    src = (ROOT / "official/evaluate_qa.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_anscheck_prompt")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "evaluate_qa.py", "exec"), ns)
    return ns["get_anscheck_prompt"]

get_anscheck_prompt = load_official_prompt_fn()

def judge_one(model, qid, entry, answer, outdir):
    out = outdir / f"{qid}.json"
    if out.exists():
        return "cached"
    prompt = get_anscheck_prompt(entry["question_type"], entry["question"], entry["answer"],
                                 answer, abstention="_abs" in entry["question_id"])
    try:
        p = subprocess.run(
            ["claude", "-p", "--model", model, "--system-prompt", "You are a helpful assistant.",
             "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
             "--settings", '{"disableAllHooks":true,"alwaysThinkingEnabled":false}',
             "--output-format", "json", "-"],
            input=prompt, capture_output=True, text=True, timeout=300, cwd=ROOT / "run_cwd")
        ev = json.loads(p.stdout)
        ev = ev if isinstance(ev, list) else [ev]
        res = next(e for e in reversed(ev) if isinstance(e, dict) and e.get("type") == "result")
        text = (res.get("result") or "").strip()
        models = list((res.get("modelUsage") or {}).keys())
        if res.get("is_error") or not text or models != [model]:
            raise RuntimeError(f"is_error={res.get('is_error')} empty={not text} models={models}")
    except Exception as e:
        with open(ROOT / "logs/judge_failures.log", "a") as f:
            f.write(f"{model}\t{qid}\t{type(e).__name__}: {str(e)[:120]}\n")
        return "failed"
    label = "yes" in text.lower()          # official rule, verbatim semantics
    rec = {"question_id": qid, "question_type": entry["question_type"],
           "abstention": "_abs" in qid, "judge_model": model, "judge_raw": text, "label": label}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1)); os.replace(tmp, out)
    return "judged"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    ds = {e["question_id"]: e for e in json.load(open(DATASET))}
    meta = {m["idx"]: m for m in json.load(open(ROOT / "prompts_meta.json"))}
    outdir = ROOT / f"judge_{a.tag}"; outdir.mkdir(exist_ok=True)
    work = []
    answers_dir = Path(os.environ.get("LME_ANSWERS", ROOT / "answers"))
    for f in sorted(answers_dir.glob("q*.json")):
        r = json.loads(f.read_text())
        qid = meta[r["idx"]]["question_id"]
        work.append((qid, ds[qid], r["answer"]))
    with ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(lambda w: judge_one(a.model, w[0], w[1], w[2], outdir), work))
    print({k: results.count(k) for k in set(results)}, f"of {len(work)} answers")

if __name__ == "__main__":
    main()
