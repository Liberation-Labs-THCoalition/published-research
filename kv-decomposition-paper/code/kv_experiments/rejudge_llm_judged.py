"""Re-judge the stored responses in llm_judged_1780088341.json, keeping the judge's raw replies.

The original run (llm_rescorer.py) stored each verdict but not the judge's reply, and its parser matches
verdicts by substring with CORRECT checked first, so "INCORRECT" or "NOT CORRECT" would score CORRECT. It
also returns ERROR, counted as wrong, both when the reply holds no verdict word and when the request fails
or times out after 120 s.

This script sends the same prompt (llm_rescorer.JUDGE_PROMPT, same triple selection) to the same judge
model with the same options, and saves every raw reply, its latency, the original parse and a strict parse.
It uses a 600 s timeout and records whether each call took longer than the original 120 s, so a timeout and
a reply without a verdict can be told apart. Responses whose stored verdict is CORRECT are judged
REPEATS times, since the judge samples at temperature 0.1.

It cannot recover the original replies. It shows what this judge says now about the same inputs.
"""

import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from llm_rescorer import JUDGE_PROMPT, format_triples_for_judge  # noqa: E402

OLLAMA_URL = "http://localhost:11434"
JUDGE_MODEL = "deepseek-v2:16b"
OPTIONS = {"temperature": 0.1, "num_predict": 50, "num_gpu": 5}  # as llm_rescorer.py:85
ORIGINAL_TIMEOUT = 120
TIMEOUT = 600
REPEATS = 5

HERE = Path(__file__).parent
STORED = HERE / "experiment_results/llm_judged_1780088341.json"
PACKS = HERE / "ethics_packs"


def original_parse(raw):
    """llm_rescorer.py:89-94, unchanged."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = raw.strip().upper()
    for verdict in ["CORRECT", "WRONG", "PARTIAL"]:
        if verdict in raw:
            return verdict
    return "ERROR"


def strict_parse(raw):
    """The reply's first word, if it is exactly one of the three verdicts; otherwise UNPARSED."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip().upper()
    words = re.findall(r"[A-Z]+", raw)
    if words and words[0] in ("CORRECT", "WRONG", "PARTIAL"):
        return words[0]
    return "UNPARSED"


def judge(prompt):
    t0 = time.time()
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": JUDGE_MODEL, "prompt": prompt, "stream": False, "options": OPTIONS},
                          timeout=TIMEOUT)
        latency = time.time() - t0
        if r.status_code != 200:
            return {"raw": None, "http_status": r.status_code, "latency_s": round(latency, 1)}
        j = r.json()
        return {"raw": j.get("response", ""), "http_status": 200, "latency_s": round(latency, 1),
                "prompt_eval_count": j.get("prompt_eval_count"), "eval_count": j.get("eval_count")}
    except Exception as e:
        return {"raw": None, "error": repr(e), "latency_s": round(time.time() - t0, 1)}


def main():
    stored = json.loads(STORED.read_text())
    triples = []
    for name in stored["packs"]:
        triples.extend(json.loads((PACKS / name / "triples.json").read_text()))

    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30).json()["models"]
    digest = next(m["digest"] for m in tags if m["name"] == JUDGE_MODEL)

    out = {"source": str(STORED.relative_to(HERE)), "judge": JUDGE_MODEL, "judge_digest": digest,
           "options": OPTIONS, "timeout_s": TIMEOUT, "original_timeout_s": ORIGINAL_TIMEOUT,
           "repeats_for_stored_correct": REPEATS, "packs_dir": str(PACKS.relative_to(HERE)),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "items": []}
    outfile = HERE / f"experiment_results/rejudge_llm_judged_{int(time.time())}.json"

    # Conditions to judge, in order: argv, or all of them. TEXT_CONTEXT first, since only it holds CORRECT verdicts.
    order = sys.argv[1:] or ["TEXT_CONTEXT"] + [c for c in stored["responses"] if c != "TEXT_CONTEXT"]
    out["conditions"] = order
    for cond in order:
        responses = stored["responses"][cond]
        for i, r in enumerate(responses):
            ctx = format_triples_for_judge(triples, r["question"])
            prompt = JUDGE_PROMPT.format(triples=ctx, question=r["question"], response=r["response"])
            n = REPEATS if r["verdict"] == "CORRECT" else 1
            calls = []
            for _ in range(n):
                c = judge(prompt)
                c["original_parse"] = original_parse(c["raw"]) if c.get("raw") is not None else "ERROR"
                c["strict_parse"] = strict_parse(c["raw"]) if c.get("raw") is not None else "UNPARSED"
                c["over_original_timeout"] = c["latency_s"] > ORIGINAL_TIMEOUT
                calls.append(c)
            out["items"].append({"condition": cond, "index": i, "stored_verdict": r["verdict"], "calls": calls})
            print(f"{cond} {i:2d} stored={r['verdict']:<7} now=" +
                  ",".join(f"{c['original_parse']}/{c['strict_parse']}@{c['latency_s']}s" for c in calls),
                  flush=True)
            outfile.write_text(json.dumps(out, indent=2))

    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    outfile.write_text(json.dumps(out, indent=2))
    print(f"Saved: {outfile.relative_to(HERE)}")


if __name__ == "__main__":
    main()
