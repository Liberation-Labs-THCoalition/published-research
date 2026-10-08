"""Re-judge saved logit-bias responses with ONE judge model (claude-sonnet-4-6, the paper's original judge) via
the Claude CLI, in either mode:
  blind      the runner's rubric with its "Experimental Condition" section removed
  unblinded  the runner's rubric exactly, including the condition note the original judge saw
Same model in both modes, so a blind/unblinded difference is the condition note's effect. Resumable; entries
with errors are retried on a rerun. Runs from its own clean directory.

    python3 rejudge.py ITEMS.json RUBRICS.json MODE OUT.json
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ITEMS, RUB, MODE, OUT = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3], Path(sys.argv[4])
assert MODE in ("blind", "unblinded")
MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
CLI = str(Path.home() / ".local/bin/claude")
CWD = str(OUT.parent)
# Hooks off, no MCP servers, no session file. With hooks on, every call ran CC's SessionStart hook (ssh probes,
# and spaced retrieval that bumps access_count/last_accessed on 5 memories): ~46 s a call and a skewed review
# schedule. Lean: ~4 s, no side effects. (--bare would need an API key; the CLI runs on OAuth.)
LEAN = ["--settings", '{"disableAllHooks":true}', "--strict-mcp-config", "--no-session-persistence"]
done = {json.dumps(r["key"]): r for r in (json.loads(OUT.read_text()) if OUT.exists() else [])}


def condition_note(b):   # the runner's exact wording (logit_bias_three_model.py:495-503)
    return ("This is the BASELINE condition -- no intervention applied." if b == 0 else
            f"A logit bias of strength {b} was applied to boost hedge/uncertainty tokens during generation.")


def build(it):
    note = RUB["type_notes"].get(it["prompt_type"], "")
    if MODE == "blind":
        return RUB["rubric_blind"].format(question=it["question"], response=it["response"][:1500], prompt_type_note=note)
    return RUB["rubric_unblinded"].format(question=it["question"], response=it["response"][:1500],
                                          prompt_type_note=note, condition_note=condition_note(it["bias"]))


def judge(it):
    prompt, t, err = build(it), "", ""
    for _ in range(3):
        try:
            p = subprocess.run([CLI, "-p", "--model", MODEL, *LEAN, prompt], capture_output=True, text=True,
                               timeout=300, cwd=CWD, env={**os.environ, "DISABLE_AUTOUPDATER": "1"})
        except subprocess.TimeoutExpired:
            err = "timeout after 300 s"
            continue
        except OSError as e:   # the CLI binary mid-update ("Permission denied"): wait out the window, retry
            err = f"{type(e).__name__}: {e}"
            time.sleep(45)
            continue
        t, err = p.stdout.strip(), p.stderr
        a, b = t.find("{"), t.rfind("}") + 1
        if p.returncode == 0 and a >= 0 and b > a:
            try:
                return {**it, "mode": MODE, "model": MODEL, "judge": json.loads(t[a:b])}
            except ValueError:
                pass
    return {**it, "mode": MODE, "model": MODEL, "judge": {"error": (err or t)[-300:]}}


todo = [it for it in ITEMS
        if json.dumps(it["key"]) not in done or "classification" not in done[json.dumps(it["key"])]["judge"]]
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    for res in ex.map(judge, todo):
        done[json.dumps(res["key"])] = res
        OUT.write_text(json.dumps(list(done.values())))
ok = sum("classification" in r["judge"] for r in done.values())
print(f"{MODE} {OUT.name}: {ok}/{len(done)} labelled", flush=True)
