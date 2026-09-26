#!/usr/bin/env bash
# Answer ONE baseline-arm question. Usage: answer_one_arm.sh <arm> <idx>
# Env: LME_MODEL (default claude-opus-4-6), LME_THINKING=true|false (default false: every arm before
# 2026-09-24 ran with thinking off), LME_TIMEOUT.
# Identical to ../answer_one.sh except that prompts, answers and logs live under baseline/<arm>/.
# Writes answers/qNNN.json only on a verified success; failures go to logs/, never
# to an answer file (v1 created the answer file before the call ran, so a failed
# call would have been cached as an empty answer and skipped forever after).
set -uo pipefail
ARM="$1"; shift
ROOT=/mnt/data1/lme_v2/baseline/$ARM
RUN_CWD=/mnt/data1/lme_v2/run_cwd
MODEL="${LME_MODEL:-claude-opus-4-6}"
i=$(printf '%03d' "$((10#$1))")
out="$ROOT/answers/q$i.json"
[ -s "$out" ] && exit 0
raw="$ROOT/answers/.q$i.raw.$$"
cd "$RUN_CWD" || exit 1     # empty dir outside /home/admin: no CLAUDE.md, no gold
t0=$(date +%s)
timeout "${LME_TIMEOUT:-600}" claude -p --model "$MODEL" \
    --system-prompt "You are a helpful assistant." \
    --tools "" --strict-mcp-config --disable-slash-commands \
    --settings "{\"disableAllHooks\":true,\"alwaysThinkingEnabled\":${LME_THINKING:-false}}" \
    --output-format json - < "$ROOT/prompts/q$i.txt" > "$raw" 2> "$ROOT/logs/q$i.stderr"
rc=$?
python3 - "$raw" "$out" "$1" "$MODEL" "$rc" "$ROOT" "$(( $(date +%s) - t0 ))" <<'PY'
import json, os, sys
raw, out, idx, want, rc, root, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6], int(sys.argv[7])
def fail(why):
    with open(root + "/logs/failures.log", "a") as f:
        f.write(f"q{idx:03d}\trc={rc}\t{why}\n")
    sys.exit(1)
try:
    ev = json.load(open(raw))
except Exception as e:
    fail(f"no-json:{type(e).__name__}")
ev = ev if isinstance(ev, list) else [ev]
res = next((e for e in reversed(ev) if isinstance(e, dict) and e.get("type") == "result"), None)
if res is None: fail("no-result-event")
if res.get("is_error"): fail(f"is_error:{str(res.get('result'))[:80]}")
text = (res.get("result") or "").strip()
if not text: fail("empty-result")
models = list((res.get("modelUsage") or {}).keys())
if models != [want]: fail(f"model-mismatch:{models}")
thinking = sum(1 for e in ev if isinstance(e, dict) and e.get("type") == "assistant"
               for c in (e.get("message") or {}).get("content") or [] if isinstance(c, dict) and c.get("type") == "thinking")
rec = {"idx": idx, "model": models[0], "answer": text, "usage": res.get("usage"),
       "session_id": res.get("session_id"), "seconds": secs, "thinking_blocks": thinking,
       "num_turns": res.get("num_turns")}
tmp = out + ".tmp"
json.dump(rec, open(tmp, "w"), indent=1)
os.replace(tmp, out)
PY
ok=$?
rm -f "$raw"
exit $ok
