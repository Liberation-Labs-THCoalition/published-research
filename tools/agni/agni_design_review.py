#!/usr/bin/env python3
"""
Agni Design Review — Pre-run red-team of experiment methodology.
Submits experiment script + pre-registration to a local model for hostile review.

Usage:
  python3 agni_design_review.py <script.py> <prereg.json>
  ssh margaret "cd ~/experiments && python3 agni_design_review.py script.py prereg.json"

CHANGELOG
  1.2.0 (2026-11-07): No silent cuts, as in agni_phases.py. The script is sent whole, with no
        function-prioritized excerpt, and hypotheses and threats are no longer sliced. A prompt over
        AGNI_LOCAL_CHAR_BUDGET (default 60000) is verdict INVALID, exit 2, before any model call,
        written over any older record. num_ctx is now set (LLM_NUM_CTX, default 32768): unset,
        Ollama trims the prompt with no signal, whatever was sent. The payload goes over stdin:
        a whole prompt can pass the command-line limit.
        Round 2: every prereg key reaches the model (a singular `hypothesis` key, or any key
        beyond hypotheses and threats such as falsification_criteria, used to be dropped without a
        word); the record lists them. The verdict comes from a line that starts VERDICT, OVERALL
        VERDICT or FINAL VERDICT, outside code: the fallback to a verdict word anywhere read "I would
        not call this design approved" as APPROVED, exit 0. INVALID is written before the model call
        and on an empty reply, so an older APPROVED cannot outlive a run that gave no review.
  1.1.0 (2026-05-29, Lyra): Fix four reliability bugs found when the gate
        silently returned empty on the persona_per_level review.
        - B1 empty-output: read `thinking` field as fallback (reasoning models
          on ollama put output there); raise num_predict 4000->12000.
        - B2 silent-empty-pass: FAIL LOUD (exit 2) if the review is empty —
          a gate must never let "no output" read as "no findings / pass".
        - B3 clobbering: output file named per-experiment, not the hardcoded
          decision_moment_agni_review.json (every review overwrote the same file).
        - B4 wrong-content review: section extraction was decision_moment-specific
          (hardcoded markers); for any other script it sent only text[:3000]
          (docstring), so the reviewer never saw the methodology and hallucinated
          findings. Now sends the full script (function-prioritized if oversized).
        - Also: verdict parsed and surfaced explicitly; prereg fields read defensively.
  1.1.1 (2026-05-29, Lyra): Raise SCRIPT_CHAR_CAP 14000->30000. v1.1.0's cap
        truncated a 16.6k-char script and the function-prioritized rebuild hid
        the implementation, producing false "not implemented" findings — a
        softer rerun of B4. qwen3:30b has 32k context; whole scripts up to 30k
        chars fit with room to spare.
  1.0.0 (prior): original single-experiment version.
"""

import json, os, sys, subprocess, textwrap, re
from pathlib import Path
from datetime import datetime, timezone

AGNI_REVIEW_VERSION = "1.2.0"
OLLAMA_MODEL = "qwen3:30b-a3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
NUM_PREDICT = 12000  # headroom so reasoning + verdict both fit (B1)
LOCAL_CHAR_BUDGET = 60000  # same default and env var as agni_phases.py; see the reasoning there

KNOWN_KILLS = [
    "C1: model.generate() rebuilds cache, invalidating geometry",
    "C2: Emotion directions computed on test data (circularity)",
    "C3: Different code paths for experimental vs control arms",
    "L1: Hook registration during generation alters behavior",
    "L3: No null control arm",
    "L4: hidden_states indexing off-by-one",
    "S1: No logit margin tracking (can't verify model uncertainty)",
    "S4: No FWL residualization on token count",
    "M2: No seed setting (non-reproducible)",
    "N_eff=1: Greedy decoding with multiple reps gives identical outputs",
    "Floor effect: Model too well-calibrated, always hedges",
    "Fabricated numbers: Reporting values not in actual data",
]

REVIEW_PROMPT = textwrap.dedent("""\
You are a hostile methodological reviewer for AI interpretability research.
Review this EXPERIMENT DESIGN (no results yet). Find every way it could mislead.

CONTEXT: W_K means the key projection weight matrix in transformer self-attention.
KV-cache geometry means SVD/spectral features of key-value cache entries.
This lab extracts geometric features from transformer caches to detect confabulation,
emotion, and other cognitive states. Prior work achieved AUROC 0.829-1.000.

KNOWN FATAL ERRORS from this lab's history:
{kills}

IMPORTANT: Only flag a known fatal error if you can point to the specific line or
construct in the SCRIPT below that exhibits it. Do NOT assume a failure is present
because the topic is similar — verify against the actual code. Flagging an error the
code does not contain is itself a review failure.

FULL EXPERIMENT SCRIPT:
{script}

HYPOTHESES:
{hypotheses}

SELF-IDENTIFIED THREATS:
{threats}

OTHER PRE-REGISTRATION FIELDS (every other key in the prereg, whole):
{other}

INSTRUCTIONS:
1. Check each known fatal error against this design. PASS or FAIL each, citing the
   specific line/construct. If the construct is absent, mark PASS (not applicable).
2. Identify NEW concerns not in the checklist.
3. For each concern: severity (CRITICAL/MAJOR/MINOR), attack, mitigation.
4. End with exactly one line: "OVERALL VERDICT: APPROVED" or "...: CONDITIONAL" or "...: REJECTED".
5. If CONDITIONAL, state what must change.
""")


_VERDICT_LINE = re.compile(
    r"^(?:(?:OVERALL|FINAL)\s+)?VERDICT\b[\s:=\u2013\u2014-]*(APPROVED|CONDITIONAL|REJECTED)(?![\w-])"
    r"(?!\s*(?:/|\||,|\bOR\b)\s*(?:APPROVED|CONDITIONAL|REJECTED)\b)", re.I)


def parse_verdict(review):
    """The verdict all of the review's verdict lines agree on, else UNPARSED. The rule of
    agni_phases.py, kept here because this harness runs on its own: a verdict line starts VERDICT,
    OVERALL VERDICT or FINAL VERDICT once markup is stripped, and a line in a code fence or indented
    as code is quoted code, not the reviewer. The old fallback to a verdict word anywhere read
    "I would not call this design approved" as APPROVED, exit 0 (round 2, 2026-11-07)."""
    found, fenced = set(), False
    for line in review.splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
            continue
        if fenced or re.match(r"^( {4}|\t)", line):
            continue
        m = _VERDICT_LINE.match(re.sub(r"[*_`#]+", "", line).strip())
        if m:
            found.add(m.group(1).upper())
    return found.pop() if len(found) == 1 else "UNPARSED"


def write_invalid(out_path, prereg, reason, prompt_length):
    """INVALID over whatever is at out_path, so an older APPROVED cannot stand for this run."""
    try:
        with open(out_path, "w") as f:
            json.dump({
                "agni_version": AGNI_REVIEW_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": OLLAMA_MODEL, "experiment": prereg.get("experiment"),
                "verdict": "INVALID", "review": "",
                "invalid_reason": reason,
                "prompt_length": prompt_length,
            }, f, indent=2)
    except OSError as e:
        print(f"ERROR: could not write the record to {out_path} ({e}). Anything already there is STALE.",
              file=sys.stderr)


def run_review(script_path, prereg_path):
    script = Path(script_path).read_text()
    with open(prereg_path) as f:
        prereg = json.load(f)

    # Every key reaches the model (round 2): a singular "hypothesis" fell back to [] and any other
    # key, falsification criteria included, was dropped without a word, and the run exited 0 APPROVED.
    hyp_key = next((k for k in ("hypotheses", "hypothesis", "primary_hypothesis") if prereg.get(k)), None)
    threat_key = next((k for k in ("threats_to_validity", "red_team_controls") if prereg.get(k)), None)
    hypotheses = json.dumps(prereg[hyp_key] if hyp_key else [], indent=2)
    threats = json.dumps(prereg[threat_key] if threat_key else [], indent=2)
    other_keys = [k for k in prereg if k not in (hyp_key, threat_key)]
    other = json.dumps({k: prereg[k] for k in other_keys}, indent=2) if other_keys else "(none)"
    kills = "\n".join(f"  - {k}" for k in KNOWN_KILLS)

    prompt = REVIEW_PROMPT.format(
        kills=kills, script=script, hypotheses=hypotheses, threats=threats, other=other)

    # B3 fix: name output per-experiment, not hardcoded
    exp = re.sub(r"[^A-Za-z0-9_.-]", "_", prereg.get("experiment", Path(script_path).stem))
    out_path = Path(script_path).parent / f"{exp}_agni_review.json"

    # No silent cuts: whole, or not at all. INVALID goes over any older record at this path.
    budget = int(os.environ.get("AGNI_LOCAL_CHAR_BUDGET", LOCAL_CHAR_BUDGET))
    if len(prompt) > budget:
        print("ERROR: REFUSED -- verdict INVALID, exit 2. No model saw this input.", file=sys.stderr)
        print("  This is NOT a pass. The design is UNGATED.", file=sys.stderr)
        print(f"  Prompt {len(prompt):,} chars, AGNI_LOCAL_CHAR_BUDGET {budget:,}: script {len(script):,}, "
              f"hypotheses {len(hypotheses):,}, threats {len(threats):,}.", file=sys.stderr)
        print("  What to do: pass a narrower script, or raise AGNI_LOCAL_CHAR_BUDGET and LLM_NUM_CTX "
              "together, deliberately.", file=sys.stderr)
        write_invalid(out_path, prereg, f"prompt {len(prompt)} chars over budget {budget}", len(prompt))
        sys.exit(2)

    print(f"Agni Design Review v{AGNI_REVIEW_VERSION}")
    print(f"Submitting to {OLLAMA_MODEL}...")
    print(f"  Script: {script_path} ({len(script)} chars sent, whole)")
    print(f"  Experiment: {prereg.get('experiment','?')}")
    print()

    # Before the call, as in agni_phases.py: a run that dies, fails in curl or gets no review then
    # leaves INVALID, not an older APPROVED (round 2).
    write_invalid(out_path, prereg, "a review started and no result was recorded: it is still running, "
                  "or it died", len(prompt))
    payload = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"num_predict": NUM_PREDICT, "temperature": 0.3,
                    "num_ctx": int(os.environ.get("LLM_NUM_CTX", "32768"))},
    })
    result = subprocess.run(["curl", "-s", OLLAMA_URL, "--data-binary", "@-"], input=payload,
                            capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"ERROR: curl failed: {result.stderr}", file=sys.stderr)
        sys.exit(3)

    try:
        resp = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("ERROR: could not parse Ollama response", file=sys.stderr)
        print(result.stdout[:500], file=sys.stderr)
        sys.exit(3)

    # B1 fix: reasoning models put output in `thinking` when `response` is empty
    review = (resp.get("response") or "").strip()
    thinking = (resp.get("thinking") or "").strip()
    if not review and thinking:
        review = "[recovered from thinking field]\n\n" + thinking

    # B2 fix: a gate must FAIL LOUD on empty — never let silence read as pass
    if not review:
        print("ERROR: Agni returned an EMPTY review. This is NOT a pass.", file=sys.stderr)
        print(f"  done_reason={resp.get('done_reason')} eval_count={resp.get('eval_count')}", file=sys.stderr)
        print("  Raise NUM_PREDICT or check the model. Gate result is INVALID.", file=sys.stderr)
        write_invalid(out_path, prereg, "the model gave an empty review", len(prompt))
        sys.exit(2)

    verdict = parse_verdict(review)
    print("=" * 60)
    print(f"AGNI DESIGN REVIEW — VERDICT: {verdict}")
    print("=" * 60)
    print(review)
    print("=" * 60)
    print(f"VERDICT: {verdict}")

    with open(out_path, "w") as f:
        json.dump({
            "agni_version": AGNI_REVIEW_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": OLLAMA_MODEL, "experiment": prereg.get("experiment"),
            "verdict": verdict, "review": review,
            "recovered_from_thinking": bool(not resp.get("response") and thinking),
            "prompt_length": len(prompt),
            "prereg_keys_sent": sorted(prereg),
        }, f, indent=2)
    print(f"Saved to {out_path}")
    return verdict


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 agni_design_review.py <script.py> <prereg.json>")
        sys.exit(1)
    v = run_review(sys.argv[1], sys.argv[2])
    # B2: non-zero exit on REJECTED/UNPARSED so callers can gate on exit code
    sys.exit(0 if v in ("APPROVED", "CONDITIONAL") else 4)
