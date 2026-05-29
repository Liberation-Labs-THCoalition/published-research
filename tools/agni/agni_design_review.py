#!/usr/bin/env python3
"""
Agni Design Review — Pre-run red-team of experiment methodology.
Submits experiment script + pre-registration to a local model for hostile review.

Usage:
  python3 agni_design_review.py <script.py> <prereg.json>
  ssh margaret "cd ~/experiments && python3 agni_design_review.py script.py prereg.json"

CHANGELOG
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

import json, sys, subprocess, textwrap, re
from pathlib import Path
from datetime import datetime, timezone

AGNI_REVIEW_VERSION = "1.1.1"
OLLAMA_MODEL = "qwen3:30b-a3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
NUM_PREDICT = 12000  # headroom so reasoning + verdict both fit (B1)
SCRIPT_CHAR_CAP = 30000  # fit whole experiment scripts; 32k ctx leaves room (B4-residual)

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
{script_excerpt}

HYPOTHESES:
{hypotheses}

SELF-IDENTIFIED THREATS:
{threats}

INSTRUCTIONS:
1. Check each known fatal error against this design. PASS or FAIL each, citing the
   specific line/construct. If the construct is absent, mark PASS (not applicable).
2. Identify NEW concerns not in the checklist.
3. For each concern: severity (CRITICAL/MAJOR/MINOR), attack, mitigation.
4. End with exactly one line: "OVERALL VERDICT: APPROVED" or "...: CONDITIONAL" or "...: REJECTED".
5. If CONDITIONAL, state what must change.
""")


def extract_script(script_path):
    """Send the full script. If it exceeds the cap, keep the docstring plus all
    function definitions (the methodology), which is what a reviewer needs to see.
    (B4 fix: previously used decision_moment-specific markers and fell back to
    text[:3000] for any other script, hiding the actual methodology.)"""
    text = Path(script_path).read_text()
    if len(text) <= SCRIPT_CHAR_CAP:
        return text
    lines = text.split("\n")
    kept, in_func, blank = [], False, 0
    # keep module docstring + every def/class block
    for i, line in enumerate(lines):
        if re.match(r"^\s*(def |class )", line):
            in_func = True
            blank = 0
        if i < 40 or in_func:
            kept.append(line)
        if in_func and line.strip() == "":
            blank += 1
            if blank >= 2:
                in_func = False
    out = "\n".join(kept)
    return out[:SCRIPT_CHAR_CAP] + "\n# [truncated — function-prioritized]"


def parse_verdict(review):
    m = re.search(r"OVERALL VERDICT:\s*(APPROVED|CONDITIONAL|REJECTED)", review, re.I)
    if m:
        return m.group(1).upper()
    # fallback: last standalone verdict word
    for word in ("REJECTED", "CONDITIONAL", "APPROVED"):
        if re.search(rf"\b{word}\b", review, re.I):
            return word
    return "UNPARSED"


def run_review(script_path, prereg_path):
    script_excerpt = extract_script(script_path)
    with open(prereg_path) as f:
        prereg = json.load(f)

    hypotheses = json.dumps(prereg.get("hypotheses", prereg.get("primary_hypothesis", [])), indent=2)
    threats = json.dumps(prereg.get("threats_to_validity", prereg.get("red_team_controls", [])), indent=2)
    kills = "\n".join(f"  - {k}" for k in KNOWN_KILLS)

    prompt = REVIEW_PROMPT.format(
        kills=kills, script_excerpt=script_excerpt[:SCRIPT_CHAR_CAP],
        hypotheses=hypotheses[:2500], threats=threats[:2000])

    print(f"Agni Design Review v{AGNI_REVIEW_VERSION}")
    print(f"Submitting to {OLLAMA_MODEL}...")
    print(f"  Script: {script_path} ({len(script_excerpt)} chars sent)")
    print(f"  Experiment: {prereg.get('experiment','?')}")
    print()

    payload = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"num_predict": NUM_PREDICT, "temperature": 0.3},
    })
    result = subprocess.run(["curl", "-s", OLLAMA_URL, "-d", payload],
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
        sys.exit(2)

    verdict = parse_verdict(review)
    print("=" * 60)
    print(f"AGNI DESIGN REVIEW — VERDICT: {verdict}")
    print("=" * 60)
    print(review)
    print("=" * 60)
    print(f"VERDICT: {verdict}")

    # B3 fix: name output per-experiment, not hardcoded
    exp = re.sub(r"[^A-Za-z0-9_.-]", "_", prereg.get("experiment", Path(script_path).stem))
    out_path = Path(script_path).parent / f"{exp}_agni_review.json"
    with open(out_path, "w") as f:
        json.dump({
            "agni_version": AGNI_REVIEW_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": OLLAMA_MODEL, "experiment": prereg.get("experiment"),
            "verdict": verdict, "review": review,
            "recovered_from_thinking": bool(not resp.get("response") and thinking),
            "prompt_length": len(prompt),
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
