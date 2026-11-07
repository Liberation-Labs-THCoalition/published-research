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
        Round 3: fences are read as CommonMark reads them, and a verdict line in code can veto but
        never decide (see parse_verdict). A prereg that is missing, unreadable, not UTF-8, not JSON,
        not an object or has a duplicate key, a script that is not UTF-8, and a curl or backend
        failure each write INVALID with its cause over any older record and exit non-zero; they
        used to raise a traceback (exit 1, which reads as REJECTED) and leave that record standing.
        Files are read as UTF-8, not in the platform's encoding.
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

import json, os, sys, subprocess, textwrap, re, unicodedata
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


_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_LIST_ITEM = re.compile(r"^( {0,3})([-+*]|\d{1,9}[.)])( +|$)")
_THEMATIC_BREAK = re.compile(r"^ {0,3}([-*_])(?: *\1){2,} *$")
# CommonMark HTML blocks. Kinds 1 to 5 (<pre>, <script>, <style>, <textarea>, comments, <?, <!X,
# CDATA) run to their end marker; any other tag runs to a blank line (end None).
_HTML_BLOCKS = (
    (re.compile(r"^ {0,3}<(?:pre|script|style|textarea)(?=\s|>|$)", re.I),
     re.compile(r"</(?:pre|script|style|textarea)>", re.I)),
    (re.compile(r"^ {0,3}<!--"), re.compile(r"-->")),
    (re.compile(r"^ {0,3}<\?"), re.compile(r"\?>")),
    (re.compile(r"^ {0,3}<!\[CDATA\["), re.compile(r"\]\]>")),
    (re.compile(r"^ {0,3}<![A-Za-z]"), re.compile(r">")),
    (re.compile(r"^ {0,3}</?[A-Za-z][A-Za-z0-9-]*(?=[\s/>]|$)"), None),
)
# CommonMark ends a line at LF, CR or CRLF only. str.splitlines also ends one at U+2028, NEL, a
# form feed and others, so "a<U+2028>```" closed a fence CommonMark keeps open (round 4).
_LINE_END = re.compile(r"\r\n|\r|\n")
_VERDICT_STEM = re.compile(r"approved|conditional|rejected")
# Cyrillic and Greek letters that look Latin, as lower case, read as the Latin letter.
_LOOKALIKES = str.maketrans(
    "\u0430\u0432\u0435\u043a\u043c\u043d\u043e\u0440\u0441\u0442\u0443\u0445\u0456\u0458\u0455\u0501"
    "\u03b1\u03b2\u03b5\u03b6\u03b7\u03b9\u03ba\u03bc\u03bd\u03bf\u03c1\u03c4\u03c5\u03c7",
    "abekmhopctyxijsd" "abezhikmnoptyx")
# Between the words of the contract read back: punctuation and "or", nothing else.
_ECHO_GAP = re.compile(r"^(?:[\s\"'/|,.:;()\[\]*_`~-]|\bor\b)*$")
# A word that says no, between "verdict" and the verdict it names. "no" only when nothing follows it.
_NEGATION = re.compile(r"\b(?:not|never|nor)\b|n't\b|\bno\b(?!\s+\w)", re.I)
# "verdict", at most two words, then a verdict word, on folded (lower-case) text.
_NAMED_VERDICT = re.compile(r"verdict\W*((?:\w+\W+){0,2}?)(approved|conditional|rejected)")
# After a pass verdict, what takes it back: a negation, a contrast, a question or an arrow.
_TAKE_BACK = re.compile(r"\b(?:not|never|nor|but|however|instead|rather|except|although|though)\b"
                        r"|n't\b|\bno\b(?!\s+\w)|\?|->|=>|<-|[\u2190-\u21ff\u27f0-\u27ff\u2900-\u297f]",
                        re.I)
# After APPROVED, what makes it depend on something, which is not APPROVED.
_CONDITION = re.compile(r"\b(?:if|unless|otherwise|provided|pending|until|once|subject\s+to|"
                        r"condition\w*)\b", re.I)


def _spaces(line):
    return len(line) - len(line.lstrip(" "))


def _code_lines(review):
    """Each line of the review as (line, in code), and whether a fence or an HTML block that runs to
    an end marker was left open.

    CommonMark, as far as a verdict needs it. Lines end at LF, CR or CRLF only. A fence opens on up
    to three spaces of indent, then three or more backticks or tildes; after backticks the rest of
    the line may hold no backtick, so ```x``` is a span, not a fence. Only the same character, at
    least as long, with nothing after it, closes it. Four spaces or a tab of indent is code, counted
    from the content of the list item the line continues. A list marker is consumed before the rest
    of its line is read, so "- ```" opens a fence and "-     x" is indented code; "- - -" is a
    thematic break, not a list. HTML blocks count as code: a <pre> or a comment holds lines a reader
    does not see as text. Where this reading knowingly differs from CommonMark, it keeps more in
    code: a closing fence must sit at or right of its opening one, a line left of the list item that
    holds a fence leaves the fence open where CommonMark would end both, and any line that starts
    with a tag starts an HTML block."""
    lines, fence, html, item = [], None, None, 0
    for raw in _LINE_END.split(review):
        line = raw.expandtabs(4)
        indent, blank = _spaces(line), not line.strip()
        if fence:
            char, length, opened_at, container = fence
            if (opened_at <= indent <= container + 3
                    and re.match(r"^ *%s{%d,}[ \t]*$" % (re.escape(char), length), line)):
                fence = None
            else:
                lines.append((raw, True))
            continue
        if html:
            lines.append((raw, True))
            if html[0].search(line) if html[0] else blank:
                html = None
            continue
        if not blank and indent < item:
            item = 0
        pos, in_code = item, False
        while not blank:
            rest = line[pos:]
            if _spaces(rest) >= 4:
                in_code = True
                break
            m = _FENCE_OPEN.match(rest)
            if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                fence = (m.group(1)[0], len(m.group(1)), pos + _spaces(rest), pos)
                in_code = True
                break
            start, end = next(((s.match(rest), e) for s, e in _HTML_BLOCKS if s.match(rest)), (None, None))
            if start:
                in_code = True
                if not (end and end.search(rest, start.end())):
                    html = (end,)
                break
            if _THEMATIC_BREAK.match(rest):
                break
            m = _LIST_ITEM.match(rest)
            if not m:
                break
            gap = len(m.group(3))
            pos += len(m.group(1)) + len(m.group(2)) + (gap if gap <= 4 and rest[m.end():].strip() else 1)
            item = pos
        lines.append((raw, in_code))
    return lines, fence is not None or bool(html and html[0])


def _fold(line):
    """Lower case, with compatibility forms and accents taken apart, invisible format characters (a
    zero-width space) dropped, and Cyrillic and Greek lookalikes read as Latin."""
    s = "".join(c for c in unicodedata.normalize("NFKD", line) if unicodedata.category(c) not in ("Cf", "Mn"))
    return s.lower().translate(_LOOKALIKES)


def _is_echo(folded):
    """The contract read back, "APPROVED / CONDITIONAL / REJECTED": all three words with nothing
    but punctuation and "or" between them. It gives no verdict, so it neither decides nor vetoes;
    otherwise a reviewer that restates its instructions could never parse."""
    hits = list(_VERDICT_STEM.finditer(folded))
    return ({h.group() for h in hits} == {"approved", "conditional", "rejected"}
            and all(_ECHO_GAP.match(folded[a.end():b.start()]) for a, b in zip(hits, hits[1:])))


def _named_verdicts(folded):
    """The verdict words a line names as a verdict, and whether one is negated ("Verdict: not
    APPROVED"). Named: "verdict", then at most two words, then the word ("| Verdict | REJECTED |",
    "**Verdict:** <emoji> REJECTED", "the verdict would be REJECTED"), or a line whose letters are one
    verdict word and nothing else ("| REJECTED |"). Farther apart is prose, not a verdict: replayed
    over 80 stored reviews, "the registered verdict is CONFOUND_SUSPECTED, ... rejected" turned a
    real CONDITIONAL into UNPARSED when any word on the line counted."""
    letters = re.sub(r"[^a-z]+", "", folded)
    if letters in ("approved", "conditional", "rejected"):
        return {letters.upper()}, False
    hits = [(m.group(2).upper(), bool(_NEGATION.search(m.group(1)))) for m in _NAMED_VERDICT.finditer(folded)]
    return {w for w, _ in hits}, any(negated for _, negated in hits)


def _takes_back(verdict, tail):
    """Whether the text after a pass verdict's word negates it, or makes APPROVED conditional. A
    REJECTED is not checked: however it is qualified, it cannot pass."""
    if verdict == "REJECTED":
        return False
    return bool(_TAKE_BACK.search(tail) or (verdict == "APPROVED" and _CONDITION.search(tail)))


def _verdicts(review, decide):
    """(verdict, decides) for each verdict the review's lines name, and whether a fence or an HTML
    block was left open. decide(text) matches a verdict line once markup is stripped.

    Round 4: a line is read whole. Only a verdict line outside code decides, and only if nothing
    after its word takes it back; one that is taken back lists UNCLEAR. A verdict line lists every
    verdict word on it as not deciding, and any other line, in code or not, in a form decide() knows
    or not, lists the verdict words it names (see _named_verdicts), so a word that differs from the
    decided verdict vetoes. A line outside code that names a verdict and negates it lists UNCLEAR.
    UNCLEAR is never decided, so it vetoes anything. The contract read back is skipped."""
    lines, unclosed = _code_lines(review)
    found = []
    for line, in_code in lines:
        folded = _fold(line)
        if _is_echo(folded):
            continue
        words, negated = _named_verdicts(folded)
        text = re.sub(r"[*_`#]+", "", line).strip()
        m = None if in_code else decide(text)
        if m:
            verdict = m.group(1).upper()
            found.append(("UNCLEAR", False) if _takes_back(verdict, text[m.end():]) else (verdict, True))
            words = {w.upper() for w in _VERDICT_STEM.findall(folded)}
        elif negated and not in_code:
            found.append(("UNCLEAR", False))
        found += [(w, False) for w in sorted(words)]
    return found, unclosed


def parse_verdict(review):
    """The verdict all of the review's verdict lines agree on, else UNPARSED. The rule of
    agni_phases.py, kept here because this harness runs on its own: a verdict line starts VERDICT,
    OVERALL VERDICT or FINAL VERDICT once markup is stripped, and a line in a code fence or indented
    as code is quoted code, not the reviewer. The old fallback to a verdict word anywhere read
    "I would not call this design approved" as APPROVED, exit 0 (round 2, 2026-11-07).
    Round 3 (2026-11-07): any line starting with three backticks or tildes flipped "in code", so a
    one-line ```span``` or a fence left open hid a REJECTED and let a quoted APPROVED decide. Now
    fences follow CommonMark; only a line outside code decides; a verdict line in code that disagrees,
    a fence that never closes, or verdict lines only in code give UNPARSED.
    Round 4 (2026-11-07): a line is read whole, as in agni_phases.py. A second verdict word on a
    verdict line, a REJECTED in a form the regex does not know (a table cell, after an emoji, a
    lookalike letter, a zero-width space), or a pass verdict that is negated or made conditional
    gives UNPARSED."""
    found, unclosed = _verdicts(review, _VERDICT_LINE.match)
    decided = {v for v, decides in found if decides}
    if unclosed or len(decided) != 1 or any(v not in decided for v, _ in found):
        return "UNPARSED"
    return decided.pop()


def write_invalid(out_path, prereg, reason, prompt_length):
    """INVALID over whatever is at out_path, so an older APPROVED cannot stand for this run."""
    try:
        with open(out_path, "w", encoding="utf-8") as f:
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


def _record_path(script_path, experiment):
    exp = re.sub(r"[^A-Za-z0-9_.-]", "_", experiment if isinstance(experiment, str) else Path(script_path).stem)
    return Path(script_path).parent / f"{exp}_agni_review.json"


def fail_invalid(paths, prereg, reason, prompt_length=0, code=2):
    """Exit non-zero with INVALID written at every path an older record of this run could be at."""
    print(f"ERROR: {reason}", file=sys.stderr)
    print("  Verdict INVALID. This is NOT a pass. The design is UNGATED.", file=sys.stderr)
    for p in dict.fromkeys(paths):
        write_invalid(p, prereg, reason, prompt_length)
    sys.exit(code)


def _no_duplicate_keys(pairs):
    """json.loads keeps the last of two equal keys and drops the first without a word."""
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}: one of its values would never reach the model")
        seen.add(key)
    return dict(pairs)


def _read_utf8(path):
    """UTF-8, strictly. The platform encoding (cp1252 on Windows) used to turn an arrow into three
    other glyphs without a word, or crash with a traceback that left an older record standing."""
    return Path(path).read_bytes().decode("utf-8")


def read_prereg(script_path, prereg_path):
    """The prereg as a dict, or INVALID and exit 2. A prereg that cannot be read cannot name its
    experiment, so INVALID goes at the script-named path and at any path its text still names (round 3)."""
    paths = [_record_path(script_path, None)]
    try:
        raw = Path(prereg_path).read_bytes()
    except OSError as e:
        fail_invalid(paths, {}, f"the prereg {prereg_path} cannot be read: {e.__class__.__name__}: {e}")
    paths += [_record_path(script_path, m) for m in
              re.findall(r'"experiment"\s*:\s*"((?:[^"\\]|\\.)*)"', raw.decode("utf-8", errors="replace"))]
    try:
        prereg = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, ValueError, RecursionError) as e:   # deep nesting: round 4
        fail_invalid(paths, {}, f"the prereg {prereg_path} is not valid UTF-8 JSON: {e}")
    if not isinstance(prereg, dict):
        fail_invalid(paths, {}, f"the prereg {prereg_path} is not a JSON object")
    return prereg


def run_review(script_path, prereg_path):
    prereg = read_prereg(script_path, prereg_path)
    # B3 fix: name output per-experiment, not hardcoded
    out_path = _record_path(script_path, prereg.get("experiment"))
    try:
        script = _read_utf8(script_path)
    except (OSError, UnicodeDecodeError) as e:
        fail_invalid([out_path], prereg, f"the script {script_path} cannot be read as UTF-8: "
                     f"{e.__class__.__name__}: {e}")

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

    # No silent cuts: whole, or not at all. INVALID goes over any older record at this path.
    # Both read before anything is sent, and a value that is not a whole number is INVALID, not a
    # traceback (round 4): int("60k") raised here, exit 1, before the placeholder, so an older
    # APPROVED stood; int("32k") for num_ctx raised after it, exit 1.
    try:
        budget = int(os.environ.get("AGNI_LOCAL_CHAR_BUDGET", LOCAL_CHAR_BUDGET))
        num_ctx = int(os.environ.get("LLM_NUM_CTX", "32768"))
    except ValueError as e:
        fail_invalid([out_path], prereg, "AGNI_LOCAL_CHAR_BUDGET and LLM_NUM_CTX must be whole "
                     f"numbers: {e}", len(prompt))
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
                    "num_ctx": num_ctx},
    })
    # Each failure writes its own INVALID with its cause, exit 3 (round 3): the placeholder above
    # says only that a review started, and a traceback here used to exit 1, which reads as REJECTED.
    try:
        result = subprocess.run(["curl", "-s", OLLAMA_URL, "--data-binary", "@-"], input=payload,
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        fail_invalid([out_path], prereg, f"curl did not run: {e.__class__.__name__}: {e}", len(prompt), 3)
    if result.returncode != 0:
        fail_invalid([out_path], prereg, f"curl failed, exit {result.returncode}: "
                     f"{(result.stderr or '').strip()[:500] or '(no stderr)'}", len(prompt), 3)
    try:
        resp = json.loads(result.stdout)
    except ValueError:
        resp = None
    if not isinstance(resp, dict):
        fail_invalid([out_path], prereg, "the backend reply is not a JSON object: "
                     f"{(result.stdout or '').strip()[:500] or '(empty)'}", len(prompt), 3)

    # B1 fix: reasoning models put output in `thinking` when `response` is empty
    review = resp.get("response") if isinstance(resp.get("response"), str) else ""
    thinking = resp.get("thinking") if isinstance(resp.get("thinking"), str) else ""
    review, thinking = review.strip(), thinking.strip()
    recovered = bool(not review and thinking)
    if recovered:
        review = "[recovered from thinking field]\n\n" + thinking

    # B2 fix: a gate must FAIL LOUD on empty — never let silence read as pass
    if not review:
        print("ERROR: Agni returned an EMPTY review. This is NOT a pass.", file=sys.stderr)
        print(f"  done_reason={resp.get('done_reason')} eval_count={resp.get('eval_count')}", file=sys.stderr)
        print("  Raise NUM_PREDICT or check the model. Gate result is INVALID.", file=sys.stderr)
        write_invalid(out_path, prereg, "the model gave an empty review"
                      + (f"; backend error: {resp['error']}" if resp.get("error") else ""), len(prompt))
        sys.exit(2)

    verdict = parse_verdict(review)
    # Round 4: thinking is not the answer. A model that ran out of tokens mid-thought, having
    # drafted the contract line there, recorded APPROVED and exited 0. A pass read only from the
    # thinking is UNPARSED; a rejection read there is kept, since it cannot pass.
    if recovered and verdict in ("APPROVED", "CONDITIONAL"):
        print(f"WARNING: {verdict} was read only from the model's thinking (done_reason="
              f"{resp.get('done_reason')}), not from an answer. Recorded UNPARSED: NOT a pass.",
              file=sys.stderr)
        verdict = "UNPARSED"
    print("=" * 60)
    print(f"AGNI DESIGN REVIEW — VERDICT: {verdict}")
    print("=" * 60)
    print(review)
    print("=" * 60)
    print(f"VERDICT: {verdict}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "agni_version": AGNI_REVIEW_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": OLLAMA_MODEL, "experiment": prereg.get("experiment"),
            "verdict": verdict, "review": review,
            "recovered_from_thinking": recovered, "done_reason": resp.get("done_reason"),
            "prompt_length": len(prompt),
            "prereg_keys_sent": sorted(prereg),
            "prereg_other_keys_sent": sorted(other_keys),
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
