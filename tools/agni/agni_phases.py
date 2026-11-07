#!/usr/bin/env python3
"""
Agni Lifecycle Review — gate every phase, not just design.

WHY THIS EXISTS
  agni_design_review.py / agni_review_claude.py review an EXPERIMENT DESIGN:
  they take (script.py, prereg.json) and ask "will this waste GPU hours?"

  But KNOWN_KILLS has outgrown that harness. Four entries --
  UNRESOLVED_REVIEW_FLAG, CLAIMED_FIX_NOT_IMPLEMENTED, PROSE_ONLY_CLAIM,
  BIB_FABRICATION -- are PAPER defects. Nothing runs them. Feeding a finished
  paper to the design harness reproduces documented bug B4: the reviewer gets
  the wrong content type and hallucinates findings. (Confirmed 2026-08-13 --
  the gate was declined for exactly this reason and the review done by hand.)

  This adds the missing phases. Same kill list, same verdict contract, phase-
  appropriate prompts and inputs.

PHASES
  design     script + prereg      -> will this design mislead?      [pre-compute]
  preflight  script + prereg      -> does the CODE do what the      [pre-compute]
                                     prereg SAYS?
  results    artifact + data      -> does every number in the       [post-compute]
                                     artifact appear in the data?
  paper      paper + data + code  -> claims vs evidence; label ->   [pre-submit]
                                     dependent statistic; architecture
  style      paper(s)             -> style guide, twin sync, bib,   [pre-submit]
                                     unresolved review flags

VERDICT CONTRACT (round 5): the reply's LAST line, alone, is "AGNI-VERDICT: <word>", where <word> is
APPROVED, CONDITIONAL or REJECTED, and that line decides. It must be outside code and, once markdown
markup (* _ ` #) and whitespace are stripped, exactly that, with nothing after it. Every other
verdict line outside code must agree with it: a line that is exactly one of the words, one that
starts VERDICT, OVERALL VERDICT or FINAL VERDICT followed by one of them (trailing text allowed,
unless after a pass it negates, contrasts, asks or points elsewhere, or after APPROVED sets a
condition: then UNPARSED), or another AGNI-VERDICT line. Prose never decides or vetoes. A line in a
code fence, an indented code block or an HTML block is quoted code, not the reviewer speaking: it
never decides or vetoes either.
APPROVED and CONDITIONAL exit 0, REJECTED exit 1. But APPROVED or CONDITIONAL on input the reviewer
did not get whole (input_complete false) exits 2 and is NOT a pass, unless --accept-incomplete-input
says to take a review of what it got; the record then carries incomplete_input_accepted.

The gate adds two verdicts of its own. Both exit 2 and neither is a pass:
  UNPARSED  no AGNI-VERDICT line last, or a verdict line that disagrees with it.
  INVALID   the reviewer did not get all of its input, or gave nothing back: a named input missing,
            unreadable or empty, a binary primary, a prompt over budget, or an empty reply. Also
            the record's value while a review is running, so an old APPROVED at the same path
            cannot outlive a run that dies.
A phase that returns empty FAILS LOUD (exit 2). A gate must never let
"no output" read as "no findings."

NO SILENT CUTS (v1.1.0): a review may never look complete when the reviewer did not get all of
its input. No input is sliced. The whole prompt is checked once against a budget, and over budget
means INVALID before any reviewer call. The record's input_coverage says what the reviewer got.
See README.md.

Usage
  python3 agni_phases.py design    <script.py> <prereg.json>
  python3 agni_phases.py preflight <script.py> <prereg.json>
  python3 agni_phases.py results   <artifact.md|tex> <data_dir>
  python3 agni_phases.py paper     <paper.md|tex> [data_dir] [code_dir]
  python3 agni_phases.py style     <paper.md|tex> [twin.md|tex]

  --model MODEL   override (default: session default in llm.py)
  --out PATH      where to write the JSON record
  --accept-incomplete-input
                  exit 0 on APPROVED/CONDITIONAL although input_complete is false (binaries in a
                  data folder, undecodable bytes); the record is stamped

  AGNI_PROMPT_CHAR_BUDGET  whole-prompt budget in chars (default 400000)
  AGNI_LOCAL_CHAR_BUDGET   budget for anything bound for Ollama, in chars (default 60000)
  AGNI_CLI_TIMEOUT         seconds per CLI reviewer call (default 1800)
"""

import argparse
import sys as _sys
# Windows consoles default to cp1252; review text carries checkmarks, arrows
# and em-dashes. Without this a rendering failure aborts the whole run.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path.home() / "lab/projects/agni-open"))
import llm  # noqa: E402
# Every CLI reviewer call goes through this; see claude_isolation.py and README.md.
from claude_isolation import isolated_claude_cmd  # noqa: E402

AGNI_PHASES_VERSION = "1.1.0"
DEFAULT_MODEL = "opus"  # CLI alias -> claude-opus-5 (verified 2026-08-30).
# Was "claude-opus-4-6", a stale literal the CLI would now reject outright.
# Whole-prompt budget (2026-11-07), replacing the per-input caps that cut silently. It counts
# everything the reviewer is sent, system text included. 4 chars per token was assumed here, and is
# wrong: measured on claude-haiku-5-5 it is about 2.45 (a filler prompt) to 2.48 (prose with
# numbers), so 400000 chars is about 160k tokens, not 100k. Haiku took that without a cut; the rate
# for the default model (opus) has not been measured. Past that the honest move is narrower input.
PROMPT_CHAR_BUDGET = 400000
# Ollama trims a prompt to its context length and says nothing, so nothing here can see the cut
# afterwards. 60000 chars fits llm.py's default 32768-token context with room for the answer, down
# to about 2.5 chars per token. It is a proxy: tokens are not measured here.
LOCAL_CHAR_BUDGET = 60000
# A file this many times over budget cannot fit whatever it holds (UTF-8 is at most 4 bytes a char).
MAX_BYTES_PER_CHAR = 4
TIMEOUTS_ALLOWED = 2    # one retry: see the retry loop in main()
LARGEST_SHOWN = 10

SNIFF_BYTES = 8192
BINARY_REPLACEMENT_RATIO = 0.01
JUNK_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", ".ipynb_checkpoints"})
SKIP_JUNK = "junk folder"
SKIP_LINK_INSIDE = "link to a path inside this tree, not followed"
SKIP_LINK_INTO_JUNK = "link to a path inside a junk folder of this tree, not followed or read"
SKIP_LINK_OUTSIDE = "link to outside this tree, not followed"
SKIP_LINK_BROKEN = "broken link"
SKIP_SPECIAL = "not a regular file"
# Skips that cost the reviewer nothing: junk is not input, and an inner link's target is shown
# under its own name. Anything else skipped means the input is not complete. A link into a junk
# folder is not harmless: nothing there is read, so its target is shown nowhere (round 4).
HARMLESS_SKIPS = (SKIP_JUNK, SKIP_LINK_INSIDE)
PREREG_PHASES = ("design", "preflight")

SYSTEM = ("You are Project Agni, a hostile but fair methodological reviewer for "
          "Liberation Labs. You find fatal flaws before they reach a venue. You "
          "would rather kill a good finding than ship a bad one.")

# ---------------------------------------------------------------- kill lists

DESIGN_KILLS = [
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
    "Hook ordering: Observation hook fires before injection hook on same module",
    "Constant additive shift: SVD Vt[1:] is blind to mean shifts from constant injection",
    "UNFALSIFIABLE: every outcome branch confirms the headline claim. Both-outcomes-"
    "informative is fine for THEORY; it is fatal when the PRODUCT claim survives "
    "every branch. Ask: what result would make the title wrong?",
    "CONTROL WITHOUT A BRANCH: a control arm is specified but the discussion has no "
    "branch for the control ALSO firing. If the control reproduces the effect, the "
    "effect is an artifact -- that outcome must be written down in advance.",
    "PREREG NOT LOCATABLE: paper claims 'we pre-registered' but the predictions live "
    "in another file the paper never cites, or are a placeholder.",
    "DIRECTIONAL WITHOUT THRESHOLD: 'A > B' with no effect size, no alpha, no stopping "
    "rule. Cannot fail.",
]

RESULTS_KILLS = [
    "Fabricated numbers: reporting values not in the actual data",
    "Rounding in the self-favorable direction",
    "Estimator mismatch: comparing a point estimate against a bootstrap mean",
    "Denominator drift: prose excludes a case that is still in the numerator",
    "Silent N change between the prereg and the artifact",
    "ddof convention unstated where it is load-bearing (sample vs population SD)",
    "Impossible table: stated percentages admit no integer count vector at the stated N",
    "Censoring undisclosed: values at a cap/ceiling reported as if uncensored",
]

PAPER_KILLS = [
    "UNRESOLVED_REVIEW_FLAG: paper ships with its own Agni-gate or REVIEW_INDEX flags "
    "still open while asserting flagged claims as established",
    "CLAIMED_FIX_NOT_IMPLEMENTED: paper claims a methodological improvement but the "
    "shipped code does not implement it",
    "CLAIMED_REVIEW_NOT_RUN: paper asserts it was adversarially reviewed / verified when "
    "no such review artifact exists for THIS paper",
    "PROSE_ONLY_CLAIM: claim added in a commit that changed only prose, no data or code "
    "artifact shipped",
    "BIB_FABRICATION: citation metadata wrong, or a number attributed to a paper that "
    "does not contain it",
    "LABEL WITHOUT ITS STATISTIC: a label/definition/count/scope qualifier is corrected "
    "but the statistic that DEPENDS on it is not. THIS IS THE HIGHEST-YIELD CHECK. "
    "For every label claim, ask: which reported number changes if this label is wrong?",
    "RETRACTED UPSTREAM: a cited number was retracted at its source. Correcting a "
    "finding at source is exactly when it stops being audited downstream.",
    "DOUBLE-COUNTED EXPERIMENT: the same experiment appears twice under different "
    "labels, often on opposite sides of a ledger (pre- and post-correction).",
    "OUTCOME-INFORMED TAXONOMY: categories assigned after outcomes were known, by an "
    "interested party, and the split is the outcome the taxonomy was built to describe.",
    "ARCHITECTURE VS CONFIG: layer-structure claims not checked against the shipped "
    "config. Qwen3.5-27B is full_attention_interval=4, full attention at [3,7,...,63] "
    "-- an INTERLEAVE, 16 full + 48 GatedDeltaNet, d_model=5120, NO boundary at L16. "
    "Qwen3-8B is dense, 36 layers, d_model=4096.",
]

STYLE_KILLS = [
    "TWIN_DESYNC: flight and academic editions diverge in a claim, a number, or the "
    "SECTION a table row is filed under. A string grep cannot see position.",
    "STALE_PDF: rendered artifact older than its source",
    "FABRICATED_AUTHOR_NAMES",
    "Byline vs contribution mismatch: named authors who have not signed off",
    "Orphaned bib entries, or cited keys with no entry",
    "Dead URLs in the reference list",
    "Correction made silently instead of visibly (no note stating what was wrong)",
    "Footnote or cross-reference pointing at a position ('line 4 of this table') that "
    "rots when rows move -- must name the row, not its index",
]

# ---------------------------------------------------------------- prompts

COMMON_TAIL = """
KNOWN KILLS -- lab-specific failures that have actually happened here. Check each:
{kills}

Return, in this order:
1. VERDICT on its own line: exactly one of APPROVED / CONDITIONAL / REJECTED
2. Numbered findings, each with: severity (CRITICAL/MAJOR/MINOR), the exact quote or
   location, why it is wrong, and the concrete fix.
3. What you could NOT check, and what you would need to check it.

Be specific. "Consider adding controls" is useless; name the control and the arm.
If you cannot verify something, say so plainly rather than assuming it is fine.

Make the LAST line of your reply, alone, exactly "AGNI-VERDICT: <word>", with the same word
as in item 1, and nothing after it.
"""

PROMPTS = {
    "design": """You are Project Agni reviewing an EXPERIMENT DESIGN before any compute is spent.
No results exist yet. Find every way this design could mislead.

CONTEXT: W_K is the key projection weight matrix in transformer self-attention.
FWL = Frisch-Waugh-Lovell residualization (mandatory here for token-count confounds).

SCRIPT:
{primary}

PRE-REGISTERED HYPOTHESES:
{secondary}

THREATS TO VALIDITY the authors already identified:
{tertiary}

Pay special attention to whether the falsification criteria can actually FAIL.
""" + COMMON_TAIL,

    "preflight": """You are Project Agni running a PRE-FLIGHT check. The design was approved;
compute is about to be spent. Your only question: DOES THE CODE DO WHAT THE PREREG SAYS?

Do not re-litigate the design. Compare implementation against specification, line by line.

SCRIPT AS WRITTEN:
{primary}

WHAT THE PREREG PROMISES:
{secondary}

THREATS THE AUTHORS CLAIMED TO HAVE MITIGATED:
{tertiary}

For each promised control, mitigation, seed, residualization or arm: is it IN the code?
Quote the implementing line, or state plainly that it is absent.
""" + COMMON_TAIL,

    "results": """You are Project Agni checking RESULTS against DATA. Compute has run.
Your only question: DOES EVERY NUMBER IN THIS ARTIFACT APPEAR IN THE SHIPPED DATA?

ARTIFACT:
{primary}

SHIPPED DATA (filenames and excerpts):
{secondary}

For every numeral in the artifact: trace it to a field in the data, or flag it.
Recompute what you can. Show as-published vs recomputed side by side.
Flag any number you cannot locate -- an untraceable number is a CRITICAL finding.
""" + COMMON_TAIL,

    "paper": """You are Project Agni reviewing a PAPER before submission. Results are in.

THE HIGHEST-YIELD CHECK, do it first and exhaustively:
For every LABEL claim in this paper -- a definition, a count, a category, a scope
qualifier, an architecture description, an exclusion -- identify which REPORTED
STATISTIC depends on it. A label error is never only a label error. In 27 papers
audited at this lab, this check found a defect in every single one, and in every
case a careful reviewer had verified the label and stopped.

PAPER:
{primary}

DATA:
{secondary}

CODE:
{tertiary}
""" + COMMON_TAIL,

    "style": """You are Project Agni doing a FINAL STYLE AND INTEGRITY pass. The science is settled;
you are checking whether the artifact is fit to ship.

PRIMARY EDITION:
{primary}

TWIN EDITION (if provided -- compare them):
{secondary}

Compare the two editions for divergence in claims, numbers, AND STRUCTURE --
specifically, whether any table row is filed under a DIFFERENT SECTION HEADING in
one edition than the other. That defect is invisible to string search and has
shipped here before: a retracted finding stayed under "Confirmed" in the
venue-facing edition while the other edition filed it under "Falsified."
""" + COMMON_TAIL,
}

PHASE_KILLS = {
    "design": DESIGN_KILLS,
    "preflight": DESIGN_KILLS,
    "results": RESULTS_KILLS,
    "paper": PAPER_KILLS,
    "style": STYLE_KILLS,
}


def _find_claude_bin():
    """Locate the Claude Code binary. `command -v claude` fails under non-login
    ssh because ~/.local/bin is not on the non-interactive PATH -- which is how
    this box looked like it had no CLI at all."""
    import shutil
    for p in (Path.home() / ".local/bin/claude",
              Path("/mnt/data1/local/bin/claude"),
              Path("/usr/local/bin/claude"),
              # Starship keeps it here; shutil.which() misses it because the
              # non-interactive SSH PATH does not include homebrew.
              Path("/opt/homebrew/bin/claude"),
              Path("/home/linuxbrew/.linuxbrew/bin/claude")):
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return shutil.which("claude")


CLAUDE_BIN = _find_claude_bin()


HONESTY_CLAUSE = """
CRITICAL CONSTRAINT ON YOUR OWN EVIDENCE.
You may have file access. If so, verify claims against PRIMARY sources -- the code and
data themselves.

If you cannot read a file you need, you MUST say "I could not read <path>" and mark any
finding that depends on it as UNVERIFIED. You must NOT substitute a secondary
description of that file -- a prior review, a README, a summary, another agent's report
-- and then state the conclusion at CRITICAL confidence.

This has already happened once. On 2026-08-14 an Agni run was sandboxed to one
directory, could not read the implementation it was reviewing, reasoned from an earlier
review that described a bug fixed an hour previously, and reported it as CRITICAL
finding #1. The finding was stale. Reasoning from a secondary while the primary is
unreachable is the exact failure this reviewer exists to catch. Do not commit it.

A finding you could not verify is still worth reporting -- as UNVERIFIED, with the path
you could not reach.
"""


def infer_cli(prompt, system, timeout, add_dirs=None, model=None):
    """Route through the headless Claude Code CLI instead of a raw HTTP POST.

    WHY THIS IS THE DEFAULT (2026-08-14): during the sprint, llm.py's direct
    POST to api.anthropic.com returned 429 continuously for hours, while the
    CLI succeeded on the SAME OAuth token at the SAME moment. The token was
    valid (max subscription, default_claude_max_20x) -- the problem was
    contention on one shared quota against five Haiku servers, a live Claude
    Code session and Penumbra. The CLI has its own queueing and backoff; the
    raw POST has none, so it loses every race.

    Prompt goes over stdin, not argv: these run past 30k chars and would blow
    ARG_MAX. Verified working at that size.
    """
    if not CLAUDE_BIN:
        raise RuntimeError("no claude binary found")
    full = cli_stdin(prompt, system)
    cmd = [CLAUDE_BIN, "-p"]
    # Forward the model. Ollama tags ("qwen3.8:27b-mxfp8") carry a colon and
    # are meaningless to the CLI, so they are never passed through.
    if model and ":" not in model:
        cmd += ["--model", model]
    for d in (add_dirs or []):
        cmd += ["--add-dir", str(d)]
    # encoding MUST be explicit: on Windows text=True decodes with the ANSI
    # codepage (cp1252), and review text contains em-dashes / arrows / math
    # symbols whose UTF-8 bytes are undefined there (crashed on 0x81,
    # 2026-08-30). errors="replace" so a stray byte degrades one glyph
    # rather than destroying a completed review.
    #
    # Isolation (2026-11-02): until then this call ran as Lyra -- it loaded their
    # CLAUDE.md, ran their SessionStart sync, and its exchange landed in their
    # episodic memory. The helper adds what its docstring lists (no-capture env,
    # CLAUDE.md excludes, strict empty MCP, no auto-memory, no skills); nothing
    # else changes. Tested by test_isolation.py.
    cmd, env = isolated_claude_cmd(cmd)
    r = subprocess.run(cmd, input=full, capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       timeout=timeout, env=env)
    if r.returncode != 0:
        # The CLI prints an API error (a refusal, say) on stdout, and stderr can be empty: a real
        # refusal came back as "claude CLI exit 1: " with no cause (2026-11-07). Report both.
        raise RuntimeError("claude CLI exit %d: stderr: %s | stdout: %s" % (
            r.returncode, (r.stderr or "").strip()[:300] or "(empty)", (r.stdout or "").strip()[:300] or "(empty)"))
    return r.stdout.strip()


def cli_stdin(prompt, system):
    """Exactly what infer_cli sends. The budget is checked against this, not the prompt alone."""
    return (system + "\n\n" + HONESTY_CLAUSE + "\n\n" + prompt) if system else prompt


# ---------------------------------------------------------------- inputs
#
# NO SILENT CUTS (2026-11-07). read_capped used to slice files to 30000 chars, folders to their first
# 25 entries at 4000 chars each, secondary to 12000 and tertiary to 8000, with prereg sections cut
# to 6000 mid-JSON. A named path that did not exist became the text "[NOT PROVIDED ...]". Every one
# of those still returned a confident verdict and recorded nothing. Now every input arrives whole
# or the run is refused.

def _decode(raw):
    """(text, replacement count). errors="replace" as before: a stray byte degrades one glyph
    rather than losing a file, but the count makes the input not complete. A U+FFFD the file
    really holds, validly encoded, is the file's own text and is not counted. Newlines as read_text
    gave them."""
    text = raw.decode("utf-8", errors="replace")
    replaced = text.count("\ufffd") - raw.count("\ufffd".encode("utf-8"))
    return text.replace("\r\n", "\n").replace("\r", "\n"), replaced


def _mostly_replaced(text, replaced):
    return replaced > BINARY_REPLACEMENT_RATIO * len(text)


def _read_file(path, unread_above):
    """One file, whole. text is None if it is binary or was left unread. Only the first 8 KB is read
    until the file is known to be text, so a 5 GB .pt in a data folder costs 8 KB. A text file over
    unread_above is measured and not read: it can only end in a refusal, and reading it could
    exhaust memory on a small review machine."""
    with open(path, "rb") as fh:
        head = fh.read(SNIFF_BYTES)
        size = os.fstat(fh.fileno()).st_size
        if b"\0" in head:
            return {"text": None, "bytes": size, "binary": True, "replaced": 0}
        if size > unread_above:
            text, replaced = _decode(head)
            return {"text": None, "bytes": size, "binary": _mostly_replaced(text, replaced), "replaced": 0}
        raw = head + fh.read()
    text, replaced = _decode(raw)
    if _mostly_replaced(text, replaced):
        return {"text": None, "bytes": size, "binary": True, "replaced": 0}
    return {"text": text, "bytes": size, "binary": False, "replaced": replaced}


def _inside(root, target):
    try:
        return os.path.commonpath([root, target]) == root
    except ValueError:      # different drives on Windows
        return False


def _under_junk(real_root, real):
    """Whether a real path inside the tree lies in a junk folder, which the walk never enters."""
    junk = {os.path.normcase(j) for j in JUNK_DIRS}
    return any(part in junk for part in os.path.relpath(real, real_root).split(os.sep))


def _list_tree(root):
    """Every regular file under root as (relative name, path), in a fixed order, plus what was
    skipped and why. Only real folders are walked: a link (symlink or Windows junction, both caught
    by comparing real paths) is never followed, so a loop cannot hang this and nothing outside the
    tree is read."""
    files, skipped, problems = [], [], []
    real_root = os.path.normcase(os.path.realpath(root))
    pending = [(str(root), "")]
    while pending:
        folder, rel = pending.pop()
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name)
        except OSError as e:
            problems.append("%s cannot be listed: %s" % (rel or folder, e))
            continue
        real_folder = os.path.realpath(folder)
        for entry in entries:
            name = rel + entry.name
            real = os.path.normcase(os.path.realpath(entry.path))
            if real != os.path.normcase(os.path.join(real_folder, entry.name)):
                if not _inside(real_root, real):
                    reason = SKIP_LINK_OUTSIDE
                elif not os.path.exists(real):
                    reason = SKIP_LINK_BROKEN
                elif _under_junk(real_root, real):
                    reason = SKIP_LINK_INTO_JUNK
                else:
                    reason = SKIP_LINK_INSIDE
                skipped.append({"name": name, "reason": reason})
            elif entry.is_dir(follow_symlinks=False):
                if entry.name in JUNK_DIRS:
                    skipped.append({"name": name + "/", "reason": SKIP_JUNK})
                else:
                    pending.append((entry.path, name + "/"))
            elif entry.is_file(follow_symlinks=False):
                files.append((name, entry.path))
            else:
                skipped.append({"name": name, "reason": SKIP_SPECIAL})
    files.sort(key=lambda f: f[0].split("/"))
    skipped.sort(key=lambda s: s["name"])
    return files, skipped, problems


def _read_dir(root, budget, got):
    """Fill got from a folder: recursive, fixed order, relative names, every text file whole."""
    cov = got["coverage"]
    files, cov["skipped"], problems = _list_tree(root)
    got["problems"] += ["%s: %s" % (cov["path"], p) for p in problems]
    parts, total = [], 0
    for name, path in files:
        try:
            # Already over budget means refusal: stop holding text, keep measuring for the message.
            one = _read_file(path, 0 if total > budget else MAX_BYTES_PER_CHAR * budget)
        except OSError as e:
            got["problems"].append("%s: %s cannot be read: %s" % (cov["path"], name, e))
            continue
        if one["binary"]:
            cov["binary_not_shown"].append({"name": name, "bytes": one["bytes"]})
        elif one["text"] is None:
            got["unread"] += one["bytes"]
            total += one["bytes"]
            got["sizes"].append((one["bytes"], name))
        else:
            parts.append("--- %s ---\n%s" % (name, one["text"]))
            total += len(one["text"])
            got["sizes"].append((len(one["text"]), name))
            cov["files_included"] += 1
            cov["replacement_chars"] += one["replaced"]
            got["has_text"] = got["has_text"] or bool(one["text"].strip())
    got["text"] = "\n\n".join(parts)


def _new_coverage(role, path, kind):
    return {"role": role, "path": str(path), "kind": kind, "chars": 0, "files_included": 0,
            "binary_not_shown": [], "skipped": [], "replacement_chars": 0, "complete": False}


def read_input(role, path, budget, kind="file"):
    """Read one named input whole. Returns text, coverage, problems (any problem means refusal),
    sizes (chars, name) for the refusal message, and unread (bytes measured but not read)."""
    cov = _new_coverage(role, path, kind)
    got = {"text": "", "coverage": cov, "problems": [], "sizes": [], "unread": 0, "has_text": False}
    if not os.path.exists(path):
        got["problems"].append("%s: %s does not exist" % (role, path))
        return got
    if os.path.isdir(path):
        cov["kind"] = "dir"
        _read_dir(path, budget, got)
    elif os.path.isfile(path):
        try:
            one = _read_file(path, MAX_BYTES_PER_CHAR * budget)
        except OSError as e:
            got["problems"].append("%s: %s cannot be read: %s" % (role, path, e))
            return got
        if one["binary"]:
            cov["binary_not_shown"].append({"name": Path(path).name, "bytes": one["bytes"]})
            got["text"] = "[NOT SHOWN: %s is a binary file; see COVERAGE]" % Path(path).name
        elif one["text"] is None:
            got["unread"] = one["bytes"]
            got["sizes"].append((one["bytes"], str(path)))
        else:
            got["text"] = one["text"]
            got["sizes"].append((len(one["text"]), str(path)))
            got["has_text"] = bool(one["text"].strip())
            cov["files_included"] = 1
            cov["replacement_chars"] = one["replaced"]
    else:
        got["problems"].append("%s: %s is not a file or a folder" % (role, path))
        return got
    # Binaries alone do not invalidate a data folder, but a primary with no text is a review of
    # nothing, and so is any named input with nothing in it at all.
    if not got["has_text"] and not got["unread"] and not got["problems"]:
        if cov["binary_not_shown"] and role == "primary":
            got["problems"].append("primary: %s has no text to review, only binary files" % path)
        elif not cov["binary_not_shown"]:
            got["problems"].append("%s: %s is empty" % (role, path))
    cov["chars"] = (0 if cov["kind"] == "file" and cov["binary_not_shown"] else len(got["text"])) + got["unread"]
    if got["unread"]:
        cov["unread_chars"] = got["unread"]
    # A reviewer shown "12.3" and U+FFFD where the file says 12.3 times did not get all of its input.
    cov["complete"] = not (got["problems"] or got["unread"] or cov["binary_not_shown"]
                           or cov["replacement_chars"]
                           or any(s["reason"] not in HARMLESS_SKIPS for s in cov["skipped"]))
    return got


def _no_duplicate_keys(pairs):
    """json.loads keeps the last of two equal keys and drops the first without a word."""
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError("duplicate key %r: one of its values would never reach the reviewer" % key)
        seen.add(key)
    return dict(pairs)


def read_prereg(path, budget):
    """Returns (got, secondary, tertiary). The reviewer gets the whole prereg. Secondary is the
    hypotheses, then every other key but threats_to_validity under OTHER PRE-REGISTRATION FIELDS;
    tertiary is the threats. A prereg with neither hypotheses nor hypothesis goes whole as
    secondary: it used to send the reviewer an empty {}.

    Round 3 (2026-11-07): other keys used to be named and not sent, which made the input not
    complete, and nearly every tracked prereg has such keys, so their design and preflight reviews
    could not pass without --accept-incomplete-input. Falsification criteria and analysis plans are
    what a design reviewer most needs to see."""
    got = read_input("secondary", path, budget, kind="prereg")
    cov = got["coverage"]
    if got["problems"] or got["unread"]:
        cov["kind"] = "prereg"
        return got, "", ""
    if cov["kind"] == "dir" or cov["binary_not_shown"]:
        cov["kind"], cov["complete"] = "prereg", False
        got["problems"].append("secondary: %s must be a prereg JSON file" % path)
        return got, "", ""
    try:
        prereg = json.loads(got["text"], object_pairs_hook=_no_duplicate_keys)
    # RecursionError is not a ValueError: a prereg nested a few thousand deep was a traceback,
    # exit 1, and the older APPROVED record stood (round 4).
    except (ValueError, RecursionError) as e:
        cov["complete"] = False
        got["problems"].append("secondary: %s is not valid JSON: %s" % (path, e))
        return got, "", ""
    if not isinstance(prereg, dict):
        cov["complete"] = False
        got["problems"].append("secondary: %s is not a JSON object" % path)
        return got, "", ""
    if "threats_to_validity" not in prereg:
        print("WARNING: prereg has no threats_to_validity; the scaffold reads it",
              file=sys.stderr)
    key = next((k for k in ("hypotheses", "hypothesis") if prereg.get(k)), None)
    tertiary = json.dumps(prereg.get("threats_to_validity", []), indent=2)
    if key:
        secondary = json.dumps(prereg[key], indent=2)
        other = [k for k in prereg if k not in (key, "threats_to_validity")]
        if other:
            secondary += ("\n\nOTHER PRE-REGISTRATION FIELDS (every other key in the prereg but "
                          "threats_to_validity, whole):\n" + json.dumps({k: prereg[k] for k in other}, indent=2))
        cov["prereg_other_keys_sent"] = sorted(other)
    else:
        secondary = json.dumps(prereg, indent=2)
        cov["secondary_is_whole_prereg"] = True
    # Every key reaches the prompt, so none is left out. Kept, empty, so a record says so outright.
    cov["prereg_keys_not_sent"] = []
    cov["chars"] = len(secondary) + len(tertiary)
    cov["complete"] = True
    got["sizes"] = [(cov["chars"], str(path))]
    return got, secondary, tertiary


def gather_inputs(args, budget):
    """Every input the user named, whole. Returns (sections, reads). A named input is never dropped:
    one the phase has no place for is a problem, not a quiet no-op, and an empty string is still
    named ("$DATA" with DATA unset), so it is refused as missing rather than read as absent."""
    reads = [read_input("primary", args.primary, budget)]
    sections = {"primary": reads[0]["text"], "secondary": "[NOT PROVIDED]", "tertiary": "[NOT PROVIDED]"}
    if args.phase in PREREG_PHASES:
        if args.secondary is not None:
            got, sections["secondary"], sections["tertiary"] = read_prereg(args.secondary, budget)
            reads.append(got)
        else:
            reads[0]["problems"].append("the %s phase requires a prereg.json" % args.phase)
    elif args.secondary is not None:
        reads.append(read_input("secondary", args.secondary, budget))
        sections["secondary"] = reads[-1]["text"]
    if args.tertiary is not None:
        if "{tertiary}" in PROMPTS[args.phase] and args.phase not in PREREG_PHASES:
            reads.append(read_input("tertiary", args.tertiary, budget))
            sections["tertiary"] = reads[-1]["text"]
        else:
            reads.append({"text": "", "coverage": _new_coverage("tertiary", args.tertiary, "file"), "sizes": [],
                          "unread": 0, "problems": ["tertiary: %s was named, but the %s phase has no place "
                                                    "for it, so the reviewer would never see it"
                                                    % (args.tertiary, args.phase)]})
    return sections, reads


def _coverage_line(item):
    bits = ["%s: %s (%s)" % (item["role"], item["path"], item["kind"]),
            "{:,} chars".format(item["chars"]),
            "%d file(s)" % item["files_included"]]
    if item.get("unread_chars"):
        bits.append("{:,} of those bytes on disk, measured and not read".format(item["unread_chars"]))
    if item["binary_not_shown"]:
        bits.append("%d binary not shown" % len(item["binary_not_shown"]))
    if item["skipped"]:
        bits.append("%d skipped" % len(item["skipped"]))
    if item["replacement_chars"]:
        bits.append("%d undecodable byte(s) shown as U+FFFD" % item["replacement_chars"])
    if item.get("prereg_keys_not_sent"):
        bits.append("%d prereg key(s) not sent" % len(item["prereg_keys_not_sent"]))
    bits.append("complete" if item["complete"] else "NOT COMPLETE")
    return ", ".join(bits)


def coverage_block(inputs):
    """The reviewer is told exactly what it was given and what it was not."""
    lines = ["COVERAGE -- exactly what this prompt gives you. Every text file included is whole;",
             "nothing was cut to fit. If a finding depends on something listed here as not shown,",
             "say so under what you could NOT check."]
    for item in inputs:
        lines.append("- " + _coverage_line(item))
        lines += ["    not shown, binary: %s (%d bytes)" % (b["name"], b["bytes"]) for b in item["binary_not_shown"]]
        lines += ["    skipped: %s (%s)" % (s["name"], s["reason"]) for s in item["skipped"]]
        if item.get("prereg_keys_not_sent"):
            lines.append("    prereg keys not sent to you: %s" % ", ".join(item["prereg_keys_not_sent"]))
        if item.get("prereg_other_keys_sent"):
            lines.append("    other prereg keys, sent whole after the hypotheses: %s"
                         % ", ".join(item["prereg_other_keys_sent"]))
        if item.get("secondary_is_whole_prereg"):
            lines.append("    the hypotheses section below is the whole prereg: it has no hypotheses key")
    return "\n".join(lines)


# ---------------------------------------------------------------- the record

def _record_path(args):
    if args.out:
        return Path(args.out)
    try:
        return Path(args.primary).with_suffix(".agni_%s.json" % args.phase)
    except ValueError:      # "." or a drive root has no name to hang a suffix on
        return Path("agni_%s.json" % args.phase)


def _record(args, model, verdict, review, inputs, **extra):
    # INVALID means no review of these inputs exists, however well they were read.
    items = [dict(i, complete=False) for i in inputs] if verdict == "INVALID" else inputs
    record = {
        "version": AGNI_PHASES_VERSION,
        "phase": args.phase,
        "model": model,
        "primary": str(args.primary),
        "secondary": str(args.secondary) if args.secondary else None,
        "tertiary": str(args.tertiary) if args.tertiary else None,
        "verdict": verdict,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "known_kills_checked": 0 if verdict == "INVALID" else len(PHASE_KILLS[args.phase]),
        "input_coverage": {"inputs": items, "input_complete": all(i["complete"] for i in items)},
        "review": review,
    }
    record.update(extra)
    return record


def _write_record(out, record):
    """Returns None, or what went wrong. Never raises: a traceback exits 1, which reads as REJECTED."""
    try:
        Path(out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    except (OSError, ValueError) as e:
        return "%s: %s" % (e.__class__.__name__, e)
    return None


def _say_record_unwritten(out, err):
    print("ERROR: could not write the record to %s (%s)." % (out, err), file=sys.stderr)
    print("       Anything already at that path is STALE. Do not read it as this run's result.",
          file=sys.stderr)


def _refuse(args, out, inputs, sizes, reasons, advice):
    """No reviewer call happens after this. Writes INVALID over whatever is at the output path, so a
    stale APPROVED cannot survive, and exits 2 even if that write fails."""
    err = _write_record(out, _record(args, args.model, "INVALID", "", inputs, invalid_reasons=reasons))
    print("ERROR: REFUSED -- verdict INVALID, exit 2. No reviewer saw this input.", file=sys.stderr)
    print("       This is NOT a pass. The phase is UNGATED.", file=sys.stderr)
    for reason in reasons:
        print("       Why: %s" % reason, file=sys.stderr)
    print("       Inputs:", file=sys.stderr)
    for item in inputs:
        print("         %s" % _coverage_line(item), file=sys.stderr)
    if sizes:
        print("       Largest files:", file=sys.stderr)
        for chars, name in sorted(sizes, key=lambda s: -s[0])[:LARGEST_SHOWN]:
            print("         {:>12,} chars  {}".format(chars, name), file=sys.stderr)
    print("       What to do: %s" % " ".join(advice), file=sys.stderr)
    if err:
        _say_record_unwritten(out, err)
    else:
        print("       Record: %s" % out, file=sys.stderr)
    return 2


def _env_budget(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default, []
    try:
        return int(raw), []
    except ValueError:
        return default, ["%s=%r is not a whole number of chars" % (name, raw)]


def _warn_if_foreign_llm():
    """sys.path puts agni-open's llm.py first on machines that have it. This repo's llm.py sets
    num_ctx; another copy may not, and then Ollama can still cut the prompt with no signal."""
    if Path(llm.__file__).resolve().parent != Path(__file__).resolve().parent:
        print("WARNING: llm.py comes from %s, not from beside this script. If it does not set "
              "num_ctx, Ollama trims the prompt to its context length with no signal, under any "
              "budget here." % llm.__file__, file=sys.stderr)


def _llm_review(backend, **kw):
    """The reply over llm.py, as (review, (fields for the record, why a pass in it is not one)).

    Round 6 (2026-11-07): over Ollama, llm.py joined a reasoning model's thinking to its answer,
    thinking first, or returned the thinking alone when the answer was empty. A draft "AGNI-VERDICT:
    APPROVED" in the thinking of a reply with no answer was recorded APPROVED, exit 0, and a draft
    "VERDICT: REJECTED?" in the thinking vetoed a well-formed answer. Now only the answer is parsed,
    and the thinking is kept beside it in the record. A reply that is all thinking is parsed, as the
    design harness parses it, but a pass read there is not a pass. An llm.py that cannot give the two
    apart (another copy first on sys.path) gives no pass over Ollama either."""
    if getattr(llm, "SPLITS_THINKING", False):
        got = llm.infer(backend=backend, split_thinking=True, **kw)
    else:
        got = llm.infer(backend=backend, **kw)
    if not isinstance(got, tuple):
        unsure = None
        if backend == "ollama" and got and got.strip():
            unsure = "through %s, which may join the model's thinking to its answer" % llm.__file__
        return got, ({}, unsure)
    answer, thinking = got[0] or "", got[1] or ""
    if answer.strip() or not thinking.strip():
        return answer, ({"thinking": thinking} if thinking.strip() else {}, None)
    return ("[recovered from thinking field]\n\n" + thinking.strip(),
            ({"recovered_from_thinking": True}, "only from the model's thinking, not from an answer"))


# ---------------------------------------------------------------- the verdict

_VERDICT_WORD = r"(APPROVED|CONDITIONAL|REJECTED)"
# Not ~: a struck-out verdict is not a verdict. A > line is read by _seen (round 6): round 5 took
# every quote for someone else's, and a blockquoted "> **VERDICT: REJECTED**" callout did not veto.
_MARKUP = re.compile(r"[*_`#]+")
# Round 5: the reply's last line, which decides. Exactly this once markup is stripped, nothing after.
_PROTOCOL = re.compile(r"^AGNI-VERDICT:\s*" + _VERDICT_WORD + r"$", re.I)
_BARE_VERDICT = re.compile(r"^" + _VERDICT_WORD + r"[.!]?$", re.I)
# The label of a contract-form line, shared with agni_design_review.py. Round 6 added "Agni verdict"
# and a parenthesis after the label ("VERDICT (design phase): REJECTED"); neither vetoed in round 5.
_LABEL = (r"^(?:(?:[-+]|\d+[.)])\s+)?"                   # the prompt's own "1. VERDICT ..." numbering
          r"(?:(?:OVERALL|FINAL|AGNI)(?:\s+|-)){0,2}VERDICT\b(?:\s*\([^()]*\))?")
_LABELLED_VERDICT = re.compile(
    _LABEL + r"[\s:=\u2013\u2014-]*" + _VERDICT_WORD +
    r"(?![\w-])"                          # not CONDITIONALLY, not APPROVED-ish
    r"(?!\s*(?:/|\||,|\bOR\b)\s*(?:APPROVED|CONDITIONAL|REJECTED)\b)",    # an echo of the contract
    re.I)
# A label alone, "## Verdict", with its word on the next line (round 6).
_LABEL_ONLY = re.compile(_LABEL + r"[\s:=\u2013\u2014-]*$", re.I)
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
# After a pass verdict, what takes it back: a negation, a contrast, a question or an arrow.
_TAKE_BACK = re.compile(r"\b(?:not|never|nor|but|however|instead|rather|except|although|though)\b"
                        r"|n't\b|\bno\b(?!\s+\w)|\?|->|=>|<-|[\u2190-\u21ff\u27f0-\u27ff\u2900-\u297f]",
                        re.I)
# After APPROVED, what makes it depend on something, which is not APPROVED.
_CONDITION = re.compile(r"\b(?:if|unless|otherwise|provided|pending|until|once|subject\s+to|"
                        r"condition\w*)\b", re.I)


def _spaces(line):
    return len(line) - len(line.lstrip(" "))


# An ATX heading or a setext underline: not a paragraph line, so indented code may follow it.
_HEADING = re.compile(r"^ {0,3}(?:#{1,6}(?:[ \t]|$)|=+[ \t]*$)")


def _code_lines(review, faithful=False):
    """Each line of the review as (line, in code). A fence's closing line is code too, and a fence or
    an HTML block left open keeps every line after it in code.

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
    with a tag starts an HTML block.

    faithful (round 6, 2026-11-07) is the reading as close to CommonMark as this gets, and where it
    is unsure it keeps less in code: a fence closes at up to three spaces of indent and ends with its
    list item; an HTML block other than <pre>, <script>, <style>, <textarea>, a comment, <?, <!X or
    CDATA is text, since a reader sees its text; and four spaces of indent right after a paragraph
    line are more of that paragraph, since indented code cannot interrupt one. Round 5 let code
    neither decide nor veto, so every line the wider reading kept in code was a place a visible
    REJECTED could hide. _contract_verdicts reads both: a line decides only when neither puts it in
    code, and is skipped as code only when both do."""
    lines, fence, html, item, para = [], None, None, 0, False
    for raw in _LINE_END.split(review):
        line = raw.expandtabs(4)
        indent, blank = _spaces(line), not line.strip()
        if fence:
            char, length, opened_at, container = fence
            if not (faithful and not blank and indent < container):
                if ((container if faithful else opened_at) <= indent <= container + 3
                        and re.match(r"^ *%s{%d,}[ \t]*$" % (re.escape(char), length), line)):
                    fence = None
                lines.append((raw, True))
                continue
            fence, item = None, 0                 # faithful: the list item ends, and its fence with it
        if html:
            lines.append((raw, not faithful or html[0] is not None))
            if html[0].search(line) if html[0] else blank:
                html = None
            para = False
            continue
        if not blank and indent < item:
            item = 0
        pos, in_code, text, marker = item, False, False, False
        while not blank:
            rest = line[pos:]
            if _spaces(rest) >= 4:
                in_code = not (faithful and para and not marker)
                text = not in_code
                break
            m = _FENCE_OPEN.match(rest)
            if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                fence = (m.group(1)[0], len(m.group(1)), pos + _spaces(rest), pos)
                in_code = True
                break
            start, end = next(((s.match(rest), e) for s, e in _HTML_BLOCKS if s.match(rest)), (None, None))
            if start:
                in_code = not faithful or end is not None
                if not (end and end.search(rest, start.end())):
                    html = (end,)
                break
            if _THEMATIC_BREAK.match(rest) or _HEADING.match(rest):
                break
            m = _LIST_ITEM.match(rest)
            if not m:
                text = True
                break
            gap = len(m.group(3))
            pos += len(m.group(1)) + len(m.group(2)) + (gap if gap <= 4 and rest[m.end():].strip() else 1)
            item, marker = pos, True
        lines.append((raw, in_code))
        para = text
    return lines


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


# After a CONDITIONAL, what takes it back: a question or an arrow right after the word (round 6).
_CONDITIONAL_TAKE_BACK = re.compile(r"^\W*?(?:\?|->|=>|<-|[\u2190-\u21ff\u27f0-\u27ff\u2900-\u297f])")
# CommonMark inline code: a run of backticks, closed by a run of the same length.
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_QUOTE = re.compile(r"^(?:\s*>)+")


def _takes_back(verdict, tail):
    """Whether the text after a pass verdict's word takes it back. A REJECTED is not checked: however
    it is qualified, it cannot pass. An APPROVED is taken back by a negation, a contrast, a question,
    an arrow or a condition after it. A CONDITIONAL is qualified by nature ("CONDITIONAL -- not
    runnable until fix 1 lands"), so only a question or an arrow right after its word takes it back
    (round 6: round 4 read its conditions as a take-back, and a well-formed CONDITIONAL was UNPARSED)."""
    if verdict == "REJECTED":
        return False
    if verdict == "CONDITIONAL":
        return bool(_CONDITIONAL_TAKE_BACK.match(tail))
    return bool(_TAKE_BACK.search(tail) or _CONDITION.search(tail))


def _clear(word, tail, named):
    """The word a contract-form line gives, or UNCLEAR. Any other verdict word on the line makes it
    unclear, except an APPROVED on a CONDITIONAL line, which names where its conditions lead:
    "VERDICT: CONDITIONAL (can be APPROVED once fix 1 is in)" (round 6)."""
    others = named - {word} - ({"APPROVED"} if word == "CONDITIONAL" else set())
    return "UNCLEAR" if others or _takes_back(word, tail) else word


def _plain(line):
    """A line as a reader sees its words: markup characters and surrounding white space gone."""
    return _MARKUP.sub("", line).strip()


def _quotes_a_verdict(span):
    folded = _fold(span)
    return "verdict" in folded and bool(_VERDICT_STEM.search(folded))


def _seen(line):
    """A line as a reader sees it, for a veto, and whether it is blockquoted (round 6). An inline code
    span that holds a verdict line, "`VERDICT: APPROVED`", quotes a fixture and is dropped; a span
    round only the word, "VERDICT: `REJECTED`", is the reviewer's own formatting and stays. Then
    blockquote markers, HTML tags and markup go: "> **VERDICT: REJECTED**" and "<b>VERDICT:
    REJECTED</b>" are verdict lines a reader sees."""
    line = _CODE_SPAN.sub(lambda m: " " if _quotes_a_verdict(m.group(2)) else m.group(0), line)
    return _plain(_TAG.sub("", _QUOTE.sub("", line))), bool(_QUOTE.match(line))


def _contract_verdicts(review, protocol, contract_line):
    """The verdict the review gives by protocol, or None, and what each other verdict line in a form
    of the contract gives.

    The protocol line is the last non-blank line: outside code, and matched by protocol() once
    markup is stripped. Any other line not in code is in a form of the contract when
    contract_line() matches it as _seen gives it, as written or folded (a Cyrillic E or a zero-width
    space does not hide one). It gives its verdict, or UNCLEAR when the line does not clearly give
    it (see _clear; round 4: a verdict line is read whole). The contract read back is skipped. Prose
    and lines in code give nothing (round 5).

    Round 6 (2026-11-07). Code is read twice (see _code_lines): the protocol line counts only when
    neither reading puts it in code, and a line is skipped as code only when both do, so a line the
    reading is unsure of cannot decide and can veto. A label alone, "## Verdict", reads the next
    line as its verdict when that line starts with a verdict word. A blockquoted line can stop a pass
    but cannot turn a rejection into UNPARSED, since a quote may be someone else's verdict."""
    wide, faithful = _code_lines(review), _code_lines(review, faithful=True)
    lines = [(line, a, b) for (line, a), (_, b) in zip(wide, faithful) if line.strip()]
    if not lines:
        return None, []
    last, last_wide, last_faithful = lines[-1]
    said = None if last_wide or last_faithful else protocol.match(_plain(last))
    said = said.group(1).upper() if said else None
    others, label = [], False
    for line, in_wide_code, in_faithful_code in lines[:-1]:
        folded = _fold(line)
        after_label, label = label, False
        if (in_wide_code and in_faithful_code) or _is_echo(folded):
            continue
        named = {w.upper() for w in _VERDICT_STEM.findall(folded)}
        for text, quoted in (_seen(line), _seen(folded)):
            if _LABEL_ONLY.match(text):
                label = True
                continue
            if after_label and not contract_line(text):
                text = "VERDICT: " + text
            m = contract_line(text)
            if m and not (quoted and said == "REJECTED"):
                others.append(_clear(m.group(1).upper(), text[m.end():], named))
    return said, others


def verdict_lines(review):
    """The verdict the review's AGNI-VERDICT line gives, or None, and what each other verdict line
    outside code gives (a bare word, or a VERDICT, OVERALL VERDICT, FINAL VERDICT or AGNI-VERDICT
    line): see _contract_verdicts."""
    return _contract_verdicts(review, _PROTOCOL,
                              lambda text: _BARE_VERDICT.match(text) or _LABELLED_VERDICT.match(text))


def parse_verdict(review):
    """The verdict the review's last line gives by protocol, if every other verdict line agrees,
    else UNPARSED.

    Scan the WHOLE review, not the first 15 lines: thinking models emit reasoning before the review
    header, which pushed "## VERDICT REJECTED" out of range and recorded a rejection as UNPARSED
    (2026-08-30). But the scan took the first line with any of the three words anywhere, in any
    case, so reasoning like "I would not call this approved" recorded APPROVED and exited 0.
    Anchored to the contract 2026-11-07, after replaying old and new parsers over 928 stored records:
    no record in the three-word vocabulary changed, and every form below that occurs there still
    parses. The 95 that differ are from a harness with another vocabulary (ACCEPT, MINOR_REVISIONS,
    MAJOR_REVISIONS), where the old parser had matched prose.

    Code is skipped (round 2, 2026-11-07): a fenced block, or a line indented as code (a tab, or four
    spaces). Agni reviews its own tooling, whose fixtures hold "VERDICT: APPROVED" lines, and a
    quoted one turned a "Final verdict: REJECTED" into APPROVED, exit 0.

    Code can veto but never decide (round 3, 2026-11-07). Round 2 flipped "in code" on any line
    that started with three backticks or tildes, so a one-line ```span```, a tilde fence "closed" by
    backticks, or a fence left open hid the reviewer's REJECTED and let a quoted APPROVED decide,
    exit 0. Now fences follow CommonMark (see _code_lines); only a verdict line outside code decides;
    a verdict line anywhere, in code too, that disagrees makes it UNPARSED; and so does a fence that
    never closes, or a review whose only verdict lines are in code. Even wrong about what is code,
    a quoted verdict that differs from the reviewer's can never pass.

    A verdict line is read whole (round 4, 2026-11-07). The decision regex anchors only the start of
    the line, so "VERDICT: APPROVED is not warranted here; this design must be REJECTED" and
    "APPROVED `VERDICT: REJECTED`" decided APPROVED, exit 0, and a REJECTED it did not know (a table
    cell, after an emoji, with a Cyrillic E or a zero-width space) could not veto. Now every line
    that names a verdict vetoes with each verdict word on it, and a pass verdict that is negated, or
    an APPROVED made conditional, is UNPARSED (in _verdicts, replaced in round 5). The cost, taken on purpose: a pass
    qualified that way ("CONDITIONAL - not before a null arm") is UNPARSED, exit 2, and so is a review
    whose prose names a different verdict within two words of the word "verdict".

    Decided by protocol (round 5, 2026-11-07). Round 4 vetoed with every verdict word a line
    named within two words of "verdict", prose included. Replayed over the stored reviews, that
    turned 2 of 85 legitimate three-word reviews (stored CONDITIONAL) into UNPARSED, on prose about
    an experiment's own verdict logic: "If H2 had failed ..., the verdict should be REJECTED
    regardless of drift". Our experiments are full of verdict logic, so a prose veto is a false
    alarm aimed at our own corpus. Now COMMON_TAIL asks for a last line "AGNI-VERDICT: <word>", and
    that line decides: the last non-blank line must be outside code and, markup stripped, exactly
    that, or the review is UNPARSED. Every other verdict line outside code in a form of the contract
    (a bare word; a VERDICT, OVERALL VERDICT or FINAL VERDICT line; another AGNI-VERDICT line) must
    give the same word, read whole and not taken back. A verdict decided by protocol needs no prose
    reading, so prose never decides or vetoes, and neither does a line in code; a fence or an HTML
    block left open puts the last line in code, which is UNPARSED. The cost, taken on purpose: a
    different verdict stated only in prose, in a table cell or after an emoji no longer vetoes.
    Reviews stored before this round have no AGNI-VERDICT line and parse UNPARSED; nothing re-parses
    them.

    Read as a reader reads it (round 6, 2026-11-07). Round 5's verifier found a REJECTED a reader
    sees that did not veto an APPROVED, exit 0: in a blockquote, under a "## Verdict" heading, after
    "Agni verdict" or "VERDICT (design phase)", in an HTML block other than a comment or <pre>, and
    in four spaces of indent right after a paragraph line. A fixture verdict in an inline code span,
    or a CONDITIONAL that states its conditions, gave UNPARSED on a correct review. See
    _contract_verdicts, _code_lines, _seen and _takes_back. Over Ollama only the answer is parsed,
    never the model's thinking (see _llm_review)."""
    said, others = verdict_lines(review)
    if said is None or any(v != said for v in others):
        return "UNPARSED"
    return said


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=list(PROMPTS))
    ap.add_argument("primary")
    ap.add_argument("secondary", nargs="?", default=None)
    ap.add_argument("tertiary", nargs="?", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", choices=["cli", "http"], default="cli",
                    help="cli = headless Claude Code (shares OAuth, has its own "
                         "queueing); http = raw POST via llm.py (429s under contention)")
    ap.add_argument("--add-dir", action="append", default=None, dest="add_dir",
                    help="extra directory the reviewer may read; repeatable. The repo "
                         "root containing the artefact is granted automatically")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-local-fallback", action="store_true",
                    help="fall back to local 30b if Claude is unavailable; output is stamped DEGRADED")
    ap.add_argument("--accept-incomplete-input", action="store_true",
                    help="exit 0 on APPROVED/CONDITIONAL although the reviewer did not get all of its "
                         "input (see input_coverage); the record is stamped incomplete_input_accepted")
    args = ap.parse_args()

    out = _record_path(args)
    budget, problems = _env_budget("AGNI_PROMPT_CHAR_BUDGET", PROMPT_CHAR_BUDGET)
    local_budget, local_problems = _env_budget("AGNI_LOCAL_CHAR_BUDGET", LOCAL_CHAR_BUDGET)
    problems += local_problems

    sections, reads = gather_inputs(args, budget)
    inputs = [r["coverage"] for r in reads]
    sizes = [s for r in reads for s in r["sizes"]]
    problems += [p for r in reads for p in r["problems"]]

    kills = "\n".join("  - %s" % k for k in PHASE_KILLS[args.phase])
    prompt = coverage_block(inputs) + "\n\n" + PROMPTS[args.phase].format(kills=kills, **sections)
    # Measured but unread bytes count: they are only ever there when the prompt is over budget.
    # So does the system text: infer_cli puts SYSTEM and HONESTY_CLAUSE before the prompt, and the
    # http backend sends SYSTEM beside it. Counting the prompt alone let 1,179 chars past (round 2).
    sent = len(cli_stdin(prompt, SYSTEM)) if args.backend == "cli" else len(SYSTEM) + len(prompt)
    prompt_chars = sent + sum(r["unread"] for r in reads)

    # REFUSE BEFORE ANY REVIEWER CALL. The one check on the whole prompt, and every named input.
    ollama_bound = (args.backend == "http"
                    and os.environ.get("AGNI_LLM_BACKEND", "anthropic") == "ollama")
    advice = []
    if problems:
        advice.append("Fix or drop the named input; a review of what is left is not a review of "
                      "what you asked for.")
    if prompt_chars > budget:
        problems.append("the assembled prompt is {:,} chars and AGNI_PROMPT_CHAR_BUDGET is {:,}".format(
            prompt_chars, budget))
        advice.append("Pass narrower files, or raise AGNI_PROMPT_CHAR_BUDGET deliberately if this "
                      "reviewer can really take it.")
    if ollama_bound and prompt_chars > local_budget:
        problems.append("the prompt is {:,} chars, bound for Ollama, and AGNI_LOCAL_CHAR_BUDGET is {:,}".format(
            prompt_chars, local_budget))
        advice.append("Pass narrower files, use a frontier backend, or raise AGNI_LOCAL_CHAR_BUDGET "
                      "and LLM_NUM_CTX together, deliberately.")
    if problems:
        for item in inputs:
            item["complete"] = False

    print("Agni Lifecycle Review v%s" % AGNI_PHASES_VERSION)
    print("  Phase       : %s" % args.phase.upper())
    print("  Primary     : %s" % args.primary)
    print("  Secondary   : %s" % (args.secondary or "-"))
    print("  Tertiary    : %s" % (args.tertiary or "-"))
    print("  Known kills : %d" % len(PHASE_KILLS[args.phase]))
    print("  Backend     : %s%s" % (args.backend, " (" + str(CLAUDE_BIN) + ")" if args.backend == "cli" else ""))
    for item in inputs:
        print("  Coverage    : %s" % _coverage_line(item))
    print("  Input whole : %s" % ("yes" if all(i["complete"] for i in inputs) else "NO"))
    print("  Prompt      : {:,} chars, budget {:,}".format(prompt_chars, budget))
    print()

    if problems:
        return _refuse(args, out, inputs, sizes, problems, advice)
    if ollama_bound:
        _warn_if_foreign_llm()

    # Replace whatever is at the output path BEFORE the reviewer runs. A run killed mid-review then
    # leaves INVALID, not an old APPROVED, and a record that cannot be written is found before a
    # review is spent on it.
    err = _write_record(out, _record(args, args.model, "INVALID", "", inputs, invalid_reasons=[
        "a review started and no result was recorded: it is still running, or it died"]))
    if err:
        return _refuse(args, out, inputs, sizes, ["the record cannot be written: %s" % err],
                       ["Point --out at a path that can be written."])

    # Retry with backoff. During the Aug-2026 sprint the shared Anthropic key was
    # 429-ing continuously, which made the gate unrunnable exactly when it was most
    # needed. A gate nobody can run is a gate nobody uses. We queue instead of dying.
    #
    # We do NOT silently fall back to the local 30b: it misreads geometry, and a
    # degraded review that looks like a real one is worse than no review. Local
    # fallback is opt-in and stamps the record.
    # Grant the reviewer read access to the whole repo containing the artefact.
    # Without this the CLI agent is sandboxed to its cwd; on 2026-08-14 that made it
    # unable to read the implementation under review, so it reasoned from a stale
    # secondary review and filed a fixed bug as CRITICAL. Give it the primary, always.
    grant_dirs = list(args.add_dir or [])
    anchor = Path(args.primary).resolve()
    d = anchor if anchor.is_dir() else anchor.parent
    for parent in [d] + list(d.parents):
        if (parent / ".git").exists():
            grant_dirs.append(str(parent))
            break
    else:
        grant_dirs.append(str(d))
    for extra in (args.secondary, args.tertiary):
        if extra and Path(extra).exists():
            p = Path(extra).resolve()
            grant_dirs.append(str(p if p.is_dir() else p.parent))
    grant_dirs = sorted(set(grant_dirs))
    if args.backend == "cli":
        print("  Read access : %s" % ", ".join(grant_dirs))

    review, reply = None, ({}, None)
    delays = [0, 20, 60, 150, 300, 600]
    attempts = timeouts = 0
    for i, wait in enumerate(delays):
        if wait:
            print("  backend busy; retry %d/%d in %ds..." % (i, len(delays) - 1, wait))
            time.sleep(wait)
        attempts += 1
        reply = ({}, None)
        try:
            if args.backend == "cli":
                review = infer_cli(prompt, SYSTEM,
                                   # was hardcoded 600s; the http path already
                                   # honours AGNI_TIMEOUT, so the cli path should
                                   # too. Large bundles exceed 10 min on Opus.
                                   # 1800 since 2026-11-07: whole inputs take longer than cut ones.
                                   timeout=int(os.environ.get("AGNI_CLI_TIMEOUT", "1800")),
                                   add_dirs=grant_dirs,
                                   model=args.model)
            else:
                # backend/limits made configurable so the SAME prompt can be
                # sent to a frontier model and a local one for comparison.
                # Reasoning models need a far larger budget: their analysis is
                # emitted as thinking tokens before any answer text appears.
                review, reply = _llm_review(
                    os.environ.get("AGNI_LLM_BACKEND", "anthropic"),
                    prompt=prompt, system=SYSTEM, model=args.model,
                    max_tokens=int(os.environ.get("AGNI_MAX_TOKENS", "6000")),
                    temperature=0.2,
                    timeout=int(os.environ.get("AGNI_TIMEOUT", "180")),
                )
        except subprocess.TimeoutExpired as e:
            # A timeout is not contention: the same prompt on the same model will likely time out
            # again, and all six delays at 1800s would burn hours and tokens (2026-11-07). One retry.
            timeouts += 1
            review = None
            print("  reviewer timed out after %ss (%d of %d allowed)" % (e.timeout, timeouts, TIMEOUTS_ALLOWED))
            if timeouts >= TIMEOUTS_ALLOWED:
                break
        except Exception as e:
            print("  backend error: %s: %s" % (e.__class__.__name__, str(e)[:700]))
            review = None
        if review and review.strip():
            if i:
                print("  succeeded after %d retries" % i)
            break

    backend_used = args.model
    if (not review or not review.strip()) and args.allow_local_fallback:
        if prompt_chars > local_budget:
            # Ollama would trim it with no signal, so this never sends. Same message as any refusal.
            return _refuse(args, out, inputs, sizes,
                           ["the local fallback was refused: the prompt is {:,} chars and "
                            "AGNI_LOCAL_CHAR_BUDGET is {:,}, and the primary reviewer gave no "
                            "review".format(prompt_chars, local_budget)],
                           ["Rerun when the primary reviewer is back, pass narrower files, or raise "
                            "AGNI_LOCAL_CHAR_BUDGET and LLM_NUM_CTX together, deliberately."])
        print("  falling back to LOCAL model -- review will be marked DEGRADED")
        _warn_if_foreign_llm()
        try:
            review, reply = _llm_review(
                "ollama", prompt=prompt,
                system="You are Project Agni, a hostile methodological reviewer.",
                model="qwen3:30b-a3b", max_tokens=6000, temperature=0.2, timeout=300,
            )
            # Parentheses matter: this was two adjacent literals on separate statements until
            # 2026-11-07, so the second half was a no-op and the stamp itself was cut off.
            backend_used = ("qwen3:30b-a3b (DEGRADED — misreads geometry, do not "
                            "trust for high-stakes shape claims)")
        except Exception as e:
            print("  local fallback failed: %s" % e.__class__.__name__)

    # B2: a gate must never let "no output" read as "no findings."
    if not review or not review.strip():
        print("ERROR: no review after %d attempts -- FAILING LOUD (exit 2)" % attempts,
              file=sys.stderr)
        print("       This is NOT a pass. The phase is UNGATED.", file=sys.stderr)
        err = _write_record(out, _record(args, backend_used, "INVALID", "", inputs, invalid_reasons=[
            "the reviewer gave no review after %d attempts (%d timed out)" % (attempts, timeouts)]))
        if err:
            _say_record_unwritten(out, err)
        return 2

    verdict = parse_verdict(review)
    notes, unsure = reply
    if unsure and verdict in ("APPROVED", "CONDITIONAL"):
        # Round 6, as the design harness's round 4: a rejection read there is kept, since it cannot pass.
        print("WARNING: %s was read %s. Recorded UNPARSED: NOT a pass." % (verdict, unsure), file=sys.stderr)
        verdict = "UNPARSED"

    # A non-empty review whose verdict cannot be classified is NOT a pass.
    # The guard above only covers EMPTY output; without this, an
    # unclassifiable REJECTED exits 0 and reads downstream as no finding.
    if verdict == "UNPARSED":
        print("ERROR: review has no single parseable verdict (no AGNI-VERDICT line last, outside "
              "code, or a verdict line that disagrees with it) -- FAILING LOUD (exit 2)", file=sys.stderr)
        print("       This is NOT a pass. Inspect the review text by hand.",
              file=sys.stderr)
        err = _write_record(out, _record(args, backend_used, "UNPARSED", review, inputs, **notes))
        if err:
            _say_record_unwritten(out, err)
        return 2

    # PERSIST BEFORE DISPLAY. A completed review is the deliverable; the console
    # is a convenience. This block used to sit BELOW the prints, so a cp1252
    # encoding failure on a checkmark destroyed a finished Opus review
    # (2026-08-30). Nothing that only affects rendering may discard the payload.
    incomplete = not all(i["complete"] for i in inputs)
    stamp = {"incomplete_input_accepted": args.accept_incomplete_input} if incomplete else {}
    err = _write_record(out, _record(args, backend_used, verdict, review, inputs, **stamp, **notes))

    print("=" * 70)
    print("AGNI %s REVIEW" % args.phase.upper())
    print("=" * 70)
    print(review)
    print("=" * 70)
    print("VERDICT: %s" % verdict)

    if err:
        # The placeholder or an older record is what is on disk, not this verdict.
        _say_record_unwritten(out, err)
        return 2
    print("Saved: %s" % out)

    if verdict == "REJECTED":
        return 1
    # Round 2 (2026-11-07): input_complete false used to stop at the record, so a twin read as
    # binary was reviewed against a stub and exited 0 APPROVED. A rejection on part of the input is
    # still a rejection; a pass on part of it is not a pass unless someone says so on purpose.
    if incomplete and not args.accept_incomplete_input:
        print("ERROR: %s on INCOMPLETE INPUT -- exit 2. The reviewer did not get all of its input "
              "(see Coverage above and input_coverage in the record)." % verdict, file=sys.stderr)
        print("       This is NOT a pass. Fix the input, or rerun with --accept-incomplete-input "
              "to take a review of what it got; the record is stamped.", file=sys.stderr)
        return 2
    if incomplete:
        print("NOTE: input NOT COMPLETE, accepted by --accept-incomplete-input; the record says so.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
