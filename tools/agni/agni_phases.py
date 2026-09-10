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

VERDICT CONTRACT (unchanged): APPROVED / CONDITIONAL / REJECTED.
A phase that returns empty FAILS LOUD (exit 2). A gate must never let
"no output" read as "no findings."

Usage
  python3 agni_phases.py design    <script.py> <prereg.json>
  python3 agni_phases.py preflight <script.py> <prereg.json>
  python3 agni_phases.py results   <artifact.md|tex> <data_dir>
  python3 agni_phases.py paper     <paper.md|tex> [data_dir] [code_dir]
  python3 agni_phases.py style     <paper.md|tex> [twin.md|tex]

  --model MODEL   override (default: session default in llm.py)
  --out PATH      where to write the JSON record
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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path.home() / "lab/projects/agni-open"))
import llm  # noqa: E402

AGNI_PHASES_VERSION = "1.0.0"
DEFAULT_MODEL = "opus"  # CLI alias -> claude-opus-5 (verified 2026-08-30).
# Was "claude-opus-4-6", a stale literal the CLI would now reject outright.
CHAR_CAP = 30000

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
    full = (system + "\n\n" + HONESTY_CLAUSE + "\n\n" + prompt) if system else prompt
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
    r = subprocess.run(cmd, input=full, capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("claude CLI exit %d: %s" % (r.returncode, r.stderr[:300]))
    return r.stdout.strip()


def read_capped(path, cap=CHAR_CAP):
    p = Path(path)
    if not p.exists():
        return f"[NOT PROVIDED: {path} does not exist]"
    if p.is_dir():
        parts = []
        for f in sorted(p.iterdir())[:25]:
            if f.is_file():
                try:
                    parts.append(f"--- {f.name} ---\n{f.read_text(encoding='utf-8', errors='replace')[:4000]}")
                except Exception:
                    parts.append(f"--- {f.name} --- [unreadable]")
        return "\n\n".join(parts)[:cap] or f"[EMPTY DIR: {path}]"
    return p.read_text(encoding="utf-8", errors="replace")[:cap]


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
    args = ap.parse_args()

    primary = read_capped(args.primary)
    secondary = tertiary = "[NOT PROVIDED]"

    if args.phase in ("design", "preflight"):
        if not args.secondary:
            sys.exit("FATAL: %s requires a prereg.json" % args.phase)
        prereg = json.loads(Path(args.secondary).read_text(encoding='utf-8'))
        if "threats_to_validity" not in prereg:
            print("WARNING: prereg has no threats_to_validity; the scaffold reads it",
                  file=sys.stderr)
        secondary = json.dumps(
            prereg.get("hypotheses", prereg.get("hypothesis", {})), indent=2)[:6000]
        tertiary = json.dumps(prereg.get("threats_to_validity", []), indent=2)[:6000]
    else:
        if args.secondary:
            secondary = read_capped(args.secondary, 12000)
        if args.tertiary:
            tertiary = read_capped(args.tertiary, 8000)

    kills = "\n".join("  - %s" % k for k in PHASE_KILLS[args.phase])
    prompt = PROMPTS[args.phase].format(
        primary=primary, secondary=secondary, tertiary=tertiary, kills=kills)

    print("Agni Lifecycle Review v%s" % AGNI_PHASES_VERSION)
    print("  Phase       : %s" % args.phase.upper())
    print("  Primary     : %s" % args.primary)
    print("  Secondary   : %s" % (args.secondary or "-"))
    print("  Tertiary    : %s" % (args.tertiary or "-"))
    print("  Known kills : %d" % len(PHASE_KILLS[args.phase]))
    print("  Backend     : %s%s" % (args.backend, " (" + str(CLAUDE_BIN) + ")" if args.backend == "cli" else ""))
    print()

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

    review = None
    delays = [0, 20, 60, 150, 300, 600]
    for i, wait in enumerate(delays):
        if wait:
            print("  backend busy; retry %d/%d in %ds..." % (i, len(delays) - 1, wait))
            time.sleep(wait)
        try:
            if args.backend == "cli":
                review = infer_cli(prompt, SYSTEM,
                                   # was hardcoded 600s; the http path already
                                   # honours AGNI_TIMEOUT, so the cli path should
                                   # too. Large bundles exceed 10 min on Opus.
                                   timeout=int(os.environ.get("AGNI_CLI_TIMEOUT", "600")),
                                   add_dirs=grant_dirs,
                                   model=args.model)
            else:
                # backend/limits made configurable so the SAME prompt can be
                # sent to a frontier model and a local one for comparison.
                # Reasoning models need a far larger budget: their analysis is
                # emitted as thinking tokens before any answer text appears.
                review = llm.infer(
                    prompt=prompt, system=SYSTEM, model=args.model,
                    max_tokens=int(os.environ.get("AGNI_MAX_TOKENS", "6000")),
                    temperature=0.2,
                    timeout=int(os.environ.get("AGNI_TIMEOUT", "180")),
                    backend=os.environ.get("AGNI_LLM_BACKEND", "anthropic"),
                )
        except Exception as e:
            print("  backend error: %s: %s" % (e.__class__.__name__, str(e)[:200]))
            review = None
        if review and review.strip():
            if i:
                print("  succeeded after %d retries" % i)
            break

    backend_used = args.model
    if (not review or not review.strip()) and args.allow_local_fallback:
        print("  falling back to LOCAL model -- review will be marked DEGRADED")
        try:
            review = llm.infer(
                prompt=prompt,
                system="You are Project Agni, a hostile methodological reviewer.",
                model="qwen3:30b-a3b", max_tokens=6000, temperature=0.2,
                timeout=300, backend="ollama",
            )
            backend_used = "qwen3:30b-a3b (DEGRADED — misreads geometry, do not "
            "trust for high-stakes shape claims)"
        except Exception as e:
            print("  local fallback failed: %s" % e.__class__.__name__)

    # B2: a gate must never let "no output" read as "no findings."
    if not review or not review.strip():
        print("ERROR: no review after %d attempts -- FAILING LOUD (exit 2)" % len(delays),
              file=sys.stderr)
        print("       This is NOT a pass. The phase is UNGATED.", file=sys.stderr)
        sys.exit(2)

    verdict = "UNPARSED"
    # Scan the WHOLE review, not the first 15 lines: thinking models emit
    # reasoning before the review header, which pushed "## VERDICT REJECTED"
    # out of range and recorded a rejection as UNPARSED (2026-08-30).
    for line in review.splitlines():
        for v in ("APPROVED", "CONDITIONAL", "REJECTED"):
            if v in line.upper():
                verdict = v
                break
        if verdict != "UNPARSED":
            break

    # A non-empty review whose verdict cannot be classified is NOT a pass.
    # The guard above only covers EMPTY output; without this, an
    # unclassifiable REJECTED exits 0 and reads downstream as no finding.
    if verdict == "UNPARSED":
        print("ERROR: review produced no parseable verdict -- FAILING LOUD (exit 2)",
              file=sys.stderr)
        print("       This is NOT a pass. Inspect the review text by hand.",
              file=sys.stderr)
        Path(args.out).write_text(json.dumps(
            {"phase": args.phase, "model": args.model, "verdict": "UNPARSED",
             "review": review}, indent=2),
            encoding="utf-8") if args.out else None
        sys.exit(2)

    # PERSIST BEFORE DISPLAY. A completed review is the deliverable; the console
    # is a convenience. This block used to sit BELOW the prints, so a cp1252
    # encoding failure on a checkmark destroyed a finished Opus review
    # (2026-08-30). Nothing that only affects rendering may discard the payload.
    out = Path(args.out) if args.out else Path(args.primary).with_suffix(
        ".agni_%s.json" % args.phase)
    out.write_text(json.dumps({
        "version": AGNI_PHASES_VERSION,
        "phase": args.phase,
        "model": backend_used,
        "primary": str(args.primary),
        "secondary": str(args.secondary) if args.secondary else None,
        "verdict": verdict,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "known_kills_checked": len(PHASE_KILLS[args.phase]),
        "review": review,
    }, indent=2), encoding="utf-8")

    print("=" * 70)
    print("AGNI %s REVIEW" % args.phase.upper())
    print("=" * 70)
    print(review)
    print("=" * 70)
    print("VERDICT: %s" % verdict)

    print("Saved: %s" % out)

    return 1 if verdict == "REJECTED" else 0


if __name__ == "__main__":
    sys.exit(main())
