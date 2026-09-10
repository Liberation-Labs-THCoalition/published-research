#!/usr/bin/env python3
"""Build-integrity gate for PAPER_PIPELINE step 7 (Zenodo).

WHY THIS EXISTS
---------------
On 2026-09-05 I rebuilt two papers, checked `pdflatex` exit code 0, extracted the
PDF text, confirmed my new sentences were present, and reported both as verified.
bibtex had never run. Neither PDF had a References section; every citation
rendered as `[?]`. The PDF was *newer* than the source and still broken.

The check I ran confirmed that my change was present. It could not detect what I
had broken. It failed toward reassurance, which is the direction nobody audits.

The pipeline's step 7 says "compile both PDFs". Compiling is not the check.
This is the check.

DESIGN RULES, from the failure above
------------------------------------
1. Every check must be able to FAIL. `--selftest` deliberately breaks a build and
   asserts this gate catches it. A gate nobody has watched fail is not a gate.
2. A check that CANNOT RUN reports FAILURE, never success. A missing `pdftotext`
   is an unknown, and an unknown is not a pass.
3. Failures name the remedy, not just the symptom.

USAGE
-----
    python build_integrity_gate.py PAPER_DIR [--tex main.tex] [--expect expectations.json]
    python build_integrity_gate.py --selftest PAPER_DIR

Exit code 0 = shippable. Non-zero = do not publish.

expectations.json (all keys optional):
    {"required_strings": ["body count is 6 confirmed"],
     "forbidden_strings": ["AUROC 1.000 across seven"],
     "min_pages": 10}
`forbidden_strings` is the travelled-caveat guard: retracted numbers must not
appear in a shipped artifact.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RESET, RED, GRN, YEL, DIM = "\033[0m", "\033[31m", "\033[32m", "\033[33m", "\033[2m"


class Result:
    def __init__(self):
        self.rows = []

    def add(self, ok, name, detail, remedy=""):
        """ok=True pass, ok=False fail, ok=None UNAVAILABLE (neither).

        The third state exists because reporting an un-runnable check as PASS
        inflates the denominator and reads as verification that did not happen.
        That is this gate's own Rule 2, and the source-less mode violated it on
        the day it was written.
        """
        self.rows.append((ok, name, detail, remedy))

    @property
    def failed(self):
        return [r for r in self.rows if r[0] is False]

    @property
    def unavailable(self):
        return [r for r in self.rows if r[0] is None]

    def report(self):
        for ok, name, detail, remedy in self.rows:
            mark = (f"{GRN}PASS{RESET}" if ok is True
                    else f"{RED}FAIL{RESET}" if ok is False
                    else f"{YEL}N/A {RESET}")
            print(f"  [{mark}] {name}: {detail}")
            if ok is False and remedy:
                print(f"         {YEL}remedy:{RESET} {remedy}")
        bad = len(self.failed)
        na = len(self.unavailable)
        runnable = len(self.rows) - na
        print()
        if bad:
            print(f"  {RED}GATE FAILED{RESET} — {bad} of {runnable} runnable checks failed. "
                  f"DO NOT PUBLISH.")
        else:
            print(f"  {GRN}GATE PASSED{RESET} — {runnable}/{runnable} runnable checks.")
        if na:
            print(f"  {YEL}{na} check(s) UNAVAILABLE{RESET} and counted as neither pass nor "
                  f"fail. Coverage is partial.")
        return 1 if bad else 0


def normalize(s: str) -> str:
    """Fold typographic variants so a needle cannot miss on punctuation alone.

    N2 (2026-09-05): the needle `7/7--0/10` COULD NOT FIRE. The rendered PDF
    extracts an en-dash, `7/7-0/10`, and the two-hyphen needle missed it. A
    forbidden-string check that cannot match is a check that cannot fail, which
    is the exact defect this gate exists to catch — built into the gate itself.
    """
    for a, b in (("–", "--"), ("—", "---"), ("−", "-"),
                 ("‘", "'"), ("’", "'"),
                 ("“", '"'), ("”", '"'), (" ", " ")):
        s = s.replace(a, b)
    return " ".join(s.split())


def run(cmd, cwd, timeout=300):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except FileNotFoundError:
        return -2, "NOT FOUND"


def full_build(paper_dir: Path, stem: str, res: Result, skip_bibtex=False):
    """pdflatex -> bibtex -> pdflatex -> pdflatex. Records whether bibtex ran."""
    if not shutil.which("pdflatex"):
        res.add(False, "toolchain", "pdflatex not on PATH",
                "install a LaTeX distribution; this check cannot run without it")
        return None
    rc1, _ = run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], paper_dir)

    if skip_bibtex:
        res.add(True, "bibtex", f"{DIM}deliberately skipped (selftest){RESET}", "")
    else:
        rcb, outb = run(["bibtex", stem], paper_dir, timeout=200)
        blg = paper_dir / f"{stem}.blg"
        blg_txt = blg.read_text(encoding="utf-8", errors="replace") if blg.exists() else ""
        errs = len(re.findall(r"^I couldn't|error message|Warning--", blg_txt, re.M))
        res.add(rcb == 0, "bibtex ran", f"exit {rcb}, {errs} log warnings",
                "run `bibtex <stem>` between pdflatex passes")

    run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], paper_dir)
    rc3, _ = run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], paper_dir)
    res.add(rc3 == 0, "final pdflatex pass", f"exit {rc3}",
            "read the .log; the last pass must complete")
    return paper_dir / f"{stem}.log"


def check_log(logfile: Path, res: Result):
    if not logfile or not logfile.exists():
        res.add(False, "log readable", f"missing {logfile}",
                "the build did not produce a log — treat as failed, not as unknown")
        return
    txt = logfile.read_text(encoding="utf-8", errors="replace")
    cit = len(re.findall(r"Citation [`'\"].*?[`'\"] .*undefined", txt))
    ref = len(re.findall(r"Reference [`'\"].*?[`'\"] .*undefined", txt))
    res.add(cit == 0, "undefined citations", f"{cit} found",
            "bibtex did not run, or a \\cite key is missing from the .bib")
    res.add(ref == 0, "undefined references", f"{ref} found",
            "a \\label is missing, or another pdflatex pass is needed")


def check_pdf(pdf: Path, res: Result, expect: dict):
    if not pdf.exists():
        res.add(False, "pdf exists", f"missing {pdf.name}", "the build produced no PDF")
        return
    if not shutil.which("pdftotext"):
        # RULE 2: a check that cannot run is a FAILURE, never a silent pass.
        res.add(False, "rendered-text checks", "pdftotext unavailable — CANNOT VERIFY",
                "install poppler-utils; an unverifiable artifact must not be published")
        return
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "t.txt"
        rc, _ = run(["pdftotext", str(pdf), str(out)], pdf.parent)
        if rc != 0 or not out.exists():
            res.add(False, "pdf text extraction", f"pdftotext exit {rc}",
                    "PDF may be corrupt")
            return
        txt = out.read_text(encoding="utf-8", errors="replace")

    has_refs = bool(re.search(r"^\s*(References|Bibliography)\s*$", txt, re.M))
    res.add(has_refs, "References section rendered", "present" if has_refs else "ABSENT",
            "bibtex did not run — this is the exact 2026-09-05 failure")

    # NOTE: the first version of this regex was `\[\?\]` and it MISSED the real
    # 2026-09-05 failure, which rendered as `framework [? ]` — with a space. The
    # selftest caught that. Whitespace inside the brackets is the normal LaTeX
    # rendering of an undefined \cite, so it must be tolerated here.
    ph = (len(re.findall(r"\[\s*\?\s*\]", txt))
          + len(re.findall(r"\(\s*\?\s*\)", txt))
          + len(re.findall(r"(?<![\w?])\?\?(?![\w?])", txt)))
    res.add(ph == 0, "no unresolved placeholders", f"{ph} '[?]'/'??' in rendered text",
            "undefined citations or refs are reaching the reader")

    # --- de-escape artifacts: a UNIVERSAL check, added 2026-09-06 -----------
    # The corpus audit found three papers that PASSED this gate 5/5 while
    # rendering `extttmp_norm_per_token` to readers. A literal TAB replacing the
    # backslash in 	exttt compiles silently and only shows in rendered text.
    # The old needle system could not catch it: required/forbidden strings only
    # fire if someone thought to specify them, and nobody specifies `exttt`.
    # These signatures need no per-paper configuration -- they are what an eaten
    # backslash leaves behind, for any paper, always.
    # Needles for RENDERED text carry NO braces -- LaTeX consumes them, so a
    # broken 	exttt{foo} renders as `extttfoo`, not `exttt{foo}`. My first
    # version used braces and could never fire.
    # Every needle below is a string that CANNOT occur in ordinary English.
    # Deliberately excluded because they DO occur and would cry wolf:
    #   "imes"  -> inside "times"      "ootnote" -> inside "footnote"
    #   "egin"  -> inside "begin"      "ext"     -> inside "text", "next"
    # A needle that fires on correct prose is worse than no needle: it trains
    # the reader to ignore the gate.
    DEESCAPE = ("exttt", "extbf", "extit", "extsc", "extrm", "extsl",
                "extsuperscript", "extsubscript", "extnormal")
    hits = [(s, txt.count(s)) for s in DEESCAPE if s in txt]
    res.add(not hits, "no de-escaped LaTeX commands",
            "clean" if not hits else "; ".join(f"{s!r}x{n}" for s, n in hits),
            "a control character replaced a backslash in the source (TAB for "
            "\t, formfeed for \f). Compiles silently and is invisible to "
            "brace-balance checks. Byte-scan the .tex for 0x09/0x0C/0x07 -- but "
            "note a de-escaped \n is an ordinary newline, invisible to a byte "
            "scan, so trust this rendered-text check over a source grep.")



    pages = txt.count("\f") or 1
    minp = expect.get("min_pages")
    if minp:
        res.add(pages >= minp, "page count", f"{pages} (min {minp})",
                "document is shorter than expected — truncated build?")

    ntxt = normalize(txt)
    for s in expect.get("required_strings", []):
        ok = normalize(s) in ntxt
        res.add(ok, f"required: {s[:52]!r}", "present" if ok else "MISSING",
                "the intended correction did not reach the rendered artifact")
    for s in expect.get("forbidden_strings", []):
        ok = normalize(s) not in ntxt
        res.add(ok, f"forbidden: {s[:52]!r}",
                "absent" if ok else "PRESENT IN SHIPPED PDF",
                "a retracted or withdrawn claim is still visible to a reader")


def check_freshness(pdf: Path, tex: Path, res: Result, built_here: bool):
    """Only meaningful in --no-build mode.

    N3 (2026-09-05): when the gate rebuilds first, this check CANNOT FAIL —
    the PDF is newer by construction. It was reported as a PASS on every run,
    which is worse than useless: it looked like evidence.
    """
    if built_here:
        res.add(True, "pdf freshness",
                f"{DIM}not applicable — gate rebuilt the PDF; use --no-build to "
                f"gate the shipped artifact{RESET}", "")
        return
    if not pdf.exists() or not tex.exists():
        res.add(False, "freshness", "pdf or tex missing", "cannot compare timestamps")
        return
    ok = pdf.stat().st_mtime >= tex.stat().st_mtime
    res.add(ok, "pdf newer than source", "yes" if ok else "PDF IS STALE",
            "rebuild — necessary, not sufficient: a fresh PDF can still be broken")


def gate(paper_dir: Path, stem: str, expect: dict, skip_bibtex=False,
         no_build=False) -> int:
    res = Result()
    mode = "verifying SHIPPED artifact (no rebuild)" if no_build else "rebuild + verify"
    print("")
    print(f"  Build-integrity gate -- {paper_dir}/{stem}.tex  [{mode}]")
    print("")
    pdf = paper_dir / f"{stem}.pdf"
    if no_build:
        # N3 (2026-09-05): gating a PDF the gate itself just built verifies the
        # gate's rebuild, not the artifact that ships. This mode audits what ships.
        log = paper_dir / f"{stem}.log"
        if not log.exists():
            res.add(False, "build log present", f"no {stem}.log beside the PDF",
                    "cannot audit a shipped PDF without its build log")
            log = None
    else:
        log = full_build(paper_dir, stem, res, skip_bibtex=skip_bibtex)
    check_log(log, res)
    check_pdf(pdf, res, expect)
    check_freshness(pdf, paper_dir / f"{stem}.tex", res, built_here=not no_build)
    return res.report()


def selftest(paper_dir: Path, stem: str) -> int:
    """Positive control: break a build on purpose, assert the gate catches it.

    Copies the paper to a temp dir and builds WITHOUT bibtex — reproducing the
    2026-09-05 failure exactly. If the gate passes that, the gate is worthless.
    """
    print(f"\n  {YEL}SELFTEST{RESET} — reproducing the 2026-09-05 failure "
          f"(build with no bibtex) and asserting this gate catches it.\n")
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / paper_dir.name
        shutil.copytree(paper_dir, dst, ignore=shutil.ignore_patterns("*.pdf"))
        for aux in dst.glob("*.bbl"):
            aux.unlink()          # remove any prebuilt bibliography
        rc = gate(dst, stem, {}, skip_bibtex=True)
    print()
    if rc != 0:
        print(f"  {GRN}SELFTEST PASSED{RESET} — the gate fired on a deliberately "
              f"broken build. It can fail, so a pass means something.")
        return 0
    print(f"  {RED}SELFTEST FAILED{RESET} — the gate reported success on a build "
          f"with no bibliography. It cannot fail and must not be trusted.")
    return 1


def gate_pdf_only(pdf: Path, expect: dict) -> int:
    """Audit a DISTRIBUTION PDF that has no source beside it.

    THE BLIND SPOT this closes (found by the 2026-09-06 corpus audit): the gate
    paired a PDF to a same-stem .tex, so every publication-named file at a repo
    root -- the ones a reader actually receives -- was never checked at all.
    One of them ships 18 unresolved citations as "[? ]".

    A scope defined by "what happens to have a sibling source" is a scope chosen
    by convenience. Checks that cannot run here are reported UNAVAILABLE, never
    silently omitted, and the verdict says the audit was partial.
    """
    res = Result()
    print("")
    print(f"  Build-integrity gate -- {pdf.name}  [SOURCE-LESS: distribution artifact]")
    print("")
    check_pdf(pdf, res, expect)
    for name in ("bibtex ran", "final pdflatex pass", "undefined citations",
                 "undefined references", "pdf freshness"):
        res.add(None, name, "no source or build log beside this PDF", "")
    rc = res.report()
    print(f"  {YEL}PARTIAL AUDIT{RESET}: rendered-text checks only. A pass here means the "
          f"page a reader sees is clean;")
    print(f"  it does NOT mean the build was sound. Locate the source to audit that.")
    return rc


def coverage(root: Path) -> int:
    """Report what this gate WOULD and would NOT check under a directory.

    This is the real fix for the blind spot. A gate that silently decides its
    own scope cannot be audited; one that reports its scope can. Run this before
    trusting any claim of the form "the corpus is clean".
    """
    pdfs = sorted(p for p in root.rglob("*.pdf") if ".git" not in p.parts)
    paired, orphan = [], []
    for p in pdfs:
        if (p.with_suffix(".tex")).exists():
            paired.append(p)
        else:
            orphan.append(p)
    print("")
    print(f"  Coverage report -- {root}")
    print("")
    print(f"  PDFs found                     : {len(pdfs)}")
    print(f"  {GRN}gateable (same-stem .tex){RESET}      : {len(paired)}")
    print(f"  {YEL}source-less (--pdf only){RESET}       : {len(orphan)}")
    if orphan:
        print("")
        print("  Source-less artifacts -- these are what readers receive, and the")
        print("  directory-mode gate is BLIND to every one of them:")
        for p in orphan:
            log = "log present" if p.with_suffix(".log").exists() else "no log"
            print(f"    {p.relative_to(root)}   ({log})")
        print("")
        print(f"  Audit each with:  python {Path(__file__).name} --pdf <file> [--expect ...]")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper_dir", nargs="?", default=None)
    ap.add_argument("--tex", default="main.tex")
    ap.add_argument("--expect", default=None,
                    help="JSON with required_strings / forbidden_strings / min_pages")
    ap.add_argument("--no-build", action="store_true",
                    help="verify the PDF AS IT SITS instead of rebuilding it. This is the "
                         "pre-publication mode: gate what ships, not what the gate made.")
    ap.add_argument("--pdf", default=None,
                    help="audit a single distribution PDF that has no source beside it. "
                         "Rendered-text checks only; unavailable checks are reported.")
    ap.add_argument("--coverage", default=None,
                    help="report which PDFs under DIR this gate can and cannot check. "
                         "Run this before believing any 'the corpus is clean' claim.")
    ap.add_argument("--selftest", action="store_true",
                    help="break a build on purpose and assert this gate catches it")
    a = ap.parse_args()

    expect = json.loads(Path(a.expect).read_text(encoding="utf-8")) if a.expect else {}

    if a.coverage:
        return coverage(Path(a.coverage).resolve())
    if a.pdf:
        p = Path(a.pdf).resolve()
        if not p.exists():
            print(f"  {RED}ABORT{RESET}: {p} not found")
            return 2
        return gate_pdf_only(p, expect)
    if not a.paper_dir:
        ap.error("give a PAPER_DIR, or --pdf FILE, or --coverage DIR")

    d = Path(a.paper_dir).resolve()
    stem = Path(a.tex).stem
    if not (d / f"{stem}.tex").exists():
        print(f"  {RED}ABORT{RESET}: {d / (stem + '.tex')} not found")
        print(f"  {YEL}hint{RESET}: for a distribution PDF with no source, use --pdf")
        return 2
    if a.selftest:
        return selftest(d, stem)
    return gate(d, stem, expect, no_build=a.no_build)


if __name__ == "__main__":
    sys.exit(main())
