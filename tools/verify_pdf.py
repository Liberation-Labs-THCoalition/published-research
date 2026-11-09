#!/usr/bin/env python3
"""Check that a built PDF actually says what you think it says.

WHY THIS EXISTS. On 2026-09-09 five separate verification checks in this corpus
returned "the fix did not land" on PDFs where the fix HAD landed. Three distinct
causes, all of which make a naive `pdftotext | grep` lie to you:

  1. LIGATURES. In these MiKTeX builds pdftotext (4.00) does NOT drop the
     ff/fi/fl/ffi/ffl ligatures: it emits each as ONE CONTROL BYTE, its slot
     in the T1 font encoding (0x1B ff, 0x1C fi, 0x1D fl, 0x1E ffi, 0x1F ffl).
     A terminal does not display those bytes, so "filter" LOOKS like "lter"
     and "We offer" like "We oer"; and Python's \\s counts 0x1C-0x1F as
     whitespace. ANY search string containing fi, fl or ff silently fails --
     which covers effect, difference, significant, coefficient, confirm, file,
     verify... normalise() maps the bytes back to letters. (Until 2026-11-08
     this tool assumed they were dropped and never matched the byte, so a
     ligature needle was never found: --present always failed and --absent
     always passed. --self-test now guards that.)
  2. HYPHENATION. LaTeX splits words across line breaks: "25 independent"
     extracts as "25 inde- pendent".
  3. LINE WRAPPING. A phrase spanning a line break never matches a flat string.

And the fourth, which no normalisation fixes and which you must think about:

  4. A CORRECTION QUOTES WHAT IT CORRECTS. "previously reported at AUROC 1.0,
     since withdrawn" contains the string you are grepping for. A substring
     cannot distinguish asserting a claim from retracting one. Search for the
     RETRACTION LANGUAGE, not the number.

Usage:
    python verify_pdf.py <file.pdf> --control "<a string you KNOW is present>" \\
        [--present "<must appear>" ...] [--absent "<must not appear>" ...]
    python verify_pdf.py --self-test

Exit 0 if every assertion holds, 1 otherwise. The --control is mandatory: a
check that cannot fail is not a check, and an empty extraction looks exactly
like a clean bill of health.
"""
import argparse, re, subprocess, sys

LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
    # T1 font slots that pdftotext 4.00 passes through as control bytes. They
    # must be mapped BEFORE whitespace is collapsed: \s matches 0x1C-0x1F.
    "\x1b": "ff", "\x1c": "fi", "\x1d": "fl", "\x1e": "ffi", "\x1f": "ffl",
}


def extract(pdf):
    out = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", pdf, "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        sys.exit("pdftotext failed on %s: %s" % (pdf, out.stderr.strip()[:200]))
    return out.stdout


def normalise(t):
    for lig, plain in LIGATURES.items():
        t = t.replace(lig, plain)
    t = t.replace("\r", "")
    t = re.sub(r"-\s*\n\s*", "", t)   # undo hyphenation across line breaks
    t = re.sub(r"\s+", " ", t)        # collapse wrapping
    return t


def needle_regex(s):
    """Build a pattern that also tolerates a fi/fl/ff ligature that some other
    pdftotext build really drops (these builds emit a control byte instead,
    which normalise() maps back)."""
    parts, i = [], 0
    while i < len(s):
        two = s[i:i + 2]
        if two in ("fi", "fl", "ff"):
            parts.append("(?:%s)?" % re.escape(two))   # present or dropped
            i += 2
        else:
            parts.append(re.escape(s[i]))
            i += 1
    return re.compile("".join(parts), re.I)


def self_test():
    """Synthetic extraction in the form pdftotext 4.00 gives on these builds."""
    sample = ("We o\x1ber a signi\x1ccant e\x1bect; the \x1doat is su\x1ecient,\n"
              "not ba\x1fing, and the hyphen-\n  ated ﬁlter holds.")
    t = normalise(sample)
    cases = [("We offer", True), ("significant", True), ("effect", True),
             ("float", True), ("sufficient", True), ("baffling", True),
             ("hyphenated filter", True), ("insignificant", False)]
    ok = True
    for needle, want in cases:
        got = bool(needle_regex(needle).search(t))
        ok &= got == want
        print("  %s self-test %-8s %r" % ("OK  " if got == want else "**FAIL**",
                                          "present" if want else "absent", needle))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--control",
                    help="a string you KNOW is in the paper; proves extraction ran (mandatory)")
    ap.add_argument("--present", action="append", default=[])
    ap.add_argument("--absent", action="append", default=[])
    ap.add_argument("--self-test", action="store_true",
                    help="check the normalisation on a synthetic extraction and exit")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.pdf or not a.control:
        ap.error("a PDF and --control are required: a check that cannot fail is not a check")

    text = normalise(extract(a.pdf))
    if not text.strip():
        sys.exit("ABORT: extracted no text from %s" % a.pdf)

    ok = True
    ctl = needle_regex(a.control).search(text)
    print("  %s control  %r" % ("OK  " if ctl else "**FAIL**", a.control))
    if not ctl:
        print("  extraction produced %d chars but the control is missing --" % len(text))
        print("  every result below is meaningless. Fix the control first.")
        sys.exit(1)

    for s in a.present:
        hit = needle_regex(s).search(text)
        ok &= bool(hit)
        print("  %s present  %r" % ("OK  " if hit else "**FAIL**", s))
    for s in a.absent:
        hit = needle_regex(s).search(text)
        ok &= not hit
        print("  %s absent   %r" % ("OK  " if not hit else "**FAIL**", s))

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
