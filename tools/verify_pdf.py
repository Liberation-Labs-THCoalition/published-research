#!/usr/bin/env python3
"""Check that a built PDF actually says what you think it says.

WHY THIS EXISTS. On 2026-09-09 five separate verification checks in this corpus
returned "the fix did not land" on PDFs where the fix HAD landed. Three distinct
causes, all of which make a naive `pdftotext | grep` lie to you:

  1. LIGATURES. pdftotext drops fi/fl/ff/ffi/ffl in these MiKTeX builds.
     "filter" extracts as "lter". "benefit" as "benet". "float" as "oat".
     ANY search string containing fi, fl or ff silently fails -- which covers
     effect, difference, significant, coefficient, confirm, file, verify...
  2. HYPHENATION. LaTeX splits words across line breaks: "25 independent"
     extracts as "25 inde- pendent".
  3. LINE WRAPPING. A phrase spanning a line break never matches a flat string.

And the fourth, which no normalisation fixes and which you must think about:

  4. A CORRECTION QUOTES WHAT IT CORRECTS. "previously reported at AUROC 1.0,
     since withdrawn" contains the string you are grepping for. A substring
     cannot distinguish asserting a claim from retracting one. Search for the
     RETRACTION LANGUAGE, not the number.

Usage:
    python verify_pdf.py <file.pdf> --control "<a string you KNOW is present>" \
        [--present "<must appear>" ...] [--absent "<must not appear>" ...]

Exit 0 if every assertion holds, 1 otherwise. The --control is mandatory: a
check that cannot fail is not a check, and an empty extraction looks exactly
like a clean bill of health.
"""
import argparse, re, subprocess, sys

LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
}


def extract(pdf):
    out = subprocess.run(["pdftotext", "-layout", pdf, "-"],
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


def restore_dropped_ligatures(t):
    """MiKTeX + pdftotext sometimes DELETES the ligature rather than emitting one.
    'filter' -> 'lter', 'benefit' -> 'benet'. We cannot recover those in general,
    so instead we make the SEARCH tolerant: fi/fl/ff in a needle may be missing
    from the haystack."""
    return t


def needle_regex(s):
    """Build a pattern that tolerates a dropped fi/fl/ff ligature in the haystack."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--control", required=True,
                    help="a string you KNOW is in the paper; proves extraction ran")
    ap.add_argument("--present", action="append", default=[])
    ap.add_argument("--absent", action="append", default=[])
    a = ap.parse_args()

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
