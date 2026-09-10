#!/usr/bin/env python3
"""Summarize the Agni STYLE survey: verdict + severity counts per paper,
and dump every CRITICAL / MAJOR finding heading.

Rebuilt from the per-paper JSONs, NOT from the run log -- the four parallel
workers share one log file and race each other's writes, so the log is not
authoritative. The JSON artifacts are.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "style"

PAPERS = [
    "cache-tracing", "decision-state-paper", "delta-manifold-paper",
    "emotion-accumulation-paper", "emotional-trajectory-paper", "formulary-paper",
    "graph-topology-paper", "identity-geometry", "kv-cloak-defense-paper",
    "lyra-technique-ii", "meta-pattern", "mine5-selective-sharpener",
    "null-swarm-paper", "oracle-loop-paper", "presence-metric",
    "spectral-shape-paper", "waystations-paper",
]

# Reviews come in TWO layouts and a parser that knows only one reports zero
# findings on the other -- which reads as "clean" on a REJECTED paper. Both
# formulary-paper and lyra-technique-ii were scored 0/0/0 by the first version
# of this parser while their reviews carried multiple CRITICALs.
#
#   layout A (inline):  "## 3. CRITICAL - <title>"
#   layout B (grouped): "## CRITICAL"  then  "**1. <title>**", "**2. ...**"
INLINE = re.compile(r"^\s*(?:#{1,4}\s*)?\**\s*(\d+)[.)]\s*\**\s*[^\n]*?"
                    r"\b(CRITICAL|MAJOR|MINOR)\b[^\n]*", re.M)
SEV_HEADER = re.compile(r"^\s*(?:#{1,4}\s*|\*\*)\s*(CRITICAL|MAJOR|MINOR)"
                        r"(?:\s*(?:FINDINGS?|ISSUES?))?\s*\**\s*:?\s*$", re.M | re.I)
NUMBERED = re.compile(r"^\s*(?:#{1,4}\s*)?\**\s*(\d+)[.)]\s+(.{0,200})", re.M)


def load(name):
    p = OUT / ("%s.agni_style.json" % name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"verdict": "UNPARSEABLE_JSON", "review": ""}


def findings(review):
    """Return list of (severity, heading_text) for numbered findings.
    Handles both layouts; de-duplicates by (severity, title) so a finding that
    matches under both is counted once."""
    out = []
    seen = set()

    for m in INLINE.finditer(review):
        key = (m.group(2), m.group(1), m.start() // 400)
        if key not in seen:
            seen.add(key)
            out.append((m.group(2), m.group(0).strip()))

    # layout B: severity section header, then numbered items until the next
    # severity header or end of review
    hdrs = [(m.start(), m.end(), m.group(1).upper()) for m in SEV_HEADER.finditer(review)]
    for i, (s, e, sev) in enumerate(hdrs):
        end = hdrs[i + 1][0] if i + 1 < len(hdrs) else len(review)
        block = review[e:end]
        for n in NUMBERED.finditer(block):
            title = " ".join(n.group(0).split())[:200]
            # skip if an inline match already claimed this position
            abspos = e + n.start()
            key = (sev, n.group(1), abspos // 400)
            if key in seen:
                continue
            if any(abs(abspos - review.find(h, 0)) < 5 for _, h in out):
                continue
            seen.add(key)
            out.append((sev, title))
    return out


def main():
    rows = []
    details = []
    for name in PAPERS:
        d = load(name)
        if d is None:
            rows.append((name, "NO OUTPUT - GATE DID NOT RUN", 0, 0, 0, 0))
            continue
        review = d.get("review") or ""
        if not review.strip():
            rows.append((name, "EMPTY REVIEW - FAIL", 0, 0, 0, 0))
            continue
        f = findings(review)
        c = sum(1 for s, _ in f if s == "CRITICAL")
        j = sum(1 for s, _ in f if s == "MAJOR")
        n = sum(1 for s, _ in f if s == "MINOR")
        rows.append((name, d.get("verdict", "?"), c, j, n, len(f)))
        for sev, head in f:
            if sev in ("CRITICAL", "MAJOR"):
                details.append((name, sev, head))

    print("%-30s %-12s %4s %4s %4s %5s" % ("PAPER", "VERDICT", "CRIT", "MAJ", "MIN", "TOT"))
    print("-" * 68)
    for r in rows:
        print("%-30s %-12s %4s %4s %4s %5s" % r)

    print()
    print("=" * 68)
    print("CRITICAL AND MAJOR FINDING HEADINGS")
    print("=" * 68)
    cur = None
    for name, sev, head in details:
        if name != cur:
            print("\n--- %s ---" % name)
            cur = name
        print("  [%s] %s" % (sev, head[:300].replace("\n", " ")))


if __name__ == "__main__":
    main()
