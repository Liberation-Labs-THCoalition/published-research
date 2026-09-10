#!/usr/bin/env python3
"""TWIN_DESYNC structural check: is any table row filed under a DIFFERENT
heading in the flight edition than in the academic twin?

WHY THE OBVIOUS VERSION IS USELESS
  The defect that shipped here (closed 2026-09-01) was a retracted finding
  sitting under the group label "Confirmed" in the venue-facing edition while
  the other edition filed it under "Falsified." Those group labels are
  \\multicolumn rows INSIDE the tabular -- they are not \\section headings and
  they contain no "&". A checker that tracks only \\section, or that keys rows
  by splitting on "&", is structurally blind to the exact defect it exists to
  catch. The first version of this file was blind that way and reported all 17
  papers "clean"; the positive control below is what caught it.

CONTEXT tracked per row, in document order:
  (section, table caption/label, nearest preceding in-table group label)

Run --selftest to prove the checker can fail before believing a clean result.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

PAPERS = ("cache-tracing decision-state-paper delta-manifold-paper "
          "emotion-accumulation-paper emotional-trajectory-paper formulary-paper "
          "graph-topology-paper identity-geometry kv-cloak-defense-paper "
          "lyra-technique-ii meta-pattern mine5-selective-sharpener "
          "null-swarm-paper oracle-loop-paper presence-metric "
          "spectral-shape-paper waystations-paper").split()

SEC = re.compile(r"\\(?:sub)*section\*?\{([^}]*)\}")
BEGIN_TAB = re.compile(r"\\begin\{(tabular|tabularx|longtable)\}")
END_TAB = re.compile(r"\\end\{(tabular|tabularx|longtable)\}")
BEGIN_TABLE = re.compile(r"\\begin\{table\*?\}")
END_TABLE = re.compile(r"\\end\{table\*?\}")
# Line-scoped, non-greedy. With re.S + greedy this matched across the whole
# document and found almost nothing -- which silently disabled the group-label
# tracking that is the entire point of this file.
MULTICOL = re.compile(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.+?)\}\s*\\\\\s*$")
RULE = re.compile(r"^\s*\\(top|mid|bottom|cmid)rule|^\s*\\hline")


def clean(s):
    s = re.sub(r"\\(text(bf|it|tt|sc)|emph|mathbf|checkmark|times|footnotesize)\b", " ", s)
    s = re.sub(r"[\\{}$&%~^_]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def collect(texfile):
    """-> dict: row key -> (section, caption, group_label)"""
    root = texfile.parent
    body = texfile.read_text(encoding="utf-8", errors="replace")

    def expand(t, depth=0):
        if depth > 3:
            return t
        def sub(m):
            name = m.group(1)
            f = root / (name if name.endswith(".tex") else name + ".tex")
            if f.exists():
                return expand(f.read_text(encoding="utf-8", errors="replace"), depth + 1)
            return ""
        return re.sub(r"\\(?:input|include)\{([^}]*)\}", sub, t)
    body = expand(body)

    rows = {}
    section = "(none)"
    caption = "(none)"
    group = "(none)"
    in_tab = 0
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("%"):
            continue
        m = SEC.search(line)
        if m and not in_tab:
            section = clean(m.group(1))
        if BEGIN_TABLE.search(line):
            caption = "(none)"
        if "\\caption" in line:
            c = re.search(r"\\caption\{(.*)", line)
            if c:
                caption = clean(c.group(1))[:80]
        if "\\label{" in line and caption == "(none)":
            c = re.search(r"\\label\{([^}]*)\}", line)
            if c:
                caption = clean(c.group(1))[:80]
        if BEGIN_TAB.search(line):
            in_tab += 1
            group = "(none)"
            continue
        if END_TAB.search(line):
            in_tab = max(0, in_tab - 1)
            group = "(none)"
            continue
        if not in_tab or RULE.match(s):
            continue
        # In-table group label: a \multicolumn row spanning the table. This is
        # the "Confirmed" / "Falsified" divider the historical defect lived on.
        mc = MULTICOL.search(s)
        if mc and "&" not in s.split("\\multicolumn")[0]:
            group = clean(mc.group(1))[:80]
            continue
        if "&" not in s:
            continue
        cells = [clean(c) for c in s.split("&")]
        key = ""
        for c in cells:
            # numeric / single-char first columns must not veto the row
            if len(c) >= 4 and not c.startswith("hline"):
                key = c
                break
        if key:
            rows.setdefault(key, (section, caption, group))
    return rows


def compare(paper, root=ROOT, quiet=False):
    f = root / paper / "main.tex"
    t = root / paper / "academic" / "main.tex"
    if not (f.exists() and t.exists()):
        if not quiet:
            print("%-30s SKIP (missing edition)" % paper)
        return None
    rf, rt = collect(f), collect(t)
    shared = sorted(set(rf) & set(rt))
    moved = [(k, rf[k], rt[k]) for k in shared if rf[k] != rt[k]]
    only_f = sorted(set(rf) - set(rt))
    only_t = sorted(set(rt) - set(rf))
    if not quiet:
        status = "clean" if not moved else "%d ROW(S) IN DIFFERENT CONTEXT" % len(moved)
        print("%-30s rows f=%-4d a=%-4d shared=%-4d  %s" % (
            paper, len(rf), len(rt), len(shared), status))
        for k, a, b in moved:
            print("    ROW %r" % k[:60])
            print("        flight  : sec=%r cap=%r group=%r" % a)
            print("        academic: sec=%r cap=%r group=%r" % b)
        if only_f or only_t:
            print("    ROWS ONLY IN FLIGHT (%d): %s" % (len(only_f), [x[:45] for x in only_f[:8]]))
            print("    ROWS ONLY IN TWIN   (%d): %s" % (len(only_t), [x[:45] for x in only_t[:8]]))
    return moved, only_f, only_t


def selftest():
    """Move a row under a different in-table GROUP LABEL in a copy of the twin
    and confirm the checker reports it."""
    target = "meta-pattern"
    tmp = Path(tempfile.mkdtemp())
    pr = tmp / "pr"
    pr.mkdir()
    shutil.copytree(ROOT / target, pr / target)
    tw = pr / target / "academic" / "main.tex"
    t = tw.read_text(encoding="utf-8", errors="replace")
    # find two group labels and a data row under the first; move it under the second
    lines = t.splitlines()
    label_idx = [i for i, ln in enumerate(lines) if MULTICOL.search(ln.strip())]
    if len(label_idx) < 2:
        print("SELFTEST: cannot set up (need 2 group labels, found %d)" % len(label_idx))
        return 1
    # first real data row under the FIRST group label
    row_i = None
    for i in range(label_idx[0] + 1, label_idx[1]):
        s = lines[i].strip()
        if "&" in s and not RULE.match(s) and "multicolumn" not in s:
            row_i = i
            break
    if row_i is None:
        print("SELFTEST: cannot set up (no data row under first label)")
        return 1
    print("SELFTEST moving row: %s" % lines[row_i].strip()[:70])
    print("        from group : %s" % lines[label_idx[0]].strip()[:70])
    print("        to   group : %s" % lines[label_idx[1]].strip()[:70])
    row = lines.pop(row_i)
    # label_idx[1] shifted down by one after the pop
    lines.insert(label_idx[1], row)
    tw.write_text("\n".join(lines), encoding="utf-8")
    res = compare(target, root=pr, quiet=True)
    moved = res[0] if res else []
    ok = bool(moved)
    if moved:
        for k, a, b in moved:
            print("  detected: %r  %r -> %r" % (k[:50], a[2], b[2]))
    print("POSITIVE CONTROL: %s" % (
        "PASS - checker detects a row moved between group labels" if ok
        else "FAIL - checker is blind; a clean result means nothing"))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 2


def main():
    if "--selftest" in sys.argv:
        return selftest()
    hits = 0
    for p in PAPERS:
        r = compare(p)
        if r and r[0]:
            hits += 1
    print("\nPAPERS WITH ROWS IN A DIFFERENT CONTEXT: %d" % hits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
