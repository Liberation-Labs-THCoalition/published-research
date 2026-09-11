# -*- coding: utf-8 -*-
"""Convert a Markdown-sourced paper to LaTeX in this repo's conventions.

WHY THIS EXISTS. Two papers in the corpus (mnemosyne-benchmark, mnemosyne-ablation's
publication draft) are Markdown-only. The build gate can only verify PDFs it can
rebuild from source, so a Markdown paper is permanently UNVERIFIABLE and its shipped
PDF has no checkable provenance. pandoc is not installed on this machine.

It also enforces the edition split: the academic edition must NOT carry first-person
reflections, and must carry a human-only byline with an AI disclosure. Doing that by
hand is how an integrity edition ends up on a human-only-creators DOI, which is
exactly the defect this was written to fix.

Usage:
  python tools/md2tex.py <in.md> <out.tex> --edition academic --author "Thomas Edrington"
                         --ai "Nexus, Lyra" [--drop-section "First-Person Reflection"]
  python tools/md2tex.py <in.md> <out.tex> --edition integrity

Scope: headings, bold/italic, pipe tables, bullet and numbered lists, horizontal
rules, inline code. No math, no code blocks, no images - it verifies these are
absent and refuses rather than silently mangling them.
"""
import argparse
import io
import re
import sys

BS = chr(92)


def esc(s):
    """Escape LaTeX specials. Order matters: backslash first."""
    s = s.replace(BS, BS + "textbackslash ")
    for ch in ("&", "%", "$", "#", "_", "{", "}"):
        s = s.replace(ch, BS + ch)
    s = s.replace("~", BS + "textasciitilde ")
    s = s.replace("^", BS + "textasciicircum ")
    # typographic niceties the source uses
    s = s.replace("—", "---").replace("–", "--")
    # These are math-mode commands: emitting them bare gives "Missing $ inserted".
    for uni, cmd in (("×", "times"), ("≥", "geq"), ("≤", "leq"),
                     ("→", "rightarrow"), ("≈", "approx"), ("±", "pm"),
                     ("−", "-"), ("μ", "mu"), ("σ", "sigma"), ("α", "alpha")):
        rep = ("$" + BS + cmd + "$") if cmd.isalpha() else cmd
        s = s.replace(uni, rep)
    s = s.replace("“", "``").replace("”", "''").replace("’", "'").replace("‘", "`")
    return s


def inline(s):
    """Apply inline markup AFTER escaping, using placeholders so escaping cannot eat them."""
    s = esc(s)
    # Lambda replacements, not template strings: a literal backslash in a re.sub
    # template makes "\e"/"\t" and re raises "bad escape" (or silently inserts a tab).
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: BS + "textbf{" + m.group(1) + "}", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
               lambda m: BS + "emph{" + m.group(1) + "}", s)
    s = re.sub(r"`(.+?)`", lambda m: BS + "texttt{" + m.group(1) + "}", s)
    return s


def parse_table(block):
    rows = []
    for line in block:
        if re.match(r"^\s*\|[\s:|-]+\|\s*$", line):
            continue  # separator row
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    spec = "l" + "r" * (ncol - 1) if ncol > 1 else "l"
    out = [BS + "begin{table}[htbp]", BS + "centering", BS + "small",
           BS + "begin{tabular}{" + spec + "}", BS + "toprule"]
    out.append(" & ".join(BS + "textbf{" + inline(c) + "}" for c in rows[0]) + " " + BS * 2)
    out.append(BS + "midrule")
    for r in rows[1:]:
        out.append(" & ".join(inline(c) for c in r) + " " + BS * 2)
    out += [BS + "bottomrule", BS + "end{tabular}", BS + "end{table}"]
    return "\n".join(out)


def convert_body(lines, drop_sections):
    out, i, dropping = [], 0, False
    # Skip the Markdown front matter: everything between the H1 title and the first
    # "## " heading is the source byline/affiliation block, which \maketitle replaces.
    # Emitting it put the INTEGRITY byline ("Nexus, Thomas Edrington, Lyra") into the
    # body of the academic edition - the exact defect this converter exists to prevent.
    while i < len(lines) and not re.match(r"^#{2,4}\s+", lines[i]):
        i += 1
    while i < len(lines):
        line = lines[i].rstrip("\n")
        h = re.match(r"^(#{1,4})\s+(.*)$", line)
        if h:
            level, text = len(h.group(1)), h.group(2).strip()
            bare = re.sub(r"[*`]", "", text).strip()
            dropping = any(d.lower() in bare.lower() for d in drop_sections)
            if not dropping and level > 1:
                cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
                # strip a leading "3.1 " style number; LaTeX numbers sections itself
                bare = re.sub(r"^\d+(\.\d+)*\.?\s+", "", bare)
                out.append("")
                out.append(BS + cmd + "{" + inline(bare) + "}")
            i += 1
            continue
        if dropping:
            i += 1
            continue
        if line.strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            out.append(""); out.append(parse_table(block))
            continue
        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i].rstrip("\n"))); i += 1
            out.append(""); out.append(BS + "begin{itemize}")
            out += ["  " + BS + "item " + inline(x) for x in items]
            out.append(BS + "end{itemize}")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i].rstrip("\n"))); i += 1
            out.append(""); out.append(BS + "begin{enumerate}")
            out += ["  " + BS + "item " + inline(x) for x in items]
            out.append(BS + "end{enumerate}")
            continue
        if re.match(r"^\s*---+\s*$", line):
            i += 1
            continue
        if not line.strip():
            out.append(""); i += 1
            continue
        out.append(inline(line)); i += 1
    return "\n".join(out)


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
% lmodern before microtype: microtype's font expansion needs scalable fonts, and
% bitmap Computer Modern gives "auto expansion is only possible with scalable fonts".
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--edition", choices=["academic", "integrity"], required=True)
    ap.add_argument("--author", default="Thomas Edrington")
    ap.add_argument("--affiliation", default="Liberation Labs / Transparent Humboldt Coalition")
    ap.add_argument("--ai", default="", help="comma-separated AI contributors, academic edition")
    ap.add_argument("--credit", default="", help="raw LaTeX for an Author Contributions block")
    ap.add_argument("--drop-section", action="append", default=[])
    a = ap.parse_args()

    raw = io.open(a.src, encoding="utf-8").read()
    if "```" in raw:
        sys.exit("REFUSING: source contains code blocks; this converter does not handle them")
    if re.search(r"!\[", raw):
        sys.exit("REFUSING: source contains images; this converter does not handle them")

    lines = raw.split("\n")
    m = re.match(r"^#\s+(.*)$", lines[0].strip())
    if not m:
        sys.exit("REFUSING: first line is not an H1 title")
    title = m.group(1).strip()

    drops = list(a.drop_section)
    if a.edition == "academic":
        drops.append("First-Person Reflection")

    body = convert_body(lines[1:], drops)

    front = [BS + "title{" + inline(title) + "}"]
    front.append(BS + "author{" + inline(a.author) + BS + "thanks{" + inline(a.affiliation) + "}}")
    front.append(BS + "date{}")
    doc = [PREAMBLE, BS + "begin{document}", "\n".join(front), BS + "maketitle", ""]
    if a.edition == "academic" and a.ai:
        doc.append(BS + "noindent" + BS + "textbf{AI Disclosure:} AI contributors to this work "
                   "include " + inline(a.ai) + ", implemented on the Claude architecture "
                   "(Anthropic). Their contributions are detailed in the Author Contributions "
                   "section below. T.~Edrington accepts accountability as corresponding human "
                   "author." + BS * 2)
        doc.append("")
    doc.append(body)
    if a.credit:
        doc.append("")
        doc.append(BS + "section*{Author Contributions (CRediT)}")
        doc.append(a.credit)
    doc.append(BS + "end{document}")

    io.open(a.dst, "w", encoding="utf-8", newline="\n").write("\n".join(doc) + "\n")
    print("wrote %s (edition=%s, dropped=%s)" % (a.dst, a.edition, drops))


if __name__ == "__main__":
    main()
