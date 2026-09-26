#!/usr/bin/env python3
"""Build main.tex and main.pdf for both editions from their Markdown sources.

    paper.md          -> main.tex, main.pdf            (integrity edition)
    academic/paper.md -> academic/main.tex, main.pdf   (academic edition)

Edit the Markdown; the .tex files are generated. The .tex follows the repository's LaTeX house style
(article, lmodern, microtype, booktabs) and builds with plain pdflatex, so scripts/build_and_verify.sh
can rebuild it (--deep runs `latexmk -pdf`, which is pdflatex run to a fixed point, as here).

The head of each Markdown file is: "# Title", one "**Name** (role) · e-mail" line per author, the
affiliation, an italic version line, and optionally an "**AI Disclosure:**" paragraph. Everything from
"## Abstract" on is the body, converted by pandoc with headings shifted up one level. Section numbers
are part of the heading text, so LaTeX numbering is off.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PANDOC = ["pandoc", "-f", "markdown+lists_without_preceding_blankline", "-t", "latex", "--wrap=preserve"]

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{textcomp}
% lmodern before microtype: microtype's font expansion needs scalable fonts.
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{longtable,array,calc}
\usepackage{etoolbox}
\makeatletter
\patchcmd\longtable{\par}{\if@noskipsec\mbox{}\fi\par}{}{}
\makeatother
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{3em}
\sloppy
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setcounter{secnumdepth}{-\maxdimen}
% Characters in the text that pdflatex's utf8 does not map by default.
\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
\DeclareUnicodeCharacter{207B}{\textsuperscript{\ensuremath{-}}}
\DeclareUnicodeCharacter{207D}{\textsuperscript{(}}
\DeclareUnicodeCharacter{207E}{\textsuperscript{)}}
\DeclareUnicodeCharacter{2070}{\textsuperscript{0}}
\DeclareUnicodeCharacter{2079}{\textsuperscript{9}}
\DeclareUnicodeCharacter{03BA}{\ensuremath{\kappa}}
"""


def gfm_list_rule(md: str) -> str:
    """lists_without_preceding_blankline lets ANY numbered line start a list, so a wrapped sentence whose
    next line begins "2025) tests..." became a list item. GitHub lets an ordered list interrupt a paragraph
    only when it starts at 1; apply the same rule by joining such a line back onto its paragraph."""
    lines = md.split("\n")
    out = []
    for ln in lines:
        m = re.match(r"^(\d+)[.)]\s", ln)
        prev = out[-1] if out else ""
        in_list = (not prev.strip()) or prev.startswith((" ", "\t")) or re.match(r"^(\d+[.)]|[-*+])\s", prev)
        if m and int(m.group(1)) != 1 and not in_list:
            out[-1] = prev + " " + ln
        else:
            out.append(ln)
    return "\n".join(out)


def pandoc(md: str) -> str:
    return subprocess.run(PANDOC + ["--shift-heading-level-by=-1"], input=gfm_list_rule(md), capture_output=True,
                          text=True, check=True).stdout


def inline(md: str) -> str:
    out = subprocess.run(PANDOC, input=md, capture_output=True, text=True, check=True).stdout.strip()
    return out


def build_tex(md_path: str) -> str:
    text = open(md_path, encoding="utf-8").read()
    head, sep, body = text.partition("\n---\n\n## Abstract")
    if not sep:
        sys.exit(f"{md_path}: no '---' + '## Abstract' boundary")
    body = "## Abstract" + body
    lines = head.strip("\n").split("\n")
    if not lines[0].startswith("# "):
        sys.exit(f"{md_path}: first line is not '# Title'")
    title = inline(lines[0][2:])
    authors, rest = [], []
    for ln in lines[1:]:
        m = re.match(r"\*\*(.+?)\*\*\s*(\((.+?)\))?\s*·\s*(\S+@\S+?)\\?$", ln.strip())
        if m:
            authors.append((m.group(1), m.group(3), m.group(4)))
        elif ln.strip():
            rest.append(ln.strip())
    if not authors:
        sys.exit(f"{md_path}: no author lines")
    affiliation, blocks = rest[0], rest[1:]
    names = []
    for name, role, email in authors:
        label = f"{inline(name)} ({inline(role)})" if role else inline(name)
        names.append(label + r"\thanks{" + inline(affiliation) + ". " + r"\href{mailto:" + email + "}{" + inline(email) + "}}")
    front = "\n\n".join(r"\noindent " + inline(b) for b in blocks)
    return (PREAMBLE + "\n\\begin{document}\n"
            + "\\title{" + title + "}\n\\author{" + " \\and ".join(names) + "}\n\\date{}\n\\maketitle\n\n"
            + front + "\n\n" + pandoc(body) + "\n\\end{document}\n")


def compile_pdf(directory: str) -> None:
    for _ in range(2):  # the second pass settles cross-references, as latexmk would
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                           cwd=directory, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"pdflatex failed in {directory}:\n" + r.stdout[-3000:])
    log = open(os.path.join(directory, "main.log"), encoding="latin-1").read()
    missing = re.findall(r"Missing character: There is no (.) in font", log)
    if missing:
        sys.exit(f"{directory}: missing glyphs {sorted(set(missing))}")
    for ext in ("aux", "log", "out", "toc"):
        p = os.path.join(directory, "main." + ext)
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    for d in (HERE, os.path.join(HERE, "academic")):
        tex = build_tex(os.path.join(d, "paper.md"))
        open(os.path.join(d, "main.tex"), "w", encoding="utf-8").write(tex)
        compile_pdf(d)
        print("built", os.path.relpath(os.path.join(d, "main.pdf"), HERE))
