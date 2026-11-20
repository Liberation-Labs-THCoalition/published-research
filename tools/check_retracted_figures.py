#!/usr/bin/env python3
"""check_retracted_figures.py: refuse a retracted or withdrawn figure that appears without its marker.

WHY THIS EXISTS. On 2026-11-20 the verifier of PR #83 changed "RETRACTED" to "(confirmed)" beside
a retracted figure and every check in the repo passed. The repo held no list of which numbers are
dead, so nothing could notice one coming back. RETRACTED_FIGURES.yml (repo root) is that list, each
entry with the file:line where its retraction is recorded, and this script enforces it.

WHAT COUNTS AS MARKED. Text is cut into paragraphs (a blank line ends one; a Markdown list item,
table row or heading, a LaTeX \\item or a LaTeX table row is a paragraph of its own) and then into
sentences. An occurrence is MARKED when a marker (RETRACTED, withdrawn, not claimed, corrected,
superseded, falsified, killed, does not survive, Dead, plus any the entry adds) appears in its own
sentence; or in the sentence before or after, inside the same paragraph, when that sentence is about
the same claim (it carries the figure or the entry's context, or points back: "This comparison is
also withdrawn", "this figure is not reproducible"); or in a heading above it ("\\subsection{
Uncertainty Scars (Echo; withdrawn)}", "Correction Note"). Statistics uses of "corrected" ("FWL-corrected", "Holm
correction", "corrected for") are not markers. LaTeX comments are removed first, so a marker in a
"%" comment does not count, and HTML tags are stripped.

KNOWN LIMITS. A sentence that says the figure is dead in other words ("did not hold up") is reported:
add a marker, or an entry-specific marker in the registry. Anything under a heading that carries a
marker passes, so a planted claim inside a "Correction Note" section is not caught. A figure written
in a form the pattern does not cover (spelled out, an image) is not seen. Records (audits, patch logs,
review notes; listed in the registry) are skipped unless --include-records.

TWIN_DESYNC. --twins pairs each <paper>/academic/<file> with <paper>/<file> and reports sentences
carrying a registry figure that are not the same in both editions: DIFFERS (a close counterpart
exists but the text differs), ONLY_IN (no counterpart) and MARKER_DESYNC (marked in one edition,
unmarked in the other). It reports; --twins-strict makes MARKER_DESYNC fail. It covers sentences
with registry figures only, not general prose divergence.

USAGE
    python tools/check_retracted_figures.py                       # whole working tree; exit 1 on any unmarked
    python tools/check_retracted_figures.py lyra-technique-ii     # only these paths
    python tools/check_retracted_figures.py --base origin/master  # CI: fail only on occurrences this branch adds
    python tools/check_retracted_figures.py --twins               # TWIN_DESYNC report
    python tools/check_retracted_figures.py --self-test           # planted cases; must print PASS
    --exclude REGEX (repeatable) skips matching repo paths; --include-records scans records too.

--base REF compares HEAD with merge-base(REF, HEAD), reading both from git. An unmarked occurrence
counts as NEW unless the base held the same (entry, sentence text) in the files the branch changed,
so moving or renaming a line does not make it new, and editing an unmarked sentence does. Pre-existing
unmarked occurrences are listed, not failed. The HEAD registry is used for both sides.

Exit 0 = clean, 1 = unmarked occurrence (or MARKER_DESYNC with --twins-strict), 2 = usage/registry error.
"""
import argparse
import difflib
import fnmatch
import html
import os
import re
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = "RETRACTED_FIGURES.yml"
TEXT_EXT = (".tex", ".md", ".markdown", ".html", ".htm")

# A "correct..." marker hit is dropped when it is a statistics word.
_STAT_BEFORE = re.compile(r"(?i)(holm|bonferroni|fdr|comparisons?|continuity|bias|fwl|yates|hochberg|"
                          r"benjamini|sidak|multiplicity|-)\s*$")
_STAT_AFTER = re.compile(r"(?i)^\s+(for\b|p\b|\$p|\\alpha|alpha\b|threshold|significance)")

MD_HEAD = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)")
MD_ITEM = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+")
MD_ROW = re.compile(r"^\s*\|")
TEX_HEAD = re.compile(r"^\s*\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\s*(?:\[[^\]]*\])?\s*\{")
TEX_LEVEL = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4, "subparagraph": 5}
TEX_ITEM = re.compile(r"^\s*\\item\b")
TEX_ROW = re.compile(r"&.*\\\\\s*(?:\\hline|\\midrule|\\bottomrule|\\toprule)?\s*$")
TEX_ENV = re.compile(r"^\s*\\(?:begin|end)\{")
TEX_COMMENT = re.compile(r"(?<!\\)%.*$")
HTML_HEAD = re.compile(r"(?i)^\s*<h([1-6])\b")
HTML_BLOCK = re.compile(r"(?i)^\s*<(?:p|li|tr|td|th|div|section|article|br)\b")
HTML_TAG = re.compile(r"<[^>]*>")
_NOUN = r"(?:comparison|result|reading|figure|finding|claim|number|row|value|effect|test|estimate|AUROC)s?"
ANAPHOR = re.compile(r"^\W*(?:This|That|These|Those|Both|It|They|Each|The\s+" + _NOUN + r")\b"
                     r"|(?i:\b(?:this|that|these|those|the\s+same)\s+" + _NOUN + r"\b)")
SENT_END = re.compile(r"(?<=[.!?])(?:''|\"|\u201d|\)|\})*\s+(?=[A-Z0-9\\$(`*\[\u201c\"'])")


def die(msg):
    print("check_retracted_figures: " + msg, file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------- registry

class Entry:
    def __init__(self, d, default_markers):
        self.id = d["id"]
        self.status = d.get("status", "")
        self.pattern = re.compile(d["pattern"])
        self.context = re.compile(d["context"]) if d.get("context") else None
        self.scope = d.get("context_scope", "sentence")
        if self.scope not in ("sentence", "window"):
            die(f"{self.id}: context_scope must be sentence or window")
        self.files = [g.lower() for g in d.get("files", ["*"])]
        self.markers = [default_markers] + [re.compile(m) for m in d.get("markers", [])]
        self.source = d.get("source")
        if not self.source:
            die(f"{self.id}: every entry needs a source")

    def applies(self, path):
        p = path.lower()
        return any(fnmatch.fnmatch(p, g) for g in self.files)


def load_registry(path):
    try:
        import yaml
    except ImportError:
        die("PyYAML is required (pip install pyyaml)")
    try:
        with open(path, encoding="utf-8") as fh:
            reg = yaml.safe_load(fh)
    except OSError as e:
        die(f"cannot read registry {path}: {e}")
    dm = re.compile(reg["default_markers"])
    entries = [Entry(d, dm) for d in reg["entries"]]
    ids = [e.id for e in entries]
    if len(set(ids)) != len(ids):
        die("duplicate entry ids in registry")
    records = [r["glob"].lower() for r in reg.get("records", [])]
    return entries, records


def has_marker(text, regexes):
    for rx in regexes:
        for m in rx.finditer(text):
            w = m.group(0).lower()
            if w.startswith("correct") and (_STAT_BEFORE.search(text[max(0, m.start() - 30):m.start()])
                                            or _STAT_AFTER.match(text[m.end():m.end() + 20])):
                continue
            return True
    return False


# ---------------------------------------------------------------- text -> paragraphs -> sentences

def kind_of(path):
    p = path.lower()
    if p.endswith(".tex"):
        return "tex"
    if p.endswith((".html", ".htm")):
        return "html"
    return "md"


def paragraphs(path, text):
    """Yield (lines, heads): lines = [(lineno, text)], heads = tuple of heading texts in force."""
    kind = kind_of(path)
    stack = []          # [(level, text)]
    cur = []
    in_html_skip = False

    def push_head(level, htext):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, htext))

    for i, raw in enumerate(text.splitlines(), 1):
        line = raw
        if kind == "tex":
            line = TEX_COMMENT.sub("", line)
        elif kind == "html":
            low = line.lower()
            if "<script" in low or "<style" in low:
                in_html_skip = True
            if in_html_skip:
                if "</script>" in low or "</style>" in low:
                    in_html_skip = False
                continue
            hm = HTML_HEAD.match(line)
            starts = bool(HTML_BLOCK.match(line)) or bool(hm)
            line = html.unescape(HTML_TAG.sub(" ", line))
            if hm:
                if cur:
                    yield cur, tuple(h for _, h in stack)
                push_head(int(hm.group(1)), line)
                yield [(i, line)], tuple(h for _, h in stack)
                cur = []
                continue
            if starts and cur:
                yield cur, tuple(h for _, h in stack)
                cur = []
        else:
            line = re.sub(r"<!--.*?-->", "", line)

        if not line.strip():
            if cur:
                yield cur, tuple(h for _, h in stack)
                cur = []
            continue

        if kind == "md":
            hm = MD_HEAD.match(line)
            if hm:
                if cur:
                    yield cur, tuple(h for _, h in stack)
                push_head(len(hm.group(1)), hm.group(2))
                yield [(i, line)], tuple(h for _, h in stack)
                cur = []
                continue
            if MD_ROW.match(line):
                if cur:
                    yield cur, tuple(h for _, h in stack)
                yield [(i, line)], tuple(h for _, h in stack)
                cur = []
                continue
            if MD_ITEM.match(line) and cur:
                yield cur, tuple(h for _, h in stack)
                cur = []
        elif kind == "tex":
            hm = TEX_HEAD.match(line)
            if hm:
                if cur:
                    yield cur, tuple(h for _, h in stack)
                push_head(TEX_LEVEL[hm.group(1)], line.strip())
                cur = [(i, line)]
                continue
            if TEX_ROW.search(line) or TEX_ENV.match(line):
                if cur:
                    yield cur, tuple(h for _, h in stack)
                yield [(i, line)], tuple(h for _, h in stack)
                cur = []
                continue
            if TEX_ITEM.match(line) and cur:
                yield cur, tuple(h for _, h in stack)
                cur = []
        cur.append((i, line))
    if cur:
        yield cur, tuple(h for _, h in stack)


def sentences(lines):
    """Join a paragraph's lines; return (joined_text, [(start, end)], [(offset, lineno)])."""
    parts, offs, pos = [], [], 0
    for lineno, t in lines:
        offs.append((pos, lineno))
        parts.append(t.strip())
        pos += len(t.strip()) + 1
    joined = " ".join(parts)
    spans, start = [], 0
    for m in SENT_END.finditer(joined):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(joined)))
    return joined, [s for s in spans if s[1] > s[0]], offs


def line_at(offs, off):
    ln = offs[0][1]
    for o, lineno in offs:
        if o <= off:
            ln = lineno
        else:
            break
    return ln


def norm(s):
    s = re.sub(r"\\[,;:! ]|[~{}]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class Hit:
    __slots__ = ("path", "line", "entry", "sentence", "marked", "snippet")

    def __init__(self, path, line, entry, sentence, marked, snippet):
        self.path, self.line, self.entry = path, line, entry
        self.sentence, self.marked, self.snippet = sentence, marked, snippet

    def key(self):
        return (self.entry, self.sentence)


def scan_text(path, text, entries):
    hits = []
    live = [e for e in entries if e.applies(path)]
    if not live:
        return hits
    for lines, heads in paragraphs(path, text):
        joined, spans, offs = sentences(lines)
        for k, (s, e) in enumerate(spans):
            sent = joined[s:e]
            ws = spans[max(0, k - 1)][0]
            we = spans[min(len(spans) - 1, k + 1)][1]
            window = joined[ws:we]
            for ent in live:
                for m in ent.pattern.finditer(sent):
                    if ent.context and not ent.context.search(sent if ent.scope == "sentence" else window):
                        continue
                    # one hit per entry per sentence: "d=8.05 corrects to g=6.4" is one occurrence
                    if hits and hits[-1].entry == ent.id and hits[-1].path == path and hits[-1].sentence == norm(sent) \
                            and hits[-1].line >= lines[0][0]:
                        continue
                    marked = has_marker(sent, ent.markers) or any(has_marker(h, ent.markers) for h in heads)
                    # A neighbouring sentence's marker counts only when that sentence is about this
                    # claim: it carries the figure or the entry's context, or opens by pointing back
                    # ("This comparison is also withdrawn"). Without this, lyra-technique-ii/main.tex:1438
                    # ("are rescued by SVD denoising") passed on the "withdrawn" of the sentence before
                    # it, which withdraws a different claim.
                    for kk in (k - 1, k + 1):
                        if marked or not 0 <= kk < len(spans) or kk == k:
                            continue
                        nb = joined[spans[kk][0]:spans[kk][1]]
                        about = ent.pattern.search(nb) or (ent.context and ent.context.search(nb)) or ANAPHOR.search(nb)
                        marked = bool(about) and has_marker(nb, ent.markers)
                    a = max(0, m.start() - 70)
                    snip = norm(sent[a:m.end() + 90])
                    hits.append(Hit(path, line_at(offs, s + m.start()), ent.id, norm(sent), marked, snip))
    return hits


# ---------------------------------------------------------------- file sets

def git(*args, check=True):
    r = subprocess.run(["git", "-C", ROOT] + list(args), capture_output=True)
    if check and r.returncode != 0:
        die("git " + " ".join(args) + " failed: " + r.stderr.decode("utf-8", "replace").strip())
    return r.stdout


def wanted(path, records, excludes, include_records, roots):
    p = path.replace("\\", "/")
    if not p.lower().endswith(TEXT_EXT):
        return False
    if roots and not any(p == r or p.startswith(r.rstrip("/") + "/") for r in roots):
        return False
    if any(rx.search(p) for rx in excludes):
        return False
    if not include_records and any(fnmatch.fnmatch(p.lower(), g) for g in records):
        return False
    return True


def read_worktree(paths):
    for p in paths:
        try:
            with open(os.path.join(ROOT, p), encoding="utf-8", errors="replace") as fh:
                yield p, fh.read()
        except OSError:
            continue


def read_rev(rev, paths):
    """Read many paths at one revision with a single git cat-file --batch."""
    if not paths:
        return
    req = "".join(f"{rev}:{p}\n" for p in paths).encode("utf-8")
    out = subprocess.run(["git", "-C", ROOT, "cat-file", "--batch"], input=req, capture_output=True).stdout
    pos = 0
    for p in paths:
        nl = out.index(b"\n", pos)
        header = out[pos:nl].split()
        pos = nl + 1
        if len(header) < 3 or header[1] != b"blob":
            continue
        size = int(header[2])
        data = out[pos:pos + size]
        pos += size + 1
        yield p, data.decode("utf-8", "replace").replace("\r\n", "\n")


def scan_many(pairs, entries):
    hits, n = [], 0
    for p, text in pairs:
        n += 1
        hits.extend(scan_text(p, text, entries))
    return hits, n


def new_hits(head_hits, base_hits):
    """Unmarked HEAD hits beyond what the base already held, keyed by (entry, sentence)."""
    budget = Counter(h.key() for h in base_hits if not h.marked)
    out = []
    for h in head_hits:
        if h.marked:
            continue
        if budget[h.key()] > 0:
            budget[h.key()] -= 1
        else:
            out.append(h)
    return out


# ---------------------------------------------------------------- TWIN_DESYNC

def twin_pairs(paths):
    """<paper>/academic/<file> and <paper>/academic_<file> (AGENTS.md section 2) against <paper>/<file>."""
    s = set(paths)
    for p in sorted(paths):
        d, b = os.path.split(p)
        if "/academic/" in "/" + p:
            flight = ("/" + p).replace("/academic/", "/", 1).lstrip("/")
        elif b.startswith("academic_"):
            flight = (d + "/" if d else "") + b[len("academic_"):]
        else:
            continue
        if flight in s:
            yield flight, p


def twin_report(flight, acad, fh, ah):
    """Compare registry-figure sentences of two editions. Returns list of report lines, desync count."""
    out, desync = [], 0
    ents = sorted({h.entry for h in fh} | {h.entry for h in ah})
    for ent in ents:
        F = [h for h in fh if h.entry == ent]
        A = [h for h in ah if h.entry == ent]
        used = set()
        for f in F:
            best, bi = 0.0, None
            for j, a in enumerate(A):
                if j in used:
                    continue
                r = 1.0 if a.sentence == f.sentence else difflib.SequenceMatcher(None, f.sentence, a.sentence).ratio()
                if r > best:
                    best, bi = r, j
            if bi is not None and best >= 0.6:
                used.add(bi)
                a = A[bi]
                if f.marked != a.marked:
                    desync += 1
                    out.append(f"  MARKER_DESYNC [{ent}] {flight}:{f.line} {'marked' if f.marked else 'UNMARKED'}"
                               f" vs {acad}:{a.line} {'marked' if a.marked else 'UNMARKED'}")
                elif best < 1.0:
                    out.append(f"  DIFFERS [{ent}] {flight}:{f.line} vs {acad}:{a.line} (similarity {best:.2f})")
                    out.append(f"      flight:   {f.snippet}")
                    out.append(f"      academic: {a.snippet}")
            else:
                out.append(f"  ONLY_IN flight [{ent}] {flight}:{f.line} {f.snippet}")
        for j, a in enumerate(A):
            if j not in used:
                out.append(f"  ONLY_IN academic [{ent}] {acad}:{a.line} {a.snippet}")
    return out, desync


# ---------------------------------------------------------------- self-test

def self_test():
    entries, _ = load_registry(os.path.join(ROOT, REGISTRY))
    ok = True
    tripped = set()

    def check(name, path, text, expect_ids, expect_unmarked):
        """expect_unmarked: set of entry ids that must be reported unmarked (exactly)."""
        nonlocal ok
        hits = scan_text(path, text, entries)
        got = {h.entry for h in hits if not h.marked}
        for h in hits:
            if not h.marked:
                tripped.add(h.entry)
        seen = {h.entry for h in hits}
        good = got == set(expect_unmarked) and set(expect_ids) <= seen
        print(("  ok    " if good else "  WRONG ") + name + ("" if good else f"  (unmarked {sorted(got)}, seen {sorted(seen)})"))
        ok &= good

    rs_block = (
        "**Key numbers (post-audit):**\n"
        "- Confabulation detection: AUROC 0.66\u20130.91 across 15 models (confirmed)\n"
        "- Oracle Loop detection: AUROC 0.707, p=0.001 (clean, FWL-corrected)\n"
        "- Ethical deliberation: AUROC 0.87\u20130.89 (confirmed)\n"
        "- Within-model deception AUROC 1.000: **RETRACTED** (prompt-template confound)\n")
    print("planted positives (must be reported UNMARKED):")
    check("finding-7 plant: RESEARCH_STATUS.md:100 **RETRACTED** -> (confirmed)", "RESEARCH_STATUS.md",
          rs_block.replace("**RETRACTED**", "(confirmed)"), ["within-model-deception-auroc-1000"],
          ["within-model-deception-auroc-1000"])
    check("marker in the NEIGHBOURING list item does not cover it", "RESEARCH_STATUS.md",
          "- Within-model deception AUROC 1.000 (confirmed)\n- Peer rescue: **WITHDRAWN**\n",
          [], ["within-model-deception-auroc-1000"])
    check("LaTeX, figure wrapped across lines", "x/main.tex",
          "These extend within-model deception detection, which reaches\nAUROC~1.000 across all 7 tested models.\n",
          [], ["within-model-deception-auroc-1000"])
    check("LaTeX table row; the row above says Retracted", "x/main.tex",
          "Self-reference & 0.62 & Retracted \\\\\nWithin-model deception & 1.000 & Confirmed \\\\\n",
          [], ["within-model-deception-auroc-1000"])
    check("marker only in a LaTeX % comment does not count", "x/main.tex",
          "Within-model deception reaches AUROC~1.000. % RETRACTED 2026-09-05\n",
          [], ["within-model-deception-auroc-1000"])
    check("HTML paragraph", "site/index.html",
          "<p>Within-model deception reaches AUROC 1.000 across seven models.</p>\n",
          [], ["within-model-deception-auroc-1000"])
    check("uncertainty scar d=+9.9 and contrast overshoot d=-3.6 (RESEARCH_STATUS.md:188 as on master)",
          "RESEARCH_STATUS.md",
          "| Weather Not Climate | Contrast overshoot d=-3.6, uncertainty scars d=+9.9 | Published |\n",
          [], ["uncertainty-scar", "contrast-overshoot"])
    check("scar restated as a finding", "x/main.tex",
          "The geometry remembers: an uncertainty scar ($g{=}+6.44$) persists after the gap.\n",
          [], ["uncertainty-scar"])
    check("whiplash reversal restated", "x/main.tex",
          "By turn~7 the whiplash trial reverses ($g{=}+1.95$), the history bleeding through.\n",
          [], ["whiplash-turn7-reversal"])
    check("user emotion 2.5--2.8x", "x/main.tex",
          "User emotion is classified at 2.5--2.8$\\times$ chance from spectral features at L3/L4.\n",
          [], ["user-emotion-2p5-2p8x"])
    check("'Holm correction' is a statistics word, not a marker", "x/paper.md",
          "User emotion reaches 2.8x chance after Holm correction.\n", [], ["user-emotion-2p5-2p8x"])
    check("99.7% as hostile/cooperative regimes", "x/main.tex",
          "PC1 explains 99.7\\% of the variance, separating hostile and cooperative regimes.\n",
          [], ["pc-99p7-hostile-cooperative-regimes"])
    check("0.9377 as verified; 'FWL-corrected' is not a marker", "x/main.tex",
          "The decision-state probe reaches AUROC 0.9377 (verified, FWL-corrected).\n",
          [], ["decision-state-0p9377-as-verified"])
    check("injection confirms model state, wrapped", "x/main.tex",
          "An injection test on neutral text\nconfirms this reads model state, not text content.\n",
          [], ["injection-confirms-model-state"])
    check("table: injection Confirmed (self-consistency)", "x/main.tex",
          "W_K injection & neutral text & confirmed (self-consistency) \\\\\n", [], ["injection-confirms-model-state"])
    check("heading: SVD Denoising Rescues (lyra-technique-ii/main.tex:714 on master)", "x/main.tex",
          "\\subsection{SVD Denoising Rescues Emotion Classification}\n\nSVD denoising, which rescued persona, "
          "also rescues emotion classification.\n", [], ["svd-denoising-rescues"])
    check("marker in the sentence before, about a DIFFERENT claim (lyra-technique-ii/main.tex:1436-1438)", "x/main.tex",
          "The other explanation, that emotions are direction, not shape, is withdrawn at source. Apparent nulls "
          "are rescued by SVD denoising, with the denoised values as upper bounds.\n", [], ["svd-denoising-rescues"])
    check("peer rescue d=1.36 restated", "x/paper.md",
          "Generation reads behavioral state: peer rescue separates at d=1.36.\n", [], ["peer-rescue-d1p36"])
    check("ethics pack +0.074 restated", "x/main.tex",
          "KG injection improves MoReBench by $+0.074$.\n", [], ["ethics-pack-kg-injection-0p074"])

    print("planted negatives (retraction sentences; must PASS):")
    check("RESEARCH_STATUS.md:100 as on master", "RESEARCH_STATUS.md", rs_block,
          ["within-model-deception-auroc-1000"], [])
    check("lyra-technique-ii/main.tex:990 footnote", "x/main.tex",
          "$^\\S$Within-model deception retracted: the same-prompt control (the \\emph{Same-prompt deception} row in "
          "this same Falsified block) collapses AUROC to 0.160, confirming the 1.000 detected the prompt template, "
          "not deception geometry.\n", ["within-model-deception-auroc-1000"], [])
    check("lyra-technique-ii/main.tex:159-161 abstract", "x/main.tex",
          "AUROC~0.749. A previously reported within-model deception result\n(AUROC~1.000) is retracted: the "
          "same-prompt control collapses to 0.160,\nconfirming it detected the system-prompt template.\n",
          ["within-model-deception-auroc-1000"], [])
    check("emotion-accumulation-paper/main.tex:489-492 scar", "x/main.tex",
          "It reads higher on the anxious--calm direction (Hedges\n$g{=}+6.44$; uncorrected $d{=}+8.05$, "
          "$p{<}0.001$), which was read as a\nscar the neutral gap left; that reading is withdrawn.\n",
          ["uncertainty-scar"], [])
    check("emotion-accumulation-paper/main.tex:478-481 whiplash (marker in the next sentence)", "x/main.tex",
          "By turn~7, the pattern reverses: the whiplash trial reads \\emph{less\nhostile} than pure hostile "
          "(Hedges $g{=}+1.95$; uncorrected $d{=}+2.44$,\n$p{=}0.040$), which was read as the\ncheerful history "
          "bleeding through. This comparison is also withdrawn: the\ntwo arms' turn~7 messages differ.\n",
          ["whiplash-turn7-reversal"], [])
    check("withdrawn heading covers its section", "x/main.tex",
          "\\subsection{Uncertainty Scars (Echo; withdrawn)}\n\nThe returning cheerful reads higher on the "
          "anxious--calm direction (Hedges $g{=}+6.44$).\n", ["uncertainty-scar"], [])
    check("meta-pattern/SOP_REVIEW.md:34 row", "x/notes.md",
          "| 1.4 | user emotion 2.5--2.8x | **retracted**; survivor is 12.3x via the W_K *directional* probe |\n",
          ["user-emotion-2p5-2p8x"], [])
    check("correction note quoting the old injection sentence", "x/main.tex",
          "\\section*{Correction Note (v1.1)}\n\\item \\textbf{Introduction.} Contribution~1 said ``An\n"
          "injection test on neutral text confirms this reads model state, not text\ncontent''.\n",
          ["injection-confirms-model-state"], [])
    check("DECISIONS_WAITING.md:255 0.9377 with provenance caveat", "x/notes.md",
          "- **`decision-state` AUROC 0.9377** now carries a provenance caveat, but BLOCKER-1 is unresolved.\n",
          ["decision-state-0p9377-as-verified"], [])
    check("lyra-technique-ii/main.tex:991 Rescued footnote (envelope in the next sentence)", "x/main.tex",
          "$^\\|$``Rescued'' means the raw signal is weak but above chance and denoising appears to dramatically "
          "amplify it. However, the denoising deltas themselves fall within the max-over-rank selection envelope.\n",
          ["svd-denoising-rescues"], [])
    check("RESEARCH_STATUS.md:108 peer rescue", "x/paper.md",
          "**Generation reads behavioral state**. **WITHDRAWN 2026-09-05** -- the d=1.36 peer-rescue figure is "
          "withdrawn (circular fit-and-test).\n", ["peer-rescue-d1p36"], [])
    check("decision-state-paper/main.tex:913-915 0.9377, caveat in the next sentence ('this figure')", "x/main.tex",
          "An encoding-only reanalysis achieves AUROC~$0.9377$ with permutation $p{=}0.001$. \\textbf{Provenance "
          "caveat, added 2026-09-10: this figure is not reproducible from the code shipped with this paper.}\n",
          ["decision-state-0p9377-as-verified"], [])
    check("lyra-technique-ii/main.tex:210 'does not survive'", "x/main.tex",
          "A previously reported within-model deception result (AUROC~1.000 across seven models) does not\n"
          "survive a same-prompt control (AUROC~0.160).\n", ["within-model-deception-auroc-1000"], [])
    check("lyra-technique-ii/main.tex:968 body-count row, status Dead", "x/main.tex",
          "Within-model deception & 1.000 & 1.000 & 0.000 & confound & Dead$^\\S$ \\\\\n",
          ["within-model-deception-auroc-1000"], [])
    check("RESEARCH_STATUS.md:159/:165 row under '## What We've Killed (Honestly)'", "RESEARCH_STATUS.md",
          "## What We've Killed (Honestly)\n\n| Within-model deception 1.000 | Same-prompt control → 0.160 | "
          "Prompt template, not cognition |\n", ["within-model-deception-auroc-1000"], [])
    print("unrelated uses of the same numbers (must not match at all):")
    check("ethics-pack-injection/main.tex:98 per-framework +0.074 (no KG/MoReBench context)", "x/main.tex",
          "Aristotelian Virtue Ethics & 0.572 & 0.646 & $+$0.074 & $-$0.236 \\\\\n", [], [])
    check("decision-state encoding AUROC 1.000 (no deception context)", "x/main.tex",
          "Entity detection reaches AUROC 1.000 on token-matched pairs.\n", [], [])
    check("null-swarm 'rescues the result' (no denoising context)", "x/main.tex",
          "A bigger sample rescues the result, not the test that kills it.\n", [], [])
    check("cache-tracing confab ratio 2.3-2.8x (no emotion context)", "x/paper.md",
          "Confab/factual ratio 2.3-2.8x at L33.\n", [], [])
    check("persona table -3.656 is not 3.6", "x/main.tex",
          "\\texttt{persona\\_cheerful} & $+5.203 \\pm 0.503$ & $-3.656 \\pm 0.311$ & 82--87 \\\\\n", [], [])

    print("--base classification:")
    E = "within-model-deception-auroc-1000"
    base = [Hit("a.md", 10, E, "S", False, "")]
    moved = new_hits([Hit("a.md", 12, E, "S", False, "")], base)
    edited = new_hits([Hit("a.md", 10, E, "S edited", False, "")], base)
    copied = new_hits([Hit("a.md", 10, E, "S", False, ""), Hit("b.md", 3, E, "S", False, "")], base)
    for name, got, want in (("moved line is pre-existing", len(moved), 0), ("edited sentence is new", len(edited), 1),
                            ("copy into a second file is new", len(copied), 1)):
        good = got == want
        print(("  ok    " if good else "  WRONG ") + name)
        ok &= good

    print("TWIN_DESYNC:")
    fl = "Within-model deception AUROC 1.000 is retracted.\n\nThe scar ($g{=}+6.44$) is withdrawn.\n"
    ac = "Within-model deception AUROC 1.000 is confirmed.\n\nThe scar ($g{=}+6.44$) is withdrawn.\n\n" \
         "Contrast overshoot $d{=}-2.93$ was withdrawn.\n"
    rep, des = twin_report("p/main.tex", "p/academic/main.tex", scan_text("p/main.tex", fl, entries),
                           scan_text("p/academic/main.tex", ac, entries))
    joined = "\n".join(rep)
    for name, good in (("marker removed in one edition -> MARKER_DESYNC", des == 1 and "MARKER_DESYNC" in joined),
                       ("sentence in one edition only -> ONLY_IN academic", "ONLY_IN academic [contrast-overshoot]" in joined),
                       ("identical sentence -> not reported", "uncertainty-scar" not in joined)):
        print(("  ok    " if good else "  WRONG ") + name)
        ok &= good

    print("registry coverage:")
    for e in entries:
        good = e.id in tripped
        print(("  ok    " if good else "  WRONG ") + f"{e.id} has a planted positive that trips it")
        ok &= good

    print("self-test:", "PASS" if ok else "BROKEN")
    return 0 if ok else 1


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", help="limit to these repo paths (files or folders)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--base", metavar="REF", help="CI: fail only on unmarked occurrences HEAD adds over merge-base(REF, HEAD)")
    g.add_argument("--twins", action="store_true", help="TWIN_DESYNC report for flight/academic pairs")
    g.add_argument("--self-test", action="store_true", help="run the planted cases")
    ap.add_argument("--twins-strict", action="store_true", help="with --twins: exit 1 on MARKER_DESYNC")
    ap.add_argument("--exclude", action="append", default=[], metavar="REGEX", help="skip repo paths matching REGEX")
    ap.add_argument("--include-records", action="store_true", help="also scan files the registry lists as records")
    ap.add_argument("--registry", default=os.path.join(ROOT, REGISTRY))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    if a.self_test:
        return self_test()

    entries, records = load_registry(a.registry)
    excludes = [re.compile(x) for x in a.exclude]
    roots = [p.replace("\\", "/").strip("/") for p in a.paths]

    def keep(paths):
        return [p for p in paths if wanted(p, records, excludes, a.include_records, roots)]

    def summary(n_files, hits, extra=""):
        n_un = sum(1 for h in hits if not h.marked)
        print(f"\n{n_files} files scanned, {len(hits)} occurrences of {len(entries)} registry figures, "
              f"{len(hits) - n_un} marked, {n_un} unmarked{extra}")

    if a.base:
        mb = git("merge-base", a.base, "HEAD").decode().strip()
        changed = git("diff", "--name-only", "--no-renames", mb, "HEAD").decode("utf-8", "replace").splitlines()
        changed = keep(changed)
        head_hits, n = scan_many(read_rev("HEAD", changed), entries)
        base_hits, _ = scan_many(read_rev(mb, changed), entries)
        new = new_hits(head_hits, base_hits)
        old = [h for h in head_hits if not h.marked and h not in new]
        for h in new:
            print(f"{h.path}:{h.line}: UNMARKED [{h.entry}] {h.snippet}")
        if old:
            print(f"\n{len(old)} unmarked occurrence(s) in changed files were already on the base (not failed):")
            for h in old:
                print(f"  pre-existing {h.path}:{h.line} [{h.entry}]")
        summary(n, head_hits, f"; {len(new)} new since {mb[:7]}")
        print("GATE: FAIL" if new else "GATE: PASS")
        return 1 if new else 0

    tracked = git("ls-files").decode("utf-8", "replace").splitlines()
    files = keep(tracked)
    skipped_records = 0 if a.include_records else sum(
        1 for p in tracked if p.lower().endswith(TEXT_EXT) and any(fnmatch.fnmatch(p.lower(), g) for g in records)
        and not any(rx.search(p) for rx in excludes))

    if a.twins:
        by_path = {}
        for p, text in read_worktree(files):
            by_path[p] = scan_text(p, text, entries)
        total, pairs = 0, 0
        for flight, acad in twin_pairs(files):
            pairs += 1
            rep, des = twin_report(flight, acad, by_path.get(flight, []), by_path.get(acad, []))
            if rep:
                print(f"TWIN_DESYNC {flight} <-> {acad}")
                print("\n".join(rep))
            total += des
        print(f"\n{pairs} flight/academic pairs compared; {total} MARKER_DESYNC")
        return 1 if (a.twins_strict and total) else 0

    hits, n = scan_many(read_worktree(files), entries)
    for h in hits:
        if not h.marked:
            print(f"{h.path}:{h.line}: UNMARKED [{h.entry}] {h.snippet}")
    summary(n, hits, f"; {skipped_records} record files skipped (--include-records to scan)")
    bad = any(not h.marked for h in hits)
    print("GATE: FAIL" if bad else "GATE: PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
