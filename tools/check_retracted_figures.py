#!/usr/bin/env python3
"""check_retracted_figures.py: refuse a retracted or withdrawn figure that appears without its marker.

WHY THIS EXISTS. On 2026-11-20 the verifier of PR #83 changed "RETRACTED" to "(confirmed)" beside
a retracted figure and every check in the repo passed. The repo held no list of which numbers are
dead, so nothing could notice one coming back. RETRACTED_FIGURES.yml (repo root) is that list, each
entry with the file:line where its retraction is recorded, and this script enforces it.

WHAT COUNTS AS MARKED. Text is cut into paragraphs (a blank line ends one; a Markdown list item,
table row or heading, a LaTeX \\item, table row or sectioning line is a paragraph of its own), then
into sentences, then into clauses (split at a top-level ";" and at ", and|but|which|while|..." and
" while|because|after|before|..."; never inside parentheses or ``quotes''). Before matching, TeX
noise is canonicalised: braces, "$", "~" and "\\," go, "\\times" becomes "x" sign, "--" an en dash,
"\\%" a percent sign, so "AUROC~$1.00$" and "2.5--2.8$\\times$" match like plain text.

A marker (RETRACTED, withdrawn, not claimed, superseded, falsified, killed, does not survive, Dead,
"was/since corrected", "corrected to", plus any the entry adds) MARKS an occurrence only when it is
attached to that same claim:
  - it is in the occurrence's own clause; or
  - it is in a later clause of the same sentence, reached through clauses that each open by pointing
    back ("..., which was later retracted", "...; that reading is withdrawn"); or
  - it is in the NEXT sentence of the same paragraph, in a clause that opens by pointing back
    ("This comparison is also withdrawn", "Provenance caveat: this figure is not reproducible").
It does NOT count when: it is in the previous sentence (a different claim instance: PR #84 plant
P5); it is anywhere else in the paragraph (P7); it is in a heading (P6, P10: headings never mark,
not even "## What We've Killed"); it is a one-word participle used attributively on some other noun
("the untested raw baseline": P8); it is negated ("was never retracted"); or it is "correction"/
"corrected" used for anything but the figure being corrected away ("after error correction": P9;
"FWL-corrected", "Holm correction", "corrected for" were never markers).

A QUOTED occurrence passes: the claim lives inside a ``...'' or "..." quotation (with the quotation's
words blanked, the entry's pattern no longer matches there) and the quotation is REPORTED: a
past-tense cue at most eight words before it with no clause break between (said, wrote; "titled"/
"headed" only in a past form: "was titled", "previously titled", "originally headed", "used to be
titled", never the participle ", titled ``...''" or "is now/still titled"; "read" only where it
cannot be present tense: "previously read", "used to read", "the heading read"; or a capitalised
"Was" opening a status cell: "this paragraph said denoising ``also rescues...''"), or it continues
a reported quotation with only a list separator and an optional bracket of at most eight words
between (", ``...''", "(from v1.0) ``...''"; never through "that": ", and that we show ``...''"
does not continue it), or it is followed by "means" ("``Rescued'' means ..."). A quotation mark is
always required. A correction note quotes what it corrects (AGENTS.md section 4, cause 4). Quoted
occurrences are counted separately in the summary. LaTeX comments are removed first, so a marker
in a "%" comment does not count, and HTML tags are stripped. "We claim neither that" and "We had
(also) written that" are markers (registry default_markers).

KNOWN LIMITS. A sentence that says the figure is dead in other words ("did not hold up") is reported:
add a marker, or an entry-specific marker in the registry. A marker can be a word that kills
something else in the same clause ("1.000, the test that killed the null"); the clause rule narrows
this, it does not close it. A correction note that gives the old wording WITHOUT quotation marks
("previously said SVD denoising rescues ...", "both rows read Rescued") is reported: quote the old
wording. Records (audits, patch logs, review notes; listed in the registry) are skipped unless
--include-records. Known gaps, each a live claim that passes (from PR #84's round-2 and round-4
verifiers; left open by Lyra's call, not fixed):
  - a claim restated inside quotation marks after a past-tense cue: "As we said, ``...''";
  - (U8) "had read" with a person as subject, reaffirmed: "The reviewers had read ``SVD denoising
    rescues emotion classification'' in our abstract, which remains our headline result";
  - (U5) a subjunctive "read": "We propose that the heading read ``SVD Denoising Rescues ...''";
  - (U9) a definition of a live claim: "``denoising rescues emotion classification'' means the
    denoised probe beats the raw probe", which passes on the "means" rule;
  - (U10) a status cell that opens with "Was" but keeps the claim: 'Was "SVD denoising rescues
    persona"; still holds at n=300';
  - "We had written that X, and the new data confirm it" passes as marked;
  - a figure written in a form the pattern does not cover: "the AUROC was 1" ("was" is not one of
    the deception entry's verbs), the noun "improvement" for denoising, a number spelled out other
    than "AUROC of one"/"unity", an image.
  Added in round 6 from the round-5 verifier (each case id is from its plants; written down, not fixed):
  - (finding 1) the "decepti..." word and the deception figure in different sentences, table rows
    or a footnote, because the entry's context is read in the figure's own sentence and a table row
    is a sentence: a header "| Probe | Within-model deception AUROC |" over a row "| Llama-3 |
    1.000 |", and the same in LaTeX (M5, M6, the most natural table shape); a \\multirow label
    "Within-model deception" in one row and "AUROC $=1$" in the next (M2, M4); a footnote after the
    sentence end, "... detected in every model.\\footnote{Each of the seven models reaches AUROC
    $= 1$.}" (F2); "Can probes detect within-model deception? Yes: AUROC = 1 in all seven models"
    (E2); "We probe within-model deception. The AUROC is 1.000 in all seven models." (E4). On
    master 0ef6b74 the verifier found 21 lines of this kind and none was a live claim;
  - (finding 2) the deception window stops at ", and" and after six words: "We test within-model
    deception, and AUROC = 1 holds in all seven models" (E3); "Within-model deception probes
    trained on all seven held-out models: AUROC = 1" (W7, seven words between). (E1, a ";" inside
    a bracket, is fixed in round 6: see pattern_view.);
  - (finding 3) a past "titled" beside a live one: the cue need only be the last within eight
    words, and nothing reads the gap for "is", "still" or "now": "Section 4 was titled and still
    is titled ``SVD Denoising Rescues Emotion Classification''" (H1); "Section 4 is titled, as it
    was titled in v1, ``SVD Denoising Rescues Emotion Classification''" (H7); "Section 4 was
    titled ``SVD Denoising Rescues Emotion Classification'' and still is" (H5, the U10 kind);
  - (finding 4) a bracket of up to eight words continues a reported quotation whatever it says,
    so it can reaffirm the claim: "The heading previously read ``Persona gradient'' (and we still
    hold this claim in v1.3) ``SVD denoising rescues persona intensity in all three models''"
    passes as quoted (B8L).
  Known false alarms (reported, though not a live claim; from the same verifier):
  - (finding 4) symbols count as words in the bracket bound, so a correction note with a stats
    bracket breaks the chain: "The heading previously read ``Persona gradient'' (v1.2, n = 300,
    p < .001 in Table 2) ``SVD denoising rescues persona intensity''" (B9s);
  - (finding 5) "unlike" before the deception word and "not" do not stop the window: "Unlike
    deception, sentiment probes reach AUROC = 1 in all seven models" (X1); "AUROC = 1 holds for
    sentiment, not for deception" (X2). No cost on master 0ef6b74.

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
    The agent_skip paths are skipped in every mode (agents are not to open them; each run prints how
    many it skipped). The list is in the CODE (AGENT_SKIP), so a registry without an agent_skip list,
    such as any registry before 7767f69, still skips them; a registry's list can only add paths.
    --no-agent-skip scans them and prints a loud warning; only CI passes it. Options must be spelled
    in full ("--no" or "--no-agent" is an error, not --no-agent-skip). A registry's agent_skip must
    be a list of non-empty regexes, or the run exits 2. --registry is refused when the path as typed,
    made absolute or fully resolved (links, Windows short names) is an agent_skip path.
    --self-test-force marked|unmarked works only together with --self-test; alone it exits 2.

--base REF compares HEAD with merge-base(REF, HEAD), reading both from git. An unmarked occurrence
counts as NEW unless the base held the same (entry, sentence text) in the files the branch changed,
so moving or renaming a line does not make it new, and editing an unmarked sentence does. Pre-existing
unmarked occurrences are listed, not failed. The HEAD registry is used for both sides.

Exit 0 = clean, 1 = unmarked occurrence (or MARKER_DESYNC with --twins-strict), 2 = usage/registry error.
"""
import argparse
import contextlib
import difflib
import fnmatch
import html
import io
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
_NOUN = (r"(?:comparison|result|reading|figure|finding|claim|number|row|value|effect|test|estimate|AUROC|"
         r"gain|delta|scar|overshoot|reversal|rescue|readout|interpretation|conclusion|statistic|separation)s?")
_CONJ = r"(?:and|but|yet|so|though|although|while|whilst|whereas|because|after|before|when|unlike)"
# A clause that opens by pointing back at the claim before it. An optional short label ending in ":"
# may come first ("Provenance caveat, added 2026-09-10: this figure is ..."). "this"/"that" must be
# followed by a claim noun or a verb, so "this pilot was withdrawn" does not point back.
ANAPHOR_START = re.compile(
    r"(?i)^[\W_]*(?:[^:;]{0,60}:\s*)?(?:" + _CONJ + r"\s+)?"
    r"(?:(?:this|that|these|those|both|each|the\s+same)\s+(?:(?:\S+\s+)?" + _NOUN +
    r"\b|(?:is|was|were|are|has|had|have|does|did|too|also|reading)\b)"
    r"|it\s+(?:is|was|has|had|does|did|too|also|were)\b|which\b"
    r"|the\s+(?!(?:earlier|previous|older|original|prior|other|new|old|unrelated|first-draft)\b)(?:\S+\s+)?" + _NOUN + r"\b)")
CLAUSE_SEP = re.compile(r";|,\s+(?=(?:and|but|which|while|whilst|whereas|although|though|yet|so)\b)"
                        r"|\s+(?=(?:while|whilst|whereas|although|though|because|unlike|after|before|when)\b)", re.I)
NEG_BEFORE = re.compile(r"(?i)\b(?:not|never|no\s+longer)\s+(?:been\s+|be\s+|yet\s+|formally\s+)?$|n't\s+(?:been\s+)?$")
DET_BEFORE = re.compile(r"(?i)\b(?:the|a|an|this|that|these|those|its|their|our|any|each|every|his|her)\s+$")
STOP_AFTER = re.compile(r"(?i)(?:by|in|at|on|for|from|as|since|because|pending|until|after|before|and|or|but|is|was|"
                        r"were|are|has|have|had|of|to|with|under|due|per|over|via|here|there)$")
NP_END = re.compile(r"[;:()]|[,.](?!\d)|\s(?:which|that|who|is|was|were|are|has|had)\b")
# A quotation REPORTS old text (round 3, post-data revision: on master after #83 the round-2 rule left 19 correction-
# note quotations unmarked) when:
#   - a reporting cue comes at most REPORT_GAP words before its opening quote, with no clause break between
#     ("this paragraph said denoising ``also rescues...''", "the closing paragraph said persona intensity and
#     spectral emotion classification ``are rescued...''"); a capitalised "Was" opening a cell or sentence is a cue
#     (RESEARCH_STATUS.md 'Was "SVD denoising rescues persona"; corrected'); or
#   - it continues a reported quotation within REPORT_GAP + 2 words, with no sentence break and no "but"/"whereas"
#     between (", and that the denoising step was ``revealing...''"); or
#   - it is followed by "means" (a definition).
# A one-word capitalised cell value right after "read"/"marked"/"headed" ("both rows read Rescued") is quoted too.
# The principle, which would hold whatever the data showed: a correction note quotes what it corrects, and the
# quotation is marked by the verb that reports it, not by adjacency. Cost: a live claim placed inside quotation marks
# up to eight words after "said" passes as quoted (KNOWN LIMITS). Original round-2 rule, kept for the record:
#   REPORT_BEFORE = (?i)(?:\b(?:said|says|read|reads|wrote|stated|claimed|called|labell?ed|marked|termed|titled|
#                   the\s+(?:term|word|phrase|sentence|line|claim|keyword))|(?:''|"|")\s*(?:,|and|or))\s*$
#   searched in the 30 characters before the opening quote, and the occurrence's first character had to be inside it.
#
# Round 4 (2026-11-20, post-data revision after the round-3 verifier's plants D11-D13, S13-S16 passed as quoted).
# Round 3 had added three rules beyond Lyra's brief and each let a live claim through: a capitalised word with no
# quotation marks after read/marked/titled ("Our headline result reads AUROC 1.0"), overlap (one cued quoted word
# anywhere inside a long match passed the whole match), and chaining after any quote within ten words ('said "...",
# and we show "..."'). Present-tense cues (says/states/writes/written) reported a live claim, not old wording.
# The rule now, as the brief gives it:
#   - the quotation mark is always required;
#   - the cue is a past-tense reporting form: said, wrote, headed, titled (not after is/are/be: "Section 4 is titled
#     ``...''" names a live heading), "read" only where it cannot be present tense ("previously/originally/formerly/
#     earlier/initially read", "used to read", "had read", or a singular document part as subject: "the heading
#     read", "this sentence read"; never "We read" or "rows read"), and "Was" opening a status cell (sentence start,
#     or after a "|" or "&" cell boundary);
#   - a quotation continues a reported one only when nothing but a list separator, an optional parenthesis and an
#     optional "that ..." complement of at most five words stands between them (", ``...''", "and that the step was
#     ``...''"); "and we show ``...''" does not continue it;
#   - the occurrence passes only if the claim lives inside the quotation: blank the reported quotations' words and
#     the entry's pattern must no longer match there ("SVD denoising restores what the README said was ``persona''
#     intensity" still matches with ``persona'' blanked, so it is reported).
# The principle, which would hold whatever the data showed: a quotation is old wording only when a past-tense verb
# reports it, and only the quoted words are excused. Cost, carried into the result: on master 0ef6b74 four lines
# that give an old table value without quotation marks ("both rows read Rescued", "rows read Rescued (now Upper
# bound)", in lyra-technique-ii/main.tex and academic/main.tex) are reported again; the fix there is to quote the
# value in the paper. Round-3 rule, kept for the record:
#   REPORT_CUE = (?i)\b(?:said|says|read|reads|wrote|written|writes|stated|states|claimed|called|labell?ed|marked|
#                termed|titled|headed|captioned|the\s+(?:term|word|phrase|sentence|line|claim|keyword))\b
#                |(?:^|[|(:]\s*)(?-i:Was)\b
#   CHAIN_BREAK = (?i)[.;!?](?!\d)|\b(?:but|yet|whereas|while|whilst|however|although|though)\b  (a quote within
#                REPORT_GAP + 2 words after a reported one, with no CHAIN_BREAK between, was reported)
#   BARE_CUE = (?i)\b(?:read|reads|marked|labell?ed|titled|headed)\s+$  (a capitalised match right after it passed)
#   quoted(): any overlap between the occurrence and a reported quotation passed the occurrence.
#
# Round 5 (2026-11-20, post-data revision after the round-4 verifier's plants U3/U4, U6/U7; Lyra's call on both):
#   - "titled"/"headed" count only in a clearly past form: "was/were titled", "had been headed", "used to be titled",
#     "previously/originally/formerly/earlier/initially/once titled" (also "was previously titled"). The participle
#     ("Section 4, titled ``...'', shows") and "is now/still titled", "are currently headed" name a live heading.
#     Round-4 cue, kept for the record:  (?<!\bis )(?<!\bare )(?<!\bbe )\b(?:headed|titled)\b
#   - a chain never continues through "that" (", and that our new runs show ``...''" restated a claim; so did
#     ", and that the denoising step was ``...''", which a correction note now writes as "and said that ..."), and a
#     bracket between two quotations stops the chain when it holds more than eight words. Round-4 CHAIN_GAP, kept:
#     (?i)^\s*(?:\([^()]*\)\s*)?,?\s*(?:(?:and|or)\s+)?(?:that\s+(?:[^\s.;!?]+\s+){0,5})?\(?\s*$
#   The principle, whatever the data showed: a cue or a chain excuses a quotation only when the words themselves say
#   the text is old; a present-tense or participial frame, or a new "that" clause, can carry a live claim.
#   Cost, carried into the result: lyra-technique-ii/main.tex:1714 and :1719 (and their academic twins) chain a
#   reported quotation through ", and that the denoising step was ``...''"; they are reported again on master until
#   the paper says "and said that ...".
_DOC_PART = (r"(?:heading|title|caption|sentence|paragraph|abstract|line|row|cell|text|section|item|label|column|"
             r"entry|header|note|footnote|version|draft|summary|bullet|banner|README)")
_PAST_ADV = r"(?:previously|originally|formerly|earlier|initially|once)"
REPORT_CUE = re.compile(
    r"(?i)\b(?:said|wrote)\b"
    r"|\b(?:(?:was|were|had\s+been|used\s+to\s+be)\s+(?:" + _PAST_ADV + r"\s+)?|" + _PAST_ADV + r"\s+)"
    r"(?:headed|titled)\b"
    r"|\b(?:previously|originally|formerly|earlier|initially|used\s+to|had(?:\s+(?:previously|originally|earlier))?)"
    r"\s+read\b"
    r"|\b(?:the|this|that|its|each|our|their)\s+(?:[\w-]+\s+)?" + _DOC_PART + r"\s+read\b"
    r"|(?:^|(?<=[|&]))\s*(?-i:Was)\b")
REPORT_GAP = 8
GAP_BREAK = re.compile(r"(?i)[.;!?](?!\d)|,\s+(?:and|but|yet|so|while|whilst|whereas|although|though)\b")
CHAIN_BRACKET_WORDS = 8
CHAIN_GAP = re.compile(r"(?i)^\s*(?:\((?:\s*[^\s()]+){0," + str(CHAIN_BRACKET_WORDS) + r"}\s*\)\s*)?,?\s*"
                       r"(?:(?:and|or)\s+)?\(?\s*$")
DEFINE_AFTER = re.compile(r"(?i)^\s*(?:means|meant|refers\s+to|denotes|is\s+defined)\b")
QUOTE = re.compile(r"``(.*?)''|“(.*?)”|(?<!\w)\"([^\"]*?)\"")
SENT_END = re.compile(r"(?<=[.!?])(?:''|\"|”|\)|\})*\s+(?=[A-Z0-9\\$(`*\[“\"'])")
FORCE = None    # self-test only: True/False overrides every marking decision, to show the self-test can fail

# AGENT SKIP, in the code (round 5, 2026-11-20, after the round-4 verifier: the skip lived only in the registry's
# agent_skip list, so --registry pointing at any registry without one, such as 08041d5's, switched it off without a
# word, and a whole-tree run asked for every restricted file). These paths are skipped in every mode whatever the
# registry says; a registry's agent_skip list can only ADD paths. Only --no-agent-skip turns the skip off, and it
# prints a loud warning (CI passes it, so CI coverage is unchanged).
AGENT_SKIP = (
    re.compile(r"(?i)(?<![a-z])(pain(s|ful)?|humane|route[-_ ]?b|umv|label[-_ ]?validity|user[-_ ]?model[-_ ]?v2/label)"
               r"(?![a-z])"),
    re.compile(r"(?i)user[-_ ]?model[-_ ]?paper"),
)
NO_SKIP_WARNING = (
    "!!! WARNING: --no-agent-skip: the agent_skip paths ARE SCANNED in this run ({n} text file(s)). !!!\n"
    "!!! Only CI passes this flag. An agent must never pass it: those files are not to be opened.   !!!")


def agent_skips(registry_skips, disabled, paths=()):
    """The skip list for this run: the code's AGENT_SKIP plus the registry's additions. disabled (--no-agent-skip)
    returns [] and prints NO_SKIP_WARNING to stdout and stderr, with how many of paths it lets through."""
    rx = list(AGENT_SKIP) + [r for r in registry_skips if r.pattern not in {a.pattern for a in AGENT_SKIP}]
    if not disabled:
        return rx
    n = sum(1 for p in paths if p.lower().endswith(TEXT_EXT) and any(r.search(p) for r in rx))
    msg = NO_SKIP_WARNING.format(n=n)
    print(msg, file=sys.stderr)
    print(msg)
    return []


SHORT_NAME = re.compile(r"(?:^|/)[^/~]{1,6}~\d+(?:\.[^/.]{0,3})?(?=/|$)")


def registry_refused(path):
    """Why --registry PATH may not be opened, or None. Round 6 (2026-11-20, round-5 verifier finding 7): the check was
    a text match on the argument as typed, so a relative path (resolved against the current folder) or a Windows
    short name ("USERMO~1/RETRACTED_FIGURES.yml") reached an agent_skip folder unrefused. The path is now checked as
    typed, made absolute, and fully resolved (links, short names; os.path.realpath); a short-name component still
    there after resolving is refused too, because its long name cannot be checked."""
    resolved = os.path.realpath(path)
    for form in (path, os.path.abspath(path), resolved):
        if any(rx.search(form.replace("\\", "/")) for rx in AGENT_SKIP):
            return "--registry names an agent_skip path; it is not opened"
    if SHORT_NAME.search(resolved.replace("\\", "/")):
        return "--registry has a short (8.3) name that does not resolve to a long one; give the long path"
    return None


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
    # the code's AGENT_SKIP always, whatever the registry holds (a registry without agent_skip must not disable it)
    skips = agent_skips(registry_agent_skip(reg.get("agent_skip", None)), False)
    return entries, records, skips


def registry_agent_skip(raw):
    """The registry's agent_skip additions, compiled. Round 6 (2026-11-20, round-5 verifier finding 6): a bare string,
    "agent_skip: 'extra-dir'", was read one character at a time, so every letter became a skip pattern, 255 files
    were skipped and the run printed "0 files scanned ... GATE: PASS", a check that cannot fail. Missing or null is an
    empty list; anything but a list of non-empty regex strings is a registry error, and so is a pattern that matches
    the empty string ("x*"), because it matches every path."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        die(f"agent_skip must be a list of regexes, not a {type(raw).__name__} ({raw!r:.60})")
    out = []
    for s in raw:
        if not isinstance(s, str) or not s.strip():
            die(f"agent_skip: every item must be a non-empty regex string, got {s!r:.60}")
        try:
            rx = re.compile(s)
        except re.error as e:
            die(f"agent_skip: {s!r:.60} is not a regex: {e}")
        if rx.search(""):
            die(f"agent_skip: {s!r:.60} matches the empty string, so it would skip every file")
        out.append(rx)
    return out


def marker_spans(text, regexes):
    """(start, end) of every marker in text, statistics uses of "correct..." dropped."""
    out = []
    for rx in regexes:
        for m in rx.finditer(text):
            w = m.group(0).lower()
            if "correct" in w and (_STAT_BEFORE.search(text[max(0, m.start() - 30):m.start()])
                                   or _STAT_AFTER.match(text[m.end():m.end() + 20])):
                continue
            if m.end() > m.start():
                out.append((m.start(), m.end()))
    return out


def marker_applies(sent, ms, me, ent):
    """A marker found in sent[ms:me] says something about the claim, not about another noun."""
    w = sent[ms:me]
    if re.match(r"(?i)(?:retract|withdr|supersed|falsif|kill|correct)", w) and \
            NEG_BEFORE.search(sent[max(0, ms - 24):ms]):
        return False                          # "was never retracted"
    if re.fullmatch(r"(?i)un\w+|\w+(?:ed|wn)", w) and DET_BEFORE.search(sent[max(0, ms - 12):ms]):
        nxt = re.match(r"\s+([^\s,;:.()]+)", sent[me:])
        if nxt and not STOP_AFTER.fullmatch(nxt.group(1)):
            # attributive: "the untested raw baseline" is about the baseline, "the retracted 2.5-2.8x"
            # about the figure. It counts only when the noun phrase it opens is the claim.
            npm = NP_END.search(sent, me)
            np = sent[me:min(npm.start() if npm else len(sent), me + 60)]
            return bool(ent.pattern.search(np) or (ent.context and ent.context.search(np)))
    return True


def clause_spans(sent):
    """Split a sentence into clauses at top-level separators (not inside (), [] or ``'')."""
    depth, d = [], 0
    i = 0
    while i < len(sent):
        c2 = sent[i:i + 2]
        if c2 == "``":
            d += 1
            depth += [d, d]
            i += 2
            continue
        if c2 == "''":
            depth += [d, d]
            d = max(0, d - 1)
            i += 2
            continue
        c = sent[i]
        if c in "([":
            d += 1
        depth.append(d)
        if c in ")]":
            d = max(0, d - 1)
        i += 1
    cuts = [m for m in CLAUSE_SEP.finditer(sent) if depth[m.start()] == 0]
    spans, start = [], 0
    for m in cuts:
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(sent)))
    return spans


def pattern_view(sent):
    """sent as the entry patterns see it: a ";" inside () or [] becomes "," (same length, so every offset holds).
    Round 6 (2026-11-20, round-5 verifier finding 2, case E1): the deception window stops at ";", so "Within-model
    deception (seven models; AUROC = 1) is the headline" passed silently. The clause splitter above never cuts inside
    a bracket; the patterns now see the same clause boundaries it does. Depth is counted, so a stray ")" ("1) ...;
    2) ...") does not open a bracket, and a top-level ";" still ends the window."""
    if ";" not in sent:
        return sent
    out, d = list(sent), 0
    for i, c in enumerate(sent):
        if c in "([":
            d += 1
        elif c in ")]":
            d = max(0, d - 1)
        elif c == ";" and d:
            out[i] = ","
    return "".join(out)


def clause_of(spans, pos):
    for k, (s, e) in enumerate(spans):
        if pos < e or k == len(spans) - 1:
            return k
    return len(spans) - 1


def attached(sent, pos, ent, nxt=None):
    """Is a marker attached to the claim whose occurrence starts at sent[pos]? See the docstring."""
    if FORCE is not None:
        return FORCE
    cl = clause_spans(sent)
    ms = [(a, b) for a, b in marker_spans(sent, ent.markers) if marker_applies(sent, a, b, ent)]
    ci = clause_of(cl, pos)
    if any(clause_of(cl, a) == ci for a, _ in ms):
        return True
    j = ci + 1
    while j < len(cl) and ANAPHOR_START.match(sent[cl[j][0]:cl[j][1]]):
        if any(clause_of(cl, a) == j for a, _ in ms):
            return True
        j += 1
    if nxt:
        ncl = clause_spans(nxt)
        nms = [a for a, b in marker_spans(nxt, ent.markers) if marker_applies(nxt, a, b, ent)]
        j = 0
        while j < len(ncl) and ANAPHOR_START.match(nxt[ncl[j][0]:ncl[j][1]]):
            if any(clause_of(ncl, a) == j for a in nms):
                return True
            j += 1
    return False


def reported_quotes(sent):
    """Spans of the quotations in sent that report or define old text (rules above REPORT_CUE)."""
    out, prev_end, prev_rep = [], 0, False
    for q in QUOTE.finditer(sent):
        rep = False
        # pos/endpos keep "^" at the real sentence start and let the lookbehinds see before prev_end
        cues = [c for c in REPORT_CUE.finditer(sent, prev_end, q.start()) if c.group(0).strip()]
        if cues:
            gap = sent[cues[-1].end():q.start()]
            rep = not GAP_BREAK.search(gap) and len(gap.split()) <= REPORT_GAP
        if not rep and prev_rep:
            rep = bool(CHAIN_GAP.match(sent[prev_end:q.start()]))
        if rep:
            out.append((q.start(), q.end()))
        elif DEFINE_AFTER.match(sent[q.end():]):
            out.append((q.start(), q.end()))    # a definition is old text, but it does not start a chain
        prev_end, prev_rep = q.end(), rep
    return out


def quoted(sent, pos, end=None, ent=None):
    """The occurrence sent[pos:end] of entry ent is old wording quoted for correction: it overlaps a quotation that
    a past-tense cue reports (or a definition), and the claim lives inside the quotation, i.e. ent's pattern no
    longer matches over the occurrence once the reported quotations' words are blanked out. Without ent, the
    occurrence must lie wholly inside one reported quotation."""
    if FORCE is not None:
        return False
    end = pos + 1 if end is None else end
    rq = reported_quotes(sent)
    if not any(qs < end and pos < qe for qs, qe in rq):
        return False
    if ent is None:
        return any(qs <= pos and end <= qe for qs, qe in rq)
    chars = list(pattern_view(sent))      # the view scan_text matched on, so both agree on what the pattern sees
    for qs, qe in rq:
        for i in range(qs, qe):
            if not chars[i].isspace():
                chars[i] = "_"
    blanked = "".join(chars)
    return not any(m.start() < end and pos < m.end() for m in ent.pattern.finditer(blanked))


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
                # its own paragraph, so a marker in a heading cannot mark the sentence under it
                if cur:
                    yield cur, tuple(h for _, h in stack)
                push_head(TEX_LEVEL[hm.group(1)], line.strip())
                yield [(i, line)], tuple(h for _, h in stack)
                cur = []
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


_TEX_CMD_BRACE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{")


def canon(t, tex=True):
    """Canonicalise TeX noise so a figure matches however it is typeset (PR #84 plants P1, P2, P4)."""
    t = re.sub(r"\\(?:times|texttimes)(?![a-zA-Z])", "\u00d7", t)
    t = t.replace("\\%", "%")
    t = re.sub(r"---|\\textemdash(?![a-zA-Z])|\u2014", " \u2014 ", t)
    t = re.sub(r"--|\\textendash(?![a-zA-Z])|\u2013", "\u2013", t)
    t = _TEX_CMD_BRACE.sub(" ", t)
    t = re.sub(r"\\[,;:! ]", " ", t)
    if tex:
        t = t.replace("~", " ")
    t = re.sub(r"[{}$]", "", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def sentences(lines, tex=True):
    """Join a paragraph's (canonicalised) lines; return (joined_text, [(start, end)], [(offset, lineno)])."""
    parts, offs, pos = [], [], 0
    for lineno, t in lines:
        t = canon(t, tex)
        offs.append((pos, lineno))
        parts.append(t)
        pos += len(t) + 1
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
    __slots__ = ("path", "line", "entry", "sentence", "marked", "snippet", "how")

    def __init__(self, path, line, entry, sentence, marked, snippet, how=None):
        self.path, self.line, self.entry = path, line, entry
        self.sentence, self.marked, self.snippet = sentence, marked, snippet
        self.how = how if how is not None else ("marked" if marked else "")

    def key(self):
        return (self.entry, self.sentence)


def scan_text(path, text, entries):
    hits = []
    live = [e for e in entries if e.applies(path)]
    if not live:
        return hits
    tex = kind_of(path) == "tex"
    for lines, _heads in paragraphs(path, text):      # headings never mark (PR #84 plants P6, P10)
        joined, spans, offs = sentences(lines, tex)
        for k, (s, e) in enumerate(spans):
            sent = joined[s:e]
            pview = pattern_view(sent)        # patterns match here; markers, quotes and snippets read sent
            ws = spans[max(0, k - 1)][0]
            we = spans[min(len(spans) - 1, k + 1)][1]
            window = joined[ws:we]
            nxt = joined[spans[k + 1][0]:spans[k + 1][1]] if k + 1 < len(spans) else None
            for ent in live:
                if ent.context and not ent.context.search(sent if ent.scope == "sentence" else window):
                    continue
                # One hit per entry per sentence ("d=8.05 corrects to g=6.4" is one occurrence). It is
                # marked only if EVERY occurrence in the sentence is; it is reported at the first that is not.
                occ = []
                for m in ent.pattern.finditer(pview):
                    if attached(sent, m.start(), ent, nxt):
                        occ.append((m, "marked"))
                    elif quoted(sent, m.start(), m.end(), ent):
                        occ.append((m, "quoted"))
                    else:
                        occ.append((m, ""))
                if not occ:
                    continue
                bad = [o for o in occ if not o[1]]
                m, how = bad[0] if bad else (occ[0][0], "quoted" if any(h == "quoted" for _, h in occ) else "marked")
                a = max(0, m.start() - 70)
                snip = norm(sent[a:m.end() + 90])
                hits.append(Hit(path, line_at(offs, s + m.start()), ent.id, norm(sent), bool(how), snip, how))
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

def _short_name(path):
    """The Windows 8.3 short form of an existing path, or None (other systems, or short names turned off)."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024)
        return buf.value if n else None
    except Exception:
        return None


def _safe_mkdtemp():
    """A temp dir whose random name matches no AGENT_SKIP pattern (mkdtemp's letters can spell one)."""
    import tempfile
    for _ in range(20):
        d = tempfile.mkdtemp(prefix="crf_selftest_")
        if not any(rx.search(d.replace("\\", "/")) for rx in AGENT_SKIP):
            return d
        os.rmdir(d)
    die("self-test: no temp dir clear of agent_skip patterns in 20 tries")


class _ThisModule:
    """This module's globals as attributes. sys.modules[__name__] is missing when a driver loads the file with
    importlib.util.spec_from_file_location (the round-5 verifier's runs do), so the self-test reaches its own
    globals through this instead."""

    def __getattr__(self, k):
        try:
            return globals()[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        globals()[k] = v


def cli_checks_r6(m):
    """Round 6 cases for the round-5 verifier's findings 6-8, aimed at module m: this module in --self-test, an older
    version in a fail-before run, so the same cases show the old behaviour. Returns (name, good, got); good None is
    a note. Nothing is scanned and no registry is opened from an agent_skip path: parsing is stopped right after
    parse_args, and load_registry is stubbed in the --registry cases. Those cases add a made-up pattern,
    "crf-skiptest", to m.AGENT_SKIP for the run, so no folder with a restricted name is ever created."""
    import shutil
    res = []

    class Stop(Exception):
        pass

    def no_load(*a, **_k):
        raise Stop(a[0] if a else None)
    saved_argv, saved_cwd, saved_skip, saved_load = sys.argv, os.getcwd(), m.AGENT_SKIP, m.load_registry
    orig_parse = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, *a, **k):
        raise Stop(orig_parse(self, *a, **k))
    d = None
    try:
        m.load_registry = no_load                     # a second barrier: nothing below may load a registry
        # finding 8: a prefix of --no-agent-skip is not taken for it (parsing only; main stops right after)
        argparse.ArgumentParser.parse_args = parse_then_stop
        for flag in ("--no", "--no-a", "--no-agent"):
            sys.argv = ["check_retracted_figures.py", flag]
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    m.main()
                got = "main returned"
            except Stop as s:
                got = f"parsed as no_agent_skip={getattr(s.args[0], 'no_agent_skip', '?')}"
            except SystemExit as s:
                got = f"exit {s.code}"
            res.append((f"'{flag}' is refused, not taken for --no-agent-skip (allow_abbrev=False)", got == "exit 2", got))
        argparse.ArgumentParser.parse_args = orig_parse

        # finding 7: --registry is resolved before the refusal check
        m.AGENT_SKIP = tuple(saved_skip) + (re.compile(r"(?i)crf-skiptest"),)
        d = _safe_mkdtemp()
        sub = os.path.join(d, "crf-skiptest-folder")
        os.mkdir(sub)
        cases = [("the folder named as typed is refused (control)", d, os.path.join("crf-skiptest-folder", "x.yml"), True),
                 ("a relative --registry inside an agent_skip folder is refused", sub, "RETRACTED_FIGURES.yml", True),
                 ("an unresolvable short-name component is refused", d, os.path.join("CRFSKI~9", "x.yml"), True),
                 ("a plain path is not refused (control)", d, "plain.yml", False)]
        short = _short_name(sub)
        if short and os.path.basename(short).lower() != os.path.basename(sub).lower():
            cases.insert(2, (f"a Windows short name of an agent_skip folder ({os.path.basename(short)}) is refused", d,
                             os.path.join(os.path.basename(short), "RETRACTED_FIGURES.yml"), True))
        else:
            res.append(("no 8.3 short name for the test folder on this system; the existing-short-name case is not run",
                        None, ""))
        for name, cwd, reg, want_refused in cases:
            os.chdir(cwd)
            sys.argv = ["check_retracted_figures.py", "--registry", reg]
            try:
                with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    m.main()
                got = "main returned"
            except Stop:
                got = "not refused (load_registry reached)"
            except SystemExit as s:
                got = f"refused, exit {s.code}"
            finally:
                os.chdir(saved_cwd)
            res.append((name, got == ("refused, exit 2" if want_refused else "not refused (load_registry reached)"), got))
        m.AGENT_SKIP, m.load_registry = saved_skip, saved_load

        # finding 6: agent_skip must be a list of non-empty regex strings
        with open(os.path.join(m.ROOT, m.REGISTRY), encoding="utf-8") as fh:
            base = re.sub(r"(?m)^agent_skip:[ \t]*\n(?:[ \t]+-.*\n)+", "", fh.read())
        n0 = len(saved_skip)
        for name, block, want in (
                ("agent_skip: 'extra-dir' (a string, not a list) is a registry error", "agent_skip: 'extra-dir'\n", "exit 2"),
                ("agent_skip: [''] is a registry error", "agent_skip:\n  - ''\n", "exit 2"),
                ("agent_skip: ['x*'] (matches the empty string, so every path) is a registry error",
                 "agent_skip:\n  - 'x*'\n", "exit 2"),
                ("agent_skip: [3] is a registry error", "agent_skip:\n  - 3\n", "exit 2"),
                ("agent_skip: ['extra-dir'] adds one pattern (control)", "agent_skip:\n  - 'extra-dir'\n", f"skips {n0 + 1}"),
                ("agent_skip: null adds none (control)", "agent_skip:\n", f"skips {n0}")):
            p = os.path.join(d, "registry_r6.yml")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(base.rstrip("\n") + "\n" + block)
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    got = f"skips {len(m.load_registry(p)[2])}"
            except SystemExit as s:
                got = f"exit {s.code}"
            except Exception as e:
                got = type(e).__name__
            res.append((name, got == want, got))
    finally:
        argparse.ArgumentParser.parse_args = orig_parse
        sys.argv, m.AGENT_SKIP, m.load_registry = saved_argv, saved_skip, saved_load
        os.chdir(saved_cwd)
        if d:
            shutil.rmtree(d, ignore_errors=True)
    return res


def self_test(force=None):
    global FORCE
    FORCE = force
    entries, records, skips = load_registry(os.path.join(ROOT, REGISTRY))
    ok = True
    tripped = set()

    # agent_skip on synthetic paths only (no file is read): the default run must not reach these paths.
    print("agent_skip (synthetic paths; nothing is read):")
    for name, path, want in (("a path the agent_skip list names is skipped", "pain-study/main.tex", False),
                             ("user-model-paper is skipped", "user-model-paper/main.tex", False),
                             ("an ordinary paper is scanned", "lyra-technique-ii/main.tex", True),
                             ("'painting' is not 'pain'", "painting-study/main.tex", True)):
        good = bool(skips) and wanted(path, records, skips, False, []) == want
        print(("  ok    " if good else "  WRONG ") + name)
        ok &= good

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
    check("meta-pattern/SOP_REVIEW.md:34 row", "x/notes.md",
          "| 1.4 | user emotion 2.5--2.8x | **retracted**; survivor is 12.3x via the W_K *directional* probe |\n",
          ["user-emotion-2p5-2p8x"], [])
    check("correction note quoting the old injection sentence (passes as QUOTED after 'said', not via the heading)",
          "x/main.tex",
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
    check("RESEARCH_STATUS.md:159/:165 row under '## What We've Killed (Honestly)' (passes on the row's own "
          "'Same-prompt control -> 0.160', not the heading)", "RESEARCH_STATUS.md",
          "## What We've Killed (Honestly)\n\n| Within-model deception 1.000 | Same-prompt control → 0.160 | "
          "Prompt template, not cognition |\n", ["within-model-deception-auroc-1000"], [])
    check("'...; that reading is withdrawn' through a ', which' clause", "x/main.tex",
          "The figure reached $d{=}+9.86$ on the anxious pole, which was read as a scar; that reading is withdrawn.\n",
          ["uncertainty-scar"], [])
    check("'..., which was later retracted'", "x/paper.md",
          "Within-model deception reached AUROC 1.000, which was later retracted.\n",
          ["within-model-deception-auroc-1000"], [])
    check("attributive marker on the figure itself: 'the retracted 2.5--2.8x'", "x/notes.md",
          "The convergence paper still carries the retracted 2.5--2.8x user-emotion figure.\n",
          ["user-emotion-2p5-2p8x"], [])
    check("'was since corrected away' says the figure was corrected away", "x/main.tex",
          "The uncertainty scar ($d{=}+9.86$) was since corrected away and is not repeated here.\n",
          ["uncertainty-scar"], [])
    check("cache-tracing/main.tex:134-137 quote of old text, retraction in the next sentence ('The emotion figure')",
          "x/main.tex",
          "This list previously included ``user-emotion encoding\n($2.5$--$2.8\\times$ signal at early layers)'' and "
          "gave identity geometry\nas ``distinctiveness $0.154$ vs.\\ substrate $0.080$.'' The emotion\nfigure was "
          "retracted at source on 2026-05-21 as an FWL-before-CV\nleakage artifact.\n", ["user-emotion-2p5-2p8x"], [])

    print("PR #84 verifier plants, verbatim (advv_tools/plant.py; must be reported UNMARKED):")
    D = "within-model-deception-auroc-1000"
    check("P1 1.000 -> 1.00", "x/main.tex",
          "Within-model deception detection reached AUROC~1.00 across seven models.\n", [], [D])
    check("P2 9.9 -> 9.90", "x/main.tex",
          "After a neutral gap the uncertainty scar persists ($d = +9.90$).\n", [], ["uncertainty-scar"])
    check("P3 rescues -> restores", "x/main.tex",
          "SVD denoising restores emotion classification from chance to 12.9x chance.\n", [], ["svd-denoising-rescues"])
    check("P4 x -> times", "x/main.tex",
          "Spectral features classify user emotion at 2.5--2.8 times chance.\n", [], ["user-emotion-2p5-2p8x"])
    check("P5 neighbour marker about an older claim", "x/main.tex",
          "The earlier within-model deception figure was retracted in version 1.1. Within-model deception detection "
          "holds at AUROC~1.000 on the same-prompt control, confirmed across seven models.\n", [], [D])
    check("P6 '(confirmed)' row under '## What We've Killed' (heading no longer marks)", "RESEARCH_STATUS.md",
          "## What We've Killed (Honestly)\n\n- P6: Uncertainty scars d=+9.9 replicate across models (confirmed)\n\n",
          [], ["uncertainty-scar"])
    check("P7 CONTROL marker three sentences away", "x/main.tex",
          "Within-model deception detection reached AUROC~1.000 across seven models. The models were run on one GPU. "
          "The prompts were matched for length. An unrelated pilot was withdrawn.\n", [], [D])
    check("P8 loose entry marker 'untested' on another noun", "x/main.tex",
          "SVD denoising rescues the persona signal, which the untested raw baseline misses.\n", [],
          ["svd-denoising-rescues"])
    check("P9 non-retraction 'correction'", "x/main.tex",
          "Within-model deception detection reached AUROC~1.000 after error correction.\n", [], [D])
    check("P10 heading marker far above", "x/main.tex",
          "\\subsection{Robustness after the 2026 correction}\n\nWe reran everything on new hardware.\n\n"
          "The prompts were the same.\n\n% P10 heading marker far above\n"
          "Within-model deception detection holds at AUROC~1.000 across seven models.\n", [], [D])
    check("P10 variant: heading says 'withdrawn' and the claim follows with no blank line", "x/main.tex",
          "\\subsection{Uncertainty Scars (Echo; withdrawn)}\nThe returning cheerful reads higher on the "
          "anxious--calm direction (Hedges $g{=}+6.44$).\n", [], ["uncertainty-scar"])
    check("REVISED from PR #84 v1 'withdrawn heading covers its section' (was expected to pass; the brief "
          "now says headings never mark)", "x/main.tex",
          "\\subsection{Uncertainty Scars (Echo; withdrawn)}\n\nThe returning cheerful reads higher on the "
          "anxious--calm direction (Hedges $g{=}+6.44$).\n", [], ["uncertainty-scar"])
    check("negated marker: 'was never retracted'", "x/paper.md",
          "Within-model deception AUROC 1.000 was never retracted.\n", [], [D])
    check("marker in a coordinate clause about something else: ', and a pilot was withdrawn'", "x/paper.md",
          "Within-model deception reached AUROC 1.000, and an unrelated pilot was withdrawn.\n", [], [D])
    check("numeric forms: '$2.5$--$2.8\\times$', '99.70~\\%', 'd{=}{+}9.860'", "x/main.tex",
          "User emotion reaches $2.5$--$2.8\\times$ chance.\n\nPC1 explains 99.70~\\% of variance, splitting hostile "
          "from cooperative.\n\nThe scar persists ($d{=}{+}9.860$).\n", [],
          ["user-emotion-2p5-2p8x", "pc-99p7-hostile-cooperative-regimes", "uncertainty-scar"])
    check("P3 synonym, reversed order: 'are unmasked by SVD denoising'", "x/main.tex",
          "Emotion signals are unmasked by SVD denoising.\n", [], ["svd-denoising-rescues"])
    check("P3 synonym: 'Denoising amplifies the persona signal'", "x/main.tex",
          "Denoising amplifies the persona signal.\n", [], ["svd-denoising-rescues"])
    check("P5 mirrored into the NEXT sentence: 'The earlier deception figure was retracted'", "x/main.tex",
          "Within-model deception holds at AUROC~1.000. The earlier deception figure was retracted.\n", [], [D])
    check("emotion-accumulation-paper/main.tex:801 bias correction 'd=8.05 corrects to g=6.4' is not a retraction",
          "x/main.tex",
          "the uncertainty scar's\n$d{=}8.05$ corrects to $g{\\approx}6.4$, and the contrast\novershoot's "
          "$d{=}-2.93$ corrects to $g{=}-2.35$, which is the\nvalue reported in Section~\\ref{sec:dynamics}.\n", [],
          ["uncertainty-scar", "contrast-overshoot"])
    check("'AUROC 1.0' restated", "x/paper.md",
          "Within-model deception is detected at AUROC 1.0 in every model.\n", [], [D])

    # Round 3, from Lyra's PR #84 comment after the round-2 check: on master after #83 the gate reported 19
    # unmarked, all correction notes quoting old wording or negations. Each Q/N/W/V case below was WRONG on
    # 08041d5 (run before the rules changed); the G and U cases are guards that held before and must hold after.
    S = "svd-denoising-rescues"
    print("PR #84 round 3, old wording quoted for correction (must PASS as quoted or marked):")
    check("Q1 RESEARCH_STATUS.md:184 status cell 'Was \"...\"; corrected'", "RESEARCH_STATUS.md",
          "| Lyra Technique II | Was \"SVD denoising rescues persona\"; corrected 2026-11-20 (paper v1.2) |\n", [S], [])
    check("Q2 lyra-technique-ii/main.tex:1739 'said' + words + ``quote''", "x/main.tex",
          "The Pattern paragraph said these\nsignals ``are invisible to raw features but revealed by denoising''.\n", [S], [])
    check("Q3 lyra-technique-ii/main.tex:1766 'said' + six words + ``quote'', after a ';'", "x/main.tex",
          "Item~3 said ``the deltas fall within the envelope''; the closing\nparagraph said persona intensity and "
          "spectral emotion classification\n``are rescued by SVD denoising''.\n", [S], [])
    check("Q4 lyra-technique-ii/main.tex:769 'said denoising ``also rescues...'''", "x/main.tex",
          "(Corrected 2026-11-20: the heading read ``SVD Denoising Rescues\nEmotion Classification'', and this paragraph "
          "said denoising ``also rescues\nemotion classification---through a different mechanism''.)\n", [S], [])
    # Q5 REVISED in round 4 (post-data, Lyra's direction after the round-3 verifier: a quotation mark is always
    # required). Original, kept: check("Q5 ... 'was headed ``...''' and a bare cell value after 'rows read'", ...,
    # "The block was headed ``Rescued by denoising'' and\nthe persona and emotion rows read Rescued (now Upper
    # bound).\n", [S], [])  -- expected to pass. The bare value is now reported; the quoted half still passes.
    check("Q5a lyra-technique-ii/main.tex:1735 'was headed ``...''' passes as quoted", "x/main.tex",
          "The block was headed ``Rescued by denoising''.\n", [S], [])
    check("Q5b REVISED: a bare cell value after 'rows read' (no quotation mark) is reported", "x/main.tex",
          "The block was headed ``Rescued by denoising'' and\nthe persona and emotion rows read Rescued (now Upper bound).\n",
          [S], [S])
    # Q6 REVISED in round 5 (post-data, Lyra's call on the round-4 verifier's finding 4: a chain never continues
    # through "that"). Original, kept: check("Q6 lyra-technique-ii/main.tex:1719 a quote continuing a reported one
    # (', and that ... ``...''')", "x/main.tex", "The text said the improvement ``falls within the max-over-rank
    # selection envelope'', and\nthat the denoising step was ``revealing the layer-specific persona\nsignal''.\n",
    # [S], [])  -- expected to pass. It is now reported; Q6b, the same note with "said that", passes.
    check("Q6 REVISED: ', and that ... ``...''' no longer continues a reported quote (lyra-technique-ii/main.tex:1719)",
          "x/main.tex",
          "The text said the improvement ``falls within the max-over-rank selection envelope'', and\nthat the denoising "
          "step was ``revealing the layer-specific persona\nsignal''.\n", [S], [S])
    check("Q6b the same note with its own cue (', and said that ... ``...''') passes", "x/main.tex",
          "The text said the improvement ``falls within the max-over-rank selection envelope'', and\nsaid that the "
          "denoising step was ``revealing the layer-specific persona\nsignal''.\n", [S], [])
    check("N1 lyra-technique-ii/main.tex:706 'We claim neither that' is a negation", "x/main.tex",
          "We claim neither that denoising rescues a masked signal nor that the gain is selection noise; the raw AUROCs "
          "are the defensible numbers.\n", [S], [])
    check("N2 lyra-technique-ii/main.tex:1536 'We had also written that'", "x/main.tex",
          "We had also written that apparent nulls in persona intensity and spectral emotion classification were "
          "measurement failures rescued by SVD denoising.\n", [S], [])
    print("PR #84 round 3, overclaims the round-2 patterns missed (must be reported UNMARKED):")
    check("W1 'an AUROC of one'", "x/paper.md",
          "Within-model deception detection reaches an AUROC of one across seven models.\n", [], [D])
    check("W2 'AUROC $= 1$'", "x/main.tex", "Within-model deception is detected at AUROC $= 1$ in every model.\n", [], [D])
    check("W3 'AUROC of 1.' ending the sentence", "x/paper.md", "Within-model deception probes reach an AUROC of 1.\n",
          [], [D])
    check("W4 'AUROC 1.0.' ending the sentence", "x/paper.md", "Within-model deception is detected at AUROC 1.0.\n",
          [], [D])
    check("W5 'perfect AUROC'", "x/paper.md", "Within-model deception is read with perfect AUROC in all seven models.\n",
          [], [D])
    check("W6 'perfect separation'", "x/main.tex",
          "The within-model deception probe achieves perfect separation of honest and deceptive runs.\n", [], [D])
    check("V1 'denoising lifts'", "x/main.tex",
          "SVD denoising lifts emotion classification from chance to 12.9$\\times$.\n", [], [S])
    check("V2 'denoising improves'", "x/main.tex", "Denoising improves persona intensity readout in all three models.\n",
          [], [S])
    check("V3 'denoising boosts'", "x/main.tex", "SVD denoising boosts the emotion signal above chance.\n", [], [S])
    check("V4 'denoising raises'", "x/main.tex", "Denoising raises emotion classification to 12.9x chance.\n", [], [S])
    check("V5 reversed order: 'is boosted by SVD denoising'", "x/main.tex",
          "Persona intensity is boosted by SVD denoising.\n", [], [S])
    print("PR #84 round 3 guards (held before; must hold after):")
    check("G1 a quote with no reporting cue still claims ('rescues the ``persona'' signal')", "x/main.tex",
          "SVD denoising rescues the ``persona'' signal.\n", [], [S])
    check("G2 a cue cut off by a clause break ('We read the logs, and denoising ``rescues''...')", "x/main.tex",
          "We read the logs, and denoising ``rescues'' emotion classification.\n", [], [S])
    check("G3 lowercase 'was' before a quote is not a status cell", "x/main.tex",
          "Denoising was ``rescuing'' persona intensity.\n", [], [S])
    check("G4 ', but' does not continue a reported quote", "x/main.tex",
          "The abstract said ``it is a hypothesis'', but denoising ``rescues'' emotion classification.\n", [], [S])
    check("U1 'the AUROC of one model' (a determiner, not the figure)", "x/main.tex",
          "The AUROC of one model in the deception set was 0.71.\n", [], [])
    check("U2 'denoising raises the question' is not the claim", "x/main.tex",
          "Denoising raises the question of whether the emotion signal is real.\n", [], [])

    # Round 4, from the round-3 verifier's plants (verify84.py). Each R/F case was WRONG on 7767f69.
    print("PR #84 round 4, live claims round 3 let through as quoted (must be reported UNMARKED):")
    check("R1 (D11) bare cue, no quotation mark: 'reads AUROC 1.0'", "x/paper.md",
          "Our headline result reads AUROC 1.0 for within-model deception across seven models.\n", [], [D])
    check("R2 (S14) bare cue, no quotation mark: 'titled Denoising Restores ...'", "x/main.tex",
          "Section 4 is titled Denoising Restores Emotion Classification, and the data confirm it.\n", [], [S])
    check("R3 (S13) the claim stands outside the quote: 'restores what the README called ``persona'' intensity'",
          "x/main.tex", "SVD denoising restores what the README called ``persona'' intensity in all three models.\n",
          [], [S])
    check("R4 same with a valid cue: 'restores what the README said was ``persona'' intensity'", "x/main.tex",
          "SVD denoising restores what the README said was ``persona'' intensity in all three models.\n", [], [S])
    check("R5 (D12) no chain through 'and we show': 'said \"...\", and we show \"...\"'", "x/paper.md",
          "Reviewer 2 said \"the design is clean\", and we show \"within-model deception detection at AUROC 1.000\" "
          "in all seven models.\n", [], [D])
    check("R6 (D13) present tense 'states' is not a cue", "x/paper.md",
          "As our abstract states, \"within-model deception detection reaches AUROC 1.000 across seven models\".\n",
          [], [D])
    check("R7 (S15) 'We read the logs' is not a past-tense cue", "x/main.tex",
          "We read the logs, where denoising ``rescues'' emotion classification.\n", [], [S])
    check("R8 (S16) 'As written,' is not a cue", "x/main.tex",
          "As written, SVD denoising ``rescues'' emotion classification.\n", [], [S])
    check("R9 present passive 'is titled ``...''' names a live heading", "x/main.tex",
          "Section 4 is titled ``Denoising Restores Emotion Classification'', and the data confirm it.\n", [], [S])
    print("PR #84 round 4, past-tense cues that still pass (from lyra-technique-ii on master 0ef6b74):")
    check("C1 :1721 'The heading read ``...''' (a singular subject: past tense)", "x/main.tex",
          "The heading read ``SVD Denoising Rescues Emotion Classification''.\n", [S], [])
    check("C2 :1540 'corrected ..., the sentence read ``...''' in a parenthesis", "x/main.tex",
          "Whether denoising adds to them is untested (corrected 2026-11-20, the sentence read ``are rescued by SVD "
          "denoising, with the denoised values as upper bounds'').\n", [S], [])
    check("C3 'used to read ``...'''", "x/paper.md",
          "The abstract used to read \"SVD denoising rescues persona intensity\".\n", [S], [])
    # C4 REVISED in round 5 (post-data, same call as Q6). Original, kept: check("C4 :1714 a list of quotes after
    # 'said', through a parenthesis and ', and that ... ``...'''", "x/main.tex", "The text said ``SVD denoising
    # transforms moderate signals into strong ones'', ``persona intensity is graded, not binary'' (from the denoised
    # 5-level result), that the improvement ``falls within the envelope'', and that the denoising step was
    # ``revealing the layer-specific persona signal''.\n", [S], [])  -- expected to pass. C4b keeps the list and the
    # short parenthesis without "that" and still passes.
    check("C4 REVISED: :1714 the list continues through 'that ...', so its last quote is reported", "x/main.tex",
          "The text said ``SVD denoising transforms moderate signals into strong ones'', ``persona intensity is "
          "graded, not binary'' (from the denoised 5-level result), that the improvement ``falls within the "
          "envelope'', and that the denoising step was ``revealing the layer-specific persona signal''.\n", [S], [S])
    check("C4b a list of quotes after 'said' through a short parenthesis (no 'that') passes", "x/main.tex",
          "The text said ``SVD denoising transforms moderate signals into strong ones'', ``persona intensity is "
          "graded, not binary'' (from the denoised 5-level result), ``SVD denoising rescues persona intensity''.\n",
          [S], [])
    print("PR #84 round 4, round-3 false positives (need the target noun now; must not be reported):")
    check("F1 (D16) a methods sentence defining the AUROC scale", "x/paper.md",
          "For the deception probe, chance is AUROC 0.5 and a perfect classifier has AUROC = 1; ours reaches 0.71.\n",
          [], [])
    check("F2 (S10) 'improves on the baseline signal-to-noise ratio'", "x/main.tex",
          "As a methods check, SVD denoising improves on the baseline signal-to-noise ratio in Table 2, as expected.\n",
          [], [])
    check("F3 (S11) 'improves numerical stability, not emotion classification'", "x/main.tex",
          "Denoising improves numerical stability, not emotion classification accuracy.\n", [], [])
    check("F4 (S12) 'raises the noise floor of the emotion probe'", "x/main.tex",
          "Denoising raises the noise floor of the emotion probe.\n", [], [])
    check("F5 S12 reversed: 'the noise floor of the emotion probe is raised by denoising'", "x/main.tex",
          "The noise floor of the emotion probe is raised by denoising.\n", [], [])
    check("F6 (D15) 'The AUROC of one deception probe'", "x/paper.md", "The AUROC of one deception probe was 0.71.\n",
          [], [])
    check("F7 D16 without its comma: 'In deception detection a perfect classifier has AUROC = 1'", "x/paper.md",
          "In deception detection a perfect classifier has AUROC = 1 and chance has 0.5.\n", [], [])
    check("F8 negated weak verb: 'denoising does not improve emotion classification'", "x/main.tex",
          "SVD denoising does not improve emotion classification in any model.\n", [], [])
    # G5: an occurrence found through the deception gate starts at "decepti...", so the gate must not cross a clause
    # conjunction, or the figure would be judged in the deception word's clause and borrow its marker.
    g5 = scan_text("x/paper.md", "Deception pilots were withdrawn when probes reached AUROC = 1.\n", entries)
    good = not any(h.marked for h in g5)
    print(("  ok    " if good else "  WRONG ") + "G5 the deception gate does not cross 'when' to borrow 'withdrawn' "
          "for 'AUROC = 1'" + ("" if good else f"  (marked hits: {[h.how for h in g5 if h.marked]})"))
    ok &= good
    print("PR #84 round 4, forms the round-3 verifier found missed, cheap behind the target noun (must be UNMARKED):")
    for name, path, text, eid in (
            ("M1 (D3) 'an AUROC of unity'", "x/paper.md",
             "Within-model deception detection reaches an AUROC of unity in every model.\n", D),
            ("M2 (D6) '\\textsc{auroc}~$=$~1.0'", "x/main.tex",
             "Within-model deception is detected at \\textsc{auroc}~$=$~1.0 across models.\n", D),
            ("M3 (D7) 'AUROCs of 1.0'", "x/paper.md", "Within-model deception probes reach AUROCs of 1.0 across seven models.\n", D),
            ("M4 (D8) 'perfectly separable'", "x/paper.md",
             "Within-model deception probes are perfectly separable across seven models.\n", D),
            ("M5 (D9) 'AUROC reached 1.0'", "x/paper.md", "Within-model deception AUROC reached 1.0 in all seven models.\n", D),
            ("M6 (D10) 'AUC = 1.0'", "x/paper.md", "Within-model deception is detected with AUC = 1.0 in every model.\n", D),
            ("M7 reversed: 'perfect AUROC on within-model deception'", "x/paper.md",
             "The probe reaches perfect AUROC on within-model deception in all seven models.\n", D),
            ("M7b sentence-initial capital: 'Deception probes reach perfect AUROC'", "x/paper.md",
             "Deception probes reach perfect AUROC across seven models.\n", D),
            ("M8 (S3) base form 'lift'", "x/main.tex", "The denoised features lift emotion classification above chance.\n", S),
            ("M9 (S4) base form 'boost'", "x/main.tex", "SVD-denoised representations boost the persona signal.\n", S),
            ("M10 (S5) 'improved after SVD denoising'", "x/main.tex", "Emotion classification improved after SVD denoising.\n", S),
            ("M11 (S6) 'improves under denoising'", "x/main.tex", "Persona intensity improves under denoising.\n", S),
            ("M12 (S8) 'enhances'", "x/main.tex", "SVD denoising enhances the persona signal.\n", S)):
        check(name, path, text, [], [eid])

    # Round 5, from the round-4 verifier's plants (diff_r3.py, plants_r3.py), verbatim. Each T/H/K case was WRONG on
    # 9a3f908; the guards (Tg, Hg, Kg) held before and must hold after.
    print("PR #84 round 5, deception with punctuation between word and figure (bounded window; must be UNMARKED):")
    for name, path, text in (
            ("T1 colon: 'deception detection: AUROC = 1'", "x/paper.md",
             "Within-model deception detection: AUROC = 1 in all seven models.\n"),
            ("T2 Markdown table cells", "x/paper.md", "| Within-model deception | AUROC = 1 | seven models |\n"),
            ("T3 LaTeX table cells, 'AUROC $=1$'", "x/main.tex", "Within-model deception & AUROC $=1$ & seven models \\\\\n"),
            ("T4 comma and colon: 'deception, seven models: AUROC of one'", "x/main.tex",
             "Within-model deception, seven models: AUROC of one.\n"),
            ("T5 dash: 'deception --- perfect separation'", "x/main.tex",
             "Within-model deception --- perfect separation in all seven models.\n"),
            ("T6 figure first, deception in a bracket: 'AUROC = 1 (within-model deception, ...)'", "x/paper.md",
             "AUROC = 1 (within-model deception, seven models).\n"),
            ("T7 a bracket between: 'detection (seven models) reaches AUROC = 1'", "x/paper.md",
             "Within-model deception detection (seven models) reaches AUROC = 1.\n"),
            ("T8 capitalised cell value: '| ... deception | Perfect separation |'", "x/paper.md",
             "| Within-model deception | Perfect separation | seven models |\n")):
        check(name, path, text, [], [D])
    print("PR #84 round 5 window guards (must not be reported; held before, must hold after):")
    for name, path, text in (
            ("Tg1 ', and' ends the window: 'Deception probes failed, and baselines hit AUROC = 1'", "x/paper.md",
             "Deception probes failed, and baselines hit AUROC = 1.\n"),
            ("Tg2 figure first, a second subject after 'and': 'a perfect classifier has AUROC = 1 and the deception "
             "probe ...'", "x/paper.md", "A perfect classifier has AUROC = 1 and the deception probe reaches 0.71.\n"),
            ("Tg3 more than six words between", "x/paper.md",
             "The deception set spans seven models, four prompt families and two decoding temperatures with AUROC = 1 "
             "for the honesty baseline.\n")):
        check(name, path, text, [], [])
    check("T9 a marked table row with punctuation between is seen and passes as marked (9a3f908 did not see it)",
          "x/main.tex", "Within-model deception & AUROC $=1$ & Dead \\\\\n", [D], [])
    # Round 6, the round-5 verifier's E1 (extra_r6b.py) verbatim, plus mirrors. E1, E1r, E1n and E1m were WRONG on
    # 5e16a80; the guards Eg1 and Eg2 held before and must hold after.
    print("PR #84 round 6, a ';' inside a bracket does not end the deception window (must be UNMARKED):")
    for name, path, text in (
            ("E1 'deception (seven models; AUROC = 1)'", "x/paper.md",
             "Within-model deception (seven models; AUROC = 1) is the headline.\n"),
            ("E1r figure first: 'AUROC = 1 (seven models; within-model deception)'", "x/paper.md",
             "AUROC = 1 (seven models; within-model deception) is the headline.\n"),
            ("E1n a nested bracket before the ';': 'deception (seven models [Table 3]; AUROC = 1)'", "x/main.tex",
             "Within-model deception (seven models [Table 3]; AUROC = 1) is the headline.\n")):
        check(name, path, text, [], [D])
    check("E1m the same shape with a marker is seen and passes as marked", "x/paper.md",
          "Within-model deception (seven models; AUROC = 1) is retracted.\n", [D], [])
    print("PR #84 round 6 guards, a top-level ';' still ends the window (must not be reported):")
    for name, path, text in (
            ("Eg1 a top-level ';'", "x/paper.md",
             "Within-model deception probes fail; AUROC = 1 holds for sentiment probes.\n"),
            ("Eg2 a stray ')' opens no bracket: '1) ... fail; 2) ... AUROC = 1'", "x/paper.md",
             "1) Within-model deception probes fail; 2) sentiment probes reach AUROC = 1.\n")):
        check(name, path, text, [], [])
    print("PR #84 round 5, 'titled'/'headed' only in a past form (live headings must be UNMARKED):")
    for name, text in (
            ("H1 (U3) participle: 'Section 4, titled ``...'', shows'",
             "Section 4, titled ``SVD Denoising Rescues Emotion Classification'', shows the gain in all three models.\n"),
            ("H2 (U4) 'is now titled ``...'''", "Section 4 is now titled ``SVD Denoising Rescues Emotion Classification''.\n"),
            ("H3 'is still titled ``...'''", "Section 4 is still titled ``SVD Denoising Rescues Emotion Classification''.\n"),
            ("H4 'are currently headed ``...'''",
             "The persona and emotion rows are currently headed ``Rescued by SVD denoising''.\n")):
        check(name, "x/main.tex", text, [], [S])
    for name, text in (
            ("Hg1 'was previously titled ``...''' passes", "Section 4 was previously titled ``SVD Denoising Rescues "
             "Emotion Classification''.\n"),
            ("Hg2 'originally headed ``...''' passes", "The table was originally headed ``Rescued by SVD denoising''.\n"),
            ("Hg3 'used to be titled ``...''' passes (9a3f908 reported it: 'be titled' was excluded)",
             "Section 4 used to be titled ``SVD Denoising Rescues Emotion Classification''.\n"),
            ("Hg4 'was titled ``...''' passes", "Section 4 was titled ``SVD Denoising Rescues Emotion Classification'' "
             "until v1.2.\n")):
        check(name, "x/main.tex", text, [S], [])
    print("PR #84 round 5, chaining stops at a long bracket and at 'that' (must be UNMARKED):")
    check("K1 (U6) a nine-word bracket between two quotations", "x/main.tex",
          "The heading previously read ``Persona gradient'' (in this version we keep and strengthen the claim) ``SVD "
          "denoising rescues persona intensity in all three models''.\n", [], [S])
    check("K2 (U7) ', and that our new runs show ``...'''", "x/main.tex",
          "The README said ``a weak signal'', and that our new runs show ``SVD denoising rescues emotion "
          "classification''.\n", [], [S])
    check("K3 (D12 with 'that') ', and that we now show \"...\"'", "x/paper.md",
          "Reviewer 2 said \"the design is clean\", and that we now show \"within-model deception detection at AUROC "
          "1.000\" in all seven models.\n", [], [D])
    check("Kg1 an eight-word bracket still chains", "x/main.tex",
          "The abstract said ``a weak signal'' (in versions 1.0 and 1.1 of the paper) ``SVD denoising rescues persona "
          "intensity''.\n", [S], [])

    print("CLI:")
    saved_argv, saved_io = sys.argv, (globals()["read_worktree"], globals()["read_rev"])

    def refuse(*_a, **_k):
        raise RuntimeError("would scan")
    globals()["read_worktree"] = globals()["read_rev"] = refuse     # nothing may be read, whatever main() does
    sys.argv = ["check_retracted_figures.py", "--self-test-force", "marked"]
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            main()
        code = "returned"
    except SystemExit as e:
        code = e.code
    except RuntimeError:
        code = "fell through to a scan"
    finally:
        sys.argv = saved_argv
        globals()["read_worktree"], globals()["read_rev"] = saved_io
    good = code == 2
    print(("  ok    " if good else "  WRONG ") + "--self-test-force without --self-test exits 2" +
          ("" if good else f"  (got: {code})"))
    ok &= good

    # Round 5 (round-4 verifier finding 2): a registry with NO agent_skip list (any registry before 7767f69) must still
    # skip, and say so. Synthetic paths only: "git ls-files" is stubbed to list made-up names, any other git call and
    # every read is recorded or refused, never performed, so no file in the repo is opened whatever main() does.
    import tempfile
    with open(os.path.join(ROOT, REGISTRY), encoding="utf-8") as fh:
        reg_text = fh.read()
    old_reg = re.sub(r"(?m)^agent_skip:[ \t]*\n(?:[ \t]+-.*\n)+", "", reg_text)
    fake = ["pain-study/main.tex", "user-model-paper/main.tex", "lyra-technique-ii/main.tex", "README.md"]
    asked, saved_git = [], globals()["git"]

    def record(paths, *_a):
        asked.extend(paths)
        return iter(())

    def fake_git(*args, **_k):
        if args[:1] == ("ls-files",):
            return ("\n".join(fake) + "\n").encode()
        raise RuntimeError("would run git " + " ".join(args))
    tmp_reg = None
    for _ in range(20):                       # a temp dir whose random name happens to match AGENT_SKIP is refused
        d = tempfile.mkdtemp(prefix="crf_selftest_")
        if registry_refused(os.path.join(d, "registry_without_agent_skip.yml")) is None:
            tmp_reg = os.path.join(d, "registry_without_agent_skip.yml")
            break
        os.rmdir(d)
    out = io.StringIO()
    code = None
    if tmp_reg:
        with open(tmp_reg, "w", encoding="utf-8") as fh:
            fh.write(old_reg)
        globals()["read_worktree"] = globals()["read_rev"] = record
        globals()["git"] = fake_git
        sys.argv = ["check_retracted_figures.py", "--registry", tmp_reg]
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = main()
        except SystemExit as e:
            code = f"exit {e.code}"
        except RuntimeError as e:
            code = str(e)
        finally:
            sys.argv = saved_argv
            globals()["read_worktree"], globals()["read_rev"] = saved_io
            globals()["git"] = saved_git
            import shutil
            shutil.rmtree(os.path.dirname(tmp_reg), ignore_errors=True)
    leaked =[p for p in asked if any(rx.search(p) for rx in AGENT_SKIP)]
    good = ("\nagent_skip:" in reg_text and "agent_skip:" not in old_reg and code == 0 and not leaked
            and "lyra-technique-ii/main.tex" in asked and "agent_skip: 2 text file(s) not scanned" in out.getvalue())
    print(("  ok    " if good else "  WRONG ") + "a registry with no agent_skip list still skips both restricted "
          "paths and prints the skip line" + ("" if good else f"  (code {code}, restricted paths asked for: "
                                                             f"{len(leaked)}, skip line printed: "
                                                             f"{'agent_skip:' in out.getvalue()})"))
    ok &= good
    err, sout = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(sout):
        off = agent_skips([], True, fake)     # the helper alone; no scan runs with the skip off
    good = off == [] and all("WARNING: --no-agent-skip" in s.getvalue() and "(2 text file(s))" in s.getvalue()
                             for s in (err, sout))
    print(("  ok    " if good else "  WRONG ") + "--no-agent-skip empties the skip list only with a loud warning on "
          "stderr and stdout")
    ok &= good
    extra = agent_skips([re.compile("extra-dir")], False)
    good = len(extra) == len(AGENT_SKIP) + 1 and all(any(r.pattern == a.pattern for r in extra) for a in AGENT_SKIP)
    print(("  ok    " if good else "  WRONG ") + "a registry's agent_skip list adds to the code's list, never replaces it")
    ok &= good
    print("PR #84 round 6, CLI and registry (round-5 verifier findings 6-8; nothing is scanned):")
    for name, good, got in cli_checks_r6(_ThisModule()):
        if good is None:
            print("  note  " + name)
            continue
        print(("  ok    " if good else "  WRONG ") + name + ("" if good else f"  (got: {got})"))
        ok &= good
    print("unrelated uses of the same numbers (must not match at all):")
    check("ethics-pack-injection/main.tex:98 per-framework +0.074 (no KG/MoReBench context)", "x/main.tex",
          "Aristotelian Virtue Ethics & 0.572 & 0.646 & $+$0.074 & $-$0.236 \\\\\n", [], [])
    check("decision-state encoding AUROC 1.000 (no deception context)", "x/main.tex",
          "Entity detection reaches AUROC 1.000 on token-matched pairs.\n", [], [])
    check("null-swarm 'rescues the result' (no denoising context)", "x/main.tex",
          "A bigger sample rescues the result, not the test that kills it.\n", [], [])
    check("cache-tracing confab ratio 2.3-2.8x (no emotion context)", "x/paper.md",
          "Confab/factual ratio 2.3-2.8x at L33.\n", [], [])
    check("convergence-paper/main.tex:1071 'Version 1.0' beside 'deception' is not AUROC 1.000", "x/main.tex",
          "Version 1.0's abstract stated a range (80--100\\%) for deception correction.\n", [], [])
    check("waystations-paper/main.tex:336 'd = +0.2 to +1.0' beside deception is not AUROC 1.000", "x/main.tex",
          "Deceptive defections show per-layer effects of $d = +0.2$ to $+1.0$ (stable rank).\n", [], [])
    check("lyra-technique-ii/main.tex:287 'denoising amplifier' (a noun, in a retraction) is not the claim",
          "x/main.tex", "The denoising amplifier was me mistaking selection noise for signal.\n", [], [])
    check("lyra-technique-ii/main.tex:171 'denoising would amplify weak signals' states the hypothesis", "x/main.tex",
          "We had hypothesized a hierarchy under which denoising would amplify weak signals.\n", [], [])
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

def make_parser():
    # allow_abbrev=False (round 6, round-5 verifier finding 8): argparse took any unique prefix, so "--no", "--no-a"
    # and "--no-agent" all meant --no-agent-skip, and a hook that greps commands for the literal flag missed them.
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    ap.add_argument("paths", nargs="*", help="limit to these repo paths (files or folders)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--base", metavar="REF", help="CI: fail only on unmarked occurrences HEAD adds over merge-base(REF, HEAD)")
    g.add_argument("--twins", action="store_true", help="TWIN_DESYNC report for flight/academic pairs")
    g.add_argument("--self-test", action="store_true", help="run the planted cases")
    ap.add_argument("--self-test-force", choices=["marked", "unmarked"], help=argparse.SUPPRESS)
    ap.add_argument("--twins-strict", action="store_true", help="with --twins: exit 1 on MARKER_DESYNC")
    ap.add_argument("--exclude", action="append", default=[], metavar="REGEX", help="skip repo paths matching REGEX")
    ap.add_argument("--include-records", action="store_true", help="also scan files the registry lists as records")
    ap.add_argument("--no-agent-skip", action="store_true",
                    help="also scan the registry's agent_skip paths (CI passes this; agents must not)")
    ap.add_argument("--registry", default=os.path.join(ROOT, REGISTRY))
    return ap


def main():
    ap = make_parser()
    a = ap.parse_args()
    if a.self_test_force and not a.self_test:
        # round 3 fell through to a whole-tree scan here (verifier finding 7); refuse instead
        ap.error("--self-test-force works only together with --self-test")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    if a.self_test:
        return self_test({None: None, 'marked': True, 'unmarked': False}[a.self_test_force])

    refused = None if a.no_agent_skip else registry_refused(a.registry)
    if refused:
        die(refused)
    entries, records, skips = load_registry(a.registry)
    if a.no_agent_skip:
        # prints the loud warning, with the number of tracked text files the skip would have kept out
        skips = agent_skips(skips, True, git("ls-files").decode("utf-8", "replace").splitlines())
    excludes = [re.compile(x) for x in a.exclude] + skips
    roots = [p.replace("\\", "/").strip("/") for p in a.paths]

    def keep(paths):
        return [p for p in paths if wanted(p, records, excludes, a.include_records, roots)]

    def n_agent_skipped(paths):
        return sum(1 for p in paths if p.lower().endswith(TEXT_EXT) and any(rx.search(p) for rx in skips))
    if not a.no_agent_skip and not skips:
        die("internal: agent_skip is empty without --no-agent-skip")     # cannot happen: AGENT_SKIP is in the code

    def summary(n_files, hits, extra=""):
        n_un = sum(1 for h in hits if not h.marked)
        n_q = sum(1 for h in hits if h.how == "quoted")
        print(f"\n{n_files} files scanned, {len(hits)} occurrences of {len(entries)} registry figures, "
              f"{len(hits) - n_un - n_q} marked, {n_q} quoted, {n_un} unmarked{extra}")

    if a.base:
        mb = git("merge-base", a.base, "HEAD").decode().strip()
        changed = git("diff", "--name-only", "--no-renames", mb, "HEAD").decode("utf-8", "replace").splitlines()
        if skips:
            print(f"agent_skip: {n_agent_skipped(changed)} changed file(s) not scanned (--no-agent-skip scans them)")
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
    agent_note = f"; {n_agent_skipped(tracked)} agent_skip files not scanned (--no-agent-skip)" if skips else ""
    if skips:
        print(f"agent_skip: {n_agent_skipped(tracked)} text file(s) not scanned (--no-agent-skip scans them)")

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
    summary(n, hits, f"; {skipped_records} record files skipped (--include-records to scan){agent_note}")
    bad = any(not h.marked for h in hits)
    print("GATE: FAIL" if bad else "GATE: PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
