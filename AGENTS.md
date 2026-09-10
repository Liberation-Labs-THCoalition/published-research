# AGENTS.md — operating rules for changing this repository

**For any agent or person editing this corpus.** Written 2026-09-09 after a day in which every
rule below was broken at least once, most of them by me. Extends the Repo Hygiene Rules in
[README.md](README.md); it does not replace them.

Read this before your first edit. It is short on purpose.

---

## 1. Before you change anything, read the thing you are changing

- **Never overwrite a file you have not read.** *(Broken 2026-09-09: `.gitignore` was clobbered by
  a `cat >` — a 2 KB file carrying steering-vector redaction rules, a documented gitlink incident,
  and policy notes, replaced by 600 bytes of guesses. Restored from git.)*
- **Never `git add <file>` without reading its full diff first.** Staging by *filename* is not
  scoping. A file can hold changes you did not make. *(Broken 2026-09-09: an author-line change and
  a result withdrawal rode into a commit titled "fix a table caption.")*

```bash
git diff -- <path>                        # what you are about to stage
git diff -w --ignore-cr-at-eol -- <path>  # content only; strips CRLF churn
```

**Most large diffs in this repo are line-ending churn.** Always check the second form before
calling a change substantive.

### The two rules that would have prevented all of it

> **1. Never `git add <dir>/` and never `git add -A`. Stage an explicit file list.**
> **2. Before every commit, run `git diff --cached --stat` and read it. If a file you did not
>    intend is listed, or a count is bigger than your edit, STOP.**

Rule 2 is the one that keeps getting skipped, and skipping it is what caused **three** bad commits
on 2026-09-09 alone:

| commit | intended | actually contained |
|---|---|---|
| `b2547b1` | a table caption | + an author-line change, + a result withdrawal |
| — | append to `.gitignore` | overwrote a 2 KB file unread (caught, restored) |
| `fd2b727` | 11 byline removals | + a retitled paper, + a corrected chance baseline, + a body-count withdrawal, + an *n* correction, + 3 new files |

**The pattern is always the same: scoping by PATH when the unit that matters is the HUNK.** A path
contains whatever anyone left there. `git diff --cached --stat` is four seconds and it is the only
step that catches it.

## 2. Every paper has twins. Check all editions.

Papers ship as `main.tex` (flight) and `academic/main.tex` (venue), sometimes plus
`academic_main.tex` or `paper/main.tex`. **A fix applied to one edition is not applied.**
Twin desync is the most common defect class in this corpus and it has its own register rows.

```bash
grep -rn "<the string>" --include="*.tex" <paper-dir>/   # find every edition first
```

## 3. Authorship is never a side effect

**The standing policy** (README "Review and auditing", confirmed by Thomas 2026-09-09):

> **Dwayne Wilkes and Kavi are advisory.** Not on a byline until they sign off on that individual
> paper; credited in Acknowledgments of papers they reviewed. **Exception:** the six
> `digital-minds-hackathon-2026` submissions, which Kavi signed off — Kavi is a byline author there.

Rules:

- **An authorship change gets its own commit, named in the subject line.** If a binary artifact
  makes that impossible — a rebuilt PDF carries both — then **name both changes in the subject.**
- **Moving someone off a byline requires their credit to land somewhere.** Check the Acknowledgments
  line exists *and is unchanged*, in every edition, and in the rendered PDF.
- **CRediT is not a byline.** Deleting a CRediT row does not implement a byline policy; it destroys
  a contributor-role record. If a sweep does this, stop and ask. *(2026-09-09: Kavi's CRediT rows
  were deleted in two academic editions. Kavi was never on those bylines.)*
- **Never leave a document that both lists someone as an author and thanks them as a non-author.**
  That is a defect whichever direction is correct. Nine papers were in this state on 2026-09-09.

## 4. Verify in the artifact a reader sees, not in the source

A corrected `.tex` beside an unrebuilt `.pdf` means the defect still ships. This is the single most
repeated failure in this corpus.

**USE THE TOOL. Do not hand-roll this — five hand-rolled checks lied on 2026-09-09.**

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
python tools/verify_pdf.py main.pdf \
    --control "<a string you KNOW is in the paper>" \
    --present "<the corrected claim>" \
    --absent  "<the wrong claim, if it should be gone>"
```

The `--control` is mandatory and the tool refuses without it: **an empty extraction looks exactly
like a clean bill of health.**

### Why `pdftotext | grep` lies, four ways

1. **LIGATURES — the one that got us five times.** pdftotext **drops** `fi`/`fl`/`ff` in these
   MiKTeX builds. *"filter"* extracts as **"lter"**, *"benefit"* as **"benet"**, *"float"* as
   **"oat"**, *"quantified"* as **"quantied"**. So any needle containing `fi`, `fl` or `ff`
   silently fails — which covers **effect, difference, significant, coefficient, confirm, file,
   verify, fifteen.** `verify_pdf.py` makes those optional in the pattern.
2. **HYPHENATION.** LaTeX splits words across line breaks: *"25 independent"* extracts as
   *"25 inde- pendent"*.
3. **WRAPPING.** A phrase spanning a line break never matches a flat string.
4. **A CORRECTION QUOTES WHAT IT CORRECTS**, and no normalisation fixes this — you have to think.
   *"previously reported at AUROC 1.0, since withdrawn"* contains the string you are grepping for.
   **A substring cannot distinguish asserting a claim from retracting one.** Search for the
   *retraction language*, not the number.

**If both the old and the new string return zero, your check is broken, not your fix.** That is the
tell, and it fired four times before anyone read it correctly.

## 5. A sweep is not evidence about a file it did not touch

If you run a corpus-wide pass, **close rows per paper, from the working tree, not from the sweep's
own claim.** Count what is actually true now:

```bash
grep -rl "<the thing that should be gone>" --include="*.tex" . | wc -l
```

*(The register recorded a byline sweep as "closed across 41 dirs (diff-verified)". Measured on
2026-09-09: 11 papers still failed it, 9 self-contradictorily. The register had **already retracted
two identical closures** four hundred lines below, with the rule written out. Same error, one entry
from its own correction.)*

**Fix the class, not the instance.** When you find one, grep the corpus for the pattern.

## 6. Committing

- **A commit message claiming "fix" or "correction" must match the actual diff** (README rule).
- **Say what you did NOT do.** If a number was unrecoverable, say it was deliberately not invented
  and open a register row. Silence reads as completeness.
- **Cite the register row or audit document** a change traces to. A correction whose reasoning
  lives on one laptop is a correction nobody can check.
- **Do not bulk-commit a working tree you have not inventoried.** 98 files sat uncommitted on
  2026-09-09 and committing them would have published nine broken papers.

## 7. Never

- Delete files force-committed past ignore rules — **they are audit evidence** (README rule).
- Remove a README status table row — **update it; it is an accountability surface** (README rule).
- Commit `*.bak*`, build artifacts, or `*.pt` steering vectors. See `.gitignore`, which carries the
  reasons.
- Invent a number you cannot source. Open a register row instead.

---

## The one-line version

**Read the diff, check every twin, verify in the PDF, give authorship its own commit, and never let
a sweep close a row it did not personally check.**
