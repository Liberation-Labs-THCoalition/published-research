# Verifier 2: port fidelity (PR #16, commit 500fc9a)

Written 2026-10-31 04:34 PDT. I only read and ran things. Nothing in the repo was edited except this file, which is
untracked and not committed.

## Verdict

**The port is faithful.** I found no blocking defect.

- Every hunk from PR #4 is present.
- code/ and data/ are byte-identical to PR #4.
- Both editions build cleanly, and the committed PDFs match a fresh build.
- The files are LF only, and the diff stays in scope.
- One published-only correction (academic, 2026-08-20) loses its dated history. That is finding 1.
  The other findings are follow-ups or cosmetic.

## Checks and evidence

1. **PR #4's corrections are all present.**
   - The worktree blob ids for `main.tex` (`23f7065…`) and `academic/main.tex` (`21c5495…`) equal
     human-review `54c7f9d1`'s.
   - PR #4's diff base is `b05561db` (the merge base with human-review origin/main). It spans 9 commits,
     `a49b2ce`..`54c7f9d1`.
   - A line-set check finds 234 lines (flight) and 248 lines (academic) that PR #4 added, with 0 missing
     from HEAD.
   - Missing hunks: **none**.
2. **Published-only changes.** These are lines in origin/master `e667f41` that are not in PR #4's base.
   - Flight: 30 lines, 28 absent from HEAD. Academic: 95 lines, 78 absent.
   - Almost all are the published copy's July/August/September H2 "TOST inconclusive" and H3
     "unsupported / exploratory" wording. PR #4 supersedes that wording, and its *Correction (2026-10-20)*
     paragraph (`main.tex:175-202`) quotes and withdraws it.
   - Kept, word for word:
     - the academic title (`academic/main.tex:25-27`);
     - the byline: flight `main.tex:30-31` has Lyra and Thomas; academic `:30` has Thomas with the AI
       contributor note;
     - the CRediT rows (`academic/main.tex:500-504`);
     - the Acknowledgments for Dwayne Wilkes and Kavi (`main.tex:520,522`; `academic/main.tex:525,527`);
     - the contact addresses.
   - The `e667f41`/`87545fd` contact-address commits never touched mine5 (`git log origin/master --
     mine5-selective-sharpener`).
   - Both `references.bib` files and `SOP_REVIEW.md` are unchanged.
   - Only one old-base line came back: the academic H2 heading "--- Confirmed". It is PR #4's text.
3. **code/ and data/ match by sha256.** I compared each `git show 54c7f9d1:<path> | sha256sum` with the
   worktree file. All 6 match: `.gitattributes` x2, `mine5_three_model.py 8155afcb…`,
   `recompute_registered.py 2c08a6aa…`, `…021501.json 2d950924…` and `prereg.json 57b1811b…`.
   `* -text` is in effect (`git check-attr`: `text: unset`).
4. **Recompute, my own run.** I ran `python -I code/recompute_registered.py` from an empty scratch
   directory.
   - Result: "real checks: 55, failures: 0; planted false value caught: True; data hashes OK: True", exit 0.
   - The worktree was still clean afterwards.
   - The PR table's values all check OK: 0.684, 0.021, 0.746, 0.055, 0.279, 1.04 and [+0.893, +1.224].
5. **Build.** I built both editions into scratch with pdflatex, then bibtex, then pdflatex twice, and
   never wrote over the committed PDFs.
   - 0 undefined references or citations, 0 errors, 8 pages each.
   - Text extracted from the committed PDFs with pypdf is identical to my rebuild: 0 diff lines.
   - "Trend Only" appears once in each PDF, inside the Correction quote. "Koriat" appears 0 times.
   - There are 3 overfull boxes per edition (finding 3), and none of them hides text.
6. **Hygiene.**
   - CR bytes are 0 in the worktree and in the HEAD blob for all 11 changed text files (Python count).
   - `git diff --check origin/master HEAD` exits 0.
   - `git diff --stat` touches only `mine5-selective-sharpener/**` and `README.md` (1 line).
   - Scope of the README line: the new Summary at `README.md:55` agrees with the registered verdicts. The
     old "RLHF improves calibration" contradicted the falsified H3. This is justified at
     `PORT_NOTES.md:139-148`.
7. **The PR.** `gh pr view 16` shows isDraft true, OPEN, base master, head `lyra/mine5-port`, headRefOid `500fc9a1…`.
   - The body's claims agree with the checks above.
   - The body ends exactly with the robot line, a blank line and the session URL.
   - The commit message ends with exactly the two attribution lines (`git log -1 | cat -A`).
   - In human-review, PR #4 is unchanged (head `54c7f9d1`, OPEN). The only mine5/wip/local branch there is
     `lyra/mine5-correction`.
8. **d = 0.91** is called "a schema-commitment signature, not a confidence result" at `main.tex:66-67` and
   `:81-82`. The deep-content figure, 0.37 with p > 0.08, is at `:98-99`. "metacogn" and "Koriat" appear 0
   times in either .tex.

## Findings, ranked

1. **The academic edition loses its own 2026-08-20 correction history.** (minor, fidelity)
   - Published `academic/main.tex` at origin/master, lines 200-206, carried a footnote saying "Corrected
     2026-08-20: this edition previously concluded equivalence from the point estimate…".
   - The ported Correction paragraph (`academic/main.tex:176`) dates the TOST wording only to "Revisions of
     28--29 July 2026". That is true of the integrity edition (`6e23b0a`), but in the academic edition the
     TOST wording arrived on 08-20.
   - So the published reader's history for this edition is misdated.
   - **Smallest fix:** in `academic/main.tex:176`, after "Revisions of 28--29 July 2026", add "(20 August 2026
     in this edition)". Rebuild the academic PDF, rerun the recompute, and record that the academic `.tex`
     no longer matches PR #4 byte for byte.
2. **`SOP_REVIEW.md` still certifies the retracted analysis.** (`REMEDIATION_REGISTER.md:367`, MAJOR there)
   - Its title is still "RLHF Is a Selective Sharpener" (`SOP_REVIEW.md:1`).
   - The PR flags this, but it now ships next to a paper that contradicts it.
   - **Smallest fix:** a two-line banner at the top: "Superseded 2026-10-20: reviews the July analysis; see the
     Correction paragraph in main.tex and code/recompute_registered.py."
3. **Three overfull hboxes per edition, at 1.0pt, 7.9pt and 14.5pt** (`main.log`, paragraphs at `main.tex`
   112-115, 131-136 and 382-390).
   - The text runs into the margin and none is hidden.
   - The overflowing lines are text unchanged since origin/master. I did not rebuild origin/master to confirm
     it overflowed there too.
   - **Smallest fix:** none needed. Optionally add `\sloppy` or a `\linebreak` at "Workspace bypass".
4. **bibtex: "empty journal in gurnee2026gwt".** It comes from the unchanged `references.bib`, so it is
   cosmetic and not part of the port. **Fix:** add a `journal`/`howpublished` field later, in a separate change.
5. **Port working notes ship inside the public paper folder** (`notes_port_20261031/PORT_NOTES.md`,
   `recompute_output.txt`).
   - The `SOP_REVIEW.md` precedent exists, but these are process notes.
   - This file will also sit there untracked.
   - **Smallest fix:** Lyra decides whether to keep them. If not, drop them from the commit before
     un-drafting.
6. **Follow-ups the PR already names, which still hold.**
   - PR #13 cites "[+0.010, +0.270]" (present here only at `main.tex:301,358`), so #16 must land first.
   - Stale rows: `REMEDIATION_REGISTER.md:396`, `:928` and `DECISIONS_WAITING.md:86`.
   - **Fix:** none in this PR. They are tracked in the PR body.
