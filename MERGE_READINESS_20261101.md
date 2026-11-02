# PR #18 merge readiness (SOP gate re-run), 2026-11-01 17:02 PST

Branch `lyra/downstream-user-model-cites`, gated at head `6564032` (pushed; upstream equals HEAD).
`origin/master` is `1748bde` and is the merge-base: `6564032` merged it in. The branch is 11 commits ahead and
0 behind. PR stays a DRAFT. Nothing was merged. The only change in this commit is this file.
This replaces the first pre-run (gated at `9f79966`, commit `d354de2`).

**Verdict: NOT READY.** All six gate items pass, and the gate-1 path from the first pre-run is gone (`2140fb7`
redacted it). Two things are still Lyra's:
- **Agni paper r2.** Agni paper r1 was CONDITIONAL, and no Agni round has seen `3f8e35b`, the commit that
  applied its conditions. That is the rule #17's register row N5 sets for #17; it is applied here for parity,
  and it is my reading, not a recorded r1 condition.
- **A sentence to check.** Decision-state's Methods, line 326, says "FWL residualization on prompt length applied". It is
  in both editions, not only the academic one as the first pre-run said. Line 390 says the headline estimator
  applies none. Read it against the source and correct it or record why it stands.

In any case #18 cannot merge before #17.

## Gate results

| # | Gate | Result |
|---|------|--------|
| 1 | Private-info scan of added lines | PASS, no new exposure (Thomas decides on the listed mentions) |
| 2 | verify_pdf on every rebuilt PDF, both ways | PASS |
| 3 | Every edition changed together | PASS |
| 4 | Credit changes in their own named commits | PASS (none) |
| 5 | No stale PDFs | PASS |
| 6 | `git merge-tree` against `origin/master` | PASS (exit 0) |

### 1. Private-info scan: PASS, no new exposure
- **What was scanned:** every added line of `git diff 1748bde HEAD`, excluding PDFs, plus the text of the four
  changed PDFs: 3,892 lines across 17 files and texts.
- **Patterns and planted control:** the same as #17. All 7 classes were caught. A new-form path in an
  unregistered file was classed FAIL, and the same path in a `SHA256SUMS`-listed file was classed as a Thomas decision.
- **Classed FAIL by the script (4), all false positives on reading:** the `C:/Users/<user>` placeholder in the
  previous copy of this file; `~/.credentials` in this file and in `notes/verify_2026-11-01_2_fidelity.md`,
  where it describes an earlier planted control; and `/Users/runner` in the same note, quoting a log line from #17.
- **Master already carries the same form (Thomas decision, not FAIL):**
  - `info@digitaldisconnections.com`, the correspondence address in the `\thanks` lines.
  - The GPU machine's name and "Tailscale": 14 mentions in this file, the notes and
    `notes/agni_paper_downstream_um_cites_r1.json`.
- **0** IPv4, **0** Tailscale IPs, **0** keys, **0** home paths. The r1 JSON now reads `<home>/...`.

### 2. verify_pdf both ways: PASS
- `tools/verify_pdf.py`, `--control "Edrington"`. Each check counts only if it passes on the new PDF and fails on
  the previous build (`66012ca`). A planted present-check fails on all four PDFs, as it should.
- All four PDFs are the same blobs as at `9f79966`; the master merge did not touch them.

  | PDF | Checks |
  |---|---|
  | `decision-state-paper/main.pdf` | 4/4 |
  | `decision-state-paper/academic/main.pdf` | 4/4 |
  | `emotion-accumulation-paper/main.pdf` | 5/5 |
  | `emotion-accumulation-paper/academic/main.pdf` | 5/5 |

- **Present:** "one trial from May's 368 on a changed software stack", "both at 5 folds", "0.985".
- **Absent:** "0.358 against 0.382)". Also absent, on emotion-accumulation only: "which the latter corrected in
  October 2026: the probe". On decision-state that string is absent from the previous build too, so it cannot fail
  there and is not counted (as in the first pre-run).

### 3. Editions changed together: PASS
- Every non-merge commit in `1748bde..HEAD` that touches a flight `main.tex` or `main.pdf` touches its academic
  twin. There are no exceptions.

### 4. Credit changes: PASS (none)
- The only net author-field change is the new `lyra2026usermodel` entry in
  `emotion-accumulation-paper/references.bib` (`089bb2e`), with `author={Lyra and Edrington, Thomas}`. That matches #17's
  `paper.json`. No byline, `\thanks` or CRediT line changed.

### 5. No stale PDFs: PASS
- All four PDFs were last committed in `9f79966`, after the last source commit `3f8e35b`. The worktree is clean
  in the paper directories.
- **Fresh rebuild:** `git archive HEAD` of both paper directories into an empty folder, then MiKTeX
  `latexmk -pdf`. All four exit 0 with no undefined references. The text is identical to the committed PDFs after whitespace
  normalisation (decision-state 16 pages, emotion-accumulation 10 pages).
- The previous build fails the gate 2 checks, so this comparison can tell builds apart.

### 6. Mergeability: PASS
- `git merge-tree --write-tree origin/master HEAD` exits 0.

## Still Lyra's
1. **Agni paper r2** on `3f8e35b` (the r1-conditions commit), both editions. Then apply what it finds.
2. **Decision-state `main.tex:326`** in both editions: the FWL-on-prompt-length sentence against `:390`.

## Waiting on Thomas
1. **#17 first.** #18 is gated on #17 and stays a draft until #17 lands.
2. **Host-name mentions** that master already carries (gate 1): accept them on master's precedent, or scrub them.
3. **Older-commit paths:** the r1 JSON's home path is still in history before `2140fb7`. History is not rewritten, and no force-push.
4. **Out of scope this round:** paper r1 item 10 (check the Qwen3.5 template, re-read Experiment 2's conclusions)
   and item 12d (rerun the bootstrap by pair group).

Not part of this branch: `notes_verify_20261101/` sits untracked in this worktree and was not committed.

## Merge order
**#17 first, then #18.** #18 cites #17's new title ("Story Emotion in the KV Cache: A Preregistered Test Against
a Text (TF-IDF) Baseline").
