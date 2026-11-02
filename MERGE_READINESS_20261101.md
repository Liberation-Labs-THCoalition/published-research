# PR #18 merge readiness (SOP gate pre-run), 2026-11-01 16:37 PST

Branch `lyra/downstream-user-model-cites`, gated at head `9f79966` (pushed; upstream equals HEAD).
Merge-base with `origin/master` is `f1ee90f`. The branch is 8 commits ahead and 45 behind.
PR stays a DRAFT. Nothing was merged. The only change in this commit is this file.

**Verdict: GATE PASSES EXCEPT ONE LOW-SEVERITY PATH, AND IT IS BLOCKED ON #17.** Gates 2 to 5 pass. Gate 1
fails on one local path in a review record. `git merge-tree` against `origin/master` is clean. #18 must not merge
before #17.

## Gate results

| # | Gate | Result |
|---|------|--------|
| 1 | Private-info scan of added lines | **FAIL** (low severity: one local path) |
| 2 | verify_pdf on every rebuilt PDF, both ways | PASS |
| 3 | Every edition changed together | PASS |
| 4 | Credit changes in their own named commits | PASS (none) |
| 5 | No stale PDFs | PASS |
| x | Clean merge into `origin/master` (not an SOP item) | PASS (merge-tree exit 0) |

### 1. Private-info scan: FAIL (low severity)
- **What was scanned:** every added line of `git diff f1ee90f HEAD`, excluding PDFs: 3,802 lines including the
  text of the four changed PDFs.
- **Patterns:** the same as #17: IPv4, Tailscale range, home paths, emails, keys/tokens/passwords, host names.
- **Planted control:** a Tailscale IP, a `C:\Users` path, a `/home` path, a GitHub token, a host name, a personal
  email and a `PGPASSWORD`. All 7 pattern classes caught it.
- **Clean:** 0 IPv4, 0 Tailscale IPs, 0 keys.
- **Fail:** `notes/agni_paper_downstream_um_cites_r1.json` carries one
  `C:/Users/<user>/.../published-research-um-correction/...` path.
- **Pre-existing, not added:** the emails in the PDF text are `info@digitaldisconnections.com` (the correspondence
  address in the `\thanks` lines) and `lyra@liberationlabs.tech`. Neither is on an added tex line.
- **Benign hits:** "Starship" and "Tailscale" in `notes/verify_2026-11-01_*.md`. Starship is on `origin/master` in
  45 files. The Tailscale and `~/.credentials` mentions only describe an earlier planted control.
- **History:** I scanned the net diff only. Older-commit paths are a known Thomas item, noted here and not failed
  on.

### 2. verify_pdf both ways: PASS
- **Tool and control:** `tools/verify_pdf.py`, with `--control "Edrington"`.
- **Rule:** each check must pass on the new PDF and fail on the previous build (`66012ca`).
- **Results:**

  | PDF | Checks |
  |---|---|
  | `decision-state-paper/main.pdf` | 4/4 |
  | `decision-state-paper/academic/main.pdf` | 4/4 |
  | `emotion-accumulation-paper/main.pdf` | 5/5 |
  | `emotion-accumulation-paper/academic/main.pdf` | 5/5 |

- **Present:** "one trial from May's 368 on a changed software stack", "both at 5 folds", "0.985".
- **Absent:** "0.358 against 0.382)". The new text reads "0.358 against 0.382, at 10 folds".
- **Absent, emotion-accumulation only:** "which the latter corrected in October 2026: the probe".
- **Discarded check strings:**
  - Three strings were dropped because they passed on the previous build as well, so they could not fail:
    "(reproduced 0.410", "result (AUROC 0.992) beats", and the "latter corrected" string on decision-state.
  - That is a fault in my check strings, not in the PDFs. They are not counted above.

### 3. Editions changed together: PASS
- Every commit in `f1ee90f..HEAD` that touches a flight `main.tex` or `main.pdf` touches its academic twin.
- There are no exceptions.

### 4. Credit changes: PASS (none)
- The only net author-field change is the new `lyra2026usermodel` entry in
  `emotion-accumulation-paper/references.bib`, with `author={Lyra and Edrington, Thomas}`.
- That matches #17's `paper.json` authors. No byline, `\thanks` or CRediT line changed.

### 5. No stale PDFs: PASS
- **Commit order:** all four PDFs were last committed in `9f79966`, after the last source commit `3f8e35b`.
  The worktree is clean in the paper directories.
- **Fresh rebuild:** `git archive HEAD` into an empty folder, then MiKTeX `latexmk -pdf`. All four exited 0 with
  0 undefined.
- **Comparison:** the rebuilt text is IDENTICAL to the committed PDFs after whitespace normalisation
  (decision-state 16 pages, emotion-accumulation 10 pages).
- **Control:** the same comparison was shown able to fail against a previous build. That control ran on #17's
  flight PDF.

## Waiting on Thomas
1. **#17 first.** #18 is gated on #17 and stays a draft until #17 lands.
2. **Gate 1 path:** the one local path in `notes/agni_paper_downstream_um_cites_r1.json`. Scrub it, or accept it
   on the precedent of master.
3. **Older-commit paths:** history is not rewritten, and no force-push.
4. **Open items, out of scope this round:**
   - Paper r1 item 10: check the Qwen3.5 template and re-read Experiment 2's conclusions.
   - Item 12d: rerun the bootstrap by pair group.
5. **Possible leftover:** decision-state's academic Methods (around `:326`) still says "FWL residualization on
   prompt length applied" for the combined classifier. It was not touched this round.

## Merge order
**#17 first, then #18.** #18 cites #17's new title ("Story Emotion in the KV Cache: A Preregistered Test Against
a Text (TF-IDF) Baseline").
