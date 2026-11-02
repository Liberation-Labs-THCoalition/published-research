# PR #17 merge readiness (SOP gate re-run), 2026-11-01 17:02 PST

Branch `lyra/user-model-correction-v2-final`, gated at head `61ab741` (pushed; upstream equals HEAD).
`origin/master` is `1748bde` and is the merge-base: `b255dde` merged it in. The branch is 45 commits ahead and
0 behind. PR stays a DRAFT. Nothing was merged. The only change in this commit is this file.
This replaces the first pre-run (gated at `ebc197b`, commit `b9a25ff`).

**Verdict: NOT READY.** All six gate items pass, with a list for Thomas. One thing is still Lyra's: Agni paper
r5's condition 1 (CRITICAL, register N5). No Agni paper round has seen `a736d36..HEAD`, which holds the r4 fixes
(`f64ee5d`) and the r5 fixes (`61ab741`), and the academic edition has never been an Agni input. r5 does not
approve HEAD. **Lyra's to do: run Agni paper r6 on that range, both editions and the bib, and apply what it finds.**

## Gate results

| # | Gate | Result |
|---|------|--------|
| 1 | Private-info scan of added lines | PASS, no new exposure (Thomas decides on the listed paths) |
| 2 | verify_pdf on both rebuilt PDFs, both ways | PASS (10/10 each) |
| 3 | Every edition changed together | PASS (one named exception) |
| 4 | Credit changes in their own named commits | PASS (one change awaits Thomas) |
| 5 | No stale PDFs | PASS |
| 6 | `git merge-tree` against `origin/master` | PASS (exit 0, no conflicts) |

### 1. Private-info scan: PASS, no new exposure
- **What was scanned:** every added line of `git diff 1748bde HEAD`, excluding PDFs, plus the text of both
  changed PDFs: 1,168,518 lines across 44 files and texts.
- **Patterns:** IPv4, Tailscale range, Windows home paths (single and doubled backslash, and forward slash), POSIX
  home paths, emails, keys/tokens/passwords, host names.
- **Planted control:** a synthetic file held a Tailscale IP, a `C:\Users` path, a doubled-backslash path, a
  `/home` path, a GitHub token, a host name, a personal email and a `PGPASSWORD`. All 7 classes were caught and
  classed FAIL. The same path placed in a file listed in a `SHA256SUMS` was classed as a Thomas decision, so the
  classifier can return either answer. (The first control run missed the IPv4 class because of a filter bug of
  mine; it was fixed and the control re-run before any result was read.)
- **Classed FAIL by the script (6), all false positives on reading:** the `C:\Users\<user>` placeholders in the
  previous copy of this file; `~/.credentials` in this file and in `notes/verify_2026-11-01_2_fidelity.md`,
  where it describes an earlier planted control; `/Users/runner` in the same note, quoting a line of the
  registered `encode.log`; and `python@3.12`, a Homebrew formula name. None is a credential or a new path.
- **In hash-registered files (Thomas decision, not FAIL):**
  - `correction_v2/R1_INVESTIGATION.md`: one `C:\Users\<user>\...` path.
  - `correction_v2/results_20261019/` (`um_v2_results.json`, `manifest.json`, `encode.log`, `analysis.log`),
    `PREREG_um_v2_R1R2.md`, `prereg_um_v2_R1R2.json` and `R1R2_DESIGN_NOTES.md`: 17 home paths for the GPU
    machine's user.
  - The GPU machine's name and user name: 70 mentions in the same files.
- **Master already carries the same form (Thomas decision, not FAIL):**
  - `review/agni_paper_um_v2_corrections.json` and `..._r2.json`: one `C:/Users/<user>` path and one
    `/Users/<user>` path each.
  - `review/agni_style_um_v2_corrections_r2.json`: one `C:\\Users\\<user>` path.
  - The GPU machine's name and "Tailscale": 15 mentions, in the review JSONs (r5 included), the README and the notes.
- **0** IPv4, **0** Tailscale IPs, **0** keys or tokens, **0** personal emails.

### 2. verify_pdf both ways: PASS
- `tools/verify_pdf.py`, `--control "Edrington"`. Each check counts only if it passes on the new PDF and fails on
  the previous build (`ebc197b`). A planted present-check of a sentence in no paper fails on both new PDFs, as it should.
- **Flight `paper/main.pdf`: 10/10. Academic `paper/academic/main.pdf`: 10/10.**
  - Present: "The splits cover the 30-way task only", "18 positive, 16 negative", "scored at 5 folds where the
    others use 10", "against 9.4 on the other 816", "with three exceptions", "the binary valence split was not
    checked", "but itself and".
  - Absent: "against 9.5 on the other 816", "copied byte for byte: the preregistration", "with them set aside".

### 3. Editions changed together: PASS
- Every non-merge commit in `1748bde..HEAD` that touches `paper/main.tex` or `main.pdf` also touches its academic
  twin. `61ab741` touches both.
- **One exception:** `878c219` "AUTHOR LINE (flight edition only)", named as edition-specific in its subject.

### 4. Credit changes: PASS
- **Named commits:** `878c219` AUTHOR LINE (flight only), `d4f11da` CREDIT DISCLOSURE (no byline change), and
  `fd73609` CREDIT CHANGE (flag for Thomas).
- **Net author-field changes against master:**
  - `lyra2026oracle` goes from `{Lyra and Edrington, Thomas and CC}` to `{Lyra and Edrington, Thomas K. and Vera}`.
    The last commit to touch it is `fd73609`, the named one. `ff379e3` first undid the change that `e9c16d8` had made.
  - The academic `\thanks` line for Thomas is carried by `d4f11da`.
  - `paper.json` was reformatted and is still `["Lyra", "Thomas Edrington"]`.
  - Third-party entries changed: `Jackrong`, `{Qwen Team}`, and `barrett2006solving` removed. These are citation
    fixes, not author credit.
- `61ab741` (r5) changes no author field.

### 5. No stale PDFs: PASS
- Both PDFs were last committed in `61ab741`, which also carries their last source change. The worktree is clean
  in the paper directories.
- **Fresh rebuild:** `git archive HEAD user-model-paper` into an empty folder, then MiKTeX `latexmk -pdf`. Both
  editions exit 0 with no undefined references, and the text is identical to the committed PDFs after whitespace
  normalisation (30 pages each).
- The previous build fails all 10 of the gate 2 checks, so this comparison can tell the two builds apart.

### 6. Mergeability: PASS
- `git merge-tree --write-tree origin/master HEAD` exits 0 with no conflicts. The `REMEDIATION_REGISTER.md`
  conflict from the first pre-run was resolved by `b255dde`.

## Still Lyra's
1. **Agni paper r6** on `a736d36..HEAD`, both editions and `references.bib` (N5, CRITICAL). Then apply its
   in-scope conditions and re-run this gate.

## Waiting on Thomas
1. **The `lyra2026oracle` credit change** (`fd73609`): it drops CC and adds Vera, to match
   `oracle-loop-paper/main.tex`. It removes a person, so it is yours to approve or revert.
2. **"Available on request"** (`3683b44`): the paper says Project-Oracle's May scripts are available on request.
   That is a release promise; please confirm it can be honoured.
3. **Bylines and CRediT:** Wilkes's CRediT sign-off; the credit framing (style r3 item 3); the `lyra2026detecting`
   per-report author lists (paper r4 item 8; rewriting them would remove Wilkes and Watson).
4. **New analyses (OUT for you):** N3 (recompute the binary-split AUROCs with the leaked stories set aside) and N4
   (the leak's share of the residual-stream LR's lead over tuned text (TF-IDF)).
5. **Paths and host names in hash-registered files** (gate 1): scrubbing them means regenerating `SHA256SUMS` lines
   for released data, or you accept them as released.
6. **Paths master already carries in the same form** (gate 1, the review JSONs): scrub them or accept them on master's precedent.
7. **Older-commit paths:** local paths in older commits on this branch. History is not rewritten, and no force-push.
8. **Open flags:** whether the implicit-arm follow-up is really in progress; the `anthropic2026emotions` arXiv ID
   (needs a web check).

Not part of this branch: `notes_verify_20261101/` sits untracked in this worktree and was not committed.

## Merge order
**#17 first, then #18.** #18 cites #17's new title ("Story Emotion in the KV Cache: A Preregistered Test Against
a Text (TF-IDF) Baseline").
