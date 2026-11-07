# Home-path scrub: what changed, what stayed, and why

**Decision (Thomas, 2026-11-03):** forward scrub, the lab machine's paths first, plus a
pre-commit check. **No history rewrite, no force-push.**
**Branch:** `lyra/home-path-scrub`, cut from `origin/master` at `b7e0613`.
**Traces to:** `notes/verify_2026-11-01_2_fidelity.md`, the private-info pass that listed the
home paths.

Placeholders in this note: `/Users/<lab user>` is the lab machine's macOS home,
`C:\Users\<user>` is Thomas's Windows home, `/Users/<runner>` is a runner account's home.
This note does not spell them out, so the new check does not trip on it.

## 1. History still holds every path

This is a **forward** scrub. Every path removed here is still in the commits before `adf4c2d`,
and anyone who can clone the repository can read them. That was Thomas's choice: no rewrite
and no force-push. If the paths ever have to be gone from history, that needs a separate
decision and a coordinated rewrite. Nothing in this branch does that.

## 2. What was searched, and how files were sorted

Four patterns, over every tracked text file at `origin/master` (`b7e0613`):

| pattern | files | occurrences |
|---|---|---|
| the lab machine's home, `/Users/<lab user>` | 43 | 79 |
| Thomas's home, forward slashes, `C:/Users/<user>` | 52 (either slash form) | 13 |
| Thomas's home, backslashes, `C:\Users\<user>` | (in the 52) | 81 |
| the runner home, `/Users/<runner>` | 3 | 5 |
| **any of the four** | **93** | |

The 11-02 count of 42 files is not reproduced here and is not used. 43 files carry the lab
machine's path, and 93 carry at least one of the four.

**Hash lists checked (32 files):** 3 `SHA256SUMS`, 2 `FROZEN.sha256`, 2 `manifest.json`,
1 `RESULTS_MANIFEST.json`, and 24 JSON files that mention sha256 (preregs, review records,
`*.meta.json`). A file counted as **hashed** if its sha256 appeared in *any* tracked text file.
Three forms were tried: as committed, LF-normalised, and as checked out. A file also counted if
its name appeared in one of the 32 lists.

| class (at `origin/master`) | files | action |
|---|---|---|
| RECORD | 56 | left untouched |
| AUTHORED (md) | 13 | scrubbed |
| SCRIPT, unregistered (.py) | 24 | 23 scrubbed, 1 moved to RECORD (below) |

**Two judgment calls, both conservative:**
- `logit-bias-confab/supplementary/base_rerun/evidence/pilot2/pilot2_driver.py` is not in any
  hash list. It is still left as **evidence**, because it lives in `evidence/` beside the pilot
  data it produced, and it is the record of what ran. Final: **57 records, 36 scrubbed files.**
- `MERGE_READINESS_20261101.md` was first matched as "named in a hash list". On a re-check, the
  r6 review JSONs that name it cite it by line and carry **no** sha256 for it. It is still a
  record: it is an independent verifier's merge-readiness check, cited by line in those review
  records. Its three mentions quote a known false positive from a private-info scan. It stays
  untouched, but the reason is **evidence**, not "hashed".

## 3. What was scrubbed (36 files, two commits)

**`adf4c2d`: the lab machine's paths (15 scripts, 10 md).**
**`443d836`: Thomas's paths and the one runner path (8 scripts, 5 md).** Two md files are in
both commits, so 13 md files in all.

- **Prose:** the home prefix became `<home>`, and `host:<home prefix>` became
  `<lab machine>:<home>`. The shell line in `base_rerun/RUNBOOK.md` uses `~`, like the lines
  around it.
- **Lab-machine scripts:** each hardcoded prefix became `LAB_HOME`, an environment variable
  that defaults to `os.path.expanduser("~")`. Every literal kept its exact shape as an f-string,
  so the same user on the same machine gets byte-identical path strings. The only additions are
  `import os` and a two-line `LAB_HOME` definition before the first use.
- **Thomas's scripts:**
  - `classify_batch*.py` use `JUDGE_DIR` (default `~/Desktop`).
  - `emotion_factor_analysis.py` uses `ORACLE_DIR` (default `~/Desktop/Project-Oracle`) and
    `OUTPUT_DIR` (default `~/Desktop`).
  - `tools/byline_audit.py` and `tools/agni/style/{emit_register_section,parse_style_findings}.py`
    now resolve paths relative to their own file. From the main checkout that is the same file.
    From a worktree it is now that worktree's copy. This one difference is deliberate.
- **Checks run:** every changed script passes `python -m py_compile`. No logic was changed.
  `git diff -w --ignore-cr-at-eol --stat` equals `git diff --stat` for both commits, so there is
  no line-ending churn.

## 4. What was left, and why (57 records)

**36 Agni style-review records** under `tools/agni/style/` (`*.agni_style.json` and
`_run_log.json`) are **evidence**: review outputs, committed as produced. All 36 carry Thomas's
path. Two of them, `decision-state-paper.agni_style.json` and `hr-mode-switching.agni_style.json`,
also carry the lab machine's path.

The other 21:

| file | home path of | why it stays |
|---|---|---|
| `MERGE_READINESS_20261101.md` | runner | **evidence**: an independent verifier's merge-readiness record (098d602, bd69b57), cited by line in the r6 review records; its 3 mentions quote a known scan false positive. Not hashed (section 2) |
| `consequentiality-decomposition/supplementary/per_trial_data/consequentiality_control_results.json` | lab machine | **evidence**: a result / data JSON |
| `consequentiality-decomposition/supplementary/per_trial_data/lat_transfer_results.json` | lab machine | **evidence**: a result / data JSON |
| `consequentiality-decomposition/supplementary/per_trial_data/nonthreat_transfer_results.json` | lab machine | **evidence**: a result / data JSON |
| `logit-bias-confab/supplementary/base_rerun/evidence/pilot2/pilot2.meta.json` | lab machine | **evidence**: a result / data JSON |
| `logit-bias-confab/supplementary/base_rerun/rerun_generate.py` | lab machine | **hashed**: its sha256 is in `logit-bias-confab/supplementary/base_rerun/FROZEN.sha256`, `logit-bias-confab/supplementary/base_rerun/results/generations.meta.json` |
| `logit-bias-confab/supplementary/base_rerun/results/generations.meta.json` | lab machine | **evidence**: a result / data JSON |
| `user-model-paper/correction_v2/PREREG_um_v2_R1R2.md` | lab machine | **hashed**: its sha256 is in `user-model-paper/correction_v2/SHA256SUMS`, `user-model-paper/correction_v2/prereg_um_v2_R1R2.json`; **registered**: a prereg |
| `user-model-paper/correction_v2/R1R2_DESIGN_NOTES.md` | lab machine | **hashed**: its sha256 is in `user-model-paper/correction_v2/SHA256SUMS` |
| `user-model-paper/correction_v2/R1_INVESTIGATION.md` | Thomas | **hashed**: its sha256 is in `user-model-paper/correction_v2/SHA256SUMS` |
| `user-model-paper/correction_v2/prereg_um_v2_R1R2.json` | lab machine | **hashed**: its sha256 is in `user-model-paper/correction_v2/SHA256SUMS`, `user-model-paper/correction_v2/results_20261019/manifest.json`, `user-model-paper/correction_v2/results_20261019/um_v2_results.json`; **registered**: a prereg |
| `user-model-paper/correction_v2/results_20261019/analysis.log` | lab machine | **hashed**: its sha256 is in `user-model-paper/correction_v2/SHA256SUMS` |
| `user-model-paper/correction_v2/results_20261019/encode.log` | lab machine, runner | **hashed**: its sha256 is in `user-model-paper/correction_v2/SHA256SUMS` |
| `user-model-paper/correction_v2/results_20261019/manifest.json` | lab machine | **hashed**: its sha256 is in `user-model-paper/correction_v2/SHA256SUMS`, `user-model-paper/correction_v2/results_20261019/um_v2_results.json` |
| `user-model-paper/correction_v2/results_20261019/um_v2_results.json` | lab machine | **hashed**: its sha256 is in `user-model-paper/correction_v2/SHA256SUMS`, `user-model-paper/correction_v2/leak_split.py` |
| `user-model-paper/correction_v2/um_v2_encode.py` | lab machine | **hashed**: its sha256 is in `user-model-paper/correction_v2/SHA256SUMS`, `user-model-paper/correction_v2/prereg_um_v2_R1R2.json`, `user-model-paper/correction_v2/results_20261019/manifest.json` |
| `user-model-paper/review/agni_paper_um_v2_corrections.json` | Thomas | **evidence**: a review record |
| `user-model-paper/review/agni_paper_um_v2_corrections_r2.json` | Thomas | **evidence**: a review record |
| `user-model-paper/review/agni_style_um_v2_corrections_r2.json` | Thomas | **evidence**: a review record |
| `waystations-paper/data/induction_pilot_results.json` | lab machine | **evidence**: a result / data JSON |
| `logit-bias-confab/supplementary/base_rerun/evidence/pilot2/pilot2_driver.py` | lab machine | **evidence**: the driver that produced pilot 2, kept beside its data (judgment call, §2) |

**At the branch head, the four patterns remain in exactly these 57 files and in no other**
(re-inventoried at the head of this branch).

## 5. Found, not in scope

The check's positive control (§6) found **Linux** home paths, which are not one of the four
patterns, in 9 non-record files:
- `oracle-loop-paper/code/detection/oracle_clean.py` (3 lines). These name **two people's** homes.
- `mnemosyne-longmemeval-v10/code/{answer_one_arm.sh,build_prompts.py}`
- `mnemosyne-longmemeval-v10/docs/PROTOCOL.md`
- `mnemosyne-longmemeval-v10/reviews/release_edits.py` (2 lines)
- `logit-bias-confab/REVIEW_INDEX.md`
- `logit-bias-confab/supplementary/agni_base_model_audit.md`
- `consequentiality-decomposition/supplementary/agni_manuscript_audit.md`
- `notes/corrections_2026-10-09b_logit.md` (3 lines)

They are left for a follow-up decision. The new check already blocks *new* ones.

Also untouched:
- the bare host alias and user name where they are not part of a path;
- the Claude project-folder name in `PATCH_oracleloop_background_2026-09-07.md`, which encodes
  a user name (`C--Users-<user>`).

## 6. The check: `tools/check_home_paths.py`

- **What it rejects:** ADDED lines carrying a macOS, Windows, Git Bash or Linux home path. It
  covers any user, not only the three above. Placeholders, `~`, URLs and `/home/linuxbrew` pass.
- **Records:** allow-listed by path pattern (`RECORD_PATTERNS`). They are data, logs and
  results, review records, preregs, hash lists, `evidence/` and `results_*/` folders, and the
  hash-pinned files named in section 4.
- **CI:** `.github/workflows/home-paths.yml` runs it as one step on every pull request
  (`--base origin/<base>`). **There was no CI workflow on `master` before this branch**, so this
  is a new workflow file, not a step added to an existing one.
- **Pre-commit (optional):** `.pre-commit-config.yaml` has the entry. Without the framework, a
  local `.git/hooks/pre-commit` can run both checks:
  ```sh
  #!/bin/sh
  python tools/check_home_paths.py --staged || exit 1
  exec scripts/pre-commit-scope-guard.sh
  ```
- **Tested:**
  - `--self-test` passes 10 planted cases: 5 must fail and do (macOS, Windows-backslash,
    Git Bash, Linux, correct line number). 5 must pass and do (a record JSON, a review record,
    a path only removed, placeholders and URLs, a deleted file).
  - On real git, a staged `.md` with a planted Windows-backslash path exits 1 and names
    file:line. A staged record JSON with a home path exits 0, because it is allow-listed. An
    empty stage exits 0.
  - `--base origin/master` over this branch exits 0.
  - **Positive control:** the whole tree fed in as one added diff flags the out-of-scope Linux
    homes in section 5, and none of the four patterns outside records.
