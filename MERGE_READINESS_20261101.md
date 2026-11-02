# PR #17 merge readiness (SOP gate pre-run), 2026-11-01 16:37 PST

Branch `lyra/user-model-correction-v2-final`, gated at head `ebc197b` (pushed; upstream equals HEAD).
Merge-base with `origin/master` is `f1ee90f`. The branch is 42 commits ahead and 45 behind.
PR stays a DRAFT. Nothing was merged. The only change in this commit is this file.

**Verdict: NOT READY TO MERGE.** Gates 2 to 5 pass. Gate 1 fails on local paths only (no IPs, keys or private
emails). Separately, the branch does not merge cleanly into current `origin/master`.

## Gate results

| # | Gate | Result |
|---|------|--------|
| 1 | Private-info scan of added lines | **FAIL** (low severity: local paths only) |
| 2 | verify_pdf on every rebuilt PDF, both ways | PASS |
| 3 | Every edition changed together | PASS (one named exception) |
| 4 | Credit changes in their own named commits | PASS (one change awaits Thomas) |
| 5 | No stale PDFs | PASS |
| x | Clean merge into `origin/master` (not an SOP item) | **FAIL**: conflict in `REMEDIATION_REGISTER.md` |

### 1. Private-info scan: FAIL (low severity)
- **What was scanned:** every added line of `git diff f1ee90f HEAD`, excluding PDFs: 1,168,257 lines including the
  data files. The text of both changed PDFs was scanned too.
- **Patterns:** IPv4, Tailscale range, Windows and POSIX home paths, emails, keys/tokens/passwords, and host names
  (MTH, the GPU and sibling hosts, the postgres containers, `localhost:<port>`).
- **Planted control:** one synthetic line set held a Tailscale IP, a `C:\Users` path, a `/home` path, a GitHub
  token, a host name, a personal email and a `PGPASSWORD`. All 7 pattern classes caught it.
- **Clean:** 0 IPv4, 0 Tailscale IPs, 0 keys or tokens, 0 personal emails. The only email-pattern hits are
  `python@3.12`, a Homebrew formula name.
- **Local paths in added lines (the fail):**
  - `user-model-paper/correction_v2/R1_INVESTIGATION.md`: a `C:\Users\<user>\AppData\...\scratchpad\...`
    path. This file is listed in `correction_v2/SHA256SUMS`, so scrubbing it also means regenerating that line.
  - `user-model-paper/review/agni_paper_um_v2_corrections.json` and `..._r2.json`: one
    `C:/Users/<user>/.../corrections/...` path each.
  - `correction_v2/results_20261019/um_v2_results.json` (6) and `manifest.json` (3): `/Users/<gpu-host-user>/...` paths
    on the GPU machine. These are released data files under SHA256SUMS.
- **Host names:** the GPU machine's name (about 70 lines across the correction_v2 docs, prereg and notes) and its
  user name (the paths above). Both are already on `origin/master`, in 45 and 47 files respectively, and
  9 files carry `C:\Users` paths. So this is not new exposure of a kind master lacks, but the gate names local
  paths, and they are present.
- **Benign hits:** `notes/verify_2026-11-01_2_fidelity.md` mentions `~/.credentials` and Tailscale only as the
  description of an earlier planted control. It holds no actual credential.
- **History:** I scanned the net diff only. Paths in older commits on this branch that are no longer in the tree
  are a known Thomas item, noted here and not failed on.

### 2. verify_pdf both ways: PASS
- **Tool and control:** `tools/verify_pdf.py`, with `--control "Edrington"`. Every check ran on the new PDF and
  again on the previous build (`ea7f255`).
- **Rule:** a check counts only if it passes on the new PDF and fails on the previous build, which shows it could
  fail.
- **Flight `paper/main.pdf`: 10/10.**
  - Present: "fine-tune of Qwen3.5-27B on reasoning traces", "Unpublished campaign reports describe", "author
    date on a rebased commit", "against a pass mark of 0.2", "revised November", "Spread across prompts
    (withdrawn)", "Linear probes only".
  - Absent: "a fifth to a third", "Implicit Emotion Arm: Word-Spotting", "mp_signal_rank".
- **Academic `paper/academic/main.pdf`: 9/9.** The same checks minus "Linear probes only", which the academic
  edition always kept.

### 3. Editions changed together: PASS
- Every commit in `f1ee90f..HEAD` that touches `paper/main.tex` also touches `paper/academic/main.tex`, and the
  same holds for the two PDFs.
- **One exception:** `878c219` "AUTHOR LINE (flight edition only)". It is named as edition-specific in its subject.

### 4. Credit changes: PASS
- **`878c219` AUTHOR LINE (flight only):** Lyra's footnote no longer reads as an Anthropic affiliation. No name is
  added or removed.
- **`d4f11da` CREDIT DISCLOSURE (no byline change):** the first-person boxes are attributed to Lyra, and the
  academic AI disclosure matches its CRediT.
- **`ff379e3` then `fd73609`:**
  - `e9c16d8` (a non-credit commit) had changed the `lyra2026oracle` byline.
  - `ff379e3` restores the original list.
  - `fd73609` "CREDIT CHANGE (flag for Thomas)" re-applies it on its own: `{Lyra and Edrington, Thomas and CC}`
    becomes `{Lyra and Edrington, Thomas K. and Vera}`.
  - Net effect: the change is carried only by the named commit. **It removes CC from a citation byline, so it
    waits on Thomas.**
- **Net author-field diff against the merge-base:**
  - Otherwise only whitespace realignment.
  - `paper.json` `"authors"` was reformatted only and is still `["Lyra", "Thomas Edrington"]`.
  - `a39e9dd` changed third-party bibliography entries: it added `Jackrong` (the model's fine-tuner), changed
    `Yang, An and others` to `{Qwen Team}`, and removed `barrett2006solving` along with its two cites in each
    edition. These are citation fixes, not credit changes for this paper's authors.

### 5. No stale PDFs: PASS
- **Commit order:** both PDFs were last committed in `ebc197b`, after the last source commit `f64ee5d` (tex and
  bib). The worktree is clean in the paper directories.
- **Fresh rebuild:** `git archive HEAD` into an empty folder, then MiKTeX `latexmk -pdf`. Both editions exited
  0 with 0 undefined.
- **Comparison:** the rebuilt text is IDENTICAL to the committed PDFs after whitespace normalisation (29 pages
  each).
- **Control:** the same comparison against the previous build (`ea7f255`) DIFFERS, so the comparison can fail.

### x. Mergeability (not an SOP item)
- `git merge-tree --write-tree origin/master HEAD` exits 1 with a **content conflict in
  `REMEDIATION_REGISTER.md`**. `README.md` auto-merges.
- The branch needs `origin/master` merged in, or a rebase with a non-force push to a new branch, before it can
  land. Rows N1 and N2 were added on this branch, and master's register has moved on since.

## Waiting on Thomas
1. **The `lyra2026oracle` credit change** (`fd73609`): CC is dropped and Vera added, to match
   `oracle-loop-paper/main.tex`. It removes a person, so it is yours to approve or revert.
2. **"Available on request"** (`3683b44`): the paper says Project-Oracle is private and its May scripts are
   available on request. That is a release promise; please confirm it can be honoured.
3. **Bylines and CRediT:**
   - Wilkes's CRediT sign-off.
   - The credit framing (style r3 item 3, OUT).
   - The `lyra2026detecting` per-report author lists (paper r4 item 8, OUT): rewriting them would remove Wilkes
     and Watson.
4. **Older-commit paths:** local paths in older commits on this branch. History is not rewritten, and no
   force-push.
5. **Gate 1 paths in the current tree** (listed above): scrub them, which means regenerating SHA256SUMS lines
   for the released data, or accept them on the precedent of master.
6. **Other open flags:**
   - The `REMEDIATION_REGISTER.md` conflict: how to bring master in.
   - Whether the implicit-arm follow-up is really in progress.
   - The `anthropic2026emotions` arXiv ID, which needs a web check.

## Merge order
**#17 first, then #18.** #18 cites #17's new title ("Story Emotion in the KV Cache: A Preregistered Test Against
a Text (TF-IDF) Baseline"). Merging #18 first would publish citations to a title master does not yet carry.
