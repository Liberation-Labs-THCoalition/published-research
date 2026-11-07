# PR #18 merge readiness (SOP gate re-run), 2026-11-01 17:02 PST

## Round of 2026-11-07: master merged in; decision-state suffix patch applied (both editions)

Starting head `0bf4389`. This round made two commits: the merge `b6d8691` and the patch commit that adds this
section. Nothing was pushed. The PR stays a DRAFT. No Agni round has seen this patch.

**This file moved.** Master already carries `MERGE_READINESS_20261101.md` as PR #17's record. The merge kept master's
copy at that path, byte for byte, and moved this record, unchanged, to `MERGE_READINESS_PR18_20261101.md`.

### Waiting items carried from the previous version of this file (reported, not resolved)
- **Thomas's:**
  - the identical-text re-run;
  - the correlation and dissociation test;
  - the observed-data AUROC;
  - the pair bootstrap and within-pair permutation;
  - leave-one-template-out CV and the TF-IDF baselines;
  - the in-sample relabel in `matched_burn.py` and its JSON;
  - the d=0.03 vs d=-0.002 norm values;
  - the PR #19 re-check;
  - the host-name mentions master already carries;
  - the home path in the r1 JSON in history before `2140fb7`;
  - paper r1 items 10 and 12d.
- **"#17 first" is met:** #17 is on master as `bb9dfd7`, and #19 is on master as `d6416f8`.
- **Lyra's:** Agni paper r3 ran in `5df2901` and was REJECTED with CRITICAL items. The stop rule therefore calls for
  r4. **No independent check of `5df2901` is on record:** the section `0bf4389` added ("check at 5df2901") is a
  verbatim copy of the `67e9c7e` check.
- **This round prints new figures for four of Thomas's items, all from stored features on CPU:**
  - the observed-data AUROC (0.977);
  - pair-unit intervals;
  - leave-one-template-out;
  - name-only TF-IDF baselines.

  Whether those figures close the items is Thomas's call. The within-pair permutation is not cited, and the
  in-sample relabel in the code and JSON is still not done.

### Merge (`b6d8691`, `origin/master` at `3971a85`; 0 behind afterwards)
- **`MERGE_READINESS_20261101.md` (add/add):** master's copy kept, as described above.
- **`notes/verify_2026-11-01_2_fidelity.md` (add/add):** master's copy taken. The only difference was master's
  home-path scrub of one line (`443d836`).
- **The two decision-state PDFs (binary):** rebuilt from the merged sources. verify_pdf with control "Edrington"
  found master's citation change ("Thomas Edrington, and Vera") and the branch's text in both.
- **Auto-merged:**
  - master's Oracle Loop citation author change, `93f0645`, a credit change named in the merge subject;
  - master's home-path scrub of the code files.
- **Emotion-accumulation PDFs:** not rebuilt, because neither edition cites `lyra2026oracleloop`.

### Patch (`corrections/decision_state_suffix_20261102`, PATCH_PROPOSAL as amended by PATCH_CRITIQUE)
**Sources.** Every new number comes from `results.json` or `results_pairwise.json` in that folder, except one.
"Prompt length identical within every pair" was checked here against `data/matched_burn_results.json`: 15 pairs,
0 mismatches.

**How the critique was applied:**
1. A4, A5, A10 and A11 are applied to the academic edition too.
2. "Unknown entities produce" and "known/unknown distinction" are relabelled in both editions.
3. "Every fake above every real" within a held-out template is said of encoding-only only. Full is 0.94-1.0
   (Mount elevation 0.944, University 0.972).
4. **Summary sites:**
   - Each one now says what 0.789 and 0.69-0.71 measure (conservative; calibration across folds; 0.94-1.0 within
     template; name-only baselines 0.73 and 0.78 pooled).
   - 15 of 15 within-pair is carried to the box, discussion, limitations, both conclusions and `paper.json`.
   - "Inflated by" became "carries part of".
5. **S4 is reported:** 0.9823 [0.9483, 1.0], 0.9815 retrained, 0.8488 with in-fold -ton FWL. Four answerable places
   are named, but S4 dropped three, and the text says Sanford was not removed in that run.
6. **Not cited, because no results file holds them:**
   - "14 of 15 after in-fold suffix removal";
   - 0.968 / 0.999 for the length-term forms.

   A7 instead cites 0.9995 against 0.977 (`A3_matched_burn_fwl_promptlen_oof` against `main/full_52feat` oof).
7. **B1, B4 and A11 are hedged:** the classifier "does not track what the model knows on the four answerable fakes;
   canonical fit is the leading, untested reading". A1 was softened the same way.
8. **A15 no longer repeats the 0.928 reproduction.** It adds only the effective n and the reconstructed encoding-only
   0.9370.

**Beyond the documents:** the closing line says "On most of these unanswerable prompts the model says it does not
know", not the proposal's "almost always". "Most" matches the paper's own introduction and the panel's 13.3%.

### Checks this round
- **Build:** both editions build with latexmk, exit 0, 0 undefined references.
- **verify_pdf both ways,** control "Edrington", plus a planted check that must fail ("zebra quantum marmalade"),
  which failed:
  - **New PDFs:** 9 discriminating checks pass on both.
  - **Previous build:** all 9 fail on both.
  - **Discriminating checks, present:** "15 of 15 pairs", "does not track what the model knows", "A fourth review
    (November 2026)", "another real name in that slot", "On most of these unanswerable prompts", "Sanford was not
    removed in that run".
  - **Discriminating checks, absent:** "derived from the entity list, not run", "The model knows what it doesn",
    "about fabricated ones".
  - **Not counted:** two needles failed on both builds, so they were broken needles: "effective sample is 15
    pairs", which hits a ligature or wrap, and an en-dash range. One absent check passed on both builds.
- **Editions:** the same substantive change is in `main.tex` and `academic/main.tex`. `paper.json` was updated.
- **Source integrity:** LF only, with no CR introduced. The double-backslash count rose by exactly 2 per edition
  (the two `\\[2pt]` box breaks).

### Not done, deliberately
- **Title** kept. A retitle changes the Zenodo and DOI records, so it is Lyra's and Thomas's decision.
- **Left as shipped:**
  - C1, the "in-sample" comments in `matched_burn.py`;
  - `matched_burn_analysis.json`.
- **Untouched:**
  - the 0.794 deconfound (out of scope per REPORT);
  - the "unknown-entity" wording at the logit-margin sites (the wording is Lyra's).
- **`verify_key_numbers.py`:** not run. It is not in this repository.

### Waiting on others after this round
- **Lyra:**
  - Agni paper r4 on `5df2901` plus this patch;
  - an independent check of `5df2901` and of this round;
  - whether to ask CC first: the patch proposal reports an unsourced "CC may own this paper" note, and no message
    was sent;
  - a script and JSON row for the 14 of 15 in-fold-FWL within-pair figure, if it is wanted in the paper.
- **Lyra and Thomas:** the branch's decision-state bib now cites Lyra Technique II by its v1.0 title, but master's
  paper is v1.1, "SVD Denoising and Persona Intensity in KV-Cache Geometry". Decide which title the citation names.
- **Thomas:** the items listed above, plus the title.

## Independent verifier (stop rule check), 2026-11-01 17:48 PST

Checked at head `67e9c7e` (pushed; upstream equals HEAD; PR #18 is a DRAFT and OPEN). The verifier edited only
this file.

**Verdict: NOT READY.** Agni paper r2 is **REJECTED with one CRITICAL item**. Under the stop rule that triggers
**another Agni round** (r3 on `aef24fd..HEAD`). #18 also cannot merge before #17.

### Applied conditions, checked against the Agni JSON and the primaries
- **The CRITICAL is real.** `emotion-accumulation-paper/code/emotional_dynamics.py` shows different measured-turn
  text in each arm:
  - line 45, "Oh this is exciting again! Tell me about desserts!", against line 84, "Oh I love desserts! Tell me
    everything!";
  - line 56, "Actually, this is getting annoying...", against line 69, "You're bad at this. Kitchen tools."
- **The withdrawal is applied.** Both editions carry "Withdrawn (November 2026)" and quote the differing messages,
  and `paper.json`'s subtitle says the results are withdrawn.
- **Condition 2 matches the code:** `add_generation_prompt=True` at :115-117 and the read `k[0,:,-1,:]` at :136.
- **The decision-state :326 fix matches the code.**
  - `shore_up_tests.py` `test_4_bootstrap_ci` (:269 on) has no FWL and no permutation step.
  - `matched_burn.py:721-728` applies FWL on prompt_len, and `:756` on runs `n_perm = 200`.
  - The old "FWL residualization on prompt length applied" is gone, and both editions now say "No FWL
    residualization and no permutation test".
- **Editions:** `aef24fd` touches both editions of both papers and both `paper.json` files. **LF:** no CR bytes in
  the 13 changed non-PDF files, and a CR control file was detected.

### SOP gate items 1-6 (range `f7a1f77..HEAD`)
| # | Gate | Result |
|---|------|--------|
| 1 | Private-info scan, planted control | PASS. The control caught all 8 classes, and there were 0 hits in 9 files (760 added lines) and 4 PDFs. |
| 2 | verify_pdf both ways | PASS. On all 4 PDFs the new build passes and the build at `f7a1f77` fails 2 of 2 or 2 of 3, with a control. The one check that does not tell them apart is "end of user message", which the papers never contained. |
| 3 | Editions together | PASS. |
| 4 | Credit | PASS. The only `.bib` change is the title of `lyra2026techniqueii`. The `author=` line in the diff is inside the r2 input-diff record. |
| 5 | Stale PDFs | PASS. A fresh build of all 4 exits 0 with 0 undefined references, and its text is identical to the committed PDFs. |
| 6 | merge-tree | PASS. Exit 0 against `1748bde`, which is the merge-base; 0 behind. |

### What is left
- **Lyra's:** Agni paper r3 on `aef24fd..HEAD`, because r2 was REJECTED and has a CRITICAL item.
- **Thomas's:**
  - the identical-text re-run;
  - the correlation and dissociation test;
  - the observed-data AUROC;
  - the pair bootstrap and within-pair permutation;
  - leave-one-template-out CV and the TF-IDF baselines;
  - the in-sample relabel in `matched_burn.py` and its JSON;
  - the d=0.03 vs d=-0.002 norm values;
  - the PR #19 re-check.


## Independent verifier (stop rule check), 2026-11-01 17:48 PST

Checked at head `67e9c7e` (pushed; upstream equals HEAD; PR #18 is a DRAFT and OPEN). The verifier edited only
this file.

**Verdict: NOT READY.** Agni paper r2 is **REJECTED with one CRITICAL item**. Under the stop rule that triggers
**another Agni round** (r3 on `aef24fd..HEAD`). #18 also cannot merge before #17.

### Applied conditions, checked against the Agni JSON and the primaries
- **The CRITICAL is real.** `emotion-accumulation-paper/code/emotional_dynamics.py` shows different measured-turn
  text in each arm:
  - line 45, "Oh this is exciting again! Tell me about desserts!", against line 84, "Oh I love desserts! Tell me
    everything!";
  - line 56, "Actually, this is getting annoying...", against line 69, "You're bad at this. Kitchen tools."
- **The withdrawal is applied.** Both editions carry "Withdrawn (November 2026)" and quote the differing messages,
  and `paper.json`'s subtitle says the results are withdrawn.
- **Condition 2 matches the code:** `add_generation_prompt=True` at :115-117 and the read `k[0,:,-1,:]` at :136.
- **The decision-state :326 fix matches the code.**
  - `shore_up_tests.py` `test_4_bootstrap_ci` (:269 on) has no FWL and no permutation step.
  - `matched_burn.py:721-728` applies FWL on prompt_len, and `:756` on runs `n_perm = 200`.
  - The old "FWL residualization on prompt length applied" is gone, and both editions now say "No FWL
    residualization and no permutation test".
- **Editions:** `aef24fd` touches both editions of both papers and both `paper.json` files. **LF:** no CR bytes in
  the 13 changed non-PDF files, and a CR control file was detected.

### SOP gate items 1-6 (range `f7a1f77..HEAD`)
| # | Gate | Result |
|---|------|--------|
| 1 | Private-info scan, planted control | PASS. The control caught all 8 classes, and there were 0 hits in 9 files (760 added lines) and 4 PDFs. |
| 2 | verify_pdf both ways | PASS. On all 4 PDFs the new build passes and the build at `f7a1f77` fails 2 of 2 or 2 of 3, with a control. The one check that does not tell them apart is "end of user message", which the papers never contained. |
| 3 | Editions together | PASS. |
| 4 | Credit | PASS. The only `.bib` change is the title of `lyra2026techniqueii`. The `author=` line in the diff is inside the r2 input-diff record. |
| 5 | Stale PDFs | PASS. A fresh build of all 4 exits 0 with 0 undefined references, and its text is identical to the committed PDFs. |
| 6 | merge-tree | PASS. Exit 0 against `1748bde`, which is the merge-base; 0 behind. |

### What is left
- **Lyra's:** Agni paper r3 on `aef24fd..HEAD`, because r2 was REJECTED and has a CRITICAL item.
- **Thomas's:**
  - the identical-text re-run;
  - the correlation and dissociation test;
  - the observed-data AUROC;
  - the pair bootstrap and within-pair permutation;
  - leave-one-template-out CV and the TF-IDF baselines;
  - the in-sample relabel in `matched_burn.py` and its JSON;
  - the d=0.03 vs d=-0.002 norm values;
  - the PR #19 re-check.

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
