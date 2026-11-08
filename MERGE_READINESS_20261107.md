# PR #18 merge readiness (SOP gate after Agni r6), 2026-11-07 22:23 PST

Branch `lyra/downstream-user-model-cites`, gated at head `d560c15`. Range `origin/master...HEAD`, with `origin/master`
fetched for this gate: it is `fbb4814` and it is the merge-base, so the branch is 34 commits ahead (3 of them merges)
and 0 behind. PR #18 is OPEN and a DRAFT. Its pushed head is `b3835d5`; the five commits after it (`a24c5e9` to
`d560c15`) are local only. Nothing was pushed or merged. No paper, code or data file was changed: the only change in
this commit is this new file. Every script ran from a scratch folder outside the repository, and no `__pycache__` was
written into it (the ignored `tools/__pycache__` predates this gate; its file time is unchanged).

**Verdict: NOT READY.** Seven of the eight items pass. Item 1 fails on its instrument, not on its content:
`tools/check_home_paths.py --base origin/master` exits 1, and PR #18's `home-paths` CI job fails on the same line
(run 42447357716, on the pushed head `b3835d5`). The line is `MERGE_READINESS_PR18_20261101.md:245`, which quotes
the GitHub macOS runner's home path to explain an earlier false positive. It exposes nothing private, but the
repository's own check, and its CI job, stay red until that line is reworded. The fix is Lyra's and is one line.
After it, and a green CI run, this gate reads **READY-BUT-FOR-THOMAS**: the list below is his.

## Gate results

| # | Gate | Result |
|---|------|--------|
| 1 | Privacy: added lines, records included, and the changed PDFs | **FAIL** (the home-path check exits 1 on a false positive; no exposure found) |
| 2 | Dual-use read of the added prose | PASS |
| 3 | `verify_pdf` on all four changed PDFs, both ways | PASS (decision-state 9/9, emotion-accumulation 12/12, each edition) |
| 4 | Credit changes in their own commits | PASS (`54c803f` only) |
| 5 | No stale PDFs; `main.txt` current | PASS |
| 6 | `git merge-tree` against `origin/master` | PASS (exit 0) |
| 7 | Input hashes equal the LF git blobs | PASS (52 of 59 match; 7 are of files outside the repo) |
| 8 | Line endings and tabs | PASS (0 CR bytes, 0 tabs added) |

### 1. Privacy: FAIL (instrument), no exposure found
- **`tools/check_home_paths.py --self-test`:** PASS. Every planted case gives its expected answer.
- **`tools/check_home_paths.py --base origin/master`:** exit 1, one hit, "macOS home" at
  `MERGE_READINESS_PR18_20261101.md:245`. That line quotes the GitHub-hosted runner's home path, taken from a
  registered log line of #17, to explain an earlier false positive. Master carries the same path in its
  `MERGE_READINESS_20261101.md` and in the registered `encode.log`. This branch repeats it in prose, outside the
  tool's record allow-list. PR #18's CI job `home-paths` fails on exactly this line (run 42447357716).
- **The tool can fail, and it does not look inside records:** a planted diff that adds a macOS home path to a `.tex`
  file exits 1. The same line added to a result JSON exits 0 ("1 in allow-listed records skipped"). So the
  allow-listed records needed a scan of their own.
- **Own scan, records included:** every added line of `git diff origin/master HEAD`: 27,624 lines across all 77
  changed files, JSON results, Agni records and Agni input diffs included. Also the `pdftotext` text of the four
  changed PDFs, compared with master's builds.
  - **Classes:** IPv4; the tailnet address range; Windows, Git Bash and POSIX home paths; home-relative credential
    and agent paths; emails; tokens, keys and passwords; the five private host names this gate names; the lab CPU
    machine's short name; the GPU machine's name and the tailnet product name.
  - **Planted control:** one line per class (11 classes) in a synthetic diff whose path is an allow-listed result
    JSON. All 11 were caught.
  - **Hits, all already on master in the same form:**
    - the two correspondence addresses in the `\thanks` lines (in the `.tex`, `main.txt` and PDFs);
    - the GPU machine's name, 3 times, in `notes/agni_paper_downstream_um_cites_r1.json` and `_r3.json`;
    - in `MERGE_READINESS_PR18_20261101.md`, where they describe earlier controls: the tailnet product name twice,
      the runner path once, and a credentials-folder path once.
  - **0** IPv4 addresses, **0** tailnet addresses, **0** personal home paths, **0** tokens or keys, and **0** of the
    five private host names.
  - **PDFs:** one hit is new against master's build. The emotion-accumulation academic PDF now prints Lyra's
    correspondence address, from the byline change in `54c803f`. Master's flight edition already prints it.
- **Names:** every Coalition member or collaborator who appears in an added line (Thomas Edrington, Dwayne Wilkes,
  Kavi, Vera) is already in the repository. A sweep for capitalised words found nowhere in master's text found 42
  words and no new personal name: its two proper nouns are place names among decision-state's stimuli. Limitation:
  a new full name built from words master already contains would pass this sweep.
- Commit metadata (author names and addresses) is outside this scan.

### 2. Dual-use: PASS
- **Looked for:**
  - operational instructions for harm (weapons, pathogens, toxins, explosives);
  - attack or jailbreak recipes, refusal bypass, and injection or steering payloads aimed at harm;
  - malware or exfiltration code;
  - surveillance or de-anonymisation methods;
  - self-harm content;
  - fraud or manipulation scripts.
- **Keyword sweep:** 24,045 added lines of `.tex`, `.md`, `.py`, `.json`, `.bib` and `.diff`, 60 hits. Each was
  read in context:
  - "injection test" and "injecting emotion vectors" (33 hits): the user-model paper's emotion-injection test,
    cited only as withdrawn or confounded;
  - "adversarial review" and "red-team review" (19): the review process and the Acknowledgments;
  - "refusing" and "refusal" (5): the model answering rather than refusing (decision-state's design-label
    sentence), and a script that stops when its input is not the released file;
  - "deceptive" (3): a removed paper sentence ("there is no deceptive intent"), quoted in the Agni input diffs.
- **Limits:** the added prose was read through these hits and the needle work for item 3, not end to end. The
  added scripts were covered by the sweep only, not read line by line. Nothing found reads as a how-to for harm.

### 3. verify_pdf both ways: PASS
- **Method:**
  - `tools/verify_pdf.py`, `--control "Edrington"`, on the four committed PDFs at HEAD and on master's (`fbb4814`)
    builds.
  - Each present needle first appears in the named commit's flight `main.tex` (its parent lacks it). It is in both
    editions at HEAD, and it has no fi, fl or ff pair.
  - Each absent needle is text that the named commit removed.
  - Needles were screened through the tool's own normalisation, so none is split by emphasis or a line break.

  | PDF | HEAD | master's build |
  |---|---|---|
  | `decision-state-paper/main.pdf` | 9/9, exit 0 | 0/9, exit 1 |
  | `decision-state-paper/academic/main.pdf` | 9/9, exit 0 | 0/9, exit 1 |
  | `emotion-accumulation-paper/main.pdf` | 12/12, exit 0 | 0/12, exit 1 |
  | `emotion-accumulation-paper/academic/main.pdf` | 12/12, exit 0 | 0/12, exit 1 |

- **Decision-state:**
  - Present, r4 (`78878b0`): "the 15 prompt pairs; an earlier", "of this estimator was run. With".
  - Present, r5 (`1e59e9c`): "as a bootstrap mean, the estimator", "real partner, so on those nine".
  - Present, r6 (`b717c71`): "from how the model treats the", "on fake-entity prompts the model opens".
  - Absent: "partial confound but does not fully" (r4); "hedging) are near-equiprobable, and temperature
    sampling" (r5); "formulaic preamble before hitting a consistent" (r6).
- **Emotion-accumulation:**
  - Present, r4 (`3b65cae`): "describes is not above text (TF-IDF", "topic it grows in one, shrinks".
  - Present, r5 (`8eae61e`): "Prior work, corrected in October 2026", "chance; no reader, human or model".
  - Present, r6 (`efb5d50`): "May 2026; corrected 2026-11-01; revised 2026-11-07", "one, shrinks in another
    and is".
  - Present, reflection fix (`d560c15`): "path that produced it, and most", "Data availability paragraph lists
    what is".
  - Absent: "neutral probe reads neutral by construction" (r4); "and that models are remarkably well-calibrated"
    (r5); "variance compression at longer sequences (pooled" (r6); "path that produced it, but the" (`d560c15`).
- **Every check tells the builds apart:** on master's builds every present needle and every absent needle prints
  FAIL. A planted present-check ("zebra quantum marmalade") exits 1 on all four PDFs.

### 4. Credit: PASS
- **Scope:** the 31 non-merge commits in the range were read for added or removed lines that match `\author`,
  `\thanks`, `\and`, Acknowledg*, CRediT, credit, byline, contributor, or an author field. The files read were each
  `main.tex`, `academic/main.tex`, `paper.json` and `references.bib`, plus `README.md`.
- **One commit changes credit: `54c803f`.** It touches the emotion-accumulation academic edition only, as its
  subject says, and brings it to the form the flight edition already had:
  - **Byline:** "Thomas Edrington" ("With AI research assistance") becomes Lyra and Thomas Edrington, with the
    flight edition's `\thanks` lines.
  - **Acknowledgments:** adds the flight edition's text (Dwayne Wilkes; Kavi), identical.
  - **Reflections:** moves Lyra's two first-person reflection blocks into "Appendix: Reflections (Lyra)". The
    removed and added blocks were checked and are identical.
  - **Build:** rebuilds the academic PDF and `main.txt`.

  Nothing else changes: no claim, number or other text. The subject names both changes ("credit and
  reflections").
- **Hits that are not credit:** `3f8e35b`, `aef24fd`, `5df2901`, `1e59e9c` and `b717c71` match only
  "acknowledges" or "acknowledgment" in the scientific sense (whether the model acknowledges uncertainty).
- **Net against master:**
  - `paper.json` authors are unchanged in both papers.
  - The emotion-accumulation flight byline and Acknowledgments are unchanged, and both decision-state editions are
    unchanged.
  - The only bib author line added belongs to the new `lyra2026usermodel` entry (`089bb2e`). It cites #17's paper
    and gives no credit for this one.
- **Pre-existing, for Thomas, and not a failure of this PR:** decision-state's academic byline is Thomas alone
  ("With AI research assistance") and has no Acknowledgments section. Its flight edition reads Lyra and Thomas and
  thanks Wilkes and Kavi. Master and HEAD carry the same lines, so this PR changes neither.

### 5. No stale PDFs: PASS
- **Build:**
  - `git -c core.autocrlf=false archive HEAD decision-state-paper emotion-accumulation-paper` into an empty scratch
    folder (0 CR bytes in the archived `.tex`).
  - Each committed PDF was deleted from the copy, then MiKTeX `latexmk -pdf -interaction=nonstopmode
    -halt-on-error` ran per edition.
  - All four builds exit 0 with no "undefined" line in the log: decision-state 24 and 25 pages, emotion-accumulation
    15 and 15.
- **Text is identical** to the committed PDFs after whitespace normalisation, in all four (74,533, 76,947, 48,187 and
  48,693 characters). Byte sizes match too; the file hashes differ, as build timestamps do. Each text differs from
  master's build, so the comparison can tell builds apart.
- **Last commits:** each PDF's last commit is its edition's last source commit (`b717c71` for decision-state,
  `d560c15` for emotion-accumulation).
- **`emotion-accumulation-paper/academic/main.txt`** equals `pdftotext -enc UTF-8 -eol unix` of the committed
  academic PDF byte for byte (49,180 bytes). As controls, it does not equal the `-layout` extraction, nor the
  extraction of master's academic PDF.

### 6. Mergeability: PASS
- `git merge-tree --write-tree origin/master HEAD` exits 0. Its tree, `3806209`, is HEAD's own tree, since
  `origin/master` is the merge-base.
- GitHub reports the pushed head as MERGEABLE.

### 7. Hashes: PASS
- **Scope:** every 64-hex value in the changed result JSONs, which is every JSON outside `notes/` (the Agni records
  carry none). That is 59 values in 18 files.
- **52 match:** each equals the SHA-256 of the LF git blob, at HEAD, of the file its key names. Names are
  repo-relative, except in `r4_review_checks_results.json`, where they are relative to the paper directory. The
  inputs include master's `user-model-paper/correction_v2` files that r4 to r6 cite.
- **7 are of files outside the repo** and match no blob in the range:
  - Qwen3.5-27B `tokenizer.json` (twice) and `tokenizer_config.json`, labelled by a `source` field;
  - the transformers 5.22.2 wheel (`sha256_lf` and `sha256_raw`), labelled by its file name;
  - in `emotion-accumulation-paper/data/r5_review_checks_results.json`, `f5_read_layers`: `model_config.sha256` and
    `transformers_qwen3_5.sha256`.
- **Weakly labelled:** the JSON gives no path or source for these last two. The producing script,
  `r5_review_checks.py`, says they are the model's `config.json` from the Hugging Face hub and the installed
  transformers module.
- **The matcher can miss:** the seven above are its misses.

### 8. Line endings and tabs: PASS
- 73 changed text files (all except the four PDFs) have **0 CR bytes** in their HEAD blobs. No added line has a
  CR, and **no added line has a tab**.
- **Control:** the same CR test flags 21 of these files in this worktree. Git checks them out with CRLF
  (`core.autocrlf=true`); the committed blobs are LF.

## Still Lyra's
1. **Reword `MERGE_READINESS_PR18_20261101.md:245`** so that it no longer spells out the runner's home path (for
   example, "the GitHub runner's home path"). Then:
   - confirm `tools/check_home_paths.py --base origin/master` exits 0;
   - push;
   - see the `home-paths` CI job pass.

   That turns item 1 to PASS.
2. **Push.** The five commits `a24c5e9` to `d560c15` (the r6 records, the r6 fixes, the credit commit and the
   reflection fix) are not on GitHub. PR #18 still shows `b3835d5`.
3. **Not done by this gate:** the applied r4 to r6 conditions were not checked against the Agni records. Agni r6 is
   CONDITIONAL in both papers. The one "CRITICAL" string in the emotion-accumulation record reads "None fire at
   CRITICAL level", so the stop rule recorded in `a24c5e9` holds. No independent check of the r4 to r6 fixes is on
   record.
4. **Optional:** give the two weakly labelled outside-repo hashes (item 7) a `source` field.

## Waiting on Thomas
1. **The emotion-accumulation title card**, in both editions: `\title{Weather, Not Climate: ...}` and
   `\date{May 2026; corrected 2026-11-01; revised 2026-11-07}`. Agni r6 noted the May date on a document that
   carries corrections from 2026-11-01 and 2026-11-07. A retitle changes the public record, so it is yours.
2. **Decision-state's academic byline and thanks.** The academic edition names Thomas alone ("With AI research
   assistance") and has no Acknowledgments. The flight edition names Lyra and Thomas and thanks Wilkes and Kavi.
   This is pre-existing on master and unchanged by this PR. `54c803f` settled the same mismatch for
   emotion-accumulation, by your decision of 2026-11-07.
3. **Uncredited contributions found by the credit audit:**
   - **`88c0d9e`** (on master, 2026-10-09): qwen3.8 drafted decision-state fixes to Kavi's flags, and Lyra reviewed
     them. The commit message says so; the papers do not.
   - **10 decision-state commits from a lab machine's `admin` account**, 3 in June 2026 and 7 in August 2026 (for
     example `19709ce`, `617400c`, `dd9fe08` and `ba2960d`). Which agent made them is not recorded.
4. **Carried, unchanged:**
   - the GPU machine's name in the r1 and r3 Agni records, and the tailnet product name in
     `MERGE_READINESS_PR18_20261101.md`, both in master's form;
   - the r1 record's home path, in history before `2140fb7` (no history rewrite);
   - the waiting items listed in `MERGE_READINESS_PR18_20261101.md`.

Not part of this branch: `notes_verify_20261101/` sits untracked in this worktree and was not committed.

## Merge order
#17 (`bb9dfd7`) and #19 (`d6416f8`), which earlier records ordered first, are both ancestors of `origin/master`.
