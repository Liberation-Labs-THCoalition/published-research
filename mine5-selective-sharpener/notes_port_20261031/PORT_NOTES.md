# mine5 port notes: human-review PR #4 → published-research

Port made 2026-10-31 (date from `date -u` in `recompute_output.txt`, line 2), on branch
`lyra/mine5-port` from `origin/master` = `e667f41`. **Lyra's full read is still pending.**

## Why

human-review `main` archived this paper on 2026-10-20 (`bd428f5c`) because a published copy
exists, so the correction in human-review PR #4 can't land there. Lyra's 2026-10-29 comment on
PR #4 says not to merge it in human-review. The published copy still headed H4 "Trend Only",
headed H2 "Not Significant (TOST Inconclusive)" (academic), had no registered verdicts and shipped
no data or recompute script. This port brings PR #4's correction into both published editions.

## Source

- human-review PR #4: `lyra/mine5-correction`, head `54c7f9d12aecafff72bc9f31598438ea302c25e6`,
  merge base with `origin/main` `b05561db7eab3bf2f3ecc18c90150e49e7f6a2f8`, 9 commits
  (`a49b2ce` `6e2debd` `c7bfd1f` `64b9c20` `c69167c` `7fa1777` `f0477fb` `371752a` `54c7f9d`).
- human-review is read-only for this port. Nothing was pushed there and PR #4 was not touched.

## Method

1. Three-way merge of each `.tex` in scratch: `git merge-file --diff3` with published
   (`origin/master`) as ours, PR #4's base `b05561db` as base and PR #4's head as theirs.
   This gave 5 conflicts in `main.tex` and 11 in `academic/main.tex`.
2. In every conflict, PR #4's text replaces the published text (table below). I resolved
   them all to PR #4's side with `git merge-file --theirs` and diffed the result against PR #4's head:
   - `main.tex`: no difference at all;
   - `academic/main.tex`: one line, 229. The published H2 heading "--- Not Significant (TOST
     Inconclusive)" merged cleanly, because PR #4 left the base's "--- Confirmed" untouched.
     PR #4's Correction paragraph withdraws that heading, so line 229 must read
     `\subsection{H2: Abliteration Does Not Reverse Compression --- Confirmed}`, as PR #4's head has it.
3. So both `.tex` files are PR #4's head, byte for byte. They were written with
   `git show 54c7f9d1:<path> > <file>`, and `git hash-object --no-filters` matches PR #4's blobs:
   `main.tex` `23f7065…`, `academic/main.tex` `21c5495…`.

## Hunk-by-hunk mapping (conflicts; PR #4 commits by `git blame` on the head lines)

| # | edition | PR #4 head lines | published text replaced | PR #4 commits |
|---|---|---|---|---|
| C1 | main | 54-70 | abstract: "contrary to our pre-registered prediction … moves in the improved direction" | 7fa1777, c69167c |
| C2 | main | 76-83 | intro: "confidence paradox … What they say instead, at this sample size" | 7fa1777, c69167c, a49b2ce |
| C3 | main | 260 | H3 heading "Prediction Unsupported, Selective Effect Exploratory" → "Calibration Ratio Preserved --- Falsified" | 7fa1777 |
| C4 | main | 325-327 | "RLHF as miscalibrator … unsupported" → "Calibration preserved (H3) … falsified" | 7fa1777 |
| C5 | main | 469-480 | Conclusion: "not supported by these data" → "fails"; both H2 criteria | 7fa1777, 371752a |
| C1 | academic | 47-51 | abstract H2: "no significant difference … underpowered for formal TOST" → both registered criteria | 7fa1777, a49b2ce |
| C2 | academic | 53-69 | abstract H3 (as main C1) | 7fa1777, c69167c |
| C3 | academic | 75-82 | intro (as main C2) | 7fa1777, c69167c, a49b2ce |
| C4 | academic | 231-253 | H2 body: "a properly specified TOST at n=30 does not confirm equivalence" + safety check → registered criteria; safety check scoped to the abliterated model only | 7fa1777, 54c7f9d |
| C5 | academic | 259 | H3 heading (as main C3) | 7fa1777 |
| C6 | academic | 278-316 | H3 body incl. interval [−0.129, +0.069], p = 0.758 → registered spread 0.279 with BCa CI [0.044, 0.538]; H4 section | 7fa1777, 371752a |
| C7 | academic | 324-326 | "RLHF as miscalibrator" (as main C4) | 7fa1777 |
| C8 | academic | 354-362 | "Selective sharpening is suggested but not confirmed" → "… and exploratory" | 7fa1777, a49b2ce |
| C9 | academic | 418-431 | "For Understanding RLHF": "not confirmed at the current sample size" → "not pre-registered and remains exploratory" | a49b2ce, 54c7f9d, 7fa1777, c69167c |
| C10 | academic | 468-482 | Conclusion (as main C5) | 7fa1777, c69167c, 371752a, a49b2ce |
| C11 | academic | 491-497 | closing first-person block: "bootstrap CI crosses zero" → "not pre-registered" | c69167c, 7fa1777 |

Plus the one clean-merge fix: academic line 229, the H2 heading (Method, step 2).

What PR #4 withdraws and what replaces it is set out in its Correction (2026-10-20) paragraph,
`main.tex:175-203` (`academic/main.tex:174-202`). It withdraws:
- the July/August "TOST Inconclusive" heading and its interval;
- the [−0.129, +0.069] interval, which has "no traceable source and cannot be reproduced";
- the "3/3 … instruct model refused all three" claim (instruct was never tested);
- the "Trend Only" H4.

## Published-only changes kept

Every change that exists only in the published copy is either in PR #4's head in the same words
or is replaced and quoted by the Correction paragraph:
- academic title "RLHF Compresses Spectral Geometry …" (both editions, line 25);
- Dwayne Wilkes off the byline (both editions);
- Acknowledgments line (`main.tex:520`, `academic/main.tex:525`);
- academic CRediT block with the Dwayne Wilkes and Kavi rows (`academic/main.tex:500-503`).

There is no authorship change relative to the published copy: `git diff -U0 origin/master` on both
`.tex` files shows no changed line naming Dwayne, Kavi, Edrington, `\author`, `\thanks`,
CRediT or Acknowledgments. That grep can find such lines: run on PR #4's own base..head diff,
it does find them, including c7bfd1f's byline and Acknowledgments lines.

Also kept:
- **Both `references.bib` files are unchanged.** They carry the DOI fields (`c5be832`,
  `3861c20`) that PR #4's copies lack. Every `\cite` key in PR #4's head resolves in them.
- **`SOP_REVIEW.md` is unchanged.** Its blob is identical in both repos and PR #4 did not touch
  it. It is now stale: `REMEDIATION_REGISTER.md:367` says it "certifies the retracted analysis as
  PASS, with no supersession banner". Not fixed here; it's flagged for Lyra.

## Added files (byte-identical to PR #4's head)

`code/.gitattributes`, `code/mine5_three_model.py`, `code/recompute_registered.py`,
`data/.gitattributes`, `data/mine5_three_model_20260715_021501.json`,
`data/mine5_three_model_prereg.json`.

- For each file, the sha256 of `git show 54c7f9d1:<path>` equals the sha256 of the written file,
  and `git hash-object --no-filters` equals PR #4's blob id.
- Independent check: the data file's sha256 `2d950924…` matches the prefix the paper prints
  (`main.tex:506`).
- The script's sha256 `8155afcb…` matches `code/.gitattributes`.
- Both `.gitattributes` set `* -text`, so this repo's `core.autocrlf=true` cannot rewrite these files.

## Recompute (`recompute_output.txt`)

`python mine5-selective-sharpener/code/recompute_registered.py` was run from an empty directory on
Python 3.11.0, numpy 2.3.1 and scipy 1.16.1.

- It ended with "real checks: 55, failures: 0; planted false value caught: True; data hashes OK:
  True", exit status 0.
- It checks every printed value against **both** editions (`EDITIONS`, script line 42), so it
  checked this port's `.tex` files.

## PDFs

- **Build:** MiKTeX pdfTeX-1.40.25 (`pdflatex`, `bibtex`, then `pdflatex` twice), the same
  producer as the published PDFs. Each edition is 8 pages. Build byproducts were removed.
- **Bibtex warning:** "empty journal in gurnee2026gwt". That entry is unchanged by this port.
- **`tools/verify_pdf.py`, control "Data and Code", both editions:**
  - present: H1 "— Confirmed", H2 "— Confirmed", H3 "Preserved — Falsified", H4 "— Confirmed";
  - absent: "Stable Rank — Trend Only", "Compression — Not Significant",
    "Ratio — Prediction Unsupported", "Ratio — Falsified (Opposite Direction)".
- **Absent checks can fail:** run on the old published PDF, the same tool reports FAIL for
  "Stable Rank — Trend Only", "Ratio — Prediction Unsupported" and "improves calibration".
- **Quoted phrases:** "Trend Only", "Not Significant (TOST Inconclusive)" and "−0.129" each
  appear once in the rebuilt PDFs, inside the Correction paragraph that quotes them as withdrawn.
- **Banned words:** "metacogn" and "Koriat" appear 0 times.
- **Compared with PR #4's own PDFs:** the only word-level differences in pdftotext output are the
  DOI lines from the published `.bib` files and one page-number shift.

## d = 0.91

The paper frames d = 0.91 as "a schema-commitment signature, not a confidence result"
(`main.tex:67`, `:82`).

- That agrees with `notes/schema_commitment_20260907.md`:
  - lines 35 and 94: deep content d = 0.37, p > 0.08;
  - line 42: decay to non-significance by token 20;
  - line 77: Koriat is a contrast, not a convergence.
- The paper does not mention Koriat or metacognition.

## README decision

`README.md:55` had the Summary "Geometric evidence that RLHF improves calibration rather than
degrading it". The registered verdicts contradict this, because H3 (calibration ratio preserved
within 10%) is falsified and the ratio rise rests on selective sharpening, which was not
pre-registered and remains exploratory. **I replaced the Summary cell only**, with: "RLHF compresses logit
entropy and V-projection stable rank, and abliteration does not undo the entropy compression
(H1, H2, H4 confirmed); the pre-registered prediction that the calibration ratio is preserved (H3)
is falsified".

Left alone:
- the link text "MINE5 Selective Sharpener" and the directory name (open decision D2,
  `DECISIONS_WAITING.md:122`);
- the Authors cell, which still lists Dwayne Wilkes. That contradicts the byline policy
  (`README.md:61`, `AGENTS.md` section 3), but ten other rows have the same problem (README lines
  17, 18, 27, 28, 34-37, 48 and 54). It's a corpus-wide fix, not a mine5 claim, so it's for Lyra.

## Not done here (flagged)

- **`convergence-paper/main.tex:379-385`** still says "selective sharpener, not a miscalibrator"
  and "improving the calibration ratio", without H3's falsification. PR #13 (`lyra/convergence-v11`)
  rewrites this.
  - PR #13's `main.tex:446` cites "the corrected Mine~5 paper" for [+0.010, +0.270], an interval
    that exists in published-research only after this port lands.
  - **So this port must merge before or together with PR #13.**
- **Audit records that go stale when this lands, left unedited:**
  - `REMEDIATION_REGISTER.md:396` ("no results artifact ships");
  - `REMEDIATION_REGISTER.md:928` and `DECISIONS_WAITING.md:86` (B4: "a pre-registered hypothesis
    is neither reported nor withdrawn"; H4 is now reported);
  - `REMEDIATION_REGISTER.md:367` (`SOP_REVIEW.md`).
