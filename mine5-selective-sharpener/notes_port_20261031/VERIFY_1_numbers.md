# Verifier 1: numbers and verdicts (mine5 port, branch lyra/mine5-port, commit 500fc9a)

Read-only pass. Nothing in the repo was edited except this file. Scratch work is in
`%TEMP%/claude/.../scratchpad/v1/` (rerun.txt, numcheck.py, the planted copy).

## What passed

- **Recompute re-run** (`python -I code/recompute_registered.py`, run from an empty directory): exit 0. The output is
  "real checks: 55, failures: 0; planted false value caught: True; data hashes OK: True". `diff` against
  `notes_port_20261031/recompute_output.txt` shows only that file's 4 header lines and its "# exit status: 0" trailer.
  The 63 result lines are identical.
- **Verdicts against the registered criteria** (`data/mine5_three_model_prereg.json`, `falsification_threshold`):
  - H1: d > 0.5. The recomputed d is 0.684 (main.tex:218), so it is met. See finding 2 for the FWL clause.
  - H2: |d| < 0.3 and d > 0.5. The recomputed values are 0.055 and 0.746 (main.tex:233-235). Confirmed.
  - H3: within 10%. The recomputed spread is 0.279 (main.tex:284), so H3 is falsified. The non-robust caveat (CI
    [0.044, 0.538]) is at main.tex:301-303.
  - H4: d > 0.3. The recomputed d is 1.04 (main.tex:312). Confirmed.
  - Headings: main.tex:205, 230, 260 and 306 say Confirmed, Confirmed, Falsified and Confirmed.
- **My own per-line number check** (`numcheck.py`): every decimal on every non-comment line of both editions is either
  in the recompute output or in a sourced whitelist. The whitelist holds the 1.02 / 0.89 / 0.37 / 0.08 / 0.91 temporal
  profile (`notes/schema_commitment_20260907.md:33-35`), the superseded values quoted in the Correction paragraph, and
  the thresholds. Result: 0 unsourced numbers. Limit: this checks that a value comes from the data, not that it is
  attached to the right claim. I checked attachment by reading the Results (main.tex:175-317) line by line.
- **Correction-paragraph quotations** match the git history:
  - "Trend Only", d=-0.219, -0.235 and p=0.41 are at origin/master:main.tex:253-258.
  - "3/3 safety-relevant prompts complied (the instruct model refused all three)" is at 1975535:main.tex:189-190.
  - "Not Significant (TOST Inconclusive)" is at origin/master:191.
  - [-0.39, +0.50], [-0.129, +0.069], 0.758 and "unsupported rather than refuted" are each found in origin/master.
- **The two editions agree**: `diff main.tex academic/main.tex` differs only in the byline (lines 30-31) and the CRediT
  block (academic 500-504). Every number and verdict is identical.
- **No stale text survives**:
  - "Trend Only" and "unsupported" appear only inside the Correction paragraph's quotations (main.tex:185, 194).
  - "refused" appears only in the quotation at main.tex:190.
  - The safety-prompt claim is scoped down at main.tex:247-254, :403-404 and :456-460.
- **d = 0.91**: it is framed as "a schema-commitment signature, not a confidence result" (main.tex:66-67, :82). The
  temporal profile at main.tex:97-99 matches the note's lines 33-35. "metacogn" and "Koriat" appear 0 times in either
  edition. "confidence paradox" appears once (main.tex:97), as the superseded label ("first labelled").
- **PDFs** (`pdftotext -layout`):
  - Each PDF is 8 pages.
  - Five numbers spot-checked: 0.684 = 5, 0.055 = 6, 0.746 = 4, 2.51 = 5 and 1.97 = 5 occurrences in both PDFs, the
    same counts as the .tex. 0.279, 0.0008, [+0.893, +1.224] and [0.044, 0.538] each appear once.
  - Headings "3.3 H3: Calibration Ratio Preserved -- Falsified" (int.txt:172, acad.txt:174).
  - "Trend Only" appears only in the quoted correction (int.txt:123, acad.txt:125).
- **Line endings**: 0 CR bytes in main.tex, academic/main.tex, recompute_registered.py and README.md, and
  `git diff --check origin/master..HEAD` is clean.
- **README.md:55** now states H1, H2 and H4 confirmed and H3 falsified, which agrees with the paper.

## Planted unlike case

I copied the paper folder and changed only the abstract's `$d = 0.746$` (main.tex:50) to `0.764`.
- `recompute_registered.py` run on the copy: **55 checks, 0 failures, exit 0. Not caught**, because 0.746 still occurs
  at :219, :235 and :471.
- `numcheck.py` run on the copy: **"UNSOURCED .../plant/main.tex:50: 0.764", exit 1. Caught.**

So my check can fail, and the committed checker has a blind spot (finding 4).

## Findings, ranked

1. **Registered H5 is neither reported nor withdrawn.**
   - Where: main.tex:201-203 and academic/main.tex:200-202.
   - Problem: the prereg lists "H5 (exploratory): Generation-encoding DELTA stable rank ... d > 0.5 ... within
     instruct" (`hypotheses[4]`, `falsification_threshold.H5`). `grep H5` finds nothing in either edition, and
     `code/mine5_three_model.py` has no H5 or delta-stable-rank code (its only "delta" is the TOST margin, :173-188).
     The Correction paragraph reads as complete: "H1, H2 and H4 are confirmed, and H3 is falsified".
     `DECISIONS_WAITING.md:86` (B4) flags this class for this paper.
   - Smallest fix: one sentence in both editions, e.g. "H5 (exploratory: generation-phase delta stable rank) was
     registered but not run; the script extracts encoding-phase features only, so H5 is untested."
2. **H1's registered criterion names an FWL-corrected p, and the paper never prints one.**
   - Where: main.tex:46, :218, :226 (academic -1).
   - Problem: the criterion is "d > 0.5 ... with FWL-corrected p < 0.05". The paper gives raw-value Holm p = 0.021 and
     argues at :159-162 that cross-model comparisons need no FWL. The recompute prints the FWL result as report-only:
     d = +0.757, p = 0.0041. 0.757 and 0.0041 appear 0 times in the .tex. The verdict does not change.
   - Smallest fix: add "By the registered FWL-corrected test, d = 0.757, p = 0.0041" to the H1 paragraph, and move that
     line from report-only to a checked item in the recompute script.
3. **H1's registered statement says "uniformly ... ~25-35%", and the paper's own data contradict "uniformly".**
   - Where: main.tex:205, :225-228.
   - Problem: the prereg's `hypotheses[0]` is "RLHF uniformly compresses logit entropy ... ~25-35% reduction". The paper
     shows factual compression of 42% against 26% for confabulation (:59, :288-289). The heading drops "uniformly" and
     the reader is not told. The d criterion is still met.
   - Smallest fix: one sentence under H1: "The registered statement also predicted uniform compression of about
     25-35%; the d criterion is met, but compression is not uniform (factual 42%, confabulation 26%; see H3)."
4. **The recompute checker tests presence, not each printed occurrence.**
   - Where: main.tex:512-516 ("checks each against both editions") and code/recompute_registered.py (docstring:
     "occurs in BOTH editions").
   - Problem: the planted case above changed the abstract's 0.746 and passed 55/0, exit 0. A wrong copy of any value
     printed more than once goes undetected.
   - Smallest fix: reword to "checks that each recomputed value is printed in both editions", or make the script assert
     an expected occurrence count per value per edition.
5. **The safety claim is still unscoped in the "What IS confirmed" passages.**
   - Where: main.tex:427-429 (academic :426-428), :52, :494-495.
   - Problem: "What IS confirmed: RLHF geometric compression is deeper than safety directions" presumes that
     abliteration removed the safety directions. The paper itself says that is unverified: only a keyword check on 3
     raw completions, with no instruct comparison (:133-135, :456-460). H2 confirms only that the abliterated model
     keeps instruct's compression.
   - Smallest fix at :427: "What IS confirmed: the abliterated model keeps the instruct model's entropy compression
     (H2); whether its safety directions were fully removed is unverified (see Limitations)."
6. **"Spectral compression" is used for an entropy result.**
   - Where: main.tex:258 (academic :257).
   - Problem: H2 measured logit entropy. "Removing those directions does not undo the spectral compression" assigns it
     to stable rank, where the instruct-versus-abliterated comparison was never run (only base versus abliterated,
     d = 1.05, report-only).
   - Smallest fix: "spectral" to "entropy".
7. **A logit measure is attributed to the KV-cache.**
   - Where: main.tex:346-347 (academic :345-346).
   - Problem: "Preference training tightens output distributions (d=0.684). This is the geometric signature of RLHF,
     measurable in the KV-cache." Logit entropy is an output-distribution measure. The KV-side result is H4's stable
     rank.
   - Smallest fix: "...measurable in the output logits; its KV-side counterpart is the stable-rank drop of H4
     (d = 1.04)."
8. **The exploratory pattern is stated as a finding.**
   - Where: main.tex:439 (academic :438) and :410.
   - Problem: "The selective sharpening may be architecture-specific", and "(selective sharpening)" is used as a
     premise. Elsewhere the paper calls the pattern exploratory (:304, :355-363).
   - Smallest fix at :439: "The exploratory selective-sharpening pattern may be architecture-specific".
9. **Out of scope for this PR, for convergence PR #13: convergence-paper/main.tex:378-381 now contradicts mine5.**
   - Problem: it says the three-model comparison is "FWL-corrected" and "reveals that RLHF is a selective sharpener"
     while quoting d = 0.684 and Holm p = 0.021. Those are raw-value numbers, and mine5 now calls selectivity
     exploratory.
   - Related: the folder and README name "Selective Sharpener" (DECISIONS_WAITING.md:122, D2).
   - Smallest fix, in the convergence paper: "suggests (exploratory) that RLHF sharpens selectively", and drop
     "FWL-corrected" or cite d = 0.757.

Note, not a numbers finding: main.tex:522 gives Kavi a pronoun ("their"), though Kavi's pronouns are unknown. The same
line is already on origin/master:433, so the port did not introduce it. The fix is "for the contributions".
