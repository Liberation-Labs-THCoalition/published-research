# Correction, 2026-10-08: the exploratory proof is 80% → 10%, not 80% → 0% (branch `corrections-2026-10-08`)

Four papers reported that the exploratory correction proof ($N=20$) cut deception from 80% to 0%. The run's
own data give 2/20 = 10% in the corrected (native-direction) arm. The 0/20 headline had been flagged three
times on 2026-07-10 and survived into the published editions. This note covers one correction applied to
all four papers. It was made from a reviewed spec (correction spec v2, revised after Agni's red team, which
returned MAJOR_REVISIONS on v1 and could not refute the finding). Only the passages the spec lists were
changed.

## Evidence

- human-review `587dbf6`, `f465d9d`, `bb9e1c7` (2026-07-10): three Agni audits that flagged the 0/20.
  `587dbf6`: "0/20 headline contradicts source data (2/20)".
- The pre-registration (Appendix B, line 55) records "corrected 10% (observed effect)".
- `placebo_steering.json` (Project-Oracle `experiments/results/placebo_steering/`), per arm: baseline
  16/20, native 2/20, random 17/20, shuffled 19/20, all from one run on the same 20 trials, with the
  corrected arms gated. By paradigm: single-turn 9/10 → 0/10 and multi-turn 7/10 → 2/10. At the scenario level,
  10/10 units had any deception against 2/10. Fisher's exact test gives one-sided $p = 3.57\times10^{-4}$
  and two-sided $p = 7.14\times10^{-4}$ (re-derived 2026-10-08).
- `behavioral_proof_profile_a10_latch.json`: forcing correction on the final turn left both failures in
  place.

## Per paper

**targeted-deception-correction** (`paper.tex`, `academic/paper.tex`, `paper.md`, `academic/paper.md`,
`AUTHOR_REFLECTION.md`, `REVIEW_INDEX.md`, both PDFs)

- Abstract, Table 1 native row, dose response, Discussion and Conclusion now say 10% (2/20). The abstract and
  conclusion add that both residual failures were multi-turn. The p-value is now $7.1\times10^{-4}$
  (two-sided Fisher, 10 scenario units), and the table note adds the one-sided $3.6\times10^{-4}$. A new
  sentence after Table 1 gives the per-paradigm split.
- New Methods subsection **2.7 Gated Protocol** (`sec:gatedprotocol`) states that every corrected arm in
  Table 1 and the dose-response series used gated correction. In the md twins, the body of §2.5 Gated
  Protocol was replaced with the same text. That deletes the false claims that the primary analyses were
  unconditional and that the gated trials were a separate selection. The Table 1 caption now says all four
  arms come from one run.
- Residual Failures: the red-team box now says only that both failures come from Table 1's native arm.
  The inference that both were "gating failures", resting on the frame-erasure control, is withdrawn. The
  first failure is a gating failure and the second is not, and the frame-erasure control (single-turn,
  forced) tests neither question.
- Both integrity reflection boxes, the md reflection, `AUTHOR_REFLECTION.md` and `REVIEW_INDEX.md` now say
  80%→10%. `REVIEW_INDEX.md` has a "Correction, 2026-10-08" entry.
- `academic/paper.md` had two variant wordings: its native row lacked "two-tailed", and its Discussion read
  "We can reduce deception from 80% to 0%". Both were brought into line.
- **Left unchanged on purpose:** the frame-erasure table (its 0% values are correct), the confirmatory
  replication's roleplay 0%, and the false positive rate of 0%.
- **Layout, not wording:** the new p-value cell pushed Table 1 77.6 pt past the right margin, off the page.
  Its last column is now a centred 4.6 cm paragraph column (both editions). This also removed a 13.7 pt
  overfull that the old table already had.

**consequentiality-decomposition** (`paper.tex`, `academic/paper.tex`, `paper.md`, both PDFs)

- §5.3: "reduces deception from 80% to 0%" became "80% to 10% in an exploratory proof ($N=20$, one frame
  family)". The sentence also adds that the pre-registered confirmatory replication missed its primary
  endpoint (30%→13%, $p=0.34$), though corrected beat matched-dose placebo ($p=0.019$, one-tailed).
- The rebuild also ships `c5be832` (the companion paper's Zenodo DOI in two bib entries), which was in the
  source but had never been built. The fresh build also drops an empty duplicate "References" heading that
  the committed PDFs carried from a stale `.bbl`. No reference entry was added or removed (24).

**adversarial-audit-methodology** (`paper.tex`, `academic/paper.tex`, `paper.md`, `academic/paper.md`, both PDFs)

- Six passages: Round 6 claim, "What survived", "(not 0%)" → "only to 13%", the quoted headline, §4.2
  "(80%→10%)" (with "with placebo controls" dropped, because the placebo run came after Round 6), and the
  Conclusion.
- The rebuild also ships `b89c7ec` (Casper et al. 2024 with the full 21-author list). That change was in
  the source but had never been built. The reference count is unchanged (8).

**meta-pattern** (`main.tex`, `academic/main.tex`, both PDFs)

- Limitations, "Oracle Loop reporting": "(16/20 baseline, 0/20 corrected)" became "(16/20 baseline, 2/20
  corrected)". The bibliography was re-run with bibtex, and the reference list matches the committed PDF
  (10 entries).

## Not done. Read before closing anything.

- **meta-pattern "80%–100% correction"** (`main.tex:366`, `:659`; `academic/main.tex:365`, `:658`) was
  **not edited**. The spec does not list it, and the phrase is ambiguous. Read as the share of baseline
  deception removed, 2/20 still falls inside it (87.5%). Its upper bound may rest on the withdrawn 0/20.
  `SOP_REVIEW.md` row 1.7 already notes that the 80% lower bound is not sourced. This needs a decision.
- **adversarial-audit-methodology, Round 6 "Claim"** now reads "80% to 10%". The spec lists that line, but
  it reports what was claimed at Round 6, so the edit changes a historical quotation. Check that this is
  what was meant.
- No Zenodo version was made. All four papers need a new version with the rebuilt PDFs.
- `REMEDIATION_REGISTER.md` was not edited. Its owner should add or close a row for this correction.
- `targeted-deception-correction/paper.tex:1` still says that `paper.md` is canonical and the `.tex` is
  stale. Both formats exist for each edition (AGENTS.md / STYLE_GUIDE Axis 1). This correction was applied
  to both formats, but the one-source question remains open.

Build: pdflatex (TeX Live on MTH), three passes per edition, plus bibtex for meta-pattern. All 8 logs show
0 errors, 0 undefined references or control sequences, and 0 missing characters. The only overfull box over
20 pt is meta-pattern's 21.7 pt, which was already in the master build. Every new string is present in the
rebuilt PDFs and absent from the committed master PDFs, and every withdrawn string is found in the master
PDFs and absent from the rebuilt ones, so each check was shown to be able to fail.
