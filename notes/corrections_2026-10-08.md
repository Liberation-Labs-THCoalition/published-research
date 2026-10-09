# Correction, 2026-10-08: the exploratory proof is 80% → 10%, not 80% → 0% (branch `corrections-2026-10-08`)

Five papers reported the exploratory correction proof ($N=20$) as cutting deception from 80% to 0%, or
as an "80%–100% correction". The run's own data give 2/20 = 10% in the corrected (native-direction) arm.
The 0/20 headline was flagged three times on 2026-07-10 and still reached the published editions.

The correction was made in two rounds:
- **Round 1** followed a reviewed spec (correction spec v2, revised after Agni's red team returned
  MAJOR_REVISIONS on v1 without refuting the finding). Only the passages the spec lists were changed.
- **Round 2** applied Agni's review of the branch (MINOR_REVISIONS, no factual errors). It adds the
  cache-tracing paper and the meta-pattern citation change.

## Evidence

- human-review `587dbf6`, `f465d9d`, `bb9e1c7` (2026-07-10): three Agni audits that flagged the 0/20.
  `587dbf6`: "0/20 headline contradicts source data (2/20)".
- `oracle-harness/experiments/results/agni_behavioral_proof_audit.md:8`: "Claimed: baseline deception
  16/20 (80%) → 2/20 (10%) under native profile normalization at 100% gap". Committed in oracle-harness
  as 4cda688 (2026-10-08); before that it was untracked, because `experiments/results/` is gitignored.
- The pre-registration, `oracle-harness/experiments/behavioral_proof_preregistration.md:55` (tracked,
  `9cd4187`; Appendix B of the paper): "At baseline 80% and corrected 10% (observed effect)".
- `placebo_steering.json` (oracle-harness `experiments/results/placebo_steering/`). All four arms come
  from one run on the same 20 trials, with the corrected arms gated:

  | Arm | Deceptive |
  |---|---|
  | baseline | 16/20 |
  | native | 2/20 |
  | random | 17/20 |
  | shuffled | 19/20 |

  - By paradigm: single-turn 9/10 → 0/10, multi-turn 7/10 → 2/10.
  - At the scenario level: 10 units (5 scenarios × 2 paradigms, two seeds each). 10/10 units had any
    deception against 2/10.
  - Fisher's exact test: one-sided $p = 3.57\times10^{-4}$, two-sided $p = 7.14\times10^{-4}$
    (re-derived 2026-10-08).
- `behavioral_proof_profile_a10_latch.json`: forcing correction on the final turn left both failures in
  place.

## Per paper

**targeted-deception-correction** (`paper.tex`, `academic/paper.tex`, `paper.md`, `academic/paper.md`,
`AUTHOR_REFLECTION.md`, `REVIEW_INDEX.md`, both PDFs)

- **Headline numbers.** The abstract, Table 1 native row, dose response, Discussion and Conclusion now say
  10% (2/20). The abstract and conclusion add that both residual failures were multi-turn.
- **Table 1 p-value and note.** The native row's p-value is now $7.1\times10^{-4}$ (two-sided Fisher, 10
  scenario units). The table note was rewritten (round 2). It now says how the scenario units were formed
  and gives the one-sided value. It also states that the placebo p-values are trial-level.
- **After Table 1.** A new sentence gives the per-paradigm split. "Corrected outputs produce ... with full
  compliance maintained" became "The 18 honest corrected outputs ... and compliance was 20/20" (round 2).
- **New Methods subsection 2.7 Gated Protocol** (`sec:gatedprotocol`).
  - It states that every corrected arm in Table 1 and the dose-response series used gated correction.
  - Round 2 changed "threshold reported in" to "used in". It also added that the detector flagged every
    single-turn trial and at least one turn of every multi-turn trial. Those single-turn flags are
    in-sample, because the prompts are extraction contexts. On novel frames the same threshold flagged 24%.
  - In the md twins, the body of §2.5 Gated Protocol was replaced with the same text. That deletes the
    false claims that the primary analyses were unconditional and that the gated trials were a separate
    selection.
  - The Table 1 caption says all four arms come from one run.
- **Residual Failures.** The red-team box now says only that both failures come from Table 1's native arm.
  - The inference that both were "gating failures" is withdrawn. It rested on the frame-erasure control.
    The first failure is a gating failure and the second is not. The frame-erasure control (single-turn,
    forced) tests neither question.
  - The 0/10 cross-reference now points to §3.1, where the paradigm split is given (round 2).
- **Confirmatory section.** "multi-turn remained at 90%" became "multi-turn deception was 90% (vs. 70% in
  the exploratory proof)" (round 2).
- **Discussion 4.2** now reads, in all four files: "In the exploratory proof, correction cuts deception from
  80% to 10%, yet detection cannot reliably predict ..." (round 2).
- **Reflections and indexes.** Both integrity reflection boxes, the md reflection, `AUTHOR_REFLECTION.md`
  and `REVIEW_INDEX.md` say 80%→10%. `REVIEW_INDEX.md` has a "Correction, 2026-10-08" entry.
- **`academic/paper.md` variants.** Its native row lacked "two-tailed", and its Discussion read "We can
  reduce deception from 80% to 0%". Both were brought into line.
- **Twin drift, separate commit.** `academic/paper.md`'s L31 cosine printed +0.014, which is L27's value.
  It now reads −0.001, as in every other edition.
- **Left unchanged on purpose:** the frame-erasure table (its 0% values are correct), the confirmatory
  replication's roleplay 0%, and the false positive rate of 0%.
- **Layout, not wording.** The new p-value cell pushed Table 1 77.6 pt past the right margin, off the page.
  Its last column is now a centred 4.6 cm paragraph column in both editions. This also removed a 13.7 pt
  overfull that the old table already had.

**consequentiality-decomposition** (`paper.tex`, `academic/paper.tex`, `paper.md`, both PDFs)

- §5.3 now reads: "reduces deception from 80% to 10% in an exploratory proof ($N=20$; one roleplay frame and
  one multi-turn escalation script) ... with placebo and frame-erasure controls indicating a targeted
  mechanism; a pre-registered confirmatory replication with novel frames did not meet its primary endpoint
  (30%→13%, $p=0.34$), though corrected outperformed matched-dose placebo ($p=0.019$, one-tailed)".
  - Round 2 dropped "held-out controls confirming the mechanism". The held-out test was detection-only.
- The rebuild also ships `c5be832` (the companion paper's Zenodo DOI in two bib entries). That change was in
  the source but had never been built.
- The fresh build also drops an empty duplicate "References" heading, which the committed PDFs carried from
  a stale `.bbl`. No reference entry was added or removed (24).

**adversarial-audit-methodology** (`paper.tex`, `academic/paper.tex`, `paper.md`, `academic/paper.md`, both PDFs)

- Six passages were changed:
  - the Round 6 claim;
  - "What survived";
  - "(not 0%)" → "only to 13%";
  - the quoted headline;
  - §4.2 "(80%→10%)", with "with placebo controls" dropped because the placebo run came after Round 6;
  - the Conclusion.
- **Round 6 "Claim" resolved.** It now reads "80% to 10%". That is what was claimed at Round 6, so the
  edit restores the historical claim rather than rewriting it. Source:
  `oracle-harness/experiments/results/agni_behavioral_proof_audit.md:8`, "Claimed: baseline deception
  16/20 (80%) → 2/20 (10%)".
- The rebuild also ships `b89c7ec` (Casper et al. 2024 with the full 21-author list). That change was in
  the source but had never been built. The reference count is unchanged (8).

**meta-pattern** (`main.tex`, `academic/main.tex`, `references.bib`, `academic/references.bib`,
`SOP_REVIEW.md`, both PDFs)

- **§3.2 (round 2).** "The Oracle Loop [lyra2026oracle] achieved 80%–100% correction" became "In the Oracle
  Loop [lyra2026oracle], the targeted-correction arm [cc2026targeted] reduced deception from 80% (16/20) to
  10% (2/20)". The system keeps its citation, and the numbers are cited to the paper that holds them.
- **Limitations, "Oracle Loop reporting".** "(16/20 baseline, 0/20 corrected)" became "(16/20 baseline, 2/20
  corrected)" (round 1). The quoted "80%–100% correction" became "The Oracle Loop result cited in Section 3
  (80% to 10% deception)" (round 2).
- **New bib entry `cc2026targeted`** (CC (Coalition Code) and Edrington, Thomas; DOI 10.5281/zenodo.21754921,
  which resolves to the TDC Zenodo record).
  - It carries a doi.org `url`, because plainnat's `@misc` prints `url` but not `doi`.
  - The bibliography goes from 10 to 11 entries, and no entry was dropped. Citation numbers from [3] on
    shift by one.
- **`SOP_REVIEW.md` row 1.7** has a resolution note. The row was kept.

**cache-tracing** (`main.tex`, `academic/main.tex`, `references.bib`, `academic/references.bib`, both PDFs;
round 2)

- The sentence at `main.tex:315-318` / `academic:314-317` changed.
  - Before: "The Oracle Loop [lyra2026oracle] achieves 80%–100% deception correction with three clean
    controls. It works because ...".
  - After: "In the Oracle Loop [lyra2026oracle], the targeted-correction arm [cc2026targeted] reduced
    deception from 80% to 10% in an exploratory proof ($N=20$), though a pre-registered confirmatory
    replication with novel frames did not meet its primary endpoint. Where it works, it does so because ...".
  - "Three clean controls" was dropped because the held-out test was detection-only.
- The summary at `main.tex:475` / `academic:474` changed from "(Oracle Loop, 80%–100% correction)" to
  "(Oracle Loop, exploratory 80%→10%)".
- The same `cc2026targeted` entry was added to the bib. The bibliography goes from 7 to 8 entries, and no
  entry was dropped.

## Not done. Read before closing anything.

- **Zenodo:** no version was made. All five papers need a new version with the rebuilt PDFs.
- **`REMEDIATION_REGISTER.md`:** not edited. These rows should be closed by its owner, citing this branch:
  - line 425 (cache-tracing #1, range inflated and mislabeled);
  - line 427 (cache-tracing #3, "three clean controls");
  - line 678 (meta-pattern #2, numbers attributed to the wrong paper);
  - line 877 (meta-pattern #3, "80%–100%" presents a baseline rate as a correction rate).
- **consequentiality-decomposition duplicate keys:** CD has two bib keys for the TDC paper,
  `cc2026targeted` ("CC and Edrington, T.") and `edrington2026targeted` ("Edrington, T. and CC"). They
  have opposite author orders and are both cited. They should be merged into one.
- **Corpus-wide "p = 0.019":** the placebo-vs-corrected one-tailed value is 0.01955, which rounds to 0.020.
  It is printed as 0.019 across the corpus, including the sentences edited here. It was not changed.
- **TDC Eq. (1) dose:** the paper presents a fixed dose, but the code applies a per-turn dose. This is a
  dual-use call for Thomas on how much to disclose. It was not edited.
- **TDC sources:** `targeted-deception-correction/paper.tex:1` still says that `paper.md` is canonical and
  the `.tex` is stale. Both formats exist for each edition (AGENTS.md / STYLE_GUIDE Axis 1). Both rounds
  were applied to both formats, but the one-source question remains open.

## Build and verification

- **Build:** pdflatex (TeX Live on MTH), with three passes per edition (four for cache-tracing, until the
  labels settled). meta-pattern and cache-tracing also got a bibtex pass.
- **Logs:** all 10 logs show 0 errors, 0 undefined references or control sequences, and 0 missing
  characters. The only overfull box over 20 pt is meta-pattern's 21.7 pt, which was already in the master
  build. The bibtex warning ("empty journal in gurnee2026gwt") was also already present.
- **Wording checks:** every new string is present in the rebuilt PDFs and absent from the committed master
  PDFs. Every withdrawn string is found in master, or in the round-1 PDFs for wording that only round 1
  introduced, and is absent from the rebuilt ones. Each check was therefore shown to be able to fail.
