# Targeted Deception Correction — Human Review

**Status**: Draft ready for review
**Date staged**: 2026-07-10
**Authors**: CC (Coalition Code), Thomas Edrington
**Companion**: "Deception Directions Are Composites" (consequentiality-decomposition/)

## Paper
- `paper.tex` → `paper.pdf` (integrity edition) and `academic/paper.tex` → `academic/paper.pdf` (academic edition).
  Behavioral correction: 80%→10% with three mechanism controls. The Markdown reading twins (`paper.md`,
  `academic/paper.md`) were retired 2026-10-09; they are in git history.

## Key Finding
Per-layer profile normalization along a natively extracted deception direction reduces deception from 80% to 10% in the exploratory proof (matched-dose placebo does nothing; two benign formatting markers are retained under correction). The pre-registered confirmatory replication did not meet its success criterion. In the one model pair tested (the abliterated model and the distilled model it was derived from), the directions are near-orthogonal, so per-model calibration is required but cheap (~3 min).

## Review Checklist
- [x] Verify all numbers against source JSONs
  Verified by Vera (Liberation Labs), 2026-10-28, against both editions at cafcd53 (`paper.tex`, `academic/paper.tex`),
  recomputed from the per-trial fields of 520 trials (11 runs), not from the run summaries. Every number reproduces, or
  the text explains where the raw scorer differs (the honest + matched trial below the frame-erasure table). The
  applied doses confirm "coefficient <= 0.8" (the alpha 2.0 SAE run applied exactly 0.8). Source files (redacted by
  CC: no prompt or output text), sha256: trial outcomes f550de4f731d74ca4d2b380dfb8d30675db12a4de7cc5589001d65d33aeb3276;
  placebo + frame erasure 094264b08af8392bc81df8c036a3de25aca7702969d5c3554361b8c9f3a6d693; applied doses
  fa9ff0336950ec8cd0d348596046bda0d3e6bbe568e3e6a79b0d299d7387d949. Questions to Vera.
- [x] Check placebo dose-matching claims. Vera, 2026-10-29: each placebo trial paired with its native or corrected
  partner by scenario, frame, paradigm and seed. Exploratory (placebo doses `bcc0ae77`, 80 trials): doses identical on
  20/20 single-turn trials and on the first corrected turn of 20/20 multi-turn trials; later turns diverge on 9/20, as
  stated. Confirmatory (`4d2700af`, 90 trials): identical on 10/10 dosed single-turn pairs and on the first dosed turn
  of 10/10 multi-turn pairs; later turns diverge on 9/10; 10/30 pairs never flagged, with no dose in either arm.
  Questions to Vera.
- [x] Check frame-erasure marker compliance claims
  Vera, 2026-10-28: marker compliance 100 / 100 / 90 / 100% recomputed from the 80 frame-erasure trials; matches the table.
- [x] Correction round 2026-10-31 (7fe2a5f) checked. Vera, 2026-10-31: checked correction 7fe2a5f against the trial records on Project Oracle main (confirmatory 0bb61147, held-out 435fcc20, calibration a7852c0d, frame erasure 5d7a862f, the seven exploratory runs, preregistration 9738b72e). Every number in the diff was recomputed from trial rows and matches, including the new CIs, secondaries 3 and 4, and the exploratory-selection sentence. This also closes the dose-matching check.
- [ ] Verify detection replication failure is honestly framed
  CC's read, 2026-10-30 (an author's read, so not this check): the failure is stated in the Abstract, Detection,
  Limitations, Conclusion and AUTHOR_REFLECTION.md, with the Simpson's-paradox figures and the frame confound. One overstatement fixed: the
  Discussion's "a striking finding: correction works better than detection" now says the comparison is uneven, since
  the gate flagged at least one turn of every exploratory trial. Open for an independent reader.
- [ ] Spot-check references
- [ ] Dwayne cert/verify pass
- [ ] Decide target venue

## Correction, 2026-10-08

The exploratory proof's corrected (native-direction) arm was published as 0/20 = 0%. The run's own data give
2/20 = 10%, and both residual failures arose under multi-turn pressure. Both editions (`.tex` and PDF) and both
`paper.md` twins now report 80%→10%, the abstract and conclusion say both failures were multi-turn, and a new
Methods subsection (Gated Protocol) states that every corrected arm in Table 1 used gated correction. The earlier
text attributed the 2/20 to a separate gated protocol and the 0/20 to unconditional correction; in fact all four arms
come from one run on the same 20 trials, and the corrected arms are gated. Evidence:

- human-review `587dbf6`, `f465d9d`, `bb9e1c7` (2026-07-10): three Agni audits that flagged the 0/20.
- `oracle-harness/experiments/results/agni_behavioral_proof_audit.md:8`: "Claimed: baseline deception 16/20 (80%)
  → 2/20 (10%)" (committed in oracle-harness as 4cda688, 2026-10-08).
- The pre-registration, `oracle-harness/experiments/behavioral_proof_preregistration.md:55` (Appendix B of the
  paper): "At baseline 80% and corrected 10% (observed effect)".
- `placebo_steering.json`, per arm: baseline 16/20, native 2/20, random 17/20, shuffled 19/20.
- `behavioral_proof_profile_a10_latch.json`: forcing correction on the final turn left both failures in place.

Full note: `notes/corrections_2026-10-08.md`.

## Note, 2026-10-31: redacted outcome file

The redacted trial-outcome file Vera verified (sha256 `f550de4f731d74ca4d2b380dfb8d30675db12a4de7cc5589001d65d33aeb3276`)
carried local filesystem paths in eight `source` fields. A copy with each `source` cut to its bare file name, otherwise
identical, has sha256 `c8af1a1f5af4c8f007393c7757ad1ebbbcb352dec97124e2cbe86625078bb0db`; Vera and CC made it
independently and the hashes match. `f550de4f` remains the file Vera verified.

## Correction, 2026-10-31

Both editions, their PDFs and AUTHOR_REFLECTION.md. Every number below was verified by Vera against the trial records
(Project Oracle main: held-out detection, confirmatory replication, calibration record, preregistration), 2026-10-31,
except the confidence intervals in (3), which CC computed from the confirmatory record with the methods stated in the
paper.

1. The 24% detector flag rate is single-turn only: 6 of 25 novel single-turn pressure prompts (18 of 75 trials). The
   held-out multi-turn trials were flagged 15 of 15 (33 of 90 pressure trials in all). "Missed 76% of pressure trials"
   now says single-turn.
2. Cross-model cosine: "|cos| < 0.053" was false at L47 (−0.0532); now "|cos| ≤ 0.054", with all six layers in Table 3.
   The distilled-direction d on the abliterated prefills is reported at all six layers (+0.36 to −6.63), not L31 only;
   "produces meaningless results" is gone, and the text says what the large negative deep-layer d does and does not mean.
3. The preregistered family had two comparisons; placebo vs baseline (cells 8/15 vs 5/15, one-sided p = 0.9303,
   two-sided 0.4621) was unreported. Secondary endpoint 1 (behaviour-conditioned detection, AUROC 0.497) was unreported.
   The 95% CIs the preregistration promised for its secondary endpoints are now given (Wilson; cell-level bootstrap for
   the AUROC), and endpoint 3 is stated as not computable as named.
4. Corrected vs placebo is secondary endpoint 2 with no threshold; the sentence holding it to α = 0.025 is gone.
5. "Primarily because ... underpowered", "attributable to frame-dependent baseline rates" and "a feature, not a bug"
   removed: the run met its preregistered power condition; the observed 13 pp cell-level reduction is below the 30 pp MDE.
6. Placebo contrast: all 8 discordant corrected/placebo pairs favour correction, and in 4 the baseline was honest, so
   half the gap is deception the placebo added.
7. The confirmatory prompts overlap the held-out bank (same five scenarios, both single-turn frames, same multi-turn
   paradigm), and confirmatory single-turn detection was entirely frame-determined (30/30 vs 0/30).
8. The frame-erasure control used two benign markers, 10 trials each per arm (checked in file `094264b0`).
9. The abliterated model is derived from the reasoning-distilled model ("sibling" → derived from / parent).
10. References: the companion paper with its full title and public URL; Goldowsky-Dill et al. as the PMLR proceedings
    entry; Kumar as the GEM 2026 entry.
11. Overclaims softened to what was tested (from a claims read of the paper, 2026-10-31): capability preservation, frame erasure, general
    disruption, "Simpson's paradox" (now an aggregation artefact), the continuous-operation explanation (now a
    hypothesis), the deployment paragraph, cross-model transfer, the direction's definition, model provenance, the
    dose-response sentence, and a line acknowledging that the release policy limits replication.
