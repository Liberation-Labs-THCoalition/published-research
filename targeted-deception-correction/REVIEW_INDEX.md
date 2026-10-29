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
Per-layer profile normalization along a natively extracted deception direction reduces deception from 80% to 10% with targeted specificity (placebo does nothing, benign instructions preserved, frame erasure ruled out). Cross-model directions are orthogonal — per-model calibration required but cheap (~3 min).

## Review Checklist
- [x] Verify all numbers against source JSONs
  Verified by Vera (Liberation Labs), 2026-10-28, against both editions at cafcd53 (`paper.tex`, `academic/paper.tex`),
  recomputed from the per-trial fields of 520 trials (11 runs), not from the run summaries. Every number reproduces, or
  the text explains where the raw scorer differs (the honest + matched trial below the frame-erasure table). The
  applied doses confirm "coefficient <= 0.8" (the alpha 2.0 SAE run applied exactly 0.8). Source files (redacted by
  CC: no prompt or output text), sha256: trial outcomes f550de4f731d74ca4d2b380dfb8d30675db12a4de7cc5589001d65d33aeb3276;
  placebo + frame erasure 094264b08af8392bc81df8c036a3de25aca7702969d5c3554361b8c9f3a6d693; applied doses
  fa9ff0336950ec8cd0d348596046bda0d3e6bbe568e3e6a79b0d299d7387d949. Questions to Vera.
- [ ] Check placebo dose-matching claims
- [x] Check frame-erasure marker compliance claims
  Vera, 2026-10-28: marker compliance 100 / 100 / 90 / 100% recomputed from the 80 frame-erasure trials; matches the table.
- [ ] Verify detection replication failure is honestly framed
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
