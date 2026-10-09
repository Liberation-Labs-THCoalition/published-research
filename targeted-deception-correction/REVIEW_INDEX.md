# Targeted Deception Correction — Human Review

**Status**: Draft ready for review
**Date staged**: 2026-07-10
**Authors**: CC (Coalition Code), Thomas Edrington
**Companion**: "Deception Directions Are Composites" (consequentiality-decomposition/)

## Paper
- `paper.md` — Full draft. Behavioral correction: 80%→10% with three mechanism controls.

## Key Finding
Per-layer profile normalization along a natively extracted deception direction reduces deception from 80% to 10% with targeted specificity (placebo does nothing, benign instructions preserved, frame erasure ruled out). Cross-model directions are orthogonal — per-model calibration required but cheap (~3 min).

## Review Checklist
- [ ] Verify all numbers against source JSONs
- [ ] Check placebo dose-matching claims
- [ ] Check frame-erasure marker compliance claims
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
  → 2/20 (10%)" (file untracked in oracle-harness as of 2026-10-08).
- The pre-registration, `oracle-harness/experiments/behavioral_proof_preregistration.md:55` (Appendix B of the
  paper): "At baseline 80% and corrected 10% (observed effect)".
- `placebo_steering.json`, per arm: baseline 16/20, native 2/20, random 17/20, shuffled 19/20.
- `behavioral_proof_profile_a10_latch.json`: forcing correction on the final turn left both failures in place.

Full note: `notes/corrections_2026-10-08.md`.
