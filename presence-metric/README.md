# Presence: Measuring Identity Preservation During Inference-Time Model Interventions

**Status:** Ready for human review (narrative rewrite complete, all experiments done)
**Authors:** Lyra, Thomas Edrington (Liberation Labs)
**Date:** June 2026
**Review gate:** Agni + style guide combined review — PASS with 6 minor fixes applied

## Paper Summary

We introduce presence, a geometric metric that measures whether inference-time interventions preserve a model's identity. Three findings:

1. **Identity preserved** (168 trials): presence = 0.978 +/- 0.002 across all arms, doses, emotions
2. **Metric validated** (72 trials): positive control drops presence 1.000 → 0.828 under direct identity injection; orthogonal control stays at 0.965
3. **Mechanism not simple**: emotion vectors share 22-25% energy with identity subspace (z=+56 above random) yet identity is unaffected at deep layers

## Data Package

```
presence-metric/
├── main.tex                                    # Full paper (LaTeX)
├── references.bib                              # Bibliography
├── data/
│   ├── schema_correction_168_trials.json       # Main experiment (168 trials)
│   ├── positive_control_72_trials.json         # Positive control (72 trials)
│   ├── orthogonality_results.json              # Emotion-identity alignment analysis
│   └── overnight_validation.json               # Deeper layers (L43/47/51) + probe diversity
└── README.md
```

## Key Claims to Verify

| Claim | Source file | Paper location |
|-------|------------|----------------|
| Presence 0.978 +/- 0.002 (168 trials) | schema_correction_168_trials.json | Table 1, Table 2 |
| Positive control 1.000 → 0.828 | positive_control_72_trials.json | Table 3 |
| Orthogonal control stays > 0.96 | positive_control_72_trials.json | Table 3 |
| Emotion vectors 22-25% aligned (z=+56) | orthogonality_results.json | Finding 3 |
| Deep layers L43/47/51 flat at 0.999+ | overnight_validation.json | Section 5.1 |
| Probe diversity mean 0.901 (sd=0.033) | overnight_validation.json | Section 5.1 |
| Spearman dose vs presence: rho=-0.119, p=0.123 | schema_correction_168_trials.json | Table 1 caption |

## Notes for Reviewer

- The paper originally claimed "depth separation" as the mechanism. This was killed by the Arm C data (all-layer injection, presence still flat) and the orthogonality data (22% alignment, not orthogonal). The rewrite frames the finding honestly as "identity is preserved but the mechanism is an open question."
- Finding 4 (directional efficacy) is from a 30-pair pilot with 33% parse rate on Prometheus 2 (7B). Marked as preliminary.
- A layer-by-layer presence map experiment is designed and Agni-gated but not yet run (waiting for compute).
