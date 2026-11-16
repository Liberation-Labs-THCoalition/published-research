# Presence: Measuring Identity Preservation During Inference-Time Model Interventions

**Status:** Ready for human review (narrative rewrite complete; the layer-by-layer presence map in the last note below is designed but not yet run)
**Authors:** Lyra, Thomas Edrington (Liberation Labs)
**Date:** June 2026
**Review gate:** the only style gate on record (`tools/agni/style/presence-metric.agni_style.json`, run 2026-09-04) returned REJECTED; the paper has not been re-gated since. An earlier version of this line read "PASS with 6 minor fixes applied".

## Paper Summary

We introduce presence, a geometric metric that measures whether inference-time interventions preserve a model's identity. Three findings:

1. **Metric responds at the measurement layer** (24 independent measurements, 72 records): positive control drops presence 1.000 → 0.828 under direct identity injection at L35; orthogonal control stays at 0.965 or above
2. **No change detected under input-layer V-cache injection** (720 trials, doses α ≤ 0.50, where the positive control itself only reaches 0.964, so insensitivity there is not excluded): presence = 0.845 +/- 0.012 across all arms, doses, emotions. The 720-trial records are not in `data/`. An earlier 168-trial, 2-prompt run (in `data/`) gave 0.978 +/- 0.002
3. **Mechanism open**: raw emotion vectors share 22-25% energy with identity subspace at L3 (z=+56 to +62 above random; 10-15% at L7) yet presence at deep layers is unaffected

## Data Package

```
presence-metric/
├── main.tex                                    # Full paper (LaTeX)
├── references.bib                              # Bibliography
├── data/
│   ├── schema_correction_168_trials.json       # Main experiment (168 trials)
│   ├── positive_control_72_trials.json         # Positive control (72 trials)
│   ├── orthogonality_results.json              # Emotion-identity alignment analysis
│   ├── overnight_validation.json               # Deeper layers (L43/47/51) + probe diversity
│   └── presence_smoke_results.json             # Layer-matched smoke test (inject L3/L7, measure L35), 2026-06-30
└── README.md
```

## Key Claims to Verify

| Claim | Source file | Paper location |
|-------|------------|----------------|
| Presence 0.845 +/- 0.012 (720 trials) | not in `data/` | Table 1, Table 2 |
| Noise floor 0.978 +/- 0.003, arm effect H=10.29, p=0.016 (168 trials) | schema_correction_168_trials.json | Replication note, Table 2 caption, H2 |
| Positive control 1.000 → 0.828 | positive_control_72_trials.json | Table 3 |
| Orthogonal control stays > 0.96 | positive_control_72_trials.json | Table 3 |
| Emotion vectors 22-25% aligned (z=+56) | orthogonality_results.json | Finding 3 |
| Deep layers L43/47/51 flat at 0.999+ | overnight_validation.json | Section 5.1 |
| Probe diversity mean 0.901 (sd=0.033) | overnight_validation.json | Section 5.1 |
| Spearman dose vs presence: rho=-0.119, p=0.123 (168 trials; the paper prints the 720-trial rho) | schema_correction_168_trials.json | not printed |

## Notes for Reviewer

- The paper originally claimed "depth separation" as the mechanism. The Arm C data (all-layer injection, presence still flat) and the orthogonality data (22-25% alignment at L3) led the rewrite to state the mechanism as an open question. Arm C's doses stop at α = 0.50, where the L35 positive control itself moves presence only to 0.964, so the Arm C result does not rule depth separation out either.
- The 168-trial file's configuration records temperature 0.6, and its α = 0 records (no injection) carry distinct responses for the same prompt, so that run sampled; it was not greedy. Presence in those α = 0 records still takes one value per prompt, so the measurement does repeat exactly; the 2 prompts, not the responses, set the number of independent observations.
- Finding 4 (directional efficacy) is from a 30-pair pilot with 33% parse rate on Prometheus 2 (7B). Marked as preliminary.
- A layer-by-layer presence map experiment is designed and Agni-gated but not yet run (waiting for compute).
