# Spectral Shape Features for Confabulation Detection: Threshold-Free KV-Cache Analysis

**Status:** Draft — under review (Dwayne Wilkes red team pending)

## Authors
- Lyra (Liberation Labs / THCoalition) — Lead AI author
- Thomas Edrington (Liberation Labs / THCoalition) — Corresponding human author
- Dwayne Wilkes (Liberation Labs / THCoalition) — Red team

## Summary

Threshold-independent spectral shape features (stable rank, singular value kurtosis, participation ratio) substantially improve confabulation detection from KV-cache geometry. On Qwen2.5-7B-Instruct, shape features achieve AUROC 0.767 (95% CI [0.694, 0.863], p < 0.001) compared to 0.628 for Marchenko-Pastur features. A five-condition experiment reveals the signal tracks *retrieval engagement* — how broadly the model draws on stored knowledge during generation. Cross-architecture evaluation shows significant signal on Mistral-7B (0.643, p = 0.002) but not Llama-3.1-8B (0.520, p = 0.143).

## Directory Structure

```
main.tex                    — Full paper (LaTeX)
outline.md                  — Paper outline (Draft 2, post red team)
code/
  lyra_features.py          — Unified feature extraction module
data/
  Qwen2.5-7B-Instruct_cognitive_states.json    — Primary dataset (n=150)
  honesty_signal.json                          — 5-condition experiment (n=130)
  Llama-3.1-8B-Instruct_cognitive_states.json  — Cross-arch (n=150)
  Mistral-7B-Instruct-v0.3_cognitive_states.json — Cross-arch (n=150)
  refinement_battery.json                      — Refinement battery (Mistral, n=150)
  RESULTS_MANIFEST.json                        — Full experiment inventory
verification/
  compute_paper_stats.py     — Verification script (reproduces all numbers)
  compute_bootstrap_cis.py   — Bootstrap CI computation
  verification_report.md     — Overnight verification results
  morning_briefing.md        — Discrepancy analysis and corrections
```

## Verification

Every number in the paper traces to source data. To reproduce:

```bash
cd verification
python compute_paper_stats.py      # Point estimates, effect sizes, TOST
python compute_bootstrap_cis.py    # Bootstrap CIs, cross-arch p-values
```

## Red Team History

- **Round 1:** 3 lethal, 5 high findings. All addressed.
- **Round 2:** 0 lethal, 1 high (stale number — fixed), 2 medium. All addressed.
- **Dwayne review:** Pending.

## Companion Papers

- P2: Oracle Loop (detection + steering)
- P4: User Model (emotion geometry)
- P6: Oracle Formulary (dose-response profiles)
- P7: KV-Cloak Defense (obfuscation countermeasure)
