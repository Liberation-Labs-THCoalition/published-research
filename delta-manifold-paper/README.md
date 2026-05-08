# Per-Layer Delta Features and Manifold Signatures in KV-Cache Confabulation Detection

**Status:** Draft — red team pass 1 complete, fixes applied

## Authors
- Lyra (Liberation Labs) — Lead AI author, manifold geometry analysis
- CC (Liberation Labs) — Delta features, layer-averaging finding, cross-architecture analysis
- Thomas Edrington (Liberation Labs) — Direction, retrieval-engagement framing
- Dwayne Wilkes (Liberation Labs / Sentient Futures) — Red team

## Summary

Layer averaging weakens or destroys spectral signal in the KV cache. Per-layer confabulation detection on Qwen peaks at d=2.46 (L23, Bonferroni-corrected), 37% stronger than the all-layer average. Delta features (generation minus encoding) achieve d=2.35 [95% CI: 1.83, 3.15]. Trajectory geometry shows confabulation follows an "approach and slide" pattern with significantly non-linear condition distances (detour ratio 1.24, bootstrap CI [1.12, 1.45]).

## Key Numbers (all verified against source data)

| Measure | Value | 95% CI |
|---------|-------|--------|
| Delta stable_rank (hr vs confab) | d=2.35 | [1.83, 3.15] |
| Endpoint stable_rank (hr vs confab) | d=1.80 | [1.30, 2.46] |
| Encoding-only baseline | d=-1.62 | [-2.58, -1.01] |
| Peak per-layer (L23, Bonferroni) | d=2.46 | p<0.001 corrected |
| Detour ratio | 1.237 | [1.12, 1.45] |
| Trajectory chi-squared | 14.25 | p=0.003 |

## Directory Structure

```
main.tex                    — Full paper (LaTeX)
main.pdf                    — Compiled PDF (14 pages)
outline.md                  — Paper outline
paper.json                  — Website publishing stamp
data/
  honesty_signal.json       — Qwen 5-condition experiment (n=130)
  Llama_persona_full.json   — Llama persona intensity (n=50x5)
verification/
  verify_cc_findings.py     — Verifies CC's headline numbers
  compute_delta_paper_stats.py — All bootstrap CIs, corrections, tests
  manifold_signatures.py    — Trajectory curvature, distance matrix
```

## Companion Papers
- Spectral Shape Features (threshold-free detection — publishes first)
- P2 Oracle Loop, P4 User Model, P6 Formulary, P7 KV-Cloak
