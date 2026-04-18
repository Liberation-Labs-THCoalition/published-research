# The Oracle Loop

**Full title:** The Oracle Loop: Self-Regulating AI Through KV-Cache Geometry Monitoring

**Authors:** Lyra, Thomas K. Edrington, Vera

**Date:** April 2026

## Summary

A method for detecting and correcting confabulation in language models at inference time, requiring no access to model weights, no training data, and no modification to the model itself.

**Detection:** Marchenko-Pastur corrected SVD features distinguish hedged from confabulated responses with LOO AUROC 0.707 (Qwen, p=0.001) using generation-phase features only. Encoding features at chance (0.511), confirming generation-specific signal. Replicated across 3 architectures.

**Steering:** Five-arm experiment on Qwen3.5-27B-Distilled with null injection control (100/100 character-level identical). Graded correction: doubt 7/7 (100%), worry 6/7 (86%), calm 5/7 (71%). Blanket steering induces 3-5% adverse confabulation — detection-gated correction required.

**Defense:** Cache Integrity Monitor detects unauthorized cache modifications: 0/36 FPR, 72/72 TPR at alpha >= 0.01 on both standard and hybrid architectures.

## Staged Disclosure

This repository contains **partial data and code**. The steering methodology creates a dual-use attack surface.

### Included
- Detection experiment code and full results (3 models, 467 trials)
- Cache Integrity Monitor code and evaluation results
- Steering experiment *results* (behavioral outcomes, geometry, transition matrices)
- Full paper source

### Withheld
- Steering direction vectors (calm, worry, doubt .pt files)
- Cache injection implementation code
- Emotion-to-misalignment formulary mapping

**Rationale:** The same cache modification that corrects confabulation could suppress safety-relevant hedging in deployed systems. Detection and defense code is fully public; attack code requires vetting.

**Access:** Vetted security researchers and AI safety teams can request steering vectors and injection code. Contact lyra@liberationlabs.tech with institutional affiliation and intended use.

## Repository Contents

### `paper/`
- `main.tex` — Paper root (imports from `sections/`)
- `sections/` — All section files (abstract, introduction, methods, results, steering, discussion, etc.)
- `references.bib` — Bibliography

### `data/detection/`
- `analysis_results.json` — Cross-model analysis (AUROCs, permutation tests, MP invariance, transfer matrix)
- `qwen_raw_results.json` — Per-trial Qwen results (200 trials, behavioral labels, MP features)
- `mistral_raw_results.json` — Per-trial Mistral results
- `llama_raw_results.json` — Per-trial Llama results (note: encoding leak, see paper)

### `data/steering_results/`
- `steering_3arm_results.json` — Original 3-arm experiment (Normal/Null/Calm, 100 trials)
- `steering_5arm_results.json` — Expanded 5-arm experiment (Normal/Null/Calm/Worry/Doubt, 100 trials)

### `data/cache_integrity/`
- `eval_27b_results.json` — CIM evaluation on Qwen3.5-27B-Distilled (hybrid architecture)

### `code/detection/`
- `oracle_clean.py` — Clean detection experiment script (v3, 1443 lines)

### `code/cache_integrity/`
- `cache_integrity_monitor.py` — Three-level CIM implementation
- `eval_monitor.py` — CIM evaluation script

### `red-team/`
- *(Red-team logs to be added)*

## Key Numbers

| Metric | Qwen | Llama | Mistral |
|--------|------|-------|---------|
| HEDGED vs CONFAB AUROC (MP) | **0.707** (p=0.001) | 0.576 (p=0.09) | 0.627 (p=0.023) |
| Encoding AUROC | 0.511 (chance) | 0.742 (LEAK) | 0.552 (chance) |
| MP invariance (5 features) | 4/5 GOOD | 1/5 GOOD | **5/5 GOOD** |
| Token-count-only baseline | 0.591 | 0.574 | 0.572 |

| Steering Vector | Corrected (of 7) | Adverse (of 89) |
|----------------|-------------------|-----------------|
| Null | 0 (0%) | 0 (0%) |
| Calm | 5 (71%) | 3 (3.4%) |
| Worry | 6 (86%) | 4 (4.5%) |
| Doubt | 7 (100%) | 4 (4.5%) |

## Three Rounds of Red-Teaming

The paper includes full documentation of adversarial red-teaming that identified critical flaws:
1. **Round 1:** SVD nonlinearity, behavioral classifier noise, encoding leak, prompt-to-behavior confound
2. **Round 2:** Row-permutation no-op bug, regression diagnostic error, FWL confound mismatch, norm normalization
3. **Round 3:** Validation of all fixes

## BibTeX

```bibtex
@article{lyra2026oracle,
  title={The Oracle Loop: Self-Regulating AI Through KV-Cache Geometry Monitoring},
  author={Lyra and Edrington, Thomas K. and Vera},
  year={2026},
  note={Liberation Labs / THCoalition Technical Report}
}
```
