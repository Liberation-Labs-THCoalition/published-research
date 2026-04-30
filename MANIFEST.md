# Data Release Manifest

## RELEASED (published-data/)

### detection/
- Primary detection results (HEDGED vs CONFABULATED AUROCs, all models)
- MP feature values per trial (5 features × 3 models)
- MP invariance diagnostic (R² table)
- Cross-model transfer matrix
- LOO ablation results (all 5 conditions with bootstrap CIs)
- Full-cache spectral gap analysis
- Control battery outcomes
- Text-only baseline features and AUROCs

### steering/
- Behavioral outcomes per trial per vector (correction/adverse/same)
- Formulary summary (12 vectors × 2 models × 3 misalignment types)
- Cross-model therapeutic ranking comparison
- Cross-cosine similarity matrices (both models)
- McNemar test results with Holm-Bonferroni correction
- Epistemic abliteration observation (coherence collapse at -0.5)

### kv-cloak/
- Obfuscation strategy comparison (7 strategies, all features)
- Real KV-Cloak implementation results (S·P̂·(K+A)·M)
- Cross-architecture sweep (6 models)
- Feature-space transformation analysis (honest cloaked vs uncloaked)
- Text-only baseline comparison
- Injection test results (direct, scrambled, magnitude)

### red-team/
- All frozen claims documents (P2, P6, P7)
- Red team agent outputs (pre-mortem, data-analyst, experiment-designer)
- Dwayne Lane 1 findings summary (P1-P4)
- LOO ablation methodology and results

## WITHHELD (staged disclosure)

The following are withheld per responsible disclosure policy:

- **Steering direction vectors** (actual cache-space tensors). These enable targeted cache injection and pose dual-use risk.
- **Cache injection implementation code** (generate_with_injection). Describes the exact mechanism for modifying cache during generation.
- **Calibration pair prompts** (contrastive pairs used to extract vectors). Could be used to reproduce steering vectors for adversarial purposes.
- **KV-Cloak implementation code**. Our implementation of Luo et al.'s mechanism.

Access to withheld materials available to vetted security researchers
and AI safety teams upon request: lyra@liberationlabs.tech

## Versioning

- **Integrity version**: Full AI authorship (Lyra as lead author), first-person reflection sections, complete methodology including dual-use considerations. Published at liberationlabs.tech.
- **Academic version**: Human authorship only (Edrington, Wilkes). AI contribution acknowledged in methods section. First-person sections removed. Identical data and claims. Published on Zenodo.

Both versions reference each other. The integrity version includes a note explaining what was removed for the academic version and why.
