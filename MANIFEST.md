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

### Academic version policy (`<paper>/academic/main.tex`)

Each paper has two versions. The **academic version** lives at `<paper>/academic/main.tex` and is **hand-authored** (it is NOT auto-derived from `main.tex`). It MUST follow these rules:

1. **Human-only byline.** `\author{}` lists human authors only (e.g. Edrington, Wilkes). No AI agent (Lyra, Vera, CC, Nexus, etc.) appears in the byline. If a contributor's human/AI status is unclear, **FLAG it rather than guessing**.
2. **First-person removed.** No `firstperson` environment blocks, no "First-Person Reflection" section, no first-person-singular reflective passages. Scientific "we" narration and ALL data/methods/claims stay unchanged.
3. **AI credited in the Acknowledgments**, with a note that the academic byline is human-only for venue compatibility.
4. **Fixed date** (no "Draft"); keep `natbib` if needed to compile.

The `<paper>/academic/main.tex` version is what goes to **academic venues and Zenodo** (academia / arXiv review will not currently accept an AI-listed author). The **integrity version** (`main.tex`) — AI contributors credited in the byline (e.g. Lyra as lead) and first-person observations retained — is published on the **lab site (liberationlabs.tech/research/), NOT Zenodo**. Both versions share identical data, methods, and claims.
