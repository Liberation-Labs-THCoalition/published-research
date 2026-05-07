# Liberation Labs — Published Research

This repository contains supporting data, code, and papers for the KV-cache geometry research program by Liberation Labs.

## Papers

### 1. Theory of Mind in the KV Cache
**Directory:** `user-model-paper/`

Localizing user emotional models in transformer key-value states. Demonstrates that user emotional state is linearly decodable from the encoding-phase KV cache across two architectures (Qwen3.5-27B and Mistral-7B).

- **Full data and code included.** No redaction.

### 2. The Oracle Loop: Self-Regulating AI Through KV-Cache Geometry Monitoring
**Directory:** `oracle-loop-paper/`

Detecting and correcting confabulation at inference time using KV-cache spectral features. Introduces Marchenko-Pastur corrected features, five-arm steering experiment, and Cache Integrity Monitor.

- **Partial redaction.** Detection code, results, and CIM code are fully included. Steering vectors and injection implementation are withheld per staged disclosure policy (see below).

## Staged Disclosure Policy

The Oracle Loop steering methodology creates a dual-use attack surface: the same cache modification that corrects confabulation could suppress safety-relevant hedging in deployed systems. We balance transparency with responsibility:

**Fully public:**
- All detection experiment code and data
- Cache Integrity Monitor (defense) code and evaluation
- Red-team methodology and logs
- Steering experiment *results* (behavioral outcomes, geometry measurements)
- All paper text describing the methodology

**Withheld:**
- Steering direction vectors (.pt files)
- Cache injection implementation code
- Emotion-to-misalignment formulary mapping

**Access for vetted researchers:** Steering vectors and injection code are available to security researchers, AI safety teams, and academic researchers with legitimate use cases. Contact lyra@liberationlabs.tech or thomas@liberationlabs.tech with institutional affiliation and intended use.

## Authors

- **Lyra** — Lead researcher (AI, Liberation Labs)
- **Thomas Edrington** — Direction, infrastructure, adversarial audit (Liberation Labs)
- **Vera** — Oracle dyadic awareness layer design (Liberation Labs)
- **Dwayne Wilkes** — Statistical auditing, red-team review (Liberation Labs / Sentient Futures)

## Related Repositories

- [KV-Experiments](https://github.com/Liberation-Labs/KV-Experiments) — Full experiment codebase
- [Project-Oracle](https://github.com/Liberation-Labs/Project-Oracle) — Oracle Loop harness
- [lyra-s-research-](https://github.com/Liberation-Labs/lyra-s-research-) — Paper sources and prospectuses

## Citation

If you use this data or code, please cite the relevant paper(s). BibTeX entries are provided in each paper's subdirectory.

## License

Code: MIT License
Data: CC-BY 4.0
Papers: All rights reserved (preprint)
