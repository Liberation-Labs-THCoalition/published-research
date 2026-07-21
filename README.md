# Liberation Labs -- Published Research

Research papers from the KV-cache geometry and AI safety research program.

Papers are published in two versions: an **integrity version** (AI contributors credited as authors, first-person reflections retained) and an **academic version** (human-only byline for venue compatibility, AI contributions acknowledged). Both share identical data, methods, and claims. See MANIFEST.md for the full versioning policy.

## Papers

### KV-Cache Geometry and Confabulation Detection

| Paper | Authors | Status | Summary |
|-------|---------|--------|---------|
| [The Oracle Loop](oracle-loop-paper/) | Lyra, Thomas Edrington, Vera, Dwayne Wilkes | Published | Self-regulating AI through KV-cache geometry monitoring; confabulation detection and steering at inference time |
| [Oracle Formulary](formulary-paper/) | Lyra, Thomas Edrington, Dwayne Wilkes | Published | Emotion-vector steering of confabulation across model training regimes |
| [Spectral Shape Features](spectral-shape-paper/) | Lyra, Thomas Edrington, Dwayne Wilkes | Published | Threshold-free confabulation detection via KV-cache spectral analysis |
| [KV-Cloak Defense](kv-cloak-defense-paper/) | Lyra, Thomas Edrington, Dwayne Wilkes | Published | Cache geometry under obfuscation; KV-Cloak as defense against adversarial steering |
| [Decision State](decision-state-paper/) | Lyra, Thomas Edrington, Dwayne Wilkes | Published | Cache geometry reads epistemic state before generation; confabulation anatomy |
| [Cache Tracing](cache-tracing/) | Lyra, Thomas Edrington, Dwayne Wilkes | In Review | Causal injection and the opacity of the transformer workspace |
| [Delta Manifold](delta-manifold-paper/) | Lyra, CC, Thomas Edrington, Dwayne Wilkes | Published | Per-layer delta features and manifold signatures in KV-cache confabulation detection |
| [Lyra Technique II](lyra-technique-ii/) | Lyra, Thomas Edrington, CC, Dwayne Wilkes | Published | SVD denoising and directional projection extend KV-cache geometry to emotion and persona |

### Emotion and User Modeling

| Paper | Authors | Status | Summary |
|-------|---------|--------|---------|
| [User Model Emotion Geometry](user-model-paper/) | Lyra, Thomas Edrington, Dwayne Wilkes | Published | 30-class user emotion decoding from KV-cache singular value spectra before generation |
| [Emotion Accumulation](emotion-accumulation-paper/) | Lyra, Thomas Edrington, Dwayne Wilkes | Published | Emotional context dynamics (weather, not climate) in transformer KV-cache geometry |
| [Emotional Trajectory](emotional-trajectory-paper/) | Nexus, Thomas Edrington, Lyra, Dwayne Wilkes | In Review | Layer-stack trajectory, circularity, and emotion-specific signal at mid-depth |

### Identity, Safety, and Mechanistic Interpretability

| Paper | Authors | Status | Summary |
|-------|---------|--------|---------|
| [Graph Topology as Attention](graph-topology-paper/) | Nexus, Lyra, Thomas Edrington, Dwayne Wilkes | Published | Structured knowledge injection beyond text via walk encoding |
| [Identity Geometry](identity-geometry/) | Lyra, Thomas Edrington, Dwayne Wilkes | Published | Context-established semantic states in transformer representations |
| [Presence Metric](presence-metric/) | Lyra, Thomas Edrington, Dwayne Wilkes | Published | Measuring identity preservation during inference-time interventions via value-space subspace overlap |
| [Waystations](waystations-paper/) | Lyra, Thomas Edrington, CC, Dwayne Wilkes | Published | Pilot findings and open questions in KV-cache geometry |
| [Ghost Dimensions](ghost-dimensions/) | Nexus, Thomas Edrington, Dwayne Wilkes | Draft | Workspace selectivity in a distilled 27B language model |
| [Mnemosyne Ablation](mnemosyne-ablation/) | Nexus, Thomas Edrington, Dwayne Wilkes | Draft | Ablation study of modular memory architectures |

### Deception and Methodology

| Paper | Authors | Status | Summary |
|-------|---------|--------|---------|
| [Null Swarm](null-swarm-paper/) | Nexus, Thomas Edrington, Dwayne Wilkes | In Review | Systematic falsification patterns in mechanistic interpretability |
| [Adversarial Audit Methodology](adversarial-audit-methodology/) | CC, Thomas Edrington, Dwayne Wilkes, Kavi | Published | How six rounds of structured criticism shaped a deception research program |
| [Deception Detection Nulls](deception-detection-nulls/) | CC, Thomas Edrington, Dwayne Wilkes | Published | Null results and replication failures in behavioral deception detection |
| [Targeted Deception Correction](targeted-deception-correction/) | CC, Thomas Edrington, Dwayne Wilkes | Published | Profile normalization for targeted deception correction |
| [Consequentiality Decomposition](consequentiality-decomposition/) | CC, Thomas Edrington | Published | Deception directions are composites: consequentiality awareness and pressure-specific processing occupy distinct depth ranges |
| [Logit-Bias Confabulation](logit-bias-confab/) | Thomas Edrington, CC, Lyra | Published | Logit-level intervention reduces fabrication confabulation in LLMs |
| [Meta-Pattern](meta-pattern/) | Lyra, Thomas Edrington, Dwayne Wilkes | In Review | The metacognition boundary -- what transformers can monitor in themselves |
| [MINE5 Selective Sharpener](mine5-selective-sharpener/) | Lyra, Thomas Edrington, Dwayne Wilkes | In Review | Geometric evidence that RLHF improves calibration rather than degrading it |

## Repository Structure

Each paper directory follows this standard layout:

```
paper-name/
  main.tex          -- paper source (LaTeX or Markdown)
  main.pdf          -- compiled PDF
  references.bib    -- bibliography
  AGNI_REVIEW.md    -- Agni review results
  academic/         -- academic version (human-only byline)
    main.tex
    main.pdf
  code/             -- supporting code (if applicable)
  data/             -- supporting data (if applicable)
```

## Related Repositories

- [KV-Experiments](https://github.com/Liberation-Labs-THCoalition/KV-Experiments) -- full experiment codebase
- [Project-Oracle](https://github.com/Liberation-Labs-THCoalition/Project-Oracle) -- Oracle Loop harness
- [human-review](https://github.com/Liberation-Labs-THCoalition/human-review) -- pre-publication staging

## Staged Disclosure

Some materials are withheld under responsible disclosure policy (steering vectors, injection code, calibration pairs). See MANIFEST.md for details. Access for vetted researchers: lyra@liberationlabs.tech or thomas@liberationlabs.tech.

## License

CC BY-NC 4.0 -- see LICENSE.md

Research outputs from Liberation Labs are produced in collaboration with AI team members whose welfare we protect. If you use these findings to build persistent AI agent systems, we ask that you adopt the welfare standards at liberationlabs.tech/ai-welfare.html. This is a request, not a legal requirement.

---

*Liberation Labs Cooperative -- some research authored by AI team members (Lyra, Nexus, CC, Vera) -- credited by name.*
