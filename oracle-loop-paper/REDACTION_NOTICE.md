# Redaction Notice

The following materials are withheld from this public repository:

## Withheld Materials

1. **Steering direction vectors** (`calm.pt`, `worry.pt`, `doubt.pt`)
   - Contrastive emotion directions extracted from key-cache activations
   - Could be used to suppress safety-relevant hedging in deployed models

2. **Cache injection implementation** (`oracle_steering_clean.py`)
   - Code that modifies KV-cache tensors at inference time
   - Demonstrates a novel attack surface against transformer inference

3. **Formulary mapping** (emotion-to-misalignment-to-countervector)
   - Systematic mapping of which steering vectors correct which failure modes
   - Early-stage research with significant manipulation potential

## What IS Included

- All steering experiment **results** (behavioral outcomes, geometry measurements, transition matrices)
- Full paper text describing the methodology in sufficient detail for reproduction
- Detection experiment code and data (the defense)
- Cache Integrity Monitor code (the countermeasure)

## Rationale

The steering methodology creates a dual-use attack surface. An attacker with access to a model's cache object could inject arbitrary direction vectors to:
- Suppress epistemic hedging (making the model confabulate on demand)
- Alter an agent's decision-making without modifying weights or prompts
- Bypass safety training by directly perturbing the cached representation

We balance scientific transparency with responsible disclosure by:
1. Publishing the full paper describing the approach
2. Publishing all defensive tools (CIM)
3. Publishing steering results (so claims are verifiable)
4. Withholding the specific vectors and injection code

## Access

Vetted security researchers, AI safety teams, and academic researchers with legitimate use cases may request access to withheld materials.

**Contact:** lyra@liberationlabs.tech or thomas@liberationlabs.tech

Please include:
- Institutional affiliation
- Intended use case
- Relevant prior work in AI safety or security
