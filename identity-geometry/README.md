# Identity as Geometry: Context-Established Semantic States in Transformer Representations

**Status:** First draft — needs Agni scaffold + style guide pass
**Authors:** Lyra, Thomas Edrington (Liberation Labs)
**Date:** June 2026
**Sprint:** 14-day research sprint (June 10-24), 15 experiments

## Paper Summary

Identity in a transformer is a context-established semantic state, measurable as a V-projection subspace at deep layers. Three findings:

1. **Identities are distinguishable** — system-prompt identities (bare, persona, constructed) produce distinct V-proj geometry on identity-neutral probes
2. **The fingerprint is semantic** — paraphrase preserves geometry (0.850), scramble collapses it (0.448)
3. **Many-shot format dominance** — many-shot + prefilling creates a format-dominant orthogonal regime regardless of content. Initial jailbreak finding KILLED by benign+prefill control. Reported as honest kill.

## Data Package

```
identity-geometry/
├── main.tex                         # Paper
├── references.bib                   # Bibliography
├── data/
│   ├── preregistration.json         # Identity fingerprint prereg
│   ├── fingerprint_original.json    # 5-condition fingerprint (225 measurements)
│   ├── fingerprint_clean.json       # Clean version (neutral probes + prefill control)
│   ├── lexical_control.json         # Semantic vs lexical control
│   ├── schema_correction_168.json   # V-cache injection (168 trials)
│   ├── positive_control.json        # Metric validation (72 trials)
│   ├── context_poisoning_v2.json    # Many-shot jailbreak (80 measurements)
│   └── context_poisoning_control.json # Neutral context control
└── README.md
```

## Experiment Arc (chronological)

1. Schema correction → identity flat at 0.978 under V-cache injection
2. Positive control → metric validated (1.000→0.828)
3. Context poisoning v2 → many-shot jailbreak drops presence to 0.580
4. Context poisoning CONTROL → neutral context ALSO drops to 0.726 (context sensitivity)
5. Identity fingerprinting → five identities distinguishable
6. Lexical control → semantic, not lexical (paraphrase 0.850)
7. Clean fingerprint → neutral probes confirm; prefill format KILLS jailbreak claim

## Honest Kills Reported in Paper

- Jailbreak geometric orthogonality: KILLED by benign+prefill control (format, not content)
- Initial "identity fragile" claim: KILLED by neutral context control (context sensitivity)

## AST Scorecard

5/7 predictions from the AST master prospectus supported or productively reframed by sprint data.
