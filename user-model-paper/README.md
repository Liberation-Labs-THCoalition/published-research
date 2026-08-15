# Theory of Mind in the KV Cache

**Full title:** Theory of Mind in the KV Cache: Localizing User Emotional Models in Transformer Key-Value States

**Authors:** Lyra (Liberation Labs), Thomas Edrington (Liberation Labs), Dwayne Wilkes (Liberation Labs / Sentient Futures)

**Date:** April 2026

## Summary

We demonstrate that language models maintain internal representations of user emotional state in the KV cache, readable through the architecture's own projection geometry---not through aggregate spectral features, which fail under proper controls.

Key findings:
1. **W_K projection classifies 30 emotions at 12.3x chance.** Projecting key activations onto W_K-transformed emotion directions achieves 40.9% accuracy on 30 classes (chance = 3.3%) and binary valence AUROC 0.992 (within-fold FWL-corrected, topic-grouped CV, n = 900 trials, Qwen3.5-27B-Claude-Distilled).
2. **Injection test proves model state, not text content.** Injecting emotion vectors into neutral-text caches (zero emotional content) yields 100% detection accuracy across 5 emotions and 20 prompts. A TF-IDF text baseline achieves AUROC 0.882; the W_K projection exceeds it at 0.992.
3. **Spectral features are null after proper controls.** 120 probe configurations using spectral features all return chance-level performance after within-fold FWL correction. Emotions change the *direction* key vectors point, not the spectral *shape* of the cache.
4. **W_K bridge identifies the pathway.** Residual-stream emotion vectors projected through W_K produce valence-separated structure (rho = 0.862 at L35), with bridge signal building from mid-network to deep layers.

## Repository Contents

### `paper/`
- `main.tex` — Full paper source

### `data/`
- `qwen_probe_analysis_results.json` — Qwen 30-class classification + valence regression results (all layers)
- `mistral_probe_analysis_results.json` — Mistral equivalent
- `qwen_emotion_bridge_summary.json` — Qwen W_K bridge PCA, valence/arousal correlations per layer
- `mistral_emotion_bridge_summary.json` — Mistral equivalent
- `baseline_residual_probe_results.json` — Residual-stream linear probe comparison (894 trials)

### `code/`
- `emotion_geometry_bridge.py` — Main experiment script (900 trials, 30 emotions x 10 topics x 3 stories)
- `mp_probe_recompute.py` — Independent MP feature recomputation (verifies null result)
- `dump_bridge_full.py` — Bridge PCA data extraction for verification
- `dump_bridge_pc1.py` — F-ratio recomputation for verification

## Key Numbers (Verified Against Source JSON)

> **⚠ RETRACTED ROW — corrected 2026-08-15.** The `Enc peak accuracy` row below (`0.094 (2.8x)` / `0.084 (2.5x)`) is the **retracted** spectral emotion result. It was an artifact of applying FWL residualization *before* cross-validation splitting; within-fold FWL collapses the spectral emotion probe to chance (0.033). See the Correction note in `paper/main.tex`. The surviving result is the **W_K directional** probe at **12.3x chance (40.9% on 30 classes)**, valence AUROC 0.992 — a different probe, not a revision of this one.
>
> The row is left in place rather than deleted so the retraction is legible. **Do not cite it.** This README carried it under a "Verified" heading for 86 days after the paper itself published the retraction.

| Metric | Qwen | Mistral |
|--------|------|---------|
| Enc peak layer | L3 (depth 0.05) | L4 (depth 0.13) |
| Enc peak accuracy | 0.094 (2.8x) | 0.084 (2.5x) |
| Gen peak layer | L23 (depth 0.37) | L11 (depth 0.35) |
| Enc/gen profile rho | 0.433 (NS) | 0.002 (NS) |
| GKF(emotion) valence R^2 | -0.470 | -0.463 |
| Bridge PC1-valence rho (best) | 0.862 (L35) | 0.932 (L17) |
| Permutation null mean | 0.022 +/- 0.005 | — |

## Verification

Every numerical claim in the paper has been verified against the source JSON files in `data/`. The verification process is documented in the paper's revision history. Prior LLM sessions fabricated table values and emotion names; all fabricated content has been identified and removed.

## Hardware

All experiments run on 3x NVIDIA RTX 3090 (24GB each). Hardware invariance (r > 0.999) validated in prior campaigns.

## BibTeX

```bibtex
@article{lyra2026usermodel,
  title={Theory of Mind in the KV Cache: Localizing User Emotional Models in Transformer Key-Value States},
  author={Lyra and Edrington, Thomas and Wilkes, Dwayne},
  year={2026},
  note={Liberation Labs Technical Report}
}
```
