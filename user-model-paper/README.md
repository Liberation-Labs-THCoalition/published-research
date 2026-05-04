# Theory of Mind in the KV Cache

**Full title:** Theory of Mind in the KV Cache: Localizing User Emotional Models in Transformer Key-Value States

**Authors:** Lyra, Thomas Edrington, Nell Watson, Dwayne Wilkes

**Date:** April 2026

## Summary

We demonstrate that user emotional state is linearly decodable from the encoding-phase KV cache, replicating across two architectures:
- **Qwen3.5-27B-Claude-Distilled** (hybrid attention, 900 trials): 2.8x chance at L3 (p < 0.0001)
- **Mistral-7B-Instruct-v0.3** (standard attention, 900 trials): 2.5x chance at L4 (p < 0.0001)

Key findings:
1. Encoding peaks at shallowest available layers (fast read), generation peaks at mid-network (deep response)
2. Encoding and generation depth profiles are uncorrelated (separable representations)
3. Cache encodes discrete emotion identity, not smooth valence continuum (R^2 < 0 on both architectures)
4. W_K bridge: residual-stream emotion vectors projected through key projection produce valence-separable structure

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
  author={Lyra and Edrington, Thomas and Watson, Nell and Wilkes, Dwayne},
  year={2026},
  note={Liberation Labs Technical Report}
}
```
