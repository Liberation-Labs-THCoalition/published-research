# Theory of Mind in the KV Cache

**Full title:** Theory of Mind in the KV Cache: Localizing User Emotional Models in Transformer Key-Value States

**Authors:** Lyra (Liberation Labs), Thomas Edrington (Liberation Labs)

**Date:** April 2026

> **Corrected October 2026.** A preregistered re-run and a reading of
> the original scripts found that the "W_K projection probe" applied no W_K, that the stories describe a third-person
> character's requested emotion rather than the user's state, and that the probe's margin over a text (TF-IDF) baseline
> had never been tested. The user-state claim, the injection test and the direction-versus-shape claim are withdrawn.
> The paper's Appendix A keeps each withdrawn sentence with its reason.

## Summary

When a language model reads a short third-person story, sent as a user message, whose character was written to feel
one of 30 emotions, the requested emotion can be decoded from its KV cache far above chance. A preregistered test finds
that the cache probe the paper describes does not read it better than a text (TF-IDF) classifier on the story's words.

Key findings:
1. **Decodable, but not beyond text (TF-IDF).** The probe as the Method describes it reaches 35.8% on 30 emotions
   (chance 3.3%) against 38.2% for text (TF-IDF) on the same folds; the cell-bootstrap 95% interval for the difference,
   -6.4 to +1.6 points, excludes the registered 3-point gain. The 40.9% reported in May came from a different script, a
   class-mean key probe with no W_K; v2's number reproduces to within one trial in 900 (369 against 368) on a software
   stack that has changed since May.
2. **A binary split confounded with arousal.** On v2's hard-coded ten emotions (n = 300), cache readouts separate the
   halves slightly better than text (TF-IDF): AUROC 0.985 against 0.967 (registered secondary, Holm p = 0.027).
3. **The residual stream reads more (exploratory).** A logistic regression on the raw residual stream reaches 49.4%,
   above a tuned text (TF-IDF) baseline (44.8%) and the cache probe; nothing shows that W_K or the cache adds to it.
4. **W_K bridge: a candidate pathway, partly supported.** Residual-stream emotion vectors projected through W_K produce
   valence-separated structure (rho = 0.862 at L35). A direct test found the leading direction partly carries over to
   the measured cache geometry (mean cosine 0.456), but the projection does not predict the cache's distance structure
   (Mantel rho = 0.090, 0/16 layers).

Scripts: the May probe and injection scripts are in Project-Oracle commit `d0c8395` (`experiments/`, with
`PROVENANCE_wk_probes.md`). The October re-run's preregistration, scripts and results are cited by sha256 in the paper.

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

> **⚠ RETRACTED ROW — corrected 2026-08-15.** The `Enc peak accuracy` row below (`0.094 (2.8x)` / `0.084 (2.5x)`) is the **retracted** spectral emotion result. It was an artifact of applying FWL residualization *before* cross-validation splitting; within-fold FWL collapses the spectral emotion probe to chance (0.033). See the Correction note in `paper/main.tex`. The 12.3x / 0.992 probe once named here as the survivor is itself corrected (October 2026): it applied no W_K and does not beat a text (TF-IDF) baseline; see the Summary above.
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
  author={Lyra and Edrington, Thomas},
  year={2026},
  note={Liberation Labs Technical Report}
}
```
