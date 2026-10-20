# Theory of Mind in the KV Cache

**Full title:** Theory of Mind in the KV Cache: Localizing User Emotional Models in Transformer Key-Value States

**Authors:** Lyra (Liberation Labs), Thomas Edrington (Liberation Labs)

**Date:** May 2026; corrected October 2026

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
   -6.4 to +1.6 points, lies below the registered 3-point gain. The 40.9% reported in May came from a different script, a
   class-mean key probe with no W_K; v2's number reproduces to within one trial in 900 (369 against 368) on a software
   stack that has changed since May.
2. **A binary split confounded with arousal.** On v2's hard-coded ten emotions (n = 300), cache readouts separate the
   halves slightly better than untuned text (TF-IDF): AUROC 0.985 against 0.967 (registered secondary, Holm p = 0.027).
3. **A larger residual-stream readout reads more (exploratory).** A 20,480-feature logistic regression on the raw residual
   stream reaches 49.4%, above the 30-feature cache probe; whether it beats a fully tuned text (TF-IDF) baseline (at least
   44.8%) is open, and nothing shows that W_K or the cache adds to it.
4. **W_K bridge: a candidate pathway, not established.** Residual-stream emotion vectors projected through W_K separate
   the emotions' assigned valence (|rho| up to 0.872 at L43). There is no unprojected or random-matrix control, the code
   applies W_K to each layer's output rather than the normalized input W_K reads, and the one direct test (Mantel rho
   0.090 on an earlier 300-trial run) is not released and could not be re-verified.

Scripts: the May probe and injection scripts are in Project-Oracle commit `d0c8395` (`experiments/`, with
`PROVENANCE_wk_probes.md`). The October re-run's preregistration, sealed prediction, scripts, results and logs are in `correction_v2/`
(byte-exact copies; `SHA256SUMS` lists every file's hash).

## Repository Contents

### `paper/`
- `main.tex` — Full paper source

### `data/`
- `qwen_probe_analysis_results.json` — Qwen 30-class classification + valence regression results (all layers)
- `mistral_probe_analysis_results.json` — Mistral equivalent
- `qwen_emotion_bridge_summary.json` — Qwen W_K bridge PCA, valence/arousal correlations per layer
- `mistral_emotion_bridge_summary.json` — Mistral equivalent
- `baseline_residual_probe_results.json` — Residual-stream linear probe comparison (894 trials)

### `correction_v2/`
- The October 2026 correction run: preregistration (markdown and JSON), the sealed prediction, the encode, analysis
  and dry-run scripts, the results with the encode manifest and logs (`results_20261019/`), the results write-up,
  the R1 investigation and the design notes. `.gitattributes` keeps them byte-exact so the registered hashes hold.

### `code/`
- `emotion_geometry_bridge.py` — Main experiment script (900 trials, 30 emotions x 10 topics x 3 stories)
- `mp_probe_recompute.py` — MP feature recomputation (described in May as verifying the spectral null; that null is withdrawn, paper §4.4)
- `dump_bridge_full.py` — Bridge PCA data extraction for verification
- `dump_bridge_pc1.py` — F-ratio recomputation for verification

## Key Numbers (Verified Against Source JSON)

> **⚠ RETRACTED ROW — corrected 2026-08-15.** The `Enc peak accuracy` row below (`0.094 (2.8x)` / `0.084 (2.5x)`) is the **retracted** spectral emotion result. The May version attributed it to applying FWL residualization *before* cross-validation splitting; the corrected paper says that mechanism cannot be re-verified (April script and per-trial data unavailable), and the within-fold table that replaced it is withdrawn too. See the Correction note in `paper/main.tex`. The 12.3x / 0.992 probe once named here as the survivor is itself corrected (October 2026): it applied no W_K and does not beat a text (TF-IDF) baseline; see the Summary above.
>
> The row is left in place rather than deleted so the retraction is legible. **Do not cite it.** This README carried it under a "Verified" heading for 86 days after the paper itself published the retraction.

| Metric | Qwen | Mistral |
|--------|------|---------|
| Enc peak layer (withdrawn with the retracted accuracies; paper §4.6) | L3 (depth 0.05) | L4 (depth 0.13) |
| Enc peak accuracy | 0.094 (2.8x) | 0.084 (2.5x) |
| Gen peak layer (withdrawn, as above) | L23 (depth 0.37) | L11 (depth 0.35) |
| Enc/gen profile rho (withdrawn, as above) | 0.433 (NS) | 0.002 (NS) |
| GKF(emotion) valence R^2 | -0.470 | -0.463 |
| Bridge PC1-valence rho (best) | 0.872 (L43) | 0.932 (L17) |
| Permutation null mean | 0.022 +/- 0.005 | — |

## Verification

Not every number in the paper traces to a file in `data/`. The April-run numbers (bridge, regression, hypothesis outcomes) do; the withdrawn per-layer spectral table did not. The May probe numbers trace to the Project-Oracle scripts at `d0c8395`, and the October re-run's numbers to `correction_v2/results_20261019/um_v2_results.json`. Prior LLM sessions fabricated table values and emotion names; that content was identified and removed.

## Hardware

The May probe scripts and the October re-run ran on Apple MPS (Starship; `PROVENANCE_wk_probes.md` and the re-run's manifest). This line used to say that all experiments ran on 3x NVIDIA RTX 3090, which is true at most of the April run.

## BibTeX

```bibtex
@article{lyra2026usermodel,
  title={Theory of Mind in the KV Cache: Localizing User Emotional Models in Transformer Key-Value States},
  author={Lyra and Edrington, Thomas},
  year={2026},
  note={Liberation Labs Technical Report}
}
```
