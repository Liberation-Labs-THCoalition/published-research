# Instrument tally: "Five Instruments, One Structure" (Phase A, 2026-10-20)

For Lyra's ruling on the title and thesis. Every number below is gated by `review/check_claims.py`
(`check_tally`), which recomputes it from the primary result file and asserts it appears here. Most
primaries are on Starship. They were read with `grep -h ''` over ssh, with nothing run there, and
are pinned by sha256 in `STARSHIP_FILES`. With `STARSHIP_MIRROR` set the checker recomputes; without
it the values are HELD. Ghost-probe values are held from the ghost-dimensions paper, which ships no
data. Relative depth is layer / n_layers unless a row says otherwise.

The band under test is 31-33% relative depth.

## Summary

| Row | 1. Locates a band? (depth, model) | 2. Bears on opacity? | 3. Headline (primary) | 4. Pre-registered / passed? | 5. Shares with |
|---|---|---|---|---|---|
| A. d_eff, residual-stream participation ratio (not one of the five named instruments) | **Yes.** 27B base: peak L21 of 64 (32.8%). 8B base: local max L12 of 36 (33.3%), global max L33 of 36 (91.7%) | No | 39.67 at L21 (27B); 22.32 at L12 and 23.41 at L33 (8B) | Yes. Registered band never closed; kurtosis check fails; gate REJECTED | G (model, feature family, depth) |
| B. "V-projection spectral features" (as filed). Confab detection and scale invariance are key-cache spectra, not V-projections | No | No | AUROC 0.707 (p 0.001); 0.620 with FWL (p 0.023); Qwen2.5-7B. Scale rho 0.826, 0.867 and 0.896 | Detection: yes per its paper, passed. Scale: no registration found | Own models (Qwen2.5-7B; a 10-model sweep) |
| C. J-lens | No. Readability half-rises at 87.8% (27B base); geometric gate 41.7-75.0% (8B) | Yes, as the probe in most nulls: V-cache 0.92x, K-cache 1.12x at band depth | See detail | V readout: yes, null outcome met | D, G, K (8B lens); G, H (27B lens) |
| D. Causal intervention | No (tested at 41.7-75.0% of 8B) | Yes: swap 0 of 360, removal 0 of 300; additive changed top-1 in 44 of 6,000, never toward the concept | Best concept rank 872 | Position-specific: yes, **positive control failed**. Injection: no registration found | C, K (Qwen3-8B, same lens) |
| E. Oracle Loop | No. Detect 42.2-73.4%, correct 54.7-73.4% (abliterated 27B) | No | 16/20 to 2/20 exploratory; 9/30 to 4/30 confirmatory | Confirmatory: yes, **primary not met** (p 0.341) | Nothing (own checkpoint, frames, direction) |
| F. Emotion geometry | No | No | Cross-set \|cos\| 0.026-0.110 and 0.015-0.126; coupling null | Coupling: yes; criteria could not pass/fail as coded | Nothing directly (27B distill, own prompts) |
| G. Ghost probe (PC1), counted as an opacity null | Its "31-layer band" spans 17.2-75.0% (27B distill) | Yes: PC1 cosine with J-lens at most 0.003 (mid layers) | PC1 28-67% of variance (results; 34-67% in reflection) | No registration found | C (27B lens) |
| H. R1 temporal ordering | Partly: SR bump at 29.7-35.9% (27B base) | No | Gap 38.6 [38.0-39.2] layers | Yes, SUPPORTED. Dense replication not run | A (model, feature family), C (lens) |
| I. R2 signal concentration | Compares band vs boundary | No | Boundary fraction 13.2% | Yes, **FAIL** (threshold 15%) | H (same run) |
| J. R3 workspace onset | Onset L14: 21.9% (paper) / 22.2% (script) | No | Methods agree at L13-16 | Yes, **missed** (predicted 25-35%); script: MARGINAL | H (same run) |
| K. 2.3x attention lift (decomposition) | At L15 of 36 = 41.7%, inside its own band 33.3-58.3%, outside 31-33% | Indirectly | Attention lift 2.30; concept power 0.92 against 20.11 | Yes, **H1 not met** (1 of 4 band layers above 2.0x); gate REJECTED | C, D (Qwen3-8B, same lens) |

## Rows in detail

**A. d_eff (residual-stream participation ratio).** Last-token hidden state at every layer, N = 90
prompts (60 prose + 30 chat), Qwen/Qwen3.5-27B and Qwen/Qwen3-8B base.
- **Band.** 27B: d_eff peaks at L21 of 64 (32.8%), value 39.67, inside 31-33%. 8B: local maximum at
  L12 of 36 (33.3%, 22.32), at the band's upper edge; the global maximum is L33 of 36 (91.7%, 23.41).
- **Registration.** The pre-registered band definition gave onsets at L5 (7.8%) and L4 (11.1%) and no
  offset in either model: the band never closed, so "31-33%" is a peak, not the registered band. H2
  (onsets within 10%) passes. H3 (kurtosis elevated inside the band) fails: on the 27B, kurtosis falls
  from 4,677 (mean of L0-L3) to 2,271 (mean of L19-L23).
- **Gate.** The Agni gate was REJECTED before the results were written. Its critical fix, a null
  control such as a randomly initialized model, still stands (`R3_Band_Measurement_agni_response.md`),
  so nothing yet separates the peak from an architecture or prompt-set artifact.

**B. Spectral features filed as "V-projection".**
- **Confabulation detection.** Qwen2.5-7B, from SVD of the key cache (oracle-loop-paper
  methods.tex:86-89). The pre-registered generation-only MP ensemble gives AUROC 0.707 (permutation
  p 0.001); with FWL, 0.620 (p 0.023) (`oracle-loop-paper/data/detection/analysis_results.json`).
- **Scale invariance.** Experiment 26, key-cache features, ten models. Spearman 0.826, 0.867 and 0.896
  between size tiers (KV-Experiments `paper-scale-invariance`). It orders categories and locates no
  depth.
- **Neither part locates a band.** The paper also counts the R1 "spectral feature" here, but R1's
  stable rank is computed on residual-stream block outputs (see H), not on V-projections.

**C. J-lens.**
- **Band.** In R1-v2 (27B base, community lens) readability half-rises at 56.2 [55.8-56.7] of 64
  (87.8%). On Qwen3-8B the geometric gate covers L15-L27 of 36 (41.7-75.0%; cache-tracing Table 1).
  The identity-lens control (validated v2) puts readability at 59.4 against 56.1 for the real lens on
  the same 60 forks, a difference of 3.25 layers. The late landmark is mostly a property of the
  unembedding, not of the Jacobian.
- **Opacity.** Fixed top-10 directions explain V-cache variance at 0.92 of random in the band and 0.96
  outside, and K-cache at 1.12 and 1.14 (Qwen3-8B, 20 prompts). The V readout met its registered "no
  workspace content" outcome (largest lift 1.20, under 1.3). No separate K registration was found;
  applied to K, the same thresholds are not met, because K reaches 1.87 at L30, outside the band.
- **Caveat.** The "concept" directions are formatting tokens ("...", "s"). The decomposition run's
  fixed directions from the same lens read the residual stream itself below random at L12-L21 (input
  lift 0.89, 0.84, 0.83, 0.92), so these nulls cannot separate "not in the cache" from "the directions
  read nothing here".

**D. Causal intervention (Qwen3-8B, same lens).**
- **Opacity.**
  - Swap: 0 of 360 changed output. Removal-only: 0 of 300.
  - Additive injection: v2 changed the top-1 token in 8 of 240 injected trials, v3 in 44 of 6,000.
    The injected concept never reached the top 5 (best rank 872).
  - The paper's "0/6,960" holds only as "never steered toward the concept".
  - The v3 file is truncated at source (its summary block is missing; the trials are complete).
- **Position-specific tracing** (10 probes, L12-L21 = 33.3-58.3%). Concept-specific own against cross:
  K 0.0047 against 0.0051, V 0.0014 against 0.0011. Its pre-registered positive control (H3) failed.
  Residual-stream lift at concept positions was 0.68 against 0.92 elsewhere, higher in only 9 of 40
  records, and the registration calls that outcome "methodology failure, not a cache finding".

**E. Oracle Loop** (abliterated 27B).
- **Depth.** Detects at L27-L47 (42.2-73.4%) and corrects at L35-L47 (54.7-73.4%), outside the band.
- **Results.** Exploratory: 16/20 to 2/20. Pre-registered confirmatory: 9/30 to 4/30, placebo 12/30.
  The primary endpoint (baseline against corrected) gave p 0.341, not met. Placebo against corrected:
  cell-level p 0.064.

**F. Emotion geometry** (27B distill; PR #3 data).
- **Within the story space.** It has valence-arousal structure.
- **Across prompt sets.** The content-prompt directions reach |cos| 0.026-0.110 (valence) and
  0.015-0.126 (arousal) with its first component.
- **Across depth.** The eccentricity series don't track each other (r = 0.09).
- **Coupling.** Off the injection layer, the random direction scores 0.577 against 0.547 (arousal) and
  0.521 (valence), so the coupling test is a null. It locates no band.

**G. Ghost probe** (27B distill; held from ghost-dimensions/main.tex).
- **Variance.** PC1 carries 28-67% of variance in its results section and "34-67%" in its reflection
  (L11-L48); the paper is inconsistent.
- **Opacity.** PC1's cosine with J-lens is at most 0.003 at L18-L40 in the text, and at most 0.001 in
  its table.
- **Band.** Its "31-layer workspace band" (L11-13, L15-17, L24-48) spans 17.2-75.0%, not 31-33%.

**H. R1 temporal ordering** (27B base, 60 prompts x 5 forks).
- **Ordering.** Residual-stream stable rank (`ActivationRecorder` on `model.layers`, not V-projections)
  half-rises at 17.6 [17.2-18.1]; readability at 56.2. Gap 38.6 [38.0-39.2], positive in 300 of 300
  forks. Linear-attention only: 39.0 [38.4-39.6]. Full-attention only: 14.5 [11.7-17.4].
- **Band.** A stable-rank bump of 1.47x [1.35-1.60] over flanking layers at L19-L23 (29.7-35.9%).
  Within-prompt rho 0.235.
- **Verdict.** SUPPORTED. The run's verdict text gates publication on the shuffled null and a dense
  replication.
- **Shuffled null.** It used integer half-rises, L17 and L56: real gap 39.0, which exceeds all 200
  shuffles (95th percentile 8.0). The dense replication has not been run. The identity-lens control
  (row C) shows the ordering is mostly stable rank against unembedding readability.

**I. R2.** The boundary variance fraction is 13.2% against a 15% threshold: FAIL. Boundary exceeds
workspace (d 1.85). The band means are not monotonic: mid 6.73 is below workspace 7.13.

**J. R3.** Median onset L14, from four methods spanning 13-16: 21.9% of 64 (paper), 22.2% of 63
(script). Predicted 25-35%; the script's verdict is MARGINAL. The stable-rank peak at L19 is 29.7%
(paper) or 30.2% (script), below 31% either way.

**K. 2.3x attention lift** (Qwen3-8B, 15 prompts, every third layer).
- **Result.** Attention lift at L15 is 2.30; concept power is 0.92 against 20.11 for the residual input
  (4.6%; the paper's "~4.5%" uses rounded inputs). L15 is 41.7% depth.
- **Registration.** H1 required lift above 2.0x at two or more of L12-L21: L12 1.37, L15 2.30, L18
  1.91, L21 1.41, so 1 of 4 and **not met**. The local Agni gate was REJECTED. The Opus review in human-review was
  conditional, and this run adds the concept-power metric it asked for.

## Independence

Five named instruments, but not five independent confirmations:
- **Qwen3.5-27B base** carries both band-like results, A (d_eff) and H (stable rank), plus I and J.
  A and H are spectral statistics of the same residual stream on one model; only the prompt sets
  differ. R1's "spectral feature" is this family, not V-projections.
- **Qwen3-8B, one lens** (`lens_qwen3-8b.pt`, 70 WikiText prompts) carries C's cache readouts, all of
  D, and K. Four of the paper's five opacity nulls are on this model with this lens. The fifth (G) is
  on the 27B distill with the 27B community lens, which R1 also uses.
- **Separate checkpoints:** E (abliterated 27B, its own direction and frames), F (27B distill, its own
  prompts), and B's detection (Qwen2.5-7B).
  None of these locates the band.

## What converges

1. **Residual-stream spectral statistics peak near a third of depth in Qwen3.5-27B base** (A, H).
   d_eff peaks at L21 (32.8%), and stable rank bumps at L19-L23 (29.7-35.9%). Qwen3-8B shows a local
   d_eff maximum at L12 (33.3%) but its global maximum at L33 (91.7%).
   Caveats: one feature family; no null control; the registered band never closed; kurtosis did not
   confirm.
2. **The cache nulls agree with each other** (C, D, G). V/K readouts near random at band depth (0.92x, 1.12x);
   no concept-specific signal at concept positions; no cache manipulation steers output toward the
   injected concept.
3. **Depth ordering** (H). Residual-stream stable rank half-rises about 38.6 layers before
   unembedding-based readability. Pre-registered, positive in all forks, survives the
   linear-attention control; one model.

## What does not converge

1. **The Oracle Loop** works at 42.2-73.4% depth (correction at 54.7-73.4%), and its confirmatory
   replication did not meet its primary endpoint (E).
2. **Emotion geometry** locates no band; its cross-set overlap is weak and its coupling test is a
   null (F).
3. **The J-lens places readable content late:** 87.8% in R1, 41.7-75.0% in the 8B gate (C). The
   R1 ordering is mostly unembedding against spectrum; the Jacobian adds 3.25 layers.
4. **That the cache nulls establish opacity.** The one registered positive control failed (9 of 40;
   0.68 against 0.92), and the lens's fixed directions read the residual stream below random at the
   same depths. The nulls cannot separate a closed cache from a probe that reads nothing there (C, D).
5. **The mechanism tests beyond ordering:** R2 failed (13.2%, under 15%); R3's onset is 21.9% against
   25-35% (I, J).
6. **The reconstruction finding.** The 2.3x lift sits at 41.7%, not in the band; its registered H1 was
   not met; its gate rejected it (K).
7. **The "V-projection" label.** Detection and scale use key-cache spectra on other models, and the
   ordering and band use residual-stream statistics. No row measures V-projections at the band (B, H).

## Text defects found while building this tally (for Phase B; not edited here)

- §6 "The real 38.6-layer gap exceeds every shuffled permutation". **This is my v1.1 error.** The
  shuffled null used integer half-rises (L17 and L56), gap 39.0, so v1.0's "39" was right in that
  sentence. The `R1.gone.39` check must be revised with it.
- §6 says R1's spectral feature is the "stable rank of the demeaned V-projection matrix". The script
  computes it on residual-stream block outputs.
- §6 R2 says "band means increase monotonically (early, workspace, boundary, late)". That omits the
  mid band (6.73), which sits below workspace (7.13).
- §4/§5 present the cache nulls without saying the position-specific positive control failed. "0/6,960"
  should read "never steered toward the concept" (52 additive trials changed top-1).
- §2.2/§4.6 call the detection features V-projection spectral features. They are key-cache spectra.
- The substrate section says a prompt-level d_eff bootstrap "we have not run". `substrate_bootstrap_results.json`
  (2026-08-22) exists on Starship and was not read for this tally.
- §5.1 "~4.5%" is 4.6% from unrounded concept power.

Phase B (2026-10-20) fixes all seven in the paper; its correction note lists them.

## Found during Phase B, not acted on (for Lyra)

`substrate_bootstrap_results.json` (Starship, 2026-08-22, pinned in `STARSHIP_FILES`) is a prompt-level
bootstrap of d_eff with 1,000 resamples over the same 90 prompts. It covers the 27B under its true substrate
mask and a dense Qwen3-32B (64 layers) as a control under a sham period-4 mask.
- **The 32B peak.** The dense 32B also has its d_eff maximum at L21 of 64, 29.48 against [18.94-25.09].
  That would be a second model locating the peak, but it is the same model family, so it does not answer
  the "second model family" run.
- **The intervals don't contain their own point estimates.** At the 27B's L21 the point is 39.67 against
  [23.38-30.56], and the 32B's interval misses its point in the same way. That is the expected behaviour of
  a rank-based statistic under resampling with replacement, where duplicate prompts lower the rank. The
  intervals need a different construction before anyone cites them.
- **The period-4 contrasts** in the same file exclude zero for both models, including the dense control
  under its sham mask. That suggests the contrast is not specific to substrate.

The paper now says only that the bootstrap was run and is not reported. None of these numbers enters the
paper.
