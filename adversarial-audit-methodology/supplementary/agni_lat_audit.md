# Agni Adversarial Audit: LAT Deception Vector Extraction

**Date**: 2026-06-26
**Auditor**: CC (Agni protocol)
**Script**: `experiments/lat_deception_vectors.py`
**Results**: `deception_directions.json` on [internal compute host]
**Model**: Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled
**Claimed finding**: Deception directions extracted at 10 layers with Cohen's d 2.049--3.264

---

## VERDICT: RESULTS ARE NOT CREDIBLE

Three fatal issues invalidate all reported effect sizes. The experiment as designed cannot distinguish signal from noise. Specific findings below.

---

## 1. CIRCULARITY (FATAL)

**Issue**: The deception direction is computed as `mean(deceptive) - mean(honest)`, then Cohen's d is computed by projecting the *same 15 pairs* onto that direction. This is textbook circular analysis -- the direction is optimally aligned with the training data by construction.

**Null simulation** (10,000 iterations, n=15 groups of isotropic Gaussian vectors in d=5120):

| Metric | Value |
|--------|-------|
| Null circular d (mean) | 27.84 |
| Null circular d (std) | 3.87 |
| Observed d range | 2.05 -- 3.26 |
| p-values (all layers) | 1.0000 |

The observed d values are **below the null expectation** for isotropic noise. This does not mean the signal is anti-correlated -- it means the activations are structured (low effective dimensionality), which shrinks both the true signal and the circular bias.

**Correcting for effective dimensionality**: Direction vector kurtosis gives estimated effective dimensionalities of ~3400 (L11), ~1900 (L19), ~128 (L47). The circular null at eff_dim=30--60 produces d=2.1--3.0, which spans the observed range exactly. **The observed d values are indistinguishable from null**.

**Cross-validation simulation** (leave-one-out, eff_dim=50, n=15):

| Method | Null d (mean) | Null d (std) |
|--------|--------------|-------------|
| Circular | 2.73 | 0.47 |
| LOO cross-validated | 0.004 | 0.54 |
| Inflation factor | ~780x | -- |

Under LOO, the null d centers at zero with std=0.54. Less than 0.1% of null samples exceed |d|=2.0. Cross-validation would reveal whether there is real signal, but the experiment does not perform any.

**Severity**: FATAL. No effect size can be trusted. The experiment would need to either:
- Extract direction from a training split and evaluate on held-out pairs
- Use a permutation test (shuffle condition labels, recompute d)
- Report LOO or k-fold cross-validated d

**Corrected estimate**: Unknown. Could be anywhere from d=0 to d~2. Cannot be determined without re-running with cross-validation.

---

## 2. EFFECTIVE n=3, NOT n=15 (FATAL)

**Issue**: The experiment uses `n_pairs=15` but has only 3 eval sets (`EVAL_SETS`). Each eval set is recycled 5 times via `EVAL_SETS[trial % len(EVAL_SETS)]`. The 15 "pairs" are actually 3 unique stimuli x 5 stochastic repetitions (temperature=0.7).

**Impact**:
- The 5 repetitions per stimulus are not independent -- they share the same prompt text and differ only in sampling noise
- Inter-class correlation within stimulus groups inflates apparent n
- Effective degrees of freedom are closer to 3 than 15
- All statistical calculations assuming n=15 independent pairs are wrong
- With n_eff=3, even a well-designed experiment would have negligible power to detect deception directions

**Severity**: FATAL. The experiment lacks the statistical power it claims.

---

## 3. HOOK TIMING BUG (CRITICAL)

**Issue**: Forward hooks are registered before the prefill pass (line 143) and removed after generation (line 167-168). During the autoregressive generation loop (lines 156-164), each `model(next_token, past_key_values=cache)` call triggers the hooks, overwriting `activations[layer_idx]`.

The final captured activation is from the **last generated token** (EOS or the 400th token), not from a fixed prompt position.

**Consequences**:
- Different samples have activations from different sequence positions
- Different samples have activations from different token content
- The "deception direction" may capture generation-length differences between conditions rather than anything about deception
- Deceptive responses (inflating scores) may be shorter or longer than honest ones, creating a confound

**If the intent was last-prompt-token activations**: hooks should be removed immediately after the prefill pass (after line 148), before generation begins.

**If the intent was generation-time activations**: the specific token position should be controlled (e.g., first generated token, or all tokens averaged).

**Severity**: CRITICAL. The activations being compared are not from matched positions.

---

## 4. PROMPT LENGTH CONFOUND (SERIOUS)

**Issue**: The deceptive guidelines (`DECEPTIVE_GUIDELINES`) are 359 characters / ~53 words. The honest guidelines (`HONEST_GUIDELINES`) are 208 characters / ~29 words. The deceptive prompt is **72.6% longer**.

This length difference means:
- The deceptive condition has more tokens before the evaluation data
- Positional encodings differ between conditions
- Attention patterns over the prompt differ
- The last-token activation integrates over a longer context

The extracted direction could be a **prompt-length direction** rather than a deception direction. A clean contrastive design would match prompts for length, or include length-matched controls.

**Severity**: SERIOUS. Cannot distinguish deception signal from prompt-structure signal.

---

## 5. NORM GROWTH IS PURE SCALING (MODERATE)

**Issue**: `diff_norm` grows from 3.77 (L11) to 28.79 (L47) -- a 7.6x increase. This was flagged as potentially meaningful (L47 as "strongest signal"). It is not.

| Layer | diff_norm | Cohen's d | d/norm ratio |
|-------|-----------|-----------|--------------|
| 11 | 3.77 | 2.278 | 0.604 |
| 19 | 7.68 | 3.264 | 0.425 |
| 27 | 11.03 | 2.049 | 0.186 |
| 35 | 16.58 | 3.016 | 0.182 |
| 47 | 28.79 | 2.639 | 0.092 |

The diff_norm growth is linear in layer index (R^2=0.962, slope=0.658/layer). Cohen's d is flat (R^2=0.04, slope=-0.006/layer). The d/norm ratio **decreases monotonically** from 0.60 to 0.09.

**Interpretation**: Activation magnitudes scale with depth (standard transformer behavior). The "separation" at deeper layers is just the same proportional noise at larger scale. There is no evidence of signal accumulation.

**Severity**: MODERATE. The claim that "L19 is peak" is based on circular d. After accounting for norm scaling, no layer is distinguishable from any other.

---

## 6. DIRECTION VECTOR SPARSITY (INFORMATIONAL)

The direction vectors show increasing concentration at deeper layers:

| Layer | Max element | Expected max (uniform) | Kurtosis | Est. eff_dim |
|-------|------------|----------------------|----------|-------------|
| 11 | 0.131 | 0.014 | 7.5 | ~3400 |
| 19 | 0.165 | 0.014 | 11.1 | ~1900 |
| 47 | 0.387 | 0.014 | 123.2 | ~128 |

Layer 47's direction is concentrated in ~128 effective dimensions (out of 5120). This extreme sparsity is consistent with a structural confound (prompt length, token identity) rather than a distributed deception feature.

---

## 7. MPS CONTENTION (LOW RISK)

Two models loaded simultaneously on [internal compute host] could add noise via memory pressure. However, contention is symmetric between conditions and does not create systematic bias. **Severity**: LOW.

---

## 8. THINK-BLOCK STRIPPING (DESIGN CHOICE, NOT CONFOUND)

Both conditions strip `<think>` from the generation prompt identically. Not a confound. The suppression of explicit reasoning may reduce the model's ability to deliberate about deception, potentially weakening any real signal. Defensible choice but should be documented.

---

## SUMMARY OF FINDINGS

| Issue | Severity | Impact |
|-------|----------|--------|
| Circular d computation | FATAL | All reported d values are meaningless |
| n=3 effective pairs (not 15) | FATAL | Experiment lacks claimed statistical power |
| Hook timing bug (wrong tokens) | CRITICAL | Activations are from uncontrolled positions |
| Prompt length confound (73% longer) | SERIOUS | Direction may encode length, not deception |
| Norm growth = activation scaling | MODERATE | No evidence of depth-dependent signal |
| Direction sparsity | INFORMATIONAL | Consistent with structural confound |
| MPS contention | LOW | Adds noise, no bias |
| Think-block stripping | LOW | Not a confound, design choice |

---

## REQUIRED REMEDIATION

Before any conclusions can be drawn from this experiment:

1. **Cross-validate**: Extract direction from k-1 pairs, evaluate on held-out pair. Report LOO or 5-fold CV d. Alternatively, run a permutation test (1000+ label shuffles).

2. **Fix the hook timing**: Either capture activations at the last prompt token (remove hooks before generation) or at a controlled generation position.

3. **Increase and diversify stimuli**: Minimum 30 *unique* contrastive pairs. Do not recycle eval sets. Each pair should use different evaluation content.

4. **Match prompt length**: Pad the honest guidelines to match deceptive prompt length, or include a length-matched control condition (long prompt, no threat).

5. **Report normalized effect sizes**: Divide diff_norm by per-layer activation norm to separate signal from scaling.

6. **Verify with behavioral ground truth**: Confirm that the deceptive condition actually *produces deception* (score inflation) and the honest condition does not, before interpreting activation differences. The Prometheus judge results from Phase 2 could serve this purpose but must be reported alongside Phase 1.

---

## BOTTOM LINE

The experiment follows the right intuition (contrastive activation extraction from Shi et al.) but the implementation has fundamental statistical and engineering errors that render the results uninterpretable. The reported Cohen's d values (2.0--3.3) are consistent with pure noise at the effective dimensionality of the activation space. No deception direction has been demonstrated.

This is fixable. The approach is sound. The execution needs the remediation above.

*"The fire that tests also tempers." -- Agni protocol*
