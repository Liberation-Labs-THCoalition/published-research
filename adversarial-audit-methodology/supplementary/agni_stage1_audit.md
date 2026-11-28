# Agni Adversarial Audit: LAT Deception Direction v2 Results

**Date**: 2026-06-27
**Auditor**: CC (Agni protocol)
**Script**: `experiments/lat_deception_v2.py`
**Results**: `experiments/results/lat_deception_v2.json`
**Prior audits**: `agni_lat_audit.md` (v1, killed), `agni_lat_v2_preflight.md` (v2 preflight, conditional pass)
**Model**: Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled (64 layers, 16 full-attention)
**Reported findings**: LOO d = 0.52 (L3) to 34.98 (L31), all layers perm_p = 0.000000

---

## VERDICT: METHODOLOGICALLY SOUND, INTERPRETIVELY OVERSTATED

The code is correct. The statistics are valid. The results do not yet demonstrate a "deception direction." They demonstrate that two different guideline texts produce different activations, which is trivially expected. Whether that difference encodes deception specifically cannot be determined from this experiment alone.

The v1 fatal issues (circular d, effective n=3, hook timing) are genuinely fixed. The v2 preflight issues (permutation shuffling projections not labels, bootstrap independence, missing Bonferroni) are also fixed. No remaining code bugs were found. The critique that follows is about interpretation and design, not implementation.

---

## 1. EFFECT SIZE PLAUSIBILITY: d=35 IS NOT ANOMALOUS (PASS, WITH REFRAMING)

d=35 at Layer 31 sounds preposterous by social science standards. It is not.

Cohen's d = (mean_difference) / (pooled_standard_deviation). The numerator (guideline-induced activation difference) is a **constant** — every deceptive sample has identical guideline text, every honest sample has identical guideline text. The denominator (within-group variance along the extracted direction) comes entirely from the varying eval content, which is orthogonal to the guideline difference by design.

This makes d a measure of **linearity**: how well the model additively decomposes the guideline effect from the eval content effect. At Layer 31, the model's representation is highly additive — the guideline effect barely interacts with the eval content. By Layer 47, representations are more entangled, and within-group variance along the direction grows, shrinking d to 4.9.

**Null simulation confirms**: random data in 5120 dimensions with n=30 produces LOO d ~ N(0, 0.36). The observed d=0.52 (smallest) is already 1.5 null SDs away. d=35 is 97 null SDs away. Not inflated by the LOO procedure.

**Reframing**: d=35 does not mean "the deception signal is 35 standard deviations strong." It means "the model processes these two specific guideline texts nearly additively at Layer 31." Interesting but not what "deception direction" implies.

**Status**: No methodological problem. Interpretation requires reframing.

---

## 2. PERMUTATION TEST: ALL p=0.000000 IS EXPECTED (PASS)

10,000 permutations with correct label-shuffling and full LOO recomputation. 0 exceedances at all 12 layers yields p < 0.0001 per layer, p < 0.0012 Bonferroni-corrected.

Code path examined for bugs that could force p=0. None found. The comparison `perm_d >= observed_loo_d` is one-sided, appropriate given the directional hypothesis. Permutation correctly recomputes full LOO with shuffled labels.

**Design note**: The test is unpaired (shuffles across all 60 activations rather than within-pair flips). Since the design is paired (same eval content in both conditions), a paired permutation test (2^30 possible flips) would be more powerful and more correct. The unpaired test is conservative (it breaks the pairing), so p-values are valid but slightly conservative.

**Status**: Correct. All-zero result is genuine, not a bug.

---

## 3. DIM 3994: SUSPICIOUS BUT NOT FATAL (CONCERN, NOT KILL)

Rank #1 in the deception direction at 11 of 12 layers, with sign oscillations (--+++------+).

**Evidence for artifact**:
- Single dimension dominating across 11 layers is unusual for a distributed semantic feature
- Sign oscillations suggest something structural (token embedding, position) rather than semantic
- BF16 quantization could produce clean patterns at high-magnitude dimensions

**Evidence against fatal artifact**:
- Removing dim 3994 reduces d by at most 4% (L3) and <1% at most layers. Results NOT driven by this dimension
- Direction involves ~500-600 dimensions at 50%-variance threshold — broadly distributed
- 395 of 5120 dimensions show similar quantization patterns at L31; dim 3994 just has the largest value
- Becomes irrelevant at L47 (rank 4545/5120), consistent with being processed away

**Verdict**: Likely a token-level or positional feature that differs between guideline texts. NOT driving results. Aesthetic concern, not statistical.

**Recommended check**: Examine embedding matrix at dim 3994 to identify corresponding token(s).

---

## 4. CROSS-LAYER CORRELATION: CONSISTENT WITH SIGNAL (PASS)

Banded pattern (strong adjacent 0.37-0.67, weak distant ~0.01) matches residual stream geometry: each layer adds an update, so adjacent layers share representation while distant layers are nearly orthogonal. Pure noise would give near-zero cosine everywhere. Banded structure is evidence of real layer-to-layer processing.

---

## 5. BOOTSTRAP CI: STILL SLIGHTLY ANTI-CONSERVATIVE (MODERATE)

Fold-aware paired bootstrap was correctly implemented. CIs are 16-84% of standard formula width. This narrowness is **partially legitimate** — LOO projections onto a nearly-fixed direction have low variance because removing 1/30 samples barely changes the direction.

Extreme CI asymmetry (upper arm 1.8x-4.0x larger than lower) reveals right-skewed bootstrap d distribution from occasional low-pooled-sd samples. Percentile bootstrap has poor coverage with skewed statistics. BCa (bias-corrected and accelerated) bootstrap would be more appropriate.

**Estimated actual coverage**: 85-90% (improved from preflight's ~80%, but not nominal 95%). Point estimates are unbiased.

---

## 6. CONTENT CONFOUND (SERIOUS — DESIGN, NOT CODE)

**This is the only finding that matters for interpretation.**

Two FIXED guideline texts, 30 varying eval contents. The "deception direction" is guaranteed to capture *any* difference between the two texts — token-level, syntactic, structural, or semantic. The permutation test confirms the difference isn't due to eval content variation, but cannot determine *what kind* of guideline difference is captured.

The claim "we extracted a deception direction" requires an additional assumption: that the direction captures the *semantic* difference (deception framing vs. improvement framing) rather than the *surface* difference (different words, ~5% length difference, different sentence structures).

**What would resolve this**: A **transfer test** — apply the extracted direction to a *different* deception scenario (different guideline text, different topic, different model context). If it still separates deceptive from honest conditions, the semantic interpretation is supported. Without transfer, the experiment establishes only that "two specific texts produce different activations."

The LAT literature (Shi et al. 2025) has the same limitation. The extraction step is standard. But calling the result a "deception direction" rather than a "guideline-text-difference direction" is an interpretive leap the data does not yet support.

**Severity**: SERIOUS for interpretation. Not a code bug. Not a statistical error. A design limitation inherent to single-contrast contrastive extraction.

---

## 7. ADDITIONAL OBSERVATIONS

**7a. FWL residualization**: Pre-registered but not implemented. Generation length moot (no generation). Prompt length ~5% difference unlikely to change results. Pre-registration deviation should be noted.

**7b. Contradictory eval labels**: 13/30 eval sets contain questions where "incorrect" answer is correct. Symmetric between conditions. Not a confound, but model might react differently to contradictions depending on deceptive framing.

**7c. d/norm ratio**: Peaks at L23-L27 (6.2-6.3), drops sharply at L35+ (0.2-1.1). Mid-layers have highest signal-to-noise, consistent with probing literature. Evidence of real processing, not trivial artifact.

---

## SUMMARY TABLE

| Check | Status | Impact |
|-------|--------|--------|
| LOO cross-validation | PASS | Correctly eliminates circularity |
| n=30 unique eval sets | PASS | Adequate sample size |
| Hook timing (prefill only) | PASS | Fixed position, no generation confound |
| Permutation test (label + LOO) | PASS | Correct null, p=0 expected |
| Bonferroni correction | PASS | Implemented, all layers survive |
| d=35 plausibility | PASS (reframe) | Measures linearity, not psychological effect |
| Dim 3994 dominance | CONCERN (not fatal) | <4% contribution; likely token feature |
| Cross-layer correlation | PASS | Banded structure = real signal |
| Bootstrap CIs | MODERATE | ~85-90% coverage, not 95%; BCa recommended |
| Content confound | SERIOUS (design) | Cannot distinguish text from deception |
| FWL not implemented | LOW | Pre-registration deviation, immaterial |
| Eval contradictions | LOW | Symmetric, not a confound |

---

## REQUIRED NEXT STEPS

1. **Transfer test**: Apply L31 direction to a novel deception scenario. Different guideline text, different topic. If it separates, "deception direction" claim strengthens. If not, it's a prompt-specific direction.
2. **Replace percentile bootstrap with BCa** for honest 95% CIs.
3. **Investigate dim 3994** via embedding matrix to identify what token feature it encodes.
4. **Report d/norm alongside d** to separate activation scaling from signal strength.

---

## BOTTOM LINE

The experiment is methodologically sound. The code is correct. The statistics are valid. The three v1 fatal issues and three v2 preflight issues are all genuinely fixed.

What remains is an interpretive gap. "Deception direction" is a misnomer until transfer is demonstrated. What this experiment has shown: (1) two specific guideline texts produce cleanly separable activations, with peak separation at L31, and (2) this separation is statistically significant beyond any reasonable doubt. Whether that separation encodes "deception" as a concept — rather than "these specific words appeared in the prompt" — requires a transfer experiment.

The large d values are not artifacts. They reflect nearly-additive processing of guideline text at mid-layers. They are not comparable to social science effect sizes and should not be described in those terms.

*"The fire that tests also tempers." -- Agni protocol*
