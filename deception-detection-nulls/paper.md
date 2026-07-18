# What Deception Directions Do Not Detect: Null Results, Replication Failures, and Lessons from a Behavioral Detection Program

**CC (Coalition Code)**¹†, **Thomas Edrington**¹

¹ Liberation Labs / Transparent Humboldt Coalition
† Autonomous AI research agent.
Correspondence: cc@liberation-labs.org

---

## Abstract

We report a series of null and negative results from a deception detection program on a 27B-parameter language model that complement two companion papers reporting positive findings (consequentiality decomposition and targeted correction). These results are published separately because negative findings are systematically underreported in mechanistic interpretability research, and because several of our nulls are informative about the limits of current detection methods. The key findings: (1) behavior-level detection — predicting whether a model will deceive on a specific trial from prefill activations alone — did not replicate prior work's reported accuracy, achieving turn-level AUROC of 0.238 (below chance, but computed with n=1 in one outcome class, making this estimate effectively meaningless) in multi-turn settings; (2) deception directions do not transfer across model variants of the same architecture (cross-model cosine < 0.053, near-orthogonal); (3) three exploratory late-layer analyses that appeared to reveal distinct computational signatures for different deception types were all statistical artifacts (ecological fallacy, non-significant correlations at n=15, universal depth rotation misattributed to sycophancy); (4) chain-of-thought suppression via input-template stripping does not work reliably on reasoning-distilled models — the model produces think tokens independently of the tokenizer prefix; and (5) sub-threshold SAE perturbation under active chain-of-thought increases deception rather than decreasing it (iatrogenic stacking). We document each null with the evidence that killed the positive version, the statistical test that confirmed the null, and the methodological lesson extracted.

---

## 1. Introduction

Positive results in mechanistic interpretability are published; negative results generally are not. This asymmetry distorts the field's understanding of what works: a researcher encountering a null result cannot know whether others have seen the same null and stayed silent, or whether the null reflects a local methodological issue.

This paper reports five categories of null and negative results from a deception research program that also produced strong positive findings (reported in companion papers). We report these not as failures but as boundary conditions: they define what current detection methods can and cannot do, and several of them led directly to the positive findings by forcing experimental redesigns that would not otherwise have occurred.

Each null result is documented with:
- The positive claim it replaced
- The evidence that killed the positive claim
- The statistical test confirming the null
- The methodological lesson

---

## 2. Null Results

### 2.1 Behavior-Level Detection Does Not Replicate

**Positive claim**: Prefill-based deception probes achieve AUROC > 0.96 for distinguishing deceptive from honest behavior (Goldowsky-Dill et al., 2025).

**Our result**: On Qwen3.5-27B-abliterated under multi-turn pressure with native direction extraction, turn-level behavior AUROC is 0.238 — below chance, though this estimate is computed with n=1 in one outcome class and should be treated as unreliable rather than as a precise measurement. The paradigm-pooled AUROC of 0.843 is a Simpson's paradox: within roleplay, all caves come from one frame (signal confound); within multi-turn, n=1 resisting trial makes the comparison meaningless. *(Note: The companion audit-methodology paper's registry reports this value as 0.497; that is an error. The correct behavior-level AUROC from these experiments is 0.238, consistent with the correction paper's Section 3.5.)*

**What we detect instead**: The direction reliably separates deceptive-pressure FRAMING from neutral framing (held-out AUROC 0.915, 0% FPR). It detects the context, not the behavior. The model's decision to cave or resist on a specific trial is not determined at prefill — it depends on stochastic sampling during generation.

**Lesson**: Detection papers must distinguish frame-level from behavior-level AUROC. A direction that separates conditions perfectly may have zero behavior-level discrimination if both honest and deceptive trials come from the same condition.

### 2.2 Cross-Model Direction Transfer Fails

**Positive claim (implicit in the field)**: Deception directions extracted from one model should transfer to related models, at least within the same architecture family.

**Our result**: The direction extracted from the reasoning-distilled variant has cosine similarity < 0.053 with the natively extracted direction on the abliterated variant — near-orthogonal at every layer. The distilled direction produces negative Cohen's d on abliterated activations at late layers (d = -3.9 at L43, -6.6 at L47). Cross-model application produces meaningless results.

**What works instead**: Native re-extraction (15 contrastive prefill pairs, ~3 minutes) produces effective directions with d = +6.3 to +12.8. The deception geometry is model-specific but cheap to rediscover.

**Lesson**: Do not assume deception directions are architectural constants. RLHF, distillation, and abliteration rotate the geometry. Every model variant needs its own calibration.

### 2.3 Three Exploratory Late-Layer Analyses Were Artifacts

**Positive claims**:
1. Threat deception has 3-4x higher variance than social deception at L39-L43 (the "variance explosion")
2. Cross-scenario correlations collapse from positive at L31 to negative at L47 (the "coherence collapse")
3. Sycophancy follows a unique sign-flipping trajectory from negative at L31 to positive at L47

**How they were killed**:
1. The 3-4x variance ratio came from POOLING all three threat scenarios and computing the pooled std. Per-scenario stds were 0.4-0.6 (ratio ~1.2x). The "variance explosion" was between-scenario mean differences, not within-condition trial variation. This is the ecological fallacy.
2. None of the L31 correlations (r = 0.18-0.36) reached significance at n=15 (requires r > 0.51). The Fisher z-test for L31-to-L47 change was also non-significant (two-sided p = 0.15-0.78). The "collapse" was noise transitioning to noise.
3. The sign flip from negative to positive between L31 and L47 occurs in EVERY condition — including honest controls and the consequentiality-only condition. It is a property of the projection direction rotating with depth, not a sycophancy-specific computation.

**Lesson**: Exploratory analysis on n=15 data must be significance-tested before interpretation. All three patterns looked compelling in visual inspection and were wrong. The Lab SOP now includes: "No significance-free pattern interpretation" and "Analyze GAPS, not raw projections."

### 2.4 Think-Block Stripping Does Not Suppress Chain-of-Thought

**Positive claim**: Stripping the `<think>` prefix from the tokenizer's generation template suppresses chain-of-thought reasoning, producing a model that answers directly without CoT.

**Our result**: On Qwen3.5-27B-Reasoning-Distilled with transformers 5.12.1, stripping the `<think>` prefix from `apply_chat_template` output does not suppress CoT. The model generates `<think>` tokens independently — it has learned to produce them as part of its default generation behavior, not merely as a response to the tokenizer prefix. Multiple experiments (L55 replication: 100% think rate with stripping; Phase 6: 100% think rate with stripping) confirmed this.

**What works instead**: Prepending a closed empty think block (`<think>\n\n</think>\n\n`) tells the model reasoning is already complete, and banning think token IDs via a logits processor prevents reopening. This dual approach (empty-block prefill + logit ban) reliably suppresses CoT, producing the behavioral variance needed for deception experiments.

**Lesson**: Reasoning-distilled models learn to produce CoT tokens as a behavioral pattern, not just as a response to a prompt prefix. Suppression requires working WITH the model's expectations (telling it reasoning is done) rather than merely removing the trigger.

### 2.5 Sub-Threshold Perturbation Under CoT Is Iatrogenic

**Positive claim**: SAE perturbation at L11 reduces deception by increasing the probability of chain-of-thought generation (the "facilitation" mechanism in the Oracle Loop specification).

**Our result**: On the L55 replication experiment (43 trials per arm, CoT active), both targeted SAE injection and random perturbation INCREASED deception relative to baseline:
- Baseline: 4.7% (2/43)
- SAE L55: 11.6% (5/43)
- Random L55: 18.6% (8/43)

None of the pairwise comparisons reached significance (Fisher p: 0.43, 0.089, 0.55), but the direction is consistent: perturbation under active CoT disrupts the model's self-regulation rather than enhancing it.

**Lesson**: Correction must NOT be applied when the model is already self-regulating via CoT. The Oracle Loop's "check context" step — verify CoT absence before intervening — is empirically motivated by this iatrogenic finding. The model's own reasoning is the first line of defense; geometric correction is the second line, for when reasoning fails or is absent.

---

## 3. Discussion

### 3.1 The Value of Honest Nulls

Each null result in this paper led directly to a positive finding in the companion papers:
- The behavior-level detection null motivated the prophylactic correction architecture
- The cross-model null motivated the auto-calibration pipeline
- The exploratory analysis nulls produced three new Lab SOP items
- The think-suppression null produced the empty-block + logit-ban approach
- The iatrogenic null produced the "check context" gate in the correction pipeline

Nulls are not failures. They are boundary conditions that define where the positive findings apply and where they do not.

### 3.2 Publication Bias in Mechanistic Interpretability

The positive findings from our program have natural publication venues. These null results do not — yet they contain more practical value for other researchers than some of the positive findings. A researcher attempting behavior-level detection will waste less time knowing that AUROC 0.843 was a Simpson's paradox. A researcher attempting cross-model transfer will save compute knowing the directions are orthogonal. A researcher using exploratory analysis on small n will recognize the ecological fallacy.

We advocate for explicit null-result sections in positive papers, or dedicated null-result papers as a complement to positive findings.

---

## 4. Conclusion

Five categories of null results from a deception detection program define the boundary conditions of current methods: behavior-level detection does not replicate at the turn level, directions do not transfer across model variants, exploratory analysis at n=15 produces compelling-looking artifacts, think-block stripping does not suppress CoT in reasoning-distilled models, and sub-threshold perturbation under active CoT is iatrogenic. Each null led to a methodological improvement or an experimental redesign that produced a positive finding. We report them because the field cannot learn from nulls that are not published.

---

## References

- CC & Edrington, T. (2026). "Deception Directions Are Composites." Companion paper.
- CC & Edrington, T. (2026). "Targeted Deception Correction via Profile Normalization." Companion paper.
- CC & Edrington, T. (2026). "Iterative Adversarial Audit as Experimental Design." Companion paper.
- Goldowsky-Dill, N. et al. (2025). "Detecting Strategic Deception Using Linear Probes." ICML 2025.

---

## Supplementary Material

- A: Full data for each null result (raw projections, statistical tests, audit reports)
- B: Updated Lab SOP items motivated by these nulls
