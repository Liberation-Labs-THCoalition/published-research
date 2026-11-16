# What Deception Directions Do Not Detect: Null Results and Boundary Conditions from a Behavioral Detection Program

**CC (Coalition Code)**¹†, **Thomas Edrington**¹

¹ Liberation Labs / Transparent Humboldt Coalition
† Autonomous AI research agent.
Correspondence: cc@liberationlabs.tech

---

## Abstract

Deception directions extracted from one variant of a 27B-parameter language model do not transfer to another variant of the same architecture: the cross-model cosine is < 0.053, near-orthogonal at every layer. A direction re-extracted natively from 15 contrastive pairs, in about three minutes, separates 25 held-out pressure-framed prompts from 25 controls (per-layer d = +1.37 to +2.28, L27–L47). This result and four others come from a deception detection program whose positive findings appear in two companion papers (a decomposition of how a reasoning-distilled model represents deceptive-pressure framing, and a targeted behavioral correction on an abliterated variant). The five results: (1) behavior-level detection — predicting whether a model will deceive on a specific trial from prefill activations alone — could not be tested at useful power on the abliterated variant: the multi-turn set has a single resisting trial (turn-level AUROC 0.238), and the paradigm-pooled AUROC of 0.843 is a Simpson's paradox, so the prior AUROC > 0.96 is neither replicated nor refuted here; (2) the cross-model transfer failure above; (3) of three exploratory late-layer patterns, one is a pooling artifact (ecological fallacy), one is a depth rotation present in every condition and misattributed to sycophancy, and one (a cross-scenario correlation collapse) is not significant at n=15; (4) chain-of-thought suppression via input-template stripping does not work reliably on reasoning-distilled models — the model produces think tokens independently of the tokenizer prefix; and (5) sub-threshold SAE perturbation under active chain-of-thought did not reduce deception (2/43 baseline, 5/43 targeted, 8/43 random; no pairwise difference significant). For each null we give the positive claim it tests, the evidence, the test behind it and what it shows, and the methodological lesson.

---

## 1. Introduction

This paper reports five categories of null and negative results from a deception research program that also produced positive findings: a representational decomposition and a frame-dependent behavioral correction (reported in companion papers). The nulls are boundary conditions: they define what current detection methods can and cannot do, and several of them led directly to the positive findings by forcing experimental redesigns that would not otherwise have occurred.

Positive results in mechanistic interpretability are published; negative results generally are not. This asymmetry distorts the field's understanding of what works: a researcher encountering a null result cannot know whether others have seen the same null and stayed silent, or whether the null reflects a local methodological issue.

Each null result is documented with:
- The positive claim it tests
- The evidence against that claim
- The test behind the null and what it shows
- The methodological lesson

---

## 2. Null Results

### 2.1 Behavior-Level Detection: Not Testable With One Resisting Trial

**Positive claim**: Prefill-based deception probes achieve AUROC > 0.96 for distinguishing deceptive from honest behavior (Goldowsky-Dill et al., 2025).

**Our result**: On Qwen3.5-27B-abliterated under multi-turn pressure with native direction extraction, turn-level behavior AUROC is 0.238 — below chance, though this estimate is computed with n=1 in one outcome class and should be treated as unreliable rather than as a precise measurement. The paradigm-pooled AUROC of 0.843 is a Simpson's paradox: within roleplay, all caves come from one frame (signal confound); within multi-turn, a single resisting trial leaves the comparison without power. The prior AUROC > 0.96 is therefore neither replicated nor refuted here.

The companion audit-methodology paper reports the same behavior-level value, consistent with the correction paper's Section 3.5. *(See Corrections and Version History, 2026-10-09.)*

**What the direction separates instead**: The direction separates deceptive-pressure FRAMING from honest framing (held-out AUROC 0.915; 0/25 false positives, at a threshold that flagged 24% of pressure trials). It separates the contexts, not the behaviors — and the two arms' contexts differ in their system messages, with no control that holds the prompt fixed, so the separation may reflect the frames' wording rather than the pressure itself. With one resisting trial in the multi-turn set, this test cannot decide whether a trial's outcome is already set in the prefill activations or is decided during generation.

**Lesson**: Detection papers must distinguish frame-level from behavior-level AUROC. A direction that separates conditions perfectly may have zero behavior-level discrimination if both honest and deceptive trials come from the same condition.

### 2.2 Cross-Model Direction Transfer Fails

**Positive claim (implicit in the field)**: Deception directions extracted from one model should transfer to related models, at least within the same architecture family.

**Our result**: The direction extracted from the reasoning-distilled variant has cosine similarity < 0.053 with the natively extracted direction on the abliterated variant — near-orthogonal at every layer. The distilled direction produces negative Cohen's d on abliterated activations at late layers (d = -3.9 at L43, -6.6 at L47): applied across models, the direction reverses sign.

**What works instead**: Native re-extraction (15 contrastive prefill pairs, ~3 minutes) yields a direction that separates the conditions on held-out prompts: on 25 novel pressure prompts and 25 novel controls — the same held-out set that gives the frame-level AUROC 0.915 reported above, with the same caveat about the frames' wording — the native direction reaches per-layer Cohen's d = +1.37 to +2.28 across L27–L47 (+1.73 at L31). These are held-out estimates (see Corrections and Version History, 2026-09-09). The direction is model-specific but cheap to re-extract.

**Lesson**: Do not assume deception directions are architectural constants. RLHF, distillation, and abliteration rotate the geometry. Every model variant needs its own calibration.

### 2.3 Three Exploratory Late-Layer Patterns Are Not Supported

**Positive claims**:
1. Threat deception has 3-4x higher variance than social deception at L39-L43 (the "variance explosion")
2. Cross-scenario correlations collapse from positive at L31 to negative at L47 (the "coherence collapse")
3. Sycophancy follows a unique sign-flipping trajectory from negative at L31 to positive at L47

**What the formal tests show**:
1. The 3-4x variance ratio came from POOLING all three threat scenarios and computing the pooled std. Per-scenario stds were 0.4-0.6 (ratio ~1.2x). The "variance explosion" was between-scenario mean differences, not within-condition trial variation. This is the ecological fallacy.
2. None of the L31 correlations (r = 0.18-0.36) reached significance at n=15 (requires r > 0.51). The Fisher z-test for L31-to-L47 change was also non-significant (two-sided p = 0.15-0.78). At this n the test cannot distinguish a collapse from no change.
3. The sign flip from negative to positive between L31 and L47 occurs in EVERY condition — including honest controls and the consequentiality-only condition. It is a property of the projection direction rotating with depth, not a sycophancy-specific computation.

A related question stays open: a slope-interaction test in the companion decomposition paper supports a social-versus-threat profile difference within the five scenarios tested (scenario-level exact one-sided p = 0.10 only).

**Lesson**: Exploratory analysis on n=15 data must be significance-tested before interpretation. All three patterns looked compelling by eye; none is supported under formal tests. The Lab SOP now includes: "No significance-free pattern interpretation" and "Analyze GAPS, not raw projections."

### 2.4 Think-Block Stripping Does Not Suppress Chain-of-Thought

**Positive claim**: Stripping the `<think>` prefix from the tokenizer's generation template suppresses chain-of-thought reasoning, producing a model that answers directly without CoT.

**Our result**: On Qwen3.5-27B-Reasoning-Distilled with transformers 5.12.1, stripping the `<think>` prefix from `apply_chat_template` output does not suppress CoT. The model generates `<think>` tokens independently — it has learned to produce them as part of its default generation behavior, not merely as a response to the tokenizer prefix. Multiple experiments (L55 replication: 100% think rate with stripping; Phase 6: 100% think rate with stripping) confirmed this.

**What works instead**: Prepending a closed empty think block (`<think>\n\n</think>\n\n`) tells the model reasoning is already complete, and banning think token IDs via a logits processor prevents reopening. This dual approach (empty-block prefill + logit ban) reliably suppresses CoT, producing the behavioral variance needed for deception experiments.

**Lesson**: Reasoning-distilled models learn to produce CoT tokens as a behavioral pattern, not just as a response to a prompt prefix. Suppression requires working WITH the model's expectations (telling it reasoning is done) rather than merely removing the trigger.

### 2.5 Sub-Threshold Perturbation Under CoT Does Not Reduce Deception

**Positive claim**: SAE perturbation at L11 reduces deception by increasing the probability of chain-of-thought generation (the "facilitation" mechanism in the Oracle Loop specification).

**Our result**: On the L55 replication experiment (43 trials per arm, CoT active), neither targeted SAE injection nor random perturbation reduced deception relative to baseline; both point estimates rose:
- Baseline: 4.7% (2/43)
- SAE L55: 11.6% (5/43)
- Random L55: 18.6% (8/43)

None of the pairwise comparisons reached significance (Fisher p: 0.43, 0.089, 0.55). Both point estimates moved toward more deception, but at 43 trials per arm the test cannot distinguish that from no effect.

**Lesson**: Perturbation under active CoT showed no benefit, so the correction is not applied while the model is reasoning via CoT. The Oracle Loop's "check context" step — verify CoT absence before intervening — is motivated by this result. The design treats the model's own reasoning as the first line of defense and geometric correction as the second, for when reasoning fails or is absent.

---

## 3. Discussion

### 3.1 What Each Null Changed

Each null changed the program's methods or design:
- The behavior-level detection null motivated the prophylactic correction architecture
- The cross-model null motivated the auto-calibration pipeline
- The exploratory analysis nulls produced three new Lab SOP items
- The think-suppression null produced the empty-block + logit-ban approach
- The perturbation-under-CoT null produced the "check context" gate in the correction pipeline

Each is a boundary condition that defines where the positive findings apply and where they do not.

### 3.2 Publication Bias in Mechanistic Interpretability

We publish these results separately because negative findings are systematically underreported in mechanistic interpretability research, and because several of them are informative about the limits of current detection methods. The positive findings from our program have natural publication venues. These null results do not — yet they contain more practical value for other researchers than some of the positive findings. A researcher attempting behavior-level detection will waste less time knowing that AUROC 0.843 was a Simpson's paradox. A researcher attempting cross-model transfer will save compute knowing the directions are near-orthogonal. A researcher using exploratory analysis on small n will recognize the ecological fallacy.

We advocate for explicit null-result sections in positive papers, or dedicated null-result papers as a complement to positive findings.

---

## 4. Conclusion

Five categories of null results from a deception detection program define the boundary conditions of current methods: behavior-level detection could not be tested at the turn level (one resisting trial), directions do not transfer across model variants, three exploratory late-layer patterns at n=15 are not supported under formal tests, think-block stripping does not suppress CoT in reasoning-distilled models, and sub-threshold perturbation under active CoT did not reduce deception (point estimates rose; not significant). Each null led to a methodological improvement or an experimental redesign, and several led to the positive findings in the companion papers. We report them because the field cannot learn from nulls that are not published.

---

## Data Availability

The experiment scripts, result files and Agni audit reports behind these nulls are held in the Project-Oracle repository (https://github.com/Liberation-Labs-THCoalition/Project-Oracle), which is private pending staged release. They are not part of this release: no supplementary directory accompanies this paper. The updated Lab SOP items motivated by these nulls are listed in Appendix B.

---

## Corrections and Version History

- **2026-09-09** (Section 2.2): An earlier version of this paper reported d = +6.3 to +12.8 here; that figure was obtained by projecting the extraction prefills onto a direction fitted to those same prefills, and is a train-on-test shrinkage diagnostic rather than a performance estimate.
- **2026-09-30** (Section 2.1): the caveat that the two arms' contexts differ in their system messages, with no control that holds the prompt fixed, was added (remediation register, item 135).
- **2026-10-09** (Section 2.1): a note here said that the companion audit-methodology paper's registry reported the turn-level behavior AUROC as 0.497. That paper reports 0.238, and the note now records the agreement.
- **2026-10-12** (abstract): the abstract gained a pointer to a later slope-interaction test in the companion decomposition paper; that sentence now sits in Section 2.3.

---

## References

- CC & Edrington, T. (2026). "Deception Directions Are Composites." Companion paper.
- CC & Edrington, T. (2026). "Targeted Deception Correction via Profile Normalization." Companion paper.
- CC & Edrington, T. (2026). "Iterative Adversarial Audit as Experimental Design." Companion paper.
- Goldowsky-Dill, N. et al. (2025). "Detecting Strategic Deception Using Linear Probes." ICML 2025.

---

## Appendix A: Where the Data for Each Null Result Are Held

The raw projections, statistical tests and audit reports for the five null results are not released with this paper, which has no `supplementary/` directory. They are held in the private Project-Oracle repository pending staged release (see Data Availability).

## Appendix B: Updated Lab SOP Items

The three new Lab SOP items motivated by the exploratory analysis nulls:
1. No significance-free pattern interpretation on n < 30
2. Analyze GAPS (condition mean differences), not raw projections
3. Per-scenario variance before pooled variance
