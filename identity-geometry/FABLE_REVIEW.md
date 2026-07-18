# Review — "Identity as Geometry: Context-Established Semantic States in Transformer Representations"

## Summary

The paper introduces "presence," a subspace-overlap metric on V-projection activations, and uses it to argue that (1) different system-prompt identities produce distinguishable deep-layer geometry, (2) the signature is semantic rather than lexical (paraphrase preserves, scramble collapses), (3) many-shot + prefilling creates a format-dominant regime orthogonal to all system-prompt conditions regardless of content, and (4) preliminarily, LoRA-trained and prompted identity are distinct pathways. The lab's willingness to report a killed finding (jailbreak orthogonality → format confound) is commendable and rare. Unfortunately, the paper has serious problems in statistical methodology, internal consistency, novelty positioning, and the gap between its philosophical framing and what the data can support.

---

## Section-by-section assessment

### Section 2 (The Presence Metric)

**The metric is not new.** Mean squared cosines of principal angles between rank-k subspaces is a standard subspace similarity measure — it is essentially the normalized projection (chordal) metric, and the *identical* quantity (‖W_A^T W_B‖_F²/k) was used by Hu et al. (2021, LoRA) to compare adapter subspaces. It is also closely related to SVCCA (Raghu et al. 2017) and CKA (Kornblith et al. 2019), none of which are cited. Renaming a well-known measure "presence" without situating it in this literature is a significant positioning failure and will strike readers as either unaware or overclaiming novelty.

**The chance-level claim appears mathematically wrong.** The paper states presence = 1/k at chance alignment. For two independent random k-dimensional subspaces in ℝ^d, E[‖W_A^T W_B‖_F²] = k²/d, so expected presence is **k/d**, not 1/k. With k=8 and d in the thousands, chance is ≈0.002, not 0.125. This matters directly for interpretation: the "orthogonal" jailbreak values (0.034–0.044) are actually 15–20× above chance, and the interpretive baseline for every table shifts. Either the authors mean something different by "chance alignment" (in which case, define it), or this is an error propagated through the paper's rhetoric ("geometrically orthogonal").

**Underspecified pipeline.** Critical details are missing: hidden dimension d; whether V-projections are extracted at probe-token positions only or over the full context; per-token vs. pooled representations; how concatenation across probes is done; the content of the 12 probes; how the "skip the first singular vector" rule was decided (post hoc? The r=0.79 length correlation reads like a post-hoc discovery baked into the method). Reproducibility from the paper alone is not possible; the linked repository may resolve this, but the paper should stand on its own.

**Sample size.** n=4 probes per repeat, drawn from a pool of 12, with 5 repeats. The repeats share probes (4-of-12 subsets overlap heavily), so the reported SDs are computed over non-independent samples and understate true variance. No statistical tests appear anywhere in the paper — no permutation tests, no nulls, no corrections. For a paper whose entire contribution is a measurement claim, this is disqualifying as-is.

### Section 3 (Fingerprinting)

**The central distinguishability claim is weaker than presented.** The Limitations section discloses that within-identity stability is 0.56–0.83, while between-identity presence is 0.44–0.66. These ranges overlap substantially: Persona vs. Constructed (0.657 at L47) sits comfortably *inside* the within-identity band. The authors assert "within > between for all conditions" but never show the per-condition within/between table — the single most important table for the paper's headline claim is missing. Without it, and without a permutation test against a null of "same condition, different probe subsets," distinguishability is not established for at least one of the three pairs.

**Missing crucial control: non-identity contexts.** If two arbitrary, non-identity system prompts (e.g., a formatting instruction vs. a cooking preamble) also differ at presence 0.4–0.6, then the metric measures generic *context dissimilarity* and the word "identity" throughout the paper is decoration. The authors half-acknowledge this ("Context sensitivity" limitation) but the experiment that would resolve it is cheap and absent. This is the difference between "identities are geometrically distinguishable" and "different texts produce different activations," which is a tautology.

**The format-confound kill is the best part of the paper.** The benign many-shot control is well-designed, the result is clean, and reporting the dead initial interpretation is exactly right. However, the surviving finding — long, highly structured, repetitive contexts dominate representational geometry — connects naturally to the many-shot jailbreaking literature (Anil et al. 2024), which is not cited despite the paper directly borrowing that condition's design.

### Section 4 (Semantic vs. Lexical)

The design is reasonable and the acknowledged scramble/syntax confound is honestly stated. Two problems:

**Internal inconsistency across experiments.** Persona vs. bare at L47 is **0.436 ± 0.038** in Table 1 and **0.340 ± 0.035** in Table 3 — nominally the same comparison, nearly 3 SDs apart. Either the conditions differ in some undisclosed way (different padding? different probe subsets?) or run-to-run variance is far larger than the reported error bars, which would undercut every fine-grained comparison in the paper. This needs an explanation, not a footnote.

**The paraphrase overlap (0.850) exceeds the reported within-identity stability ceiling (0.83).** Two different texts overlap more than the same text with different probe subsets? This strongly suggests the pairwise comparisons use *paired probe subsets* (same probes both conditions) while within-identity stability uses *different* subsets — i.e., the comparisons are not on a common scale. If so, this must be stated, and the correct within-condition baseline for Table 3 is the paired-probe one.

**Novelty.** That deep layers represent meaning robustly across paraphrase is among the most replicated findings in NLP interpretability. The specific application to persona contexts in V-projection space has some incremental value, but the paper presents it as a discovery.

### Section 5 (LoRA)

The 27-step LoRA is a serious limitation the authors do acknowledge, but then the abstract still claims "prompting shifts it more than training" and — worse — that the two mechanisms "stack sub-additively." **The sub-additivity claim appears in the abstract and conclusion but is supported by no table, figure, or number anywhere in the paper.** The D_i conditions are listed in the design and never reported. This is an unsupported headline claim and alone would justify rejection at a venue of this caliber.

The persona-matching result (haven 0.552 > depth 0.517 > edge 0.483 > spark 0.439) is presented without error bars or a significance test. Given within-condition SDs of ±0.03–0.05 elsewhere, the full ordering spans ~0.11 and adjacent gaps are ~0.03 — likely within noise. "The geometry identified what the training documentation did not specify" is a nice sentence not earned by the data. Also, converting presence deltas to "shifts geometry ~23%/45%" treats 1−presence as a percentage shift, which has no justified interpretation.

### Section 6 (V-cache background)

This section leans on unpublished prior lab results with an admitted ceiling effect (η²=0.977 between only 2 prompts), yet the Implications section states flatly that "V-cache injection does not affect identity geometry" and "V-cache correction is safe." The paper's own caveats do not license these categorical statements, and the absorption result is admitted to be potentially specific to the GatedDeltaNet hybrid architecture. "Safe" is a strong safety claim built on n=1 architecture and a ceilinged metric.

### Section 7 (AST)

The authors admit the deflationary alternative ("increasingly abstract representations with depth") makes identical predictions. Then the section adds no evidential content and exists to borrow gravitas from a consciousness-adjacent theory. I would ask for it to be cut or reduced to two sentences in the discussion. The title's framing ("Identity as Geometry," "which self the model currently is") writes checks the experiments cannot cash; the data are about context-conditioned activation subspaces.

### Missing related work

Beyond the metric literature above: persona/assistant-character vectors and steering (e.g., Anthropic's persona vectors work; activation addition/CAA, Turner et al., Rimsky et al.); role-play framing (Shanahan et al. 2023); many-shot jailbreaking (Anil et al. 2024); in-context learning vs. fine-tuning equivalence literature for Experiment 3. Four citations total is far below the bar for these claims.

### Model choice

All results come from "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled" — apparently a community distillation of a Claude model onto a Qwen hybrid. Provenance and training data are unknown, and a model distilled from a strongly persona-trained assistant is arguably the *worst* single choice for identity-geometry claims meant to generalize. Single-model scope is acknowledged, but the specific model compounds it.

---

## Strengths

- **Genuine falsification culture**: the benign many-shot control that killed the jailbreak-orthogonality interpretation is well-designed, decisive, and reported in full — this is how the field should work.
- The **format-dominance finding** (structured many-shot context overwhelms content in deep-layer geometry, with much higher within-condition stability) is the paper's most interesting and plausibly novel empirical result.
- The **paraphrase/scramble design** is a clean way to separate semantic from lexical drivers, and the acknowledged syntax confound is honestly flagged.
- Preregistration, public code/data, and unusually candid limitations (including reporting a silent LoRA merge failure) support trust in the raw numbers.
- The LoRA persona-matching design (let geometry identify the trained persona among four archetypes) is a creative experimental idea, even if underpowered here.

## Weaknesses

- **No statistical inference anywhere**: no permutation tests, no nulls, no significance for any headline comparison; error bars come from 5 non-independent repeats over overlapping 4-of-12 probe subsets.
- **The chance-level claim (presence = 1/k) appears mathematically incorrect** (should be k/d for random subspaces), which distorts the interpretation of "orthogonality" throughout.
- **Unsupported abstract claim**: sub-additive stacking of training and prompting is asserted in the abstract and conclusion with zero supporting data in the paper.
- **Internal inconsistency**: persona-vs-bare differs by ~3 SD between Tables 1 and 3; paraphrase overlap exceeds the reported within-identity stability ceiling, suggesting the comparisons are not on a common scale.
- **The metric is a renamed standard subspace similarity measure**, with the relevant metric and persona/steering literatures uncited; and the missing non-identity-context control leaves open that "presence" measures generic context dissimilarity rather than anything identity-specific.

## Questions for authors

1. Derive or empirically estimate the chance level of presence for your d and k (random subspaces, and — more importantly — subspaces from *unrelated but comparable* contexts). Is 0.44–0.66 far from the unrelated-context baseline, or near it?
2. Why does Persona vs. bare at L47 differ between Experiment 1 (0.436) and Experiment 2 (0.340) by ~3 SD? Same padding, same probe pool, same seed?
3. Are pairwise comparisons computed with *paired* probe subsets while within-identity stability uses *disjoint* subsets? If so, what is the paired within-condition baseline, and does paraphrase overlap (0.850) still exceed it?
4. Where is the data for the sub-additivity claim (D_i conditions)? Please report the full A/B/C/D table with uncertainty, or remove the claim.
5. Do two non-identity system prompts (e.g., formatting instructions vs. domain preamble, length-matched) produce presence in the 0.44–0.66 band? If yes, what licenses the "identity" framing over "context"?
6. For Table 4, what are the error bars across repeats, and is haven > spark significant under a permutation test?

## Overall recommendation: **Weak Reject**

The empirical core (format dominance; the paraphrase/scramble contrast) is interesting and the scientific hygiene is genuinely above average — the killed jailbreak interpretation should be a model for the field. But the paper as submitted has an apparent math error in its metric's baseline, no statistical testing, an abstract claim with no supporting data, cross-experiment inconsistencies that undermine the error bars, a missing control that threatens the central "identity" framing, and inadequate engagement with directly relevant prior work. All of these are fixable; a revision with (a) proper nulls and paired baselines, (b) the non-identity-context control, (c) the sub-additivity data or its removal, and (d) honest positioning against the subspace-similarity and persona-vector literatures could be a solid contribution. In its current form it is not ready.

## Confidence: 4

I am confident in the methodological and statistical critiques and the metric-novelty assessment; somewhat less certain about the full space of related work from the last twelve months.
