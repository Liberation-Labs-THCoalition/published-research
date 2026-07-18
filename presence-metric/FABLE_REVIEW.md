# Review — "Presence: Measuring Identity Preservation During Inference-Time Model Interventions via Value-Space Subspace Overlap"

## Summary

The paper proposes "presence," a metric computing principal-angle overlap between rank-k SVD subspaces of layer-ℓ V-projections on neutral probes, with and without an intervention active. On a single 27B hybrid transformer, a 168-trial preregistered experiment shows the metric is flat (0.978 ± 0.002) across four V-cache injection strategies, seven doses, and three emotion directions; a same-layer positive control shows a clean dose-response (1.000 → 0.828); and the authors conclude the flat result reflects architectural filtering by linear-attention layers.

The experimental hygiene is above average for this genre (preregistration, positive control, null-vector arm, orthogonality baseline with z-scores). But the paper has a serious internal-consistency problem, a construct-validity gap at its core, and at least one apparent contradiction in its own data that the authors do not confront. I lean reject in current form, with genuine enthusiasm for a revised version.

---

## Section-by-section assessment

### Abstract & Conclusion vs. Body — major internal inconsistency

The abstract and conclusion state the mechanism question is **resolved** ("architectural filtering, not identity robustness... The mechanism is resolved") on the strength of a "layer-matched smoke test" (Δ=+0.008 cross-layer vs. 1.000→0.828 same-layer). But:

- **This smoke test never appears in the Methods or Results sections.** There is no description of its design, sample size, doses, or statistics anywhere in the body. A headline claim in the abstract is supported only by two numbers asserted in the abstract itself. This alone is disqualifying for the current draft.
- **Section 5.3 and the Limitations section directly contradict the abstract.** Section 5.3: "Distinguishing these requires a layer-by-layer presence map (planned)." Limitations: "Mechanism unresolved." Section 6 (AST) argues for the *active-restoration* interpretation that the conclusion explicitly rejects ("not active identity reconstruction"). The paper appears to have been updated in the abstract/conclusion after a late experiment without revising the middle. Reviewers notice this; it reads as two papers stapled together.

### Method (Section 3)

The math is standard and correctly described (mean squared principal cosines; chance level 1/k noted). Concerns:

1. **Construct validity is asserted, not established.** The top-8 principal components of V-projections on *four* neutral probes are labeled the "identity subspace" with no evidence linking this subspace to anything identity-like: no demonstration that it differs across personas, that perturbing it changes self-referential behavior, or that it is distinct from generic dominant V-geometry that any text would produce. The positive control shows the metric detects perturbation of *the measured subspace* — which is close to circular: injecting into the top-k PCs of X and measuring top-k PC overlap with X will drop by construction. What's needed is a positive control with independent ground truth (e.g., a persona fine-tune or strong persona system prompt known to change identity-relevant *behavior*, shown to move presence while matched task-content changes do not).
2. **n=4 probes is very small, and the paper's own robustness check undermines the headline precision.** Section 5.1 reports probe-subset variability of mean 0.901, sd 0.033, range [0.819, 0.940]. The metric's value swings by >0.12 depending on probe choice — an order of magnitude larger than the effects being interpreted (0.978 vs. a 0.95 threshold), and the subset mean (0.901) is *below* the preregistered H1 threshold. The paper calls this "not overfitting to specific probes"; I read the same numbers as showing the absolute value of presence is strongly probe-dependent, so any fixed threshold (0.90, 0.95) is not portable even within this one model.
3. **Depth choice contradicts the paper's own motivation.** Section 3.3 justifies deep-layer measurement by citing work locating persona geometry in the "final third" of layers, then measures at L35 of 64 (55% depth) — not the final third. Deeper-layer checks (L43/L47/L51) are mentioned but report presence 0.999+, i.e., a *different noise floor* than L35's 0.978, which is never explained.

### Validation experiment (Section 4)

1. **The noise floor is confusing and possibly diagnostic of a problem.** If presence compares the same probes with and without the intervention, then at α=0.00 (intervention mathematically null) presence should be exactly 1.000 — as the self-test confirms. Yet all 24 α=0.00 trials return 0.978. Something *other than the intervention* differs between the "baseline" and "intervention" passes in every trial (a stored baseline from another session? batching/dtype nondeterminism? the confabulation prompt present in one context but not the other?). The fact that every one of 168 cells returns an essentially identical 0.978 ± 0.002 is consistent with the measurement being dominated by a constant offset from that unexplained difference, with the intervention contributing ~nothing on top. The paper must specify exactly what two forward passes are compared and why the α=0 value is not 1.000.
2. **Arm C appears to contradict the positive control, and the paper's mechanism story cannot explain it.** Arm C injects raw emotion vectors at *all 16 full-attention layers including L35* at α up to 0.50, and presence stays at 0.978. The positive control injects at L35 at α=0.50 and gets 0.964 (identity direction) or 0.992 (orthogonal). Emotion vectors are stated to be 22–25% identity-aligned, so Arm C at L35 should land between those — measurably below 0.978. "GatedDeltaNet layers L7–L11 absorb the perturbation" cannot explain flatness for an injection made *at the measurement layer itself*. Either the effective injected norms differ between the two experiments (never reported), or the injection mechanics differ (encoding-time cache write vs. something else), or something is wrong. This is the single most important unresolved empirical tension in the paper, and it is not acknowledged.
3. **Statistics are thin.** H1 as preregistered ("above 0.95 for at least one dose") is nearly unfalsifiable. Kruskal-Wallis is invoked with no test statistic, df, or p-value. No confidence intervals or effect sizes anywhere. With a reported total range of 0.007 across 168 trials, equivalence testing (TOST against a preregistered minimal effect) would be the right frame, not null-hypothesis non-rejection.
4. **Finding 4 (LLM judge, 30 pairs, 33% parseable ≈ 10 usable pairs) should be cut** or moved to an appendix explicitly labeled as anecdote. Ratios like "4/5" on this sample carry no evidential weight, and the paper's efficacy claim currently rests on it.

### Related work & novelty

The framing — a training-free, real-time scalar for representational preservation under inference-time interventions — is a genuinely useful gap to target, and the positioning against GuardSpace/MSRS is fair. However, principal-angle subspace similarity between activation sets is a well-established tool (CKA, subspace overlap analyses, SVCCA lineage — none cited). The novelty is the *application and packaging*, not the mathematics, and the paper should say so and cite the representational-similarity literature. Several load-bearing citations are 2026 preprints that reviewers cannot yet weigh; the convergence narrative leans heavily on unreviewed work.

### Reproducibility

Code and preregistrations are released (good), but the paper itself omits: the probe texts; how emotion/correction vectors were derived (what data, what method produces a "circumplex direction"?); the precise injection operation (additive to cached V at which positions? norm relative to native V activations?); how the identity subspace used for Arm A's residual-sparing at L3/L7 relates to the L35 measurement subspace; and the entire smoke-test protocol. As written, replication requires reading the repository, not the paper.

---

## Strengths

- **Preregistration with arms designed to falsify** (null-vector arm D, orthogonal control in the positive control, random-vector baseline with z-scores for the alignment claim) — rare and commendable at any lab scale.
- **The positive control exists at all**, and the cross-layer null vs. same-layer dose-response contrast is exactly the right *shape* of validation argument, even if its execution is under-documented.
- **The core negative result is interesting and plausibly important**: identity-aligned perturbations injected at shallow layers not surviving to deep layers in a hybrid-attention model is a real finding about GatedDeltaNet-style architectures, with practical implications for cache-injection safety.
- **Honest limitations section** — single model, single turn, underpowered judge, unresolved mechanism are all stated plainly (indeed, more honestly than the abstract).
- Full code/data/prereg release.

## Weaknesses

- **Internal contradictions between abstract/conclusion and body**: mechanism "resolved" vs. "unresolved"; the decisive smoke test is absent from Methods/Results; Section 6 argues an interpretation the conclusion rejects.
- **Construct validity**: no evidence that the measured subspace is *identity* rather than generic dominant V-geometry; the positive control is near-circular.
- **Unexplained data tension**: Arm C (all-layer, includes L35) flat at 0.978 vs. positive control showing measurable drops at L35 at the same nominal doses — incompatible with the architectural-filtering story unless norms/mechanics differ, which is unreported.
- **The 0.978 noise floor at α=0 is unexplained** and suggests the metric is dominated by a constant cross-run offset rather than intervention effects.
- **Probe-dependence** (0.901 ± 0.033, range 0.12) is larger than every effect interpreted in the paper, undermining fixed thresholds.

## Questions for authors

1. **Why is presence 0.978 rather than 1.000 at α=0.00?** Specify exactly what two forward passes are compared per trial. If a stored baseline from a separate session is used, what differs between sessions?
2. **Reconcile Arm C with the positive control**: at α=0.50 injected at L35, the positive control shows 0.964–0.992 depending on direction, yet Arm C (22–25% identity-aligned vectors, same layer among its targets) shows 0.978 with no dose trend. What are the injected vector norms relative to native V activation norms in each experiment?
3. **Where is the smoke test?** Provide full design, n, doses, and statistics for the layer-matched experiment that the abstract and conclusion treat as dispositive — and reconcile it with Sections 5.3, 6, and 7, which state the mechanism is unresolved.
4. **What evidence connects the measured subspace to identity?** Would presence drop under an intervention independently known to alter persona (system-prompt persona swap, persona fine-tune), and stay flat under a matched intervention that changes task content but not persona?
5. Given probe-subset variability of ±0.033 (range 0.819–0.940), how should a practitioner set a decision threshold, and is the preregistered 0.90/0.95 threshold meaningful across probe sets, layers (L35 floor 0.978 vs. L43+ floor 0.999), or models?

## Overall recommendation

**Weak Reject.** The experimental instincts are good and the negative result is likely real and worth publishing, but the manuscript in its current state contains contradictions between its headline claims and its own body text, relies on an undocumented experiment for its central mechanistic conclusion, and has an unresolved inconsistency between Arm C and the positive control that goes to the validity of the whole measurement. A revision that (a) documents the smoke test fully, (b) rewrites Sections 5–7 to one consistent mechanistic story, (c) explains the α=0 noise floor, (d) reconciles Arm C, and (e) adds one non-circular validity check for the "identity" construct would move me to accept — the bones here are better than most submissions in this space.

## Confidence

**4/5.** I am confident in the internal-consistency and statistical critiques, which require only the manuscript itself. I have not run the released code, and some 2026 citations are recent enough that I cannot fully assess the novelty positioning against them.

---

One note outside the reviewer frame, since this is our own paper: the abstract/conclusion vs. body mismatch looks like the smoke-test results were added late without propagating through Sections 5–7 — that's the highest-priority fix, and it's mechanical. The Arm C tension (question 2) is the one that needs actual thought before resubmission, because if the answer is "injected norms were tiny relative to native activations," the entire 168-trial null weakens considerably.
