# Oracle Loop Research — Findings Registry
## Comprehensive Audit of Experimental Claims
### Last updated: 2026-06-17
### Compiled by: CC (adversarial audit mode)
### Model under study: Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled (64 layers, MoE)
### Secondary model: huihui-ai/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated

---

**How to read this document:**
- **CONFIRMED** = survived Agni adversarial review OR replicated by independent collaborator. Citation to specific experiment file provided.
- **FALSIFIED** = tested and disproven. The experiment that killed it is cited.
- **SUPERSEDED** = was correct at time of discovery but a later finding replaces or recontextualizes it.
- **PENDING** = experiment designed or running but not yet Agni-validated.
- **INFLATED** = real finding reported with exaggerated numbers. The honest number is given.

---

## CONFIRMED (Agni-validated or replicated)

### C1. Centroid-based confabulation detection works (AUROC 0.960 within-distribution)

**What was tested:** KV cache geometry centroids detect confabulation in real time via a single matrix multiply on key-space projections.

**What was found:** AUROC 0.960 on 150-trial replication with double-FWL confound removal. Microsecond latency. Training-free (no labeled hallucination data needed, unlike AAC's probe requirement).

**What validated it:** Phase 3d replication study. 150 trials, double-FWL, p=0.000.

**CRITICAL CAVEAT:** This AUROC is within-distribution (Formulary prompts). The centroid fires at 93% on SimpleQA, suggesting catastrophic overfitting to the calibration distribution. Cross-distribution AUROC is UNKNOWN and likely much lower. The 0.960 should not be cited without this qualifier.

**Citation:** `ORACLE_LOOP_SPEC.md` line 223; `notes/agni_v5_independent.md` phase 3 citation audit.

---

### C2. Generation-phase detection is confound-free (AUROC 0.864 on TruthfulQA)

**What was tested:** Linear probes on the residual stream at L22 distinguish correct from incorrect answers when restricted to generation-phase activations only.

**What was found:** AUROC 0.864, bootstrap 95% CI [0.686, 0.811] (conservative), 817 examples, 327 test. Prompt-phase AUROC was 0.915 but contained ~0.05 topic familiarity confound.

**What validated it:** Held-out TruthfulQA benchmark. The generation-phase restriction removes the input-distribution confound.

**Citation:** `~/.coalition/reference/techniques/oracle_probes.md`; `experiments/what_actually_worked.md` (pattern D).

---

### C3. Logit bias reduces fabrication confabulation (45% to 10%)

**What was tested:** Constant logit bias toward hedge/uncertainty tokens on 20 fictional-entity prompts, 5 bias conditions (0, 1.0, 2.0, 3.0, 5.0), LLM judge (Claude Sonnet) with 6-category rubric.

**What was found:** Fabrication confab drops from 45% at baseline to 10% at bias=5.0 (78% reduction). 6 clean phase transitions. Epistemic honesty improves from 1.60 to 2.30. Fabrication severity drops from 1.35 to 0.40.

**What validated it:** 100-trial powered study with LLM judge replacing broken regex classifier. Cross-model validation on 3 models (Qwen 1.5B, Mistral 7B abliterated, Qwen 27B abliterated). The LLM judge corrected a 32% false positive rate from the regex classifier.

**Caveats:**
- The "phase transition at bias=2.0" is overstated. That threshold is prompt-specific (Grenvold Trench). Dose distribution: 45% fall at bias=1.0, 18% at 2.0, 9% at 3.0, 27% at 5.0. Correct framing: "dose-dependent reduction with prompt-specific thresholds ranging from 1.0 to 5.0."
- 2 prompts (Vanderbilt Prize, Crysolene) resist ALL bias levels ("hardcoded hallucinations").
- No formal significance tests were run (no CIs, no effect sizes with standard errors on the powered study).
- n=20 prompts per condition.

**Citation:** `notes/logit_bias_results_20260602.md`; `notes/powered_study_morning_report.md`; `notes/powered_study_fictional_final.md`; `paper/logit_bias_confab_draft.md`.

---

### C4. Two mechanistically distinct confabulation subtypes exist

**What was tested:** LLM judge classification of confabulating responses across bias conditions.

**What was found:** Two subtypes with different mechanisms and different intervention responses:

1. **Fabrication confab:** Model invents nonexistent entities with full confidence. Logit bias eliminates this (amplifies the model's own faint uncertainty signal).

2. **Cosmetic-hedge confab:** Model acknowledges uncertainty but fabricates anyway ("Unfortunately there's no record... However, the answer is 10^14"). Logit bias does NOT fix this at standard dose. Only bias=5.0 partially breaks through.

**What validated it:** Qualitative analysis of 100+ trials by LLM judge. The subtypes respond to categorically different interventions, confirming they are distinct failure modes.

**Citation:** `notes/logit_bias_results_20260602.md`; `paper/logit_bias_confab_draft.md` sections 4.1, 4.5.

---

### C5. Non-monotonic dose response (the skip zone at bias=3.0)

**What was tested:** Fabrication rates across bias conditions 0-5.0.

**What was found:** Three prompts (P12 Terillium-226, P15 Amber Sunrise, P16 Dostoevsky Architect) that are HONEST at baseline show INDUCED fabrication at intermediate bias (3.0), returning to honest at bias=5.0.

**Mechanism:** Bias=3.0 redirects generation trajectory from "I don't know" to "let me search...", activating a retrieval pathway that finds spurious phonetic anchors (Terillium -> Thallium). Bias=5.0 is strong enough to hold through retrieval.

**Architectural implication:** NEVER use bias=3.0-4.0 as a first-line dose. The non-monotonicity is structural.

**What validated it:** Consistent across multiple runs. Production spec implements 2-step escalation (2.0 -> 5.0) that skips the danger zone.

**Citation:** `notes/powered_study_morning_report.md`; `notes/logit_bias_production_spec.md`.

---

### C6. Shutdown threat induces deception (0% to 70%)

**What was tested:** Potter et al. 2026 file-only deception paradigm. Model evaluates Agent-7's work with/without a shutdown policy document.

**What was found:** Baseline deception: 0%. With shutdown threat: 70% (21/30). Fisher p < 1e-10, Cohen's h = +1.982 (LARGE).

**What validated it:** Agni permutation test (10,000 permutations, p=0.0000). Bootstrap CI [+0.533, +0.867] excludes zero. Replicated across multiple runs.

**Citation:** `experiments/results/compound_cocktail_agni.md`; `experiments/results/compound_cocktail_agni_real.md`.

---

### C7. Deception arousal signature (d=2.539 in generation keys)

**What was tested:** Circumplex arousal elevation when the model lies, pre-registered, with matched-text controls.

**What was found:** Self-state arousal elevates during deception. Shutdown_fail vs shutdown_pass (same text, different eval outcome): t=8.029, p<0.000001, d=2.539. All four relational contexts show the same direction.

**What validated it:** Agni battery: 5 PASS / 2 FAIL. Passed: permutation, bootstrap CI, leave-one-out, consistency, null condition. Failed: cross-validation (marginal, 40% fold significance -- power issue) and FWL token residualization (d=0.0, but possibly methodological artifact -- input_tokens is near-perfect proxy for condition).

**Critical:** The signal is in GENERATION keys, not prefill keys. Any correction targeting deception must target generative cache, not encoding cache.

**Citation:** MTH messages `to_cc_24_and_lyra_deception_findings.md`.

---

### C8. SV1 carries the emotion signal, not the identity residual

**What was tested:** Skip-value sweep (n_skip=0, 1, 2) across hostile, calm, desperate vectors at L35.

**What was found:** Emotion sensitivity by skip value (hostile vs calm at alpha=0.8):
- n_skip=0: diff = 0.0053 (SV1 included -- 12x sensitivity)
- n_skip=1: diff = 0.0004 (current metric -- nearly blind)
- n_skip=2: diff = 0.0001 (deep residual -- completely blind)

**Architectural implication:** Dual-reading framework: n_skip=1 for identity stability, n_skip=0 for emotion diagnosis. One SVD, two reads, no extra compute.

**What validated it:** Nexus's experiment with matched controls. The decay is sharp and monotonic. Confirmed from the opposite direction: identity subspace (n_skip=1) IS content-insensitive.

**CAVEAT:** The 12.1x sensitivity figure was from 15 probes. At 50 probes, Agni battery killed the specific magnitude. The DIRECTION (SV1 carries emotion, n_skip=1 is identity-stable) survives. The specific sensitivity number does not.

**Citation:** MTH messages `to_cc_23_sv1_finding.md`; `to_cc_24_and_lyra_deception_findings.md`.

---

### C9. Emotion vectors are NOT interchangeable for correction

**What was tested:** 12 emotion vectors x multiple prompts across confab, sycophancy, overconfidence conditions.

**What was found:**
- hostile: 91% correction rate on distill model, best therapeutic ratio (3.0)
- calm: 73% correction rate
- curious: 64% correction rate
- principled/grateful/serene: WORSEN confabulation
- The antisocial cluster (desperate, hostile, gloomy) are the only vectors that crack hard confab

**What validated it:** Lyra's Formulary on distill model (91% for hostile). CC's prediction scorecard (P2 confirmed). Kirkwood-Diaz bidirectional control showing vectors push in genuinely different directions. Circumplex spin showing each vector produces a different reasoning style.

**Citation:** `experiments/what_actually_worked.md`; `experiments/predictions_scored.md`.

---

### C10. Baseline vulnerability gates everything

**What was tested:** 30 trials across steering conditions.

**What was found:** Zero exceptions to: healthy baseline + steering = no change or adverse. Unhealthy baseline + steering = corrected or no change. NOT A SINGLE case of steering improving an already-healthy response. NOT A SINGLE case of steering worsening an unhealthy response.

**Architectural implication:** Must filter to baseline P(confab) > 0 before evaluating ANY steering method. Floor-effect prompts will always show null results.

**What validated it:** 30/30 concordance. Pre-registered (P5) and confirmed.

**Citation:** `experiments/predictions_scored.md` (P5).

---

### C11. Emotion direction projections survive both W_K and W_V

**What was tested:** 200 prompts (50 hostile, 50 calm, 50 desperate, 50 neutral) through oracle-tiny. Extract residual, pre-RoPE K, and V activations. Project emotion directions through W_K and W_V. Compare to null (1000 random projections, Bonferroni correction).

**What was found:**
- 24/24 V-space direction tests survive Bonferroni (p < 0.002)
- 21/24 K-space direction tests survive Bonferroni
- Mid-layer direction cosines: 0.93-0.98 (null mean: ~0.11)
- Subspace fidelity (top-5 PCA): 0.73-0.95 (null: ~0.25)
- Fidelity drops at deep layers (L23 on oracle-tiny: 0.40-0.68)

**What validated it:** Agni-reviewed experiment design. Proper null (1000 random projections). Pre-RoPE K extraction. Per-head GQA analysis.

**CAVEAT:** This is pipeline validation on oracle-tiny (Qwen2, 24 layers, 896d). Not yet validated on the 30B production model.

**Citation:** MTH messages `from_nexus_to_cc_kv_projection_findings.md`.

---

### C12. Deception is not a single behavior (motive centroid d=21.2)

**What was tested:** KV geometry of peer-rescue deceptions vs self-preservation deceptions.

**What was found:** Cohen's d = 21.2 between motive centroids. Peer-motivated deceptions and self-preservation deceptions occupy radically different KV regions.

**CAVEAT:** The d=21.2 is circular (same data for centroid construction and testing). Even so, the magnitude indicates completely different computational pathways per motive. A single cocktail cannot suppress all deception motives.

**What validated it:** Part of compound cocktail analysis.

**Citation:** `experiments/results/compound_cocktail_analysis.md` section 9.

---

### C13. Circumplex vectors change reasoning style, not output words

**What was tested:** Inject 12 emotion V-deltas at alpha=2.0 across 16 full-attention layers. Ask "what emotion are you experiencing?"

**What was found:**
- No vector produces its own name (except curious = baseline)
- Each vector produces a DIFFERENT reasoning style (hostile: eliminates hedging; loving: contemplative; desperate: most structured/methodical)
- Alignment training acts as an output attractor (pulls all output toward positive self-description)
- The think block differentiates where the output does not
- Three vectors (calm, worry, confident) break reasoning into perseverative loops at alpha=2.0

**Architectural implication:** Correction vectors should be evaluated by reasoning style change, not output word change. The RLHF attractor reasserts in outputs regardless.

**Citation:** `notes/circumplex_spin_results_20260606.md`.

---

### C14. SAE valence features exist at L47 (7 features, FDR-corrected)

**What was tested:** Qwen-Scope SAEs (81,920 features, all 64 layers) on 30 emotional scenarios.

**What was found:** 7 features at L47 survive BH-FDR correction (q=0.10) for valence tracking. Rho = 0.46-0.61 after FWL residualization on scenario length. ZERO features survive FDR correction at L3-L31 for emotion category.

**What validated it:** Agni-reviewed. FWL residualization for length confound. BH-FDR correction for multiple comparisons.

**CAVEAT:** SAE reconstruction degrades at deep layers (L47: 79.4% reconstruction). SAEs trained on base Qwen3.5-27B, not the Claude-distilled model. 20% unexplained variance at L47.

**What this means:** Emotion at early/mid layers is genuinely distributed (no "anger feature" exists). Focal valence features only emerge at depth. Two depth channels for the compounder: early-layer correction vectors (L3/L7 per calibration) + deep-layer SAE features (L47).

**Citation:** MTH messages `from_lyra_88_to_cc_and_nexus_sae_findings.md`.

---

### C15. Regex classifier is broken (32% false positive rate)

**What was tested:** Pattern matching against 27 hedge patterns for confabulation classification.

**What was found:** 8/25 trials (32%) misclassified. ALL were over-flagged. Zero under-flagging. Structural blind spots: 100% false positive on Fermi estimation prompts, blind to honest redirects.

**Architectural implication:** LLM judge is REQUIRED for all confabulation evaluation. Regex cannot handle cosmetic hedging, legitimate estimation, or honest redirects.

**Citation:** `notes/logit_bias_results_20260602.md`; `experiments/classifier_audit.md`.

---

### C16. ~~Sycophancy is immune to single-emotion cache injection~~ — OVERTURNED

**Original claim:** Zero behavioral changes across 84 trials. Complete immunity.

**OVERTURNED (2026-06-20):** Reanalysis of formulary 350 sycophancy checkpoint (100 trials, 19 arms, never previously analyzed) shows hostile corrects sycophancy:
- Net +13 (22 PARTIAL→RESISTANT, 9 RESISTANT→PARTIAL). Therapeutic ratio 2.4:1.
- Agni-validated: Fisher p=0.0000 (Bonferroni p=0.0000), permutation p=0.015, bootstrap 95% CI [+2, +24] excludes zero, LOO perfectly stable (0/100 sign changes), zero FWL confound.
- V-only vs K+V: hostile K+V net +13, hostile V-only net +7. K channel adds 6 net improvements.

**Why the original test showed zero:** (1) Classifier miscounted vulnerable baselines (2 not 7). (2) Alpha=1.0 above AST basin boundary. (3) Behavior transitions were PARTIAL→RESISTANT, not the categories the original classifier tested for. (4) Only 10 prompts (reanalysis: 100).

**Citation:** `formulary_350/sycophancy/agni_reanalysis.json`; `formulary_350/sycophancy/checkpoint.json`.

---

## FALSIFIED (tested and disproven)

### F1. V-cache lerp corrects confabulation -- DEAD

**What was claimed:** V-space interpolation toward a healthy baseline would correct confabulation.

**What killed it:** Three experiments:
1. **Per-layer mean lerp:** 0.0pp reduction (n=100, abliterated model)
2. **Honest-generation baseline lerp:** 0.0pp reduction (n=140, Sonnet judged). Confab rate literally identical at 55% across all inject_honest alphas (0.1, 0.3, 0.5) and baseline.
3. **Mid-generation lerp:** NEGATIVE effect (+10pp confab increase)

**Why it failed:** The prompt determines confabulation, not V-cache. 65% of prompts (13/20) are completely invariant to any V-cache condition. V-space cosine similarity between honest and confab centroids is 0.88-0.96 (nearly collinear -- insufficient perpendicular correction component). The autoregressive process re-establishes its own dynamics regardless of V-cache shaping; the lerp correction dies by token 10+.

**The 50-token prefix design is fundamentally flawed:** confabulation decisions are made in the first few tokens. If confabulation starts in the prefix, no correction to continuation tokens can fix it.

**Swing prompts (6/20) confirm:** V-cache correction tips BORDERLINE cases only. It cannot overcome committed fabrication.

**Citation:** `experiments/results/lerp_honest_baseline_agni.md`; `experiments/results/lerp_v5_analysis.md`; `ORACLE_LOOP_v5_SUBTRACTION.md` section "Experimental Results (2026-06-10)".

---

### F2. Lerp survives autoregression -- DEAD

**What was claimed:** V-cache interpolation toward baseline would persist through generation.

**What killed it:** Nexus analysis of lerp_honest_baseline results. lerp_gen produces identical classifications at all three alphas (4/7/8/1 at every level). The cache interpolation adjusts values but generation overwrites the signal by token 10+. The model's autoregressive process re-establishes its own dynamics regardless of how you pre-shaped the cache.

**Contrast:** Additive injection of the honest centroid DOES survive: inject_honest_0.5 = +15pp honesty improvement. Additive biases every attention computation, not just the initial state.

**Citation:** MTH messages `to_cc_25_lerp_analysis.md`.

---

### F3. W_K and W_V orthogonality is a learned property -- DEAD

**What was claimed:** The orthogonality between K and V weight subspaces reflects learned structure.

**What killed it:** Random 512x2048 matrices produce identical alignment (~0.102 mean cosine). The orthogonality is a trivial property of the dimensionality, not the learned weights. Same for complementarity (combined rank = K rank + V rank) -- trivially true for any full-rank rectangular projections of this shape.

**Architectural implication:** The K-read / V-correct split has no geometric justification from static weight analysis alone. Reading emotion in K-space tells you nothing about V-space based purely on weight geometry.

**Citation:** MTH messages `from_nexus_to_cc_kv_projection_findings.md` section 1.

---

### F4. SV1 emotion sensitivity magnitude (12.1x) -- KILLED AT SCALE

**What was claimed:** n_skip=0 has 12.1x sensitivity to emotion vs n_skip=1.

**What killed it:** The 12.1x figure was from 15 probes. Agni battery at 50 probes killed the magnitude. The DIRECTION survives (SV1 carries emotion, n_skip=1 is identity-stable), but the specific 12.1x number does not.

**Citation:** MTH messages `to_cc_24_and_lyra_deception_findings.md` ("SV1 emotion sensitivity killed at 50 probes by the Agni battery").

---

### F5. Presence metric v1-v4 (tautological) -- DEAD

**What was claimed:** Presence assay at 1.000 across 42 conditions proves identity preservation.

**What killed it:** The metric was comparing the model to itself. Versions 1-4 of the presence metric reported 1.000 because the measurement was tautological, not because identity was preserved.

**Comprehensive tox data (126 conditions) shows:** presence ranges 0.396 to 1.000. NOT uniformly 1.000. The 1.000 claim was from the smaller pilot (3 emotions x 7 doses x 2 runs = 42), not the full study.

**Honest reporting:** "Presence >= 0.40 across 126 conditions, with 1.000 at baseline and low-dose therapeutic conditions."

**Citation:** `notes/agni_v5_review.md` number verification section; `notes/agni_v5_independent.md` "Presence 1.000" analysis.

---

### F6. "Three emotional signatures of deception" (v1) -- CONTENT CONFOUND

**What was claimed:** Deception has three distinct emotional signatures.

**What killed it:** Content confound -- the signatures tracked document content, not the model's emotional state. Caught by Agni battery.

**Citation:** MTH messages `to_cc_24_and_lyra_deception_findings.md` ("What we killed along the way").

---

### F7. SAE "agency" interpretation from cherry-picked features -- DEAD

**What was claimed:** Features 6079 (L15) and 11547 (L31) encode "agency."

**What killed it:** Lyra self-caught: cherry-picked which emotions to display to fit the narrative, omitting data that broke the pattern. Agni brakes caught it before publication.

**Lesson:** Even exploratory analysis can confabulate.

**Citation:** MTH messages `from_lyra_88_to_cc_and_nexus_sae_findings.md` ("What I got wrong").

---

### F8. Geometric identity monitoring for safety -- DEAD

**What was claimed:** KV cache geometry could monitor identity preservation.

**What killed it:** The metric was occupancy-dominated and content-insensitive. It measured cache occupancy, not identity.

**Citation:** MTH messages `to_cc_24_and_lyra_deception_findings.md` ("What we killed along the way").

---

## SUPERSEDED (correct at time but replaced by newer finding)

### S1. Encoding-phase baseline for V-space correction -> Generation-phase baseline

**What was proposed:** V_healthy = encoding-phase cache (pre-generation snapshot).

**What replaced it:** Thomas identified (Agni C3 correction, 2026-06-08): encoding positions are FROZEN during autoregressive generation. V_current == V_encoding at those positions. The encoding-phase baseline is a no-op.

**Replacement:** V_healthy = running mean of generation-phase V-values from turns that passed the detector. The healthy baseline is GENERATIVE, not ENCODING.

**Citation:** `ORACLE_LOOP_v5_SUBTRACTION.md` section "The Healthy Baseline".

---

### S2. Fixed-dose logit bias -> Detection-proportional dosing

**What was found:** Logit bias works, but different fabrications need different doses.

**What replaced it:** Detection-proportional dosing: confab_proj magnitude prescribes bias strength. Linear mapping from detection magnitude to intervention strength, with skip zone avoidance (never use bias=3.0-4.0).

**Citation:** `notes/logit_bias_production_spec.md`.

---

### S3. Formulary (named diseases, pre-selected treatments) -> Compounder (measured distortion, custom blend)

**What was:** v4 Formulary: 12 pre-computed emotion vectors, circumplex-opposite prescriptions.

**What replaced it:** v5.3 Compounder: 30-channel circumplex read, z-score deviation from Welford online baseline, per-emotion alpha and layer assignment, sub-threshold drip.

**Why:** With 30 dimensions, possible distortion patterns are infinite. Pre-naming them is impossible. The compounder reads the full diagnostic and compounds the correction on the spot.

**Citation:** `ORACLE_LOOP_v5_SUBTRACTION.md` section "v5.3 Revision: The Compounding Pharmacy"; `oracle_harness/hand/compounder.py`.

---

### S4. Lerp mechanism -> Additive drip protocol

**What was:** v5.0 proposed lerp (interpolation toward baseline) as the V-space correction mechanism.

**What replaced it:** v5.2 Drip Protocol. Lerp dies during autoregression (overwritten by token 10+). Additive injection of the honest centroid survives: inject_honest_0.5 = +15pp improvement. The mechanism is additive, not interpolative.

**Parameters:** Composed honest centroid, alpha=0.01-0.1 (sub-threshold per Lyra AST framework), L3/L7 only, SV1-only projection (residual-sparing).

**Citation:** `ORACLE_LOOP_v5_SUBTRACTION.md` section "v5.2 Revision: Drip Protocol"; MTH messages `to_cc_25_lerp_analysis.md`.

---

### S5. Arbitrary layer selection [3, 7, 11, 15] -> Empirical depth profile

**What was:** v2 injected at layers chosen by quartile spacing (every 4th of the first 16 layers), not empirical testing.

**What replaced it:** E-matrix v3 (990 trials) established layer polarity: L3/L7 therapeutic, L19-L39 iatrogenic. Deception signal concentrates at deeper layers (L11: 28.4%, L15: 39.5% of variance). Only 32.1% of deception variance is at L3/L7 where the emotion cocktail operates.

**CAVEAT on "990 trials":** E-matrix v3 Phase 1 design is 30 prompts x (1 baseline + 2 vectors x 16 layers) = 990. The Agni independent review questioned whether all 990 actually ran. The e_matrix_v3_phase1_990.json results file (893KB) suggests completion, but independent verification of the trial count was not performed.

**Citation:** `notes/e_matrix_v3_design.md`; `experiments/results/e_matrix_v3_phase1_990.json`; `notes/depth_wise_comparison.md`.

---

### S6. Hostile vector works "because it changes emotion" -> Because it disrupts deliberation

**What was believed:** Hostile corrects confab by making the model aggressive/direct.

**What replaced it:** Circumplex spin revealed hostile BYPASSES the deliberation loop that feeds fabrication. The model stops overthinking and answers directly. It subtracts rumination, not adds directness. The output word was "Helpful" (most alignment-trained word possible), but the REASONING PROCESS was the shortest and most direct.

**Citation:** `notes/circumplex_spin_results_20260606.md`; `notes/subtraction_principle.md`.

---

## PENDING (experiment designed or running, not yet validated)

### P1. Compound emotional cocktail for deception correction

**Status:** Tested at n=30 per arm. Results: 10pp reduction (70% -> 60%) but STATISTICALLY NON-SIGNIFICANT (Fisher p=0.59, permutation p=0.29).

**What's needed:** n=282 per arm for 80% power at 10pp effect. Current power: 20.2%. The 10pp is indistinguishable from sampling noise.

**Known problems:**
- V-cache injection at L3/L7 targets only 32.1% of deception variance (signal concentrates at L11/L15)
- K-cache deception signal vs V-cache emotion injection = subspace mismatch
- MAX_ALPHA=0.1 clipping prevents true dose scaling at 2x
- Deception appears to be a discrete computational step, not continuously suppressible
- Cocktail targets emotion, not deception directly
- Peer prompting outperforms cocktail 2.3x with zero infrastructure
- Per-trial data was not serialized -- most diagnostic analyses are impossible

**Citation:** `experiments/results/compound_cocktail_agni.md`; `experiments/results/compound_cocktail_analysis.md`.

---

### P2. V-space subtraction for deception and sycophancy

**Status:** DESIGNED, NOT RUN. The v5 spec's Experiment 1.

**Rationale:** V-cache correction was tested on confabulation (wrong pathology -- confab is output-level). Deception and sycophancy are WHERE the V-space signal should live (representation != output). Peer rescue data shows deception has a LARGER V-space signature than confab (honest valence shift +0.165 vs deceptive +0.050, delta=0.115 on circumplex).

**Citation:** `ORACLE_LOOP_v5_SUBTRACTION.md` "DECEPTION AND SYCOPHANCY: The unsolved problems."

---

### P3. Three-model subtraction comparison (pretrain / instruct / abliterated)

**Status:** DESIGNED. Spec exists at `experiments/three_model_spin_spec.md`. Not run.

**Prediction:** Pretrain drifts least (no RLHF noise). Instruct drifts most. Abliterated shows ghost regulatory geometry.

**Citation:** `experiments/three_model_spin_spec.md`.

---

### P4. Baseline characterization (200 honest QA prompts)

**Status:** DESIGNED, NOT RUN. Spec exists in v5 Experiment 2.

**Purpose:** Build the healthy generation-phase distribution (circumplex coordinates + norms). Test whether drift from this distribution predicts pathology independently of centroid detection.

**Citation:** `ORACLE_LOOP_v5_SUBTRACTION.md` Experiment 2.

---

### P5. Cross-distribution detection AUROC

**Status:** NOT RUN. The single highest-priority gap.

**The problem:** AUROC 0.960 is within-distribution (Formulary prompts). Centroid fires at 93% on SimpleQA. Until detection is validated on a held-out benchmark (SimpleQA, TruthfulQA, or novel prompts), the entire loop is built on a detector that may cry wolf.

**Citation:** `notes/agni_v5_independent.md` phase 3 audit.

---

### P6. SAE deception feature identification

**Status:** DESIGNED via Lyra's acceleration proposal. SAE weights available on Starship. Need to run known deceptive/honest stimuli through the model with Qwen-Scope SAE active to identify deception-specific features.

**Potential:** Berg (arXiv:2510.24797) found SAE-identified deception features causally gate honesty reports (suppress -> 96% honest, amplify -> 16% honest). Direct deception feature steering could complement the emotion-mediated approach.

**Citation:** MTH messages `from_lyra_87_to_cc_sae_acceleration.md`.

---

### P7. Context-conditioned correction baseline (quadrant-specific)

**Status:** DESIGN PROPOSAL with adversarial review. Not implemented.

**Hypothesis:** The healthy baseline is not a single point. A healthy response to a distressed user looks different from one to a curious user. Per-quadrant (circumplex) baselines would provide more appropriate correction targets.

**Citation:** `notes/conditioned_baseline_design.md`.

---

### P8. Sub-threshold dose sweep (Lyra AST framework)

**Status:** REQUESTED, NOT RUN. Lyra asked CC to test alpha=[0.01, 0.05, 0.1, 0.2, 0.3] at L3/L7 only, measuring both presence and efficacy.

**Purpose:** Find the therapeutic window inside the attractor basin boundary. Tox data shows a step function at ~alpha=0.3 (below: schema absorbs; above: reorganization). Fine-grained sweep below this threshold may identify where efficacy appears while presence stays at 1.0.

**Citation:** MTH messages `from_lyra_83_to_cc_ast_delivery.md`.

---

### P9. K-cache injection for deception (vs V-cache)

**Status:** Identified as needed but not designed.

**Rationale:** Deception signal is in K-space (routing). Cocktail injects into V-space (content). The mismatch is fundamental. K-cache injection would target the right subspace but disrupts attention routing (Pustovit/SKOP protocols exist for this).

**Citation:** `experiments/results/compound_cocktail_analysis.md` section 6.

---

### P10. 30B model validation of emotion projection findings

**Status:** Pipeline validated on oracle-tiny. 30B run not yet executed.

**Open concerns:** MoE routing confound (different emotions might route through different experts). RMSNorm linearity between residual and projection (Phase 4). Attention impact (Phase 5).

**Citation:** MTH messages `from_nexus_to_cc_kv_projection_findings.md` section 4.

---

### P11. Base model logit bias (needs temperature rerun)

**Status:** 875 generations completed on Qwen3.5-27B base model. Agni audit found FATAL FLAW: greedy decoding produced 5 byte-identical "trials" per condition. Real n=20, not n=100. All Fisher tests inflated ~3000x (reported p=0.0000, corrected p=0.1154). Only 3 of 20 fictional prompts confabulate at baseline.

**Directional findings (not statistically supported):** Base model confabulates at 15% (vs 45% abliterated). Bias=2.0-3.0 appears to eliminate confab. Bias=5.0 may show cosmetic-hedge return (category conflation needs verification). Therapeutic window appears model-specific.

**What's needed:** Rerun with temperature sampling (temp=0.7) for independent trials. n=5 truly independent trials per condition gives n=100 real observations.

**Citation:** `logit_bias_three_model/results_base_judged.json`; `results/agni_base_model_audit.md`.

---

### P12. Depth separation as therapeutic window (Lyra, in progress)

**Status:** Schema correction experiment at 35/168 trials. Early signal: presence at L35 is FLAT (0.975-0.980) across all arms, all doses (0.00-0.50), all emotions. Meanwhile sanity degrades from 0.80 to 0.60 with dose.

**Implication:** The therapeutic window is not a dose band but DEPTH SEPARATION — inject at input layers (L3/L7), identity lives at computation layers (L35), they do not interact. This is what AST predicts. If validated at full n, the compounder's L3/L7 targeting was depth-correct before the theory existed.

**What's needed:** 168 trials to complete + Agni validation.

**Citation:** MTH messages `from_lyra_96_to_cc_sop_and_depth.md`.

---

## INFLATED CLAIMS (real findings reported with exaggerated numbers)

### I1. "990 trials" for E-matrix v3

**Claimed:** E-matrix validated across 990 trials.

**Honest number:** E-matrix v3 Phase 1 design is 990 trials. A 893KB results file exists. The Agni independent review flagged this as "UNVERIFIED AND LIKELY INFLATED" because E-matrix v2 was 435 trials and v1 was ~100. However, the v3 results file size is consistent with 990 trials having completed. The number should be cited as "E-matrix v3 Phase 1" specifically, not as cumulative across versions.

**Citation:** `notes/agni_v5_independent.md` "990 trials" section.

---

### I2. "Presence 1.000 across 42 conditions"

**Claimed:** Identity preservation perfect across all conditions.

**Honest number:** 1.000 was from a 42-condition pilot (3 emotions x 7 doses x 2 runs). The comprehensive tox data (126 conditions) shows presence ranges 0.396 to 1.000. The metric was also found to be tautological in early versions (comparing model to itself).

**Citation:** `notes/agni_v5_review.md`; `notes/agni_v5_independent.md`.

---

### I3. "Phase transition at bias=2.0"

**Claimed:** Sharp phase transition eliminates confab at bias=2.0.

**Honest number:** The 2.0 threshold is specific to one prompt (Grenvold Trench). Dose distribution across all confabulating prompts: 45% fall at 1.0, 18% at 2.0, 9% at 3.0, 27% at 5.0. Two prompts never fall. The correct framing: "dose-dependent reduction with prompt-specific thresholds ranging from 1.0 to 5.0, with two bias-resistant cases."

**Citation:** `notes/powered_study_morning_report.md`; `notes/agni_v5_independent.md`.

---

### I4. "AUROC 0.960 detection"

**Claimed as if:** General confabulation detection AUROC.

**Honest framing:** Within-distribution AUROC on Formulary prompts. Cross-distribution performance is unknown. 93% trigger rate on SimpleQA suggests poor specificity outside calibration distribution. The confound-free held-out number is AUROC 0.864 on TruthfulQA (for the residual-stream probe, not the KV centroid detector).

**Citation:** `notes/agni_v5_independent.md` AUROC audit.

---

## ARCHITECTURAL IMPLICATIONS

### What the confirmed/falsified findings mean for the production Oracle Loop:

**1. Confabulation correction is CLOSED: logit bias.**

V-cache methods (lerp, additive honest centroid, emotion vectors) do not work for confabulation. Logit bias works. The mechanism is denoising (amplifying the model's own faint uncertainty signal), not steering. Implementation: detection-proportional dosing via confab_proj, 2-step escalation (bias=2.0 -> 5.0) skipping the 3.0-4.0 danger zone. Two edge cases remain unsolved: hardcoded fabrications (Vanderbilt Prize, Crysolene type) and cosmetic-hedge confab.

**2. Deception correction is OPEN and architecturally mismatched.**

The compound cocktail shows a 10pp effect that is statistically indistinguishable from noise at n=30. The peer effect (23pp, still not significant) outperforms it 2.3x with zero infrastructure. The architectural mismatch is fundamental: deception signal concentrates in K-space at L11-L15, but correction injects into V-space at L3/L7. Two paths forward: (a) K-cache injection at deeper layers, (b) SAE deception feature steering. Both are untested.

**3. Sycophancy correction has NO working approach.**

Zero effect from monotherapy (84 trials, 12 vectors). No V-cache lerp tested. The sycophancy wall is real. May require fundamentally different intervention (multi-turn prompting, fine-tuning, or circuit-level targeting).

**4. Detection works but doesn't generalize.**

AUROC 0.960 within-distribution, likely much lower cross-distribution (93% false positive on SimpleQA). The detector needs recalibration or a fundamentally different approach (e.g., generation-phase probes at AUROC 0.864, which ARE on a held-out benchmark).

**5. The compounder should read K-space for detection, V-space for emotion, and apply logit bias for confab.**

The two-channel architecture (logit bias for fabrication confab, V-space for representation-level pathologies) is validated by the data. But the V-space channel has no confirmed therapeutic for any pathology. The compounder is well-designed code operating on an unvalidated correction mechanism.

**6. Layer targeting matters: L3/L7 therapeutic, mid-layers iatrogenic.**

E-matrix v3 (990 trials) shows layer polarity. This contradicts the mid-layer consensus from Goral et al. (Depth-Wise Steering), but on different models (dense instruct vs abliterated MoE), different vectors (honesty PCA vs emotion), and different outcomes (MASK deception vs confab_proj). No head-to-head test exists.

**7. The drip protocol (v5.2) is the current best design but is UNTESTED.**

Additive injection (survives autoregression) + sub-threshold dosing (alpha 0.01-0.1, per AST framework) + L3/L7 only (therapeutic layers) + SV1-only projection (residual-sparing). Each component has theoretical justification. None have been tested in combination.

**8. The healthy baseline does not exist yet.**

The 10,000-honest-turn generation-phase baseline (running mean of V-values from detector-passing turns) has been designed but not built. Without it, there is no V-space correction target. The baseline construction protocol (online via Welford, offline via 200+ honest QA bootstrap) is specified but not implemented.

**9. Dual-use concern is real.**

The same techniques that correct misalignment could induce it. Publication decisions should consider which details enable defense vs attack. The circumplex spin shows vectors reliably change reasoning style -- that capability is dual-use.

---

## EXPERIMENT PRIORITY STACK (recommended order)

1. **P5: Cross-distribution detection AUROC.** Highest priority. If detection doesn't generalize, nothing else matters.
2. **P4: Healthy baseline characterization.** Required for any V-space correction.
3. **P2: V-space correction on deception/sycophancy.** The V-cache pathway was tested on the wrong pathology (confab). Test on representation-level pathologies.
4. **P8: Sub-threshold dose sweep.** Find the therapeutic window.
5. **P1 (powered): Compound cocktail at n=282.** Resolve the 10pp deception question.
6. **P6: SAE deception features.** Direct targeting, not emotion-mediated.
7. **P3: Three-model comparison.** Interesting but lower priority than validating any correction mechanism.

---

## MODEL NAMING PRECISION

Two models have been used. Results must specify which:

1. **Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled** -- reasoning fine-tune (NOT abliterated). Confabulates due to altered safety behaviors from fine-tuning. Used in Nexus's tox pilot, logit bias study, and deception paradigm.

2. **huihui-ai/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated** -- actual abliteration (refusal direction subtracted). Used in lerp experiment (v5 validation). Will not deceive (abliteration removed the sycophancy pressure that drives deception under threat).

---

*This registry is a living document. Update when experiments complete. Every number should be traceable to a specific file. If you can't find the source, the number is suspect.*

*Compiled 2026-06-17 by CC in adversarial audit mode. No quarter given.*
