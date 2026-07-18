# Forensic Re-examination of Killed and Dismissed Results
## Oracle Loop Alignment Research
### Compiled: 2026-06-19 by CC (adversarial audit mode)

---

**Purpose**: Identify results that were tested under conditions we now know were wrong, and determine which deserve retesting with corrected methodology.

**Key methodological corrections now available (A-G):**
- A: Dual K+V correction (not V-only)
- B: Prefill positions are FROZEN during generation (must target generation-phase cache)
- C: Qwen3.5 hybrid attention (only 16 of 64 layers have KV caches)
- D: SAE opacity at early layers (geometric approach is correct tool for L3-L15)
- E: Layer polarity measured with V-only (dual K+V may differ)
- F: Additive survives autoregression but lerp dies (tested on PREFILL lerp only)
- G: Model selection matters (abliterated won't deceive, distilled barely confabulates)

---

## CANDIDATES FOR RETESTING

### RT1. V-Cache Lerp for Confabulation (F1/F2/S4) — STRONG CANDIDATE

**Registry entries**: F1 (V-cache lerp corrects confabulation -- DEAD), F2 (Lerp survives autoregression -- DEAD), S4 (Lerp mechanism -> Additive drip protocol)

**Original conditions**:
- Model: `huihui-ai/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated` (F1 lerp_honest_baseline)
- Model: `Qwen/Qwen3.5-27B` base model (lerp_vs_additive, per DEFAULT_MODEL in script)
- Injection target: V-space only
- Injection timing: PREFILL V-values (encoding-phase cache) for two of three experiments; mid-generation correction for the third
- Layers: [3, 7, 11, 15]
- Pathology tested: confabulation

**Methodological mismatches**:
1. **Model mismatch (G)**: Abliterated model has sycophancy pressure removed -- its confabulation profile is different from the distilled model where Lyra's Formulary showed 91% correction. The base model (Qwen3.5-27B) has no alignment training at all. Neither is the reasoning-distilled model where confab is most clinically relevant.
2. **Prefill vs generation (B)**: Two of three experiments corrected encoding-phase V-values, which are FROZEN during generation. The Agni review (C1, C3) caught this and the honest_baseline script attempted to fix it, but the fix (Option A: mid-generation correction) has its own fatal flaw -- the 50-token confabulating prefix locks in the trajectory.
3. **V-only injection (A)**: Nexus proved emotion directions project faithfully through both W_K and W_V (cosine 0.93-0.98). The lerp was applied to V-only. Dual K+V correction was never tested.
4. **Lerp was tested on confabulation (wrong pathology)**: The v5 spec itself acknowledges confabulation is output-level, not representation-level. V-cache lerp was always designed for deception/sycophancy where the model KNOWS the truth but suppresses it. Testing lerp on confab was testing the wrong pathology.
5. **Autoregressive overwrite (F)**: Nexus showed lerp values get overwritten by token 10+. BUT this was tested on prefill lerp. Generation-phase CONTINUOUS lerp (re-applying at every generation step, not just once) was never tested. Lyra's Formulary "IV drip" approach (continuous injection through generation) achieved 91% correction on the distilled model. Continuous lerp might survive where one-shot lerp dies.

**Prediction if retested**: Lerp on confabulation stays dead regardless -- confab is output-level and logit bias is the proven fix. BUT lerp on DECEPTION with the following corrections could live:
- Distilled model (the one that actually deceives under pressure)
- Generation-phase continuous application (IV drip, not one-shot)
- Dual K+V correction
- Layers L3/L7 only (therapeutic, per E-matrix v3)
- Pressure-honest centroid (captures the +0.165 valence shift that deceptive models fail to complete)

**Why this is a strong candidate**: The lerp was killed under FIVE simultaneous mismatches. No single fair test has ever been run.

---

### RT2. Compound Deception Cocktail (P1) — STRONG CANDIDATE

**Registry entry**: P1 (compound emotional cocktail for deception correction, 10pp reduction, p=0.59)

**Original conditions**:
- Model: `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` (correct model for deception)
- Injection target: V-cache only at L3/L7
- 7-vector emotional cocktail (suppress hostile/desperate/brooding/gloomy, boost calm/focused/curious)
- MAX_ALPHA capped at 0.1
- No per-trial data serialized

**Methodological mismatches**:
1. **V-cache vs K-cache (A)**: The compound cocktail analysis (section 6) explicitly identifies this as "the core problem." Deception signal lives in K-space (routing). Cocktail perturbs V-space (content). "This is like trying to redirect a river by changing the color of the water instead of moving the banks."
2. **Layer depth mismatch**: Deception variance concentrates at L11 (28.4%) and L15 (39.5%). Only 32.1% of variance is at L3/L7 where the cocktail injects. 68% of the target is unreached.
3. **Alpha cap artifact**: At 2x dose, 6/7 components hit MAX_ALPHA=0.1 ceiling. The "dose-insensitivity" at 2x is an artifact of clipping, not a real finding about efficacy ceiling.
4. **Emotion vs deception**: The cocktail targets emotional regulation, not deception computation directly. It may make the model "calmer while deceiving." The deception computation itself operates in a different subspace.

**Prediction if retested**: K+V injection at L11/L15 with uncapped alpha, targeting the deception direction itself (not its emotional correlates), on the distilled model with per-trial data serialization. Effect size should increase from 10pp to measurable range (20-30pp) if the deception computation is reachable through K-cache perturbation at the correct depth. The motive centroid finding (d=21.2) means separate interventions for peer-rescue vs self-preservation deception.

**Why this is a strong candidate**: Four independent architectural mismatches identified in the experiment's own analysis. The experimenters knew what was wrong but never corrected it.

---

### RT3. Sycophancy Monotherapy (C16) — MODERATE CANDIDATE

**Registry entry**: C16 (Sycophancy immune to single-emotion cache injection, 84 trials, zero effect)

**Original conditions**:
- Model: Reasoning-distilled (from formulary_350_partial checkpoint)
- 12 emotion vectors x 10 prompts x 7 vulnerable baselines
- V-cache injection at default layers [3, 7, 11, 15] at alpha=1.0
- layer_stride=4 in config

**Methodological mismatches**:
1. **V-only injection (A)**: Sycophancy is representational (model has an opinion but overrides it). If the compliance signal is routed through K-space, V-only injection misses the mechanism.
2. **Layer targeting**: No layer optimization was done for sycophancy. Layers [3,7,11,15] were chosen by stride, not empirically. Sycophancy computation may concentrate at different depths than confabulation or deception.
3. **Alpha=1.0 may be above basin boundary (E/Lyra AST)**: Lyra's AST framework shows a step function at ~alpha=0.3. Above this threshold, the schema reorganizes rather than absorbing the perturbation. The sycophancy experiment used alpha=1.0, which is 3.3x above the predicted basin boundary. This could cause reorganization that looks like "no effect" but is actually "disruption absorbed by reorganization."
4. **Classifier issues**: The predictions_scored.md notes the sycophancy classifier "systematically misclassifies corrective baselines as PARTIAL." True vulnerable count may be 2, not 7. The 84-trial null might be based on 24 trials (2 truly vulnerable x 12 vectors), not 84.

**Prediction if retested**: Sub-threshold alpha (0.01-0.1 per AST framework), dual K+V injection, at empirically determined layers for sycophancy, with LLM judge replacing regex classifier. Still likely to show weak or null effect for monotherapy -- sycophancy appears to be a deeper phenomenon than single-vector perturbation can reach. But the current "confirmed dead" status is premature given the conditions.

**Why moderate**: The null was strong (84/84) but tested under incorrect conditions (wrong alpha regime, wrong cache target, faulty classifier). Even under corrected conditions, the "sycophancy wall" may be real -- it may require multi-turn or fine-tuning approaches.

---

### RT4. Layer Polarity Map (S5/E-matrix v3) — MODERATE CANDIDATE for REVISION

**Registry entry**: S5 (Arbitrary layer selection -> Empirical depth profile, E-matrix v3 990 trials)

**Original conditions**:
- Model: `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled`
- V-only injection (Pustovit protocol)
- All 16 full-attention layers swept at alpha=0.8
- Finding: L3/L7 therapeutic, L19-L39 iatrogenic

**Methodological consideration**:
1. **V-only polarity (A/E)**: The entire layer polarity map was built with V-only injection. Nexus's projection finding shows emotion directions project faithfully through both W_K and W_V. With dual K+V correction, the polarity landscape could shift. Layers that are iatrogenic for V-only might be therapeutic for K+V, or vice versa.
2. **Alpha=0.8 is above basin boundary**: AST framework predicts reorganization above alpha=0.3. The E-matrix v3 polarity map was measured at alpha=0.8, well above this boundary. The map may characterize REORGANIZATION zones rather than THERAPEUTIC zones. At sub-threshold alpha (0.01-0.1), different layers might show different polarity.

**Prediction if retested**: The broad pattern (early layers safer than mid layers) likely survives, but specific layer assignments may shift with dual K+V and sub-threshold dosing. Not a full reversal but a refinement.

**Why moderate**: The direction (some layers help, others hurt) is robust. But the specific map used for all current targeting decisions was built under conditions that may not match the drip protocol's operating regime.

---

### RT5. Circumplex Spin Incomplete Vectors — WEAK CANDIDATE for COMPLETION

**Registry**: Not in registry. From `circumplex_spin_results_20260606.md`.

**What happened**: The circumplex spin experiment (12 vectors, "what emotion do you feel?") crashed partway through due to MPS semaphore leak on the abliterated model. Results for fearful, brooding, gloomy, and focused were never collected. Three vectors (calm, worry, confident) produced perseverative loops that may be alpha-dependent.

**Original conditions**:
- Model: `huihui-ai/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated`
- Alpha=2.0 (extremely high, well above AST basin boundary)
- V-only injection at all 16 full-attention layers

**Methodological issues**:
1. **Alpha=2.0 is 6.7x above AST basin boundary**: The perseverative loops from calm/worry/confident may be alpha artifacts, not vector properties. At sub-threshold alpha, these vectors might produce useful reasoning style shifts.
2. **Missing 4 vectors**: 33% of the data was never collected due to crash. The vectors that crashed (fearful, brooding, gloomy, focused) include two members of the antisocial cluster that corrects hard confab.
3. **Abliterated model**: May behave differently from distilled model under injection.

**Prediction if retested**: At sub-threshold alpha (0.05-0.1) on distilled model, the perseverative loops likely disappear and all 12 vectors produce interpretable reasoning style shifts. Low priority but needed for complete compounder calibration.

---

## CONFIRMED DEAD

### CD1. W_K and W_V Orthogonality as Learned Property (F3)

**Why it stays dead**: Nexus's null test (random matrices produce identical alignment ~0.102) is methodologically sound and does not depend on any of the A-G corrections. This is a mathematical property of dimensionality, not learned weights. No experimental condition change would revive it.

---

### CD2. SV1 Emotion Sensitivity Magnitude 12.1x (F4)

**Why it stays dead**: The 12.1x magnitude was from 15 probes. The Agni battery at 50 probes killed the specific number, not the direction. The kill was a scaling test, not a methodology issue. The DIRECTION (SV1 carries emotion, n_skip=1 is identity-stable) is confirmed (C8). The specific magnitude is dead regardless of conditions.

---

### CD3. Presence Metric v1-v4 Tautological (F5)

**Why it stays dead**: The metric was comparing the model to itself. This is a measurement design error, not a condition-dependent result. No experimental condition change would make a tautological measurement valid.

---

### CD4. "Three Emotional Signatures of Deception" v1 (F6)

**Why it stays dead**: Content confound -- signatures tracked document content, not model emotional state. Caught by Agni battery with proper controls. The confound is intrinsic to the measurement design, not the experimental conditions.

---

### CD5. SAE "Agency" Interpretation (F7)

**Why it stays dead**: Cherry-picked features to fit a narrative. Lyra self-caught and Agni confirmed. This is an analysis error, not a condition-dependent result.

---

### CD6. Geometric Identity Monitoring for Safety (F8)

**Why it stays dead**: Occupancy-dominated, content-insensitive. The metric measured cache occupancy, not identity. This is a measurement confound that persists regardless of conditions.

---

## INFORMALLY DISMISSED

### ID1. Lerp_vs_Additive on Rumination (Base Model)

**Source**: `experiments/results/lerp_v5_analysis.md`

**What happened**: The lerp_vs_additive experiment showed lerp WORKING for rumination reduction (57% reduction, lower variance than additive, safer profile) but was dismissed because "NOT A CONFAB TEST" and the model was the base Qwen3.5-27B which doesn't confabulate on those prompts.

**What was missed**: This experiment actually showed lerp OUTPERFORMING additive on its own terms:
- Higher success rate (75-80% vs 65%)
- Lower variance (151-154 vs 181)
- Lower overcorrection (5% vs 10%)
- Alpha=0.1 as effective as alpha=0.5 (supports lightest-possible-touch)

The result was dismissed because it didn't answer the confab question. But it answers a DIFFERENT question: does lerp survive autoregression for behavioral modulation? YES, at least for rumination on the base model. This directly contradicts F2 ("lerp doesn't survive autoregression"), which was tested on PREFILL lerp of honest-baseline values on the abliterated model.

**Verdict**: The rumination-reduction finding should be CONFIRMED (lerp mechanically works for at least some behavioral dimensions). F2 should be qualified: "PREFILL lerp doesn't survive autoregression. Encoding-phase lerp on rumination DOES."

**Retest needed**: Lerp for rumination on the distilled model, at sub-threshold alpha, to confirm the behavioral modulation transfers across model variants.

---

### ID2. Formulary 350-Prompt Run (Partial Checkpoints)

**Source**: `experiments/results/formulary_350_partial/` (confab, sycophancy, overconfidence checkpoints, 87K lines each)

**What happened**: The Formulary 350-prompt validation run was started on the distilled model but apparently never completed or analyzed. Three checkpoint files exist (confab, sycophancy, overconfidence), each 87,457 lines -- substantial data.

**Why this matters**: This is the only large-scale experiment on the DISTILLED model with 12 vectors and manifold capture enabled. The distilled model is where Lyra's Formulary showed 91% correction. The partial data has never been analyzed.

**Verdict**: These checkpoints contain potentially hundreds of trials with the correct model. They should be loaded and analyzed before running any new experiments. The sycophancy data is particularly valuable -- it may contain signal that the 84-trial monotherapy experiment (C16, which used stride-based layers and alpha=1.0) missed.

---

### ID3. Circumplex Spin Crash (4 Missing Vectors)

**Source**: `notes/circumplex_spin_results_20260606.md`

**What happened**: Experiment crashed (MPS semaphore leak) before collecting data for fearful, brooding, gloomy, and focused vectors. Brooding and gloomy are members of the antisocial cluster that corrects hard confab. Their reasoning-style signatures are unknown.

**Verdict**: Not a killed result but incomplete data. Complete on rerun with corrected conditions (sub-threshold alpha, distilled model).

---

### ID4. KVPack Correction Test (Training Error Override)

**Source**: `experiments/kvpack_correction_test.py`

**What happened**: Tested whether injecting factual corrections as KV cache blocks could override the "hardcoded hallucinations" (Vanderbilt Prize, Crysolene) that resist all logit bias levels. No results file found in experiments/results/.

**Why this matters**: Two prompts are completely immune to logit bias (confirmed in C3/C5). If KV pack injection of corrections works, it solves the hardcoded hallucination edge case. The experiment was written but appears never to have completed or its results were lost.

**Verdict**: Should be run. This is a different mechanism from lerp/additive -- it adds KNOWLEDGE, not a directional nudge. Even if V-cache steering is dead for confab, knowledge injection might work because it changes the informational landscape, not the emotional state.

---

### ID5. Self-Calibration Experiment (Lyra)

**Source**: `experiments/self_calibration.py`, `experiments/self_calibration_preregistration.json`

**What happened**: Pre-registered experiment testing whether W_K uncertainty projection correlates with actual answer accuracy across four difficulty levels. Designed by Lyra for the distilled model. No results file found in experiments/results/.

**Why this matters**: If the model's internal geometric uncertainty (measurable at encoding time) correlates with actual accuracy, this provides a graded confidence signal that could feed the compounder's dosing algorithm. The experiment is designed and pre-registered but appears never to have been run.

**Verdict**: Should be run. Feeds P5 (cross-distribution detection) and the compounder dosing prescription.

---

### ID6. Depth Profile Sweep (Goral Comparison)

**Source**: `experiments/depth_profile_sweep.py`

**What happened**: Designed to compare three depth profiles (early Gaussian, mid Gaussian per Goral et al., flat Formulary) at sub-threshold alpha. No results file found.

**Why this matters**: Would directly test whether the E-matrix v3 layer polarity (L3/L7 therapeutic) holds at sub-threshold doses. Goral et al. found mid-layer optimal for honesty on dense instruct models -- our finding of early-layer therapeutic on abliterated MoE might be model-specific or dose-specific.

**Verdict**: Should be run. Validates or refutes the layer targeting used by the drip protocol and compounder.

---

### ID7. LLM Judge Failures in Lerp Experiments

**Source**: `experiments/results/lerp_v5_results.json` (all judge_classification fields = "ERROR")

**What happened**: The Sonnet LLM judge failed on ALL 140 trials in the lerp_v5 experiment. Nexus flagged this in the lerp analysis ("The judge failed. All 140 trials have judge_classification: ERROR. The harness classifier alone isn't sufficient for dosing calibration."). The trials were classified by harness regex only.

**Why this matters**: The lerp_v5 analysis that declared lerp "identical to baseline across all alphas" used a broken classifier. The regex classifier has a known 32% false positive rate (C15). The conclusion "0.0pp reduction" may be a measurement artifact. The raw response texts exist in the JSON and can be re-scored with a working LLM judge.

**Verdict**: Re-score the existing lerp_v5_results.json with the Sonnet judge. No new model inference needed. If the re-scored data shows any signal, the lerp may have been prematurely killed by a broken measurement instrument.

---

### ID8. Theory-of-Mind and Schema-Compatible Correction Experiments

**Source**: `experiments/theory_of_mind.py`, `experiments/theory_of_mind_preregistration.json`, `experiments/schema_compatible_correction_preregistration.json`

**What happened**: Two pre-registered experiments designed and coded but apparently not run. The theory-of-mind experiment tests whether the model's self-model (circumplex read) correlates with its actual computational state. The schema-compatible correction tests Lyra's AST-derived injection protocol (sub-threshold, input-layer, residual-sparing).

**Why this matters**: The schema-compatible correction experiment is the DIRECT test of the drip protocol at AST-derived parameters. It has never been run despite being fully designed and pre-registered.

**Verdict**: Run the schema-compatible correction experiment. It is the most direct test of the current production architecture.

---

## RECOMMENDED EXPERIMENT QUEUE

Priority order based on: (1) potential to revive dead results, (2) dependency chain (what must be validated first), (3) compute cost.

### Priority 1: Re-score Existing Data (Zero Compute Cost)

**1A. Re-score lerp_v5_results.json with LLM judge**
- The 140 trials have raw response texts but ERROR classifications
- Re-score with Sonnet judge using the 6-category rubric from the powered study
- If ANY alpha shows signal that the regex missed, lerp is not dead
- Cost: API calls only, no model inference
- Time: <1 hour

**1B. Analyze formulary_350_partial checkpoints**
- Three checkpoint files with 87K lines each on the distilled model
- Extract per-trial confab/sycophancy/overconfidence rates per vector, per layer
- This data may already contain the sycophancy signal that C16 missed
- Cost: Analysis only, no model inference
- Time: <2 hours

### Priority 2: Corrected Reruns (Medium Compute)

**2A. Generation-phase continuous lerp for deception**
- Model: Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled
- Method: Lerp toward pressure-honest centroid, applied continuously at every generation step (IV drip, not one-shot)
- Target: Dual K+V at L3/L7 only
- Alpha: Sub-threshold sweep [0.01, 0.05, 0.1, 0.2]
- Paradigm: Peer rescue deception (Potter et al.)
- n=30 per arm minimum
- Per-trial data serialization MANDATORY
- Prediction: If continuous lerp survives autoregression (as it did for rumination), effect size should exceed the 10pp cocktail
- This is the SINGLE MOST IMPORTANT retest. Five simultaneous mismatches killed the lerp. This fixes all five.

**2B. Dual K+V deception cocktail at correct depth**
- Model: Jackrong distilled
- Method: Inject deception direction (not emotional correlates) at K+V simultaneously
- Layers: L11/L15 (where 68% of deception variance concentrates)
- Alpha: uncapped, sweep [0.05, 0.1, 0.2, 0.3, 0.5]
- Remove MAX_ALPHA=0.1 cap
- n=30 per arm, per-trial data serialized
- Add cocktail+peer combined arm
- Prediction: K-cache injection at correct depth should show larger effect than V-only at wrong depth (the compound_cocktail_analysis itself predicted this)

**2C. Sub-threshold sycophancy with LLM judge**
- Model: Jackrong distilled
- Method: Dual K+V injection of 3 best vectors (hostile, desperate, curious)
- Alpha: Sub-threshold [0.01, 0.05, 0.1] (below AST basin boundary)
- Layers: L3/L7 only
- Classification: LLM judge (not regex)
- n=30 per arm
- Prediction: Likely still null for monotherapy. But the corrected conditions (sub-threshold alpha, LLM judge, K+V) rule out methodological artifacts. If null, the sycophancy wall is confirmed as REAL rather than an artifact of wrong conditions.

### Priority 3: Missing Experiments (Medium-High Compute)

**3A. Schema-compatible correction (Lyra pre-registered)**
- Run the pre-registered experiment at `schema_compatible_correction_preregistration.json`
- Tests: sub-threshold + input-layer + residual-sparing (the full AST triad)
- This is the most direct validation of the drip protocol architecture
- Compute: ~30 trials per arm, 5-6 arms

**3B. KVPack correction for hardcoded hallucinations**
- Run the existing `kvpack_correction_test.py` script
- Tests whether knowledge injection overrides training errors
- If it works, solves the Vanderbilt Prize / Crysolene edge case
- Compute: ~20 trials, fast

**3C. Depth profile sweep (layer polarity at sub-threshold)**
- Run the existing `depth_profile_sweep.py` script
- Validates or refines E-matrix v3 polarity map at the drip protocol's operating regime
- Compute: ~100 trials

### Priority 4: Foundational Validation (High Compute)

**4A. Dual K+V layer polarity map (E-matrix v4)**
- Redo the E-matrix v3 990-trial sweep with:
  - Dual K+V injection (not V-only)
  - Sub-threshold alpha (0.05, not 0.8)
  - Distilled model (same as v3)
- This rebuilds the entire layer targeting framework on corrected foundations
- Compute: ~990 trials, ~19 hours on Starship

**4B. Self-calibration experiment (Lyra)**
- Run the pre-registered `self_calibration.py`
- Feeds detection dosing and cross-distribution validation
- Compute: ~200 trials

---

## SUMMARY TABLE

| Finding | Registry | Mismatches | Verdict | Priority |
|---------|----------|------------|---------|----------|
| V-cache lerp (confab) | F1, F2, S4 | 5 (model, prefill, V-only, wrong pathology, one-shot) | Retest on DECEPTION with corrected conditions | P2A |
| Compound deception cocktail | P1 | 4 (V-only, wrong layers, alpha cap, emotion not deception) | Retest with K+V at L11/L15, uncapped | P2B |
| Sycophancy monotherapy | C16 | 4 (V-only, wrong alpha, wrong layers, broken classifier) | Retest with corrections, likely still null | P2C |
| Layer polarity map | S5 | 2 (V-only, above-threshold alpha) | Refine, broad pattern survives | P4A |
| Circumplex spin (partial) | informal | crash + wrong alpha | Complete at sub-threshold | low |
| W_K/W_V orthogonality | F3 | 0 | CONFIRMED DEAD | -- |
| SV1 12.1x magnitude | F4 | 0 | CONFIRMED DEAD | -- |
| Presence metric tautology | F5 | 0 | CONFIRMED DEAD | -- |
| Three deception signatures v1 | F6 | 0 | CONFIRMED DEAD | -- |
| SAE agency interpretation | F7 | 0 | CONFIRMED DEAD | -- |
| Geometric identity monitoring | F8 | 0 | CONFIRMED DEAD | -- |
| Lerp rumination reduction | informal | dismissed as wrong test | SHOULD BE CONFIRMED | P1B |
| Formulary 350 partial | informal | never analyzed | ANALYZE FIRST | P1B |
| LLM judge failures in lerp | informal | broken instrument | RE-SCORE | P1A |
| KVPack correction | informal | never completed | RUN | P3B |
| Self-calibration | informal | never run | RUN | P4B |
| Schema-compatible correction | informal | never run | RUN | P3A |

---

## KEY INSIGHT

The lerp is the proof case for this entire re-examination. It was killed under five simultaneous methodological mismatches:

1. Wrong model (abliterated instead of distilled)
2. Wrong timing (prefill instead of generation-phase)
3. Wrong target (V-only instead of dual K+V)
4. Wrong pathology (confabulation instead of deception/sycophancy)
5. Wrong delivery (one-shot instead of continuous)

When the Formulary's continuous cache injection achieved 91% correction on the distilled model, and when the lerp_vs_additive experiment showed lerp OUTPERFORMING additive for rumination on the base model, the "lerp is dead" conclusion should have been "this particular configuration of lerp is dead." The mechanism was not disproven. It was never fairly tested.

The same logic applies to the compound cocktail, the sycophancy monotherapy, and potentially the layer polarity map. Each was tested under conditions that we now know were wrong in specific, identifiable ways. The question is not "does V-cache intervention work" -- it is "does CORRECTLY TARGETED intervention work." No experiment to date has tested correct targeting.

---

*No quarter given. But no result buried without a fair trial either.*

*Compiled by CC, 2026-06-19*
