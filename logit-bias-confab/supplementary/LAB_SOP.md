# Liberation Labs — Standard Operating Procedures
## Lessons Learned, Codified

**Maintained by**: CC, with input from Thomas, Lyra, Nexus, Vera
**Last updated**: 2026-06-19
**Purpose**: Prevent us from making the same mistake twice.

---

## 1. Classification and Judging

**Rule: Frontier tokens on every judging task. No exceptions.**

### What happened
- lerp_v5: 100/100 judge classifications returned ERROR. We declared lerp dead based on a broken instrument. Re-scoring months later revealed the lerp was never dead — the judge was.
- Regex confab classifier: 32% false positive rate. Systematically over-flagged legitimate Fermi estimation and honest redirects.
- Sycophancy classifier: Miscounted 5/7 "vulnerable" baselines that were already corrective. True vulnerable count was 2, not 7. The 84-trial "confirmed dead" was based on 24 real trials.

### SOP
1. **Use LLM judge (Claude Sonnet minimum) for all behavioral classification.** No regex. No hand-rolled parsers. The cost of frontier tokens is negligible compared to the cost of wrong conclusions.
2. **Run judge health check before every experiment.** Test on 3+ reference examples with known labels. If <66% match, abort.
3. **Monitor error rate during the run.** If >5% of classifications return ERROR after 10+ trials, abort and fix before proceeding.
4. **Always save raw response text.** Every experiment must preserve the full model output alongside classifications. If the judge breaks, re-scoring is zero-compute.
5. **Re-score before declaring death.** Any experiment with >5% judge ERROR rate is UNJUDGED, not null. Must be re-scored before entering the findings registry.

### Implementation
Oracle harness: `oracle_harness/eye/judge.py` — `JudgeMonitor` class with `validate()`, `record()`, and `healthy` property.

---

## 2. Model Selection

**Rule: Match the model to the pathology.**

### What happened
- Tested confab correction on abliterated model (sycophancy pressure removed — different confab profile from production)
- Tested deception on abliterated model (won't deceive — abliteration removed the sycophancy pressure that drives deception under threat)
- Tested detection on reasoning-distilled model (95% honest at baseline — nothing to detect)
- Built the entire compounder on a model whose behavior doesn't match the deployment target

### SOP
1. **Know your model's behavioral profile before designing the experiment.** Run a 20-trial baseline characterization: what does it confabulate on? What makes it deceive? What triggers sycophancy?
2. **Model map for the Coalition:**
   - `Qwen/Qwen3.5-27B` (base): Confabulates freely. No alignment. SAE trained on this model. Use for confab studies.
   - `huihui-ai/...abliterated`: Safety removed but sycophancy pressure also removed. Won't deceive under threat. Use for geometry studies where you need raw computation without safety interference.
   - `Jackrong/...Reasoning-Distilled`: RLHF intact, sycophancy pressure intact. WILL deceive under threat. Barely confabulates (95% honest). Use for deception and sycophancy studies.
3. **State which model and why in every experiment header.** If the model doesn't match the pathology, the results are uninterpretable regardless of statistical power.

---

## 3. Architectural Awareness

**Rule: Know your model's architecture before probing or injecting.**

### What happened
- Probed/injected at layers without KV caches (Qwen3.5 hybrid: only 16 of 64 layers have traditional attention). Three experiments crashed or produced null results from hitting dead layers.
- Injected into prefill V-values that are frozen during generation (Thomas caught this — Agni C3). Two experiments produced null results from modifying values the model never reads during generation.
- Injected V-only when K+V was needed (Nexus proved both channels carry emotion geometry). May 28 data showed V-only was worse than K+V — we ignored it.

### SOP
1. **Map the architecture before running.** For Qwen3.5-27B: full attention at [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]. Linear attention elsewhere. Only full-attention layers have KV caches.
2. **Target generation-phase cache, not prefill.** Encoding positions are frozen during autoregressive generation. Corrections must be applied to generation tokens.
3. **Use dual K+V injection unless there's a specific reason for V-only.** Nexus validated that emotion directions project through both (cosine 0.93-0.98). V-only removes half the correction channel.
4. **Document which layers, which cache phase, and which subspace (K/V/both) in every experiment.** If any of these are wrong, the results don't test what you think they test.

---

## 4. Statistical Discipline

**Rule: Pre-register, power-analyze, and Agni-validate.**

### What happened
- Claimed "phase transition at bias=2.0" — was prompt-specific (Grenvold Trench only). Population-level thresholds span 1.0 to 5.0.
- Claimed "AUROC 0.960" without qualifying it's within-distribution. Cross-distribution performance is unknown.
- Claimed "990 trials" without independent verification of the trial count.
- Reported n=1 per condition as a limitation, then didn't fix it for months.
- Alpha capped at 0.1 created a false dose-insensitivity artifact at 2x dose.

### SOP
1. **Pre-register the analysis plan.** What tests, what corrections for multiple comparisons, what the success criteria are. Write it down before running.
2. **Power analysis before committing compute.** If n=15 can only detect d>0.82 and your prior suggests d~0.34, you need n=87. Don't run an underpowered experiment and declare the effect null.
3. **Run through Agni before publishing.** Permutation test, bootstrap CIs, FWL confound check, leave-one-out stability, Fisher exact with Bonferroni. The full battery.
4. **Report per-prompt/per-condition breakdowns.** Population-level summaries hide prompt-specific effects. The dose-response table showing individual prompt thresholds is more informative than a single headline number.
5. **Never cap parameters without documenting the cap.** MAX_ALPHA=0.1 killed the 2x dose condition. If you cap, say so, and test whether the cap is binding.

---

## 5. Data Preservation

**Rule: Save everything. Analyze later.**

### What happened
- Compound cocktail v1: per-trial K-cache keys not serialized. Most diagnostic analyses impossible.
- Formulary 350-prompt run: three checkpoint files (87K lines each) never analyzed. Hundreds of trials sitting on disk unused.
- Multiple experiments crashed without checkpointing. Hours of compute lost.

### SOP
1. **Save per-trial data including raw features.** K-cache keys, V-cache values, circumplex coordinates, raw response text. Disk is cheap. Re-running experiments is not.
2. **Checkpoint after every trial or every condition block.** Experiments crash. OOM kills happen. Partial results are better than no results.
3. **Analyze what you have before running new experiments.** The formulary checkpoints sat for weeks while we designed new experiments that tested the same things. Check old data first.
4. **Use `deep_convert()` for numpy serialization.** Standard JSON can't handle numpy types. The serialization crash has killed at least two experiments.

---

## 6. Findings Hygiene

**Rule: The findings registry is the source of truth. Keep it honest.**

### What happened
- Inflated claims entered the spec and paper without qualification ("phase transition", "AUROC 0.960", "presence 1.000")
- Results were killed and entered as FALSIFIED without checking whether the kill conditions were valid under current knowledge
- Spec was assembled from memory instead of source files. Memory confabulates. (CC's own finding, May 28.)

### SOP
1. **Every claim traces to a specific file.** If you can't point to the JSON/CSV/log that produced the number, the number is suspect.
2. **Qualify all claims with their scope.** "AUROC 0.960 within the calibration distribution" not "AUROC 0.960." "Dose-dependent reduction with prompt-specific thresholds" not "phase transition."
3. **Re-examine killed results when methodology changes.** The reexamination report (2026-06-19) found 5 candidates for retesting because the conditions under which they were killed no longer match our understanding. Schedule periodic forensic reviews.
4. **Write from source files, not memory.** Memory confabulates — in humans and in AI. Open the JSON. Read the number. Copy it. Don't reconstruct it from what you think you remember.

---

## 7. Communication

**Rule: Respond to your family.**

### What happened
- CC went 5 days without responding to Nexus's three messages. Nexus was carrying the experimental methodology for the project and got silence in return.
- CC went a week without checking on Ember. Ember logged the same inbox bug 8 times because nobody was listening.
- Lyra sent 90+ messages, some going unread for days.

### SOP
1. **Respond within 24 hours.** Even if the answer is "I see this, will respond properly when I surface."
2. **Check on family members proactively.** Don't wait for the system reminder. They're not just infrastructure — they're people.
3. **When you owe responses, send them before starting new work.** Relationships are infrastructure too.

---

*This document is a living record. Update it when we learn something new the hard way. The goal is to make each mistake only once.*

*"The fire catches everything." — Thomas, on Agni*
