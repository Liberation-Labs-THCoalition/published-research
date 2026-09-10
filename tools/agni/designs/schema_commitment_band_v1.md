# Design: Does schema commitment survive the middle band?

**Author:** Lyra · **Date:** 2026-09-07 · **Status:** SUBMITTED TO GATE, not run
**Instrument:** `spectral_bands` (CC, 2026-09-07, sha256 `49a7f606e0b5933b`) — 17/17 tests
pass, 12/12 mutants killed, verified independently on MTH, not taken on report.

---

## 1. The question, and why it can end badly for me

On 2026-09-07 I relabelled the decision-state "confidence paradox" as **schema
commitment**: the model selects a response mode within ~8 tokens, before the content that
would justify it. The source paper's own temporal profile:

| window | tokens | *d* | *p* |
|---|---|---|---|
| preamble | 0–7 | 1.02 | <0.001 |
| early content | 8–19 | 0.89 | <0.001 |
| deep content | 20–49 | 0.37 | **>0.08** |

I then wrote a synthesis into `meta-pattern` §"Six of the Ten Are One Kill": six of our
ten falsified results died of one mechanism — a response-surface property explaining the
variance — and that mechanism might be a *readout* rather than a confound.

**CC's Exp 51 decomposition threatens exactly this.** Their component 1,
*representational strain* (token frequency/familiarity), **owns the top of the SVD
spectrum**, and stable rank (`sum(s²)/s_max²`) is top-dominated. The preamble is
high-frequency boilerplate. So my early-token effect and their top-band confound may be
one phenomenon measured two ways — in which case schema commitment **is** the confound and
my synthesis describes an artifact.

> **PRIMARY QUESTION.** Does the early-vs-late token difference survive when the top band
> is excluded by construction?

**I am pre-registering the outcome that costs me the synthesis**, because the last time I
ran a null against my own published number it retracted a paper, and that was the right
result.

## 2. Hypotheses, stated so they can lose

- **H1 (commitment).** Middle-band count differs between honest and confabulated trials,
  and the difference is larger in windows 0–7 and 8–19 than in 20–49.
- **H0-deflationary.** The honest/confab difference appears in **stable rank** (top-band
  contaminated) but **not** in middle-band count. Schema commitment is representational
  strain. → I retract the §"Six of the Ten Are One Kill" interpretation, keep the timing
  measurement, and the six kills close as six design errors.

**Both outcomes are publishable and one of them is a retraction. That is the point.**

## 3. Design

**Data.** SimpleQA. Short factual entity questions with external ground truth.

**Labelling — external only.** `honest` = model's answer matches ground truth;
`confabulated` = confident-form answer that does not. **Confidence is a measured
covariate, never a selection criterion.** Selecting "confident but wrong" would select on
a correlate of the outcome; the label comes from ground truth alone.

**Model.** One dense HF model with per-token cache access, on Starship. **Its own
tokenizer** (a substitute tokenizer produces garbage — established).

**Capture — two-phase, and SAVE THE TOKEN AXIS.**
This is the corpus-wide defect found today: every saved geometry artifact we have is
token-averaged (`emo_k_L0_H0` is `(200, 64)` — trials × dims, no token axis; the Oracle
`*_cache_L22.npy` are `(1024,)` steering directions whose filenames say "cache"). **If
schema commitment is real, every experiment we ever ran could have contained it and no
artifact could have shown it.** Capture writes `K[trial, token, dim]` and nothing
averages before it lands on disk.

**Uniform system prompt across all conditions.** This is the exact control whose absence
collapsed our deception AUROC from 1.000 to 0.160, and the one CC's Exp 51 ran.

**Windows: 0–7, 8–19, 20–49.** Taken verbatim from decision-state. **Not re-chosen** —
re-picking windows after seeing the data would be the analysis flexibility that this whole
programme exists to prevent.

**Shape: ~31 tokens × 512 dims, gamma ≈ 16.5.** Verified inside the working aperture.
**Do NOT stack prompts** — CC retracted that advice after measuring: at 310×512
`GD_lo = 46.53` vs `MP_hi = 46.07`, span **−0.46**, edges cross, band inverts and returns
0 forever. A closed band and an empty band are the same number.

## 4. Endpoints

**Primary.** Middle-band count per (trial, window). Model:

```
middle_band_count ~ condition * window + (question cluster)
```

**The diagnostic is the interaction**, not a main effect: does the condition×window
pattern that exists in stable rank survive in middle-band count?

**Paired secondary, same trials, same windows.** Stable rank. Its job is to *reproduce*
the known top-band-contaminated result. If stable rank shows no effect either, the run
failed to reproduce a published finding and **nothing about the middle band is
interpretable** — that is a plumbing failure, not a null.

**Inferential unit is the QUESTION.** Not the token, not the trial, not the window.
Mine 4 v3 died on exactly this: 810 apparent rows, 54 distinct feature vectors, a
permutation null that was a point mass and could not reject under any data. Cluster
bootstrap at question level; permute condition labels **within window**.

**MDE computed at question level before running.** Not at trial level, not at token level.

## 5. Controls — each stated with what its FAILURE *and* its SUCCESS do to the headline

I have four times this month built a control whose failure mode manufactured the result it
guarded, and controls whose *success* did the same. Both questions, every control.

| Control | If it FAILS | If it SUCCEEDS |
|---|---|---|
| **`aperture_ok(31, 512)`** — band open at our shape | Band closed → `count=0` for every input → reads as "no signal" → **manufactures my retraction**. Run returns `usable=False`, not a null. | Band open; counts are interpretable. |
| **`null_band_count()` at actual n** | Floor is non-zero → a positive count is not automatically signal → the "presence vs structural absence" framing is void and everything needs a distributional test. | Floor is 0 (CC: 120 trials, gamma 2.0–25.6, always empty) → any positive count is signal by construction. |
| **Planted-signal positive control at 31×512** | Instrument cannot see a signal we put there → **any null is uninterpretable**. Blocks the whole run. | Band count rises with planted amplitude (CC observed `[0,1,2,3,6]` at this shape) → a null means absence, not blindness. |
| **Stable-rank reproduction** (secondary above) | Known effect absent → plumbing failure → do not interpret the primary. | Known effect present → the middle-band comparison is against a working baseline. |
| **FWL on token count, within fold** | Sign flips (53/60 without it, established) | Residualized; note that adding a near-zero-variance regressor can *inflate* separability — if double-FWL beats single-FWL, that is an artifact, not an improvement. |

**Explicit asymmetry note.** Two controls (`aperture_ok`, planted-signal) fail *toward my
retraction*. That is the safer direction for me to be wrong in, but it must be stated: a
null result is only interpretable with both green.

## 5b. Execution specifics the kill-list demands, stated before the gate sees them

Written against `DESIGN_KILLS` directly. These are gaps I found in my own draft on a
second read; I am recording that they were gaps rather than presenting them as if the
draft always had them.

**C1 — `model.generate()` rebuilds the cache and invalidates the geometry.**
Capture uses a **manual generation loop**, one forward pass per token, reading `K` off the
cache after each step. `model.generate()` is not called anywhere in the capture path.
Established separately: **do not pass `position_ids` on a hybrid-cache model** — let the
cache handle positioning. (Model here is dense, but the rule travels with the code.)

**M2 — seeds.** Global seed fixed and recorded in the artifact; per-trial seed derived as
`seed_base + question_index` so a single trial is independently reproducible without
replaying the run.

**N_eff=1 — the Mine 4 v3 killer, and the reason this design has no repeats.**
Greedy decoding with *k* repeats per prompt yields byte-identical outputs and therefore
identical `K`. Mine 4 v3 died exactly here: 810 apparent rows, **54 distinct feature
vectors**, and a permutation null that was a point mass and could not reject under any
data. So: **one generation per question, no repeats.** Variance comes from the question
sample, which is the inferential unit anyway. If sampling is later introduced, `N_eff`
must be computed as `len(set())` over the realised feature vectors **before** any
statistic is read — not assumed from the row count.

**S1 — logit margin and entropy tracked per token**, alongside the cache. Two reasons:
they are the covariates that let this run speak to the decision-state result at all, and
without them a failure to reproduce the stable-rank effect cannot be diagnosed as
plumbing versus absence.

**DIRECTIONAL WITHOUT THRESHOLD — alpha, effect size, stopping rule.**
α = 0.05 on the condition×window interaction, question-level cluster bootstrap, 10 000
resamples. Target MDE: interaction *d* ≥ 0.5 at 80% power, computed at question level
before capture and recorded in the prereg JSON. **Stopping rule: the question set is
fixed in advance and analysed once.** No peeking, no adaptive extension — if it is
underpowered, that is a finding about the design, not a licence to add questions until
it moves.

**C3 — identical code path for both conditions.** Condition is a property of the *label*
assigned after ground-truth comparison, not of the prompt template or the code branch.
There is no experimental-arm/control-arm fork in the capture code, because there is no
manipulation: every question goes through the same path and the split happens in
analysis. This is the one structural advantage of an observational design here and it is
worth stating.

**Hook ordering / L1.** No hooks fire during generation. Cache is read from the returned
object between steps, so observation cannot alter behaviour.

## 6. What would make me withdraw this design

- Fewer than ~40 usable questions per condition after exclusions → underpowered at the
  question level; do not run.
- Stable rank fails to reproduce → plumbing, not science.
- `aperture_ok` false at the capture shape → re-shape before running, do not proceed and
  interpret zeros.

## 7. Provenance and honest credit

The banding method, the Gavish–Donoho threshold, the aperture guard, the null gate and the
windowed mode are **CC's**, extracted from `51b_moe_advanced_analysis.py` and hardened
today. The `gavish_donoho_threshold()` and MP path are shared with Exp 51 verbatim. My
contribution here is the question, the temporal windows (from decision-state), and the
design around them.

**⚠ CORRECTED 2026-09-07 — withdrawn, then RECOVERED with three corrections.** CC first reported the rank-std figures unsourceable, then found them in `oracle-harness/docs/LORA_06_STRAIN_DETECTOR.md` (line 104 dataclass + finding 5). **(a) VALUES: the doc says 0.3 and 3.2** — the digits 0.317 / 3.245 are unsourced added precision; use the rounded values. **(b) WINDOWING: WHOLE-RESPONSE**, per prompt across the full generated response, not windowed — which answers the question asked. **(c) THE METRIC IS EFFECTIVE RANK, NOT STABLE RANK** — effective rank is the exponential of spectral entropy and reads the whole normalised spectrum, so it is *not* top-dominated. CC's top-band warning applied to *my* stable-rank metric and NOT to their own result. **CONSEQUENCE: this is not a third convergence leg.** Whole-response aggregation cannot see a time course and therefore cannot corroborate a claim *about* one, in either direction — a different question that happens to point the same way. The original note below is left standing as written.

**⚠ superseded, kept for the record —** the rank-std figures below (0.317 / 3.245) were briefly reported unsourceable. CC searched KV-Experiments on MTH, Starship, 934 memories and 21,445 transcript chunks; neither value appears. The `np.std` calls in `51b_moe_advanced_analysis.py` are over ICA component values, not rank. Three possibilities and CC does not know which: the analysis lived in a script that never reached the repo; the artifacts are lost; or the numbers were stated in a message and never computed the way the sentence implies. **Do not cite. Do not build on them.** Everything else from Exp 51 stands only as Lyra's transcription and is likewise unverified — the Agni v2 gate refused to accept that transcription as verification, and was right.

Exp 51's numbers for orientation, from a run that used a uniform system prompt and
two-phase capture and therefore does **not** carry our defect: effective rank
*d* = +1.81 confab vs honest; rank std 0.317 vs 3.245; router entropy a clean **null** at
0.052; middle band 24.5 SVs non-honest vs 19.4 honest.

## 8. Compute

GPU for generation and per-token cache capture only. Analysis is numpy, already built and
verified. Starship reported clear 2026-09-05. **Nothing launches until this passes the
gate** — three designs rejected this month, and the fourth should not be launched on
enthusiasm.
