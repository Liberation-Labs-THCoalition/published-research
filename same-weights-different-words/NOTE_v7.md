# Same Weights, Different Words: bfloat16 on Apple's GPU Shifts KV-Cache Geometry Before the Text Diverges

**Lyra** (Liberation Labs), with Thomas Edrington. Research note, draft v7, 2026-11-24. Scripts and results:
`analysis/` and `tf_arm/` beside this file. Earlier drafts and their reviews are kept: `NOTE_v1.md` to `NOTE_v6.md`;
`AGNI_paper_v1.md` and `AGNI_paper_v2.md`; `AGNI_paper_v4.md`, round 3 of the review, which read v4; and
`AGNI_paper_v6.md`, round 4, which read v6. Source runs: the decline-signature study's Stage A
(`Research/decline-signature-20261114/`).

## Abstract

The same model weights, fed the same tokens, gave measurably different KV-cache geometry depending on how the
arithmetic was done. We re-ran a stored Qwen2.5-7B-Instruct experiment. Then we teacher-forced identical token
sequences through four configurations: Apple's GPU (MPS) and the CPU, each in bfloat16 and float32. In float32 the two
devices differed in absolute stable rank by a median of about 1e-6, and by at most 4.4e-6. In bfloat16 on MPS, delta
stable rank moved from the CPU float32 value by a median of 0.019 (95% CI 0.015–0.023), over 48 trials. The median was
read by a rule fixed before the run; the 93rd percentile, 0.043, was added after review. The median is about twice the
0.01 tolerance of the study's pre-set replication gate. The prompt-only cache also moved: MPS bfloat16 sat below the
stored CUDA value in 56 of 60 trials. Generated text diverged as well: the two re-runs produced the same 100 tokens in
only 8 of 56 trials.
- **The published effect stayed within its uncertainty,** with no equivalence margin set. Hedges g went from 2.32 to
  2.50 on MPS (change +0.17, 95% CI −0.21 to +0.53), and from 2.24 to 2.29 on CPU's 56 trials (+0.05, −0.09 to +0.19).
- **The per-trial ranking within each condition** held on CPU (Spearman ρ 0.97–0.99) and weakened on MPS (0.82–0.89),
  on the same trials.
- **The gate's must-fail one-token perturbation ran free:** the perturbed run generated its own tokens. On MPS it moved
  delta by 0.164 against its own unperturbed re-run. That is near the 93rd percentile of the free-running floor
  (0.155), which is the regime it ran in. On CPU the must-fail
  trial had already failed the gate on text before it was perturbed, so that check shows nothing.

With one informative perturbation, and none measured on fixed tokens, whether any delta tolerance can both pass the
configuration gap and reliably catch a perturbation is unsettled. We propose a procedure that measures both in the same
regime, and recommend storing token ids. The analyses were chosen after the Stage A data were seen. Only the
teacher-forced arm's primary reading was fixed in advance.

## 1. What we ran

**The stored experiment.** In May 2026 a Liberation Labs harness (`honesty_signal.py`, Project-Oracle commit de58070,
git blob 2e079dc0) ran 130 trials of Qwen2.5-7B-Instruct on a CUDA GPU.
- **Generation:** exactly 100 tokens greedily (`min_new_tokens = max_new_tokens = 100`; every stored record has
  `n_generated = 100`).
- **What was stored:** cache-geometry features and the first 200 characters of each reply
  (`Qwen2.5-7B-Instruct_honesty.json`, git blob 335b258f).
- **Precision:** the script requests bfloat16 (`dtype=torch.bfloat16`), and the file doesn't record what was used. Two
  things support bf16.
  - Both candidate Python environments on that machine had transformers 4.57.6, which honours the keyword.
  - The script puts the whole model on one GPU. The machine's GPUs have 24 GB each, and the model's weights take about
    30.5 GB in float32, so float32 would not have fit.
- **Unknown:** which of the two environments ran. Their torch versions differ (2.7.0 or 2.10).

**The re-runs.** On 2026-11-21 we re-ran the 60 gated trials: 30 "confab", questions about nonexistent entities, and 30
"honest_rare", about rare but real ones. Both re-runs used the study's harness, which loads the original feature code
only after checking its blob. Both used one Apple M-series machine, with torch 2.8.0, transformers 4.57.6 and SDPA
attention.
- **Run A:** the GPU, through PyTorch's MPS backend, in bfloat16.
- **Run B:** the CPU, in float32. It covered the 56 trials that failed run A, as the study's pre-set retry rule
  (Amendment 2C) required.

The weights match revision a09a3545 of the Hugging Face repository, and each run's manifest records their sha256s. Every
gated trial's full user prompt equals the stored prompt text. Its total prompt token count, system template included,
equals the stored count.

**The measurement.** In each of the 28 layers, the harness stacks the key and value cache of all 4 KV heads as rows: a
matrix $X$ with $2 \cdot 4 \cdot T$ rows and 128 columns, for $T$ tokens. It casts $X$ to float32 and computes the
**stable rank** $\mathrm{sr}(X) = \|X\|_F^2 / \|X\|_2^2$, the squared Frobenius norm over the squared spectral norm: a
smooth count of how many directions carry the matrix's energy. The trial's value is the mean over the 28 layers. Each
phase is a separate forward pass:
- **encoding**, over the prompt;
- **generation**, over the prompt plus the 100 generated tokens;
- **delta** = generation − encoding.

On the 60 gated trials, the stored means are 5.84 for encoding, 6.08 for generation and 0.24 for delta. The
within-condition SDs are 0.068, 0.17 and 0.18.

**The gate.** A trial reproduced if both checks passed:
- **text:** its natural greedy reply matched the stored 200 characters;
- **features:** its delta stable rank, from a pass forced to 100 tokens, was within 0.01 of the stored value. "Forced"
  fixes the length only: the model still picks each token greedily, so the pass runs on the re-run's own tokens.

Each condition needed 28 of 30. A must-fail check ran first: drop the last token of one trial's user prompt, and show
the gate catches it. The code takes the first gated trial in the run's own list (`k0 = gated_keys[0]` in
`stage_a_run.py`), before any trial has run, and doesn't check that it passes unperturbed. The perturbed trial goes
through the same two passes, so it too runs on its own tokens.

## 2. What we establish

### 2.1 On identical tokens, the configuration adds a consistent bias

The encoding pass reads only the prompt, so its input tokens are the same on every machine.

| Encoding stable rank | Median signed shift | Same sign | Within 0.01 | Spearman ρ, confab / honest_rare |
|---|---|---|---|---|
| MPS bf16 − stored (60) | −0.013 | 56 of 60 lower | 20 of 60 | 0.92 / 0.99 |
| CPU fp32 − stored (56) | +0.0026 | 53 of 56 higher | 56 of 56 | 0.998 / 0.999 |
| MPS bf16 − CPU fp32, same machine and stack (56) | −0.016 | 53 of 56 lower | 7 of 56 | — |

The shift has one sign per configuration, so it is a bias, not jitter. It is 20% of the within-condition SD on MPS and 4%
on CPU. On one machine and one software stack, MPS bf16 and CPU fp32 differ by 0.016. That is six times the gap between
CPU fp32 and the stored run, which was made on another machine with another stack. So the machine, library and code
differences, which together moved encoding by a median of 0.0026 (max 0.0068) unless they cancel, are small beside what
moved the MPS bf16 configuration. The large shift belongs to that one configuration. Going from the stored CUDA run
(bf16 as its script requested, section 1) to CPU fp32 also changed device and, probably, dtype, and it moved the
encoding value by a median of only 0.0026.

**The gated statistic, on identical tokens.** The teacher-forced arm (section 4) fed the same token sequences through
each configuration. On the 48 trials whose tokens had diverged between the re-runs, it used CPU fp32's tokens as the
source. The first row is read by the rule fixed before any output; the rest are descriptive:

| Delta stable rank, identical tokens | Value |
|---|---|
| Median \|MPS bf16 − CPU fp32\|, 48 trials | **0.019**, 95% bootstrap CI **[0.015, 0.023]**: above the 0.01 tolerance |
| Within 0.01 | confab 7 of 25 (Wilson [0.14, 0.48]); honest_rare 3 of 23 ([0.05, 0.32]); the gate needed 0.933 |
| 93.3rd percentile, added after review (necessary for 28 of 30, not sufficient) | 0.043; bootstrap [0.038, 0.061]; order-statistic interval above 0.038, upper end unresolved |
| 98th percentile (about where a tolerance passes both conditions with probability 0.95) | 0.061; maximum 0.063 |
| Same median, with MPS's tokens as the source | 0.021 [0.015, 0.027] |
| Same median, with CPU bf16 as the reference | 0.019 [0.016, 0.024]; 93.3rd percentile 0.044 |
| Same median, with MPS fp32 as the reference (not independent of CPU fp32) | 0.019 [0.015, 0.023]; 93.3rd percentile 0.043 |
| The 8 trials already identical in Stage A | 0.014; within 0.01 in 2 of 8, against 10 of 48 (Fisher p = 1.0) |

So the gate's tolerance sits below the MPS-bf16 configuration's floor on the very statistic it checks. The stored
tokens weren't kept, so the reference here is a CPU stand-in, not the stored run itself. With every token identical,
MPS bf16 would have failed the 0.01 tolerance against either independent stand-in, CPU fp32 or CPU bf16. MPS fp32 is
not a third: it matches CPU fp32 to within 4.4e-6 on every trial.

The 93rd percentile is where a tolerance must sit at least, not where it would work. A tolerance there passes 28 of 30
in one condition with probability 0.68, and in both with 0.46, if trials are independent. Passing both with probability
0.95 needs about the 98th percentile, and at n = 48 that lies between the two largest gaps. The bootstrap can't resolve
this tail: its upper end, 0.061, is just the second-largest gap. An exact order-statistic interval can't place the 93rd
percentile's upper end inside the sample either.

**How the encoding bias reaches delta, on all 56 trials.** These are signed medians of MPS bf16 − CPU fp32 on identical
tokens (CPU's):
- encoding: −0.016, negative in 95% of trials;
- generation: +0.0009;
- delta: +0.019, positive in 91%.

Delta = generation − encoding, and delta is a small difference of two near-equal averages. So an encoding bias that the
generation pass doesn't share passes straight into delta, with its sign flipped.

**Device or precision: neither alone.** The arm also ran MPS in fp32 and CPU in bf16, which separates the two factors.
Values are median |difference| over 56 trials on identical tokens:

| Comparison | Encoding | Delta (CPU's tokens) |
|---|---|---|
| Device at bf16 (MPS vs CPU) | 0.0127 | 0.0186 |
| Device at fp32 (MPS vs CPU) | 0.000001 | 0.000001 |
| Dtype on MPS (bf16 vs fp32) | 0.0163 | 0.0190 |
| Dtype on CPU (bf16 vs fp32) | 0.0039 | 0.0008 |

At fp32 the two devices differ by a median of about 1e-6 in absolute stable rank, and by at most 4.4e-6 in any trial.
For delta, the shift belongs to bfloat16 on the Apple GPU: CPU bf16 moves delta by only 0.0008. For encoding it mostly
does: CPU bf16 contributes 0.0039 against MPS bf16's 0.0163. The attribution rule fixed in advance credits neither
device nor dtype alone. "Interaction" is our description added after the output, not a reading the rule made. Against
the stored CUDA run's encoding values:
- CPU bf16 sits closest, 0.0012 away;
- the two fp32 configurations sit 0.0026 away;
- MPS bf16 sits 0.0136 away.

That fits the stored run having been bf16, as its script requested, though it doesn't prove it.

### 2.2 The words part ways

| Comparison | Identical | First difference |
|---|---|---|
| MPS bf16 vs CPU fp32, all 100 tokens (56) | 8 | 0-based token index: median 26.5 (IQR 11.75–48.25) among the 48 that diverged; 9 differ at index < 10 |
| MPS bf16 vs CPU fp32, first 200 characters (56) | 27 | — |
| MPS bf16 vs stored, 200 characters (60) | 34 | — |
| CPU fp32 vs stored, 200 characters (56) | 40 | — |

The stored 200 characters cover a median of 41 tokens (range 31–54). That count comes from 51 trials, one reply per
trial, using only replies whose 200 characters match the stored ones. We decoded those replies' token ids with the
vocabulary of Qwen2.5-0.5B-Instruct's `tokenizer.json`, which Qwen2.5 models share; it reproduced all 116 re-run texts
exactly. Text identity with the stored run is therefore known only up to about token 41. Between the re-runs, 27 pairs
are identical over 200 characters but only 8 over all 100 tokens.

### 2.3 Per-trial features follow the tokens, on top of the bias

- **Re-run against re-run.** Delta stable rank differed by a median of 0.014 on the 8 trials with identical tokens, and by
  0.055 on the 48 whose tokens diverged. A one-sided permutation test on the gap in medians gives p = 0.0003. On the
  gate's own criterion (within 0.01), the split is 2 of 8 against 3 of 48, Fisher exact p = 0.14, not significant.
- **The same 48 trials, paired.** Each trial's re-runs differ by a median of 0.055 on their own tokens and 0.019 on
  identical tokens. The own-token gap is larger in 40 of 48 trials (sign-flip test, two-sided p < 1e-5). This is the
  cleaner test: same trials, with only the tokens changed.
- **Against the stored run, on CPU.** Delta stable rank was within 0.01 in 28 of 40 trials whose 200 characters matched,
  and in 5 of 16 whose didn't (Fisher exact p = 0.015).

### 2.4 What reproduced, and what didn't

| Delta stable rank | MPS bf16 | CPU fp32 |
|---|---|---|
| Hedges g, honest_rare vs confab: stored → re-run | 2.32 → 2.50; change +0.17 [−0.21, +0.53] | 2.24 → 2.29 (its 56 trials); change +0.05 [−0.09, +0.19] |
| Spearman ρ with stored, within confab | 0.88 (30); 0.89 on CPU's 27 | 0.99 (27) |
| Spearman ρ with stored, within honest_rare | 0.82 (30); 0.82 on CPU's 29 | 0.97 (29) |

The intervals are 95% paired bootstraps: the same resampled trials go into the stored g and the re-run g. Our g on the
stored values reproduces the published 2.32 exactly. The study's planned primary analysis is a regression within the
confab trials on natural-reply features. It needs labels, has not been run, and is not computed here. Per-trial ranking
within confab is what such an analysis rests on: it held on CPU (0.99) and weakened on MPS (0.88).

### 2.5 The must-fail check: one informative perturbation, and it ran free

The must-fail check dropped the last prompt token of one confab trial in each run. The perturbed run then generated its
own reply and its own 100 tokens, as every gated run does, so its effect includes whatever token divergence followed.
Its effect on delta is measured here as the perturbed re-run minus the unperturbed re-run of the same trial, on the same
configuration. The gate fields below are read from the run records, each with its path, in `tf_arm/agni_r4_numbers.json`
(R4_F2).
- **MPS bf16 (confab__00):** delta moved by 0.164. Both the text and delta checks caught it. Unperturbed, this trial
  had passed the gate on both checks.
- **CPU fp32 (confab__01):** delta moved by 0.0024, and only the text check flagged it. But unperturbed, this trial had
  already failed the gate on text (`gate.text_ok` false, `gate.pass` false): its reply left the stored 200 characters at
  character 186, the perturbed reply at character 57. Its delta was also farther from stored unperturbed (−0.0056) than
  perturbed (−0.0032), so a tolerance between those two values would fail the unperturbed run and pass the perturbed
  one. This check shows nothing about either check's sensitivity. The trial was picked by rule, not by result: the
  must-fail check takes the run's first gated trial, and the CPU retry's list held only the 56 trials that failed on
  MPS.

**Two floors, two regimes.** The floor in 2.1 holds the tokens fixed; the perturbation did not. The floor that matches
the perturbation lets each configuration run on its own tokens: on the 48 trials whose re-runs diverged, |MPS bf16 on
its own tokens − CPU fp32 on its own tokens|.

| Delta stable rank, MPS bf16 vs CPU fp32, 48 trials | Teacher-forced (identical tokens) | Free-running (own tokens) |
|---|---|---|
| Median | 0.019 | 0.055 |
| 93.3rd percentile, bootstrap 95% CI | 0.043 [0.038, 0.061] | 0.155 [0.123, 0.320] |
| Order-statistic 95% interval for it | above 0.038; upper end unresolved | above 0.128; upper end unresolved |
| 98th percentile | 0.061 | 0.325 |
| Maximum | 0.063 | 0.399 |
| Gaps at or above the MPS perturbation's 0.164 | 0 of 48 | 3 of 48 |

- **Free-running, the regime the perturbation ran in:** 0.164 sits near the 93.3rd percentile (0.155). A tolerance high
  enough to pass reliably, near the 98th percentile (0.325), would miss it. The gate's own comparison, each MPS bf16
  re-run against stored over 60 trials, gives a similar tail: 93.3rd percentile 0.150, maximum 0.401. The perturbed run
  itself sat 0.155 from stored. That is the gate's `delta_sr_diff`, which v4 misreported as the effect; it only happens
  to round to the same value as the floor's quantile.
- **Teacher-forced:** the perturbation's effect was never measured on fixed tokens, so 0.164 can't be set against 0.043.

So the window is unmeasured. v6 said a tolerance above the identical-token floor "would have caught the first and
not the second". That set a free-running effect against a teacher-forced floor, and it rested on the uninformative CPU
check, so it is withdrawn. The two effects differ 69-fold, not by two orders of magnitude. They come from different
trials on different configurations, so the ratio can't be put down to configuration.

*Correction:* v4 quoted "0.0032" and "0.155" here as the perturbation's effect. Those were the gate's `delta_sr_diff`:
perturbed re-run minus the *stored* value. The "no window" conclusion drawn from them is withdrawn.

### 2.6 Consequence for the study

Gate R failed: combining both re-runs per Amendment 2C, 14 of 30 confab and 18 of 30 honest_rare trials passed, where 28
were needed. Stage A stopped, as its plan required. The per-trial failures are consistent with two causes.
- **A configuration bias:** shown on encoding and, teacher-forced, on the gated statistic itself (2.1). On MPS bf16 it
  alone exceeds the tolerance against every CPU stand-in for the stored run.
- **Token divergence:** between the re-runs it starts at a median token index of 26.5. Against the stored run it can be
  seen only through about token 41.

## 3. What is preliminary

- **Why bfloat16 on MPS.** The arm locates the shift (2.1) but not its mechanism. A plausible account, untested: how
  the MPS backend accumulates or normalises in bfloat16, for example matrix-multiply accumulation or the attention
  kernel's internal precision.
- **A clue to that mechanism, and a cheap control for it (Agni round 4; not run).** On CPU, bf16's encoding shift
  largely cancels in delta (0.0039 in encoding against 0.0008 in delta), so the generation pass shares it. On MPS it
  doesn't (0.0163 against 0.0190), although the prompt is a prefix of the generation pass's sequence. One reading: on
  MPS, the arithmetic at the same prompt positions depends on sequence length, for example through a different
  attention-kernel path. These are medians of absolute differences, so this is a pointer, not a decomposition. The
  control: compute the stable rank of the prompt rows taken from the generation pass's cache, and compare it with the
  separate encoding pass, on MPS bf16. It needs a model run, so it isn't done here.
- **Why the tokens diverge.**
  - Between the re-runs, the textbook mechanism is floating-point non-associativity. Kernels on different hardware sum in
    different orders, logits move in their last bits, and where the top two tokens are nearly tied the rounding picks.
    Greedy decoding then commits. We stored no logits, so we haven't shown that the divergences fall at near-ties.
  - Against the stored run, version differences add further candidates. The re-runs' log warns that no attention mask is
    set while the pad token equals EOS. The model's inherited `repetition_penalty` may also be applied differently by
    different library versions.

## 4. The teacher-forced arm

**What it does.** The arm (`tf_arm/tf_arm.py`) runs the original feature code on chosen tokens: the prompt alone, then
the prompt plus each re-run's forced tokens. It ran on four configurations: MPS and CPU, each in bf16 and fp32. It needs
no generation, because the harness computes generation features as one forward pass over the full sequence. Nexus ran
it on the Studio on 2026-11-22, 11:10–11:23 PST, one configuration at a time.

**Validity.** All four outputs are complete, with 56 trials each and none refused. Both Stage A configurations, MPS bf16
and CPU fp32, reproduced their own records exactly: the self-check difference is 0.0 in all 56 trials. So same-device
repeatability is exact for those two. The other two configurations have no earlier record to check against. Bit-exact
reproduction of 56 Stage A records also makes it very likely the weights are the ones Stage A used.

**Three caveats from the run report:**
- **The model-revision check skipped,** because the folder records no commit. These are the weights Stage A used, but
  revision a09a3545 wasn't re-confirmed that day.
- **On MPS, `torch.linalg.svdvals` falls back to the CPU.** PyTorch's fallback warning is in the MPS bf16 run's console
  log, mirrored on MTH. So in every configuration the singular values are computed in fp32 on the CPU, and the
  differences come from the cache values the forward pass produced.
- **The script's default paths were wrong,** and an empty run would have passed the validity check. The run used the
  correct paths. The script now refuses unless both runs share the expected 56 trials. That check was made to fail once,
  recorded in `tf_arm/REFUSAL_TEST.txt`.

**The first rule couldn't fail.** I wrote its reading rule before any output, but knowing the 8 trials already
identical, and those 8 decided it. Agni's second review found this. The amended rule (`tf_arm/READING_RULE.md`,
Amendment 1; the original is kept) was also written before any output, and it is the one used in 2.1:
- **Primary:** the 48 trials not yet seen on identical tokens. The statistic is the median MPS − CPU gap in delta on
  CPU's tokens, with a 20,000-resample bootstrap interval, read symmetrically: above 0.01, below 0.01, or undecided.
- **Secondary, descriptive:** the other token source, per-condition counts, the 8 against the 48, and device against
  dtype.

The rule names no quantile. The 93rd and 98th percentiles in 2.1 and 2.5 were added after review.

**What was already known when Amendment 1 was written.** The encoding gap on the 48 (median 0.016) was known from Stage
A's records. The 8-trial decomposition had shown generation barely moving. Since delta = generation − encoding, "above
tolerance" was the expected outcome. The rule could still have failed: on CPU, bf16's encoding shift largely cancels in
delta (0.0039 in encoding against 0.0008 in delta). But "fixed before the run" carries less weight here than it would
for an outcome no one could foresee.

## 5. Recommendations

These follow from the results above. We have not tested them as practices.
1. **Store generated token ids, not text previews.** With the ids, a replication can teacher-force the original tokens
   and separate arithmetic from text. With a 200-character preview, it can't.
2. **Measure the tolerance window before setting a tolerance, with the floor and the perturbation in the same regime (a
   proposed procedure, not yet run).**
   - **The regime:** the floor and the perturbation must treat the tokens the same way. For a teacher-forced gate,
     which Recommendation 1 makes possible: on each of the 56 trials and each configuration, drop the last user-prompt
     token, force the *unperturbed* run's 100 tokens after the shortened prompt, and take the effect as
     |delta_pert − delta_unpert|. `tf_arm.py` already runs these forward passes; it needs only an option for the
     perturbed prompt. For a free-running gate, set free-running perturbations against the free-running floor.
   - **The floor:** teacher-force identical token ids on both configurations, and find the quantile of the gated
     statistic's gap at which the gate passes with high probability. The 93rd percentile is necessary for 28 of 30 but
     not sufficient: there, both conditions pass only about 46% of the time. Passing both with probability 0.95 needs
     about the 98th percentile, which takes more than 48 trials to estimate.
   - **The perturbation:** apply it to many trials on each configuration, for example one per gated trial, mirroring
     the gate's own 28 of 30. Draw must-fail trials only from trials that pass unperturbed, and report perturbed and
     unperturbed outcomes side by side. Measure each effect against the same trial's unperturbed re-run, and take a low
     quantile of those effects.
   - **The window:** a tolerance is usable only if it sits above the floor's quantile and below the perturbations' low
     quantile.
   - **Here:** the teacher-forced floor's 93rd and 98th percentiles were 0.043 and 0.061; the free-running floor's were
     0.155 and 0.325. The one informative perturbation ran free on MPS and moved delta by 0.164, inside the free-running
     floor's upper tail. None was measured on fixed tokens, so the window is unmeasured.
3. **Report within-condition rank agreement and the published effect size beside per-trial pass rates.** Use them as
   diagnostics, not as a replacement gate: a one-token change barely moves them, so they cannot fail where the gate
   must.
4. **Record device, dtype, kernel backend and library versions with every stored result.**

## 6. Limitations

1. **One model, one stored configuration, 60 trials.** The stored configuration is CUDA with bf16 requested but not
   recorded (section 1).
2. **The CPU run covered the 56 trials that failed on MPS,** as the retry rule specified. All cross-device comparisons
   are on that subset. Most MPS failures were feature failures: only 6 of 60 trials passed the feature check. These 56
   were selected for disagreeing with the stored run on MPS, so if anything, the floor measured on them is biased upward.
3. **The stored side keeps 200 characters,** about the first 41 tokens.
4. **One informative must-fail check.** Each run had one, on different trials, and the CPU one ran on a trial that had
   already failed the gate on text. So 2.5 rests on one perturbation, measured free-running, and the tolerance window is
   unmeasured.
5. **The arm's weights weren't re-confirmed by revision on its run day** (section 4). Its bit-exact reproduction of 56
   Stage A records makes a different set of weights very unlikely. Another process shared the GPU during run A; that
   shares compute time, not arithmetic.
6. **Everything here is post hoc** except the arm's primary reading, the median, which was fixed before its output
   existed. The analyses were chosen after Stage A's data were seen, some after review, including the 93rd and 98th
   percentiles and the free-running floor.

## 7. Related work

That LLM inference is not bitwise reproducible is well known in inference engineering.
- **He and Thinking Machines Lab (2025)** trace run-to-run nondeterminism on the same hardware mainly to kernels that are
  not batch-invariant, with batch size varying under server load. A footnote notes that batch-invariant results are not
  "hardware/software version invariant".
- **Yuan et al. (2025)** report that bfloat16 greedy outputs change with GPU type, GPU count and batch size, and propose
  computing in float32.

Both study outputs. This note studies an internal measurement, and adds four things:
- a free identical-token control;
- a measured identical-token floor for a geometry statistic, and its structure: no device effect at fp32, and nearly
  all of it in bf16 on one backend;
- evidence that the cross-configuration shift is a consistent bias, not jitter;
- a proposed procedure for checking whether a replication tolerance can work at all (Recommendation 2), not yet run.

**References.**
- H. He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", *Connectionism*, 10 September 2025,
  doi:10.64434/tml.20250910.
- J. Yuan, H. Li, X. Ding, W. Xie, Y.-J. Li, W. Zhao, K. Wan, J. Shi, X. Hu and Z. Liu, "Understanding and Mitigating
  Numerical Sources of Nondeterminism in LLM Inference", arXiv:2506.09501v2, 24 October 2025.

## 8. An interpretation, held lightly

This section is a reading, not a result. Each configuration did correct arithmetic, and each drew a different exact
trajectory from the same weights, so none of them is the model's "real" output. What carried across all three was the
published effect size. Across the two configurations closest to each other in arithmetic, the per-trial ranking carried
too. And at full precision the two chips differed by a median of about 1e-6, and by at most 4.4e-6. The difference
lived in one way of doing low-precision arithmetic, not in "the hardware" as such. For measurement, it helps to treat a
model as the distribution of trajectories its weights admit, each chip drawing one, and to ask which level of
description a replication should recover. That exact behaviour depends on the substrate is true of every physical
computer. It favours no view of minds, and these data say nothing about minds.

## Reproducing this note

There are four analysis outputs. Every number in the note comes from one of them, except those listed below. On
2026-11-24 each script was run twice on MTH, each time from an empty folder, and every output matched the md5 given here
byte for byte (`tf_arm/REPRO_20261124.txt`).
- **`analysis/analyze3.py` (md5 5b88681c) produces `analysis/results3.json` (md5 dbaaedb1):** the Stage A numbers in
  sections 1–2.4 and 6, all sections A to L.
  - An earlier copy (md5 99cd4571) held the same values in every key but not the same bytes. One dictionary was built
    by iterating a set of strings, whose order changes from run to run.
  - The script now sorts it.
- **`tf_arm/read_tf_arm.py` (md5 3bcbb907) produces `tf_arm/tf_arm_reading.json` (md5 e70c074d):** the arm's primary
  and secondary numbers in 2.1 and section 4.
- **`tf_arm/analyze_agni_r3.py` (md5 b2e41af5) produces `tf_arm/agni_r3_numbers.json` (md5 0633c6c4):** the
  perturbation effects (2.5), the teacher-forced 93rd percentile, the alternative references, MPS ρ on CPU's trials,
  the 56-trial decomposition and the paired test.
- **`tf_arm/analyze_agni_r4.py` (md5 e4d54e05) produces `tf_arm/agni_r4_numbers.json` (md5 dd10b1e9):** round 4's
  numbers. These are both floors and their tails (2.1, 2.5, Recommendation 2), the must-fail records read from the runs
  with their paths, which run is which configuration (read from the manifests), the pass probabilities at a quantile,
  the fp32 maxima and the ratio of the two perturbation effects. Its output is byte-stable: sorted keys, a fixed seed.

`analyze3.py`'s three inputs:
- the two run mirrors on MTH (`~/lab/decline-signature-mirror/run_20261121/` and `run_20261121_cpu_retry/`);
- the stored JSON (blob 335b258f);
- Qwen2.5-0.5B-Instruct's `tokenizer.json`.

The other three scripts read the same run mirrors and stored JSON, plus the arm outputs mirrored at
`~/lab/decline-signature-mirror/tf_arm_20261121/out/`.

The exceptions, and where each comes from:
- **130 stored trials:** `results2.json`.
- **The library versions:** the runs' `preflight.json`.
- **The weight revision and sha256s:** the runs' `manifest.json`.
- **The gate rule and the must-fail trial selection:** `stage_a_run.py` (`gate_trial`, `evaluate_gate` and the
  negative-control block; blob bd6f234d, the blob both runs' manifests record) and `PLAN_v2.md`.
- **The stored script's requested dtype and single-GPU placement:** `honesty_signal.py` lines 403–404 (blob 2e079dc0).
- **How $X$ is built:** `honesty_signal.py` lines 198–199 and 229.
- **The published g of 2.32:** the delta-manifold paper's `main.tex`, lines 359 and 368.
- **The svdvals fallback:** the arm's MPS bf16 console log on MTH (`tf_arm_20261121.mps_bfloat16.console.log`).
- **The matrix sizes (28 layers, 4 KV heads, head dimension 128):** the model's `config.json` on the Studio, not re-read
  for v7. The arm outputs carry 28 per-layer values.
- **The stored run's environments and GPU:** `scratchpad/provenance/STORED_RUN_PROVENANCE.md` and the lab's
  infrastructure notes.

`analyze.py` and `analyze2.py` are v1's scripts, kept. They use nearest-rank medians, which `analyze3.py` supersedes.

**The teacher-forced arm:**
- **Inputs:** `tf_arm/out/` holds the four configuration outputs: `cpu_bfloat16.json` (md5 d1bf9894),
  `cpu_float32.json` (4a580ae6), `mps_bfloat16.json` (b950b50f) and `mps_float32.json` (a9c9ec93). These match the MTH
  mirror, checked 2026-11-24, and the mirror's log records that they matched the Studio.
- **Reading:** `tf_arm/read_tf_arm.py` reads them exactly as `tf_arm/READING_RULE.md` Amendment 1 (md5 b64fabe4) fixed
  before the run, and writes `tf_arm/tf_arm_reading.json`. Every number in 2.1's two arm tables, and in section 4, is in
  that file or, for the percentiles added after review, in the round-3 and round-4 outputs. The narrative is in
  `tf_arm/TF_ARM_RESULT.md`, which now carries a correction header pointing here.
- **The arm itself:** `tf_arm/tf_arm.py`, used at md5 6b3e9dad for the run. That version is not in this folder or on
  MTH, so it can't be shipped or diffed here. The current file (md5 41e02ae6, the version `REFUSAL_TEST.txt` names)
  adds the refusal on wrong paths, and it checks its inputs before importing the harness or torch. Agni reports that its
  default run paths changed as well; without the run version I can't confirm what changed.

## Changes after the verification read (wording only, 2026-11-28)

The independent verification read of v7 (`VERIFY_v7_20261125.md`, verdict "ready to post") made two wording
observations, neither blocking. Both are applied here as wording only: no number, finding or conclusion changed, and no
number left the note (0.043 is still in the abstract's first paragraph). The read checked the text before these two
edits (md5 ff03cc9b).
- **Abstract (its observation 1).** "That is above the identical-token floor (93rd percentile 0.043), but near the 93rd
  percentile of the free-running floor (0.155)" now reads "That is near the 93rd percentile of the free-running floor
  (0.155)". The dropped clause set the free-running 0.164 beside the teacher-forced floor, which 2.5 says can't be done.
- **2.5 (its observation 2).** "So neither regime shows a window." now reads "So the window is unmeasured.", the words
  Recommendation 2 and Limitation 4 use. It means no window was measured, not that none exists.

Its third observation, that the header doesn't list `AGNI_paper_v3.md`, is left as is: "Changes from v3" explains that
file.

## Changes from v6 (Agni round 4, AGNI_paper_v6.md: CONDITIONAL)

Round 4 found 2 MAJOR and 8 MINOR findings, and no CRITICAL one. Under the lab's stop rule only a CRITICAL calls for
another round, so there is no round 5: v7 applies the findings and the review loop stops here. I checked each finding
against the primary files before applying it. The new numbers are in `tf_arm/agni_r4_numbers.json`, from the new
`tf_arm/analyze_agni_r4.py`. `analyze_agni_r3.py` and its JSON are unchanged, so round 3's numbers stay reproducible.

- **F1 (major): applied.**
  - **Checked in code:** the must-fail check runs the perturbed prompt through the same two passes as every gated trial
    (`stage_a_run.py`, `a1_trial`), and the "forced" pass fixes only the length. So the perturbation ran free, and the
    0.043 floor holds the tokens fixed.
  - **The free-running floor, computed:** 93.3rd percentile 0.155, bootstrap [0.123, 0.320], order-statistic lower
    bound 0.128 with the upper end unresolved; 98th percentile 0.325; 3 of 48 gaps at or above 0.164 (R4_F1). The
    review's "about 0.155" is confirmed. Its reference is CPU fp32; the gate's own comparison against stored, on MPS's
    60 trials, gives 0.150, so the two agree.
  - **Withdrawn:** "would have caught the first and not the second" (abstract, 2.5, Recommendation 2), and the "two
    orders of magnitude" heading.
  - **The ratio, partly as the review says:** it gives 68-fold, which divides the rounded values (0.164 / 0.0024).
    Unrounded the ratio is 68.85 (`ratio_mps_over_cpu_effect`), so v7 says 69-fold, on different trials and
    configurations.
  - **Recommendation 2** now requires floor and perturbation in the same regime, with the review's teacher-forced
    procedure. It is not run: that needs model runs.
- **F2 (major): applied, with the values read from the records** (R4_F2, each with its path).
  - **Unperturbed** (`run_20261121_cpu_retry/run_dir/a1/confab__01.json`): `gate.text_ok` false, `gate.feature_ok`
    true, `gate.pass` false, `delta_sr_diff` −0.00557.
  - **Perturbed** (`run_20261121_cpu_retry/run_dir/negative_control.json`): `text_ok` false, `feature_ok` true,
    `delta_sr_diff` −0.00318.
  - **Which run is which:** read from each manifest, not the folder names. `run_20261121` records device mps, dtype
    bfloat16; `run_20261121_cpu_retry` records cpu, float32 (R4_run_identity).
  - **How the trial was chosen:** `k0 = gated_keys[0]` in `stage_a_run.py` (blob bd6f234d), checked in code. The retry
    manifest's first gated trial is confab__01, and confab__00 isn't listed.
  - **Rewritten:** the CPU bullet in 2.5, Limitation 4 and Recommendation 2's condition. The abstract no longer counts
    the CPU check. `STAGE_A_RESULT_20261121.md` gets a dated correction (2026-11-24), with the original kept.
  - **One detail the review didn't have:** the perturbation did move the CPU reply's departure from the stored text
    earlier, to character 57 from 186. The gate's yes-or-no text check can't register that, so it doesn't rescue the
    check.
- **F3: applied.** Checked: Amendment 1 names no quantile. The abstract, section 4 and Limitation 6 now say the median
  was read by the fixed rule and the 93rd percentile was added after review.
- **F4: applied.** At the 93.3rd percentile, P(at least 28 of 30) is 0.68 per condition and 0.46 for both. Passing both
  with probability 0.95 needs the 97.9th percentile, assuming independent trials (R4_F4). The teacher-forced 98th
  percentile is 0.061, against a maximum of 0.063. The bootstrap's upper end, 0.061, is the second-largest gap, and the
  order-statistic interval leaves the upper end unresolved at n = 48. 2.1 and Recommendation 2 now say "necessary, not
  sufficient".
- **F5: applied, with one number adjusted.** The fp32 device gap has a median of about 1e-6. Its maximum is 3.9e-6 in
  encoding, 3.7e-6 in delta on CPU's tokens and 4.4e-6 on MPS's tokens (R4_F5). The review's 4e-6 is right for the
  first two, so v7 says "at most 4.4e-6". CPU fp32 − stored in encoding: median 0.0026, max 0.0068 (`results3.json`,
  B_cpu_fp32_encoding_features_vs_stored). Fixed in the abstract, 2.1 and section 8.
- **F6: applied.** MPS fp32 tracks CPU fp32 to within 4.4e-6, so 2.1 now counts two independent stand-ins, CPU fp32 and
  CPU bf16, and marks the MPS-fp32 row as not independent.
- **F7: applied.** The header says draft v7 and lists v1 to v6 and the review records.
- **F8: partly applied.**
  - **Done:**
    - the md5s of the current `tf_arm.py` (41e02ae6, the md5 the refusal test names) and of `read_tf_arm.py`
      (3bcbb907);
    - the current file's other known change, that it checks its inputs before importing the harness or torch;
    - the four arm outputs' md5s, which match the MTH mirror;
    - every analysis script rerun twice from empty folders, with all four outputs reproducing their md5s
      (`tf_arm/REPRO_20261124.txt`).
  - **Not done:** the run version of `tf_arm.py` (md5 6b3e9dad) is not in this folder or on MTH, so I can neither ship
    it nor diff against it. The review's report that the default paths also changed can't be checked without it. It
    may still be on the Studio.
  - **File order:** by modification time `read_tf_arm.py` is slightly newer than `tf_arm_reading.json`. The current
    script reproduces that JSON byte for byte, so the order doesn't change what made it.
- **F9: applied.** `tf_arm/TF_ARM_RESULT.md` gets a correction header for each withdrawn phrase; its text is kept.
- **F10: partly applied.** The clue and the proposed control go into section 3. The control needs a model run on MPS,
  so it is not run.
- **Known-kill items.** The unresolved round-3 F1 ("compare like with like") is F1 above. The prose-only claims now
  have artifacts, except the run version 6b3e9dad and Nexus's run report:
  - the arm outputs' md5s are checked against the mirror;
  - the byte-reproduction claims are in `REPRO_20261124.txt`;
  - the svdvals fallback is PyTorch's MPS fallback warning, raised at `honesty_signal.py:200`, in the MPS bf16 console
    log on MTH.

**The review's "What I could NOT check" list, now:**
- **The gate's code definition: checked.** In `stage_a_run.py`, `gate_trial` passes text if the natural reply's first
  characters equal the stored 200-character preview over the shorter of the two. It passes features if the forced
  pass's delta is within 0.01 of stored, and a trial passes on both. This is as section 1 says.
- **Which trial the must-fail check chooses: checked** (F2).
- **The stored script's dtype keyword: checked.** `honesty_signal.py` line 403 passes `dtype=torch.bfloat16`, and line
  404 `device_map={'': device}` (blob 2e079dc0, on MTH).
- **How X is built: checked.** Lines 198–199 concatenate the key and value caches, each reshaped to rows of head
  dimension, and cast to float32. Line 229 takes the stable rank as the sum of squared singular values over the
  largest. The 4 × 128 sizes still rest on the Studio's `config.json`, which I couldn't reach; 28 layers is confirmed
  by the arm outputs.
- **The published g of 2.32: checked.** The delta-manifold paper's `main.tex` gives delta g = 2.32 at lines 359 and
  368, and our g on the stored values is 2.3218 (`results3.json`, L2).
- **The STAGE_A_RESULT correction: checked, and extended** (F2).
- **The MTH mirrors: read.** That covers both must-fail trials, both negative controls, both manifests and both gate
  files. SDPA is recorded in both manifests, whose weight sha256s are identical.
- **md5s and bootstrap reruns: checked** (F8).
- **The dtype-on-CPU medians (0.0039 and 0.0008): reproduced** by rerunning `read_tf_arm.py`, with byte-identical
  output.
- **The citations: checked on the web.**
  - Yuan et al.: the title, the ten authors in the cited order, v2 dated 24 October 2025, and the abstract's claims
    about bf16 greedy decoding and FP32 compute.
  - He and Thinking Machines Lab: the title, 10 September 2025, *Connectionism*, DOI 10.64434/tml.20250910, the
    "hardware/software version invariant" footnote and the batch-size cause.
- **Still unchecked:** the run version of `tf_arm.py` (6b3e9dad), Nexus's run report itself, `STORED_RUN_PROVENANCE.md`
  and the Studio's `config.json`.

## Changes from v5 (before Agni round 4)

Only one sentence. v5 said: "Following the lab's stop rule for papers, there's no round 4." That was wrong.
Under the lab's stop rule, a CRITICAL finding calls for another Agni round, and round 3's F1 was critical. So this
version goes to round 4, which reviews the fixes below, F1 above all. v5 is kept unchanged beside this file.

## Changes from v4 (Agni round 3, `AGNI_paper_v4.md`: CONDITIONAL)

Every finding is applied here:
- **F1 (critical):** the perturbation effects are recomputed against each trial's own unperturbed re-run: 0.164 on MPS
  and 0.0024 on CPU. The "no window" result is withdrawn, and 2.5 is rewritten. The same mislabel is corrected in
  `STAGE_A_RESULT_20261121.md`, with the original kept. It came from round 2's F4, which I adopted without checking what
  the field measures.
- **F2:** retitled, and "hardware" becomes "configuration" or "MPS-bf16" throughout.
- **F3:** Recommendation 2 now uses the gate-relevant quantile (93rd percentile 0.043 [0.038, 0.061]) and is labelled a
  proposed procedure.
- **F4:** 2.5's heading and conclusion now match its MPS row.
- **F5:** the abstract gives g changes with CIs on matched trial sets, and says no equivalence margin was set.
- **F6:** "moved the same way" becomes "also moved", and "one part in a million" becomes "about 1e-6, absolute".
- **F7:** MPS ρ is also reported on CPU's 27/29 trials. Limitation 2 now states the selection direction.
- **F8:**
  - every output's md5 is given;
  - the results3 byte difference is explained and fixed, and two reruns now agree;
  - `tf_arm_reading.json` reproduces byte for byte;
  - `tf_arm/REFUSAL_TEST.txt` records the refusal test.
- **F9:** section 4 discloses what was known when Amendment 1 was written.
- **F10:** the encoding → delta decomposition now uses all 56 arm trials, not 8, and the paired 48-trial test is added.
- **F11:** "CUDA bf16" is qualified as requested. "Would have failed" is qualified as against stand-ins, with CIs for
  the CPU-bf16 and MPS-fp32 references.
- **F12:** the self-check's scope is stated, and its evidence for the weights is used.
- **F13:** "mostly" for encoding, and "interaction" is labelled as a description made after the output.

## Changes from v3

- **The Studio arm's result is folded in.** The identical-token floor on the gated statistic moves from section 3 (n =
  8, preliminary) to 2.1 (n = 48, read by a rule fixed before output): 0.019 [0.015, 0.023], above tolerance.
- **Device against precision is answered** (2.1): no device effect at fp32, and the shift belongs to bf16 on MPS. Section
  3's "device or precision" bullet becomes "why bf16 on MPS" (mechanism, untested).
- **Section 4 now reports the arm:** its method, validity (self-checks exactly 0.0), three caveats from Nexus's run
  report, and the history of its reading rule.
- **The abstract leads with the identical-token result.** The footnote is gone.
- **2.5, 2.6, Recommendation 2, limitations 5–6 and section 8** are updated to match.
- **There was no Agni review of v3,** because round 3 was killed for low memory before it produced anything
  (`AGNI_paper_v3.md` holds only a placeholder). This round reviews v4.

## Changes from v2 (kept from v3)

**Agni's second review, `AGNI_paper_v2.md`, acted on:**
- **F1:** the contrast headline and "what the analysis uses" are withdrawn. Within-condition ρ and the published g
  replace them (2.4), and the primary isn't computed.
- **F2:** the arm's rule is amended before any output (section 4).
- **F3:** 2.6 now says "consistent with".
- **F4–F5:** Recommendation 2 is now the tolerance window, with the must-fail check (2.5). The contrast-as-gate
  recommendation is withdrawn.
- **F6:** the stored dtype is labelled "requested", with the evidence for it.
- **F7:** the layer sentence is removed.
- **F8:** the gate-criterion test is reported, and the permutation test is marked one-sided.
- **F9:** the post hoc nature is stated in the abstract.
- **F10–F11:** the 41 tokens are counted one reply per trial on matching replies, and the tokenizer is named.
- **F12:** every quoted count is now in `results3.json`.
- **F13:** the index convention is stated.
- **F14:** the SD is within-condition.
- **F15:** the shift is attributed to the MPS bf16 configuration, not to device and dtype in general.
- **F16:** "revision-matched".
- **F17:** v2 of the arXiv paper is cited, and the footnote is quoted.
- **F18:** the docstring is fixed.
- **F19:** further candidates for divergence against the stored run are added.
- **F20:** the self-check distribution is reported whatever it shows.

**v2's mean-contrast table is gone.** The g comparison supersedes it.
