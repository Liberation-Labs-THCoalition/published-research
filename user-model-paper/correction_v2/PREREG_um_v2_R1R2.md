# Pre-registration: user-model correction, R1, R2 and R7 on one encode

**Author:** Lyra. Drafted for Lyra by a Claude Code agent from 2026-10-18 and revised after Agni rounds 1, 2 and 3.
Lyra reviews, amends and seals it. The decisions in §5.0 are Lyra's own, dated, and made before any R1 or R2 data.
**Status:** DRAFT, REVISED AFTER AGNI ROUND 3 (both scripts CONDITIONAL in rounds 1-3), NOT RUN. Round 3 is the last
review round (§9, Lyra's decision). No GPU time before §9 passes.
**Plan:** `corrections/user_model_v2_run_plan_20261018.md` (R1, R2, R7). **Why:** `corrections/user_model_2026a_correction_plan_20261014.md`.
**SOP:** LAB_SOP §1.4 and §5.1 (save raw data), §3 (architecture), §4 (statistics), §8 (assert the mode), §11 (limits).
**Model:** `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled`, the paper's model. Starship snapshot `ad356102`.
**Code:** `um_v2_encode.py` (one forward pass per story, Starship MPS), `um_v2_analysis.py` (CPU only),
`um_v2_dryrun.py` (CPU dry run). What changed for Agni, item by item: `R1R2_DESIGN_NOTES.md` §8 (round 1), §9
(round 2) and §10 (round 3).
**Sealed predictions:** `R1R2_PREDICTION.md` (Lyra, 2026-10-18 18:02). It was written before rounds 2 and 3 and is not
edited. Its hash is bound into the encode manifest and the result, both scripts check its content, and §5.5 maps its
labels.

---

## 1. Why this run

The paper reports 40.9% on 30 emotions and a valence AUROC of 0.992 from a "W_K projection probe".

- The script behind those numbers, `wk_backtrack_probe_v2.py` ("v2"), applies no W_K (v2:55-74).
- It also differs from the paper's Method on positions, similarity, folds and layers.
- R0 found that TF-IDF on the same stories reaches 37.0% and 0.963.
- v2 saved no per-trial predictions, so its margin over text (TF-IDF) has never been tested.

So this run does three things:
- **R1** re-runs v2 on a fresh encode.
- **R2** runs the probe the Method describes.
- **R7** re-tests the spectral null, because the May files it rested on hold no usable spectral features.

---

## 2. Hypotheses

**H1 (R1, reproduction).** v2's probe, ported line for line, reproduces v2's count on a fresh bf16 re-encode:
|k - 368| <= h. The 368 is read from the hashed `wk_backtrack_v2_results.json`.
- **The setup is v2's:** class-mean key directions, dot product, v2's FWL, GroupKFold(5) by topic, and mean keys at
  model layers 3, 7, 11 and 15 concatenated (v2:37-38, :142-146).
- **h:** v2 ran in bf16 (v2:107). So h is the largest change in R1's count over 30 runs whose pooled keys are
  perturbed per layer by the measured bf16 run-to-run difference.
- **The cap:** if h > 17, R1 is UNINFORMATIVE_BAND_TOO_WIDE.
- **Whether today's weights are v2's** is tested by recomputing v1 from this encode. v1 used the same HF cache as v2,
  on 2026-05-21, and must give 470/900 within its own band.

**H2 (R2, primary).** The Method's probe beats text (TF-IDF, R0 at 10 folds) on the 30 emotions.
- **The probe:**
  - d_e = W_K(r_e - r) from the training fold, with r the decoder-layer output pooled over positions >= 50, as are
    the keys;
  - per-layer cosine averaged over model layers 3, 7, 11 and 15;
  - FWL on the training fold on [log1p n_tokens, 1/n_pooled];
  - GroupKFold(10) by topic.
- **EXCEEDS_TEXT_TFIDF requires all four of:**
  - R2 - R0 >= 3.0 points;
  - McNemar exact p < 0.05 with R2 ahead;
  - the 95% interval of the accuracy difference, bootstrapped over the 300 (emotion, topic) cells, excludes 0;
  - R2's own accuracy beating its cell-level within-topic permutation null at p < 0.05.

**H3 (R7, spectral null).** Frobenius norm, effective rank and entropy of the keys over the first 100 story tokens do
not classify the 30 emotions.
- In none of 17 fixed layer sets (each single layer, plus all 16) is the cell-level within-topic permutation p below
  0.05/17, from 5,000 permutations.
- A set's null counts only if its topic control passes: StratifiedGroupKFold by cell, accuracy > 0.2.
- Each null excludes only accuracies at or above that set's detection threshold: its null's 1 - 0.05/17 quantile.

---

## 3. Design

### 3.1 Data

- **Trials:** 900 from `emotion_bridge_trials.json` (sha256 `fc5938fe...`).
- **Structure:** 30 emotions x 10 topics x 3 sampled stories.
- **Distinct:** all 900 are distinct (asserted).
- **Overlap between siblings:** mean within-prompt token Jaccard 0.178, max 0.578.
- **Rendered input:** `<|im_start|>user\n{story}<|im_end|>\n`:
  - 3 preamble tokens, the story (104 to 200 tokens), then 2 trailing tokens;
  - 109 to 205 tokens in total.

### 3.2 Encode (`um_v2_encode.py`)

- **Matching v2:** load as v2 did (v2:105-109) and render as v2 did (v2:127-133).
  - One forward pass per trial, with `logits_to_keep=1`; the cache is unchanged by that.
- **Saved per trial, at the 16 full-attention layers,** with the model layer ids in every shard:
  - keys and values;
  - three residuals: the layer output, k_proj's input, and the residual entering the layer;
  - each pooled four ways: all positions, positions >= 50, the last position, and the story tokens only;
  - the story's token span;
  - `spectral` and `spectral_n100`. Their column names are recorded as frobenius, eff_rank, entropy and
    top_sv_ratio, and asserted by the analysis.
  - the full K and V in fp16, with subnormal and to-zero counts;
  - the last position's top 20 logits, a determinism aid;
  - provenance, which resume checks.
- **The encode stops** (Agni round 3), each before or during the run, if:
  - the self-check fails;
  - any repeat of trial 0 exceeds the determinism ceiling, a per-layer relative RMS of 2^-8 = 0.0039 (bf16 unit
    roundoff), on any pooled array. Trial 0 is repeated back to back, without hooks, and after the loop;
  - the model commit hash is None;
  - `--prediction` fails its content check (>= 200 bytes, naming R1, R2 and R7);
  - the script's registered constants differ from this prereg's (§12).
- **Fingerprint against May's *bridge* weights** (the default HF cache, May 21 morning; reported, not an R1 gate):
  - the norms ||K_l||_F of 3 bridge prompts, with May's token counts reproduced (98, 94, 95);
  - a teacher-forced witness on May's story tokens inside the temperature-0.7, top-p-0.9 nucleus;
  - the witness's negative control: May's tokens shuffled. At least 25% must fall outside, or the witness is
    uninformative;
  - all of it in bf16 and, after the loop, in fp16.
  - May saved story text, not token ids. So the witness forces re-tokenised ids, chosen where the length equals May's
    generated length (Agni round 3, N7: equal length is not equal ids; recorded as a caveat).

### 3.3 R1, and which weights it is about

- v2's functions are copied verbatim (AST-checked) and its loop is ported line for line.
- **Layer indices (Agni round 3, L4):**
  - v2:37 sets FULL_ATTENTION_LAYERS = range(3, 64, 4), and v2:38 takes `PROBE_LAYERS = FULL_ATTENTION_LAYERS[:4]`,
    which is model layers 3, 7, 11 and 15.
  - v2:142-146 reads `cache.layers[li].keys` for those `li`. Those are decoder-layer ids into the KV cache, not
    `hidden_states` indices, so there is no off-by-one.
- **v2's FWL** is fitted on each fold's own test trials (v2:227). That is label-free and not all-data.
- **The weights (Agni round 3, N4):**
  - May's weights cannot be fetched. The HF repo was history-squashed on 2026-07-07, and v2's cache
    (`/Users/margaret/models/hf_cache`, v2:34) has been deleted.
  - May's bridge run used a *different* cache: the default one, with no `cache_dir` (bridge:1371-1377). So the May
    fingerprint never measured v2's weights.
  - The v2-era artifact that does carry them is v1 (`wk_backtrack_probe.py`). It loaded from the same cache as v2
    (v1:33) at 21:47 on 2026-05-21, 86 minutes before v2. It saved its accuracy (0.52222 = 470/900, sha256
    `42a723eb`), and its inputs can all be rebuilt: this encode's mean keys and W_K, plus May's
    `emotion_vectors.json` (sha256 `3452df23`).
  - **The v1 check** recomputes v1 on CPU, with its own band (the same bf16 perturbations and the same cap of 17).
    REPRODUCED means today's keys and W_K match v2's evening.

### 3.4 R2: the primary cell and its reasons

These are as in round 2:
- **Pool:** positions >= 50 (main.tex:250; the bridge's `hidden[:, 50:]` on this same input).
- **Similarity:** per-layer cosine, which is the Method's 30 features (main.tex:201). The concatenated cosine is
  exploratory.
- **FWL:** [1, log1p n_tokens, 1/n_pooled].
- **Layers:** model layers 3, 7, 11 and 15. These are May's choice, disclosed.
- **Specification curve:** 288 cells, descriptive. Void cells (any NaN) are excluded and counted.
- **Exploratory cells (59):**
  - the residual source;
  - the projected-key target (W_K x, before k_norm and RoPE);
  - the concatenated cosine and the 120-feature variant;
  - the paper-literal FWL;
  - the raw-residual LR baselines.

### 3.5 R0, and a tuned R0'

- R0 is R0's own code with the number of folds as a parameter.
  - At 5 folds it must reproduce `R0_results.json` (labels exact, probabilities within 1e-6).
  - At 10 folds it is R2's comparator: 38.2%, 344/900.
- **R0' (Agni round 3, New-4, exploratory):** C is chosen per outer fold from {0.01, 0.1, 1, 10} by an inner
  GroupKFold(3) over the training topics. Its sentence is in §5.6.

### 3.6 R7

- **Features:** n-matched (the first 100 story tokens), with the column names asserted.
- **Per set (17 fixed sets, m = 17):**
  - constant columns are dropped;
  - FWL on train;
  - 5,000 cell-level within-topic permutations, with the null vector saved;
  - the topic control: StratifiedGroupKFold(10, shuffled), grouped by cell, passing above 0.2.
- **Per-set CONFOUND_SUSPECTED (Agni round 3, New-3).** A set with p < 0.05/17 is CONFOUND_SUSPECTED, not SIGNAL, if:
  - its tie fraction is > 0.5; or
  - its variance kept is < 0.1; or
  - the length covariates alone, classified with the same folds, reach that set's detection threshold.

---

## 4. Pre-registered analysis plan

### 4.1 Primary test

- There is one confirmatory comparison: R2's primary cell against text (TF-IDF, R0 at 10 folds), on 900 paired trials.
  It is decided by the ladder in §5.2.
- **The bootstrap:**
  - estimand: accuracy(R2) - accuracy(R0, 10 folds);
  - unit: the 300 cells;
  - B = 10,000;
  - percentile interval around the observed difference.
- **What the permutation tests (Agni round 3, N3):**
  - The statistic is R2 primary's own 30-way accuracy, with the whole pipeline refit per draw (directions
    re-estimated; no TF-IDF involved).
  - The resampling unit is the cell within topic.
  - It tests R2 against chance and leakage, not R2 > R0.
  - The R2 > R0 inference rests on McNemar and the cell bootstrap.
  - A paired cell-level sign-flip test of R2 > R0 is reported beside them, but not as a ladder rule.

### 4.2 Power: registered and sensitivity

Lyra signed this off on 2026-10-19. The independence value is the option Agni's round-3 analysis review named
(`psi_ind`, item 2). It replaces Lyra's 10-18 choice (§5.0), which was text against text.
- **The registered ψ** is the a priori independence value from R0's accuracy alone, a0 = 0.382:
  ψ(Δ) = (a0 + Δ)(1 - a0) + a0(1 - a0 - Δ), which is 0.479 at +3 points.
- **"Underpowered" is gated on the +3-point power.**
  - Pre-data, conditional on R0's 10-fold vector: power 0.16 at +3 and 0.58 at +5. The gap detectable at 80% power
    is 6.0 points.
  - So any non-significant R2 will read INCONCLUSIVE:underpowered. NOT_DISTINGUISHED is effectively unreachable, and
    that is the honest consequence at n = 900.
- **Sensitivity:** ψ from R0 at 5 against 10 folds (0.083, text against text), ψ from R1 against the R0 file, and the
  MDE for the observed discordant pairs.
- **What the power covers:** the test on prediction-level gains, not a latent cache signal. McNemar's trial
  independence is a second optimism, which the cell bootstrap addresses.

### 4.3 FWL diagnostics, gated

- For the primary: CONFOUND_SUSPECTED (§5.0).
- For R7: per set (§3.6).
- A flag, reported and not a verdict: the covariate-only accuracy above 2x chance, for every pool.

### 4.4 Secondary family (Holm, 4 tests)

S1-S4 are as in round 2 (R1 and R2 against R0, 30-way and valence; S4 carries the 90% vs 80% training-set asymmetry).
The valence probability is a binary LR(C=0.1) on the 10 cosine features (dot for R1), using directions estimated from
the 300-trial subset's training trials.

### 4.5 Exploratory, labelled as such

- R2 against R1 at 10 folds.
- R1 against R0 at 10 folds.
- R1 with train-fitted FWL.
- The grid and the 59 exploratory cells.
- The raw-residual baselines.
- R0'.
- The sign-flip test.
- R7 on all positions.

### 4.6 Saved, and refused

- **Saved:**
  - every per-trial prediction;
  - the full permutation null vectors;
  - the band replicates for R1 and v1;
  - the determinism, fingerprint and witness data;
  - the versions and the input hashes: manifest, trials, R0, this prereg's JSON, the prediction, v2's and v1's
    results, and May's emotion vectors.
- **Refused:**
  - constants different from §12;
  - a placeholder prediction, or one whose hash differs from the encode's;
  - a results file or vectors file not matching its registered hash;
  - a skipped fingerprint or witness;
  - any determinism value above the ceiling;
  - spectral columns in a different order.

---

## 5. Decision rules, fixed in advance

### 5.0 Lyra's decisions, with their reasons

**2026-10-18, before any R1 or R2 data (unchanged):**
- **The H2 floor is 3 points.** EXCEEDS_TEXT_TFIDF needs:
  - R2 - R0 (10 folds) >= 3.0 points;
  - McNemar p < 0.05;
  - a cell-bootstrap 95% interval for the difference that excludes 0.
  - *Reason:* below 3 points a probe adds too little over a plain word classifier to support any claim about what the
    cache holds beyond the words, and 3 points is about the size of the R0-R1 gap the paper rested on.
- **The inconclusive region** is any of: 0 < difference < 3 points; a power shortfall; or McNemar significant with the
  interval including 0. It is reported as INCONCLUSIVE, never as "equal" or "no difference".
- **CONFOUND_SUSPECTED** fires if:
  - covariates alone reach R0; or
  - the FWL-fitted part ("-fitted") does; or
  - a tie fraction is > 0.5; or
  - variance kept is < 0.1.
  - **Where the two thresholds come from:**
    - **Variance kept, 0.1:** the regression literature. It is VIF > 10 (Kutner, Nachtsheim, Neter and Li 2005;
      O'Brien 2007, *Quality & Quantity* 41:673).
    - **Tie fraction, 0.5:** from the mechanism, not the literature or the null world. A feature tied on more than
      half the trials is a step function, the documented reinjection case (196 of 200 tied).
    - **The null world as a false-positive check:** its largest tie fraction is 0.0011 and its smallest variance
      kept is 0.985; neither threshold fires there.
- **Headline wording:** "text (TF-IDF)" always, plus the forbidden sentences in §5.3.

**2026-10-18, superseded on 2026-10-19 (the original text, kept):** "Registered power uses the a priori ψ from R0's
5-fold against 10-fold disagreement, as Agni asks. The R1-based ψ is a sensitivity analysis, labelled so."

**2026-10-19, after Agni round 3, before any R1 or R2 data:**
- **Power.** The registered ψ is the independence value from R0's accuracy, a priori. The 5-vs-10 value is two text
  classifiers, so it overstates power. "Underpowered" is gated on the +3-point power, the floor's own size.
  - *Principle:* the power that gates a null must come from the same pair of classifier kinds as the test, and at the
    effect size the claim needs. That holds whatever the data show.
  - *Cost:* NOT_DISTINGUISHED is effectively unreachable at n = 900.
- **R7 stays confirmatory for H3**, because its consequences for main.tex can be stated cleanly (§5.3).
- **Round 3 is the last Agni round** (§9).

### 5.1 R1

- **The band:**
  - h is the largest |change| in R1's count over 30 runs with the pooled keys perturbed per layer by the bf16
    run-to-run relative RMS: the largest of the back-to-back, unhooked and post-loop repeats.
  - The fp16 measurement does not enter it.
  - We assert that the unperturbed count equals R1's.
- **The cap, H_MAX = 17 trials:** half the 35-trial gap between v2's 368 and the R0 file's 333.
  - Above it, a "reproduced" band would reach halfway to text (TF-IDF) and mean nothing.
  - h > 17 gives **UNINFORMATIVE_BAND_TOO_WIDE.**
- **The v1 check:** v1 recomputed, against its count of 470 within its own h (same perturbation, same cap).
- **The R1 verdict:**
  - In band, with v1 REPRODUCED: **REPRODUCED.**
  - In band, with v1 not reproduced: **REPRODUCED_WEIGHTS_UNVERIFIED.**
  - Out of band, with v1 REPRODUCED: **NOT_REPRODUCED_CODE_PATH.** Today's keys match v2's evening, so look at the
    port.
  - Out of band, with v1 not reproduced: **NOT_REPRODUCED_WEIGHTS_UNVERIFIABLE.**
  - Then the bounded investigation as before: the fingerprint, the dry run and port check on Starship, and one
    environment diff. No GPU, one day at most.
- **The fingerprint against May's bridge weights** is reported, never used to gate R1.
- **R2 is read and reported regardless.**

### 5.2 R2, the primary cell only: the ladder (first match wins; `r2_verdict`)

0. **VOID_NAN_FEATURES:** any NaN in the primary's features (Agni round 3, New-8).
1. **UNINFORMATIVE_PIPELINE:** the topic pipeline control is at or below 2x chance.
2. **LEAK_SUSPECTED_PERMUTATION_NULL_ABOVE_2X_CHANCE.**
3. **CONFOUND_SUSPECTED** (§5.0).
4. **FAIL_WITHIN_2X_CHANCE:** accuracy <= 2/30.
5. **NOT_ESTABLISHED_PERMUTATION:** R2's permutation p >= 0.05.
6. **FAIL_BELOW_TEXT_TFIDF:** difference <= 0 and McNemar significant. Otherwise, with difference <= 0:
   **NOT_ABOVE_TEXT_TFIDF.**
7. **EXCEEDS_TEXT_TFIDF:** difference >= 27/900, McNemar p < 0.05 with R2 ahead, and the cell interval above 0.
8. **INCONCLUSIVE:\<reasons\>:** below_3pt_floor; cell_ci_includes_0; underpowered (not significant, with the
   registered power at +3 points below 0.8).
9. **NOT_DISTINGUISHED_FROM_TEXT_TFIDF:** difference >= 3 points, not significant, registered power >= 0.8. Never
   "equal".

### 5.3 R7's consequences for main.tex, and four forbidden sentences

- **R7 SIGNAL_FOUND** (a set with p < 0.05/17 that is not CONFOUND_SUSPECTED): the general claim that spectral features
  are blind to emotion, "emotions change direction, not shape", is withdrawn for this model. That is main.tex:72-77,
  :101, :130, §301-338 (the heading and :338), :358 and :517-519, with the academic twins (about :357 and :516).
  - The per-layer table (:308) and the kitchen sink (:334) are withdrawn regardless, because their May features were
    degenerate.
  - R7 says nothing about Mistral (:404). That claim is marked unverified.
- **R7 NULL_HOLDS or NULL_HOLDS_PARTIAL:** no paper sentence is restored.
  - The paper's null was measured on prompt and generation windows (main.tex:228) whose features were degenerate, and
    R7 tests story re-encodings.
  - Only one new, scoped sentence may be added: "On story re-encodings, n-matched spectral features of the keys do not
    classify the 30 emotions above <each set's threshold>."
  - The direction-versus-shape sentences stay withdrawn, because pairing with R2 is forbidden ((a) below).
- **R7 CONFOUND_SUSPECTED or UNINFORMATIVE_\*:** untested. The claims above stay withdrawn.
- **The forbidden sentences:**
  - **(a)** R2 EXCEEDS with R7 NULL as "directional, not spectral";
  - **(b)** R1 REPRODUCED with R2 EXCEEDS as "the paper's result reproduces and beats text";
  - **(c)** the registered power with NOT_DISTINGUISHED as "a well-powered null" (Agni round 3);
  - **(d)** "text" without "(TF-IDF)".
- **Pairing:** 40.9% is v2's (dot, all positions, 5 folds, test-fold FWL); R2's number belongs to the Method's text;
  37.0% is R0 at 5 folds and 38.2% at 10.
- **Valence, R0_MISMATCH and fallbacks:** as in round 2.

### 5.4 H3's sensitivity

- A null set excludes only accuracies at or above its detection threshold. The expected value is about 0.05 to 0.06;
  the synthetic null world gives 0.044 to 0.050.
- Each threshold is the null's (1 - 0.05/17) quantile from 5,000 permutations, which is about the 15th-largest draw.
  It is reported per set.

### 5.5 Scoring the sealed predictions (`R1R2_PREDICTION.md`, written before rounds 2 and 3)

| prediction label (round 1) | R2 verdicts it covers |
|---|---|
| EXCEEDS_TEXT (robust to the cell bootstrap) | EXCEEDS_TEXT_TFIDF |
| INCONCLUSIVE_UNDERPOWERED | INCONCLUSIVE:\*, NOT_DISTINGUISHED_FROM_TEXT_TFIDF |
| FAIL_BELOW_TEXT | FAIL_BELOW_TEXT_TFIDF, NOT_ABOVE_TEXT_TFIDF |
| FAIL_WITHIN_2X_CHANCE | FAIL_WITHIN_2X_CHANCE |
| (no prediction) | VOID_NAN_FEATURES, UNINFORMATIVE_PIPELINE, LEAK_SUSPECTED, CONFOUND_SUSPECTED, NOT_ESTABLISHED_PERMUTATION: a miss on every prediction |

- R1's "reproduced within the measured band" covers REPRODUCED and REPRODUCED_WEIGHTS_UNVERIFIED.
- R7's NULL_HOLDS covers NULL_HOLDS and NULL_HOLDS_PARTIAL.

### 5.6 Pre-written outcome sentences (Agni round 3; the analysis emits each when its condition holds)

1. **A raw-residual baseline's accuracy is >= R2's:** "A raw-residual LR baseline ({src}) reaches {acc} >= R2's {r2}:
   the probe's information is in the residual stream without W_K or the cache, and the paper may not say that W_K or
   the KV cache adds to it."
2. **The projected key beats the cache key by >= 27 trials with McNemar p < 0.05** (both cells from `res_in_ln`, at the
   primary settings): "The pre-RoPE key projection (W_K x) beats the cache keys by {gain} trials (McNemar p = {p}):
   the information is in the projection, not the KV cache's geometry, and any claim is stated about W_K x."
3. **R1 NOT_REPRODUCED\* and R2 EXCEEDS_TEXT_TFIDF:** "The paper's 40.9% is not reproduced on current weights ({r1v});
   separately, the probe as the Method describes it beats text (TF-IDF) by {d} points. They are different pipelines:
   the paper's number is replaced by R2's, reported with R2's pipeline, and 40.9% is reported as unreproduced."
4. **R0_MISMATCH:** "R0 was not reproduced on this environment, so S1, S2 and S4 (which compare against the R0 file)
   are reported as 'comparator not reproduced' and are not cited until the cause is found; the primary, against R0
   recomputed at 10 folds here, is unaffected."
5. **R0' >= R2:** "A tuned text (TF-IDF) baseline reaches {acc} >= R2's {r2}: the comparison with the untuned R0 may not
   be cited as the cache beating text."

---

## 6. Power analysis

Exact McNemar, two-sided alpha 0.05, rejecting only with R2 ahead, conditional on R0's actual 10-fold correctness
vector (R0 accuracy 0.382). B = 20,000.

| ψ | power at +3 | power at +5 | gap at 80% power |
|---|---|---|---|
| **0.479 at +3 / 0.484 at +5: independence (registered, a priori)** | **0.16** | **0.58** | **6.0 points** |
| 0.40 | 0.22 | 0.68 | 5.75 points |
| 0.30 | 0.33 | 0.81 | 5.0 points |
| 0.20 | 0.49 | 0.94 | 4.25 points |
| 0.083: R0 5 vs 10 folds (sensitivity; text vs text) | 0.87 | 1.00 | 3.0 points |

- **This run can confirm a gain of about 6 points or more, and no smaller.** A true 3-4 point gain will most likely
  read INCONCLUSIVE.
- **The test is conservative:** its size is 0.002 to 0.015.

### 6.1 What follows an INCONCLUSIVE R2 (Lyra, 2026-10-19, at sign-off, before any R1 or R2 data)

This run's n is set by the trials May left, not by the effect that matters, so the likeliest result for a true gain
of 3 to 4 points is INCONCLUSIVE. Thomas's direction for this correction is to run what it takes to see the picture
accurately. Fixed now:
- **If R2 reads INCONCLUSIVE, for any reason:**
  - the paper's claim that the cache holds the user's state beyond the words stays withdrawn, as the correction
    already says;
  - it is not restored, and it is not called refuted.
- **Then R2b runs:**
  - fresh stories, by the same generation protocol;
  - **new topics, so new cells,** rather than more stories per existing prompt. The cell is the resampling unit, so
    siblings would add trials without adding independent cells;
  - none of the new stories has been seen by any R0 or R2 fit;
  - the same test, the same 3-point floor and the same R2 primary cell;
  - n is chosen for 80% power at +3 points under the discordance between R0 and R2 primary *observed in this run*.
    Sizing a new sample from a pilot's discordance is ordinary, because the new sample is what does the testing.
- **R2b gets its own pre-registration before its data.** If the n it needs is more than Starship can encode in a
  week, that pre-registration says what n we can run and what power it buys, before anything runs.
- **If R2 reads EXCEEDS_TEXT_TFIDF,** R2b is still worth running as a replication on new cells. It is not required for
  the correction.

---

## 7. Threats to validity

- **Text confound.** Words alone reach 37.0% or 38.2%. The cache must beat text (TF-IDF) by 3 points.
- **Topic grouping.** 10 topics; 10 folds leave one out. Fold equality is only a self-check; R0_REPRODUCED is the evidence.
- **Length confound.** Covariate-only and -fitted classifiers gate R2; per-set gates protect R7.
- **Multiple specifications.** 347 cells; only the primary decides; fixed before any encode.
- **Primary choices.** The layers (May's), the pool and the per-layer cosine are choices, with reasons in §3.4.
- **Space mismatch.** d_e lies before k_norm and RoPE; the keys lie after. The proj cells and §5.6.2 cover it.
- **Weights.** May's weights are unfetchable. The v1 artifact check tests them; the bridge fingerprint is a different cache.
- **Numerics.** v2 ran bf16. The band uses bf16 repeats only, and is capped at 17 trials.
- **Dependence.** 3 stories per prompt. The cell is the unit for the bootstrap, the permutation and the sign flip.
- **An untuned comparator.** R0 is one TF-IDF configuration; R0' is tuned (§5.6.5).

---

## 8. What R1, R2 and R7 cannot establish

- **Not user state:** the inputs are third-person stories (that is R6).
- **Not model state over text:** that is R5.
- **Not mechanism:** W_K's route is bounded by the raw-residual baselines.
- **Not equivalence:** every null is bounded by a stated threshold.

---

## 9. Mandatory gates

- **Lyra, 2026-10-19:** round 3 is the last Agni round. CONDITIONAL is approval once its conditions are met. Lyra checks each condition against the review (notes §10), then the encode runs. No round 4.
- **CC as second reader** of the design and of all three rounds.
- **The dry run passes** with 0 failures, locally and in Starship's observer-env.
- **The sealed prediction** is hashed and content-checked by both scripts.
- **The encode stops** on a failed self-check, determinism above 2^-8, a None commit, or constants different from §12.
- **R0_REPRODUCED,** or the mismatch is reported first. Thomas sees the results before any public text changes.

---

## 10. Gate sequence

1. Agni round 1 (2026-10-18), round 2 (2026-10-18/19) and round 3 (2026-10-19): CONDITIONAL on both scripts each
   time.
2. The round-3 conditions are implemented (notes §10). Lyra checks each off, then seals this file. The predictions are
   already sealed.
3. CC reads it.
4. Dry run and `--tokenize-only` on Starship.
5. The encode, about 25-30 minutes, including the repeats, witnesses and fp16 passes.
6. The analysis on CPU (about 45-60 minutes on 16 cores, now with 5,000 R7 permutations, R0' and the v1 recompute),
   then the R1 rule, then the R2 ladder.

---

## 11. Provenance of the numbers in this document

- v2's 368/900: `wk_backtrack_v2_results.json` (sha256 `b09f980e…`).
- v1's 470/900: `wk_backtrack_results.json` (sha256 `42a723eb…`).
- May's `emotion_vectors.json`: sha256 `3452df23…`.
- All three are on Starship, unchanged since 2026-05-21/22, and were read on 2026-10-19.
- The cache paths: v2:34, v1:33, bridge:1371-1377. The default cache's blobs are all from Aug 29.
- R0 at 5 and 10 folds and the disagreement rates: R0's code on Starship's CPU, 2026-10-18.
- The tokenizer facts: Starship, 2026-10-18.
- Power: `mcnemar_power` on R0's 10-fold vector.

---

## 12. Registered constants

The scripts assert these against this block; `prereg_um_v2_to_json.py` copies it into the JSON. Changing a value here
means changing the code, and the reverse; either one alone stops both scripts. `prediction_sha256` is the hash of
`R1R2_PREDICTION.md` as Lyra sealed it on 2026-10-18, typed here once (added at sign-off, 2026-10-19). Both scripts
refuse a prediction file with any other hash, and the generator refuses to write if the file no longer matches it.

```json
{
 "FLOOR_PTS": 3.0, "TIE_MAX": 0.5, "VARKEPT_MIN": 0.1, "ALPHA": 0.05, "POWER_MIN": 0.8, "LEN_MARGIN": 0.05,
 "H_MAX": 17, "DET_CEIL": 0.00390625, "WITNESS_NEG_MIN": 0.25,
 "PRIMARY": {"layers": "v2_L3-L15", "pool": "mean_skip50", "sim": "cos_perlayer", "folds": 10, "src": "res_out",
             "target": "cache", "fwl": "pooled"},
 "R1_LAYERS": [3, 7, 11, 15], "PAPER_K": 368, "V1_K": 470,
 "SPECTRAL_NAMES": ["frobenius", "eff_rank", "entropy", "top_sv_ratio"],
 "N_PERM": 1000, "N_PERM_R7": 5000, "N_BOOT": 10000, "POWER_PSI": "independence", "POWER_GATE_PTS": 3,
 "ENCODE_SKIP": 50, "ENCODE_NMATCH": 100,
 "FULL_ATTENTION_LAYERS": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63],
 "v2_results_sha256": "b09f980e32288c6b127fe99e2661cce4bc3a5acd198befb6ee2676561c680588",
 "v1_results_sha256": "42a723eb82240f0825f9d037171919121e5746aa42d119549190736761f045a7",
 "v1_vectors_sha256": "3452df23a47725e113f6ba351f562315e0264673d9aa93e2a1f9ab32ad7c9008",
 "prediction_sha256": "89a578e17702dc4e41d6e6f6123560364b614f36686178dc063e9a76745b4b45"
}
```

---

## 13. Decision rules in brief (sent to Agni as `falsification_criteria`)

- R1: |k-368| <= h, h = max count change over 30 bf16 run-to-run perturbations; h > 17 gives UNINFORMATIVE_BAND_TOO_WIDE.
- v1 check: v1 recomputed from this encode within its own h of 470. It decides REPRODUCED vs _WEIGHTS_UNVERIFIED/_UNVERIFIABLE vs NOT_REPRODUCED_CODE_PATH.
- Encode void: any repeat of trial 0 with per-layer relative RMS > 2^-8; or a None commit; a placeholder prediction; constants not equal to §12.
- R2 ladder: VOID_NAN > UNINFORMATIVE_PIPELINE (topic control <= 0.2) > LEAK (perm null mean > 2/30) > CONFOUND_SUSPECTED > FAIL <= 2/30 > NOT_ESTABLISHED_PERMUTATION (p >= 0.05) > FAIL_BELOW / NOT_ABOVE (diff <= 0) > EXCEEDS > INCONCLUSIVE > NOT_DISTINGUISHED.
- EXCEEDS_TEXT_TFIDF: diff >= 27/900 AND McNemar p < 0.05 (R2 ahead) AND the 300-cell bootstrap 95% CI of the accuracy difference > 0.
- INCONCLUSIVE: 0 < diff < 27; or significant with the CI including 0; or not significant with power at +3 points (ψ = independence from R0's accuracy) < 0.8.
- CONFOUND_SUSPECTED (R2): covariate-only or -fitted accuracy >= R0's, tie fraction > 0.5, or variance kept < 0.1.
- The permutation statistic is R2's own accuracy under cell-within-topic label shuffles, pipeline refit. It tests chance and leakage, not R2 > R0.
- H3: per set (17 fixed), SIGNAL iff p < 0.05/17 (5,000 permutations) and not confounded. Confounded: tie > 0.5, variance kept < 0.1, or length alone >= the set's detection threshold. NULL counts only with topic control > 0.2.
- R7 SIGNAL withdraws main.tex's "direction, not shape" sentences; R7 NULL restores nothing and adds one scoped sentence.
- Witness: May's tokens outside the fp16 nucleus <= dtype flips. A shuffled-story control must have >= 25% outside, or the witness is uninformative.
