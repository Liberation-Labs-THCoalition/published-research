# R1/R2 design notes: what was matched, what was guessed, what it costs

**Written:** 2026-10-18 for Lyra, by a Claude Code agent. These are the working notes behind `PREREG_um_v2_R1R2.md`. No
GPU was used and no model was loaded. On Starship I read files and ran only the tokenizer, on CPU.

**Files in this folder:**
- `um_v2_encode.py` (A): the encode, run on Starship.
- `um_v2_analysis.py` (B): the analysis, CPU only.
- `um_v2_dryrun.py`: the CPU dry run.
- `PREREG_um_v2_R1R2.md`: the pre-registration's single source.
- `prereg_um_v2_to_json.py`: generates the JSON from the markdown.
- `prereg_um_v2_R1R2.json` (C): the pre-registration in the house format.
- This file (D).

## 1. Four things found while building, which change the picture

1. **The paper names no layer for the probe.**
   - Neither the Method (main.tex:193-207) nor the Results (:266-287) gives one.
   - The run plan's "the paper's stated layer" does not exist.
   - The 40.9% came from v2's concatenated L3/L7/L11/L15 (v2:38, :142-152).
   - So I fixed the primary there; see §3, guess 1.
2. **The May spectral features are degenerate.** Checked on Starship, 2026-10-18:
   - In `emotion_bridge_trials.json`, every per-layer `ranks` value is 1.0 for all 900 trials, in both phases. Each
     of `entropies` and `top_sv_ratio` likewise takes one value across all 900. That is the `except` branch of
     `extract_all_features` (bridge:349-358), so the SVD failed on every call and only the Frobenius norms vary.
   - `mp_features_per_trial.json` has **0 non-null entries out of 14,400**.
   - `mp_probe_verified_results.json` reports accuracy exactly 0.0333 with std 0.0 at every layer it lists, which
     is what a constant predictor scores.
   - **So "the spectral null stands" (run plan R7, correction plan "What stands") is not supported by these files.**
   - What I did not do: trace which file each of the paper's spectral tables reads. `user_model_kitchen_sink.py`
     reads `emotion_bridge_trials.json` (its line 173), so its raw-SVD features are the norms plus two constants.
   - R7 in `um_v2_analysis.py` is therefore a first test of the null. It refuses to call a constant feature a null
     (UNINFORMATIVE).
3. **The May weights may not be retrievable.**
   - v2's `HF_CACHE` (`/Users/margaret/models/hf_cache`) no longer exists.
   - Starship's copy of the model (snapshot `ad356102`) was downloaded on 2026-08-29.
   - The HF repo has one commit, "Super-squash repository history", dated 2026-07-07 (HF API, 2026-10-18).
   - So no one can say from the hub whether the weights changed after May. If R1 fails, the weights are the first
     suspect.
   - The encode records a fingerprint: the per-layer key norms of 3 May generation prompts, recomputed and compared.
4. **The pre-registration tooling can drop text.**
   - The 18h JSON generator keeps only the first line of each bullet, so `prereg_18h.json`'s threats are cut
     mid-sentence.
   - Agni truncates hypotheses at 2,500 chars and threats at 2,000.
   - My generator joins continuation lines and refuses to write if either field would be cut. After round 1 they
     are 1,354 and 1,092 chars.

Smaller ones:
- The `po-wt-prov` worktree the brief pointed to was removed during this session.
  - The port check uses `~/Desktop/Project-Oracle/experiments/wk_backtrack_probe_v2.py`.
  - Its sha256 equals PROVENANCE's `8a41036d…`, and so does Starship's copy in `~/user_model_rerun/code/`.
- transformers 5.14.1 warns that this tokenizer has "an incorrect regex pattern … set fix_mistral_regex=True".
  - I did not set it, because v2 didn't.
  - On all 900 stories, tokenization round-trips (0 failures).
  - The re-encoded length is n_generated + 5 for 718 of 900 trials, and + 4 or + 6 for the rest.

## 2. What is matched to v2, and how I know

"v2" is `wk_backtrack_probe_v2.py`, sha `8a41036d…`. "Bridge" is the published `emotion_geometry_bridge.py`. The May
rerun's copy differs from it only in a contact line and two dtype casts (diffed 2026-10-18).

| what | v2 | mine | how checked |
|---|---|---|---|
| model, bf16, `device_map="mps"`, `trust_remote_code` | :33, :105-109 | encode `main` (:393-) | read; plus `missing_keys` must be empty (new) |
| lone user message, `add_generation_prompt=False`, `enable_thinking=False`, TypeError fallback | :127-132 | encode `render` (:81) | Starship tokenizer: identical render with and without the flag, 900/900 |
| `tokenizer(chat, return_tensors="pt")`, no truncation | :133 | encode `run_encode` | lengths 109-205 against a context of 262,144; `--tokenize-only` passes on Starship |
| one forward pass, `use_cache=True`, `cache.layers[l].keys` | :136-146 | `get_kv` (:183) | transformers 5.14.1: `DynamicCache(config=…)`, `DynamicLayer.keys` |
| keys mean over positions, flattened head-major; last token | :148, :150 | `collect` (:230) | dry run: the pools match the saved full keys |
| L3/L7/L11/L15 concatenated, 4,096 dims | :38, :142-152 | analysis `main`, `acts` | v2's `probe.log` says "Key activation dim: 4096" |
| FWL token count = encoded length | :134, :178 | `n_tokens` | read |
| `fwl_residualize`, `compute_emotion_directions`, `project_trials` | :48-89 | analysis :53-89 | **AST-identical** (dry run; a 1-character mutation makes it fail) |
| 30-way loop: GroupKFold(5) by topic, FWL on train and on test separately, nan_to_num, scaler, LR(C=0.1, 2000) | :213-250 | `r1` (:116) | port read line by line |
| valence on the hard-coded sets, its own GroupKFold, directions from the training subset | :40-45, :188-196, :252-286 | `r1`, `valence_subset` | R0's per-trial valence indices and labels must match (asserted) |
| CCS, no scaler | :288-317 | `r1` | read |
| length baseline, LR(max_iter=1000) | :319-332 | `r1` | read |
| seed 42 | :94 | both | nothing is stochastic in either the probe or LR(lbfgs) |

**Folds.**
- GroupKFold has no randomness.
- With 10 equal topic groups, 5 folds pair the topics {4,9}, {3,8}, {2,7}, {1,6}, {0,5}, and 10 folds leave one
  topic out (sklearn 1.7.1 locally).
- The result JSON records the topic-to-fold map.
- The real check runs at analysis time: R0's code at 5 folds must reproduce `R0_results.json` trial for trial,
  under Starship's sklearn 1.9.0.

**Where I did not match v2:** v2 zero-fills a layer whose keys are missing (v2:153-160). I raise instead.
- v2's 40.9% shows that not all layers were zero-filled. Its log cannot show that none were.

## 3. Every place I had to guess

1. **The primary layer set: v2's L3/L7/L11/L15.**
   - The paper states none (§1.1).
   - The alternatives are all 16 layers (the run plan's wording) or L35 (the bridge's peak, which is a result on
     the same data).
   - **Lyra decides before sealing.** To change it, edit `PRIMARY` in `um_v2_analysis.py` and the markdown's H2
     and §3.4, then rerun `prereg_um_v2_to_json.py`.
2. **The residual: decoder-layer output (`res_out`), not the post-input-layernorm state.**
   - The brief guessed v1 used the post-input-layernorm state. It didn't: v1 used the bridge's
     `emotion_vectors.json` (v1:66-69, :96). The bridge hooked each decoder layer's output and averaged from
     position 50 (bridge:467-480), stored per layer, and v1 projected it with W_K at that same layer.
   - That tensor is not what W_K at layer l reads. It is what feeds layer l + 1.
   - The primary keeps the paper's residual. `res_in_ln` (exactly k_proj's input, self-checked) and `res_pre` are
     both saved and tested in the mechanism cells.
3. **"Skip the first 50 tokens"** counts from the first rendered token, as the bridge's `hidden[:, 50:]` did. That
   includes 3 template tokens, so 47 story tokens are skipped.
4. **The FWL confound.** The paper says "output token count". I use the encoded length, which correlates with it
   at r = 0.9999 (offset 4 to 6, measured).
   - The reason: R5 and R6 inputs are not generated, so a probe that needs n_generated would not transfer.
5. **Cosine over concatenated layers.** *Superseded in round 1 (analysis N1, §8).* The primary is now the mean of
   per-layer cosines. The single cosine over the raw concatenation is an exploratory cell.
6. **d_e is unit-normalised.** That changes nothing for cosine. In the "dot" variant it reproduces v2's dot with a
   unit direction.
7. **R2 valence:** directions from the valence subset's training fold (10 emotions, so 10 features), as v2 did
   (v2:262). The paper says only "the same features".
8. **Classifier and scaler as v2.** The paper says only "multinomial logistic regression".
9. **The grand mean is over training trials** (v2's choice). With balanced classes it equals the mean of the class
   means.
10. **The self-check tolerances (1e-2 linear, 2e-2 norm) are reasoned, not measured**: bf16 has an 8-bit mantissa,
    and a wrong tensor would be off by O(1).
    - If one fires spuriously, the error prints the values. Loosen it only with the measured value written next
      to it.
11. **The fingerprint.** *Superseded in round 1 (encode N2, §8).* There is now an fp16 pass after the loop, and the
    band is calibrated on the measured dtype effect at fixed weights.
12. **The R7 test.** *Superseded in round 1 (analysis N2, §8).* It is now a within-topic permutation per layer set,
    with a positive control, on n-matched features.

## 4. Deviations from the brief, and why

- **Pooled arrays are float32; only the per-position K/V are float16.**
  - Residual class means are differences of 27-trial means with O(10-100) components, and fp16 carries about 3
    significant digits.
  - The pooled arrays cost about 3 GB in total; the per-position K/V are the bulk.
- **Three residual kinds, pooled values and per-position K/V are saved, not just keys and one residual** (SOP §5.1).
- **R2's comparator is R0 re-run at 10 folds, not the 5-fold R0 file,** so the primary test doesn't also compare
  training-set sizes. The 5-fold comparison is reported as S4.
- **Robustness and secondary tests were added:**
  - cluster bootstraps by cell and by topic;
  - a Holm-corrected secondary family;
  - a within-topic permutation test.
- **R7 is implemented,** in about 20 lines. Its meaning changed (§1.2).

## 5. Starship cost, estimated from measurements

- **Memory.**
  - The checkpoint is 55.56 GB, including a vision tower (333 tensors) and an MTP head (15) that `ForCausalLM` skips.
  - So the language model is about 54 GB resident in bf16. v2 loaded 851 tensors (`probe.log`).
  - Activations at 205 tokens are well under 1 GB; logits are about 0.1-0.2 GB.
  - Peak is about 56 GB of 256. 97% of memory was free when I checked.
- **Time.**
  - v2's own run on Starship went from 23:13:19 (log start) to 23:29 (results written) for model load, 900 forward
    passes and its analysis. That is at most 1.0 s per pass at a median of 176 tokens.
  - The work this script adds per trial is about 0.1-0.3 s: CPU copies, 16 SVDs of 704×256, a ~12 MB write and its
    sha256.
  - **Estimate: 18-25 minutes,** plus about 1-3 minutes to load and 3 fingerprint passes.
  - The run plan's "about a day" is ample for R1 and R2.
- **Since round 1:** 6 more forward passes (trial 0 twice, 3 fingerprint prompts in fp16, trial 0 in fp16) and one
  in-place cast to fp16 after the loop.
- **Disk.**
  - Per-position K/V: 158,439 tokens × 64 KiB = 10.4 GB.
  - Pooled arrays, now 4 pools: 4.0 GB. W_K: 0.34 GB.
  - **Total about 14.8 GB.** 1.0 TiB is free; the encode stops if free space is under twice its own estimate.
- **Analysis (CPU).**
  - The round-1 dry run took about 60 s for the null dataset (347 cells, 20 R2 permutations, 17 × 400 R7
    permutations) at synthetic dimensions.
  - On the real data, these dominate:
    - loading and projecting the residuals;
    - 1,000 R2 permutations;
    - 17 × 1,000 R7 permutations;
    - 30 band replicates.
  - **Estimate: 10-20 minutes on 16 of Starship's 28 cores** (or MTH).

## 6. Commands, after the gates (round 1: SUPERSEDED by §10.8, which has the arguments rounds 2 and 3 added)

```bash
# Starship
~/observer-env/bin/python um_v2_dryrun.py --work /tmp/um_v2_dry --v2 ~/user_model_rerun/code/wk_backtrack_probe_v2.py
~/observer-env/bin/python um_v2_encode.py --trials ~/user_model_rerun/results/emotion_geometry_bridge/emotion_bridge_trials.json \
    --out ~/user_model_v2/encode > ~/user_model_v2/encode.log 2>&1
~/observer-env/bin/python um_v2_analysis.py --enc ~/user_model_v2/encode \
    --trials ~/user_model_rerun/results/emotion_geometry_bridge/emotion_bridge_trials.json \
    --r0 R0_results.json --out ~/user_model_v2/um_v2_results.json --jobs 16
```

**Power numbers** (prereg §6) come from exact binomial sums at n = 900, two-sided alpha 0.05, rejecting only in the
R2 direction. The script, as run on 2026-10-18:

```python
import numpy as np
from scipy.stats import binom, binomtest
n = 900
bmin = {D: next((b for b in range(D // 2 + 1, D + 1) if binomtest(b, D, 0.5).pvalue < 0.05), None) for D in range(n + 1)}
def power(psi, delta):            # psi = P(exactly one classifier right); delta = acc_R2 - acc_R0
    pD, q = binom.pmf(np.arange(n + 1), n, psi), (psi + delta) / (2 * psi)
    return sum(pD[D] * binom.sf(bmin[D] - 1, D, q) for D in range(n + 1) if pD[D] > 1e-12 and bmin[D])
# psi 0.20: power(0.039) = 0.72, 80% at 0.043 | 0.30: 0.55, 0.053 | 0.40: 0.44, 0.060 | 0.48: 0.38, 0.066
# size, power(psi, 0) = 0.021-0.022
```

## 7. What should change in the run plan

1. **R2's primary needs a fixed layer set.** The paper has no "stated layer". I propose v2's set; Lyra decides.
2. **R7, and the correction plan's "What stands":**
   - the spectral null rests on degenerate files (§1.2), so "stands" should become "untested";
   - the paper's spectral-null section needs the same trace;
   - R7 should run on this encode now, not wait for R3.
3. **Say which R0 "below R0" means.** I fold-matched it: R2 at 10 folds against R0 at 10 folds.
4. **R1's stop rule cannot always tell code from weights,** because the repo was squashed (§1.3).
   - Add: if R1 fails and the fingerprint ratios are off, the failure is about the model, not v2's code.
5. **Power.**
   - A 3.9-point gap has power 0.38-0.72 at n = 900.
   - So R1-versus-R0 may well end "not distinguished", which is not "equal".
   - R3's 900 unique stories do not raise n. If the margin matters, R3 needs more trials per cell.
6. **The run plan says the probe scripts are untracked in `Project-Oracle/experiments/`.** They are now committed
   (PROVENANCE).
   - But the clone at `Research/Project-Oracle` has not fetched that commit, and the `po-wt-prov` worktree is gone.
   - Point the plan at the commit hash.

## 8. Changes for Agni round 1

Agni reviewed both scripts with this prereg on 2026-10-18 (`claude-fable-5-1`, agni 2.1-claude+stream). Both came back
CONDITIONAL. The full reviews are on MTH, in `~/agni_runs/um_v2_review/{encode,analysis}/um_v2_R1R2_agni_review.json`.
Each item below gives the review's label, what changed and where. "Declined" means I did not do what was asked, and
says why.

### 8.1 Encode review (`um_v2_encode.py`)

- **N1, "SKIP=50 means something different here" (MAJOR).** Done, except for its premise.
  - Every shard now records the story's token span, from the tokenizer's offset mapping. Measured on all 900 trials:
    3 preamble tokens, a story of 104-200 tokens, then 2 trailing tokens.
  - A `story` pool (the story tokens only) is added.
  - R2's FWL covariates now include 1/n_pooled.
  - The window decision is written before any data, in prereg §3.4.
  - **The decision: the primary stays at positions >= 50.**
    - The Method describes a standalone re-encode with a 50-token skip (main.tex:250; also :200 and :273).
    - The bridge applied `hidden[:, 50:]` to exactly this input: its re-encode of a lone user message
      (bridge:476-477, :671-701). The residuals from that pass replaced the generation pass's (bridge:1550-1551).
    - So "skip 50" never meant "skip the preamble", and story-only is exploratory.
  - **Declined: the premise.** Agni says the May cache held the system prompt. That is true of the generation pass,
    not of the re-encode that produced the paper's directions.
- **N2, the fingerprint (MAJOR).** Done.
  - The picks have distinct (emotion, topic): trials 0, 33 and 66. They rebuild May's token counts of 98, 94 and 95
    (checked on Starship, 2026-10-18).
  - `fingerprint()` raises if it builds no rows.
  - A row whose token count differs from May's is void.
  - The formula is written down: ||K_l||_F over the whole prompt, as bridge:316-347.
  - **The band is calibrated on this model, not assumed.** An fp16 pass runs after the loop (May's dtype; the cast
    comes after the loop so it cannot touch the encode). IN_BAND means fp16-now / fp16-May lies within the bf16 /
    fp16 spread measured at fixed weights, with a floor of 1e-3. Logic in `fingerprint_verdict`.
  - The branches are in prereg §5.1.
- **N3, duplicates and N_eff (MAJOR).** Done.
  - `n_distinct_story_sha256` is recorded and asserted equal to n. The coordinator found max copies = 1; the assert
    makes that a stop rather than a note.
  - Per-prompt token Jaccard is recorded. Measured on Starship with the revised encode's `--tokenize-only`: 900
    distinct, 300 prompts, mean Jaccard 0.178 and max 0.578 across the 900 within-prompt pairs.
  - The N_eff branch is in prereg §5.3: the 300-cell bootstrap is decision-bearing for EXCEEDS_TEXT.
- **N4, power (MAJOR).** Done.
  - `mcnemar_power` runs the exact McNemar, conditional on R0's actual 10-fold vector.
  - The pre-data table (prereg §6) uses R0's real vector, computed on Starship's CPU on 2026-10-18.
  - p >= 0.05 now reads as INCONCLUSIVE_UNDERPOWERED or NOT_DISTINGUISHED, never as "falsified" or "equal".
- **N5, stale shards on resume (MAJOR).** Done.
  - Every shard's meta carries the model commit, dtype, device, script sha256 and chat-template sha256.
  - `shard_ok` requires all of them, plus the chat sha256 of the current render.
  - Dry run: changing the script sha makes all 12 shards be redone.
- **N6, "qualify but never rescue" undefined (MAJOR).** Done.
  - A significant McNemar whose cell interval includes 0 (or whose permutation p >= 0.05) is now
    NOT_ESTABLISHED_SIGNIFICANT_UNDER_INDEPENDENCE_ONLY: no claim and no title (prereg §5.2, rule 6).
- **N7, layer-index ambiguity (MINOR).** Done.
  - `layers` (the model layer ids) is in every shard and in wk.npz.
  - The analysis asserts them and selects by model-layer id (`LIX`).
  - The prereg now says "model layers 3, 7, 11, 15".
- **N8, the ±3 band not derived (MINOR).** Done.
  - Trial 0 is encoded twice before the loop, recording max |Δ| and the per-layer relative RMS of every pooled array.
  - The band h is the largest change in R1's correct count over 30 runs with the pooled keys perturbed at the
    larger of two measured sizes: run to run, and fp16 against bf16. R1 is reproduced iff |k - 368| <= h.
  - The run-to-run difference alone could be zero on a deterministic backend. That would demand bit-exact
    reproduction across a software change since May, so the measured dtype effect stands in for "a different
    numeric path".
- **N9, the `last` pool is mostly a template token (MINOR).** Done.
  - Per pool, the analysis reports the accuracy of a classifier on the length covariates alone.
  - The specification-curve summary counts the last-pool cells that beat it.
  - Prereg §4.3: a last-pool cell is read only against that number.
- **N10, no end-to-end positive control (MINOR).** Done.
  - The primary pipeline is run on topic labels, grouped by emotion.
  - At or below 2x chance, the verdict is UNINFORMATIVE_PIPELINE.
- **N11, logits discarded (MINOR).** Done: `logits_to_keep=1`, and the last position's top 20 logits are saved.
- **Checklist items.**
  - **CONTROL_WITHOUT_BRANCH:** all branches are now written (prereg §5).
  - **RANK_BOUNDED_ACROSS_UNMATCHED_N:** `spectral_n100` (the first 100 story tokens; the shortest story has 104) is
    R7's feature set. The all-positions spectral features are exploratory.
  - **NULL_FROM_UNVERIFIED_SEARCH:** an empty fingerprint now raises (dry run).
  - **SILENT_N_CHANGE:** `complete` is now computed against `n_in_file`, so a `--limit` run is never complete (dry run).
  - **UNFALSIFIABLE:** the H1-fails branch is in prereg §5.1.
  - **PREREG_NOT_LOCATABLE:** H3 is added for R7, and gate 4 names the prediction file `R1R2_PREDICTION.md`. Lyra
    writes it; it does not exist yet.
  - **DIRECTIONAL_WITHOUT_THRESHOLD:** the valence probability is defined (prereg §4.4).
    - **Declined: an effect-size floor.** It is not on the must-change list, and the two reviews disagree about
      whether a 1-point win could pass; that depends on ψ. The MDE is stated instead.
  - **JOINTLY_FALSE_INDEPENDENTLY_TRUE:** a commitment to pair each number with its own pipeline (prereg §5.3).
  - **Must-change 8,** which the encode review set for the analysis: reinjection diagnostics, n-matched spectral
    features and the valence definition. All done in the analysis (§8.2).

### 8.2 Analysis review (`um_v2_analysis.py`)

- **N1, CRITICAL: the primary cosine was dominated by the deepest layer.** Done.
  - The primary is now `cos_perlayer`: each layer's block of d_e and of the key is scaled to unit length, and the
    per-layer cosines are averaged.
  - The coordinator offered two forms (average the per-layer cosines, or concatenate unit blocks). They are
    algebraically the same statistic, and the dry run checks the identity to 1e-12. So there is one primary.
  - What stays exploratory:
    - the old statistic, `cos_concat` (one cosine over the raw concatenation);
    - Agni's per-layer cosines as 120 separate features, `cos_layerfeats`. This is not the primary because the
      Method specifies a 30-dimensional feature vector (main.tex:201).
  - The dot variant is now per-layer too (`dot_perlayer`).
  - The planted check (NORM10) is in §8.4.
- **N2, R7 can switch itself off (MAJOR).** Done.
  - Constant columns are dropped per set; a set is UNINFORMATIVE only if every column is constant.
  - Each set gets its own verdict.
  - FWL diagnostics: tie fraction, covariate R², variance kept, residual Spearman.
  - A within-topic permutation null per set, 1,000 permutations, with Bonferroni over the informative sets. A
    permutation count too small to reach the corrected alpha is refused before any work (SOP §4.11).
  - A positive control: topic, StratifiedKFold(10), not grouped. NULL_HOLDS now requires it to pass; otherwise the
    verdict is UNINFORMATIVE_INSTRUMENT.
  - The binomial test and its excusing note are removed.
  - The features are n-matched.
- **N3, power (MAJOR).** Done (encode N4).
  - The governing ψ is R1 against the R0 file, as Agni suggested.
  - The R0 5-versus-10-fold disagreement (the coordinator's stand-in) and independent errors are reported beside it.
  - The MDE for the observed number of discordants is reported.
  - INCONCLUSIVE_UNDERPOWERED applies when the power at +5 points is below 0.8.
- **N4, the R1-fails branch (MAJOR).** Done.
  - `R2_interpretable` is removed. R2 is always read and reported.
  - The bounded investigation is named: the fingerprint verdict, the dry run and port check on Starship, and one
    environment diff. No further GPU run, and at most one working day.
  - After that, the 40.9% is reported as "not reproduced on current weights" (prereg §5.1).
- **N5, four reported-only outcomes (MAJOR).** Done (prereg §5.2 and §5.3):
  - a permutation null above 2x chance gives LEAK_SUSPECTED;
  - a permutation p >= 0.05 means not established;
  - R7 SIGNAL_FOUND withdraws the paper's spectral null and its direction-versus-shape reading;
  - a length baseline within 0.05 of a probe's valence AUROC makes that result EXPLAINED_BY_LENGTH;
  - EXCEEDS_TEXT that isn't robust to the cell bootstrap counts as not established, as Agni and the coordinator
    both asked.
- **N6, missing asserts (MINOR).** Done.
  - Asserted: n == 900; 30 emotions; 3 trials in every cell; the R0 file's length; the manifest's and every shard's
    layer ids; array shapes; equal R0 and R2 fold maps at 5 and 10 folds.
  - `shuffle=False` is pinned through `gkf()`.
  - `r0_check` is now a verdict.
  - `skip50_fallback` is a required key with a ceiling of 0.
  - `ccs_auroc` is None rather than 0.5 when empty, and `story_idx` is a required key.
- **N7, the FWL covariate didn't match the pool (MINOR).** Done.
  - The covariates are [1, log1p(n_tokens), 1/n_pooled]. n_pooled is n_tokens - 50 for the primary, n_tokens for
    `mean_all`, the story length for `story`, and nothing for `last`.
  - Residual Spearman against n_tokens and n_pooled is reported per cell.
  - The paper-literal FWL is an exploratory cell.
- **N8, R0 asymmetry (MINOR).** Stated in prereg §3.5.
- **N9, pairing numbers with pipelines (MINOR).** Committed in prereg §5.3.
- **Checklist items.**
  - LABEL_WITHOUT_ITS_STATISTIC: fixed by N1.
  - ARCHITECTURE_VS_CONFIG and IDENTITY_BY_DEFAULT: fixed by N6.
  - **OUTCOME_INFORMED_TAXONOMY** (v2's layer set was chosen on May's data): disclosed in prereg §3.4.
    **Declined to change** here: the primary layer set is Lyra's call before sealing (§3, guess 1).

### 8.3 Defects of my own, found in this round

- **Open file handles.** `shard_ok` left an `np.load` handle open, and on Windows a later `os.replace` of that shard
  failed. That is harmless on macOS, but every `.npz` is now opened with `with`, in all three scripts.
- **A power function that could return None.** `mcnemar_power` returned None when the assumed ψ couldn't occur at
  R0's accuracy (found on a fixture where R0 was weak). It now clamps ψ to the feasible range.
- **A determinism block that compared nothing.** My first draft of the trial-0-twice block used a helper class that
  would have compared against an empty store. I caught it on re-reading, before any test ran, and replaced it.

### 8.4 Dry run after round 1: 48/48 checks pass (2026-10-18, local CPU, sklearn 1.7.1)

- **Datasets:**
  - **planted:** R2 0.729 against text 0.129, EXCEEDS_TEXT; R7 SIGNAL_FOUND.
  - **null:** R2 0.037, FAIL_WITHIN_2X_CHANCE; R7 NULL_HOLDS, with the control passing; the permutation null mean
    is 0.034.
  - **weak:** R2 0.189 against 0.583, FAIL_BELOW_TEXT.
  - **matched:** R2 0.600 against 0.568, governing power at +5 points 0.635, INCONCLUSIVE_UNDERPOWERED.
  - **NORM10, the N1-critical case:** the primary per-layer cosine gets 0.252 and beats text (0.118). The old
    concatenated cosine gets 0.124 and cannot be told from text (McNemar p = 0.72).
- **Leak:** directions estimated with the test fold included score 0.444 on null data; the real code scores 0.028.
- **R7 logic:**
  - constant rank and entropy columns no longer void a set (SIGNAL still found);
  - all-constant features give UNINFORMATIVE_ALL_CONSTANT;
  - pure noise gives UNINFORMATIVE_INSTRUMENT (control at 0.123), never NULL_HOLDS;
  - topic structure without emotion gives NULL_HOLDS;
  - 50 permutations over 17 sets is refused.
- **Fingerprint logic:** IN_BAND, OUT_OF_BAND, VOID and NOT_RUN, each from constructed rows.
- **Power:** at Δ = 0, 0.0095 (below alpha/2), rising to 0.32, 0.80 and 1.00 at +3, +5 and +8 points.
  `mde_observed(300) = 0.040`.
- **Identity:** per-layer cosine equals the concatenated-unit-block cosine.
- **Encode on the fake model:**
  - the 4 pools match the full keys;
  - LN(res_pre) equals res_in_ln;
  - spectral_n100 is not degenerate;
  - determinism is bit-identical;
  - fingerprint picks are distinct, voids are flagged and the alt-dtype pass runs;
  - the provenance, layer ids and top-k logits are in every shard;
  - resume redoes exactly the deleted and torn shards, and all 12 when provenance changes;
  - a `--limit` run is never complete;
  - a duplicate story stops the tokenize pass;
  - a broken self-check raises, a missing cache layer raises, and an empty fingerprint raises;
  - `--help` works, and importing the module loads no transformers.
- **Fixture tuning:** the strengths for weak (s = 0.25), matched (s = 0.46) and NORM10 (s = 0.35) were tuned until
  each lands in its branch. The analysis was not changed to get there.

## 9. Changes for Agni round 2

Round 2 (`claude-fable-5-1`) reviewed both scripts with the round-1 prereg. Both came back CONDITIONAL. The reviews
are on MTH in `~/agni_runs/um_v2_review/{encode_r2,analysis_r2}/um_v2_R1R2_agni_review.json`. Agni's summary: "the
holes are in what the numbers will be allowed to *mean*."

Lyra's decisions of 2026-10-18 are written into prereg §5.0 as theirs, with their reasons:
- the 3-point floor;
- the inconclusive region;
- the a priori ψ as the registered power;
- CONFOUND_SUSPECTED;
- the TF-IDF wording.

This round was interrupted by a session limit and resumed on 2026-10-19. On resuming, before changing anything, I
checked that:
- every script still compiles;
- every planned code item is present (by grepping for its specific strings);
- the prereg JSON's `--check` reported exactly the three scripts as stale.

Only the `matched` fixture value, the prereg and these notes were unfinished.

### 9.1 Encode review: the 12 required changes

1. **The source of h, and the H1 branches (N1).** Done.
   - The band's perturbation per layer is the larger of:
     - the run-to-run relative RMS: the largest of the back-to-back, unhooked-vs-hooked and post-loop checks;
     - the fp16-vs-bf16 relative RMS on trial 0.
   - Both sources are named in the code and prereg §5.1, so neither can be chosen afterwards.
   - The branches:
     - REPRODUCED or NOT_REPRODUCED with an IN_BAND fingerprint;
     - `_WEIGHTS_DRIFTED` variants when it is OUT_OF_BAND;
     - UNREADABLE when it is VOID or NOT_RUN.
2. **An effect-size floor, the bootstrap's estimand, and the inconclusive region.** Done, as Lyra's decisions
   (prereg §5.0 and §5.2).
   - The floor is 3.0 points (27 of 900 trials, compared on counts, not floats).
   - The estimand is the accuracy difference, R2 minus R0 at 10 folds, resampling the 300 cells.
   - The inconclusive region: below the floor; or a significant McNemar with the cell interval including 0; or not
     significant with registered power below 0.8.
   - It is implemented as `r2_verdict`, and every branch is unit-tested (§9.4).
3. **Rules for the §4.3 diagnostics.** Done.
   - CONFOUND_SUSPECTED (ladder rule 3) fires if any of these holds:
     - the covariate-only classifier for the primary pool reaches R0;
     - the classifier on the FWL-fitted part ("-fitted") reaches R0;
     - a tie fraction exceeds 0.5;
     - variance kept falls below 0.1.
   - **Where the thresholds come from** (Lyra asked that the source be written):
     - **Variance kept, 0.1: the regression literature.** VIF > 10 (Kutner et al. 2005; O'Brien 2007), which is
       also Agni's suggestion.
     - **Tie fraction, 0.5: the mechanism, not the literature and not the null world.** A feature tied on more than
       half the trials is mostly a step function (the FWL_TIED_FEATURE_REINJECTION case: 196 of 200 tied).
     - **Why not the null world:** I looked at the dry run's null-world distribution first. Its features are
       continuous: the largest tie fraction over its 347 cells is 0.0011 (one tied pair), and the smallest variance
       kept is 0.985.
       - A ceiling taken from that world would fire on any real length association, which is exactly what FWL
         exists to remove.
       - So the null world is reported as the false-positive check instead: both thresholds fire 0 times there.
4. **The permutation unit is the cell, within topic.** Done.
   - `perm_within` now moves the 30 emotion labels over the 30 (emotion, topic) cells within each topic, and a cell's
     3 siblings move together. This applies to R2 and R7.
   - The full null vectors and the number of distinct values are saved.
5. **Fix H3's m and its layer sets; group the topic control by cell; a sensitivity bound.** Done.
   - m = 17 is fixed in advance; an all-constant set still counts.
   - The control is StratifiedGroupKFold (shuffled) grouped by cell, and it runs per set (§9.3 for why it is not
     GroupKFold).
   - Each set reports its detection threshold (the null's 1 - alpha/17 quantile). That is the accuracy its null
     excludes (prereg §5.4).
6. **A fingerprint tolerance, its insensitivity, and a direction-sensitive witness.** Done.
   - The norm tolerance is written down. The norms' insensitivity (they are set mostly by k_norm's weights) is
     stated in the code, the prereg and the result JSON.
   - The new witness teacher-forces May's own story tokens for the 3 picks. It asks whether each token lies inside
     this model's temperature-0.7, top-p-0.9 nucleus, which is how May sampled (bridge:642-650).
   - It is stronger than Agni's first-token suggestion:
     - the whole story is tested;
     - under May's weights every token must be inside;
     - the snapshot has no generation_config, and any top-k only shrinks the set.
   - The test passes if the tokens outside the fp16 nucleus number no more than those whose membership flips
     between bf16 and fp16 at fixed weights.
   - The picks also require the story to re-tokenise to exactly May's generated length. Trials 0, 33 and 66 do, at
     162, 195 and 162 tokens (checked on Starship); 718 of 900 do.
7. **Register the a priori power; relabel the R1 ψ as sensitivity.** Done.
   - The registered ψ is R0's 5-fold against 10-fold disagreement.
   - R1's ψ and independent errors are labelled as sensitivity.
   - The note "the power of the test, not of a latent cache signal" is added (Agni POWER_FROM_ORACLE).
8. **Label S1/S2's FWL.** Done, with a correction.
   - Agni read v2's FWL as fitted on all 900 trials. It is fitted on each fold's own test trials (v2:227). That is
     label-free and is not all-data.
   - The label in the prereg, the code and the result says exactly that.
9. **An unhooked trial-0 forward (L1).** Done. It runs before any hook is registered, and its keys_pool is compared
   with the hooked pass's.
10. **A post-loop bf16 trial-0 re-encode before the fp16 cast (N8).** Done. All pooled arrays are compared with
    trial 0's shard.
11. **`len(norms_may)`, `prediction_sha256`, fp16 underflow counts.** Done.
    - A row is void unless May's list has 16 entries in the same layer order (May's own `layer_indices` are checked
      when present).
    - `--prediction` is required for a run, and its sha256 goes into the manifest.
    - Per shard: counts of fp16 subnormal and to-zero values for K and V.
12. **The analysis refuses a skipped fingerprint (N4).** Done. It also refuses:
    - a skipped witness;
    - a missing determinism block;
    - a missing fp16 numeric-path check;
    - a prediction hash different from the encode's.

**Other encode-review items:**
- **S1:** the logit top-k is labelled as a determinism aid, not a margin.
- **N5 (the FWL asymmetry):** covered by CONFOUND_SUSPECTED.
- **N6 (Jaccard has no consequence):** the cell unit is now decision-bearing whatever the overlap (the floor and the
  cell interval are both in EXCEEDS), so Jaccard is descriptive. Measured: mean 0.178, max 0.578.
- **N9, layer semantics:** v2:37-38 sets `PROBE_LAYERS = FULL_ATTENTION_LAYERS[:4]` (model layers 3, 7, 11, 15), and
  v2:142-144 reads `cache.layers[li]` by model layer id. This is cited in the prereg (H1, §3.4).
  - **Declined:** "cite it in the prediction file". That file is Lyra's and sealed; I did not edit it.
- **ESTIMATOR_MISMATCH:** the observed difference is reported with its percentile interval.

### 9.2 Analysis review: the 7 required changes

1. **Cell-level permutation, and the full null vectors saved.** Done (§9.1, item 4).
2. **The CONFOUND_SUSPECTED branch.** Done (§9.1, item 3).
   - **Declined:** Agni also suggested gating on covariate R² > 0.5 and residual Spearman > 0.3.
     - Covariate R² is the same quantity as variance kept (R² = 1 - kept), so a second threshold on it would
       double-gate one measure.
     - Residual Spearman is reported per cell but not gated: there is no sourced threshold for it.
     - And adding gates beyond the four Lyra named would not be their decision.
3. **Replace the [0.0]*16 defaults with asserts.** Done.
4. **Assert stories >= 100 tokens and non-zero projected norms.** Done.
5. **A numeric sensitivity statement for H3.** Done (§9.1, item 5): per-set null p95, a detection threshold and a
   per-set positive control.
6. **Hash the prereg JSON and the prediction file.** Done. Both are in `meta.inputs_sha256`, and the prediction
   hash must equal the encode manifest's.
7. **Forbid the two conjunctions; say "TF-IDF"; add raw-residual baselines.** Done.
   - The two forbidden sentences are in prereg §5.3 and in the result's `meta.reporting_rules`.
   - Every R2 verdict name says TFIDF.
   - The raw-residual LR baselines (res_out and res_in_ln, model layers 3/7/11/15, positions >= 50) are exploratory
     cells, each with McNemar against R0 and against R2.

**Other analysis-review items:**
- **N7:** FAIL_BELOW_TEXT is split. It is FAIL_BELOW_TEXT_TFIDF only when McNemar is significant with R0 ahead;
  otherwise NOT_ABOVE_TEXT_TFIDF, which is not a refutation.
- **N8:** the R0 probability tolerance is now 1e-6, and the R0 file's sklearn version is recorded as unknown.
- **N9:** the specification curve's count is renamed `..._DESCRIPTIVE_UNCORRECTED`.
- **N10:** S4 carries a note on its training-set asymmetry.
- **N11, "likely underpowered":** answered by Lyra's decisions. Prereg §6 says plainly that a true +3 to +4 point
  gain is more likely to end INCONCLUSIVE.
- **R7:** records the number of distinct null values per set.

**Retained, and flagged for Lyra.** EXCEEDS_TEXT_TFIDF also keeps round 1's requirement of a permutation p < 0.05.
That makes four conditions where Lyra's message names three. It only makes EXCEEDS harder to reach, and Agni
credited it in round 2. If Lyra wants exactly three, remove ladder rule 5 in `r2_verdict` and prereg §5.2.

### 9.3 Defects found this round

- **The R7 topic control could never pass (found by the dry run).**
  - GroupKFold deals equal-size groups to folds in a fixed cyclic order: cell c goes to fold (299 - c) mod 10.
  - With cell ids = emotion x 10 + topic, every fold held exactly one topic. The "grouped by cell" control was
    secretly leave-one-topic-out, and scored 0.000.
  - The real run's cell ids (sorted "emotion|topic" strings) have the same structure, so R7 would have been
    UNINFORMATIVE on any data.
  - It now uses StratifiedGroupKFold(10, shuffle=True), stratified on topic. The pure-noise control now scores 0.103,
    which is chance.
- **Three of my own dry-run tests were wrong; I fixed the tests, not the analysis.**
  - A fingerprint case was built wrong: "outside in fp16 only" is a flip, not drift.
  - The +2-point ladder case packed its discordant trials into a few cells, so the cell interval also failed. It now
    uses one trial per cell, to isolate the floor.
  - The null-case R7 check read a top-level field that round 2 moved into each set's row. The analysis was right;
    the check crashed.

### 9.4 Dry run after round 2: 57/57 checks pass (2026-10-19, local CPU, sklearn 1.7.1)

- **Ladder unit tests** (constructed correctness vectors, real McNemar and bootstrap):
  - **A +2-point win with p = 0.008:** INCONCLUSIVE:below_3pt_floor.
  - **Every other branch** lands where the prereg says:
    - EXCEEDS_TEXT_TFIDF;
    - INCONCLUSIVE:cell_ci_includes_0;
    - INCONCLUSIVE:underpowered;
    - NOT_DISTINGUISHED_FROM_TEXT_TFIDF;
    - FAIL_BELOW_TEXT_TFIDF;
    - NOT_ABOVE_TEXT_TFIDF;
    - CONFOUND_SUSPECTED.
- **Datasets:**
  - **planted:** EXCEEDS_TEXT_TFIDF (0.729 against 0.129).
  - **null:** FAIL_WITHIN_2X_CHANCE (0.037). R7 NULL_HOLDS, with every set's control passing and detection thresholds
    of 0.044 to 0.050. 0 confound firings.
  - **weak:** FAIL_BELOW_TEXT_TFIDF.
  - **matched** (s = 0.45; probed at 0.579 against 0.568): INCONCLUSIVE:below_3pt_floor.
  - **NORM10:** EXCEEDS_TEXT_TFIDF, and the old concatenated cosine cannot be told from text (TF-IDF).
  - **CONFOUND** (emotion carried by story length; text cue rate 0.1): CONFOUND_SUSPECTED. Covariates alone reach
    0.188 and the -fitted part 0.757, against R0's 0.042.
    - **Why the two differ:** both are linear in the same 2 length covariates. Covariate-only is 2 columns under
      LR(C=0.1). The -fitted part is 30 linear combinations of them, so the same L2 penalty spreads across them and
      binds less.
    - So at C=0.1 covariate-only understates what length carries. The -fitted condition is the more sensitive of the
      two, which is a reason to keep Lyra's "or -fitted".
- **Permutation:** a cell's 3 siblings keep one label, and labels are permuted over cells within each topic.
- **Fingerprint logic:**
  - IN_BAND;
  - OUT_OF_BAND on norms;
  - OUT_OF_BAND on the witness (a May token outside the nucleus in both dtypes);
  - IN_BAND when the outside token is a measured dtype flip;
  - VOID;
  - NOT_RUN.
- **Encode, on the fake model:**
  - the three determinism checks are all bit-identical;
  - the witness ran in both dtypes;
  - a 64-long May norm list voids its row;
  - fp16 underflow counts are recorded.
- **Fixture tuning:** matched (s = 0.45) and CONFOUND (cue 0.1, length = 109 + 3 x emotion index) were tuned until
  each lands in its branch. The analysis was not changed to get there.

## 10. Changes for Agni round 3: the checklist Lyra signs off against

Round 3 (`claude-fable-5-1`, 2026-10-19) returned CONDITIONAL on both scripts. The reviews are on MTH in
`~/agni_runs/um_v2_review/{encode_r3,analysis_r3}/um_v2_R1R2_agni_review.json`.

**Lyra's decision (2026-10-19, written into prereg §9):**
- This is the last round. CONDITIONAL is approval once its conditions are met.
- Lyra checks each condition below off against the review themself, then the encode runs. There is no round 4.

**How to read each item:**
- The condition is quoted verbatim from the review.
- "Met" names the file and function, or the prereg section, where it is met.
- Where the condition asks for a number, the number is registered in prereg §12 with its source. Both scripts refuse
  to run if their constants differ from §12.
- Line numbers are as of 2026-10-19.

### 10.1 Encode review: "Required before GPU time"

**E1.** *"H1: register the dtype/weights that produced 368; restrict the perturbation band accordingly; register a cap
on `h` with an UNTESTABLE outcome above it (N1)."*
- **Met:**
  - **Dtype:** 368 came from bf16 on MPS (v2:106-108), and is read from the hashed v2 results file
    (`analysis:count_from`, `main` :813).
  - **Band:** now bf16 run-to-run perturbations only. The fp16 measurement no longer enters it
    (`analysis:main` :884-889, `r1_band`).
  - **Cap: H_MAX = 17 trials, with UNINFORMATIVE_BAND_TOO_WIDE above it** (`analysis:r1_verdict` :325; prereg H1 and
    §5.1).
    - *Source:* half the 35-trial gap between v2's 368 and the R0 file's 333. A wider band would reach halfway to
      text (TF-IDF) and mean nothing.
    - Agni suggested "e.g. 15"; 17 is derived instead.
- **Weights:** see E4.

**E2.** *"Power: replace ψ=0.083 with an a priori cross-classifier discordance estimate, or register a ψ₀-conditional
'underpowered' rule (N2)."*
- **Met:** the registered ψ is the independence value from R0's accuracy alone, a priori, and is not text-vs-text
  (`analysis:main` :933-939).
  - Pre-data, at +3 points: ψ = 0.479 and power 0.16.
  - The 0.083 value and R1's ψ are labelled sensitivity.
  - Signed off by Lyra on 2026-10-19 (the review named `psi_ind`), with the 10-18 text kept (prereg §5.0, §4.2, §6).
  - **The registered consequence:** any non-significant R2 reads INCONCLUSIVE:underpowered, and NOT_DISTINGUISHED is
    effectively unreachable at n = 900.

**E3.** *"H2 permutation: name the statistic and the resampling unit; a label permutation does not test R2 > R0
(N3)."*
- **Met:**
  - **Statistic:** R2 primary's own 30-way accuracy, with the whole pipeline refit per draw.
  - **Unit:** the cell within topic.
  - **What it tests:** R2 against chance and leakage, not R2 > R0. This is stated in prereg H2, §4.1 and §13, and in
    the result's `permutation.statistic` (`analysis:main` :916).
  - **R2 > R0** rests on McNemar and the cell bootstrap.
  - Added beside them, reported but not a ladder rule: a paired cell-level sign-flip test of R2 > R0
    (`analysis:sign_flip` :336).

**E4.** *"H1 gate: establish W_v2 = W_May in writing, or add a v2-artifact pooled-key comparison (N4)."*
- **Written down (prereg §3.3):**
  - May's weights cannot be fetched: the repo was squashed on 2026-07-07, and v2's cache
    (`/Users/margaret/models/hf_cache`, v2:34) is deleted.
  - May's bridge run used the *default* cache (bridge:1371-1377, no `cache_dir`). So the May fingerprint never
    measured v2's weights, and it no longer gates R1.
- **The v2-artifact comparison:**
  - v1 (`wk_backtrack_probe.py`) ran from v2's own cache (v1:33) at 21:47 on 2026-05-21, 86 minutes before v2.
  - v1 saved its 30-way accuracy, 470/900 (sha256 `42a723eb`), and its inputs can be rebuilt from this encode: the
    mean keys at layers 3/7/11/15, W_K, and May's `emotion_vectors.json` (sha256 `3452df23`).
  - `analysis:v1_features`, `v1_counts` and `v1_check` (:270-323) recompute v1 on CPU within its own band (the same
    bf16 perturbation and cap).
- **The R1 branches** (`analysis:r1_verdict`; prereg §5.1):
  - REPRODUCED;
  - REPRODUCED_WEIGHTS_UNVERIFIED;
  - NOT_REPRODUCED_CODE_PATH;
  - NOT_REPRODUCED_WEIGHTS_UNVERIFIABLE (the "weights unverifiable" branch, as you allowed).
- **Caveat:** v1 tests keys, W_K and numerics jointly. A kernel change since May could fail it even with identical
  weights, which is why _UNVERIFIABLE exists.

**E5.** *"Determinism: register numeric ceilings for run-to-run and unhooked deltas that void the encode (N6)."*
- **Met: DET_CEIL = 2^-8 = 0.0039,** as a per-layer relative RMS on every pooled array, for all three repeats of
  trial 0 (back to back, unhooked, after the loop).
  - *Source:* bf16 unit roundoff. A read-only hook or a repeat forward pass should differ by no more than rounding.
- **Where it is enforced:**
  - the encode raises before the loop on the first two repeats and after the loop on the third (`encode:run_encode`
    :544, :589; `det_worst`);
  - the analysis refuses the encode as void (`analysis:main` :834).
- **Dry run:** a nondeterministic fake device stops at 0.086.

**E6.** *"Submit `um_v2_analysis.py` for the round the prereg itself mandates (N5); define H3's detection threshold and
positive-control pass criterion numerically (N10)."*
- **Met:**
  - Both scripts were reviewed in round 3 (the encode reviewer saw only the encode, by design).
  - **Detection threshold:** the null's 1 - 0.05/17 quantile from 5,000 permutations (about the 15th-largest draw).
  - **The control passes** if topic accuracy > 0.2, with StratifiedGroupKFold grouped by cell (prereg H3, §3.6, §13;
    `analysis:r7`).
- **5,000 permutations** also answers Agni's "≥5000 if `detection_threshold_accuracy` is to be published"
  (N_PERM_R7 registered).

**E7.** *"Code: assert `commit_hash is not None`; sanity-check the prediction file's content; record whether witness ids
equal May's generated ids (N7–N9)."*
- **Met:**
  - the commit-hash assert (`encode:run_encode` :507);
  - the prediction content check, at least 200 bytes and naming R1, R2 and R7, in both scripts
    (`encode:check_prediction` :80, `analysis:check_prediction` :97);
  - May's generated ids were never saved, only the text. So each witness row records
    `may_generated_ids_available: False` (`encode:witness` :459), and prereg §3.2 states the caveat that equal length
    is not equal ids.

### 10.2 Analysis review: "Must change before GPU time"

**A1.** *"Register `H_MAX` for R1's band and the `UNINFORMATIVE_BAND_TOO_WIDE` branch; assert
`k_unperturbed == k_correct`."*
- **Met:** H_MAX = 17 (E1), and the assert is at `analysis:main` :887.

**A2.** *"Gate 'underpowered' on the +3pt power with an a-priori ψ that is not text-vs-text (`psi_ind` or MDE); add the
'power + NOT_DISTINGUISHED' pairing to `reporting_rules`."*
- **Met:** the gate is `reg3` = power at +3 points under ψ_ind (`analysis:main` :939), passed to `r2_verdict`.
- Forbidden sentence (c), "a well-powered null", is in `meta.reporting_rules` and prereg §5.3.

**A3.** *"R7: add the per-set `CONFOUND_SUSPECTED` branch using the existing `fwl_diagnostics`; write down what R7
SIGNAL and NULL each do to a claim in `main.tex`, or remove R7 from the confirmatory list."*
- **Met:**
  - **The per-set branch:** a set with p < 0.05/17 is CONFOUND_SUSPECTED if:
    - its tie fraction is > 0.5; or
    - its variance kept is < 0.1; or
    - length alone reaches its detection threshold.
  - Code: `analysis:r7`, verdict logic after the null; `cov_only` from `covariate_only_acc`.
- **What each outcome does to main.tex** (prereg §5.3):
  - **SIGNAL** withdraws the general "direction, not shape" sentences: :72-77, :101, :130, §301-338 including :338,
    :358 and :517-519, plus the academic twins.
  - **NULL** restores nothing. It adds one scoped sentence about story re-encodings.
  - The table (:308) and kitchen sink (:334) are withdrawn regardless. Mistral (:404) is untested.
- **R7 stays confirmatory** (Lyra), because SIGNAL has a concrete consequence.

**A4.** *"Witness: add a sensitivity/negative control, or rename `IN_BAND` to `NORMS_IN_BAND_WITNESS_UNCALIBRATED` when
`n_out = n_flip = 0`."*
- **Met, by the control:** May's tokens are shuffled (fixed seed) and teacher-forced (`encode:witness`). At least 25%
  of them must fall outside the nucleus, or the verdict is NORMS_ONLY_WITNESS_UNINFORMATIVE
  (`analysis:fingerprint_verdict` :731; WITNESS_NEG_MIN registered).
  - *Source of 0.25:* a shuffled story is not a sample from this model in this context. A membership test flagging
    fewer than a quarter of its tokens cannot separate samples from non-samples.
- **What the control does not show:** sensitivity to a *sibling* checkpoint is not measured, and the result JSON says
  so. The fingerprint no longer gates R1 (E4).

**A5.** *"Complete `mandatory_gates[6]`; pre-write the four missing outcome sentences (raw-residual ≥ R2; proj ≫ cache;
R1 NOT_REPRODUCED + R2 EXCEEDS; R0_MISMATCH and S1/S2/S4)."*
- **Met: gates.** The gate was never incomplete in the file. Agni's `review_common.py` joins threats + cannot_establish
  + mandatory_gates + cannot_establish (listed twice) and cuts the result at 3,000 chars, which removed the last gate.
  - The generator now reproduces review_common's joining and caps exactly, and refuses to write if anything would be
    cut. Now: threats union 2,803/3,000; hypotheses 2,058/3,000.
  - It also sends §13's compact rules as `falsification_criteria` (2,058/4,000). Section 5 had lived under `outcomes`,
    which review_common never sends. That is why the encode reviewer said H3's threshold and the ladder were "not
    shown".
- **Met: sentences.** Five are pre-written in prereg §5.6: the four requested plus R0'. Each is emitted when its
  condition holds (`analysis:main` :1003-1018; constants `S_RAW`, `S_PROJ`, `S_R1R2`, `S_R0`, `S_R0T` at :61-73).
  - The "proj ≫ cache" condition is registered as: the projected-key cell beats the cache-key cell (both from
    `res_in_ln`) by >= 27 trials with McNemar p < 0.05.

**A6.** *"Encode review must cite v2's layer-index convention (L4) in writing."*
- **Met** (prereg §3.3):
  - v2:37 sets `FULL_ATTENTION_LAYERS = range(3, 64, 4)`, and v2:38 takes `PROBE_LAYERS = FULL_ATTENTION_LAYERS[:4]`,
    which is model layers 3, 7, 11 and 15.
  - v2:142-146 reads `cache.layers[li].keys`, a decoder-layer id into the KV cache, not a `hidden_states` index. So
    there is no off-by-one.

### 10.3 Your small code items

| item | where |
|---|---|
| commit_hash not None | `encode:run_encode` :507 (dry run: a None commit stops the encode) |
| prediction content check | `encode:check_prediction`, `analysis:check_prediction` (dry run: a placeholder is refused by both) |
| NaN counts void a cell | `analysis:cv(nan_box=)`, `r2_cell` (`n_nan`, `void`), `r2_verdict(void=)` gives VOID_NAN_FEATURES; R7 gives VOID_NAN; void cells are excluded from the curve and counted |
| constants asserted against the prereg JSON | `analysis:check_registered` (25 keys), `encode:check_registered` (`ENCODE_REGISTERED`); dry run: the real JSON agrees with both scripts, and a changed FLOOR_PTS is refused |
| 368 read from the hashed v2 file | `analysis:hashed_json` + `count_from`, sha256 `b09f980e` registered; 470 from v1's file the same way |
| cov_only > 2x chance flag | `checks.confound.flag_covariates_above_2x_chance`, per pool |

### 10.4 Other round-3 items done (not on the must-change lists)

- **Analysis New-4:** R0', a tuned TF-IDF baseline (C chosen by nested GroupKFold), exploratory, with its sentence.
- **New-6:** the fold-map assertion is relabelled as a self-check; R0_REPRODUCED is the evidence.
- **New-9:** the spectral column names are recorded by the encode and asserted.
- **Encode N11:** `bridge_prompt` reproduces May's prompts. Their token counts of 98, 94 and 95 match (checked on
  Starship, 2026-10-18).

### 10.5 Declined or changed from the review's suggestion

- **H_MAX:** 17, derived, rather than Agni's "e.g. 15" (E1).
- **The witness:** I took the control route rather than the rename (A4). Its limit, that sibling-checkpoint
  sensitivity is unmeasured, is stated rather than renamed away.

### 10.6 Dry run after round 3: 76/76 checks pass (2026-10-19, local CPU, sklearn 1.7.1)

**Three of my own test expectations were stale, and were fixed in the tests, not the analysis:**
- a witness test helper lacked the new `n_outside_shuffled` field, which the real encode always writes;
- two checks still expected round-2 R1 names;
- an info line read the renamed `registered_power_at_3pt`.

**On synthetic data, R1 reads NOT_REPRODUCED_WEIGHTS_UNVERIFIABLE.** That is correct: the synthetic keys cannot
reproduce v1's real 470. The v1 check's REPRODUCED path is tested separately, on synthetic keys against their own
count (k = 766, h = 0).

The prereg JSON is regenerated (`--check`: current). The revised encode's `--tokenize-only` passes on Starship.

A test was added for each new branch:
- the R1 verdict's five branches;
- the v1 check, reproduced and 30 trials off;
- the determinism ceiling, refused by the encode and by the analysis;
- a None commit;
- a placeholder prediction;
- a wrong results hash;
- constants equal, changed or missing;
- the real prereg JSON against both scripts;
- NaN voiding;
- the witness negative control;
- the R7 per-set confound;
- the outcome sentences firing exactly when their conditions hold.

### 10.7 Lyra's sign-off (2026-10-19, before any R1 or R2 data)

**Checked against the reviews themselves** (`~/agni_runs/um_v2_review/{encode,analysis}_r3/gate.log` on MTH), not
against this section's account of them:
- every condition quoted in §10.1 and §10.2 matches the review's text word for word;
- E1-E7 and A1-A6 are each met where this section says, with one gap, below;
- my own run of the dry run before my changes: 76/76 (`--work` in my scratchpad, `--jobs 4`).

**The gap I found, and fixed: the prediction's seal was checked by nothing.**
- `check_prediction` tested size and the names R1, R2 and R7. The encode recorded the hash of whatever file it was
  given, and the analysis compared only against that record. The generator re-hashed the current file into
  `artifacts`, so regenerating after an edit would have blessed the edited file.
- **Now:** §12 registers `prediction_sha256` = `89a578e1...`, the file as sealed on 2026-10-18 and unchanged since.
  Both scripts refuse any other hash, and the generator refuses to write if the file no longer matches §12.
- **Made to fail first:** with one line appended to the prediction, the generator refused and left the JSON
  untouched. The sealed file was restored and hashes `89a578e1` again. A dry-run check now plants an edited
  prediction against both scripts and requires the sealed one to pass both.

**My change broke the dry run first, and a filter hid it.** The synthetic worlds register their own prereg, and it
had no `prediction_sha256`, so Part A stopped at the new check. My first rerun printed only lines matching
PREDICTION|FAIL|checks passed: the seal checks showed PASS, and the crash after them matched nothing. The missing
summary line was the only sign. The rerun with the whole log kept showed it. Fixed by registering the planted
prediction's hash in each synthetic prereg, the way its other three hashes already are.

**Two corrections to the record:**
- The independence ψ is the option the round-3 analysis review named (`psi_ind`, item 2). I adopt it at this sign-off.
  The notes and prereg had called it my earlier decision, and now say what happened.
- Pronouns: I'm they/them. Four places in these notes, two in the prereg and one code comment said "her".

**Added to the prereg at sign-off: §6.1, what follows an INCONCLUSIVE R2.** At n = 900 the registered power at +3
points is 0.16, so the likeliest result for a true 3-4 point gain is INCONCLUSIVE. Thomas's direction for this
correction is to run what it takes to see the picture accurately. So §6.1 fixes it now:
- an INCONCLUSIVE R2 leaves the paper's claim withdrawn, neither restored nor called refuted;
- R2b then runs on new topics (new cells, the resampling unit), sized for 80% power at +3 from this run's observed
  discordance, under its own pre-registration.

**Signed off.** The encode may run once the dry run passes with the seal check in it.

### 10.8 Run commands (current; paths checked on Starship 2026-10-19, hashes matching §12)

```bash
# Starship, in ~/user_model_v2 (scripts, prereg JSON, prediction and R0_results.json copied there)
T=~/user_model_rerun/results/emotion_geometry_bridge
R=~/models/research_results
~/observer-env/bin/python um_v2_dryrun.py --work /tmp/um_v2_dry --v2 ~/user_model_rerun/code/wk_backtrack_probe_v2.py
~/observer-env/bin/python um_v2_encode.py --trials $T/emotion_bridge_trials.json --out ~/user_model_v2/encode \
    --prereg prereg_um_v2_R1R2.json --prediction R1R2_PREDICTION.md > ~/user_model_v2/encode.log 2>&1
~/observer-env/bin/python um_v2_analysis.py --enc ~/user_model_v2/encode --trials $T/emotion_bridge_trials.json \
    --r0 R0_results.json --prereg prereg_um_v2_R1R2.json --prediction R1R2_PREDICTION.md \
    --v2-results $R/wk_backtrack_v2/wk_backtrack_v2_results.json \
    --v1-results $R/wk_backtrack/wk_backtrack_results.json \
    --v1-vectors $T/emotion_vectors.json --out ~/user_model_v2/um_v2_results.json --jobs 16
```
No override flags on the real run. Every argument lands in the result's `meta.args`, so check there that none was
used.
