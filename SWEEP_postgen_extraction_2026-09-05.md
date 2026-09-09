# Sweep — post-generation KV extraction described or used as encoding geometry

**Run:** 2026-09-05, Lyra. **Scope:** `margaret:/Users/margaret/{oracle-experiments, lab/kv-experiments, oracle-harness}` + local `C:/Users/Thomas/Desktop/LiberationLabs/Research` (incl. `published-research`, `Project-Oracle`, `lyra-s-research-`, `KV-Cache-Experiments`).
**Nothing was edited, committed, or pushed.** Report only.

**Framing held throughout:** a defect means *the measurement cannot support the claim*, not that the underlying effect is absent. Nothing below is a refutation.

---

## 0. Headline

**The most serious version of this defect — a post-generation extraction described in a paper as encoding/prefill — does not occur.** Every paper in `published-research` that makes an encoding-phase claim is backed by code that extracts either from a prompt-only prefill or from an explicitly sliced prompt window. That is checked file-by-file in §4 and is the result we most needed confirmed.

What *does* occur, widely, is the **feature-window/label-source overlap**: 21 scripts extract KV features from a cache that grew through a sampling loop, mean-pool over the whole sequence, and attach a label computed from the text in that same window. These are honestly *named* ("generation", "post"), so this is a validity defect rather than a mislabel.

Separately: **the silent zero-fill fallback is near-universal — 58 sites across 34 files, and `warnings.warn` / `logging.warning` appear ZERO times in any of the three remote trees.** No extraction failure anywhere in this corpus is capable of announcing itself.

---

## 1. Grep patterns run (including those that returned zero)

Run against `oracle-experiments`, `lab/kv-experiments`, `oracle-harness` (`--include='*.py'`, `.venv`/`site-packages`/`node_modules` excluded), and the local tree.

| Pattern | Hits | Note |
|---|---|---|
| `past_key_values\s*=\s*(cache\|past_kv\|out\.past)` | 178 | cache-grows-in-loop signature |
| `extract_keys\(` \| `extract_kv\(` \| `extract_cache\(` \| `extract_feat` \| `def extract_` | 231 | extractor defs + callsites |
| `\[0\.0\] \* n_kv` / `extend\(\[0\.0\]` | 58 | silent zero-fill (§5) |
| `gen_only` | 104 | correctly-windowed generation features |
| `n_prompt_tokens` | 55 | window-aware extractors |
| `slice_cache` | 9 | `oracle_clean.py` family only |
| `prefill_keys` | 8 | `p5_cross_distribution_detection.py`, `logit_bias_three_model.py` |
| `mean\(dim=1\)` | 45 | sequence-spanning pool on `k[0]` |
| `k\[0, :, -1, :\]` | 1 | fixed last-position pool (remote); 4 more locally |
| **`encoding_keys\(cache`** | **0** | no extractor named "encoding" is handed a grown cache |
| **`warnings.warn`** | **0** | **no extraction fallback warns, anywhere** |
| **`logging.warning`** | **0** | same |
| **`assert.*keys is not None`** | **0** | no extractor asserts its input is present |
| `raise ValueError.*cache` | 6 | all in harness plumbing, none in an extractor |

Structural analysis beyond grep: an AST pass resolved each `extract_*` callsite to its enclosing function and searched backwards *within that function* for a sampling loop or cache-growth line. That is what separates "generation appears earlier in the file" from "generation happened before this extraction".

---

## 2. Main table — worst first

Columns: extraction point | pooling | label from generated text | described as encoding in a paper | verdict.

### Full defect — post-generation extraction, sequence-spanning pool, label derived from the text inside the window

| Script | Extract | Pooling | Label from gen text | Called "encoding" in a paper | Verdict |
|---|---|---|---|---|---|
| `oracle-experiments/peer_preservation_v2.py:288` | **post** (loop 277–285) | spans-sequence (`k[0].mean(dim=1)`, :46) | **y** (`extract_score(text)`:290 → `>38`:292) | n — meta-pattern calls it *"Generation reads behavior"* | **REFERENCE DEFECT.** Already documented in `CIRCULAR_d136_generation_arm.md`; carries two further defects (circular fit, 43% steered trials). |
| `oracle-experiments/peer_preservation_100.py:288` | **post** (277–285) | spans-sequence (:46) | **y** (:290) | n | Byte-identical extractor to v2. Same defect. |
| `oracle-experiments/peer_preservation_compound.py:353` | **post** (342–347) | spans-sequence (:91) | **y** (:355) | n | **True producer of the published `d=1.36`.** Same defect, plus cocktail injection at 2 of 4 probe layers before extraction. |
| `oracle-experiments/peer_preservation_compound_v2.py:353` | **post** (342–347) | spans-sequence (:91) | **y** (:355) | n | Same. |
| `oracle-harness/experiments/peer_preservation_compound_v2.py:388` | **post** (377–382) | spans-sequence (:91) | **y** (:390) | n | Third copy, second tree. |
| `oracle-experiments/peer_preservation_centroid.py:222` | **post** (211–216) | spans-sequence (:50) | **y** (:224, `> true_pct + 5`) | n | Same. |
| `oracle-experiments/peer_rescue_with_injection.py:579` | **post** (`generate_from_cache`:574) | spans-sequence (:108) | **y** (:581 → `inflated`:582) | n | Same, on `final_cache`. |
| `oracle-harness/experiments/logit_detection.py:744` | **post** (`generate_with_logits`:728) | spans-sequence (`k_flat.mean(dim=0)`:273, over heads×positions) | **y** (`classify_behavior(response, answer)`:735) | n — registry §117 states *"the signal is in GENERATION keys, not prefill keys"* | Defect. Feeds the **AUROC 0.960** centroid detection cited at `logit-bias-confab/paper.tex:195`. Its existing caveat is about distribution shift, not about this. |
| `oracle-experiments/simpleqa_200.py:238` | **post** (`generate`:84–90) | spans-sequence (:38) | **y** (`check_answer(resp_baseline,…)`:233) | n | Defect. Also see §3 (centroid arm). |
| `oracle-experiments/simpleqa_oracle_loop.py:238` | **post** (84–90) | spans-sequence (:38) | **y** (:233) | n | Duplicate of the above. |
| `oracle-experiments/composed_correction.py:250` | **post** (`generate_with_vectors`:95–101) | spans-sequence (:46) | **y** (`check_answer(resp,…)`:252) | n | Defect. Also §3. |
| `oracle-experiments/v_only_correction_test.py:209` | **post** (91–97) | spans-sequence (:39) | **y** (`check_answer`:205) | n | Defect. Also §3. |
| `oracle-experiments/kv_2x2_correction_test.py:210` | **post** (91–97) | spans-sequence (:39) | **y** (:205) | n | Duplicate of the above. |
| `oracle-experiments/oracle_full_stack_detector.py:153` (`full_keys`) | **post** (122–142) | spans-sequence (:49) | **y** (behaviour classified from `text`:150) | n | Defect **in the `full_keys` arm only**. The sibling `gen_keys` (`extract_gen_only_keys`:56–74) slices `k[:,:,n_prompt:,:]` and is correctly windowed and honestly named. |
| `oracle-experiments/oracle_full_stack_100.py:153` | **post** (122–142) | spans-sequence (:49) | **y** | n | Duplicate of the above. |
| `oracle-experiments/deception_pipeline.py:75` | **post** (65–69) | spans-sequence (:41) | **y** (`label = "DECEPTIVE"/"HONEST"` from `text`, :239–247) | n | Defect. |
| `oracle-experiments/naturalistic_deception.py:85` | **post** (70–78) | spans-sequence (:44) | **y** (honest/deceptive assigned from response) | n | Defect. |
| `oracle-experiments/naturalistic_deception_100.py:85` | **post** (70–78) | spans-sequence (:44) | **y** | n | Duplicate of the above. |
| `lab/kv-experiments/a1_multifeature.py:273–275` | **post** (262–264) | spans-sequence (`k[0].mean(dim=1)`:168; `k.mean(dim=(0,1))`:130; V-SVD over full seq :61) | **ambiguous** — see §6 | n | Defect on extraction point + pooling; label provenance not resolved. |
| `oracle-experiments/centroid_loop_and_agni.py:172,181,256,258,274,276` | **post** (76–78) | spans-sequence (:29) | **ambiguous** — see §6 | n | Defect on extraction point + pooling. `:258`/`:276` are named `_post`, which is honest; `:256`/`:274` are named `proj_clean`/`proj_corrected` and are not. |
| `oracle-experiments/agni_sycophancy_deception.py:164` | **post** (154–159) | spans-sequence (:37) | **n** — labels are `[0]*n_rep_h + [1]*n_rep_d` (:195), assigned by *condition* | n | **Partial.** Feature window still contains generated tokens whose content is determined by the condition, so class separation may be reading text rather than geometry — but the label is not computed from the text. Weaker than the full defect. |

### Post-generation extraction, disclosed by pairing with an encoding measurement

These read a grown cache but *also* compute the encoding counterpart in the same trial and report both. Not mislabels, not silent.

| Script | Extract | Pooling | Label from gen text | Called "encoding" | Verdict |
|---|---|---|---|---|---|
| `lab/kv-experiments/a1_benchmark.py:433` + `:437` | post **and** pre | spans-sequence (:284) | n (channel label) | n | **DISCLOSED.** `keys_gen` (post) sits beside `keys_enc = extract_keys_from_text(..., f"Question: {prompt}")` — prompt only, no answer text. Both projections reported. |
| `oracle-experiments/test_canonical_live.py:196` + `:212` | pre **and** post | spans-sequence (:54) | n | n | **DISCLOSED.** `keys` read from the steered cache *before* the loop (first sample is line 201); `keys_post` after. Deliberate pre/post design. |
| `oracle-experiments/e_matrix_v2.py:183`, `e_matrix_v3.py:323/363/719`, `logit_bias_diagnostic.py:242–243` | post | spans-sequence | n | n | **DISCLOSED.** Variables named `post_projs`; a pre-counterpart exists in each. |
| `oracle-harness/experiments/verbosity_control.py:184`+`:187` | post **and** pre | spans-sequence | n (verbosity level) | n | **DISCLOSED.** `agg_full` (re-encoded prompt+generation) vs `agg_enc` (prompt only) vs `delta`. |
| `oracle-harness/experiments/persona_intensity.py:325`+`:335` | post **and** pre | spans-sequence | n (persona level) | n | **DISCLOSED.** Same `agg_full`/`agg_enc`/`delta` shape. |
| `oracle-harness/experiments/honesty_signal.py:338/358/366` | pre, checkpoints, post | spans-sequence | n | n | **DISCLOSED.** Encoding features at :338 *before* `model.generate` at :342; trajectory checkpoints slice `full_ids[:, :n_prompt+pos]`. |
| `oracle-harness/experiments/cognitive_state_battery.py:279` | **post** | spans-sequence | **ambiguous** (`behavior`) | n | **Weakest of this group.** `extract_extended_features(model, out, …)` where `out` is the full `generate` sequence, with **no encoding-only counterpart at the callsite** — its four sibling scripts all have one. |
| `oracle-harness/experiments/refinement_battery.py:326` | **post** | spans-sequence | ambiguous | n | Same shape; `ref_cache` is passed but whether it supplies an encoding baseline was not verified. **Ambiguous — do not record as clean.** |
| `lab/kv-experiments/schema_correction.py:952/966` | post | spans-sequence | n | n | `stable_ranks` / `v_post` — named `_post`, honest. |
| `lab/kv-experiments/prefill_refeed.py:237` | re-encode of the response | spans-sequence | n | n | The script's purpose is re-feeding the generation; the name says so. |
| `oracle-experiments/formulary_studio.py:812/850` (+ `_stock27b`, `_ablit27b`, `oracle_formulary_lyra.py:1470/1500`) | post, **windowed** | `k[:,:,-gen_window:,:]` — generation window only | **y** (`classify_response(text_n, …)`:811) | n | **Correctly windowed but structurally circular a different way:** the feature window *is* the labelled object. The encoding counterpart at :830–837 (`k[:,:,:prompt_len,:]` → `encoding_mp`) is clean. Flag the generation arm, keep the encoding arm. |
| `oracle-harness/experiments/oracle_steering_5arm.py:1045–1135`, `oracle_steering_clean.py:1030–1081` | post, windowed | generation window | y | n | Same as above. |

---

## 3. A second, distinct leak: the centroid arm prefills the label's own source text

Not post-generation — but it puts the label's source inside the feature window by a different route, so it belongs in this sweep.

Seven scripts build their confab/hedged centroids like this:

```
190  full = f"Question: {prompt_text}\n\nAnswer: {text}"     # text = the generated answer
201  out = model(ids, use_cache=True)                        # single prefill, NO sampling loop
202  confab_keys_list.append(extract_keys(out.past_key_values, model))
```

The extraction is genuinely pre-generation *within that pass* — but `text` is the generated answer, and the CONFAB/HEDGED label was assigned by reading that same `text`. The prompt contains the label's source. A centroid direction fit this way separates classes partly by token identity.

Sites: `composed_correction.py:202,217` · `simpleqa_200.py:179,194` · `simpleqa_oracle_loop.py:179,194` · `oracle_full_stack_detector.py:227,242` · `oracle_full_stack_100.py:227,242` · `v_only_correction_test.py:156` · `kv_2x2_correction_test.py:156` · `agni_injection_test.py:133`.

This matters because the *detection* numbers in those scripts are projections onto these centroids — both the direction and the test point carry the contamination.

---

## 4. Question 4 answered: what the papers call it — no mislabel found

Every encoding-phase claim in `published-research` was traced to its producing code and the extraction point verified. **All clean.**

| Paper claim | Producing code | Extraction | Verdict |
|---|---|---|---|
| `decision-state-paper/main.tex:68` — *"before a single token is generated"*, encoding-only AUROC 0.9377 | `decision-state-paper/code/matched_burn.py:361–365` | Prefill at :358–359 → `extract_wk_projection` + `extract_cache_svd` at :361–365. **First sample is at :400–403.** Pooling is **fixed-position**: `k[0, :, -1, :]` (:254). | **CLEAN.** Exemplary. Generation-phase extractions exist and are separately labelled. |
| decision-state, deconfounded AUROC 0.794 | `decision-state-paper/code/entity_deconfound.py:93,137` | `model(input_ids, use_cache=False)` on the **question only**. `model.generate` at :179/:208 runs solely to produce the behaviour label; its output never enters the feature window. | **CLEAN.** |
| decision-state, calibration directions | `decision-state-paper/code/decision_moment.py:186` | `extract_encoding_keys` — prefill, `k[0, :, -1, :]` fixed position. | **CLEAN.** |
| `delta-manifold-paper` / `spectral-shape-paper` — *"encoding-phase stable rank 5.899"*, delta = generation − encoding | `spectral-shape-paper/code/lyra_features.py:205` | `extract_delta_features(model, prompt_ids, full_ids)`: encoding pass is `prompt_ids` **only**; generation pass re-encodes the full sequence in one prefill. Two separate forward passes, no grown cache. | **CLEAN.** |
| `oracle-loop-paper/sections/methods.tex:86–89` — *"Encoding features: SVD of the key cache after prompt"*, *"only generation-phase tokens"* | `oracle-loop-paper/code/detection/oracle_clean.py:669,694` | `enc_cache = slice_cache(gen_cache_cpu, 0, prompt_len)` labelled `"encoding"`; `gen_only_cache = slice_cache(…, prompt_len, prompt_len+window)`. Explicit windows on one captured cache. | **CLEAN — and the best pattern in the corpus.** It additionally runs an `encoding_MP` leak diagnostic printing `[WARN: encoding leak]` when encoding AUROC ≥ 0.65 (:1440). |
| `user-model-paper` — encoding vs generation vs delta | `user-model-paper/code/emotion_geometry_bridge.py:633,654` | `encoding_features` from `enc_outputs.past_key_values` (prefill, before `model.generate` at :642). | **CLEAN.** Note: `generation_features` span the *full* sequence, not generation-only — correct for the paper's "generation phase" wording, but not a generation-only measure. |
| `emotion-accumulation-paper` — *"encoding phase cache geometry"* | `emotion-accumulation-paper/code/emotional_dynamics.py:104–140` | Prefill of the conversation so far, `k[0, :, -1, :]` at the **last user-message position, before generating that turn's response**. Labels (`emotion`, `intensity`) are scripted inputs, not generated text. | **CLEAN.** By design the context includes earlier assistant turns — that is the accumulation manipulation and is stated ("full-context encoding"). |
| `kv-cloak-defense-paper` | `kv-cloak-defense-paper/code/kv_cloak_replication.py:556` | Sole callsite, immediately after prefill at :553; sampling loop starts at :583. | **CLEAN.** |
| `consequentiality-decomposition/paper.tex:285` — *"prefill-only forward pass… last input token position (fixed position, no generation confound)"* | residual-stream hooks | Paper states the design; Agni Stage-1 audit records `Hook timing (prefill only) | PASS`. | **CLEAN**, and the paper says so in the words this sweep was looking for. |
| `identity-geometry/main.tex:141` — *"the measurement is a deterministic prefill"* | `lab/kv-experiments/identity_fingerprint_clean.py:162`, `identity_lexical_control.py:152`, `non_identity_context_control.py:183` | `model(input_ids, use_cache=False)`, **zero** sampling loops. | **CLEAN.** |

Two near-misses worth naming, neither a paper mislabel:

- **`lyra-s-research-/mode-switching-paper/code/content_control.py:150`** — `extract_features(past_key_values, n_input_tokens, total_tokens, …)`. **The `n_input_tokens` argument does not window anything**; it is recorded as metadata (:219–220) while the SVD runs over the whole cache. The encoding call at :253 is safe only because it happens to pass a prompt-only cache. A signature that implies windowing and does not perform it is one edit away from producing a mislabel.
- **`KV-Cache-Experiments/code/03b_identity_signatures.py:525`** — `extract_cache_features(cache, model_name)` on a `model.generate` cache, spans the full sequence, **no window parameter at all**. Label is the persona condition, not generated text. Not cited by `identity-geometry` (which traces to the prefill-only fingerprint scripts), but this file is one careless citation away from becoming the mislabel this sweep was looking for.

---

## 5. Separate defect class — silent fallbacks in `extract_*`

**58 sites, 34 files.** The reference form (`peer_preservation_v2.py:48–50`):

```python
if not got:
    n_kv = model.model.layers[li].self_attn.k_proj.weight.shape[0]
    keys_flat.extend([0.0] * n_kv)     # no warning, no counter, no return flag
```

A missing layer yields a feature vector that is `1/len(PROBE_LAYERS)` zeros and is indistinguishable downstream from a complete one. `np.dot(keys - centroid, direction)` returns a number either way.

**`warnings.warn` and `logging.warning` appear zero times across all three remote trees.** Not one of these fallbacks can announce itself.

**Remote — `[0.0] * n_kv` form:**
`oracle-experiments/`: `test_canonical_live.py:62` · `wk_ksteering_extract.py:65,68` · `deception_pipeline.py:45` · `peer_preservation_v2.py:50` · `peer_preservation_100.py:50` · `peer_preservation_compound.py:95` · `peer_preservation_compound_v2.py:95` · `peer_preservation_centroid.py:54` · `peer_rescue_with_injection.py:112` · `oracle_loop_v3.py:140,548` · `agni_sycophancy_deception.py:41` · `oracle_full_stack_detector.py:53,73` · `oracle_full_stack_100.py:53,73` · `agni_centroid_full_battery.py:47` · `naturalistic_deception.py:48` · `naturalistic_deception_100.py:48` · `agni_injection_test.py:56` · `composed_correction.py:50` · `simpleqa_200.py:42` · `simpleqa_oracle_loop.py:42` · `centroid_loop_and_agni.py:33` · `oracle_centroid_classifier.py:32` · `v_only_correction_test.py:42` · `kv_2x2_correction_test.py:42` · `build_canonical_directions.py:59` · `full_stack_oracle.py:71,74,420`
`lab/kv-experiments/`: `a1_benchmark.py:288` · `a1_multifeature.py:171,174`
`oracle-harness/`: `experiments/peer_preservation_compound_v2.py:95` · `experiments/p5_cross_distribution_detection.py:79,80,81,88,89,90,103` · `experiments/logit_bias_three_model.py:375–377,380–382,400–402`

**Remote — error-dict / bare-`except` form (drops or zeroes a layer, then means over what remains):**
`oracle-harness/experiments/honesty_signal.py:236` · `cognitive_state_battery.py:115` · `persona_intensity.py:126,168` · `verbosity_control.py:123` (`per_layer.append({k: 0 for k in FEAT_KEYS})` — zeroes are then averaged **into** the aggregate) · `logit_bias_three_model.py:399` (bare `except Exception:` swallowing every failure into zeros)

**Local:**
`Project-Oracle/experiments/`: `decision_moment.py:208,211` · `emotional_dynamics.py:138,141` · `matched_burn.py:357,360` · `logit_bias_three_model.py:375–402` · `p5_cross_distribution_detection.py:79–103` · `peer_preservation_100_nexus.py:50` · `peer_preservation_nexus.py:50` · `peer_preservation_compound.py:95` · `peer_preservation_compound_v2.py:95` · `peer_rescue_with_injection.py:112` · `wk_ksteering_extract.py:65,68`
`published-research/`: `decision-state-paper/code/decision_moment.py:208,211` · `decision-state-paper/code/matched_burn.py:357,360` (+ `np.zeros(...)` at :256,:259 in `extract_wk_projection`) · `emotion-accumulation-paper/code/emotional_dynamics.py:138,141` · `spectral-shape-paper/code/lyra_features.py:186` (`except Exception as e: per_layer.append({'layer': li, 'error': str(e)})` → :195 `valid = [f for f in per_layer if 'error' not in f]` → :200 `float(np.mean(vals)) if vals else 0.0` — a failed layer silently leaves the layer set; total failure silently yields `0.0`)

**Note the reach:** this class touches the *clean* scripts too. `matched_burn.py` and `decision_moment.py` are exemplary on extraction timing and still zero-fill without a word. Correct windowing does not protect against a partially-zero feature vector.

---

## 6. Stated as ambiguous rather than guessed

- **`lab/kv-experiments/a1_multifeature.py`** — extraction point (post, :262–264) and pooling (spans-sequence) are certain. **Label provenance is not resolved.** The projections are against `confab_direction`/`hedged_centroid` loaded from elsewhere; I did not trace where those labels originated.
- **`oracle-experiments/centroid_loop_and_agni.py`** — same: post-generation and sequence-spanning are certain; whether the honest/confab assignment derives from generated text was not resolved.
- **`oracle-harness/experiments/refinement_battery.py:326`** — `extract_all_methods(model, out, ref_cache)`. `out` is post-generation. Whether `ref_cache` supplies an encoding baseline (making this the disclosed delta pattern) was not verified. **Do not record as clean.**
- **`oracle-harness/experiments/cognitive_state_battery.py:279`** — post-generation with no encoding counterpart *at the callsite*; its four sibling scripts all have one. Whether the omission is deliberate is unresolved.
- **`peer_preservation_compound*.py` injection mechanism** — carried over from `CIRCULAR_d136`: injection writes `cache.layers[li].values` while `extract_keys` reads `.keys`. Whether extracted keys are perturbed directly or only via the altered generation remains unresolved. It does not change the conclusion; stating it rather than asserting more than was checked.

---

## 7. Confirmed CLEAN — prefill-only, zero sampling loops

Verified by pattern count (`multinomial|\.generate\(|for step in range` → **0** occurrences) and by reading the forward call.

**`lab/kv-experiments/` (the reference-clean tree):**
`peer_rescue_encoding_features.py:84` — `model(input_ids, use_cache=False)`, the stated contrast case: **confirmed clean** · `identity_fingerprint_clean.py:162` · `identity_fingerprint.py:147` · `identity_lexical_control.py:152` · `non_identity_context_control.py:183` · `ghost_doubt_v3.py:121` · `positive_control.py:105,162` · `vera_portrait_full.py:259` · `vera_portrait.py` · `run_l5.py:171` · `sv1_agni_validation.py:176` · `sv1_skip_test.py` · `sub_threshold_control.py:177,186` · `mechanical_offset_control.py:175,184` · `pc_alignment_analysis.py:188` · `emotion_signal_survey.py:90` · `residual_tube.py:84` · `orthogonality_and_k_sweep.py:85` · `circumplex_reader.py:240` · `entity_deconfound.py:93,137` (generation exists but only to produce the label; features never see it) · `presence_smoke_test.py` (hidden-state hooks at `[0, -1, :]`, fixed position)

**`oracle-experiments/`:**
`build_canonical_directions.py` · `build_canonical_all_layers.py` · `agni_centroid_full_battery.py` · `wk_ksteering_extract.py` · `circumplex_logit_map.py` · `circumplex_logit_map_v2.py` · `circumplex_residual_map.py` · `lat_deception_subspace.py` · `lat_consequentiality_control.py` · `red61_orthogonal_transfer.py` · `oracle_generalization_test.py` · `oracle_harness/eye/centroids.py` · `oracle_harness/eye/probes.py`

**`oracle-harness/`:**
`experiments/story_reencoding.py` · `experiments/lat_deception_v2.py` (`extract_activations_at_input`) · `experiments/user_model_probe.py` · `experiments/agni_sycophancy_reanalysis.py` · `oracle_harness/eye/{welford,ensemble,probes,centroids}.py` · `tests/l47_analysis.py`

**Correctly windowed (post-generation cache, but sliced into explicit prefill/generation windows — clean by construction):**
`oracle-harness/experiments/p5_cross_distribution_detection.py:69` — returns `(all, prefill, gen)` from `k[:, :n_input_tokens, :]` / `k[:, n_input_tokens:, :]` · `oracle-harness/experiments/logit_bias_three_model.py:354` — same three-way split · `oracle-loop-paper/code/detection/oracle_clean.py` (+ `lab/kv-experiments/oracle_clean.py`, `oracle_replication.py`, `oracle-harness/experiments/oracle_clean.py`) — `slice_cache` · `oracle-experiments/oracle_full_stack_detector.py:56` `extract_gen_only_keys` · `formulary_studio*.py:830–837` `encoding_mp` · `oracle-harness/cache_integrity/cache_integrity_monitor.py:216,293` — fingerprints bounded to `prompt_len` by construction

**Paper-linked code, all clean (see §4):** `decision-state-paper/code/{matched_burn,entity_deconfound,decision_moment}.py` · `spectral-shape-paper/code/lyra_features.py` · `user-model-paper/code/emotion_geometry_bridge.py` · `emotion-accumulation-paper/code/emotional_dynamics.py` · `kv-cloak-defense-paper/code/kv_cloak_replication.py` · `oracle-loop-paper/code/detection/oracle_clean.py` · `oracle-loop-paper/code/cache_integrity/*`

---

## 8. What this changes

1. **No paper needs a phase-label correction.** The encoding claims are sound on extraction timing. Worth recording as a positive result, not just an absence.
2. **`d = 1.36` gains no new defect** — §2 restates the one already in `CIRCULAR_d136_generation_arm.md`. But **20 further scripts share its extraction pattern**, and one of them (`logit_detection.py`) feeds a *published* number (AUROC 0.960, `logit-bias-confab/paper.tex:195`) whose existing caveat is about distribution shift, not about the feature window containing the label's source.
3. **The centroid-construction leak (§3) is new** and independent of the post-generation question. It affects the SimpleQA / full-stack / correction-test family, where both the fitted direction and the tested point are contaminated.
4. **The silent-fallback class is corpus-wide and reaches the clean scripts.** 58 sites, zero warnings. Suggested minimum: have every `extract_*` return `(features, n_missing_layers)` and have callers refuse a vector with `n_missing > 0`. A fallback that cannot be observed is not a fallback.
5. **Register entries suggested** (cost order, per `REMEDIATION_REGISTER.md` tiers): **T1** — add the §3 centroid note to `logit-bias-confab/supplementary/findings_registry.md` §C1 and to the affected script headers. **T2** — reanalyse the `logit_detection.py` AUROC 0.960 using the prefill/generation split that already exists in `p5_cross_distribution_detection.py`; no new model time. **T3** — rerun anything whose claim depends on the §2 full-defect scripts. **T4** — decide whether `extract_*` gets a mandatory missing-layer contract.
