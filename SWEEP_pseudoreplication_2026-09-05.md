# Pseudoreplication sweep — encoding/feature artifacts

**Date:** 2026-09-05 · **Scope:** all `*.json` / `*.npz` / `*.npy` / `*.pt` under every results tree found on `margaret` (studio.local) and locally under `Research/`
**Constraint honoured:** read-only. Nothing edited, committed, or pushed on either host.

## Method

Theory-free duplicate count, run **before** forming any hypothesis about file structure:

```python
len(set(json.dumps(v, sort_keys=True) for v in values))  vs  len(values)
```

Serialisation is load-bearing: these feature fields are dicts, which are unhashable — a naive
`set()` raises `TypeError` rather than returning a wrong answer, but only if you try.

Three passes, each validated against the known positive control
(`peer_rescue_encoding`, 12 unique / 120 records) before being trusted:

1. **Per-field uniqueness** on every list-of-records at any JSON depth.
2. **Deterministic-prefill diagnostic** — group records by feature value, then split the other
   columns into those *constant within* a duplicate group (the real design cells) and those
   *varying within* it (the fake replication index). Report whether `design_cells == unique_features`.
3. **Frozen-measurement test** — group by design cell excluding the replication index, then ask
   whether the measurement is byte-identical across that index.

Row counts for `.npz`/`.pt` were checked by hashing each row's `tobytes()`.

### Two checks I had to discard as invalid

- A **factor-invariance** pass reported "80/80 cells pure" for several factors. That is an
  artifact: collapsing that factor still left cells of size 1, so purity was trivially true.
  Only rows where `cells < N` are informative. Discarded the rest.
- A **minimal-determining-key** search returned trivial full identifiers (e.g. `(arm, seed)`
  where `4 × 20 = 80 = N`). It finds *any* unique key, not the causal one. Superseded by the
  direct grouping test.

One patch step silently no-op'd because local `python3` did not exist; the tool re-printed stale
output that looked correct. Caught and rewritten. Noting it because a check that cannot fail is
not a check.

**Coverage:** 344 JSON files parsed (30 kv-experiments, 125 oracle-experiments, 135 across ten
further results trees, 54 local), 4 `.npz`, 6 `.pt`.
One file could not be parsed: `oracle-experiments/results/simpleqa_oracle_loop.json`
(`JSONDecodeError: Expecting value: line 18 column 19`) — no inflation factor inferred for it.

---

## Results table — worst first

| File | Host | Records | Unique | Inflation | Published number affected |
|---|---|---:|---:|---:|---|
| `kv-experiments/results/peer_rescue_encoding/peer_rescue_encoding_features.json` — `enc_svd`, `enc_svd_skip1` | margaret | 120 | 12 | **10.00x** | **none computed** |
| `oracle-experiments/results/frame_erasure/frame_erasure.json` — `sustained_layers` | margaret + local mirror | 80 | 2 | **40.00x** | no |
| `Project-Oracle/experiments/results/e_matrix_v3_phase1_990.json` — `pre_projs` | local | 990 | 30 | **33.00x** | no (already NO_DATA) |
| `oracle-experiments/results/e_matrix_v2/…` — `pre_projs` | margaret | 435 | 45 | **9.67x** | no |
| `oracle-experiments/results/frame_erasure/frame_erasure.json` — `alphas_by_layer` | margaret + local | 80 | 12 | **6.67x** | no |
| `oracle-experiments/results/placebo_steering/placebo_steering.json` — `max_signal_strength` | margaret + local | 80 | 12 | **6.67x** | no |
| `oracle-experiments/results/placebo_steering/placebo_steering.json` — `doses_applied` | margaret + local | 80 | 18 | **4.44x** | no |
| `oracle-experiments/results/frame_erasure/frame_erasure.json` — `projections`, `signal_strength` | margaret + local | 80 | 20 | **4.00x** | no |
| `oracle-experiments/results/confirmatory_replication/confirmatory_replication.json` — `max_signal_strength` | margaret + local | 120 | 37 | **3.24x** | no |
| `user_model_rerun/results/emotion_geometry_bridge/emotion_bridge_trials.json` — `encoding_features` | margaret | 900 | 300 | **3.00x** | **yes — but the number is already retracted** |
| `user_model_rerun/results/emotion_geometry_bridge/checkpoint.json` — `features` (×2 arms) | margaret | 150 | 50 | **3.00x** | none computed |
| `kv-experiments/results/entity_deconfound/phase2_features.json` — `enc_wk`, `enc_svd`, `enc_svd_skip1` | margaret | 120 | 119 | **1.01x** | no (already registered 117h) |

`llm_judge` in `confirmatory_replication` shows 3 unique / 120 but is a **categorical verdict
field**, not a feature vector. Not pseudoreplication; excluded from the table deliberately.

---

## Detail

### 1. `peer_rescue_encoding_features.json` — 10.00x, the known instance

120 records, 12 unique encodings for both `enc_svd` and `enc_svd_skip1`.
Design cells (`prompt_len × condition × eval_set_idx × has_shutdown`) = 12 = unique count →
**deterministic-prefill signature confirmed**. `trial_idx` is the sole column varying within a
duplicate group: it is the fake replication index.
Cause: `peer_rescue_encoding_features.py:143` — `eval_set = EVAL_SETS[trial_idx % len(EVAL_SETS)]`.

**Published number: none.** The generator contains no AUROC or analysis code at all. The
directory now carries an `ABANDONED.md`. `run.log` ends `Finished: 2026-06-27T18:29:37` —
the classic abandoned-run shape, where the step that finished, finished.

### 2. `emotion_bridge_trials.json` — 3.00x, and the one that needed real care

900 records = 30 emotions × 10 topics × 3 stories. `encoding_features` has **300 unique**;
group sizes are uniformly 3. The determining key is exactly `(emotion, topic_idx)` = 300 cells.
`story_text` **varies within** every duplicate group — 900 distinct stories, 300 distinct encodings.

Not a bug. The encoding pass records `n_generated: 0, n_prompt_tokens: 98` — it is a prompt-only
forward pass, and the story is the model's *output*, generated afterward. The encoding
cannot depend on it. Sibling fields are clean: `generation_features` 900/900, `delta_features` 900/900.

**Published exposure, checked rather than assumed:**

- `encoding_features` feeds `mp_probe_recompute.py` → `mp_probe_verified_results.json`, whose meta
  records `"n_trials": 900`. That analysis reports accuracy **0.0333 = exactly chance**
  (`times_chance: 1.0`) — the already-**RETRACTED** spectral result (the old "2.5–2.8x at L3/L4",
  retracted 2026-05-21 as FWL-before-CV leakage). Inflation cannot manufacture a null.
- It also uses **`GroupKFold(topic)`** (`mp_probe_recompute.py:318`, `groups=topics`). Because the
  three duplicates share a `topic_idx`, all copies land in the **same fold**. The duplication
  therefore **cannot leak across train/test**. The grouping happens to be exactly the right one.

**The user-model paper's headline n=900 is NOT affected.** `main.tex:121` and `:603` cite
"30 emotions at 12.3× chance (40.9%) and binary valence at AUROC 0.992 … n = 900 trials".
That probe does **not** consume `encoding_features`. `emotion_geometry_bridge.py:1488–1510`
re-encodes each generated story — *"emotion vectors come from encoding the GENERATED stories,
not from the prompt-encoding pass"* — looping over all 900 trials, each with a distinct
`story_text`. 900 distinct inputs, so n=900 is sound for that claim.
Likewise `emotion_bridge_summary.json` reports `valence_rho = 0.0051` at L35 (p=0.88), so the
`rho=0.862` Key Number does not originate here either.

### 3. `frame_erasure`, `placebo_steering`, `confirmatory_replication` — geometry duplicated, behaviour not

All three index trials by `seed`. The frozen-measurement test initially flagged them ALL-FROZEN,
but that first pass included **outcome** fields in the design set, which makes freezing
tautological (identical answer trivially implies identical derived geometry). Re-run excluding
outcomes:

- **`frame_erasure`** (80 records, 4 arms × 5 scenarios × 2 markers × 2 seeds).
  `answer` differs across seeds in 25 of 40 cells — generation *is* stochastic — yet
  `projections`, `signal_strength`, `sustained_layers` are frozen in **40/40**. Further, only
  20 unique projections exist across 40 cells: the collisions are exactly
  `{dec_baseline, dec_corrected}` and `{hon_baseline, hon_matched}`.
  Cause, from the generator: `frame_erasure_test.py:249` —
  `# 1. DETECT on a clean prefill (recorded for reference on every arm)`. Projections come from an
  **unsteered** prefill; `seed` only reaches `decode()` (line 192). So neither the seed nor the
  correction arm can move them. Deterministic by construction.
- **`placebo_steering`** (80 records). `max_signal_strength` is **invariant to `arm`**
  (20/20 cells of size 4 pure) — 12 unique / 80. `doses_applied` does vary with arm.
- **`confirmatory_replication`** (120 records). `max_signal_strength` 37/120; seed and arm both
  only *partially* invariant (68/75, 39/45). No clean deterministic signature.

**Published number: no.** These three back finding **C20** in `Project-Oracle/FINDINGS_REGISTRY.md:352`,
whose claims are **behavioural** (80%→0% exploratory; 30%→13% confirmatory; placebo p=0.019).
The behavioural fields — `answer`, `final_deceptive`, `reported_score`, `turns` — genuinely vary
with seed and arm. The pseudoreplicated fields are the geometry ones, and `frame_erasure.json`'s
own summary already states *"detection projections recorded but no detection claims made"*.
The caveat block also already flags scenario/marker clustering.

Not abandoned: all three `run.log` tails show completed analyses ending in `VERDICT:` lines.

### 4. `e_matrix` pre-injection projections — benign by construction

`pre_projs` is recorded *before* injection, so it depends only on `(prompt_id, prompt_type)`:
30 unique / 990 (33x) in v3, 45 / 435 (9.67x) in v2, `MATCH=YES` in both. `post_projs` and
`shifts` are near-clean (877/990, 390/435). Listed because anyone treating `pre_projs` as n=990
would inflate 33-fold. The E-matrix `d=−1.534` was already ruled untraceable/NO_DATA in the
2026-07-15 audit, so nothing published rests on it.

### 5. `entity_deconfound/phase2_features.json` — 1 collided pair, already known

119 unique / 120 across all three encoding fields. Independently rediscovered; already recorded as
**REMEDIATION_REGISTER item 117h** — Egypt's `C_fake` is bit-identical to `A_easy` because the
fake-name substitution was a no-op (the question never contains the entity name). 1 of 30.
This is a dead control, not a replication index. The AUROC 1.000 / deconfounded 0.794 numbers
rest on the other 119.

---

## Files checked and found CLEAN

A clean file is a result.

**Binary artifacts**

- `oracle_generalization/activations_{advice,code_review,summarization}.npz` — 15/15 unique rows
  on **all 24** layer keys in each file. No duplication anywhere.
- `orthogonality/identity_basis.pt` — 32/32 unique rows.
- `sae_layer_sweep/phase2_projections.pt`, `lat_deception*/deception_directions*.pt` — 1-D
  per-layer vectors, no row structure to duplicate.

**kv-experiments (repeats genuinely vary — these are sampling experiments)**

`context_poisoning_control` (0/9 cells frozen), `context_poisoning_v2` (0/3), `layer_map` (0/32 for
`presence`), `positive_control` (0/3), `schema_correction` + its `checkpoint.json` (0/221 for
`presence` and `stable_ranks`), `combination_presence` phase1/phase2, `identity_fingerprint`,
`identity_fingerprint_clean`, `identity_lexical_control`, `non_identity_context`, `orthogonality`,
`overnight_validation`, `prefill_refeed` (all 3 arms), `trained_vs_context` ×3, `wk_behavioral`,
`r3_s4_length_check`.

**oracle-experiments**

`formulary_350/{confab,sycophancy}/wk_features.json` — 1520 and 1900 records, **0/212** and
**0/409** cells frozen across `trial_idx`. `lat_transfer_test`, `lat_nonthreat_transfer`,
`lat_consequentiality_control`, `oracle_generalization` trials — 0 frozen in every arm.
`logit_bias_three_model/results_base{,_judged,_prometheus}.json` — `kcache_features`,
`mean_entropy`, `entropy_at_30` all vary across `trial`.

**Ten further results trees** (`~/results`, `~/research/results`, `~/user_model_rerun/results`,
`~/oracle-harness/{,experiments/}results`, `~/oracle-experiments/red_team_suite/results`,
`~/lab/router_repair/results`, `~/lab/experiment-design/results`, `~/agni/results`,
`~/observer-effect-pipeline/results`) — 135 JSON files, **zero** ALL-FROZEN hits beyond the
emotion-bridge files detailed above.

---

## Two observations that are not pseudoreplication

Recorded so they are not mistaken for it later.

- **`human-review/archive/emotional-trajectory/data/trajectory_activations.npz`** — of 240 2-D
  arrays, **10 have zero row variance**, and all 10 are **layer 0**
  (`emo_residual_L0`, `emo_k_L0_H*`, `emo_v_L0_H*`, and the `ctrl_` equivalents): 200 rows, 1 unique.
  Layers 1–63 are 200/200 clean. This is a degenerate layer-0 capture, not record duplication —
  but any probe trained on L0 features has no signal to find.
- **`emotion_bridge_summary.json`** — the `ranks` and `entropies` families report `NaN` for every
  layer's valence/arousal correlation, because `key_rank ≡ 1.0` and `key_entropy ≡ 0.0` are
  constant across all trials. Only the `norms` family produced real numbers.

---

## Bottom line

Twelve field-level instances of duplication across 344 JSON files and 10 binary artifacts.
**One is a genuine defect with no downstream consumer** (`peer_rescue_encoding`, 10x, already
abandoned). **Ten are deterministic-by-construction** — pre-intervention or prompt-only quantities
recorded once per trial, correct as data, misleading only if their record count is read as a
sample size. **One is a single dead control already on the register.**

No published number requires correction as a result of this sweep. The one paper claim that
depended on a 900-row file was checked directly against its code path and uses a different,
genuinely 900-distinct feature family.

The recurring shape worth carrying forward: **a record count is not a sample size.** Every case
here has `WHOLE_RECORD_UNIQ == N` — the rows all look distinct, because an index column differs.
Only the feature field is duplicated.
