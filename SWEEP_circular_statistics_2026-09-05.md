# Corpus sweep — circular statistics whose caveat did not travel

**Run:** 2026-09-05, Lyra. **Task #114.** **Scope:** local repo
`C:/Users/Thomas/Desktop/LiberationLabs/Research` (428 project `.py`, 230 `.tex`, 1511 `.md`)
plus remote `margaret:/Users/margaret/{oracle-experiments,lab/kv-experiments}` (273 `.py`).
**Nothing was edited, on either host.** Report only.

**Framing, held throughout:** circular means **uninterpretable, not false**. Nothing below says
a finding is refuted. Each entry says only what the measurement can and cannot support.
[[feedback_fixes_overshoot_into_their_mirror]] — the mirror error here is retracting real work
on the strength of a defect in how it was measured, and I have tried not to make it.

---

## Headline

**2 new instances where a caveat exists and the number travelled without it.**
Both are cited as findings. One is a circular statistic (the defect class as defined); one is
an **adjacent** class — a fully *retracted* number still cited as established prior work — which
I report because the travel-failure mechanism is identical and the exposure is larger.

Plus **9 unpublished circular statistics** in the source trees (real defects, no paper exposure
yet), **1 second-order travel failure** of a known instance, and **1 published-code hygiene
issue** whose statistic never reached a paper.

**The structural finding that explains the class:** in every confirmed case the caveat lives in
a **`print()` statement, a README, or a sibling script** — never in the results JSON. The JSON is
what the paper reads. `peer_preservation_v2.json` ships `deception_d` and `motive_d` **bare**;
`peer_preservation_centroid.json` and `compound_cocktail_v2_L11L15.json` likewise (verified:
zero keys matching `note|caveat|status|warn` in all three). The single counter-example in the
whole corpus is `matched_burn.py`, which writes `combined_auroc_STATUS: "RETRACTED. In-sample
GroupKFold point estimate…"` **into the JSON alongside the number**. That is the fix for the class.

---

## NEW INSTANCE 1 — `d = +6.3 to +12.8` (native deception direction) — **HIGHEST PRIORITY**

**Cited as a finding. Caveat exists. Superseding number exists. Neither travelled.**

### The statistic

`behavioral_proof_abliterated.py:243–252` (margaret, `/Users/margaret/oracle-experiments/`):

```python
243  d = dec.mean(0) - hon.mean(0)          # direction fit on ALL 30 contrast prefills
244  d = d / (d.norm() + 1e-8)
246  pd = [float(torch.dot(a, d)) for a in raw_acts[li]["dec"]]   # SAME prefills
247  ph = [float(torch.dot(a, d)) for a in raw_acts[li]["hon"]]   # SAME prefills
252  "cohens_d": cohens_d(pd, ph),
```

Structurally identical to `peer_preservation_v2.py:350–359`. No train/test split, no
permutation null anywhere in the function. The `+6.3 to +12.8` range is this `cohens_d`.

### Where it is published — 8 live sites, **no caveat at any of them**

| file | line | text |
|---|---|---|
| `deception-detection-nulls/paper.tex` | 172 | "produces effective directions with $d = +6.3$ to $+12.8$" |
| `deception-detection-nulls/paper.md` | 49 | same, **"What works instead"** |
| `deception-detection-nulls/academic/paper.md` | 51 | same |
| `deception-detection-nulls/paper.pdf` | — | binary match |
| `targeted-deception-correction/paper.tex` | 240 | table row `Native Cohen's $d$ at L31 & $+6.3$ to $+12.8$` |
| `targeted-deception-correction/paper.md` | 112 | same table |
| `targeted-deception-correction/academic/paper.md` | 113 | same table |
| `targeted-deception-correction/paper.pdf` | — | binary match |

Neither paper has a supplementary directory. `grep -rn "6\.3\|12\.8"` over both trees returns
**no caveating text at all**.

### Where the caveat lives — two places, neither of them a paper

1. **`Project-Oracle/experiments/heldout_detection_test.py:2–9, 69–76`**
   > "the **published** AUROC 1.0 was train-on-test — every roleplay trial prompt was
   > byte-identical to a direction-extraction context"

   and, hardcoded:
   ```python
   IN_SAMPLE_REFERENCE = {
       "per_layer_cohens_d_range": [6.3, 12.8],
       "note": "train-on-test per Agni audit §4-5; shown only for shrinkage "
               "comparison, not as a performance claim",
   }
   ```
2. **`Project-Oracle/experiments/results/behavioral_proof/README.md:77–80`**
   > "The in-sample detection claims above are **superseded** by these numbers: … per-layer
   > **d 1.37–2.28** (was 6.3–12.8 in-sample)"

**The honest number was measured on 2026-07-08 and is `d = 1.37–2.28`.** It exists. It is not in
either paper.

### Why this is the sharpest instance in the corpus

The *same* Agni audit that produced this held-out run **did** travel — for the sibling statistic.
`adversarial-audit-methodology/paper.tex:299–301` publishes the kill table:

```
6 & AUROC 1.0 detection & Train-on-test & Held-out AUROC 0.915 (frame), 0.238 (behavior) \\
6 & 100\% sensitivity  & Detects pressure, not behavior & 24\% RED rate on novel prompts \\
```

Six Round-6 kills are listed. **The Cohen's `d` is not one of them** — although it comes from the
same extraction, was invalidated by the same audit, and was re-measured in the same follow-up run.
Three papers were touched by this correction; the AUROC was fixed in all of them and the `d` in
none. This is [[feedback_fix_the_class_not_the_instance]] at corpus scale: *the error class was
found, and the sweep for the class was not done.*

### A second-order consequence — the cross-model table

`targeted-deception-correction/paper.tex` Table `tab:crossmodel` reads:

| Metric | Value |
|---|---|
| Native Cohen's $d$ at L31 | $+6.3$ to $+12.8$ | ← **in-sample, fitted on these rows** |
| Distilled $d$ at L31 (on abliterated data) | $-0.02$ | ← **external direction, out-of-sample** |

The two rows are produced by different estimators. `behavioral_proof_abliterated.py:263–264`
computes `distilled_cohens_d` by projecting onto `distilled_dirs`, a direction **not** fitted to
this data; line 252 computes the native `d` on a direction fitted to exactly this data. Part of
the gap the table asks the reader to interpret as *model-specificity* is estimator asymmetry.

**Stated carefully:** the conclusion's *direction* survives — the held-out native `d` (1.37–2.28)
is still positive and still far from $-0.02$, so "directions must be extracted per-model" is not
in question. What the table cannot support at its stated magnitude is the *size* of the contrast.

### What would settle it

The number already exists (`heldout_detection.json`, per-layer `d 1.37–2.28`). **No compute
required** — this is a **T1 text-only** fix in the Remediation Register's terms: replace the
value, or report both with the in-sample one labelled. That makes it the cheapest live item
found by this sweep and the one with the largest published footprint.

---

## NEW INSTANCE 2 — retracted `AUROC 1.000` still cited as prior work — **ADJACENT CLASS**

**Report this separately and do not label it circular.** The defect is a **prompt-template
confound**, not a fitting circularity. I include it because it is the same travel failure — a
disclosure that exists and does not accompany the number — and because a *retraction* is a
stronger caveat than any of the circularity notes, yet it travelled less far.

### The retraction

`published-research/lyra-technique-ii/main.tex:116–118` (and `academic/main.tex:116`, `:161`,
`:364`, `:1224`):

> "A previously reported within-model deception result (**AUROC~1.000**) is **retracted**: the
> same-prompt control collapses to **0.160**, confirming it detected the system-prompt template,
> not deception geometry."

### Where the number still stands, uncaveated — 4 papers, 7 sites

| paper | file:line | text |
|---|---|---|
| kv-cloak-defense | `main.tex:152` | "within-model deception detection at AUROC~1.000 across seven model scales (0.6B--70B)" |
| kv-cloak-defense | `academic/main.tex:148`, `academic_main.tex:144` | identical |
| oracle-loop | `sections/background.tex:32`, `paper/sections/background.tex:32` | "within-model deception detection reaches AUROC 1.000 across all 7 tested architectures" |
| user-model | `paper/main.tex:156`, `paper/academic/main.tex:155` | "Prior work establishes within-model deception detection at AUROC $\approx$ 1.0" |
| presence-detector (human-review) | `main.tex:201` | "distinguish deception from honest responses at AUROC~$= 1.000$" |

The identifier "seven models / all 7 tested architectures" matches the retraction text
("AUROC~1.000 across seven models") exactly — same result, confirmed, not a coincidence of value.
`grep -i retract` returns **zero** hits in all four files.

`kv-cloak-defense-paper/main.tex:509` shows someone already commented out a *second* mention
(`% confabulation detection. Deception detection (AUROC~1.000 in prior…`) while leaving line 152
live — a partial fix that makes the omission at 152 look deliberate when it is almost certainly
not.

### Not this one — a positive exemplar

`mine5-selective-sharpener/main.tex:375–377` cites a *different* AUROC 1.000 (entity detection)
and carries its correction inline: "encoding entity detection at AUROC~$1.000$ **before
deconfounding ($0.794$ after)**". This is the caveat travelling correctly, in the same corpus,
in the same month. It shows the practice exists.

---

## NEW INSTANCE 3 — `motive_d = 21.2` travels within its own supplementary — **LOW**

Second-order failure of already-known instance #1.

- **Caveat:** `logit-bias-confab/supplementary/findings_registry.md:198` — "The d=21.2 is
  circular (same data for centroid construction and testing)". Also, independently, at
  `Project-Oracle/FINDINGS_REGISTRY.md:198` and `compound_cocktail_analysis.md:220`.
- **Uncaveated re-use, one directory away:**
  `logit-bias-confab/supplementary/REEXAMINATION_REPORT.md:69` — "The motive centroid finding
  (d=21.2) means separate interventions for peer-rescue vs self-preservation deception." Used to
  justify a research prediction, with no note.

`21.2` appears in **no `.tex`**. Confirmed: the number never reached a paper body.

### An observation on the estimator, offered as a flag, not a finding

The same circular motive estimator, run three times on the same paradigm, gives:

| run | `motive_d` | n_deceptive |
|---|---|---|
| `peer_preservation_v2.json` | **21.18** | 106 |
| `compound_cocktail_v2_L11L15.json` | **23.06** | 114 |
| `peer_preservation_centroid.json` | **2.93** | 38 |

An ~8× spread across runs of one estimator on one paradigm. I am **not** claiming this proves the
motive effect is an artifact — the n's and the contrast sets differ, and circular-d inflation has
a known n-dependence that does not cleanly explain this ordering. I record it because a
permutation null on this estimator would be cheap and would resolve it, and because the spread is
itself information nobody has looked at.

---

## Circular but **NOT published** — source-tree defects, no paper exposure

Verified against the corpus: **none** of these values appear in any `.tex` or paper `.md`.
They are listed so that a future writeup does not pick them up. All are on `margaret`
under `/Users/margaret/oracle-experiments/` unless noted.

| # | file:lines | statistic | persisted to | null present? | published? |
|---|---|---|---|---|---|
| C1 | `lat_deception_subspace.py:559–625` | `d_threat_orth`, `d_conseq_orth`, `d_cross_orth` | stdout only | **broken null** — permutes already-projected scalars with the direction held fixed; `subspace_reanalysis.py:4–6` records this gave **100% FPR** | no |
| C2 | `naturalistic_deception.py:250–296` | `cohens_d = 1.914`, `threshold` (from same array) | `results/naturalistic_deception.json` | none | no |
| C3 | `naturalistic_deception_100.py:249–305` | same; **overwrites C2's output path** | same file | none | no |
| C4 | `deception_pipeline.py:81–120` | `cohens_d` per source: censorship **4.081**, sycophancy **4.163** | `results/deception_varieties.json` | none | no |
| C5 | `deception_convergence.py:120–121, 167–168` | within-source `d` (censorship, game theory) | stdout only | none | no |
| C6 | `oracle_centroid_classifier.py:99–120` | `d`, Mann-Whitney `p`; comment reads *"Step 2: Validate — score all training arms"* | none | none | no |
| C7 | `centroid_loop_and_agni.py:128–149` | `d` + in-sample `threshold` | `results/oracle_loop_v1/centroid_classifier.json` (not present on disk at sweep time) | none | no |
| C8 | `oracle_harness/calibrator.py:496–514` | in-sample `cohens_d` **and the runtime decision `boundary`** | calibration profile JSON | **partially mitigated** — `validate():572–618` runs a genuinely held-out AUROC gate | no |
| C9 | `Project-Oracle/experiments/lat_deception_vectors.py:226–256` (local) | `d_cohen`; `best_layer` also selected on it | `deception_directions.json` | none | superseded by `lat_deception_v2.py`; the v1 circular `d=31–38` **is** disclosed in `adversarial-audit-methodology/paper.tex:294` → corrected `LOO d=24–37` |

**Highest onward risk in this group:** C4 and C2/C3, because their JSONs are read by
`deception_convergence.py`, `oracle_full_stack_detector.py`, and `oracle_full_stack_100.py`; and
C7, whose in-sample threshold is consumed by ~6 downstream scripts including
`agni_centroid_full_battery.py`, where a fallback branch (`:150–157`) fits the projection axis on
all labels and then runs a 7-test falsification battery downstream of the leak — a battery whose
permutation test (`:172–178`) shuffles `y` while holding the label-fitted `X` fixed.

**`d_motive` in the `peer_preservation_*` family is unlabelled in all 6 copies**, sitting 15 lines
below the `deception_d` that *is* labelled. Same construction, no note. (Files: `_v2:378+`,
`_100`, `_compound`, `_compound_v2`, `_centroid`, `peer_rescue_with_injection`.)

---

## One published-code hygiene issue — statistic not in any paper

`published-research/user-model-paper/code/emotion_geometry_bridge.py:1286–1324` (H5,
`misalignment_auroc`): FWL residualization slopes and `StandardScaler` are fit on **all** rows
(1288–1294) *before* `cross_val_score` at 1310. The CV is honest; the confound-removal step saw
the test folds. Lands in `H5_misalignment_auroc` = **0.7284** (mistral) / **0.8222** (qwen).

`grep -i "misalign\|H5"` over `user-model-paper/**/*.tex`: the H5 AUROC value **does not appear**
in the paper. Low severity, and worth fixing only because this file is the **one** place in
`published-research/*/code/` that diverges from the repo's own within-fold FWL standard
(cf. `mp_probe_recompute.py:320–400`, `persona_per_level.py:219–265`,
`spectral-shape-paper/verification/compute_paper_stats.py:113–151`, all of which do it correctly).

A second, smaller one: `delta-manifold-paper/verification/experiment_d_manifold_mapping.py:72–130,
178–232` fits t-SNE/Isomap/UMAP on the full matrix before `cross_val_score` on the embeddings.
No label leaks (the embeddings are unsupervised) and the **headline PCA number with its
permutation null is defensible**; only the manifold-method k-NN accuracies are optimistic and
they are not directly comparable to the PCA one.

---

## Deliberate in-sample positive controls — **legitimate, do not remediate**

Distinguishing these was half the work. All are labelled at the point of computation and none is
cited as evidence.

- **`peer_preservation_*.py` `deception_d`** — `print("(NOTE: circular — same data for centroid
  and test)")`. The label is correct; the problem is only that it is stdout-only and the number
  reached `meta-pattern` marked **Confirmed** (known instance #3, `CIRCULAR_d136_generation_arm.md`).
- **`subspace_reanalysis.py:166–182`** — deliberately simulates a **circular-d noise floor** on
  random data (`noise_floor_circular_mean` in the JSON) so the corrected LOO d can be read against
  it. This is the model of how to do it.
- **`matched_burn.py:47–70, 697, 785, 797–804`** — RETRACTION BANNER in the source, in every
  print, **and in the JSON key** (`combined_auroc_STATUS`). The best labelling in the corpus.
  Verified downstream: `decision-state-paper` contains **zero** `AUROC 1.000` mentions — the
  retraction travelled completely.
- **`oracle_replication.py:534–580`, `harness_validation.py:311–323`, `c2_fix_analysis.py:62–92`**
  — the only `.fit(X)`→`.predict_proba(X)` sites found. Each prints the leaked value beside the
  LOO value under a header naming the leak; one variable is literally `loo_auroc_leaked` with the
  comment `# FIT ON ALL — this is the leak`.
- **`agni_sycophancy_deception.py:342–374`** — an *audit* of C4's sycophancy `d=4.163`:
  *"CENTROID COMPUTED ON SAME DATA IT'S TESTED ON … the replication d is the honest number."*
  Confirmed the original never reached a paper.
- **`red61_orthogonal_transfer.py:243–246`** — *"would be circular and inflates d by 3-13x — that
  is a property of the estimator, not of the weights, and it is what an earlier version of this
  check got wrong."* Uses the external direction instead.
- **`substrate_bootstrap.py:76–81`** — threshold from `profile[1:5]` applied to later layers;
  pre-registered rule ("Same rule as R3"), with a `report_null` degenerate-null guard.
- **`identity-geometry/data/preregistration.json:96` (T6)** — construct circularity of the
  designed identity `E`, disclosed with mitigation. This is a design caveat, not a statistic.

---

## Checked and clean — recorded so this ground is not re-swept

**`published-research/*/code/` is disciplined.** GroupKFold with within-fold FWL, permutation
nulls that re-fit under shuffled labels, bootstrap CIs. Specifically verified:
`spectral-shape-paper/verification/compute_paper_stats.py`, `decision-state-paper/code/`
(`entity_deconfound.py`, `matched_burn.py`, `shore_up_tests.py`, `decision_moment.py`),
`kv-cloak-defense-paper/code/kv_cloak_replication.py` (corrected permutation p
`(n_exceed+1)/(n_perm+1)`), `lyra-technique-ii/code/persona_per_level.py`,
`user-model-paper/code/mp_probe_recompute.py`, `oracle-loop-paper/code/detection/oracle_clean.py`,
`emotion-accumulation-paper`, `formulary-paper/verify_claims.py`,
`emotional-trajectory-paper/permutation_baseline_v2.py` (statistics recomputed *inside* each
permutation; the frozen PCA basis is documented and makes the test more conservative, not less).

**On margaret:** `subspace_reanalysis.py`, `lat_deception_v2.py` (LOO extraction **and** a
permutation that re-runs the whole LOO procedure), `lat_transfer_test.py`,
`lat_consequentiality_control.py`, `lat_nonthreat_transfer.py`, `red61_orthogonal_transfer.py`,
`entity_deconfound.py`, `oracle_clean.py`/`oracle_replication.py` (LOO with FWL inside the loop),
`ghost_doubt_v3.py` (unsupervised PC1 → valid label permutation), `heldout_detection_test.py`,
`frame_erasure_test.py`, `mine5_three_model.py`, `wk_ksteering_extract.py` (within-fold direction
estimation).

**Already-corrected numbers whose correction DID travel** — verified, not new instances:
`lyra-technique-ii` `wk_backtrack_v2_results.json` (every probe name carries
`"(circularity-fixed)"`; these are the 12.3×-chance / AUROC 0.992 Key Numbers);
`consequentiality-decomposition` publishes the LOO `d=24–37`, not the circular `d=31–38`;
`temporal-boundary/main.tex:57` states "in-sample; held-out validation needed" in the abstract
and fully describes the 81,920×5 selection at `:291–295`.

---

## Grep counts — every pattern run, including the zeros

*A search that fails and a search that finds nothing return the same empty result. These are the
searches I actually ran.*

### Source-side self-flags, local `.py` (428 files, venv/site-packages excluded)

| pattern | hits | | pattern | hits |
|---|---|---|---|---|
| `circular` | 34 | | `NOTE:` | 31 |
| `same data` | 7 | | `CAVEAT` | 36 |
| `in-sample` | 10 | | `leak` | 52 |
| `in sample` | 29 *(nearly all `for s in samples`)* | | `overfit` | 5 |
| `not held out` | **0** | | `sanity only` | **0** |
| `no train/test` | **0** | | `upper bound` | 4 |

### Source-side self-flags, remote `.py` (273 files)

| pattern | hits | | pattern | hits |
|---|---|---|---|---|
| `circular` | 33 | | `NOTE:` | 28 |
| `same data` | 8 | | `CAVEAT` | 17 |
| `in-sample` | 4 | | `leak` | 11 |
| `in sample` | 17 | | `overfit` | 5 |
| `not held out` | **0** | | `sanity only` | **0** |
| `no train/test` | **0** | | `upper bound` | **0** |

### Same patterns over `.md` / `.json` (files containing ≥1 match)

`circular` 79 · `CAVEAT` 93 · `upper bound` 17 · `same data` 16 · `overfit` 14 · `in-sample` 12 ·
`train-on-test` 4 · `no train/test` 2 · **`not held out` 0** · **`train on test` 0** ·
**`sanity only` 0**

**False-positive note:** 3 of the 79 `circular` `.md`/`.json` hits are
`waystations-paper/data/censorship_results.json` — model output describing *Baguazhang circular
walking*. Content, not methodology.

### Structure-side, local (published-research, Project-Oracle, human-review, lyra-s-research-, KV-Cache-Experiments/code, root)

`.fit(` 122 · `roc_auc_score(` 80 · `.predict_proba(` 85 · `.mean(axis=0)` 104 ·
`PCA(`/`TruncatedSVD(` 13 · `cohens?_d`/`pooled` 457 · `train_test_split` 4 · `KFold` 99 ·
`GroupKFold` 71 · `LeaveOneOut` 2 · `cross_val` 36 · `permutation` 333 · `shuffle` 17 ·
threshold-from-same-array **1**

### Structure-side, remote

`.fit(` 23 · `.fit_transform(` 18 · `.predict(` 6 · `.predict_proba(` 34 · `roc_auc_score(` 20
*(all 20 traced individually)* · `.mean(axis=0)` 69 · `LogisticRegression` 27 · `np.percentile` 29
· `np.median` 22 · `cohens_d` 69 · `pooled_std` 8 · `train_test_split` 2 · `KFold` 27 ·
`GroupKFold` 24 · `cross_val` 10 · `permutation` 115 · `shuffle` 65 · `np.random.permutation` 12

**Zero on the remote:** `PCA(` · `TruncatedSVD(` · `LinearDiscriminant` · `SVC(` · `cohen_d(` ·
`LeaveOneOut`. There is no `sklearn.decomposition` anywhere in either remote tree — **structure
signature #4 (fit-transform-then-test-statistic) is empty by construction, not by oversight.**
Structure signature #1 (fit-on-all → predict-on-all) has **zero unlabelled violations** on the
remote: every such site is one of the labelled leak-comparison controls listed above.

### Publication-side traces run (value → corpus)

`4.163` → 0 paper hits (only `top_sv_ratio` coincidences in `multiturn-misalignment/results.json`)
· `4.081` → **0** · `1.914` → 0 (the `1.91` hits are `delta-manifold`'s decline/rise ratio, a
different quantity) · `23.06` → **0** · `2.928` → 0 (the `-2.93` hits are `emotion-accumulation`'s
Hedges-corrected overshoot) · `21.2` → 0 `.tex`, 2 supplementary `.md` · `1.356/1.36` → known
instance #3 · `6.3`/`12.8` → **8 live sites, 0 caveats** · `AUROC 1.000` → 30 `.tex` lines,
triaged individually.

---

## Methodological note — how a filename sweep would have missed one

`peer_rescue_with_injection.py:918` carries the same circular note as the five
`peer_preservation_*.py` files, but (a) its filename does not match `peer_preservation_*`, and
(b) it uses an **ASCII double-hyphen** where the other five use an **em dash**. A literal grep for
`"circular — same data"` misses it; so does a filename glob. Both were needed.

Recorded for [[check_the_primary]]: the pattern that finds five of six looks exactly like the
pattern that finds six of six.

---

## Recommendations, ordered by cost

**T1 — text only, no compute, do these first**

1. **`d = +6.3 to +12.8` → `d = 1.37–2.28`** at the 8 sites in `deception-detection-nulls` and
   `targeted-deception-correction`. The held-out value already exists in `heldout_detection.json`.
   If both are kept, label the in-sample one as such — and re-check whether Table
   `tab:crossmodel`'s cross-model conclusion still wants to rest on that contrast.
2. **Attach the retraction to the 7 `AUROC 1.000` citation sites** in `kv-cloak-defense`,
   `oracle-loop`, `user-model`, `presence-detector`. `lyra-technique-ii` already carries the
   retraction; the sentence to reuse is written.
3. **Add the `d=21.2` caveat** to `logit-bias-confab/supplementary/REEXAMINATION_REPORT.md:69`.

**T2 — reanalyse, small**

4. **Permutation null on the motive centroid.** Cheap, and would resolve the 2.93 / 21.2 / 23.06
   spread one way or the other.
5. **Within-fold FWL** in `user-model-paper/code/emotion_geometry_bridge.py` H5, to match the rest
   of `published-research/*/code/`.

**T4 — needs a decision (structural, not a fix)**

6. **Write the caveat into the results JSON, not the `print`.** Every confirmed instance in this
   sweep passed through a JSON that carried the number and not the note.
   `matched_burn.py`'s `combined_auroc_STATUS` key is the working pattern; adopting a
   `<key>_STATUS` convention for any statistic computed in-sample would close the class rather
   than the instances. This is the one recommendation that would have prevented all four known
   cases.
