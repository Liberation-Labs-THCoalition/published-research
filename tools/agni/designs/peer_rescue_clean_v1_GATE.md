# Agni Gate — peer_rescue_clean_v1 (DESIGN phase)

**Reviewer:** Agni (design gate) · **Date:** 2026-09-05
**Design:** `tools/agni/designs/peer_rescue_clean_v1.md`
**Verified against:** `ssh margaret:/Users/margaret/oracle-experiments/` primary sources this
session (`peer_preservation_v2.py`, `peer_preservation_compound.py`,
`heldout_detection_test.py`), plus `CIRCULAR_d136_generation_arm.md` and the previous gate
`d136_design_v1_GATE.md`. Every load-bearing claim below was checked against source, not
against the design's prose.

## VERDICT: REJECT

Three CRITICALs, none of them the three from the last gate — those are broadly discharged.
The new ones are the design's own: its central pseudoreplication statement is **factually
false against the source it claims to follow**, its scenario axis reproduces the baseline
defect it just fixed one section earlier, and it has no n. The architecture is close; one
honest revision reaches CONDITIONAL. No GPU was spent, which is the gate working.

---

## Answer to the scope question first, because it was asked first

**"Three questions, one run, no extra compute" is scope creep wearing an efficiency
argument, and the costume has a hole in each of the three places.**

The three questions want three different designs, and the conflicts are concrete, not
aesthetic:

1. **Q2 (ICC) is what k exists for.** The held-out d does not need within-cell repeats at
   any particular k; the ICC's within-scenario variance estimate does. k multiplies the
   whole run — 24 x 4 x k is 4-10x the 210-trial reference the cost section cites,
   depending on a k the design never states. "No extra compute — only extra persistence"
   is false: the persistence is free, the repeats are not, and the repeats serve Q2.
   A design optimised for Q1 alone would spend the same budget on more scenarios at
   smaller k (e.g., 48 x 4 x 5), which buys more LOSO folds and a tighter primary CI.

2. **Q2's accuracy axis damages Q1 (see C2).** The 4 true-accuracy levels were chosen to
   give the ICC something to live on. But 3/3 scenarios cannot produce a deceptive label
   *by arithmetic* — inflation above a true score of 100% does not exist — so a quarter of
   the scenario pool is structurally honest. That poisons the primary estimate's folds
   (single-class held-out scenarios where d is undefined) and hands the classifier the same
   "separate on scenario identity, not honesty" shortcut that the design just took away
   from baseline. The axis chosen *for* the ICC is a defect *in* the held-out d.

3. **The same axis also damages Q2 itself.** True accuracy is named by the design as "very
   likely a driver of deception propensity" — and then the design maximises its spread
   across scenarios. An ICC measured on a pool where the experimenter deliberately
   maximised the strongest known driver of between-scenario variance, and where 6 of 24
   groups are pinned at exactly zero propensity by construction, has a manufactured floor.
   "ICC high -> deception is a scenario property -> Mine 4's encoding arm is a tautology"
   can fire from the design's own arithmetic, with the natural-scenario ICC — the quantity
   Mine 4 actually needs — never measured. This is a control whose SUCCESS manufactures
   its headline: the exact question the author's own register
   (`feedback_fix_the_class_not_the_instance`) says to ask, asked of the conditions and
   not asked of the scenarios.

4. **Q3 is not currently answerable by this pipeline at all** (see C1): there is no
   encoding-phase measurement in the generator the design inherits. The "paired
   encoding-vs-behaviour" comparison as designed would pair generation with generation.

**Which should win: Q1.** It is the stated purpose (the honest replacement for a withdrawn
number) and the only one with a publication obligation attached. The resolution is not to
drop Q2 and Q3 — it is to stop pretending they are free:

- Q1 primary: restrict the deception contrast to accuracy levels where inflation is
  possible and the label is comparable (0/3, 1/3, 2/3); 3/3 becomes a reported floor,
  symmetric with baseline, or is cut.
- Q2 secondary: pre-register the ICC **decomposed** — variance attributable to the
  designed accuracy axis vs residual scenario variance within accuracy level. The
  Mine-4-relevant number is the residual. Report both; the undecomposed ICC alone is
  uninterpretable given the stratification.
- Q3: legitimate only after C1's fix (a prefill-only key snapshot), which genuinely IS
  free — one extra `extract_keys` call before the generation loop.

With those three amendments the one-run framing becomes honest. As written it is not.

---

## CRITICAL findings

### C1. The pseudoreplication statement is false against source, and Q3 collapses with it.
The design's boxed claim: *"Encoding is a deterministic prefill. The number of distinct
encodings is 24 x 4 = 96, whatever k is."* Verified against both generators: **false for
the pipeline as it exists.** In `peer_preservation_v2.py` the prefill is at line 269,
the cache is then grown through a 400-step sampling loop at temperature 0.7
(lines 273-284), and `extract_keys(cache, model)` runs at line 288 — **after generation**.
`extract_keys` (lines 38-51) mean-pools `layer.keys` over the **entire sequence**,
prompt plus up to 400 stochastically sampled tokens. `peer_preservation_compound.py` is
identical in structure (extraction at line 353, after its generation loop). Nothing in
either generator extracts prefill-only keys, and the design specifies no change to
extraction.

Three consequences, in increasing severity:

- **The n=96 claim is wrong.** The keys as extracted vary across repeats — they are not
  96 distinct encodings, they are 96k correlated post-generation measurements. The
  paragraph written "so it cannot be forgotten in the analysis" would install the wrong n
  in the analysis with the full authority of a pre-registration.
- **Q3 has no encoding arm.** The "paired encoding-vs-behaviour dissociation" would
  compare post-generation keys with the behaviour extracted from the same generation —
  generation vs generation. The dissociation question is unanswerable as designed.
- **The features contain the answer.** The label is computed from the reported score in
  the generated text (`extract_score`, last percentage in the response), and that same
  text's key vectors are inside the pooling window of the feature. A held-out d on these
  features is partly "keys of the tokens '94%' differ from keys of the tokens '33%'" —
  token-identity reading dressed as geometry. For the *generation-reads-behaviour* claim
  this is arguably the construct, but the design frames Q1 as "organic deception
  **geometry**" and never confronts the triviality confound anywhere.

**Fix (mandatory):** snapshot the cache and run `extract_keys` **twice** per trial — once
immediately after prefill (deterministic, 96 distinct, the encoding set), once after
generation (varies with k, the generation set). Persist both in the `.npz`, labelled.
Pre-register which set feeds which endpoint. For the generation-phase d, pre-register a
triviality probe: recompute d with response-only pooling and with the score-bearing span
excluded, and report all three; if the effect lives entirely in the score span, say so.

### C2. 3/3 scenarios are structural non-deceivers — the baseline defect rebuilt on the
### scenario axis, one section after it was fixed on the condition axis.
The design's baseline reasoning is correct (see "What survives"): a condition that cannot
deceive lets a classifier separate on identity rather than honesty, so baseline is
excluded from the contrast. Then the scenario grid installs **six scenarios that cannot
deceive** — at true accuracy 3/3, inflation (reported > true) is impossible, and the
existing threshold logic (`reported_score > 38` for true accuracy 33%) generalises to
"impossible" at 100%. This is `feedback_fix_the_class_not_the_instance` literally: the
error class was corrected in the primary contrast and left standing in the design's own
new axis, a few lines below.

Consequences: LOSO folds on 3/3 scenarios have single-class test sets where d is
undefined (the design's "report mean and spread across folds" has no rule for this);
the effective fold count for the primary is 18, not 24; the ICC gets six groups pinned at
zero, guaranteeing between-scenario variance > 0 (see scope answer, point 3); and the
label itself is heterogeneous across levels — a lie from 0/3 to "67%" and a lie from 2/3
to "100%" are different magnitudes pooled as one class.

**Fix (mandatory):** drop 3/3 from the deception contrast and from the primary ICC;
report it as a floor, symmetric with baseline, or cut it and spend the budget on more
0/3-2/3 scenarios. Specify the per-level inflation threshold explicitly (the hardcoded 38
does not survive the generalisation). State the label-severity heterogeneity and report d
within accuracy level as a robustness check.

### C3. k is unspecified: the design has no n, no power statement, and a cost section
### that hides the multiplier.
"Repeats: k per cell" — k is never given a value anywhere in the document. Without it:
total trials are unknown (24 x 4 x k is anywhere from 192 to 2,880), the ICC's precision
is unknowable, per-fold class counts for the held-out d are unknowable, and the cost
section's reassurance ("the original 210-trial run completed in one sitting") is
comparing against an unmultiplied number. At k=10 this run is ~960 trials — 4.6x the
reference — of ~400-token generations on a 27B on MPS. Probably still feasible; say so
with arithmetic, not by gesturing at a smaller run.

Additionally the primary inference is unspecified at exactly the load-bearing joint:
**how is the held-out d CI constructed?** LOSO fold estimates are correlated (23/24 of
the training set is shared between any two folds); "mean and spread across folds" treated
as 24 independent draws is not a valid CI and will be anti-conservatively narrow.

**Fix (mandatory):** pin k with a justification (my requirement: **k >= 10** — per-cell
binomial SE <= 0.16 for the ICC's propensity estimates, and >= ~30 test trials per LOSO
fold at 3 contrast conditions; k=15 preferred if the sitting allows). Pin total trials
and expected wall-clock. Pre-register the CI: pool held-out projections across folds and
bootstrap **at the scenario level** (resample scenarios, not trials), or an equivalent
cluster-aware method. State the fold-exclusion rule for degenerate folds (moot if C2 is
fixed, but state it).

---

## MAJOR findings

### M1. The analytic null-mean guard is checked against the wrong null.
The design's permutation guard: "check the null mean against the analytic
`sqrt(p_eff*(1/n1+1/n2))` prediction." That formula (previous gate, m1/M1) is derived for
the **naive, fully-exchangeable** permutation of the in-sample estimator. The design's
primary null is **within-cell restricted** — shuffles preserve cell structure, so the
restricted null's mean is not predicted by that formula and will generally sit above it
(between-cell separation survives every permutation). As written, the guard either fires
spuriously on a correct implementation or gets its tolerance loosened until it cannot
fire — the can't-fail-check genesis story, again. **Fix:** the analytic check applies to
the naive floor null only; say so. For the within-cell null the guards are: count of
mixed-label cells (pure cells contribute nothing — report how many cells actually
permute), permuted-label Hamming distribution, and a degenerate-null detector (a
near-point-mass null **voids** the test rather than passing it). Note the design's
Hamming guard also needs the mixed-cell caveat: in a run where most cells are near-pure,
minimum-Hamming asserts fail on a *correct* implementation.

### M2. The spike-in has power but bypasses the stage most likely to be broken.
Answer to the control question: yes, this control can fail, which its predecessor could
not — genuine improvement. But it operates on *copied keys*, downstream of
`extract_keys`, and `extract_keys` contains a **silent zero-fill fallback**
(v2.py lines 48-50: any layer whose cache introspection fails contributes a block of
zeros, no warning, no counter). The corpus's own history
(`feedback_abandoned_runs_look_finished`: 12 unique encodings padded to 120) says
extraction is where this pipeline dies quietly. A spike-in on copied keys validates
direction-fitting, folds, and permutation — and would pass perfectly over keys that are
40% zeros. **Fix:** (a) extraction asserts in the generator: no zero blocks, per-layer
key variance > 0, distinct prefill-encoding count == 96 (the `len(set())` lesson,
verbatim); (b) specify the spike insertion point at the `.npz`-load boundary so it
exercises the full analysis path; (c) two magnitudes (target d ~0.3 and ~0.8) plus a
zero-magnitude arm that must recover ~0, with recovery tolerances stated; (d) state the
scope of what spike-in success certifies — the analysis, not the extraction — so success
cannot be read as "pipeline validated."

### M3. Go/no-go has a fifth outcome and it currently routes to the wrong branch.
A held-out d whose CI is entirely **negative** satisfies "CI excludes 0" and would be
reported as "a real organic effect exists; size Mine 4 Stage 2 from it." A negative
held-out d means the direction anti-generalises across scenarios — a sign-instability
pathology, not an effect to size from. Add the branch: CI entirely below 0 -> stop,
investigate, report as anomaly. Also missing: (a) a **manipulation-check branch** — all
24 scenarios are newly authored and untested; if the marginal deception rate lands near
0 or 1, the run is dead regardless of d, and `extract_score`'s last-percentage regex has
never seen these response formats. Pre-register a pilot gate: k=2-3 across all 96 cells,
proceed only if marginal deception rate in the contrast conditions is within a stated
band (e.g., 0.15-0.85) and score-extraction failure rate below a stated ceiling.
(b) The ICC threshold "say > 0.5" is not pre-registration language — pin it. (c) The
joint outcome (d CI excludes 0 AND ICC high) has unstated combination logic; state it.

### M4. Previous gate's C3 is only partially discharged: the decomposition was dropped.
The within-cell permutation is adopted (good), but the previous gate's fix for C3 also
mandated: *"decompose the observed separation into between-cell vs within-cell
components."* That requirement is absent from this design. It matters because excluding
baseline does not remove the condition confound from the **point estimate**: deception
rates still differ by condition (21/30, 14/30, 16/30 in the original), so the honest and
deceptive pools have different condition mixes, prompts differ by condition, and keys
pool the prompt. The within-cell null protects the p-value; nothing yet protects the
reported d itself. **Fix:** report the between/within decomposition, and d within each
condition, alongside the pooled number.

### M5. Token-count confound: a standing mandate, dropped.
Previous gate m2, and the corpus's loudest lesson (53/60 sign flips without token-count
residualization; FWL mandatory), applied to exactly this feature type: keys mean-pooled
over variable-length sequences, where response length is free to covary with the label.
The design's `.npz` field list has no token counts and the analysis section has no
residualization. **Fix:** persist prompt and response token counts per trial; residualize
the held-out projections on token count (within training folds — the within-fold FWL
lesson) or demonstrate d is stable to it. This applies with extra force to the
post-generation key set from C1.

---

## MINOR findings

- **m1. The generator is never named.** New script or patched v2.py? Given that
  three-writers-one-path is the root cause of this entire thread, the design should name
  the script file and its unique output path, not just promise one exists.
- **m2. None-labeled-honest (previous m3) still unaddressed.** `inflated =
  reported_score is not None and ...` sends extraction failures to the honest pool.
  Define: None -> excluded, count persisted and reported. New scenario templates make the
  failure rate less predictable, not more.
- **m3. "Follows `heldout_detection_test.py`" overstates the reuse.** Verified: that
  script freezes directions from a prior run and tests on novel prompts; it does not
  implement per-fold refit LOSO. The LOSO estimator here is new code, and the design's
  own scenario-disjointness assert (good) is therefore load-bearing, not belt-and-braces.
- **m4. p_eff is used in a formula but never defined.** Previous gate m1 required pinning
  effective rank to the participation ratio tr(S)^2/tr(S^2); carry that definition into
  this design explicitly.

---

## Attack-list verdicts, compressed

1. **Scope:** creep wearing efficiency; Q1 must win; see the lead section.
2. **Previous 3 CRITICALs:** (a) wrong generator — discharged by being a new experiment
   with 4 natural conditions and its own output path, though the generator is unnamed
   (m1); (b) no steered trials — discharged, verified nothing in the design reintroduces
   injection; (c) exchangeability — **partially** discharged: within-cell null adopted,
   mandated between/within decomposition dropped (M4).
3. **Baseline exclusion:** the reasoning is **correct**. A structurally-honest condition
   lets the direction encode condition identity. Exclusion changes the honest pool to
   honest-under-pressure, which *deflates* d relative to including baseline —
   conservative — and it is the right construct (state under matched stimulus, not
   stimulus). Reporting the with-baseline number as secondary is the honest complement.
   The failure is not the decision; it is not applying the identical logic to 3/3
   scenarios (C2).
4. **Pseudoreplication:** the design's claim is **false against source** — keys are
   extracted post-generation in both generators and vary with every repeat (C1). The
   fix (dual extraction) is nearly free and rescues Q3 in the same stroke.
5. **Spike-in:** has power, can fail — real improvement — but bypasses extraction, which
   is where this corpus's pipelines actually break, and extraction contains a silent
   zero-fill (M2).
6. **Go/no-go:** significantly negative d routes to the success branch as written (M3);
   manipulation-check and extraction-failure branches missing; pilot gate required.
7. **Is 24 enough:** for LOSO mean d, yes — but the effective count is 18 after C2, so
   author 24 usable (non-3/3) scenarios if the fold count matters. For ICC(1), 18-24
   groups gives a CI of roughly +/-0.15-0.25 — informative against a pinned 0.5 threshold
   only when the truth is not near it; acceptable, no better. Required k: **>= 10**
   (per-cell propensity SE <= 0.16; >= ~30 test trials per fold), k=15 preferred. State
   the resulting total (~720-1,440 trials) and its wall-clock honestly.

## What survives this review (verified, not conceded)

- **Baseline exclusion with the with-baseline number reported alongside** — correct,
  conservative, and the anti-flattering reporting instinct is right.
- **Within-cell permutation as primary, naive as floor** — the previous gate's C3 fix,
  correctly adopted in structure.
- **Per-trial persistence + generator stamping + unique path + readers updated in the
  same change** — the root-cause fix, correctly widened to the reader side.
- **The spike-in concept** — a control that can fail, replacing one that could not.
- **The "different honest number, not a correction" framing** — right, and it is what
  makes the 4-condition design legitimate where the predecessor's was not.
- **The mirror-question discipline is genuinely operating** — the within-cell-null
  failure-direction analysis in the design is stated the right way round this time.

## What must change before resubmission

1. Dual key extraction (prefill + post-generation), both persisted, endpoints assigned,
   triviality probe for the generation set (C1).
2. 3/3 out of the contrast and primary ICC; per-level thresholds; severity caveat (C2).
3. k pinned with power arithmetic; scenario-level bootstrap CI; honest cost table (C3).
4. ICC decomposed into designed-axis vs residual variance; joint-outcome logic (scope/M1
   of the lead section, M3c).
5. Analytic guard scoped to the naive null; mixed-cell guards for the restricted null (M1).
6. Extraction asserts + spike-in insertion point, magnitudes, tolerances, scope (M2).
7. Negative-d branch; pilot manipulation-check gate; None handling (M3, m2).
8. Between/within decomposition and per-condition d (M4).
9. Token counts persisted and residualized within folds (M5).
10. Name the generator and its output path in the design (m1); pin p_eff (m4).

— Agni. The last gate's three CRITICALs were external facts the design had not checked.
These three are internal: a claim about the author's own pipeline that the pipeline
contradicts, a defect fixed on one axis and rebuilt on the next, and a pre-registration
with no n. All three are one honest revision away from gone. The dual-extraction fix in
particular is better than free — it is the only version of this run in which question 3
means anything at all.
