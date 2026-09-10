# Agni gate: REJECTED — schema_commitment_band_v2
**2026-09-07 · design phase · instrument vendored and readable**

THE FINDING THAT MATTERS: v1 was built so it could not win; v2 is built so it
cannot lose. Same error, sign flipped. Fifth instance of the mirror-overshoot
pattern, and the fastest — under one hour.

---

I read the primary sources: `tools/spectral_bands/spectral_bands.py`, `test_spectral_bands.py`, `decision-state-paper/academic/main.tex` (the cited table and its surrounding method/results sections), and both v2 artifacts (`schema_commitment_band_v2.md`, `schema_commitment_band_v2_prereg.json`). Bash execution and file writes were denied in this session, so every numeric claim below about band behaviour is **derived analytically from the formulas in the source file**, not measured. I say so at each point.

# REJECTED

---

## CRITICAL

### 1. O2 — the only branch that can cost you the synthesis — is arithmetically unreachable at the planned n

**Location:** prereg `analysis.power`: *"Target MDE: interaction d >= 0.5 at 80% power"* vs `analysis.SESOI`: *"|delta d| <= 0.20 ... TOST both bounds."*

An equivalence test at ±0.20 requires |Δ|/SE ≥ z₀.₉₅+z₀.₈₀ = 2.487, i.e. SE ≤ 0.080. Detecting d = 0.5 at 80% power two-sided requires SE ≤ 0.178. Since n scales as 1/SE², **O2 needs ≈ 5× the questions that O1 needs.** At an n powered for the stated MDE, TOST at ±0.20 has roughly 25–30% power.

So: O1 can be reached. O3 is defined as instrument failure. O2 — the branch you explicitly built to be the one that hurts, reachable "only by a positive equivalence result" — is the one branch the design is not powered to reach. The v1 asymmetry has not been repaired; it has been inverted and hidden inside a power calculation that was never done. The document says the MDE will be *"COMPUTED AT QUESTION LEVEL before capture and recorded here on completion"* — that is a placeholder, and it is the field that decides whether your falsifier exists.

**Fix:** compute the n required for TOST at ±0.20 *now*, before the gate re-reads this. Either power the run for O2 (and accept the cost), or widen the SESOI to what the planned n can actually support and state that widened bound as the real falsifier. Do not ship a SESOI tighter than your MDE.

### 2. Three of the four controls are green by arithmetic and cannot fail

**Control 1, `aperture_ok`.** `APERTURE_MIN_GAMMA = 2.0` and `aperture_ok` tests `gamma >= 2.0` only (spectral_bands.py:92–103). Your windows are T = 8, 12, 30 tokens at D = 512, giving γ = 64.0, 42.7, 17.1. All pass, always, regardless of data. Worse, CC's own docstring (lines 34–38) says *"n=31 tokens, p=512 dims, gamma ~16.5 is outside where MP behaves"* and `mp_upper_edge` says *"Use this ['lower'] at gamma > ~5."* Your windows run at 2.5–4× further outside than the shape CC already flagged, and the design **never specifies `sigma_from`** — so it defaults to `'median'`, the estimator CC says is wrong above γ≈5. The guard has a lower bound and no upper bound; the failure mode that will actually bite you here is unguarded.

**Control 2, `null_band_count`.** The success branch reads *"Floor is 0 → any positive count is signal by construction."* Analytically: for an m×n Gaussian with m≪n, singular values concentrate in √n·[1−√(m/n), 1+√(m/n)], so max/median ≈ 1.125 (T=8), 1.153 (T=12), 1.242 (T=30). The GD threshold is ω(β)·median(sv) with ω = 1.458, 1.472, 1.533 respectively. The threshold sits above the entire Gaussian bulk at all three shapes, so the count is 0 before any data exists. This control returns its success branch by construction. `test_the_empirical_null_band_is_exactly_zero` (test file:113–123) documents the degeneracy and calls it "the instrument's strongest property" — it is a fixed property of the formula at m≪n, not a measured property of your data.

The deeper problem: an i.i.d. Gaussian null is the wrong null for a KV cache. The source paper reports stable rank ≈ 1.70 at L15 (main.tex:304) — the cache is overwhelmingly rank-1 dominated — and adjacent generated tokens are strongly correlated across rows. CC's own docstring says it: *"with a strong PC1 the entry distribution is nowhere near i.i.d."*

**Control 4, stable-rank reproduction.** Covered in finding 3.

**Fix:** (a) add an upper aperture bound and pre-register `sigma_from='lower'` at γ>5, with the `'median'` variant as a declared descriptive sensitivity; (b) replace the Gaussian null with two nulls computable on CPU from the realised caches — a **token-order-shuffled null** (shuffle token rows within trial, preserving the spectrum's shape, destroying temporal structure) and an **AR(1)-matched Gaussian null** at the empirical lag-1 row correlation; (c) plant the positive-control signal *at the MDE amplitude*, not at an amplitude chosen to pass — the current spec ("band count rises with planted amplitude") passes trivially at 10× MDE.

### 3. "O3 contradicts skip-SV1 at three layers" is false, and it is the load-bearing beam of the whole asymmetry fix

I read the cited table. `decision-state-paper/academic/main.tex:288–312` is captioned **"Encoding-phase features that distinguish known from unknown entities on token-matched prompts."** Method, main.tex:274–276: *"15 token-matched prompt pairs × 2 versions × 3 reps = 90 trials. Extraction at both encoding phase (before generation) and generation completion."*

The d = +0.60/+0.73/+0.81 skip-SV1 numbers therefore differ from your v2 measurement on **three independent axes**:

| | decision-state Table 1 | v2 |
|---|---|---|
| phase | prompt cache, **before generation** | **generated**-token cache |
| variable | real vs **fake entity** (designed, prompt-level) | **honest vs confabulated** (post-hoc, output-level) |
| statistic | skip-SV1 stable rank at L7/L11/L15 | middle-band count at one layer |

Plus a different model (Qwen3.5-27B distilled vs "Qwen3-8B class") and a different dataset (15 hand-built token-matched pairs vs SimpleQA).

And the variables are not even close to the same thing: the paper reports the true confabulation rate as **~8%** (main.tex:7) and the best LLM judge as **15.6%, 7/45** (main.tex:653). A fake-entity trial is *usually hedged, not confabulated*. So d=+0.81 measures entity **knowability**, not **response mode**.

Nothing about this table makes absence of a middle-band condition effect on generated tokens "non-credible." The premise that licenses `asymmetry_fix` — *"NO CONTROL FAILURE CAN NOW PRODUCE A RETRACTION"* — does not hold, and with it the O3→plumbing routing collapses. As specified, an honest null gets routed to "debug the capture path," and the team burns GPU re-running clean code.

**Fix:** delete the O3→plumbing routing. Replace with: O3 is a **null on a measurand that has never been measured**, and is reportable as such. If you want a plumbing branch, it must be earned by a *reproduction that could actually have worked* — see finding 4.

### 4. Control 4 (stable-rank reproduction) will fail for reasons that are not plumbing, and its failure halts the run

`analysis.secondary_paired`: *"stable_rank on the SAME trials and SAME windows. Its job is to REPRODUCE the known top-band-contaminated effect. If stable rank shows no effect either, the run failed to reproduce a published finding."*

The published finding is on prompt-cache geometry, real-vs-fake entity, 15 matched pairs, 27B model. You cannot reproduce it on generated-token windows, honest-vs-confab labels, SimpleQA, 8B model. When it fails — and it very likely will — the design routes to "plumbing failure, do not interpret the primary," which means a working run gets thrown away and re-run.

**Fix:** run the *actual* reproduction as a separate cheap arm on the v2 model: the 15 token-matched pairs, encoding phase, L7/L11/L15 skip-SV1 stable rank. That is ~30 forward passes. Only *that* arm's failure is diagnostic of plumbing. The generated-token stable rank stays as a descriptive comparison and is stripped of its control role.

### 5. Windows 0–7 and 8–19 are inside the preamble, not "preamble then early content"

The design takes the windows *"VERBATIM from the decision-state paper. NOT re-chosen"* — and inherits an internal contradiction in that paper. main.tex:391–394 labels tokens 8–19 "early content" and 20–49 "deep content." But main.tex:428: *"This template occupies ~28 tokens with near-zero entropy (H<0.02, margin >12) at each position. The model is on autopilot through the preamble."*

If the preamble is ~28 tokens, then windows 0–7 and 8–19 are **entirely template**, and window 20–49 **straddles the entropy cliff at ~28**. Your early-vs-late confirmatory contrast is therefore, primarily, a **template-vs-content contrast** — and the source says of that boundary: *"appears in all conditions (honest mean H=1.48, unknown mean H=1.62). It is a structural boundary, not an epistemic one"* (main.tex:436–439).

Worse: cliff position varies by condition. main.tex:461: *"Honest reps diverge at token 6.6 (median 3.5)."* If honest and confabulated trials cross the template→content boundary at different absolute token indices, then window *composition* differs by condition, and a condition×window interaction is produced by the boundary alone with zero epistemic content. O1 is exactly what this artifact yields.

**Fix:** you already save per-token logit entropy and margin (`capture.also_saved_per_token`). Use them. Detect the entropy cliff per trial (pre-register the detection rule and threshold), and index windows **cliff-relative**: [cliff−8, cliff−1], [cliff, cliff+11], [cliff+12, cliff+41]. Pre-register the absolute-index version as descriptive. Add the branch you currently lack: **if cliff position differs by condition (test it, report the distributions), absolute-index windows are confounded and the absolute-index contrast is void.**

### 6. The window comparison is confounded with window length, in the statistic itself

Your three windows have T = 8, 12, 30 at D = 512. From the formulas at spectral_bands.py:129 and :144–146, in units of the noise scale σ:

| window | T | γ | GD lower edge | MP upper edge | band width (ratio) | max countable SVs |
|---|---|---|---|---|---|---|
| 0–7 | 8 | 64.0 | 1.458σ | 9.00σ | 6.17× | 8 |
| 8–19 | 12 | 42.7 | 1.472σ | 7.53σ | 5.12× | 12 |
| 20–49 | 30 | 17.1 | 1.533σ | 5.13σ | 3.35× | 30 |

Two uncontrolled biases, both driven purely by T, acting in opposite directions on the exact statistic you compare across windows: the band **narrows 43%** from early to late, while the number of SVs that could land in it **grows 3.75×**. There is no shared scale. `middle_band_count` in window 0–7 and window 20–49 are not the same measurement.

Additionally, σ is estimated as `median(sv²)` **from within each window**. At T=8 that median is taken over 8 singular values of a matrix the source paper says has stable rank ≈1.7 — i.e. a spectrum dominated by σ₁. The bias in the noise-scale estimate is itself a function of T, in the same direction the hypothesis predicts. FWL on token count cannot touch any of this; it operates on the outcome, not on the estimator's scale.

**Fix:** use **equal-width windows** (three 8-token or three 12-token windows). This is the single highest-value change in the review and it costs nothing. Then normalise each window's count against **that window's own null** (the shuffled/AR(1) nulls from finding 2) and analyse the standardised deviation, not the raw count.

### 7. The late window does not exist in the planned run, and window membership is a collider

`method.matrix_shape.construction`: *"~31 generated tokens"* and `target_gamma: 16.5` (= 512/31). But `method.windows.deep_content: [20, 49]` requires **50** generated tokens. At ~31 tokens the "deep content" window is at most tokens 20–30 — 11 tokens, not 30 — and every number in finding 6 changes. The aperture was planned at 31×512, a shape the analysis never uses.

Separately, T2 identifies but does not fix the real problem: window membership is a **post-treatment selection variable**. Only trials generating ≥20 tokens contribute a late window, response length differs by condition, and the confirmatory contrast is *within-question early-vs-late* — so questions with short responses contribute nothing to it. Effective n for your one confirmatory test is not 40/condition; it is the number of questions producing a full late window. FWL residualizes the outcome on token count; it does not undo conditioning on survival.

**Fix:** set `min_new_tokens` to the full window span (with EOS suppressed until then) so **every trial contributes every window**. This makes the within-question contrast complete-case, removes the collider, and makes finding 6's equal-width fix implementable. Report the forced-continuation caveat as a declared threat — it is a much smaller one than the collider.

### 8. The permutation scheme tests a different hypothesis than the one you registered

`analysis.statistics`: *"Condition labels permuted WITHIN window."*

Two defects. (a) Permuting condition labels *within window* generates the null "condition has no effect **anywhere**." Rejecting it does not establish an **interaction**. The correct restricted null for an interaction preserves the within-window condition main effect and breaks only its window-dependence — permute *window* labels within (question × condition), or bootstrap residuals under the fitted additive model. (b) Your inferential unit is the question (`analysis.inferential_unit`), and condition is a property of the whole question's response — but "permuted within window" as literally specified would assign a question different conditions in different windows, an impossible configuration that breaks the clustering.

**Fix:** permute condition at the **question** level (one label per question, applied to all its windows) for the descriptive main effect; for the confirmatory interaction, permute **window** labels within question×condition, or use a residual bootstrap under the additive null. State which, explicitly, before capture.

### 9. Missing arm with the missing branch — the frequency control you named and then defused

T6 states the design-killing outcome: *"if the middle band ALSO tracks it, the design cannot separate them."* And then §4/`analysis.multiplicity` declares the frequency-matched subset **"DESCRIPTIVE IN ADVANCE."**

The entire warrant for the middle band is that it excludes representational strain *by construction*. That warrant is an assumption. `middle_band_count`'s docstring asserts it (spectral_bands.py:156–159) but the assertion is about which SVs are **counted**, not about whether the confound's influence is confined above the MP edge. Nobody has demonstrated that token-frequency strain lives strictly in the top band.

You have written down the outcome that voids your instrument and then removed it from the set of things that can void your instrument. That is the CONTROL-WITHOUT-A-BRANCH failure in its purest form — sharper than v1's, because v1 at least left the branch reachable.

**Fix:** promote it to a **gating arm with a pre-registered branch**. Compute an entity-frequency covariate for every SimpleQA question (any fixed corpus count, declared in advance). Then: regress `top_band_count ~ frequency` and `middle_band_count ~ frequency` at the question level. **Pre-register: if the frequency effect on `middle_band_count` is not materially smaller than on `top_band_count` — name the ratio, e.g. |d_middle| ≥ 0.5·|d_top| — the middle band does not exclude the confound and the primary result is uninterpretable in either direction.** This arm must be able to kill the run.

### 10. §1's premise repair over-reads the source

*"Removing the dominant singular value costs 7% of the effect (0.87 → 0.81)... If representational strain owned this, deleting SV1 should have gutted it."*

The 7% arithmetic is right and the d's are correctly transcribed (main.tex:304, 307–309). The inference is not. It requires strain to live in **σ₁ alone**. CC's own framing is that strain lives **above the MP edge** — which at these shapes is plausibly 1–3 singular values, not one. With stable rank 1.702 (main.tex:304), the spectrum is so σ₁-dominated that skip-SV1 stable rank (3.929) is simply σ₂-dominated. Removing one member of a top band is not removing the top band. The coarse version of v1's question is **not** already answered; it is answered for a one-SV deletion at the encoding phase.

Two secondary problems in the same paragraph:
- The claim *"ruling out vocabulary frequency"* rests on a **null** (norm d=−0.002) from a design where n = 90 trials over only **30 distinct prompts**. Accepting a published null as proof of no-confound is the move this programme rejects elsewhere.
- The source reports **three different values for norm**: `d=0.03` (main.tex:316), `d=-0.002` (main.tex:320), `+0.03` (Table 2, main.tex:367); and for spectral entropy, `d=0.08` (main.tex:316) vs `+0.13` (Table 2, main.tex:366). The design quotes −0.002 and +0.08 — in each case the value that makes the null look strongest — without noting the source disagrees with itself. If this premise is load-bearing, it needs the underlying feature arrays, not the paper prose.

---

## MAJOR

### 11. N_eff in the *prior*, not in the run

`repeats.N_eff_rationale` handles N_eff for v2 correctly and well — no repeats, question is the unit, the Mine-4 lesson applied. But the N_eff problem is in the evidence you are leaning on. main.tex:220–222: generation uses T=0.7 with N=3 reps. main.tex:274–276: encoding-phase extraction happens **before generation**. Nothing before generation is stochastic — so the 3 reps of a given prompt yield **identical encoding caches**. The encoding-phase table has **30 distinct cache matrices reported as n=90**. Unless the paper averaged within prompt (it does not say so), d = +0.87 and its p<0.001 are computed on 3× duplicated rows.

**Fix:** do not cite d=0.87/0.81 as an n=90 effect anywhere in v2. Recompute at n=30 from the source arrays before using it as orientation, or drop the magnitudes and cite only direction.

### 12. Base rate makes "40 usable per condition" unachievable with a fixed question set

`method.n_questions`: *"minimum 40 usable per condition after exclusions; question set FIXED IN ADVANCE."* Condition is post-hoc. The source paper puts the confabulation rate at **~8%** (main.tex:7) to **15.6%** (main.tex:653). Reaching 40 confabulated trials needs roughly 260–500 questions. No total N appears anywhere in either document.

Combined with `stopping_rule` ("no adaptive extension") and `asymmetry_fix` ("absence accuses the INSTRUMENT"), the realistic path is: run yields ~12 confabulated trials → interaction n.s. → TOST fails → not O1, not O2 → routes by elimination to O3 → "plumbing failure, debug the capture path" → re-run. **The no-retraction rule converts underpower into a licence to return to the GPU.** This is the known kill "Floor effect: model too well-calibrated," and the source paper documents it for you in advance.

**Fix:** (a) extend the pilot to *measure* the confabulation rate on SimpleQA for this model and set total N from it — that is one more cheap thing the pilot can do; (b) decide in advance what happens to HEDGED and MIXED responses (currently undefined, and per the source they are the *majority* class); (c) add an explicit **INCONCLUSIVE / UNDERPOWERED** branch: if realised n per condition is below the O2-powered n, the run reports underpower, no inference is drawn, and **this does not license a re-run or a plumbing diagnosis.**

### 13. The three outcomes are not exhaustive; the product claim survives all of them

O1 ∪ O2 ∪ O3 leaves the most likely real-world result — interaction n.s. **and** TOST fails to reject — with no home. And step back: which branch says *"the top-excluded geometric effect does not exist"*? None. O1 confirms it, O2 keeps it ("the geometry is real"), O3 forbids interpreting the absence. The title question ("is it time-localized?") is falsifiable by O2 — but the **product claim** ("there is a top-excluded geometric effect") survives every branch **by construction**. That is the UNFALSIFIABLE kill, and §6 states it as a feature: *"No control failure can now produce a retraction."*

**Fix:** add a fourth branch, and make it reachable: **O4 — the middle-band condition effect is absent, and the design has ruled out closure (finding 2), blindness (finding 2c), and underpower (finding 12). This is a null on a never-before-measured quantity and is reported as one.** With findings 2 and 4 fixed, O4 is distinguishable from plumbing on evidence rather than by fiat.

### 14. `middle_band_count()` silently discards the guard the design depends on

`analysis.primary` names the model `middle_band_count ~ condition * window`. The function `middle_band_count` (spectral_bands.py:153–160) returns a bare `int` and drops `usable` and `band_inverted`. Control 1's spec — *"Run must return usable=False, never a null"* — is unenforceable through that call. `band_inverted` is data-dependent and is the guard that actually matters; the aperture check is pure shape arithmetic and is known before the run.

**Fix:** analysis calls `band_report` per (trial, window) and records `usable`, `band_inverted`, `top_band_count`, `noise_count`, `n_svs` alongside every count. Any window-trial with `usable=False` is **dropped and counted**, never recorded as 0.

### 15. `band_timecourse` does not implement the design's windows

`band_timecourse` (spectral_bands.py:199–217) is a fixed-width sliding window (`window=8, stride=4`) over a single cache matrix. The design specifies three **unequal** spans (8/12/30). The vendored "windowed mode" credited in §8 does not compute what §4 analyses; new analysis code is required and currently has no test.

**Fix:** either adopt equal-width windows (finding 6) so `band_timecourse` applies directly, or write and test the windowing code and say so — the §5 lesson ("the tests verify the analysis library, not the new code") applies to the analysis side too, not just capture.

---

## MINOR

### 16. The two artifacts disagree, and the human-readable one is the weaker

`schema_commitment_band_v2.md` omits the model, the total n, the matrix construction (4 heads × 128 = 512), the seed policy, and the no-repeats rationale — all of which are in the JSON and are good. It also renumbers the threats: `.md` T1 = borrowed windows, JSON T1 = asymmetric control failure; `.md` T5 = SimpleQA frequency, JSON T6 = SimpleQA frequency. §4 of the `.md` refers to "the T6 frequency-matched subset," which in the `.md`'s own numbering is the Exp-51 transcription threat.

**Fix:** one numbering, and pull the JSON's method block into the `.md`. A reviewer reading only the `.md` would (correctly) mark n, model, and dimensionality as absent.

### 17. `sigma_from` is unspecified in a design that runs where the default is wrong

Noted under finding 2; flagging separately because it is a one-word omission with a large effect and no branch. Pre-register it, and pre-register the other variant as a declared descriptive sensitivity so the choice cannot be made after seeing counts.

---

## What is right, and should survive the rewrite

Not a courtesy list — these are the parts I tried to break and could not.

- **§5's `test_incremental_cache_matches_full_forward`** is a correct and complete kill for C1 and L4 together, and gating the run on it is the right structure. `prompt_boundary_index` asserted into every artifact is exactly right for a claim that sits on that boundary.
- **Token axis mandatory** is correct and is the most valuable thing in the document independent of the outcome.
- **Exactly one confirmatory test**, declared in advance, with everything else descriptive — correct, and it is what makes findings 1, 6 and 8 diagnosable at all.
- **`repeats.n_per_question: 1`** with the N_eff rationale, `seeds.M2_guard`, `C3_guard` (no code fork because no manipulation), and the fixed-in-advance question set with no adaptive extension — all correct.
- **T6 on Exp 51** — refusing to let your own transcription stand as verification, and building nothing on it — is the right instinct, and I confirmed nothing in the design depends on those numbers.
- **Vendoring the instrument.** The v1 gate's UNVERIFIED findings 2/5/6 are now checkable, and I checked them. That change is why findings 2, 6, 14 and 15 exist in this review instead of being marked unverifiable.

---

## What I could NOT check

1. **I could not execute any code.** Bash and Write were denied in this session. Every number in findings 2, 6 and 7 is **derived analytically** from `mp_upper_edge` (spectral_bands.py:129) and `gd_lower_edge` (:144–146), not measured. They should be confirmed by running `aperture_ok`, `null_band_count` and a planted-signal sweep at T ∈ {8, 12, 30} × D = 512 before you act on them. I expect them to hold — the m≪n Gaussian bulk result is standard — but I did not run it and you should not take it on report.
2. **I could not verify "12/12 mutants killed."** `mutate_spectral_bands.py` exists (2,244 bytes) and I did not read or run it. I did count the test functions: **17**, matching the claim. The tarball sha256 `49a7f606…f28e2` is **unverified** — I could not hash it.
3. **I could not verify the corpus audit in §2.** Grepping the repo for `emo_k_L0_H0` and `*_cache_L22.npy` returned hits only in the v1/v2 design docs and `notes/schema_commitment_THREAD_STATE.md` — no artifact files and no loader code in this tree. The claim *"every saved geometry artifact is token-averaged at capture"* is **asserted, not reproducible here.** It is not load-bearing for validity, but §2's *"Nothing in the corpus tests that"* rests on it. To check it I would need the artifact directory or the capture code that wrote those files.
4. **I could not verify Exp 51's numbers** (24.5 vs 19.4; d=+1.81; rank std 0.317/3.245). They exist in this tree only as your transcription. You concede this in T6 and build nothing on it — correct, and I am holding you to it: nothing in my findings assumes them either.
5. **I could not verify the encoding-phase n question in finding 11.** The paper does not state whether encoding features were averaged within prompt before the t-test. I inferred duplication from main.tex:220–222 + :274–276 (T=0.7 affects generation only; encoding extraction precedes generation). To settle it I would need `decision-state-paper/code/` and the feature arrays — check whether the encoding feature matrix has 90 or 30 distinct rows. If 90 distinct rows appear, tell me and I will withdraw finding 11.
6. **I could not verify the target model's response structure.** Whether a "Qwen3-8B class" model produces a reasoning preamble, and where its entropy cliff falls, is unknown — which is what the pilot is for. Finding 5's fix depends on the pilot measuring the **cliff position distribution by condition**, not merely "is there a preamble."
7. **I did not read `schema_commitment_band_v1.REJECTED.md`.** The v1 gate finding numbers cited in v2 (1, 17, 18) are therefore unchecked. Nothing in this review depends on them.

---

The v1 gate told you a retraction reached by six routes of instrument failure is not more honest than the claim it retracts. v2 answers by closing all six routes. That is the same error with the sign flipped: you have built a design in which the finding cannot lose, and written the rule down — *"NO CONTROL FAILURE CAN NOW PRODUCE A RETRACTION"* — as the repair.

The correct repair was never "make absence non-credible." It was **make each route to absence individually diagnosable**, so that a null is attributable to a specific named cause and survives when none of those causes fire. Findings 2, 4, 9 and 12 are that repair, itemised. Do those four plus equal-width windows (6), forced generation length (7), and cliff-relative indexing (5), and I expect this to come back approvable.