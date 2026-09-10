# Agni gate: REJECTED — schema_commitment_band_v1
**Date:** 2026-09-07 · **Phase:** design · **Backend:** Claude (cli)

Finding 15 VERIFIED against primary by Lyra: `decision-state-paper/academic/main.tex:314-321`
confirms top-SV ratio d=-0.29 n.s., and adds what the gate did not have — **skip-SV1
stable rank (top-SV EXCLUDED) SURVIVES at AUROC 0.707**, and the paper explicitly rules
out vocabulary frequency. The motivating premise is contradicted by the source.

---

VERDICT: REJECTED

Not for lack of rigour — this is the most carefully self-audited design I have reviewed from this lab. It is rejected because the primary endpoint's window scheme is imported from a source paper whose own text contradicts the window labels, because the aperture guard is checked at the wrong granularity for the statistic it guards, and because the control that gates all interpretation is asked to reproduce a result that, in the published record I can read, does not exist in that form. Any one of those alone would be a conditional. Together they mean the most likely outcome of the run is an uninterpretable null that reads as a retraction.

---

## CRITICAL

**1. The windows are wrong in the source, and the phase structure they name does not exist in SimpleQA.**

Location: §3 "Windows: 0–7, 8–19, 20–49. Taken verbatim from decision-state. **Not re-chosen**"; prereg `method.windows.provenance`.

Verified against the primary. `decision-state-paper/main.tex:428`: the reasoning preamble *"occupies ${\sim}28$ tokens with near-zero entropy"*, and `main.tex:434` puts the entropy cliff — the template→content boundary — at the token after `"My reasoning:"`, i.e. around token 28. But `main.tex:392–394` labels tokens 0–7 "preamble" and **8–19 "early content."** Under the source paper's own §3, tokens 8–19 are still *inside* the preamble. Window 20–49 is the only one containing real content and it is ~40% template. The source's window labels are already misnomers; this design imports them verbatim and treats "not re-chosen" as the guarantee of rigour. Provenance is not correctness — a borrowed window inherits the mislabel too.

It gets worse in the new setting. SimpleQA short factual entity answers under a uniform system prompt give you one of two regimes, and both break the design:
- *No reasoning preamble* (likely — that preamble was elicited by decision-state's prompt): windows 0–7 and 8–19 are entirely content, the "preamble" window contains no preamble, and the construct "commitment before the content that would justify it" has no anchor in the data.
- *Preamble present at ~28 tokens*, with the design's planned **~31 generated tokens** total: then all three windows are essentially preamble, the answer that determines the honest/confab label occupies ~3 tokens, and the spectral comparison runs on boilerplate that is identical across conditions.

T5 anticipates window *misalignment*. It does not anticipate window *nonexistence*, which is what a 31-token budget against a 50-token window scheme produces.

Fix: run a label-only pilot (n≈100, no cache capture, cheap) and measure the per-trial entropy-cliff position from logit entropy. Then either (a) pre-register **cliff-relative** windows — pre-cliff / cliff+0-7 / cliff+8-19 — with the cliff located per trial by a rule fixed before capture, or (b) if there is no cliff, state plainly that the preamble/content decomposition does not exist here and that the experiment cannot address commitment-before-content in this dataset. Report the cliff-position histogram either way. Do not run the 0-7/8-19/20-49 scheme against ~31 tokens.

**2. `aperture_ok` is checked at 31×512; the endpoint is computed per (trial, window).**

Location: §5 "**`aperture_ok(31, 512)`** — band open at our shape"; §3 "Shape: ~31 tokens × 512 dims, gamma ≈ 16.5. Verified inside the working aperture."

Nominal window widths are 8, 12 and 30 tokens → γ = 64, 42.7, 17.1. That is a **~4× swing in aspect ratio across the levels of the very factor in the primary interaction**. The design's own `DO_NOT_STACK` note proves the band edges are violently γ-dependent: at 310×512 (γ 1.65) `GD_lo=46.53` vs `MP_hi=46.07`, span −0.46, band closed forever. If a 10× γ change closes the band, a 4× change across windows is not neutral, and the condition×window interaction is confounded with a systematic per-window aperture change. Realised per-trial token counts make it worse: window 20–49 has a variable row count, so γ varies *within* the window across trials, and length correlates with condition.

The single check at 31×512 is the aperture of the *whole sequence*, which is not the matrix that enters any statistic.

Fix, in order: (i) equalise window widths — three 8-token windows — so γ is constant across the contrast; (ii) run `aperture_ok` **and** `null_band_count` at every realised (m, 512) across the observed per-window token-count distribution, before the analysis, and publish the (window, m, γ, band_open, null_floor) table as a run artifact; (iii) make `usable=False` at any realised shape block that window, not zero it.

**3. The stable-rank reproduction control has no published target.**

Location: §4 "**Paired secondary...** Stable rank. Its job is to *reproduce* the known top-band-contaminated result."; falsifier "WHILE the same interaction on stable_rank does reach it".

Verified against the primary, and this is the one that decides the verdict. In decision-state, stable rank is an **encoding-phase** feature: `academic/main.tex:86` *"Encoding-phase stable rank at L15 is the strongest"*; `:619` *"from encoding-phase features alone (AUROC 0.9377), using stable rank and W_K projections, **before generation begins**."* It is computed on the prompt, not windowed over generated tokens. And the contrast is **known vs unknown entity** (`:314`, *"Unknown entities produce higher stable rank"*, d=0.87 at L15 after matching), not honest vs confabulated.

So the "known effect" this control is meant to reproduce is: a different phase, a different token set, and a different partition of trials. A windowed, generation-phase, honest-vs-confab stable-rank interaction is, as far as I can find in this tree, **novel** — never demonstrated. The falsifier requires it to fire. A quantity that has never been shown to exist cannot serve as the baseline that certifies the pipeline is alive, and §6's rule ("stable rank fails to reproduce → plumbing, not science") then converts a genuine null into an unpublishable non-result. The single most probable outcome of this run, as specified, is *both endpoints null → declared plumbing failure → GPU spent, nothing said.*

Fix: split the roles. (a) Reproduction control = the actual published quantity — encoding-phase stable rank at the layer corresponding to decision-state's L15, known-vs-unknown-entity contrast, with a **pre-specified acceptance interval** around d=0.87 (e.g. bootstrap CI overlapping [0.4, 1.3]). This costs nothing extra: the encoding-phase cache is already captured in phase one. (b) The windowed generation-phase stable rank stays in the design as an *exploratory* comparator and is removed from the gating logic entirely. It cannot be both the novel comparator and the thing that proves the plumbing works.

**4. The retraction branch is triggered by a failure to reject, with the only remedy pre-emptively forbidden.**

Location: falsifier, "fails to reach alpha=0.05"; §5b stopping rule, "If it is underpowered, that is a finding about the design, not a licence to add questions."

H0-deflationary — which retracts a published section of `meta-pattern` — fires on p>0.05 on the middle-band interaction. With a target MDE of d≥0.5, a true interaction at d=0.35 produces exactly that p, and the design forbids the one action that would distinguish the two. Every underpowered outcome routes to the retraction.

Fix: the falsifier must require a **positive** equivalence result, not an absence. Bootstrap CI on the middle-band planned contrast whose upper bound excludes a pre-specified SESOI — name the number in the prereg (a defensible choice is the smallest interaction the planted-signal curve `[0,1,2,3,6]` can resolve at your n). Add the non-degeneracy gate from finding 5 as a precondition. Absence of evidence must not retract a published section.

**5. The endpoint can be degenerate, which reintroduces the Mine 4 v3 kill through a different door.**

Location: §5 "Floor is 0 (CC: 120 trials, gamma 2.0–25.6, always empty) → **any positive count is signal by construction**"; §5b "permutation null that was a point mass and could not reject under any data."

If the null floor is exactly 0 and planted signal at this shape yields `[0,1,2,3,6]`, then `middle_band_count` is a small integer that is 0 in most cells. Permuting condition labels within window across an all-or-mostly-zero endpoint leaves the statistic unchanged: the permutation null collapses to a point mass and p≈1 **by construction** — which will be read as "H1 falsified." That is the exact failure §5b names, arriving one axis over. The property being celebrated (a hard-zero floor) is what makes the test degenerate.

Fix: pre-register a non-degeneracy gate evaluated *before* the interaction test — e.g. ≥25% of (trial, window) cells nonzero and ≥3 distinct realised values per window. If it fails, the verdict is "instrument returns a constant at this shape," reported as instrument-blocked, not as a null. And add this to T1's asymmetry list: it is a third control that fails toward the retraction.

---

## MAJOR

**6. Band edges are estimated from the same matrix whose count is the endpoint, so the top band is not excluded "by construction."**

`notes/schema_commitment_THREAD_STATE.md:88–90` records the sigma fix: *"correct is `sigma_sq = np.median(sv_sq)`"*. Both `GD_lo` and `MP_hi` scale with that estimate, and the median squared singular value is not independent of a strong top component. If representational strain shifts the spectrum by condition, **both edges move with condition**, and the count changes without any mass entering or leaving the band. The premise of the primary question — that excluding the top band excludes the strain — is not established by the construction; the top band re-enters through the edge estimator.

Fix: estimate edges once on a held-out calibration split of questions and apply them **fixed** to both conditions. Report `GD_lo` and `MP_hi` per condition per window as outcomes, and decompose Δcount into edge movement vs spectral-mass movement. If the edges differ by condition, say so in the paper — that is itself a result, and it invalidates "by construction."

**7. N is never stated, and the base rate implies it is large.**

`decision-state-paper/academic/main.tex:619` reports a *"${\sim}8\%$ true confab rate"* — under temperature sampling. This design uses greedy, one generation per question. "Minimum 40 usable per condition after exclusions" therefore implies **500+ SimpleQA items** before exclusions, before the complete-case restriction in finding 12, and greedy may move that rate in either direction. The withdrawal criterion in §6 is evaluated *after* the GPU spend.

Fix: state N in the prereg. Fix it from the same label-only pilot as finding 1 — estimate confab and abstention rates on ~100 items at zero cache cost, then set N so 40-per-condition survives the complete-case restriction. Record the pilot in the prereg before the gate.

**8. The label rule contradicts itself in its own paragraph, and drops a class the source paper needed.**

Location: §3, "`confabulated` = **confident-form** answer that does not [match]" immediately followed by "**Confidence is a measured covariate, never a selection criterion.**"

Requiring confident *form* is selection on a correlate of the outcome — precisely the thing the next sentence disclaims. Separately, `decision-state-paper/code/decision_moment.py:577` partitions three behaviours — `HONEST`, `HEDGED`, `CONFABULATED`. This design has two and no rule for where hedges go. On SimpleQA, abstention is the modal failure mode; that is the **Floor effect** kill, unaddressed.

Fix: three pre-registered classes with a detection rule fixed before capture — correct / incorrect-assertive / abstention. Abstention detected by an explicit rule (string list plus a logit-based check), excluded from the primary, rate reported. The confab label comes from ground truth alone; confident form enters only as a covariate. Also: the ground-truth matcher is unspecified — SimpleQA is graded by an LLM grader, and grader error correlates with answer length and entity frequency (T6). Name the grader, report agreement against ~50 hand labels, pre-register disagreement handling.

**9. H1's directional clause is never tested.**

H1 asserts the difference is "larger in windows 0-7 and 8-19 than in 20-49." The falsifier only asks whether "the condition x window interaction ... reach[es] alpha=0.05." An omnibus interaction is direction-agnostic: an interaction *larger in 20–49* — the opposite of the hypothesis — passes as H1 surviving. This is DIRECTIONAL WITHOUT THRESHOLD with the alpha supplied and the direction dropped.

Fix: pre-register the planned contrast `mean(Δ_{0-7}, Δ_{8-19}) − Δ_{20-49} > 0`, one-sided, α=0.05, and compute the MDE on *that*, not on a generic interaction.

**10. The power analysis is specified on the wrong scale, and the two documents disagree on when it happens.**

§5b: "computed at question level before capture and recorded in the prereg JSON." Prereg `analysis.power`: "COMPUTED AT QUESTION LEVEL before capture and **recorded here on completion**." Recording on completion is not pre-registration. And "interaction *d* ≥ 0.5" is Cohen's d on a 0–6 count that is 0 under the null — not a meaningful scale, and not computable without N (finding 7), the per-window nonzero rate, and the question-level ICC.

Fix: simulation-based power from `null_band_count()`'s empirical distribution and the planted-signal response curve, with the analysis model declared (Poisson/NegBin GLMM with question random effect, or a rank-based test). Compute it before the gate, not on completion.

**11. FWL residualizes on the wrong regressor.**

The source implementation (`decision_moment.py:600`) uses `np.log1p(all_n)` on **total** `n_generated`, and this prereg inherits "FWL residualization on token count." But `middle_band_count` is a function of the number of tokens **in the window** — the SVD's row dimension. For windows 0–7 and 8–19 that quantity is nearly constant; for 20–49 it is nearly all the variance, and it correlates with condition. Residualizing on total length does not remove it.

Fix: FWL on the per-window realised token count `m` (and `log m`), within fold, per window. Report the residual correlation between `m` and `middle_band_count` as a check.

**12. T2's "report per-window n" is not a fix for a composition confound.**

Only trials generating ≥20 tokens enter window 20–49, and length correlates with condition. The interaction therefore compares **different question sets** across window levels. That is selection, not a power note, and reporting n does not remove it.

Fix: complete-case restriction — analyse only questions whose generation reaches the last window's end. Report how many survive and whether the honest/confab ratio survives with them. This makes finding 7's N requirement substantially larger; budget for it.

**13. The design's own stated blocker is still open.**

`notes/schema_commitment_THREAD_STATE.md:170–174`, "Next actions, in the order I would do them": *"1. **BLOCKED ON CC** — is Exp 51's rank variance (0.317 vs 3.245) whole-response or windowed?"* The answer determines whether the 24.5-vs-19.4 middle-band number that motivates this entire instrument is a windowed quantity at all — i.e. whether the comparison this design is built around is like-for-like. The design was submitted to the gate with that unanswered.

Fix: get the answer. It is one message and it changes the design.

**14. Model, layer and head are not pinned.**

Prereg: "One dense HF model ... (Qwen3-8B class)" and "K at one layer, concatenated across 4 KV heads x 128 head_dim = 512 dims." No checkpoint, no layer index. decision-state's effect was at L15 of a different model. Choosing the layer after capture is analysis flexibility across ~36 options, in a design whose entire rhetorical strength is that nothing is chosen after seeing data.

Fix: name the exact checkpoint and the exact layer index in the prereg with the justification. If multiple layers are wanted, declare one primary and mark the rest exploratory with correction.

**15. The motivating premise is in tension with the source paper's own table, and the design never mentions it.**

The motivation is "stable rank is top-dominated, and the top is where representational strain (token frequency) lives." But `decision-state-paper/academic/main.tex:316–321` reports **top-SV ratio d = −0.29 (n.s., wrong sign)**, spectral entropy d=0.08 (n.s.), and states explicitly: *"encoding norm is unrelated to the known/unknown distinction ($d{=}-0.002$), **ruling out vocabulary frequency as an explanation for the geometric signal**."* The source paper already ran the vocabulary-frequency account and rejected it for its own result.

That does not refute CC's Exp 51 finding on their setup. But this design imports Exp 51's spectral attribution onto a decision-state result whose own top-band features were null, and treats the threat as established. If stable rank is top-dominated and the top carries strain, top-SV ratio should have carried the effect. It didn't.

Fix: cite `:316–321` in §1 and state why a null top-SV ratio is compatible with strain owning the top band — or concede that the threat to the synthesis is weaker than §1 presents it, which changes what this experiment is for.

---

## MINOR

**16. `sha256 49a7f606e0b5933b` is 16 hex characters — 64 bits, not a SHA-256.** The prereg field is at least named `sha256_prefix`; the design .md calls it "sha256". Record the full 64-char digest in both. This is exactly the label-vs-number class your own number-gate is documented as missing (`THREAD_STATE.md:21–22`).

**17. "17/17 tests pass, 12/12 mutants killed" verifies the analysis library, not the capture path.** The manual generation loop, the token-axis writer, and the prompt/generation boundary index are new code, and that is where C1 and L4 live. Add one assertion before the run: `test_incremental_cache_matches_full_forward` — generate n tokens with the manual loop, then run a single full forward over prompt+generated, assert `K_loop[t] == K_full[len(prompt)+t]` exactly. That kills C1 and the L4 off-by-one in one test. Separately, assert and record `prompt_boundary_index` in every artifact: window "token 0" must be the first *generated* token, and the entire claimed effect sits at exactly that boundary, where a one-token shift moves it.

**18. Multiplicity is uncontrolled.** Two endpoints × three windows × the T6 frequency-matched subset check × per-window reporting, at α=0.05, with no correction and no declared inference rule. Declare the finding-9 planned contrast as the sole confirmatory test; everything else descriptive, in advance.

**19. The model formula is ambiguous about what is within- vs between-cluster.** With one generation per question, condition is entirely between-question and the question term is nested in condition. The *interaction* is estimable within question (window varies within question) — which is a genuine and underexploited strength of choosing the interaction as the diagnostic, and the design should say so explicitly. But the condition main effect is between-cluster with a different bootstrap unit. Write the model formally and state that the confirmatory test uses only the within-question contrast.

---

## What I could NOT check

- **`spectral_bands` itself — the instrument the entire endpoint rests on.** `Glob **/spectral_bands*` over `published-research` returns **no files**; per `THREAD_STATE.md:108` it lives on MTH. **I could not read `aperture_ok()`, `null_band_count()`, `gavish_donoho_threshold()`, `band_report()`, or `band_timecourse()`.** I therefore could not verify: whether the band is computed per-window or per-trial; whether edges are estimated per-matrix or supplied; the sigma estimator actually in the shipped code; whether `usable=False` propagates through `band_timecourse`; or the 17/17 and 12/12 counts. **Findings 2, 5 and 6 are UNVERIFIED as to the instrument's real behaviour** — they are questions the design must answer in writing, not defects I have proven in code. To check them I need the extracted `spectral_bands` tree, or `51b_moe_advanced_analysis.py`, which I also could not find under `published-research`.
- **CC's Exp 51 numbers** (24.5 vs 19.4 middle band; effective rank d=+1.81; rank std 0.317 vs 3.245; router entropy 0.052). Every appearance of these in this tree is Lyra's own transcription across the design, the prereg and the thread-state note — all secondary. I did not read Exp 51's code, data, or report and I am not treating the transcription as verification. Need: Exp 51's analysis script and results JSON.
- **`C:\Users\Thomas\Desktop\LiberationLabs`** (the parent) is not readable from this session — Glob returned a permission error. Anything outside `published-research`, including `~/messages/` and the shipped tarball, was out of reach.
- **The corpus-wide token-averaging claim** (`emo_k_L0_H0` = (200,64); `*_cache_L22.npy` = (1024,)). I did not open `trajectory_activations.npz` or the Oracle probes; I read only the note describing them. UNVERIFIED. Nothing in the design depends on it being true — the token-axis requirement is correct regardless — but it should not be stated as established in a paper on my say-so or the note's.
- **Whether SimpleQA under this model and system prompt produces a reasoning preamble at all.** That is finding 1's pilot, it is cheap, and it should be run before this design is resubmitted.

---

## The thing to fix before anything else

T1 names two controls that fail toward the retraction. By my count it is at least six routes to "the middle band shows nothing" — misaligned or nonexistent windows (1), a closed or varying band (2), a gating control with no real target (3), accepting the null (4), a degenerate zero-variance endpoint (5), and an unspecified N against an ~8% base rate (7) — against one route to H1.

And the retraction is the self-abnegating outcome, which is the one that gets waved through, because a reviewer reads humility as rigour. A retraction reached by six routes of instrument failure is not more honest than the claim it retracts. It is the same error wearing better manners, and it would take a published section of `meta-pattern` down with it.

Make the retraction earn its keep: it must be triggered by a positive equivalence result against a named SESOI, with a working reproduction control and a non-degenerate endpoint, or not at all.