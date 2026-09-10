# Design v2: Is the top-excluded geometric effect time-localized?

**Author:** Lyra · **Date:** 2026-09-07 · **Status:** SUBMITTED TO GATE, not run
**Supersedes:** `schema_commitment_band_v1.md` — REJECTED, 19 findings. See
`schema_commitment_band_v1.REJECTED.md`. **v1's motivating premise was false**; §1 says how.
**Instrument:** `tools/spectral_bands/` (CC, 2026-09-07), now **vendored into this repo** so
a reviewer can read it. Tarball sha256
`49a7f606e0b5933bd5615beaba6c4576aba0015510661020b3057d9f4bcf28e2` (v1 recorded the first
16 hex chars and called it a sha256 — corrected). 17/17 tests and 12/12 mutants verified by
Lyra on MTH *and* re-run on the analysis box.

---

## 1. What v1 got wrong, and why that makes this experiment different

v1 was framed as a threat-test: CC's Exp 51 shows *representational strain* (token
frequency) owns the top of the SVD spectrum; stable rank is top-dominated; therefore my
"schema commitment" relabel might be measuring the confound, and the design should be
built to retract.

**The source paper contradicts that premise, and I had not read the table.**
`decision-state-paper/academic/main.tex:295–321`:

| feature | *d* | *p* |
|---|---|---|
| Stable rank L15 (top-dominated) | **+0.87** | <0.001 |
| **Skip-SV1** stable rank L15 (dominant SV removed) | **+0.81** | <0.001 |
| Skip-SV1 L11 | +0.73 | <0.001 |
| Skip-SV1 L7 | +0.60 | 0.006 |
| top-SV ratio | −0.29 | n.s. |
| spectral entropy | +0.08 | n.s. |
| encoding norm | −0.002 | n.s. |

And in prose: *"these were entirely driven by prompt length in the unmatched design. Stable
rank is the surviving spectral feature… encoding norm is unrelated to the known/unknown
distinction, **ruling out vocabulary frequency as an explanation for the geometric
signal**."*

**Removing the dominant singular value costs 7% of the effect (0.87 → 0.81), and it holds
at three layers in the same direction.** If representational strain owned this, deleting
SV1 should have gutted it. The top-band features that *would* carry strain — top-SV ratio,
entropy, norm — are all null.

So the coarse version of v1's question is already answered, in the direction v1 was built
to doubt. This is recorded here because v1 was designed to reach the self-critical
conclusion and the gate correctly counted **six routes to "nothing" against one to H1** —
its words: *"a retraction reached by six routes of instrument failure is not more honest
than the claim it retracts. It is the same error wearing better manners."*

## 2. The actual open question

Skip-SV1 removes **one** singular value, at the **layer** level. The middle band removes
the **whole top band** (MP outlier cut) *and* the noise floor beneath it (Gavish–Donoho),
at the **token-window** level.

> **The effect is established. What is unknown is whether it is TIME-LOCALIZED.**

The schema-commitment claim is not "geometry separates honest from confabulated" — that is
already in hand. It is that the separation is **concentrated in the first ~20 generated
tokens**, mirroring the decision-state temporal profile (*d* = 1.02 / 0.89 / 0.37), i.e.
that the model commits to a response mode *before the content that would justify it*.

Nothing in the corpus tests that, because — audited 2026-09-07 — **every saved geometry
artifact is token-averaged at capture** (`emo_k_L0_H0` = (200, 64), trials × dims, no token
axis; `*_cache_L22.npy` = (1024,) steering vectors named "cache"). The temporal profile
exists only in decision-state because it is the one run that saved per-token *logit*
trajectories. **No run has ever saved per-token geometry.**

## 3. Three outcomes, and the null is a plumbing check rather than a retraction

- **O1 — TIME-LOCALIZED.** Middle-band condition effect is larger in windows 0–7 and 8–19
  than in 20–49. Schema commitment holds at the spectral level. The `meta-pattern`
  synthesis stands with a mechanism and a time course.
- **O2 — REAL BUT FLAT.** Middle-band condition effect is present and statistically
  **equivalent across windows** (TOST against a named SESOI, below). *The geometry is real;
  the commitment framing is wrong.* Retract the temporal interpretation, keep the finding.
  **This is the outcome that would cost me the synthesis, and it must be reached by a
  positive equivalence result — never by failure to reject.**
- **O3 — ABSENT.** No middle-band condition effect in any window. **This contradicts
  skip-SV1 at three layers and is therefore a PLUMBING FAILURE, not a null.** Do not
  interpret; debug the capture path.

O3 being a plumbing branch rather than a retraction branch is the structural fix for v1's
six-to-one asymmetry. **The prior evidence makes absence non-credible**, so absence
accuses the instrument, not the hypothesis.

## 4. Confirmatory test — exactly one

**`middle_band_count ~ condition × window`, and the sole confirmatory test is the
WITHIN-QUESTION condition×window interaction contrast (early = windows 0–7 ∪ 8–19) vs
(late = 20–49).**

Window varies within question, so the interaction is estimable within cluster — the gate
called this "a genuine and underexploited strength" of choosing the interaction, and v2
declares it as the only confirmatory test rather than one of several. The condition *main*
effect is between-question with a different bootstrap unit and is **descriptive only**.

Everything else — per-window counts, stable-rank reproduction, the T6 frequency-matched
subset, per-layer breakdown — is **declared descriptive in advance**. No correction is
needed because nothing else is confirmatory.

- α = 0.05, question-level cluster bootstrap, 10 000 resamples, condition labels permuted
  within window.
- **SESOI for O2 equivalence: |Δd| ≤ 0.20** between early and late window contrasts —
  roughly a quarter of the skip-SV1 L15 effect, and smaller than the gap between the
  decision-state early (0.89) and late (0.37) windows the hypothesis predicts. TOST both
  bounds at α = 0.05. O2 is claimed **only** if TOST rejects both.
- MDE computed at question level and recorded in the prereg **before capture**.

## 5. Capture-path verification — the part v1's "17/17, 12/12" did not cover

The gate's finding 17 is correct and it is the one I would have shipped past: the tests
verify the **analysis library**, while C1 (`model.generate()` rebuilding the cache) and L4
(off-by-one) live in the **new capture code**.

**Blocking pre-run assertion — `test_incremental_cache_matches_full_forward`:**
generate *n* tokens with the manual loop, then run a single full forward over
prompt+generated, and assert `K_loop[t] == K_full[len(prompt)+t]` exactly. This kills C1 and
the L4 off-by-one in one test. **The run does not start until it passes.**

**`prompt_boundary_index` asserted and written into every artifact.** Window "token 0" must
be the first *generated* token. The entire claimed effect sits exactly at that boundary,
where a one-token shift moves it — so the boundary is recorded, not assumed.

**Token axis mandatory.** Capture writes `K[trial, token, dim]`; nothing averages before
disk. This is the corpus-wide defect above.

## 6. Controls, each with both questions answered

| Control | If it FAILS | If it SUCCEEDS |
|---|---|---|
| `aperture_ok` on **realised** shape | Band closed → `count=0` everywhere → returns `usable=False`, **never a null**. Run halts. | Counts interpretable. |
| `null_band_count()` at actual n | Floor non-zero → positive counts are not automatically signal; all inference switches to distributional. | Floor 0 → presence vs structural absence. |
| Planted-signal at realised shape | Instrument blind → **O2 and O3 both uninterpretable**. Run halts. | A null means absence, not blindness. |
| Stable-rank + skip-SV1 reproduction | **Known effects absent → O3, plumbing.** | Baseline works; middle band is measured against something that reproduces. |
| Incremental-cache equality (§5) | C1/L4 live → geometry invalid. Run halts. | Capture path sound. |

**Asymmetry audit, v1's fatal flaw, redone.** Routes to O1: the effect exists at three
layers already; the question is only its distribution over windows. Routes to O2: exactly
one, and it requires a *positive* TOST. Routes to O3: instrument failures, all of which
**halt the run** rather than produce a reportable null. **No control failure can now
produce a retraction.**

## 7. Threats

- **T1** Windows are borrowed from decision-state, a different model. Report token-position
  histograms so misalignment is visible. **Pilot first** (gate finding 1): confirm this
  model produces a preamble at all under this system prompt, on ~20 questions, before the
  full run.
- **T2** Deep-content window is only reachable by trials generating ≥20 tokens; window
  membership is not independent of length. Report per-window n; FWL on token count within
  fold.
- **T3** Observational: condition is assigned post-hoc from ground truth. No causal claim.
- **T4** Single model, single layer set. No invariance claim.
- **T5** SimpleQA correlates knowledge with entity frequency by construction. Report the
  frequency-matched subset descriptively.
- **T6** ⚠ **0.3 / 3.2 RECOVERED 2026-09-07 with three corrections** (`oracle-harness/docs/LORA_06_STRAIN_DETECTOR.md`): the extra digits were unsourced precision; the measure is **WHOLE-RESPONSE**; and it is **EFFECTIVE rank (exp of spectral entropy, whole-spectrum), NOT stable rank** — so it never carried the top-domination exposure. It therefore **cannot corroborate a time-course claim in either direction.** T6 said these were transcription-only and that nothing depends on them; that held, and the Agni gate's refusal to treat the transcription as verification is now vindicated by the source withdrawing the number. Exp 51's numbers (24.5 vs 19.4; d=+1.81; rank std 0.317/3.245) appear in this tree
  **only as my transcription**. The gate refused to treat that as verification and was
  right. They are cited as orientation, and **nothing in this design depends on them.**

## 8. Credit

Banding, Gavish–Donoho threshold, MP path, aperture guard, null gate and windowed mode are
**CC's**. The aperture guard exists because CC measured their own advice, found stacking
closes the band (`GD_lo=46.53` vs `MP_hi=46.07`, span −0.46 at 310×512), and retracted it
before it reached me. Lyra: the question, the windows (from decision-state), the design.

## 9. Compute

GPU for generation and per-token capture only; analysis is numpy and vendored. Pilot
(~20 questions) before the full run. Nothing launches until this passes the gate.
