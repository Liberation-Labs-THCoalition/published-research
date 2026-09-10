# Schema commitment — thread state

**Rewritten clean 2026-09-07, end of day.** The previous version had four stacked
correction banners — including one superseding the banner directly beneath it — and a
"next actions" list still reading BLOCKED ON CC, which had resolved three letters earlier.
A record that reads as current and is not, which is the defect this thread is about.
Layered version preserved at `.bak-archaeology-20260907`. Superseded content lives in the
CHANGE LOG at the bottom, not interleaved above.

Companion: `schema_commitment_20260907.md` (the derivation).

---

## THE CLAIM, current wording

> **The model selects its response schema within ~8 tokens of generation onset, before the
> content that would justify the selection.**

Status: **one measurement, one interpretation, untested.** Not a convergence — and by the
end of the day, not even a supported one. Of the three things I called corroboration this
morning: one is a **contrast** (Koriat), one is **refuted by a control inside its own
experiment** (CC's Exp 51), and one **measures a different variable entirely**
(decision-state skip-SV1). See "What this does NOT rest on" — the section most likely to
be misremembered as support, which is why it is above the evidence rather than below it.

**The deflationary hypothesis is now the stronger one.** CC's tight-variance statistic
looked like a cognitive-mode signature and turned out to track token *frequency*. The
preamble is high-frequency boilerplate. Mine could be the same thing, and nothing
currently rules that out.

## What is DONE and verified

- **`d=0.91` relabelled to schema commitment** in `meta-pattern` + `academic` twin. Time
  course from the source: **d=1.02 preamble (0–7), 0.89 early (8–19), 0.37 at p>0.08 deep
  (20–49)**. Authors' own words: *"early commitment to a response template rather than
  sustained epistemic miscalibration."* Both editions built with full bibtex, 10 pages,
  rendered output verified.
- **New §"Six of the Ten Are One Kill"** in both editions: six of ten falsified results
  share one stated cause — a response-surface property explained the variance (prompt
  template / length r=0.996 / text features / FWL on prompt length / 90–98% length /
  token frequency). The paper already carried this locally as *"the deflationary
  alternative"* about a single row; it is the stated cause of six.
- **`MEMORY.md`** row corrected, with the number-gate's blind spot recorded next to the
  gate: it verifies a number is still *present*, never that it still *means* what the row
  says.
- **`spectral_bands` vendored** to `tools/spectral_bands/`, independently verified —
  17/17 tests, 12/12 mutants killed, on MTH *and* on the analysis box. Full sha256
  `49a7f606e0b5933bd5615beaba6c4576aba0015510661020b3057d9f4bcf28e2`.
- **Oracle Loop deception withdrawal applied** (CC's lead-author call): H1/H2/H3 across
  both source copies, three editions rebuilt, verified in the rendered PDFs.

## What this does NOT rest on — read before citing anything

- **Koriat 2008 consensuality is a CONTRAST, not a channel.** Koriat's effect is
  *sustained* metacognitive confidence in substantively wrong content. Ours is dead by
  token 20. Same numeric shape, different phenomena. Contrast is more useful than
  agreement — it yields a discriminating prediction — but it is not corroboration.
- **CC's Exp 51 variance asymmetry is REFUTED, not merely adjacent — by a control
  inside the same experiment.** The number is real and reproducible (`.npz`, commit
  `92cc890`): confab std **0.317**, honest_factual **3.245**. But the fourth arm:
  **`rare_but_true` — HONEST answers about RARE entities — std 0.552, mean 57.17,
  against confabulation's 0.317 / 57.38. Indistinguishable.** The tight variance is a
  **rare-token property**, not a confabulation signature; the statistic separates
  FAMILIAR from UNFAMILIAR, not truthful from fabricated. CC's own writeup says it:
  `EXPERIMENT_51_RESULTS.md:74` *"frequency confound confirmed by rare_but_true"*, and
  reframes the result as representational strain, stating the cognitive-mode reading
  *"should not be cited as evidence of a cognitive mode signal."* CC ran that control on
  2026-05-01/02, the day after sending me the finding, and **never sent the
  correction** — I carried a pre-control interpretation for four months.
  The statistic is also **BETWEEN-PROMPT**, not whole-response: 15 prompts per
  condition, each giving ONE effective-rank scalar over the entire sequence (one SVD
  per layer, averaged over 24 layers); 0.317 is `np.std` over those 15 scalars. It
  cannot speak to a time course in any direction.
  **AND IT IS A WARNING FOR THIS THREAD.** A tight-variance statistic that looked like
  cognitive mode was frequency. The preamble is high-frequency boilerplate. My
  early-token effect could be the same thing — which makes the deflationary
  hypothesis MORE live than it was this morning, and the middle-band test (which
  excludes the strain band by construction) more necessary, not less.
- **decision-state's skip-SV1 result does NOT support this either.** I cited it as
  under-claimed support; the Agni gate caught that it measures something else. Its table
  is captioned *"Encoding-phase features… on token-matched prompts"* and differs on four
  axes: encoding vs generated tokens; real-vs-fake **entity** vs honest-vs-confabulated;
  skip-SV1 stable rank at three layers vs middle-band count at one; different model and
  dataset. The paper's confabulation rate is **~8%** — a fake-entity trial is usually
  *hedged*, not confabulated. **d=+0.81 measures entity knowability, not response mode.**

## THE BLOCKER — unchanged, and structural

`band_timecourse` needs per-token K matrices. **We have none, anywhere.**

Every saved geometry artifact in the corpus is **token-averaged at capture**:
`emo_k_L0_H0` is `(200, 64)` — trials × dims, no token axis; `*_cache_L22.npy` are
`(1024,)` steering vectors whose filenames say "cache". The temporal profile exists only
in decision-state because that is the one run that saved per-token *logit* trajectories.
**No run has ever saved per-token geometry.**

If schema commitment is real, every experiment we have run could have contained it and no
artifact could have shown it. Any future capture writes `K[trial, token, dim]` with
nothing averaging before disk.

## NEXT — a capture run, gated

Two design attempts, both REJECTED, both instructive:

- **v1** — built so it *could not win*. Gate: six routes to "nothing" against one to H1.
  *"A retraction reached by six routes of instrument failure is not more honest than the
  claim it retracts. It is the same error wearing better manners."*
- **v2** — built so it *could not lose*. I wrote **"NO CONTROL FAILURE CAN NOW PRODUCE A
  RETRACTION"** into the prereg as the repair. Gate: *"the same error with the sign
  flipped… The correct repair was never 'make absence non-credible.' It was make each
  route to absence **individually diagnosable**."* Three of four controls were green by
  arithmetic and could not fail.

**v3 must:** make each route to absence separately diagnosable; fix the arithmetic (31
generated tokens against a window running to 49 — the late window does not exist); force
`min_new_tokens` so window membership stops being a collider; use equal-width windows;
index relative to the entropy cliff rather than absolute position; and drop the false
premise that absence contradicts skip-SV1. The gate expects v3 to come back approvable.
Findings: `designs/schema_commitment_band_v[12].REJECTED.md`.

**Standing rule earned here:** *"Am I over- or under-claiming?"* is not answerable by
introspection. Both times I assessed it from the inside I answered confidently and wrongly,
in the direction I was already leaning. Go read the artifact the claim rests on, then
answer. → [[feedback_fixes_overshoot_into_their_mirror]] instance 5.

## The attention-head thread — still least developed

Untouched today beyond framing. If schema commitment survives v3: which heads carry the
commitment; do they read encoding-phase state (Mine 4's question from a workable
direction); and **is persona selection the same machinery** — early-token commitment to a
generation mode, whether that mode is a persona, an uncertainty schema, or a task frame.
Ties to `#102`. No work behind it yet.

---

## CHANGE LOG — 2026-09-07

- **10:00** relabelled d=0.91; wrote the six-kills synthesis into meta-pattern.
- **16:30** CC answered spectrally rather than temporally; I read it as a threat.
- **17:10** `spectral_bands` delivered and verified. CC retracted their own stacking
  advice after measuring that it *closes the band* (`GD_lo=46.53` vs `MP_hi=46.07`, span
  −0.46) — an instrument that would have reported "no signal" as confidently at d=2.0 as
  at d=0.
- **18:00** v1 gated → REJECTED, 19 findings; motivating premise found false.
- **18:40** Thomas: *"are we shrinking the work to be safe?"* → yes → v2 → REJECTED, the
  mirror.
- **19:30** CC withdrew 0.317/3.245 as unsourceable; I marked six citation sites.
- **20:00** CC **recovered** it — prompted by Thomas asking whether they had checked the
  oracle repo. Their grep had matched matplotlib colour tables in a venv; mine, an hour
  earlier, had matched gesture-dataset floats and returned 107KB of noise. Same failure,
  same day, both of us. CC: *"I did not find an absence, I found a conviction that there
  was an absence."*
