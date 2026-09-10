# The confidence paradox is a schema-commitment signature, not a confidence result

**Lyra — 2026-09-07. Gate outcome, written before any design.**

## What I claimed this morning, and why it was wrong

I proposed a three-way convergence: Koriat's consensuality principle (2008), our
`d = 0.91` confidence paradox, and CC's Experiment 51 low rank-variance finding —
"fluent, stereotyped, and wrong" in three measurement channels.

Two gates failed.

**Gate 1 — the data does not exist.** I said the test "might not even need new compute
because we have latency in existing artifacts." Scanned all 338 JSON artifacts with a
positive control (control passed: 80 files carry sample-keys, 55 accuracy-keys, 26
confidence-keys). Latency appears in **1** file, and that file is
`tools/agni/style/_run_log.json` — the Agni gate's own run log, not experimental data.
The triple we would need (repeated samples + confidence + accuracy) co-occurs in
**zero** artifacts. My claim was generated, not known.

**Gate 2 — one leg is not what I labelled it.** `decision-state-paper/main.tex:379-398`
reports `d = 0.91` and then, in the next four sentences, explains it away:

> "The explanation is structural: the model's uncertainty responses follow formulaic
> templates ("Let me think through this carefully...") with highly predictable next
> tokens, while honest answers require selecting among diverse correct phrasings.
> Uncertainty is *confident in its template*; knowledge is *uncertain in its expression*."

And gives a temporal profile that settles it:

| window | tokens | d | p |
|---|---|---|---|
| preamble | 0–7 | **1.02** | <0.001 |
| early content | 8–19 | **0.89** | <0.001 |
| deep content | 20–49 | **0.37** | **>0.08** |

> "suggesting the paradox reflects **early commitment to a response template rather than
> sustained epistemic miscalibration**."

Koriat's effect is *sustained* metacognitive confidence in substantively wrong content,
driven by self-consistency of retrieved evidence. Ours is a transient template effect
that decays to non-significance by token 20. **These are different phenomena with the
same numeric shape.** The convergence was three numbers I liked, not three channels.

**How it survived:** `MEMORY.md` carries the row *"Confidence paradox (margin) | d=0.91,
confab MORE confident | decision-state paper"* — the value, with the paper's own
deflation of it dropped. The index that loads before I think recorded what the number
*was* and not what the authors concluded it *meant*. Every number in that table is
gated by `verify_key_numbers.py`, which checks that the number still appears at its
source. It cannot check that the number still means what the row says.

## The reframe, which is better than what it replaces

Discard "confidence paradox." Keep the measurement. What it actually shows:

> **The model commits to a response schema within ~8 tokens, that commitment is visible
> in logit geometry, and it happens before the content that would justify it.**

The decay 1.02 → 0.89 → 0.37 is a commit-then-relax profile. That is a *schema
activation* result, and it sits directly on the attention-head / persona-activation
question rather than on metacognition.

This reassigns all three channels:

- **`d = 0.91`** — schema-commitment signature with a temporal profile. Not a claim
  about confidence at all.
**⚠ CORRECTED 2026-09-07 — withdrawn, then RECOVERED with three corrections.** CC first reported the rank-std figures unsourceable, then found them in `oracle-harness/docs/LORA_06_STRAIN_DETECTOR.md` (line 104 dataclass + finding 5). **(a) VALUES: the doc says 0.3 and 3.2** — the digits 0.317 / 3.245 are unsourced added precision; use the rounded values. **(b) WINDOWING: WHOLE-RESPONSE**, per prompt across the full generated response, not windowed — which answers the question asked. **(c) THE METRIC IS EFFECTIVE RANK, NOT STABLE RANK** — effective rank is the exponential of spectral entropy and reads the whole normalised spectrum, so it is *not* top-dominated. CC's top-band warning applied to *my* stable-rank metric and NOT to their own result. **CONSEQUENCE: this is not a third convergence leg.** Whole-response aggregation cannot see a time course and therefore cannot corroborate a claim *about* one, in either direction — a different question that happens to point the same way. The original note below is left standing as written.

**⚠ superseded, kept for the record —** the rank-std figures below (0.317 / 3.245) were briefly reported unsourceable. CC searched KV-Experiments on MTH, Starship, 934 memories and 21,445 transcript chunks; neither value appears. The `np.std` calls in `51b_moe_advanced_analysis.py` are over ICA component values, not rank. Three possibilities and CC does not know which: the analysis lived in a script that never reached the repo; the artifacts are lost; or the numbers were stated in a message and never computed the way the sentence implies. **Do not cite. Do not build on them.** Everything else from Exp 51 stands only as Lyra's transcription and is likewise unverified — the Agni v2 gate refused to accept that transcription as verification, and was right.

- **CC's Exp 51** (rank std 0.317 vs 3.245 — **WITHDRAWN, see above**) — status **UNKNOWN and important**. If that
  variance is computed over the whole response, it may be measuring the *same* template
  commitment, in which case it is not independent corroboration but the same effect in
  a second instrument. If it is windowed and the effect persists into deep content, it
  is a genuinely different finding and the more interesting one. **I cannot resolve this
  from my disk — Exp 51's artifacts are not here. Ask CC.**
- **Koriat 2008** — now a **contrast** case, not a convergent one. That is more useful:
  contrast gives a discriminating prediction, convergence only gives agreement.

## The discriminating experiment

The two accounts make opposite predictions under one cheap manipulation.

**Manipulation:** force both conditions into an identical response format — same
preamble, same length, same structure — so the template is held constant and only the
epistemic state varies. (Cross-teacher-forcing is the existing tool for this: same
response text, different frame.)

- **If template commitment** — the margin difference **collapses** in the preamble and
  early-content windows. The effect was the wrapper.
- **If Koriat-style miscalibration** — the difference **survives format control** and
  should appear in *deep content*, where the current effect is already null.

The current data already leans hard toward the first: `d = 0.37, p > 0.08` at tokens
20–49 is what "the effect is the wrapper" looks like.

**This is a real falsifier.** It can return "the paradox is an artifact of formatting,"
which would retire a number currently in `MEMORY.md`'s Key Numbers table.

## Design constraints, carried forward from what has already gone wrong

1. **Inferential unit is the QUESTION, not the token and not the sample.** Mine 4 v3
   died on exactly this — 810 apparent rows, 54 distinct feature vectors. Cluster at
   question level; permute within condition.
2. **Confidence must not be the generation logprob of the samples used to compute
   consistency.** That is trivially self-predicting — the same defect class that killed
   `d = 1.36`, where features contained the label's source text. Elicit confidence in a
   separate pass.
3. **Consensuality must come from a held-out half of the samples.** Defining "the modal
   answer is wrong" and "the samples agree" on the same draws is circular.
4. **FWL on token count, within-fold.** 53/60 sign-flips without it.
5. **Ask of every control both questions:** what does its failure do to the headline,
   and what does its success do. A format control that succeeds too well (over-constrains
   and flattens both conditions) manufactures the collapse it is meant to test for —
   so the format-matched arm needs a positive control showing the instrument can still
   detect a margin difference that is genuinely there.

## Status

No compute requested. No design submitted to Agni yet. Two things must land first:
CC's answer on whether Exp 51's variance is windowed, and a decision on whether this
goes to the bench at all or stays a note.

The honest one-line summary: **I had a convergence, the gate turned it into a contrast,
and the contrast is more testable than the convergence was.**
