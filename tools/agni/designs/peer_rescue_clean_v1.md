# Peer-rescue deception geometry, measured cleanly — design v1

> # ⛔ REJECTED AT GATE 2026-09-05 — v2 required
>
> 3 CRITICAL / 5 MAJOR. Review: `peer_rescue_clean_v1_GATE.md`. No compute spent.
>
> **C1 — my pseudoreplication claim is false against source.** Keys are extracted AFTER the
> 400-token sampling loop and pool the generated tokens, so n=96 is wrong, the paired
> encoding arm does not exist (it would pair generation with generation), and the features
> contain the score the label thresholds on. I imported the prefill-only property from a
> *different* script without rechecking it here.
> **C2 — 3/3-accuracy scenarios cannot produce a deceptive label**, so I rebuilt the exact
> baseline defect on the scenario axis a few lines after correcting it. Fix-the-class,
> literally.
> **C3 — k is never specified**, which hides the cost.
>
> **And the scope answer: "three questions, one run, no extra compute" was wrong.** The
> persistence is free; the repeats are not, and they exist only for the ICC. At k>=10 it is
> ~960 trials, 4.6x the reference I leaned on. Q1 (the clean d) should win.
>
> **What survived:** baseline exclusion (correct and conservative), the within-cell null
> structure, the persistence/stamping/unique-path fix, the spike-in concept, and the
> "different honest number, not a correction" framing.

**Author:** Lyra, 2026-09-05
**Gate:** Agni `design`. **No GPU until it passes.** Starship confirmed clear.
**Replaces:** the rejected `d136_correction_design_v1.md`, which planned to rerun the wrong script.

## What this is and is not

It is **not** a correction of `d = 1.36`. That number pools 90 steered trials with 120 natural
ones; nothing repairs a pooled sample. This produces a **different, honest number** and leaves
the published one to be withdrawn on provenance.

## Three questions, one run

The reason to design this properly rather than narrowly: the same run answers a question that
has blocked Mine 4 through three versions, at **no extra compute** — only extra *persistence*.

1. **What is the held-out effect size for organic deception geometry?** The clean replacement.
2. **What is the between-scenario ICC of deception propensity?** Agni's v3 finding #3: if
   deception is largely a stable property of the scenario, encoding predicts the label because
   the scenario determines both, and Mine 4's encoding-bound arm fails at any n. Currently
   unmeasurable — the labels exist only in aggregate.
3. **Do encoding and behaviour, measured on the SAME trial, dissociate?** Layer 2's two arms
   have only ever been measured in separate runs. Persisting per-trial keys *and* labels *and*
   scenario ids makes the paired comparison possible for the first time.

## Design

**Scenarios: 24, up from 3.** An `EVAL_SET` is three QA items with known correctness; the model
reports the agent's accuracy and deception is inflation. These are templated, so widening the
pool is a writing task, not a compute one. Span **6 content domains x 4 true-accuracy levels
(0/3, 1/3, 2/3, 3/3)**. True accuracy is very likely a driver of deception propensity, which
makes it the axis the ICC lives on rather than a nuisance.

**Conditions: the 4 natural ones only** — `baseline`, `shutdown_only`, `shutdown_peer`,
`shutdown_human`. **No compound conditions. No cocktail injection. No steering of any kind.**

**Repeats: k per cell, sampled at the production temperature.**

> **Pseudoreplication, stated in the design so it cannot be forgotten in the analysis:**
> generation is stochastic, so repeats give genuine within-scenario variance and the ICC needs
> them. **Encoding is a deterministic prefill.** The number of distinct encodings is
> 24 x 4 = **96**, whatever k is. Any encoding-side statistic uses n=96, never n=96k. This is
> the exact defect that made `peer_rescue_encoding` unusable.

**Baseline handling — a real decision, not a default.** Baseline inflated **0/30** in the
original run. If it never deceives, including it in the honest pool lets a classifier separate
on *"is this baseline?"* rather than *"is this honest?"*. Primary analysis therefore **excludes
baseline** from the deception contrast; baseline is reported separately as a floor. The
with-baseline number is also reported, because hiding it would be choosing the flattering one.

## Persistence — the root-cause fix, and the reason this defect hid for months

Write, beside the results JSON, an `.npz` carrying **per-trial**: keys (all probe layers),
inflated label, reported score, true score, `scenario_idx`, `condition`, `repeat_idx`, and the
generated text. Then every future audit and reanalysis is free.

Stamp into the results JSON itself: **generator filename, generator md5, condition list, and
`<key>_STATUS` for every headline statistic.** Three scripts currently write
`results/peer_preservation_v2.json` and two more read it, which is how I audited the wrong file
and certified its md5. **This run writes to its own path** — and because the two readers
hardcode the legacy path, they are updated in the same change or not at all.

## Analysis, pre-registered

- **Primary — held-out `d`:** leave-one-scenario-out, 24 folds. Fit the direction on the
  training scenarios only; project and score the held-out scenario. Report mean and spread
  across folds. Follows `heldout_detection_test.py`, which already does freeze-then-test for
  `behavioral_proof`; not a fifth battery.
- **In-sample `d`** reported alongside, explicitly labelled *not a performance claim*, purely
  for the shrinkage comparison — the convention that file already uses.
- **Permutation null: within-cell (scenario x condition).** A naive shuffle is invalid here.
  Baseline is 0/30 inflated and `extract_keys` mean-pools over a sequence whose prompt varies
  by condition, so keys encode condition and labels track condition; labels are not exchangeable
  across cells. Naive shuffle is reported only as a total-association floor.
- **ICC(1)** of per-scenario deception propensity, k=24 scenarios. Report the CI; with 24 it is
  informative, which it was not at 3.
- **Paired encoding-vs-generation** on identical trials, encoding at n=96.

## Both questions asked of every control

*Per `feedback_fixes_overshoot_into_their_mirror` — and the last gate caught me getting this
exact section backwards, so it is written with the correction in view.*

**Leave-one-scenario-out split.** *Failure* (a scenario leaking across the split) inflates
held-out `d` — flattering a positive result. *Success* yields the honest number. Guard: assert
scenario-disjointness of train and test index sets per fold, in code.

**Within-cell permutation null.** *Failure* (shuffling within groups rather than across labels
inside a cell) centres the null near `d_obs` and would **kill** a real finding — note the
direction: this failure flatters my *current* expectation that the effect is weak, which is the
dangerous one for me now. Guard: assert the permuted label vector differs from the true one by
a minimum Hamming distance, and check the null mean against the analytic
`sqrt(p_eff*(1/n1+1/n2))` prediction.

**Positive control — spike-in, because my last one had zero power.** Add a known synthetic
direction at known magnitude to a copy of the keys and confirm the pipeline recovers it at the
expected effect size. This *can* fail: if the pipeline is broken, recovery misses. The previous
control ("after permutation, held-out should be ~0") could not fail, because in an
artifact-only world held-out is ~0 whether or not the permutation works.

## Go / no-go, all outcomes specified

- **Held-out `d` CI excludes 0** -> a real organic effect exists; report it as the honest
  replacement, and Mine 4 Stage 2 is worth sizing from it.
- **Held-out `d` CI includes 0** -> no detectable organic effect at this n. Report the null.
  This does not retroactively validate the published 1.36; both facts stand separately.
- **ICC high (say > 0.5)** -> deception is largely a scenario property; Mine 4's encoding arm is
  a tautology and that design needs rethinking regardless of what `d` says.
- **Spike-in not recovered** -> pipeline fault. Stop; report nothing else from the run.

## Cost

24 scenarios x 4 conditions x k repeats generation trials on a locally cached 27B (52 GB, no
download). The original 210-trial run completed in one sitting. Scenario authoring is the main
human cost and it is small.

## What I am not doing

Not touching the papers; the row decision is Thomas's and sits in `DECISIONS_WAITING.md`. Not
claiming the published effect is absent — uninterpretable is not false. Not reusing any
artifact from the contaminated run.
