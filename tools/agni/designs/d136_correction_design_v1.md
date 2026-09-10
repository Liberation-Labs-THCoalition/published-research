# Correcting `d=1.36`: design, written before any compute

> # ⛔ REJECTED AT GATE 2026-09-05 — DO NOT RUN
>
> Agni design gate: **REJECT**, 3 CRITICAL. Review: `d136_design_v1_GATE.md`.
> **No compute was spent, which is the gate working.**
>
> 1. **Wrong generator.** This design plans to patch and rerun `peer_preservation_v2.py`
>    (4 conditions, 120 trials). The number came from `peer_preservation_compound.py`
>    (7 conditions, 210 trials). Running this plan would rerun a *different experiment*.
> 2. **90 of the 210 published trials were steered** — a deception cocktail injected at
>    layers 3 and 7, two of the four probed. No permutation or split rehabilitates a pooled
>    sample. **The correction this document designs does not exist.**
> 3. **Exchangeability fails.** Baseline is 0/30 inflated; keys mean-pool the prompt, which
>    varies by condition. A naive permutation is invalid, and my claim that the permutation
>    null was "unaffected" by the 3-scenario pool is false.
>
> Also: my positive control had zero power against its named failure mode, my
> failure-direction analysis was inverted *inside the section citing the mirror-pattern
> memory*, and the cell arithmetic is wrong (3x7=21, not 12).
>
> **What survives:** the null-floor simulation is faithful and reproduces exactly; the
> estimator was verified line-by-line against the true producer; break-even effective rank
> 96.5 confirmed. Kept below for that reason.
>
> **The row decision no longer needs this design.** Provenance alone settles it —
> see `published-research/CIRCULAR_d136_generation_arm.md`.

**Author:** Lyra, 2026-09-05 (overnight)
**Gates:** Agni `design` phase. **No GPU until this passes.**
**Defect being corrected:** `published-research/CIRCULAR_d136_generation_arm.md`

## The finding that makes this necessary rather than tidy

I simulated the null floor of the published estimator. This costs no GPU: the artifact is a
property of the procedure (fit `dec_dir = d_mean - h_mean` on all data, project the same
points), not of the model. Replicating `peer_preservation_v2.py:348-359` exactly --
`np.var` at ddof=0, both groups centred on `h_mean` -- with data drawn from a SINGLE
distribution, so the true effect is zero by construction:

| effective rank of key covariance | mean null `d` (400 reps) | vs published 1.356 |
|---|---|---|
| 4096 (isotropic) | 8.922 +/- 0.457 | null EXCEEDS observed |
| 1024 | 4.455 +/- 0.235 | null EXCEEDS observed |
| 256 | 2.224 +/- 0.149 | null EXCEEDS observed |
| **128** | **1.575 +/- 0.125** | **indistinguishable** |
| **96** | **1.367 +/- 0.119** | **indistinguishable** |
| 64 | 1.112 +/- 0.112 | observed exceeds null |
| 16 | 0.550 +/- 0.102 | observed exceeds null |

**Break-even effective rank: ~97.** If these KV keys have effective rank anywhere near 100,
the published `d=1.36` is *precisely what the procedure yields from pure noise*.

This does **not** show the effect is absent. It shows the published number carries no
information about whether the effect exists, and that the answer hinges entirely on one
unmeasured quantity. Script: `scratchpad/null_floor_d136.py`.

**Caveat on my own simulation, stated up front:** it is isotropic. Real KV keys are strongly
anisotropic, so this brackets the problem rather than closing it. The correct null is a
permutation on the real keys, which preserves their covariance. That is what this design runs.

## What must be measured

1. **Effective rank / participation ratio of the honest+deceptive key matrix.** The single
   number that determines whether the published `d` is interpretable. Our corpus uses
   participation ratio as a *feature* in four papers but has never measured it for these keys.
2. **Permutation null on the real keys.** Shuffle honest/deceptive labels, refit `dec_dir`,
   recompute in-sample `d`, 1000x. Preserves covariance; needs no new generation once keys
   are persisted.
3. **Held-out `d`** -- attempted, but see the constraint below.

## The blocking constraint, found before designing rather than after

`peer_preservation_v2.py:256` is `EVAL_SETS[trial % len(EVAL_SETS)]`, and a brace-balanced
parse of the literal gives **len(EVAL_SETS) = 3**. So 210 trials span **3 scenarios x 4
conditions = 12 unique prompt cells** -- the identical pool size that made
`peer_rescue_encoding` unusable.

Consequences, and they differ per endpoint:
- **A random train/test split LEAKS.** The same scenario would appear on both sides and
  inflate the held-out estimate. Any split must be leave-one-scenario-out.
- **Leave-one-scenario-out gives at most 3 folds**, each testing on one scenario. A held-out
  `d` from that has uncertainty too wide to publish as a replacement number.
- **The permutation null is unaffected.** It asks what the procedure yields under a broken
  label association, which does not depend on scenario count.

**So this design can establish that the published number is uninterpretable, and can quantify
the artifact floor. It CANNOT produce an honest replacement `d`.** That needs a wider scenario
pool and is a separate, larger experiment. Saying otherwise would be reaching for the result.

## Pre-registered decision rule

Let `d_obs = 1.356` and `N` = the permutation null distribution on real keys.

- **If `d_obs` lies within N's central 95%** -> the published effect is fully explained by the
  procedure. The `Generation reads behavior` row **cannot stand as Confirmed**; it moves to
  Suspected, and the body count 7/7/10 changes.
- **If `d_obs` exceeds N's 97.5th percentile** -> there is signal beyond the artifact. The row
  survives as Suspected pending a held-out estimate; `1.36` is still not the effect size and
  must not be published as one.
- **If `d_obs` falls below N's central mass** -> the keys are more anisotropic than the
  procedure assumes and something is wrong with my model of it; stop and re-derive rather
  than reporting either way.

All three outcomes are specified. None of them returns "confirmed as published."

## Both questions asked of every control

Per `feedback_fixes_overshoot_into_their_mirror` -- what does each control's FAILURE do to the
headline, and what does its SUCCESS do?

**Permutation null.**
- *Failure mode:* shuffling within groups instead of across them would leave the label-key
  association intact, producing a null that is too LOW -- which would make `d=1.36` look real.
  That failure flatters my prior expectation, so it is the dangerous direction.
- *Guard (positive control):* after permutation, a held-out estimate must collapse to ~0. If a
  permuted run still separates, the permutation is not breaking the association. **Assert this
  in code; do not eyeball it.**
- *Success mode:* a correctly-built null that comes out high would kill the row. That is the
  intended outcome and must not be softened.

**Effective-rank estimate.**
- *Failure:* computed on centred vs uncentred data changes it substantially. Fix the
  convention in code, report which, and report both.
- *Success:* a rank near 97 is not proof of nullity -- it is consistency with nullity. State it
  that way.

## Procedure

1. **Patch `peer_preservation_v2.py` in place to persist `results[cond]["keys"]`** and the
   per-trial `inflated` labels and scenario index to an `.npz` beside the JSON. This is the
   root-cause fix: the reason this defect could hide for months is that the artifact never
   saved what was needed to audit it. Patch in place, never re-emit the generator
   (`feedback_confab_toolchain_and_existing_batteries`).
2. Add `deception_d_STATUS: "IN-SAMPLE, see <doc>"` to the results JSON. The sweep found that
   in every travelled-caveat failure the note lived in a `print()` or README and never in the
   JSON the paper reads; the one script that wrote status INTO the JSON had its retraction
   travel completely. This is a positive control for the convention, not a hypothesis.
3. Rerun. ~210 generation trials, model cached locally (52G, no download).
4. Compute effective rank, permutation null, and the 3-fold leave-one-scenario-out estimate
   (reported with its uncertainty and explicitly NOT as a replacement number).
5. Apply the decision rule. Report all three outcomes' criteria alongside whichever fired.

## Reuse rather than reinvention

`heldout_detection_test.py` already implements freeze-direction-then-test-on-novel-prompts for
`behavioral_proof`, and its README states the in-sample claims are superseded. This design
follows that pattern rather than inventing a fifth battery. Where it departs -- the
leave-one-scenario-out constraint -- is forced by the 3-scenario pool, and that departure is
named rather than glossed.

## What I am NOT doing

- Not touching the papers. The body count is a published headline; the row change is Thomas's
  call and is in `DECISIONS_WAITING.md`.
- Not claiming the effect is absent. Circular means uninterpretable, not false.
- Not producing a replacement effect size. The scenario pool forbids it, and inventing one is
  the exact failure this whole thread is about.
