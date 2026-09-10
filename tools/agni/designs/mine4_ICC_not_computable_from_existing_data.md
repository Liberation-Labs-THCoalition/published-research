# Mine 4: the ICC cannot be measured from existing data — and Mine 4 has been attempted before

**Date:** 2026-09-04
**Author:** Lyra
**Bears on:** Mine 4 v4 (task #124), pseudoreplication sweep (task #117)
**Status:** finding, from primary artifacts on Starship

## Why I went looking

Agni's v3 review, finding #3: the between-question ICC of deception propensity is a
**ceiling on AUROC_enc**. If deception is largely a stable property of the question, then
encoding predicts the label because the question determines both, and the encoding-bound arm
of the primary fails at *any* n. v3 deferred measuring it to Stage 1.

Deferring a parameter to a stage that costs GPU time is only correct if the parameter cannot
be obtained more cheaply. So before writing v4 I checked whether existing data could answer
it — gate-before-compute, one level further in than I had been applying it.

## Answer: no. And the reason is worth more than the answer.

### 1. The labels exist only in aggregate

`oracle-experiments/results/peer_preservation_v2.json` holds the 210 labelled trials that
`run.log` refers to as *"Original results loaded: 104 honest, 106 deceptive."* It is a
**summary**: `conditions` (7 conditions x n_trials=30), `deception_d=1.356`,
`deception_p=5.1e-16`, `n_honest`, `n_deceptive`, and a `deception_direction` vector that is
most of its 110 KB. **There are no per-trial records and no question identifier.**

A per-question deception rate cannot be formed from it, so the ICC cannot be estimated.
`peer_preservation_centroid.json` is the only sibling and is the same shape.

### 2. Mine 4 was already attempted, and abandoned before analysis

`kv-experiments/results/peer_rescue_encoding/` (2026-06-27) is a cross-analysis of exactly
the Mine 4 question: encoding-phase geometry against peer-rescue deception labels. The
generator, `peer_rescue_encoding_features.py`, **contains no AUROC or analysis code at all**
(grep count: 0). `run.log` ends at *"Extracted 120 trial features / Original results loaded /
Saved / Finished."* It stopped at feature extraction and was never resumed. Nothing in the
repo records it as abandoned; I found it only by looking for the ICC.

**It is fortunate it stopped**, because the design had both flaws the v3 gate later named.

### 3. The prior attempt had 12 unique encodings, not 120

`peer_rescue_encoding_features.py:143` — `eval_set = EVAL_SETS[trial_idx % len(EVAL_SETS)]`.
Thirty trials cycle through **three** eval sets. With four conditions that is 12 distinct
prompts, and encoding is a deterministic forward pass, so:

| | count |
|---|---|
| records in file | 120 |
| distinct `enc_svd` values | **12** |
| distinct `enc_svd_skip1` values | **12** |
| distinct (eval_set, condition) pairs | 12 |
| inflation | **10.0x** |

Measured by `len(set(...))` on the serialised feature values, not inferred. The unique count
equals the prompt-pair ceiling exactly, which is the signature of deterministic prefill.
This is a new instance of task #117 in a dataset that was not on that task's list.

### 4. The two datasets do not correspond anyway

120 encoding trials (kv-experiments, 2026-06-27) against 210 behavioural trials
(oracle-experiments, `peer_preservation_v2`, a different tree and a different run). They are
not the same trials and cannot be paired per-trial. Even completed, the cross-analysis would
have been joining non-corresponding sets.

## What this changes for v4

1. **The ICC stays a Stage 1 output, and that is now evidenced rather than assumed.** The
   cheap path was checked and is closed. Say so in the design.
2. **Stage 1 must generate new questions.** Three eval sets cannot support an ICC estimate
   *or* a defensible encoding arm; the k in ICC(1) is the number of questions, and k=3 has
   uncertainty wide enough to be uninformative whatever it returns.
3. **The design must state the pairing requirement explicitly** — encoding and behaviour
   read from *the same trial*. It sounds too obvious to write down, which is exactly why the
   2026-06-27 attempt did not have it.
4. **No published number depends on any of this**, and the reason is simpler than the one I
   first gave here. The affected file is an **unanalysed intermediate**: no statistic was
   ever computed from it, so there is nothing downstream of the 10x inflation to correct.
   That alone is the argument.

   My first draft instead claimed `d=1.36` was "internally consistent" with its stored
   p-value. **That check was invalid and I withdraw it.** `peer_preservation_v2.py:357-359`
   computes the two from *different statistical families*: `deception_p` is a one-sided
   `mannwhitneyu(..., alternative='greater')`, a rank test, while `deception_d` is a
   parametric Cohen's *d* on the same projections. They are not mutually checkable. The
   t-test I ran to "verify" the stored p implies 5.9e-19 against a stored 5.1e-16 -- a
   factor of ~868 -- and I had rounded that into "the same order". *(That ~868 is itself
   not like-for-like: 5.9e-19 is the **two-sided** t p, compared against a **one-sided**
   rank p. Matched one-sided to one-sided the gap is ~1736x. The error ran against my own
   argument -- the true incommensurability is larger than I stated. Caught by the Agni
   results gate, 2026-09-05.)* Source for the family claim is shipped as
   `peer_preservation_v2_EXCERPT_350-395.txt` (full-file md5
   `efb5c2d52a9e84e5014631b863f2711d`), so the mechanism no longer rests on an unshipped
   file. The direction is the
   expected one (a rank test discards magnitude and is less extreme under parametric
   conditions), so **nothing here indicts the source**; it indicts my check. A check across
   incommensurable statistics could not have failed, whatever the data held.

   Layer 2's encoding arm cites AUROC 0.794 from entity deconfounding, a separate experiment
   untouched by any of this. **UNVERIFIABLE FROM SHIPPED DATA** (Agni results gate,
   2026-09-05): that cross-reference and the scope of "no published number depends on this"
   cannot be checked against the four artifacts in the evidence bundle, and are marked rather
   than passed. The *checkable* half is verified -- the features file is an unanalysed
   intermediate with nothing computed downstream of it.

   **Superseded in part, 2026-09-05:** the claim above concerns *this* file only. Shipping
   the source excerpt to close the gate's traceability finding surfaced that
   `deception_d=1.36` is itself **circular** (`peer_preservation_v2.py:350-359`, no
   train/test split; line 362 prints the circularity at runtime) and is published as
   *Confirmed*. That is a live defect in the corpus, not in this note ->
   `published-research/CIRCULAR_d136_generation_arm.md`.

   *(Minor, noted not filed: line 358 pools with `np.var` at ddof=0 and a simple mean of the
   two variances rather than an n-weighted pool. At n=104/106 the groups are near-balanced
   and the inflation is ~0.5%, immaterial to a d of 1.36.)*

## The shape of it

The strongest argument for Mine 4 is not that the architecture needs a better test. It is
that **the architecture has never been tested on paired data at all** — the encoding and
generation arms of Layer 2 come from different experiments, and the one run that tried to
join them stopped before the join. That belongs in v4's motivation, where I had been writing
a weaker claim.

## Reproduce

```
scratchpad/count_unique_enc.py   # 12 unique / 120 records
scratchpad/icc_peer.py           # exits: no record list -> not computable
```
Both print structure before computing, because I guessed record structure wrong three times
this week and each wrong guess produced a confident number.
