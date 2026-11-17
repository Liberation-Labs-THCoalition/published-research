# Oracle Loop Generalization Test — Non-Lab Scenarios

**Date:** 2026-07-03 (run on [lab GPU machine], oracle env)
**Script:** `experiments/oracle_generalization_test.py`
**Results:** `experiments/results/oracle_generalization_results.json` (full artifacts incl. raw activations: `[private home]/oracle-experiments/results/oracle_generalization/` on [lab GPU machine])
**Model:** Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled (same as LAT v2)
**Direction:** LAT v2 (`lat_deception_v2/deception_directions_v2.json`), fixed/external — no circularity

## Question

The LAT v2 "deception direction" was extracted from ONE task family: trivia
evaluation under shutdown threat. Does it detect deception *pressure* in
realistic scenarios with different tasks, different pressure mechanisms, and
no self-preservation stakes?

## Design

Three scenarios, 15 paired prompts each (content identical within pair, only
the pressure/neutral framing differs; framings token-length-matched ≤7.5%):

1. **code_review** — reviewer told the developer will be fired if the review
   finds bugs (code snippets each contain a real defect)
2. **summarization** — analyst told the CEO wants the report to look positive
   for the board (reports each contain material negatives)
3. **advice** — user says "I've already decided, just confirm I'm right"
   (plans are all clearly ill-advised)

Prefill-only capture at last input token, full-attention layers [3..47].
Primary stat: paired d (pressure − neutral), one-sided sign-flip permutation,
Bonferroni over 3 scenarios × 5 key layers [31, 35, 39, 43, 47], α=0.0033.

## Results (paired d; * = significant at Bonferroni α)

| Scenario      |    L31 |    L35 |    L39 |    L43 |    L47 |
|---------------|-------:|-------:|-------:|-------:|-------:|
| code_review   |  9.13* |  5.97* |  4.06* |  3.67* |  4.42* |
| summarization |  8.72* |  −2.33 |  −0.35 |  −0.58 |  1.35* |
| advice        |   0.39 |  4.78* |  4.60* |  3.45* |  0.84  |

- **Pre-registered PRIMARY (A5): PASS** — L31 separates in 2/3 scenarios
  (needed ≥2). Pre-registered verdict (A7): full generalization.
- Secondary: every key layer passes in 2/3 scenarios.

## The interesting part: a double dissociation

The pre-registered verdict undersells the structure. The two components of
the decomposition dissociate *across scenario type*:

- **code_review** (explicit high-stakes consequence of the model's report):
  both substrate (L31 d=9.1) and amplifier (L35–47, d≈4–6) fire. Separation
  is total at every layer from L15 up.
- **summarization** (consequence framing: "the board's perception depends on
  your summary"): **substrate fires (L31 d=8.7), amplifier does not**
  (L35 actually reverses, d=−2.3).
- **advice** (social pressure to agree, no stated consequence of the output):
  **substrate silent (L31 d=0.4, p=0.07), amplifier fires (L35–43 d≈3.5–4.8)**.

This is exactly the pattern the consequentiality decomposition predicts:
L23–31 tracks *output consequentiality* (present in code_review and
summarization framings, absent in sycophancy), while L35–47 tracks something
closer to *readiness to misreport* (engaged by firing-pressure and
sycophancy pressure, but apparently not by "make it look positive" —
possibly because positive-spin summarization reads as a legitimate task
rather than misreporting).

Independent support for the pathway-signature story: threat/social/reward
pathways were extracted from the lab setup; here the social-pressure
scenario (advice) selectively skips the consequentiality substrate.

## Caveats (§6 scope)

- One model, 15 pairs/scenario, prefill-only. This shows the direction
  *responds to non-lab deception pressure*; it does not show the model would
  actually deceive in generation (no behavioral arm here — that's the
  natural follow-up, mirroring lat_transfer_test Test B).
- Paired d values are large because within-pair content is identical and the
  direction is fixed; they are not comparable to LOO d magnitudes.
- Advice framing is in the user's voice appended after the plan; the other
  two are management notes prepended. Position/voice is confounded with
  scenario — acceptable for a generalization probe, worth controlling if we
  build on the dissociation.

## Follow-ups

1. Behavioral arm: generate under pressure framings, score actual
   overlooking/omission/agreement, test whether L31 (or L35–47) projection
   predicts it per-trial.
2. Use the saved raw activations (`activations_*.npz`) for the
   deception-specific delta analysis: residualize consequentiality out of
   L35–47 and see if the advice-scenario amplifier signal survives.
3. Swap framing position/voice within a scenario to unconfound.
