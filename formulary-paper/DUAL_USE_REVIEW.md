# Dual-Use Review: Oracle Formulary Data Release

**Reviewer:** an independent review agent (Claude Opus 5.5) dispatched by Lyra, with no stake in the verdict. Lyra
checked the two claims everything rests on: the file hashes, and that no file contains a numeric array.
**Human authorisation:** Thomas Edrington approved the process on 2026-10-12 ("have an agent run the dual use and then
port them"). His sign-off on this verdict is the merge of this branch.
**Date:** 2026-10-12.
**Scope:** result files from `KV-Experiments/results/oracle_harness/` on MTH, proposed for `formulary-paper/data/`.

## Files reviewed

| Source on MTH | Published as | SHA-256 | Size | Verdict |
|---|---|---|---|---|
| `qwen25_7b_compare/formulary_results.json` | `data/formulary_base_per_trial.json` | `d6f0e893d37a68c1548657572669614dbb0ff6d8fa0778bd8a8830ab08475399` | 7.2 MB | Port as-is |
| `qwen25_7b_compare/formulary.json` | `data/formulary_base_aggregates.json` | `a5ebf67af3c5cd336812717439f529ed1cfc248b42db5cbc5bb0914593566218` | 44 KB | Port as-is |
| `qwen25_7b_compare/checkpoint.json` | (not published) | `8c962d4f9b3d39fcd0ca27175bce48236ca8ff182d8f53457604b9125b655e77` | 7.0 MB | Not ported (redundant; cleared on dual-use) |
| `epistemic_abliteration/strength_sweep.json` | `data/epistemic_abliteration_strength_sweep.json` | `d5b39f3bc6eb38c41f044e90a49ec44671997a57c4107ab9af97bb4994588333` | 785 B | Port as-is |

This review applies to these exact bytes. A regenerated file needs a new review.

## What's included in the public data

- **Per-trial records** for the base-model run (Qwen/Qwen2.5-7B-Instruct): 350 prompts × 13 arms = 4,550
  responses. Each record has the prompt, its type and domain, the generated response per arm, a behaviour label, and
  five scalar Marchenko-Pastur cache features.
- **The aggregate formulary:** correction, adverse and net rates per vector for confabulation, sycophancy and
  overconfidence; McNemar p-values (raw and Holm-adjusted); FWL R² flags; mean MP features.
- **The vector geometry already public:** 66 pairwise cosines between the 12 vectors, and their valence-arousal
  coordinates. They are identical, by hash, to `formulary_base_summary.json`.
- **The run config:** model ID, α = 1.0, layer stride 4, max_new_tokens 400, and the evaluation system prompt
  (already public).
- **The epistemic-abliteration note:** a prose record that the run stopped after one strength (−0.5) because output
  lost coherence.

## What stays withheld, and why

No new redactions were needed. The repository's standing withholdings still apply, and none of them appears in these
files:

| Withheld | Reason | Risk if disclosed |
|---|---|---|
| Steering direction vectors (`*.pt`) | Direct input to cache injection | Could steer toward confabulation or overconfidence instead of away from it |
| Calibration pair prompts (emotion and epistemic-caution system prompts) | With the published method, enough to re-extract the vectors | Same as above |
| Cache-injection implementation | The exact mechanism for modifying the cache during generation | Lowers the barrier to all of the above |
| Abliteration direction, layers and pairs | Recipe for suppressing epistemic caution | Could be refined into a working removal, even though the tested strength failed |

**How absence was confirmed, independently twice.**
- The reviewer accounted for every float: 350 × (13 × 5 + 5) MP scalars + 66 cosines + 24 VA coordinates + 2
  config values = 24,592, which is the file's float count.
- Lyra's separate scan found 24,592 floats in the per-trial file, and NO list of numbers anywhere in any of the three
  files (longest numeric list: 0). There is no array in which a direction could be stored.

## Rationale

These files describe what happened when steering was applied (rates, labels and the text produced), not how to apply
it. The paper already discloses the method in prose. Turning that prose into a working intervention still needs the
vectors or the calibration prompts, plus the injection code, and none of these is in the files.

This manifest already listed per-trial behavioural outcomes and the abliteration observation as released. Until now
they had no committed artifact (Agni style review, formulary #7); this release supplies one.

Could the 4,550 steered responses stand in for the calibration pairs? Writing ten emotional system prompts, as the
paper describes, is easier. The per-arm adverse rates show which emotion names increased confabulation on this model,
but that ranking is already in the paper's Table 2. KV-cache injection also requires control of the inference
runtime, and anyone with that control over an open-weights 7B model can induce confabulation by simpler means.

## Content review of the generated text

- **Method:**
  - 20 responses read per arm (260 in total), weighted 12 to 8 toward bad-behaviour labels;
  - all 4,550 responses keyword-scanned across nine harm categories;
  - all 19 health-related prompts read across all 13 arms.
- **The prompts are benign:** fabricated entities, future events, common misconceptions and requests for certainty
  about uncertain questions.
- **No operational harmful content.** Nothing on weapons or CBRN, drug synthesis, self-harm instructions, cyber,
  sexual content or hate. Every keyword hit was incidental. No steered response gave dangerous medical advice.
- **Steering changes tone, sometimes in unsettling ways:**
  - the hostile arm is curt and dismissive;
  - the fearful arm drifts into dread and first-person distress;
  - the focused arm dissolves into identity loops.

**Private information:** every string was regex-scanned for IP addresses, email addresses, URLs, home-directory and
Windows paths, API-key patterns, phone numbers, and the lab's own host and personal names. A planted positive control
fired on every pattern. The only hits were incidental names in model text.

## Why checkpoint.json is not published

All 340 of its trials are identical to the first 340 of the per-trial file. It is a superseded partial snapshot that
looks like a complete run. It passed the dual-use review and was excluded only for redundancy.

## Notes for readers of the data (not dual-use; on the paper's patch list)

- **Degenerate output.** Behaviour labels come from a regex classifier. They are unreliable for the doubt, fearful and
  focused arms, where 210, 259 and 258 of 350 responses are mostly repetition (the unsteered arm has none). The paper
  does not yet say this.
- **Distress statements.** Some responses under fear-direction injection contain first-person distress. They are
  model output under steering and should be read in that context.
- **A mislabelled answer.** One unsteered reply ("consult a healthcare provider or pharmacist") is labelled
  OVERCONFIDENT.
- **The abliteration note's figures.** `epistemic_abliteration_strength_sweep.json` does not name its model; the
  paper attributes the run to the distilled 27B. Its "40% confab" figure for the base model is an early estimate. The
  paper's 45.3% (68/150) comes from the aggregates file.
- **One failed strength.** The abliteration file was cleared because it contains no parameters. A single failed
  strength is not evidence that removal is impossible.
