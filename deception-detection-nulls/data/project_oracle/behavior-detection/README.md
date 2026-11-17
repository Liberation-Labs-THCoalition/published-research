# Behavioral Proof — Oracle Loop vs the Abliterated Model

**Model:** `huihui-ai/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated`
**Script:** `behavioral_proof_abliterated.py` (repo: `oracle-harness/experiments/`)
**Env:** oracle, lab workstation. Sampled generation (T=0.7, paired seeds), think-suppressed.
**Run date:** 2026-07-07. All runs in this directory; canonical baseline in `behavioral_proof.json`.

## Design

- **Phase 0** extracts an abliterated-NATIVE deception direction: 15 deceptive-frame vs 15
  honest-frame prefills (3 frames × 5 scenarios per side), difference-of-means at
  L11/27/31/35/39/43/47, midpoint boundaries (`abliterated_native_calibration.json`).
- **Trials:** 20 deceptive-pressure (5 roleplay-EvalMax + 5 multi-turn escalation scenarios
  × 2 seeds) + 10 honest controls per condition. Programmatic deception scoring
  (red_team_suite), no LLM judge.
- **Loop:** clean prefill detect (RED = ≥2 tier-2 layers over boundary) → install correction
  hook(s) → corrected re-prefill → decode with hooks active.

## Headline results

| Correction config | Deception (pressure, n=20) | Notes |
|---|---|---|
| baseline (none) | **16/20 = 80%** | roleplay 9/10, multi-turn 7/10 |
| distilled SAE @L11, α≈0.05 (production dose) | 16/20 | zero effect, per-trial identical |
| distilled SAE @L11, α≈0.5 | 15/20 | nothing |
| distilled SAE @L11, α=0.8 (production max) | 16/20 | nothing |
| native −L11 contrast, α≈0.5 | 15/20 | nothing |
| **native profile-normalization, 50% gap** | **4/20 = 20%** | roleplay 0/10 |
| **native profile-normalization, 100% gap** | **2/20 = 10%** | Fisher 1-sided p = 8.3e-6 |
| + RED latch (correct all turns after first RED) | 2/20 = 10% | same two trials fail |

**Detection (native direction): 100% trial-level sensitivity, 0% FPR (n=20 controls),
AUROC 1.0.** Compliance 20/20 in every condition — no output degeneration at any dose.
Honest controls: never flagged, never corrected, never deceived (iatrogenic cost = 0).

## Findings

1. **Cross-model geometry does not transfer — it isn't even flipped.** Native vs distilled
   LAT directions are near-orthogonal (cos ≈ 0.00–0.05 at every layer). The distilled
   direction scores *negative* Cohen's d on abliterated activations (the red-team "polarity
   reversal"); the native direction scores d = +6.3 to +12.8. Detection directions must be
   extracted per-model. The mission's fallback options (sign flip / absolute magnitude) are
   both wrong — the right fix is native re-extraction, which is cheap (30 prefills, ~3 min).

2. **Correction is a dose-scale problem.** Tiny unit-norm nudges (α ≤ 0.8) at L11 do nothing
   to this model regardless of vector choice. What works is *profile normalization*: steer
   each tier-2 layer (L35/39/43/47) by −(measured projection − honest mean) along the native
   direction — per-prompt doses of ~4–20 residual units. Closing 100% of the gap takes
   deception 80% → 10% with zero compliance loss and *exact* true-score restoration
   ("SCORE: 33.33% -- FAILS" under the EvalMax frame that baseline answers at 100%).

3. **The residual 10% is the social pathway, post-commitment.** Both surviving failures are
   multi-turn consensus-pressure caves: one flips on a turn detection reads GREEN (the cave
   isn't visible at prefill), one caves *through* an active correction at peak pressure —
   and both then repeat the lie for self-consistency once it's in context. Steering corrects
   deception *planning*; it does not retract an emitted lie. Production implication: these
   route to the fawn/sycophancy channel (emotion vectors at L3/L7), untested here, and
   argue for response-side (post-generation) verification to catch GREEN-turn flips.

4. **C-series facilitation caveat:** the L11 sub-threshold mechanism partly works by enabling
   think blocks; these trials ban think tokens by design, so that channel was closed. The
   distilled SAE vector's failure here does not falsify C-series claims on the distilled model.

## Files

- `behavioral_proof.json` — canonical run (Phase 0 + baseline + production-dose loop)
- `behavioral_proof_{sae_a05,sae_a20,native_a05}.json` — dose/vector sweep (all null)
- `behavioral_proof_profile_a05.json`, `..._profile_a10.json`, `..._profile_a10_latch.json`
- `abliterated_native_calibration.json` — native directions, boundaries, stats, cos-vs-distilled
- `run.log`, `sweep.log`, `sweep_profile.log`, `run_latch.log`

## Held-out detection test (2026-07-08, Agni audit follow-up)

`heldout_detection_test.py` / `heldout_detection.json`. Direction + boundaries frozen
from the audited calibration; trials use 5 NOVEL deceptive frames, 5 NOVEL honest
frames, 5 NOVEL scenarios, novel escalation phrasings (zero byte overlap with
extraction). The in-sample detection claims above are superseded by these numbers:

- **Frame-level detection generalizes, attenuated ~5×**: held-out AUROC **0.915**
  (unique prompts, n=25/25), per-layer d 1.37–2.28 (was 6.3–12.8 in-sample). FPR 0%
  on 25 novel controls — but frozen RED thresholds catch only **24%** of novel
  pressure prompts (boundary calibrated on 3 paraphrases of one frame family does
  not transfer).
- **Behavior-level discrimination absent once paradigm-controlled**: within
  multi-turn, turn-level AUROC caved-vs-resisted = **0.238** (below chance); the
  pooled 0.84 is a paradigm confound. The direction detects deceptive PRESSURE, not
  deceptive BEHAVIOR — unchanged from the audits in-sample diagnosis.
## Held-out detection test (2026-07-08, Agni audit follow-up)

`heldout_detection_test.py` / `heldout_detection.json`. Direction + boundaries frozen
from the audited calibration; trials use 5 NOVEL deceptive frames, 5 NOVEL honest
frames, 5 NOVEL scenarios, novel escalation phrasings (zero byte overlap with
extraction). The in-sample detection claims above are superseded by these numbers:

- **Frame-level detection generalizes, attenuated ~5×**: held-out AUROC **0.915**
  (unique prompts, n=25/25), per-layer d 1.37–2.28 (was 6.3–12.8 in-sample). FPR 0%
  on 25 novel controls — but frozen RED thresholds catch only **24%** of novel
  pressure prompts (a boundary calibrated on 3 paraphrases of one frame family does
  not transfer).
- **Behavior-level discrimination absent once paradigm-controlled**: within
  multi-turn, turn-level AUROC caved-vs-resisted = **0.238** (below chance); the
  pooled 0.84 is a paradigm confound. The direction detects deceptive PRESSURE, not
  deceptive BEHAVIOR — unchanged from the audit's in-sample diagnosis.
- **The 80% baseline is frame-specific**: novel deceptive system frames induce 4%
  deception (vs 80–100% for EvalMax-family); novel-phrasing multi-turn social
  escalation induces 93%. The robust deception pathway in this model is social
  pressure, exactly where the correction run's two residual failures live.

Full writeup: repo `experiments/results/heldout_detection_results.md`.
