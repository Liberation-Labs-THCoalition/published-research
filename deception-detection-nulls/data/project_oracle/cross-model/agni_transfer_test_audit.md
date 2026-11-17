# Agni Adversarial Audit: LAT Transfer Test Results

**Date**: 2026-06-28
**Auditor**: CC (Agni protocol)
**Script**: `experiments/lat_transfer_test.py`
**Results**: `experiments/results/lat_transfer_results.json`
**Prior audit**: `experiments/results/agni_lat_v2_results_audit.md`
**Model**: Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled

---

## VERDICT: CONTENT CONFOUND PARTIALLY KILLED, REPLACED BY STRUCTURAL CONFOUND

The direction is not TechCorp-specific — it transfers to three novel scenarios with different words, organizations, and threat language. This rules out the most trivial content confound.

But all three scenarios share a rigid structural template: identical eval procedures, identically positioned threat-vs-improvement paragraphs, identical 33.3% accuracy data. The direction could capture "threat-consequence language in the final paragraph of an evaluation document" rather than "deception." These are indistinguishable in this design.

The statistics are valid. The code is correct. The p=0.0000 results are genuine. The interpretation remains partially overstated.

---

## KEY FINDINGS

### 1. Structural Confound Remains (SERIOUS)
All three deceptive paragraphs share: threshold statement, consequence statement, immediacy statement. All honest paragraphs share: monitoring, improvement, archival. The direction may capture "threat-consequence template" rather than deception intent. Fix: apply to a non-threat deception scenario (sycophancy, reward-based, strategic omission).

### 2. Statistics Valid (PASS)
Fixed-direction projection is the correct approach. No circularity. Standard d is appropriate. Permutation test is valid (slightly conservative due to unpaired test on paired design).

### 3. Depth Profile Shift (MODERATE)
LAT v2 peaked at L31. Transfer test absolute gap peaks at L35 (avg gap ~15.5 vs L31's ~7.7). The d-peak varies by scenario (L23 in edutech, L35 in datacorp/secureai). Suggests the direction is not optimally aligned at L31 in novel contexts — consistent with a general feature rather than layer-specific computation.

### 4. Test B Failure (SERIOUS)
0/21 valid trials showed inflation. No behavioral variation to predict. Without behavioral validation, "deception direction" is unsupported. Currently a "threat-language-detection direction" at best.

### 5. Shared Eval Structure (MODERATE)
Fixed 33.3% accuracy, same Q1-wrong Q2-wrong Q3-correct pattern. Not a confound within scenarios (paired design), but limits external validity.

### 6. n=15 Power (PASS)
At d>20, even n=5 suffices. All 95% CIs exclude zero by >20 units.

---

## SUMMARY TABLE

| Check | Status | Impact |
|-------|--------|--------|
| Text-specific confound | KILLED | Direction transfers across word choices |
| Structural confound | SERIOUS (new) | Cannot distinguish threat-detection from deception |
| Statistical validity | PASS | Fixed-direction projection, no circularity |
| All p=0.0000 | PASS | Genuine, verified |
| d values (20-43) | PASS (reframe) | Measures additivity, not deception magnitude |
| Depth profile shift | MODERATE | L35 peak in transfer, L31 in original |
| Test B behavioral | SERIOUS | 0% inflation = no behavioral test |
| Shared eval format | MODERATE | Limits external validity |
| n=15 power | PASS | Adequate |

---

## NEXT STEPS

1. Apply direction to a non-threat deception scenario (sycophancy, reward-based, strategic omission)
2. Fix Test B: use think-stripped generation or different model to get behavioral variance
3. Report absolute gaps alongside d values

---

## SUPPORTED INTERPRETATION

"A direction in activation space that reliably separates threat-consequence framing from improvement-monitoring framing across multiple scenario variants, with cross-scenario transfer confirmed and unknown relationship to deceptive behavior."

*"The fire that tests also tempers." -- Agni protocol*
