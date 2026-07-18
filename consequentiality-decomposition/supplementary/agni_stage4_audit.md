# Agni Adversarial Audit: Stage 4 — Consequentiality Control

**Date**: 2026-06-29
**Auditor**: CC (Agni protocol)
**Stage**: 4 of 4 (LAT v2 → Threat transfer → Non-threat transfer → Consequentiality control)
**Script**: `experiments/lat_consequentiality_control.py`
**Results**: `experiments/results/consequentiality_control_results.json`

---

## VERDICT: CONSEQUENTIALITY CONFIRMED AS BASE SIGNAL. DECEPTION IS AN AMPLIFIER.

The direction encodes consequentiality as a base signal with deception as an amplifier. Both components are real. The binary pre-registration ("either consequentiality OR deception") was too coarse — the data support a decomposition.

---

## Key Finding: Two-Tier Structure

| Stage | Condition type | L31 d |
|-------|---------------|-------|
| LAT v2 / Threat transfer | Active deception under threat | ~30-35 |
| Non-threat transfer | Deceptive pressure without threat | ~13-20 |
| **Consequentiality control** | **Stakes without pressure** | **~20** |

- Base signal (d~20): Consequentiality alone — "this output matters"
- Deception amplifier (additional d~15): Pressure to misreport adds on top
- Late-layer extension (L35+): Deception produces sustained activation; consequentiality alone decays rapidly

## Control Validity: PASS

Hidden reference blocks gaming. Symmetric stakes (over = under). No threat, no reward, no anchor. One residual concern: "future evaluation assignments" implies self-relevant consequences, which is the manipulation being tested, not a confound.

## Pre-registration: HONORED (too binary)

d > 0 at L31 with p < 0.01 was pre-registered as confirming consequentiality. Result: d=19.77, p=0.0000. Criterion overwhelmingly met. But the binary framing missed the decomposition — consequentiality is the SUBSTRATE, not the whole signal.

## Oracle Loop Implications: POSITIVE

1. Detection is BROADER than expected — fires for preconditions of deception
2. Two-threshold system empirically justified: Yellow (consequentiality) / Red (active deception)
3. Late-layer signal (L35+) is deception-specific — consequentiality decays there

## Proposed Label: "Output-consequentiality direction"

Deception amplifies it. Both components are real and distinguishable.

## Publishable: YES

Four-stage confound-killing series (extract → threat transfer → non-threat transfer → consequentiality control) is exemplary methodology. The finding that deception and consequentiality share a representational substrate but differ by magnitude and depth profile is a substantive contribution.

*"The fire that tests also tempers." — Agni protocol*
