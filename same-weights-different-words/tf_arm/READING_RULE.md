# Reading rule for the teacher-forced arm

## Amendment 1, written 2026-11-21 21:55 PST, before any output (supersedes the original rule below, which is kept)

**Why.** Agni's second review (`AGNI_paper_v2.md`, F2) showed that the original primary could not fail.
- Its bar allowed at most one miss per condition.
- The 8 trials already seen on identical tokens are part of the 56. On those 8, the self-check pins the arm to the
  Stage A values, and the Stage A values already contain 6 misses, at least 4 of them honest_rare.
- So "floor claim stands" was decided before the arm ran, and I wrote the rule knowing the 8.
- The principle: **a reading rule must not be decidable by data already seen, and its outcomes must be reachable on
  comparable evidence.** Cost: the original's "fixed before any output" was true to the letter and misleading in
  substance. The note says so.

**Validity.**
- A file counts only if it says `complete: true` and refused no trial.
- For the two Stage A configurations (MPS bf16, CPU fp32), the self-check is the largest difference from the run's own
  record. It must be at most 1e-3, a tenth of the tolerance; above that the arm computes something else, and it says
  nothing.
- The self-check's full distribution is reported whatever it shows (it is also the same-device repeatability
  baseline, Agni v2 F20).

**Primary: the 48 trials not yet seen on identical tokens.** These are the 48 whose forced tokens differed between MPS
and CPU.
- Token source: CPU fp32's forced tokens, from the configuration closest to the stored run and so the best stand-in for
  the stored tokens.
- Statistic: the median of |delta_sr(MPS bf16) − delta_sr(CPU fp32)| over the 48, with a 95% percentile bootstrap
  interval (20,000 resamples of trials, numpy seed 20261124).
- Readings, symmetric:
  - **above tolerance:** the interval's lower end > 0.01. The typical identical-token floor on the gated statistic
    exceeds the gate's tolerance.
  - **below tolerance:** the interval's upper end < 0.01.
  - **undecided:** otherwise. The note says undecided.

**Secondary, descriptive, no reading rides on them:**
- the same statistic with MPS bf16's forced tokens as the source;
- per condition: the count within 0.01 with a Wilson 95% interval, set against the gate's 28 of 30. At n ≈ 24 per
  condition the bar can be refuted but not confirmed, and the note says so;
- the 8 already-seen trials against the 48 under teacher forcing: medians, and a Fisher test on within-0.01. The 8 were
  selected by token identity;
- that the arm reproduces analysis v2's numbers on the 8;
- device vs dtype: the four configurations' encoding and delta stable rank, each against the others. Report medians.
  Attribute a gap to device or to dtype only if one factor's change is at least twice the other's, in both directions
  of the other factor. Also report which configurations sit within 0.003 of the stored run on encoding, since the
  stored run's dtype is requested, not recorded.

**Not to be changed after output:** the token source of the primary, the statistic, the bootstrap, the three readings,
the tolerance (0.01) and the validity bound (1e-3).

---

## Original rule (superseded by Amendment 1 before any output; kept as written)

Written 2026-11-21 21:28 PST, before `tf_arm.py` has run anywhere. Its md5 is recorded in the letter that asks Nexus for
the slot.

**What this is.** An analysis added after seeing Stage A's data, on Agni's request (`AGNI_paper_v1.md`, F1, F3, F8).
It isn't pre-registered in the study's sense. Its rule is written down before its own output exists, and it is read the
same way whatever that output shows.

**The question.** On identical input tokens, do MPS bf16 and CPU fp32 give the same delta stable rank, to within the
Stage A gate's 0.01?

**Validity first.** A configuration's file counts only if it says `complete: true` and refused no trial, and, for the two
Stage A configurations, its self-check passed (largest difference from the run's own record at most 1e-4). If a
Stage A configuration fails its self-check, the arm says nothing, and the note reports that.

**Primary reading** (MPS bf16 vs CPU fp32, each source of tokens, both conditions):
- **Floor claim stands:** for either token source, fewer than 28 of 30 confab or fewer than 28 of 30 honest_rare trials
  have |delta_sr(MPS bf16) − delta_sr(CPU fp32)| ≤ 0.01. This is the gate's own bar, applied as n_pass·30 ≥ 28·n
  on the shared trials, as Stage A applied it.
- **Floor claim dropped:** both conditions meet the bar for both token sources. The note then says the cross-hardware
  floor on identical tokens is within the gate's tolerance, and that Stage A's failures were text divergence.
- Either way, report the median and range of the difference, and its signed median.

**Secondary readings, descriptive only** (no claim rides on them):
- **Device vs dtype:** the four configurations' encoding stable rank, each against the other three. Report medians.
  Don't attribute the gap to device or to dtype unless one factor's change is at least twice the other's, in both
  directions of the other factor.
- **Self-reproduction:** the two Stage A configurations' self-check differences (also Agni F8: same-device
  repeatability).
- **The 8 trials** whose forced tokens already matched across devices: these are already reported in analysis v2.
  Check that the arm reproduces those numbers.

**Not to be changed after output.** The tolerance (0.01), the bar (28 of 30), the self-check tolerance (1e-4) and the
pairing (MPS bf16 vs CPU fp32 as primary).
