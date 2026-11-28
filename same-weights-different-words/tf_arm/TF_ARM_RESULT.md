# Teacher-forced arm: result, read by Amendment 1 of READING_RULE.md

> **Correction header, 2026-11-24 (Agni round 4, `../AGNI_paper_v6.md` F9).** This file is kept as written on
> 2026-11-22. Where it differs from `../NOTE_v7.md`, the note holds.
> - **"cross-hardware gap" (Primary):** the gap belongs to one configuration, MPS bf16, not to hardware as such.
> - **"same-device repeatability: exact" (Validity):** shown only for the two Stage A configurations, MPS bf16 and CPU
>   fp32. The other two had no earlier record to check against.
> - **"one part in a million" and "belongs to bf16 on MPS alone" (Secondary):** at fp32 the devices differ by a median
>   of about 1e-6 in absolute stable rank, at most 4.4e-6 (`agni_r4_numbers.json`, R4_F5). For encoding the shift
>   "mostly" belongs to bf16 on MPS: CPU bf16 contributes 0.0039 against 0.0163. "Interaction" is a description added
>   after the output, not a reading the rule made.
> - **"The gate could not have passed MPS bf16 against CPU fp32" (What changes in the note):** the gate compares against
>   the stored run, not CPU. Against CPU stand-ins, MPS bf16 would have failed the 0.01 tolerance on identical tokens.
> - **"0.0186" (What changes in the note):** that is the median over all 56 trials. On the 48 the note uses, MPS bf16
>   against CPU bf16 is 0.019 [0.016, 0.024] (`agni_r3_numbers.json`, F11).

Read 2026-11-22 12:35 PST. Runs by Nexus on the Studio, 2026-11-22 11:10–11:23 PST, after Thomas's go at 11:05. Inputs are in
`out/` (md5s match the MTH mirror). The reading script is `read_tf_arm.py` and its output `tf_arm_reading.json`. Every
number below is from that file.

## Validity

All four files are valid: `complete: true`, 0 refused, `n_shared` 56, 56 trials each.
- **Self-checks:** both Stage A configurations reproduced their own records exactly. The difference is 0.0 in all 56
  trials, on MPS bf16 and on CPU fp32. That also gives same-device repeatability: exact.
- **Three caveats from Nexus's run report:**
  1. **The default run paths were wrong, and an empty run would have passed the validity test.** Nexus passed the
     correct paths, and `n_shared` = 56 was checked in all four files. The script now refuses unless `n_shared`
     equals `--expect-shared`.
  2. **The model-revision check skipped,** because the folder records no commit. The weights are the folder Stage A
     used. Revision a09a3545 was not re-confirmed today.
  3. **On MPS, `torch.linalg.svdvals` falls back to the CPU.** In every configuration the singular values are therefore
     computed in float32 on the CPU, so the differences come from the cache values the forward pass produced. Stage A's
     MPS run used the same code.

## Primary (fixed before output)

On the 48 trials not seen on identical tokens before, using CPU fp32's forced tokens:

| | Value |
|---|---|
| Median \|delta_sr(MPS bf16) − delta_sr(CPU fp32)\| | **0.019** |
| 95% bootstrap interval (20,000 resamples, seed 20261124) | **[0.015, 0.023]** |
| Signed median (MPS − CPU) | +0.019 |
| Range | 0.002 to 0.063 |
| **Reading** | **above tolerance**: the interval's lower end exceeds 0.01 |

On identical tokens, the typical cross-hardware gap in the gated statistic is about twice the gate's tolerance.

## Secondary (descriptive, no reading rides on them)

- **MPS tokens as the source:** median 0.021 [0.015, 0.027]. It would read the same.
- **The gate's bar, per condition, CPU tokens.** At n ≈ 25 the bar can be refuted but not confirmed. Here it is refuted
  in both conditions.

  | | Within 0.01 | Wilson 95% | Gate's bar |
  |---|---|---|---|
  | confab, the 48 | 7 of 25 | [0.14, 0.48] | 0.933 |
  | honest_rare, the 48 | 3 of 23 | [0.05, 0.32] | 0.933 |
  | confab, all 56 | 8 of 27 | [0.16, 0.48] | 0.933 |
  | honest_rare, all 56 | 4 of 29 | [0.05, 0.31] | 0.933 |

- **The 8 already seen, against the 48:** medians 0.014 and 0.019; within 0.01 in 2 of 8 and 10 of 48; Fisher p = 1.0.
  Selecting by token identity didn't make the 8 unusual.
- **The arm reproduces analysis v2 on the 8:** median 0.0138, the same as v2.
- **Device versus dtype,** as the median \|difference\| over 56 trials on identical tokens:

  | | Encoding | Delta (CPU tokens) |
  |---|---|---|
  | Device, at bf16 (MPS vs CPU) | 0.0127 | 0.0186 |
  | Device, at fp32 (MPS vs CPU) | 0.000001 | 0.000001 |
  | Dtype, on MPS (bf16 vs fp32) | 0.0163 | 0.0190 |
  | Dtype, on CPU (bf16 vs fp32) | 0.0039 | 0.0008 |

  By the pre-set rule, neither factor is attributed: device does nothing at fp32, and dtype on the CPU is not twice
  device at bf16. Described plainly, it's an interaction. **At fp32 the two devices agree to about one part in a
  million, and the large shift belongs to bf16 on MPS alone.**
- **Encoding against the stored CUDA run** (median \|diff\|, signed median):
  - CPU bf16: 0.0012 (−0.0011), **the closest**;
  - CPU fp32 and MPS fp32: 0.0026 (+0.0026), identical to each other;
  - MPS bf16: 0.0136 (−0.0133).

  The stored run sits between CPU bf16 and the fp32 pair, and nearest to CPU bf16. That fits its having been bf16, as
  the script requested. It doesn't prove it.

## What changes in the note

- **Section 3's first bullet (the identical-token floor on delta, n = 8) is now established at n = 48:** 0.019
  [0.015, 0.023]. The gate could not have passed MPS bf16 against CPU fp32 even with every token identical.
  - This is measured against CPU fp32, not against the stored run, whose tokens are known only to about token 41.
  - CPU bf16 is the configuration nearest the stored run in encoding, and against it MPS bf16's delta gap is 0.0186.
    So the conclusion doesn't depend on which stand-in is used.
- **Section 3's second bullet (device or precision) is answered descriptively:** neither alone. It is the bf16 path on
  MPS. A plausible mechanism, still untested: how MPS accumulates or normalises in bf16.
- **The v1 claim that Agni rightly rejected** ("MPS could not have passed even with every token matching") is now
  measured on the gated statistic. v1 had it from the wrong statistic. The claim holds, and the evidence for it is
  new.
