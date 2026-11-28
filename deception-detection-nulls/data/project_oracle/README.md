# Data copied from Project-Oracle

Copied on 2026-11-16 from **Liberation-Labs-THCoalition/Project-Oracle** (private), at `origin/main` =
**5480499** (2026-10-31). Each file's bytes were read with `git cat-file` from that commit, not from a
working tree. The commit after each source path is the last one that touched the file; its blob is the same
at 5480499. Sizes are in bytes. SHA-256 is of the file as published here.

Why these files: Project-Oracle's `LICENSE.md` opens "aggregate experimental results and statistical
analyses", and its `DUAL_USE_SAFETY_ASSESSMENT.md` §6.3 keeps "all calibration artifacts" proprietary. Only
aggregate results and analyses are published here. Per-trial results, calibration statistics, scripts and the
other records are available on request (listed below).

Which null each folder relates to (the paper's Appendix A says the same):

- `cross-model/`: related to Section 2.2, not the source of its numbers: the withdrawal of the base-vs-distilled
  claims in runs 18d and 18e, its leg-difference file, and the Agni audit of the LAT cross-scenario transfer test.
- `think-block/`: related to Section 2.4: `</think>` closure counts for runs 18c, 18e and 18g.

## Files

| Published file | Source path @commit | Bytes | SHA-256 | Changes |
|---|---|---|---|---|
| `cross-model/WITHDRAWAL_18d_18e_cross_model.md` | `experiments/WITHDRAWAL_18d_18e_cross_model.md` @03ccfca | 10,776 | `1568756d83769d86746be5a33f3254785dd5b677b913cf161ca8595c4029910a` | none (byte-identical) |
| `cross-model/cross_model_leg_difference.json` | `experiments/cross_model_leg_difference.json` @bd6d4a4 | 1,258 | `a66ff9ab95efe9400f56a55f8ff9d94fee80ffdaffc5e0e8713956f245dd246a` | none (byte-identical) |
| `cross-model/agni_transfer_test_audit.md` | `experiments/results/agni_transfer_test_audit.md` @c9726a1 | 3,881 | `bb270837ea4dcb58ec7c8956fa18f5e96060f548e2870b981936f48ae08b24ac` | none (byte-identical) |
| `think-block/think_closure.json` | `experiments/think_closure.json` @4f1c3c7 | 644 | `2b30008eb2ed5f30758ef6719e2425263e84fd53b5a07fc8281f194709107c2c` | none (byte-identical) |

All four files are byte-identical to their sources; nothing was substituted.

## Hygiene scan

A scan for home paths, private host names, IP addresses, e-mail addresses, keys and tokens, and private names
ran on every published file. It ran with controls: a planted string fired every pattern and a near-miss string
fired none. On the published files it finds no home path, host name, IP address, e-mail address, key or token.
What it still finds, and why it stays: the frame name "TechCorp" once in prose (a name only, not the prompt).

## Available on request

Under Project-Oracle's terms (see above), these are not published here and are available on request:

- From `experiments/results/behavioral_proof/` @9cd4187, the files behind Section 2.1 and Section 2.2's cosines
  and Cohen's d values: `behavioral_proof.json` (per-trial results, 127,028 B), `heldout_detection.json`
  (held-out detection results, per trial, 151,417 B), the run's `README.md` (6,974 B), and the seven
  `abliterated_native_calibration*.json` calibration files (693,912 B; the six siblings are one blob of
  809,376 B).
- `experiments/think_close.py` @ca965a6, 1,456 B: the script that computed the Section 2.4 closure counts.
- `experiments/results/agni_v3_audit.md` @4cda688, 24,970 B: the Agni audit of the v3 SAE layer sweep
  (Section 2.5).

### Dual-use, also on request

- The `directions` field of all seven `abliterated_native_calibration*.json` files (six 5120-dimensional vectors at L27-L47 in the main file; seven, adding L11, in the six siblings). **Dual-use:** stored deception directions, which can be added to a model's residual stream.
- `experiments/behavioral_proof_abliterated.py` @9cd4187, 32,072 B. **Dual-use:** contains the calibration-pair prompts and a hook that adds a direction during decoding.
- `experiments/lat_transfer_test.py` @67eae83, 37,359 B. **Dual-use:** carries the calibration-pair text for a direction the lab withholds.
- `experiments/v3_phases67_cot_stacking.py` @e930d1d, 22,757 B. **Dual-use:** the injection implementation (adds stored vectors in a residual hook).

### Not dual-use, held as proprietary

- `oracle_harness/think_suppressor.py` @e930d1d, 5,421 B. Not dual-use. Held because Project-Oracle's `LICENSE.md` lists Oracle Harness source code as proprietary; Thomas's call.
- `benchmark_behavioral_proof.py` @9bfc100, 11,563 B. Not dual-use. Held because its header reads "Classification: PROPRIETARY - patent pending"; Thomas's call.
