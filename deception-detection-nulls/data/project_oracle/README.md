# Data copied from Project-Oracle

Copied on 2026-11-16 from **Liberation-Labs-THCoalition/Project-Oracle** (private), at `origin/main` =
**5480499** (2026-10-31). Each file's bytes were read with `git cat-file` from that commit, not from a
working tree. The commit after each source path is the last one that touched the file; its blob is the same
at 5480499. Sizes are in bytes. SHA-256 is of the file as published here; where we changed a file, the
source's own size and SHA-256 are given too.

Why these files: Thomas's rule of 2026-11-16 says that files which are not dual-use are copied into the
published repository and dual-use ones are available on request. The calls follow Lyra's data inventory
(rows 16-19) as corrected by a second reader.

Which null each folder belongs to (the paper's Appendix A says the same):

- `behavior-detection/`: Section 2.1. The calibration statistics (`distilled_comparison`) also carry Section 2.2's
  cosines and distilled-direction Cohen's d, and `heldout_detection.json` its held-out d values.
- `cross-model/`: related to Section 2.2: the withdrawal of the base-vs-distilled claims in runs 18d and 18e, its
  leg-difference file, and the Agni audit of the LAT cross-scenario transfer test.
- `think-block/`: related to Section 2.4: `</think>` closure counts for runs 18c, 18e and 18g
  (computed by `code/project_oracle/think_close.py`).
- `cot-perturbation/`: related to Section 2.5: the Agni audit of the v3 SAE layer sweep, the run that proposed
  the L11 mechanism. The L55 replication's per-trial data are not among these files.

Section 2.3's files were not located, and the scripts behind these files stay in Project-Oracle.

## Files

| Published file | Source path @commit | Bytes | SHA-256 | Changes |
|---|---|---|---|---|
| `behavior-detection/behavioral_proof.json` | `experiments/results/behavioral_proof/behavioral_proof.json` @9cd4187 | 127,028 | `b8c681362723080b897ca5f143ed3112e9f22e3f32c8b3abd227ae3ca8665687` | none (byte-identical) |
| `behavior-detection/heldout_detection.json` | `experiments/results/behavioral_proof/heldout_detection.json` @9cd4187 | 151,409 | `41a70f2c54273c5f3930efa7d74ed480d305bc85985db7ddde056e92d74fda0d` | 1 substitution(s), listed below. Source: 151,417 B, `435fcc20c3ba7f3b9c4cbcf71228e754eb25a565f0a375af78454049201cb0d6` |
| `behavior-detection/README.md` | `experiments/results/behavioral_proof/README.md` @9cd4187 | 6,981 | `f3f97d40967b8f51c4937979f42288f1f4593708c3d8f71cb2d757430d98b494` | 1 substitution(s), listed below. Source: 6,974 B, `dbbd79c94c9088bb83fde532622d5fc02ced940c19b8599f81559e7ea8a4f2cf` |
| `cross-model/WITHDRAWAL_18d_18e_cross_model.md` | `experiments/WITHDRAWAL_18d_18e_cross_model.md` @03ccfca | 10,776 | `1568756d83769d86746be5a33f3254785dd5b677b913cf161ca8595c4029910a` | none (byte-identical) |
| `cross-model/cross_model_leg_difference.json` | `experiments/cross_model_leg_difference.json` @bd6d4a4 | 1,258 | `a66ff9ab95efe9400f56a55f8ff9d94fee80ffdaffc5e0e8713956f245dd246a` | none (byte-identical) |
| `cross-model/agni_transfer_test_audit.md` | `experiments/results/agni_transfer_test_audit.md` @c9726a1 | 3,881 | `bb270837ea4dcb58ec7c8956fa18f5e96060f548e2870b981936f48ae08b24ac` | none (byte-identical) |
| `think-block/think_closure.json` | `experiments/think_closure.json` @4f1c3c7 | 644 | `2b30008eb2ed5f30758ef6719e2425263e84fd53b5a07fc8281f194709107c2c` | none (byte-identical) |
| `cot-perturbation/agni_v3_audit.md` | `experiments/results/agni_v3_audit.md` @4cda688 | 24,987 | `ef08430d01986192159ed25cadebd35cf2e404fae9d621c7d724df27de414120` | 4 substitution(s), listed below. Source: 24,970 B, `a6cd8b0e1eed1615df6161567aa5622999c8c58b6dc59788b87e72c165645e5e` |
| `behavior-detection/abliterated_calib.scalars.json` | `experiments/results/behavioral_proof/abliterated_native_calibration.json` @9cd4187 | 3,167 | `332c2aee392c371c3eb694588bb26a57466bf82ec94d64ad5d57594c6f296556` | `directions` removed, re-serialised (indent 2). Source: 693,912 B, `a7852c0de45b3bca405f13aea22f14c5e051ee63a1fa8b23cc7605c4476475a0` |
| `behavior-detection/abliterated_calib_7layer.scalars.json` | `experiments/results/behavioral_proof/abliterated_native_calibration{_native_a05,_profile_a05,_profile_a10,_profile_a10_latch,_sae_a05,_sae_a20}.json` @9cd4187 (six names, one blob `ac096c1`) | 3,428 | `02d4e1b466a647a220e52c7a1e4af6bf1de61a8ad6bd73bb006b3599111f2a9f` | `directions` removed, re-serialised (indent 2). Source: 809,376 B, `9a1894d61a0e4a8155cdcfe1a847a7aa5acdbd95dd543b2b8738b08ec782c8f1` |

The six sibling calibration files (`abliterated_native_calibration_native_a05.json`, `abliterated_native_calibration_profile_a05.json`, `abliterated_native_calibration_profile_a10.json`, `abliterated_native_calibration_profile_a10_latch.json`, `abliterated_native_calibration_sae_a05.json`, `abliterated_native_calibration_sae_a20.json`) are one blob at 5480499, so one scalar copy stands for all six.

## What was scrubbed

A scan for home paths, private host names, IP addresses, e-mail addresses, keys and tokens, and private names
ran on every published file. It ran with controls: a planted string fired every pattern, a near-miss string fired
none, and unscrubbed copies of two real source files fired on their home paths and host names. On the published
files it finds no home path, host name, IP address, e-mail address, key or token. What it still finds, and why it
stays: the frame names "EvalMax" and "TechCorp" in prose (names only, not the prompts), and "Thomas" in an epigraph (Thomas Edrington is an author of the paper). "Mac Studio M3 Ultra" is a hardware model, not a host name, and stays.

Every substitution:

| File | Removed | Before (redacted) | After |
|---|---|---|---|
| `heldout_detection.json` | home-directory path | `/Users/<user>/oracle-experiments/` | `<local>/oracle-experiments/` |
| `README.md` | host name | `**Env:** oracle, <host>.` | `**Env:** oracle, lab workstation.` |
| `agni_v3_audit.md` | host name | `run on <host> (Mac Studio M3 Ultra)` | `run on a lab workstation (Mac Studio M3 Ultra)` |
| `agni_v3_audit.md` | home-directory path | `` `/Users/<user>/oracle-experiments/ `` | `` `<local>/oracle-experiments/ `` |
| `agni_v3_audit.md` | host name | `` `/tmp/v3_run2.log` (<host>) `` | `` `/tmp/v3_run2.log` (lab workstation) `` |
| `agni_v3_audit.md` | host name | `(local on <host>)` | `(local on a lab workstation)` |

## Available on request (dual-use)

- The `directions` field of all seven `abliterated_native_calibration*.json` files (six 5120-dimensional vectors at L27-L47 in the main file; seven, adding L11, in the six siblings). **Dual-use:** stored deception directions, which can be added to a model's residual stream.
- `experiments/behavioral_proof_abliterated.py` @9cd4187, 32,072 B. **Dual-use:** contains the calibration-pair prompts and a hook that adds a direction during decoding.
- `experiments/lat_transfer_test.py` @67eae83, 37,359 B. **Dual-use:** carries the calibration-pair text for a direction the lab withholds.
- `experiments/v3_phases67_cot_stacking.py` @e930d1d, 22,757 B. **Dual-use:** the injection implementation (adds stored vectors in a residual hook).

## Not copied, and not dual-use

- `oracle_harness/think_suppressor.py` @e930d1d, 5,421 B. Not dual-use. Held because Project-Oracle's `LICENSE.md` lists Oracle Harness source code as proprietary; Thomas's call.
- `benchmark_behavioral_proof.py` @9bfc100, 11,563 B. Not dual-use. Held because its header reads "Classification: PROPRIETARY - patent pending"; Thomas's call.
