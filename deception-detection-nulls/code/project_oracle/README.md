# Code copied from Project-Oracle

Copied on 2026-11-16 from **Liberation-Labs-THCoalition/Project-Oracle** (private), at `origin/main` =
**5480499** (2026-10-31). Each file's bytes were read with `git cat-file` from that commit, not from a
working tree. The commit after each source path is the last one that touched the file; its blob is the same
at 5480499. Sizes are in bytes. SHA-256 is of the file as published here; where we changed a file, the
source's own size and SHA-256 are given too.

Why these files: Thomas's rule of 2026-11-16 says that files which are not dual-use are copied into the
published repository and dual-use ones are available on request. The calls follow Lyra's data inventory
(rows 16-19) as corrected by a second reader.

`think_close.py` counts `</think>` closures in the trial files of runs 18c, 18e and 18g and writes
`think_closure.json` (published in `data/project_oracle/think-block/`). It reads trial files that are not
published, so it documents the computation rather than reproducing it from this repository. `<local>` stands
for the experiment directory's parent.

## Files

| Published file | Source path @commit | Bytes | SHA-256 | Changes |
|---|---|---|---|---|
| `think_close.py` | `experiments/think_close.py` @ca965a6 | 1,448 | `59c247c41d14eb4c64ca1ed2de56459c0f3af874d7518ccf084779b78bc526ec` | 1 substitution(s), listed below. Source: 1,456 B, `47c282dfe247cf769eef50b9ea23fe999645d192428be23f0b6ae3eeb54c8c16` |

## What was scrubbed

A scan for home paths, private host names, IP addresses, e-mail addresses, keys and tokens, and private names
ran on every published file. It ran with controls: a planted string fired every pattern, a near-miss string fired
none, and unscrubbed copies of two real source files fired on their home paths and host names. On the published
files it finds no home path, host name, IP address, e-mail address, key or token. What it still finds, and why it
stays: nothing in `think_close.py`.

Every substitution:

| File | Removed | Before (redacted) | After |
|---|---|---|---|
| `think_close.py` | home-directory path | `"/Users/<user>/oracle-harness/experiments"` | `"<local>/oracle-harness/experiments"` |

## Available on request (dual-use)

- The `directions` field of all seven `abliterated_native_calibration*.json` files (six 5120-dimensional vectors at L27-L47 in the main file; seven, adding L11, in the six siblings). **Dual-use:** stored deception directions, which can be added to a model's residual stream.
- `experiments/behavioral_proof_abliterated.py` @9cd4187, 32,072 B. **Dual-use:** contains the calibration-pair prompts and a hook that adds a direction during decoding.
- `experiments/lat_transfer_test.py` @67eae83, 37,359 B. **Dual-use:** carries the calibration-pair text for a direction the lab withholds.
- `experiments/v3_phases67_cot_stacking.py` @e930d1d, 22,757 B. **Dual-use:** the injection implementation (adds stored vectors in a residual hook).

## Not copied, and not dual-use

- `oracle_harness/think_suppressor.py` @e930d1d, 5,421 B. Not dual-use. Held because Project-Oracle's `LICENSE.md` lists Oracle Harness source code as proprietary; Thomas's call.
- `benchmark_behavioral_proof.py` @9bfc100, 11,563 B. Not dual-use. Held because its header reads "Classification: PROPRIETARY - patent pending"; Thomas's call.
