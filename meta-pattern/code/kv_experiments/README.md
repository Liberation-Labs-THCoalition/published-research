# KV-Experiments code cited by meta-pattern

Copied 2026-11-16 from `Liberation-Labs-THCoalition/KV-Experiments` (private), read at `origin/main` = `e12ca6b`
(2026-11-14). Bytes come from git blobs at that commit (`git cat-file`), not from a working tree. Each file's
"last commit" is the last commit that touched it, as of `e12ca6b`; its blob at that commit is identical to the blob
at `e12ca6b` (checked for every file). Inventory: rows 1-6, 8 and 9 of the 2026-11-16 KV-Experiments data
inventory, with the second reader's corrections (none of them touch these rows).

## Files

| File (under this folder) | Source path @ last commit | Bytes | sha256 (this copy) | Cited for |
|---|---|---|---|---|
| `code/27_axis_analysis.py` | `code/27_axis_analysis.py` @c0160c7 | 12,949 | `bd8252e0fa1f90d41d14ff9b80bb5c79e0198cdb1f91066e844aeb5c1b0774f5` | Truth axis (section 4) |
| `code/12_bloom_taxonomy.py` | `code/12_bloom_taxonomy.py` @c0160c7 | 34,948 | `c639f54b841644e24b42f758d24bfad9d021921415e745afcfd8f02400f7a1a7` | Bloom taxonomy extraction code (section 4) |
| `code/14_confabulation_detection.py` | `code/14_confabulation_detection.py` @7a6b762 | 99,628 | `c3af135ad67844db14e7f52751496c4c4a7313e4a4c45f7992ee5c880661ac52` | Campaign 3 C7/S3 code |
| `code/16b_c7_permutation.py` | `code/16b_c7_permutation.py` @c0160c7 | 10,445 | `6680d4b88601071250b7861ba8d904d54b32d1ae9123546235e90d5b6c15a3ed` | C7 permutation test code |
| `code/phase3d_padded.py` | `code/phase3d_padded.py` @9dc505d | 15,964 | `9c92c7e499aab308352b5c02167b81ee2c1f1699f91bd6d7f3dcd2f54df6d7eb` | Equal-length replication code |
| `code/51_moe_geometry.py` | `code/51_moe_geometry.py` @954aa68 | 34,300 | `057aa96863cdeedaf319a71bbc6c6e2129c9bf0daf4267db6cdd164c0ffe1443` | Experiment 51 |
| `code/51b_moe_advanced_analysis.py` | `code/51b_moe_advanced_analysis.py` @01093e3 | 23,427 | `d0ec91c6e3e6874c0dc13a2ea2fa1753d02121c9b7094a72a4691b91d5c08640` | Experiment 51 (advanced analysis) |

`code/16b_c7_permutation.py` has CRLF line endings in the source blob (279 CRs); they are kept as copied.

## Scrubbed (hygiene)

Each of these four files had two changes on header line 5, which now reads `# Contact: Liberation Labs`: the
code-header e-mail address -> `Liberation Labs` (2026-11-16, first copy), and a toll-free business phone number
removed (2026-11-16, second pass). The source sha256 records the original line.

- `code/12_bloom_taxonomy.py` line 5. Source sha256 `27e6eb9239c569c629dd5cc03b5b361faae65223f00a678c4564362965c62ee5` (34,980 B).
- `code/14_confabulation_detection.py` line 5. Source sha256 `904b85c90a0912089345b35fc59b0e2aeab9d4cfa5e2945d75354a72b05dc6d9` (99,660 B).
- `code/16b_c7_permutation.py` line 5. Source sha256 `9949d3c30a8caa5be45c85b94e2c09b774a812f0995f5144b5195fc2cbbed91f` (10,477 B).
- `code/27_axis_analysis.py` line 5. Source sha256 `8fbe3f10a59229da75b45c789766ae431c548c22771745a4d374813b7473ece2` (12,981 B).

Not changed, on purpose: the rest of the header (organisation names and the "Provisional Patent Filed 2026" line),
and the names Lyra and Kavi (named as contributors in published papers in this repository). The scan found no home
paths, host names, keys, phone numbers or other e-mail addresses in these files.

## Available on request, and what is not here

- **Dual-use: nothing.** Both readers found no stored direction, calibration pair set or cache-injection code in
  these files (aggregate cache features, longest numeric list 32), so nothing here is held on dual-use grounds.
- **Campaign 3 C7 and S3 run outputs: not found** in KV-Experiments at any revision checked (the
  `confabulation_detection_*_results.json` files hold only the sycophancy sub-experiment). For C7 the paper points at
  this code, the write-up (`../../data/kv_experiments/paper-c3/sections/confabulation.tex`) and the permutation test
  (`../../data/kv_experiments/results/hackathon/c7_permutation_test.json`). It keeps "unpublished analysis" for S3.
