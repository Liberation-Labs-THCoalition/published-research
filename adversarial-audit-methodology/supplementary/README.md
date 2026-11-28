# Supplementary material: audit reports

Index for *Iterative Adversarial Audit as Experimental Design* (Appendix A of `paper.tex`).

The paper describes six audit rounds. This folder holds every report that survives: five reports
covering Rounds 1, 2, 4 and 6 (Round 1 has two). The Round 3 and Round 5 reports were not retained.

| Round | Report | Dated | Where |
|---|---|---|---|
| 1 (first run) | LAT deception vector extraction | 2026-06-26 | [`agni_lat_audit.md`](agni_lat_audit.md) |
| 1 (re-estimate) | LAT deception direction v2 results | 2026-06-27 | [`agni_stage1_audit.md`](agni_stage1_audit.md) |
| 2 | LAT transfer test | 2026-06-28 | [`agni_stage2_audit.md`](agni_stage2_audit.md) |
| 3 | Non-threat transfer | n/a | Not retained |
| 4 | Consequentiality control | 2026-06-29 | [`agni_stage4_audit.md`](agni_stage4_audit.md) |
| 5 | Orthogonalization and LOO reanalysis | n/a | Not retained |
| 6 | Behavioral proof of the Oracle Loop | 2026-07-07 | [`agni_behavioral_proof_audit.md`](agni_behavioral_proof_audit.md) |

## Provenance

- The three stage-named files are byte-identical to the copies in
  `consequentiality-decomposition/supplementary/` in this repository, where they are named by stage.
  Git blob ids:
  - `agni_stage1_audit.md` `4e56ddcc0472c3740295bd6bc2f1bff554dd5f77`
  - `agni_stage2_audit.md` `74e5d69c01bd10d5e1ffa0bac6fd1975ec53fc88`
  - `agni_stage4_audit.md` `7d95303cf40bedc987b54b33e8284692d53190c8`
- `agni_stage1_audit.md` and `agni_stage4_audit.md` match Project-Oracle's
  `agni_lat_v2_results_audit.md` and `agni_consequentiality_audit.md` exactly.
  `agni_stage2_audit.md` differs from Project-Oracle's `agni_transfer_test_audit.md` in one line only:
  its pointer to the prior audit names the file in this supplement.
- Three Round 6 rows of the paper's registry (variance explosion, correlation collapse, sycophancy
  trajectory) do not come from the Round 6 report above. The report that withdrew them was not found in
  Project-Oracle or in this repository.
- The August 2026 manuscript audit of the decomposition paper, cited in Round 4, is in
  `consequentiality-decomposition/supplementary/agni_manuscript_audit.md`.

## Copied from Project-Oracle (2026-11-16)

The first Round 1 report and the Round 6 report were held back from earlier versions of this folder
only because each names an internal compute host. They are now copied, with that name replaced.

**Source:** `Liberation-Labs-THCoalition/Project-Oracle` (private, staged release), commit
`4cda68835ab39be08a47513e137048f955480f17` (2026-10-08, "Track the Agni audit reports under
experiments/results"). The blobs are unchanged at `origin/main` `5480499`. Bytes were read with
`git cat-file blob`, not from a working tree.

| File here | Source path @ commit | Source bytes | Source sha256 | Published bytes | Published sha256 |
|---|---|---|---|---|---|
| `agni_lat_audit.md` | `experiments/results/agni_lat_audit.md` @ `4cda688` | 10,065 | `7b05bf28633a29bc3660bf6e11a583262a6dca36f1d41b0f74f7d57cd00c3b8c` | 10,095 | `3874dba582732cec06212a179423a1a42a14d5fac88a6826ad00aa46746e3958` |
| `agni_behavioral_proof_audit.md` | `experiments/results/agni_behavioral_proof_audit.md` @ `4cda688` | 21,399 | `be860f25413f9040f9cf211c3385be1438e1569c9f958d96ac61c905c3934e28` | 21,429 | `1234cab76e053bff747b7ceb898463db172f6749e381187dcc859e44ccc8f19c` |

**What was scrubbed** (the only change; each published file differs from its source by +30 bytes).
Each source file named a private compute host, by its machine name, twice. All four mentions were
replaced with `[internal compute host]`. The name itself is not repeated here; below it is written
`<host>`.

| File | Line | Before | After |
|---|---|---|---|
| `agni_lat_audit.md` | 6 | `` `deception_directions.json` on <host> `` | `` `deception_directions.json` on [internal compute host] `` |
| `agni_lat_audit.md` | 141 | `Two models loaded simultaneously on <host>` | `Two models loaded simultaneously on [internal compute host]` |
| `agni_behavioral_proof_audit.md` | 6 | `` <host> `results/behavioral_proof/` `` | `` [internal compute host] `results/behavioral_proof/` `` |
| `agni_behavioral_proof_audit.md` | 10 | `read from the source JSONs on <host>` | `read from the source JSONs on [internal compute host]` |

The hygiene scan also looked for other private host names, IP addresses, `localhost` ports, home
paths, e-mail addresses, keys and tokens, and private names that are not public authors. It found
nothing else in either file. Controls: a planted string fired on all seven pattern groups, a near-miss
string fired on none, and the host name appended to each scrubbed file was caught. The auditor
credited in both reports, CC, is an author of this paper.

**Available on request, not copied** (dual-use: it stores six 5120-dimensional direction vectors, and
stored directions are the first item the lab withholds): `abliterated_native_calibration.json`, which
the Round 6 report reads, in Project-Oracle `experiments/results/behavioral_proof/`. The report quotes
only scalars from it. `deception_directions.json`, which the first Round 1 report names, is not in
Project-Oracle at `origin/main` and was not located; the report contains no vectors from it.
