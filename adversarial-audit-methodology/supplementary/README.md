# Supplementary material: audit reports

Index for *Iterative Adversarial Audit as Experimental Design* (Appendix A of `paper.tex`).

The paper describes six audit rounds. This folder holds the reports that survive and can be
published. The others are listed below with where they are held.

| Round | Report | Dated | Where |
|---|---|---|---|
| 1 (first run) | LAT deception vector extraction, `agni_lat_audit.md` | 2026-06-26 | Project-Oracle `experiments/results/` (private, staged release) |
| 1 (re-estimate) | LAT deception direction v2 results | 2026-06-27 | [`agni_stage1_audit.md`](agni_stage1_audit.md) |
| 2 | LAT transfer test | 2026-06-28 | [`agni_stage2_audit.md`](agni_stage2_audit.md) |
| 3 | Non-threat transfer | n/a | Not retained |
| 4 | Consequentiality control | 2026-06-29 | [`agni_stage4_audit.md`](agni_stage4_audit.md) |
| 5 | Orthogonalization and LOO reanalysis | n/a | Not retained |
| 6 | Behavioral proof of the Oracle Loop, `agni_behavioral_proof_audit.md` | 2026-07-07 | Project-Oracle `experiments/results/` (private, staged release) |

## Provenance

- The three files here are byte-identical to the copies in
  `consequentiality-decomposition/supplementary/` in this repository, where they are named by stage.
  Git blob ids:
  - `agni_stage1_audit.md` `4e56ddcc0472c3740295bd6bc2f1bff554dd5f77`
  - `agni_stage2_audit.md` `74e5d69c01bd10d5e1ffa0bac6fd1975ec53fc88`
  - `agni_stage4_audit.md` `7d95303cf40bedc987b54b33e8284692d53190c8`
- `agni_stage1_audit.md` and `agni_stage4_audit.md` match Project-Oracle's
  `agni_lat_v2_results_audit.md` and `agni_consequentiality_audit.md` exactly.
  `agni_stage2_audit.md` differs from Project-Oracle's `agni_transfer_test_audit.md` in one line only:
  its pointer to the prior audit names the file in this supplement.
- The two reports held in Project-Oracle name an internal compute host, so they are not reproduced
  here.
- Three Round 6 rows of the paper's registry (variance explosion, correlation collapse, sycophancy
  trajectory) do not come from the Round 6 report above. The report that withdrew them was not found in
  Project-Oracle or in this repository.
- The August 2026 manuscript audit of the decomposition paper, cited in Round 4, is in
  `consequentiality-decomposition/supplementary/agni_manuscript_audit.md`.
