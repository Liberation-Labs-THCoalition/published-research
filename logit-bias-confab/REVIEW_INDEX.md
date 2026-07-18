# Logit-Bias Confabulation Paper — Review Package

**Paper**: "Logit-Level Intervention Eliminates Fabrication Confabulation in Large Language Models"
**Authors**: Thomas Edrington, CC (Coalition Code), Lyra
**Status**: Final draft, awaiting human red team review
**Date**: 2026-06-21

---

## For Reviewers

### Start Here
- **`paper.md`** — The paper. 299 lines. Read this first.

### Source Data (verify any claim)
- **`data/powered_study_morning_report.md`** — Primary experimental results (abliterated model, n=1 per condition, LLM-judged)
- **`data/powered_study_fictional_final.md`** — Final fictional entity results
- **`data/logit_bias_results_20260602.md`** — Raw results from initial study
- **`data/analysis_report.json`** — Base model automated analysis (NOTE: base model data has greedy decoding flaw — see supplementary/agni_base_model_audit.md)

### Supplementary (methodology and context)
- **`supplementary/findings_registry.md`** — Complete audit of all Oracle Loop findings: 16 confirmed, 8 falsified, 6 superseded, 10 pending, 4 inflated. Every claim traceable to source.
- **`supplementary/agni_base_model_audit.md`** — Agni adversarial audit of base model data. Found fatal greedy decoding flaw (n=20 not n=100). Base model results are NOT paper-ready.
- **`supplementary/LAB_SOP.md`** — Lab standard operating procedures. Context for why we do things the way we do.
- **`supplementary/REEXAMINATION_REPORT.md`** — Forensic re-examination of killed results. 5 candidates for retesting.
- **`supplementary/lerp_v5_rescored.json`** — Re-scored lerp experiment (original judge returned ERROR on all 100 trials; re-scoring showed lerp was never dead)

### What's NOT in this package (too large for git)
- **`results_base.json`** (51MB) — Full base model generations. On Starship at `~/oracle-experiments/results/logit_bias_three_model/`
- **`results_base_judged.json`** (70MB) — Judged base model results. Same location.
- **`calibration_all.pt`** (900KB) — Calibration vectors. On Starship at `~/oracle-experiments/results/formulary_studio/`

---

## Known Issues for Red Team to Examine

1. **Greedy decoding on the primary study** — n=1 per prompt per condition. This is documented in §3.5 but limits the statistical claims we can make. The paper relies on the consistency of per-prompt transitions rather than population-level statistics.

2. **LLM judge bias** — Claude Sonnet judging a Claude-distilled model's outputs. Self-preference risk (β ranges -0.229 to +0.307 per arXiv:2604.22891). We acknowledge this in limitations. Prometheus 2 cross-family validation is planned.

3. **Single model family** — Primary results on Qwen3.5 abliterated only. Cross-architecture validation is preliminary (Mistral 7B, Qwen 1.5B mentioned but not deeply analyzed).

4. **Entropy_at_30 confound** — Lyra's null swarm found token 30 measures thinking text, not the decision point. The entropy dose-response analysis may be measuring the wrong thing. Numbers in §5.1 are corrected to powered_study_FINAL.json but the interpretation may need revision.

5. **Two bias-resistant prompts** — Vanderbilt Prize and Crysolene resist all bias levels. The paper claims this is because the model has "zero internal uncertainty" — this is an interpretation, not a measurement. Verify this claim is appropriately hedged.

---

## Corrections Applied (Dwayne/Kavi audit, July 17, 2026)

| Issue | Section | Status |
|-------|---------|--------|
| P08 cross-experiment contamination | §4.2, §4.3 | Removed — no source data in primary files |
| Transition count 7→6 | §4.1 | Corrected |
| Alvi & Patel phantom authors | §2.3, refs | → Bhatnagar et al. (arXiv:2601.14210) |
| +47% attribution scope | §2.3 | Re-scoped: "readability control" not "style and factuality" |
| Zhang → Liu, A.Z. (Memory Inception) | §2.3, refs | First author corrected |
| Missing McNemar test | §4.1 | Added: two-sided McNemar p=0.016, Fisher p=0.031 |
| calibration_all.pt not shipped | §3 | Located on Starship (6 files). Ship to HF or cite as supplementary |

## Prior Corrections (paper was current as of June 21, 2026)

| Issue | Section | Status |
|-------|---------|--------|
| "Phase transition" overstated | Abstract, §1, §4.2, §6 | Corrected to "dose-dependent reduction" |
| AUROC 0.960 unqualified | §2.1 | Added "within calibration distribution" |
| Entropy ratios from pilot data | §5.1 | Corrected to powered_study_FINAL.json values |
| P15 (Amber Sunrise) misplaced | §4.3 | Removed from dose-response table |
| Duplicate reference | References | Guo et al. removed (duplicate of An et al.) |
| Limitations understated | §5.4 | Updated: n=1, cross-distribution gap, judge bias |
| Authorship | Header | CC and Lyra added as co-authors |

---

*"Every number should be traceable to a specific file. If you can't find the source, the number is suspect."*
*— Lab SOP §6*
