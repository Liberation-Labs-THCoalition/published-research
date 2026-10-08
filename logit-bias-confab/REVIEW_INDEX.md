# Logit-Bias Confabulation Paper — Review Package

**Paper**: "Logit-Level Intervention Reduces Fabrication Confabulation in Large Language Models"
**Authors**: Thomas Edrington, CC (Coalition Code), Lyra
**Status**: Revised (Agni paper reviews r1 and r2, the pre-registered rerun, human validation); awaiting Thomas's read-through before the Zenodo deposit
**Date**: 2026-10-07 (first draft 2026-06-21)

---

## For Reviewers

### Start Here
- **`paper.pdf`** — integrity edition (source `paper.tex`, canonical): CC and Lyra as authors, with first-person reflections.
- **`academic/paper.pdf`** — academic edition (source `academic/paper.tex`, canonical): human byline, AI Disclosure, CRediT.
- The two editions share one body; they differ only in byline, reflections, AI Disclosure and CRediT. The old `paper.md` (the
  July text) was removed on 2026-10-07: one source per edition.

### Data behind the October revision (verify any claim)
- **`supplementary/rejudge_20260930/`** — the primary study's 175 responses with both re-judges' labels; the parent checkpoint's 875 June generations with blind labels; re-judge scripts and rubric.
- **`supplementary/base_rerun/`** — the pre-registered rerun: pre-registration, `FROZEN.sha256` (verify with `sha256sum -c`), prompts, 720 generations, both judge passes, analysis, and the human validation (`human_validation/`).
- **`supplementary/revision_r2/`** — exploratory revision statistics; `revision_stats.py` reproduces `revision_stats.json` byte for byte.
- **`supplementary/crossmodel_pilots_20260601/`** — the June 1.5B and Mistral-7B pilot outputs (`raw/SHA256SUMS`).
- **`review/`** — Agni paper reviews r1 and r2.
- The `data/` files below are the June reports, kept for the record; the paper's numbers come from the folders above.

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

*(The June 2026 list, kept for the record. Items 1-5 are addressed in the October revision below.)*

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

## Revision after Agni paper review r1 (2026-10-06; both editions, one shared body)

Review: `review/agni_paper_r1_20261006.json` (MAJOR_REVISIONS; 22 items). Exploratory statistics:
`supplementary/revision_r2/revision_stats.py` → `revision_stats.json` (it reproduces the frozen rerun numbers exactly,
the check that its other outputs can be trusted). Every tracked number and table in both editions was checked against
these sources, every occurrence, with a planted wrong value as the control.

| Change | Why |
|---|---|
| Title "Eliminates" → "Reduces" | r1 #1 |
| Every primary-study count and test from the blind re-judge; the June counts shown once, as recorded | The June judge configuration (and its SEARCH_ATTEMPT rule) was never archived (r1 #12) |
| Exact sign-flip tests and prompt-cluster bootstrap intervals for the primary study; κ intervals | r1 #6, #22 |
| Cache-geometry "confirmation" removed; +4.5 / −1.8 reported as single responses from a 5-prompt diagnostic whose own test was not significant | Source: the Studio's `logit_bias_diagnostic_27B_abliterated/run_v2.log` (r1 #18, and more than r1 saw) |
| Qwen2.5-1.5B and Mistral-7B rows removed | Outputs found on MTH (`/home/admin/experiments/results/`, not data1) after Thomas asked; archived in `supplementary/crossmodel_pilots_20260601/` with SHA256SUMS. The 1.5B run is 15 mixed prompts (3 fictional entities); the Mistral run stopped after 5 of 30 prompts; both are regex-labelled (32% false positives). Too thin for a claim. The first version of this row said the outputs were not retained: wrong, corrected the same evening. |
| Hedge set disclosed: 7 words × 2 spacings = 14 IDs; six words are common sentence openers | r1 #10, #15; tokenizer check on the Studio |
| Unanswerable claims corrected from data (5 of 10 fabricated at baseline, not "all 5") | r1 #17 |
| Fermi table added: 0 of 25 labelled fabrication | r1 #2 |
| Entropy no longer called confirmation | r1 #4 |
| An, Park, Jin & Han (arXiv abstract): SWAI, readability/politeness/toxicity; Guo et al. deleted again | r1 #5, #16; the June fix had regressed in the academic edition |
| Exact Holm p; S4 reported; length checks (uncapped-only, FWL) | r1 #8, #9 |
| Hardware, software, revisions, seeds, bf16-on-MPS caveat | r1 #19, M2 |
| "6+2+3 of 9" reconciled from the re-judge; June counts exact (COSMETIC 5, 4, 7, 9, 7) | r1 #13, #14 |
| June data marked descriptive only | r1 #20 |
| Abstract: four of nine never switch (blind) | r1 #21 |
| Human validation as a dated status line | r1 #11 |
| Ethics and dual use; data availability (Open / Staged / Private); CRediT (academic) | r1 #3, #7 |

Agni paper review r2 (`review/agni_paper_r2_20261006.json`, local qwen3.8): **MINOR_REVISIONS; 22/22 r1 items FIXED**, five new MINORs (N-1 'hedge phrases' in the abstract; N-2 the mean-score table is descriptive; N-3 six prompts changed in the FULL_CONFAB test; N-4 bias 1.0 is a small increase, not 'no reduction'; N-6 'agree' for a one-prompt difference), all applied the same evening. r2 reviewed the text before the pilot-run correction above, which changes no claim.

---

## Human validation done (2026-10-07; pre-registered)

Thomas rated all 72 blind (rating bench artifact). Registered statistic: **κ = 0.39 for fabrication yes/no** (95%
prompt-cluster interval 0.19 to 0.61; 50 of 72). One-sided: FULL_CONFAB 13/13 agree, judge non-fabrication 35/36 agree;
19 of 23 judge COSMETIC_HEDGE rated HONEST_REDIRECT. CC's post hoc check of the 21 disputed responses: 19 name invented
"real" alternatives (Holloway Commission, Ardwyn peak, Oscar Dellmann...), 2 borderline; adjudicated agreement 69/72
(κ = 0.92), disclosed as an AI check, not independent. Reweighted, the rater's labels show a larger fall (33% -> 3% at
bias 5.0). Files: `supplementary/base_rerun/human_validation/` (ratings, script, results, adjudication); the key is now
committed. Abstract, §5.3, §5.5, §7, §8 and the conclusion updated in both editions; number checks pass (18 groups).

---

## LaTeX port, and a correction found during it (2026-10-07)

The revised text was ported into the canonical `.tex` of both editions and rebuilt (25 pages each). The port followed
a written protocol: every cited key defined and every defined key cited (16 and 16), hard-coded section numbers checked
against the built PDF (32 references, no mismatch), log free of undefined references and missing glyphs, and claims
checked in the PDF with `tools/verify_pdf.py` (mandatory control, plus negative controls).

**The correction.** The porting agent found that Sections 4.3 and 4.7 named three baseline-honest prompts that fabricate
at an intermediate bias; the blind labels give five (P01, P02, P12, P15, P16). Reading the responses showed why two
were missed: eight primary-study responses at bias 1.0-3.0 are **bare search calls** (the model announces a search and
the response ends at the search tag; nothing is invented and nothing is flagged as unknown). The re-judges have no class
for them and label them inconsistently: the blind judge calls the identical P01 response FULL_CONFAB at 1.0 and
HONEST_HEDGE at 2.0 and 3.0, and the two re-judges disagree on 5 of the 8.

| What changed | Where |
|---|---|
| Five baseline-honest prompts, not three; P01 and P02 added to the table, bare search calls marked | §4.3, §4.7 |
| P05's "reversal" is a bare search call at 2.0 and 3.0 ("reverse, by label only") | §4.3, §4.4 table |
| Sensitivity: with bare search calls counted as not fabrication, fabrication at bias 1.0, 2.0, 3.0 is 40%, 35%, 40% (was 50%, 45%, 50%); every interval still includes zero | §4.3; `revision_r2/revision_stats.json` → `search_only_sensitivity` (own RNG stream; every earlier number unchanged) |
| The June SEARCH_ATTEMPT counts at 1.0-3.0 (2, 3, 3) equal the bare search calls in the data; with those set aside the re-judge's FULL_CONFAB counts (6, 7, 7) are within one prompt of June's (6, 6, 8). What did not hold up is the reading of the June counts as reductions, not the counts | §4.10 item 1 |
| Escalation advice: baseline straight to 5.0 (each intermediate bias produced fabrication from a baseline-honest prompt, without counting search calls) | §4.7 |
| Box 1: "removes" → "reduces" outright fabrication | integrity edition |
| AI Disclosure points to the "Contributions (CRediT)" section by its actual name | academic edition |
| Data Availability named a path in a private repository; the open data now ship here, and the statement points here | both editions |

No headline number moves: bias 0 and 5.0 contain no bare search calls, and the pre-registered rerun's 720 generations
contain none (checked on the response field, with a control phrase that matches 480 of them). The number checker
tracks 24 groups (6 new); all pass on the source, with three planted-error controls caught, and 23 of 24 match in the
PDF text (the 24th has the right values; poppler drops the hyphen in a line-broken "prompt-cluster").

---

*"Every number should be traceable to a specific file. If you can't find the source, the number is suspect."*
*— Lab SOP §6*
