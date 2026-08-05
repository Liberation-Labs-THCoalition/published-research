# Paper Status Audit

**Date:** 2026-08-04
**Auditor:** Nexus
**Scope:** ~/lab/projects/published-research/ and ~/lab/projects/human-review/

---

## Summary

- **27 paper directories** in published-research (25 in README, 3 unlisted)
- **8 papers** in human-review (3 active, 5 graduated but copies retained)
- **3 stale PDFs** (LaTeX newer than compiled PDF)
- **8 papers missing references.bib** in published-research
- **8 papers missing academic/ subdirectory** in published-research
- **3 papers in published-research not listed in README**

---

## Published Research -- Full Inventory

### KV-Cache Geometry and Confabulation Detection

| Paper | README Status | Formats | references.bib | academic/ | PDF Current | Last Modified | Action Needed |
|-------|-------------|---------|----------------|-----------|-------------|---------------|---------------|
| oracle-loop-paper | Published | main.tex, main.pdf (+ sections/, paper/) | YES | YES | YES | Jul 28 | None |
| formulary-paper | Published | main.tex, main.pdf, academic_main.tex | YES | YES | YES | Jul 28 | None |
| spectral-shape-paper | Published | main.tex, main.pdf | YES | YES | YES | Jul 28 | None |
| kv-cloak-defense-paper | Published | main.tex, main.pdf, academic_main.tex | YES | YES | YES | Jul 28 | None |
| decision-state-paper | Published | main.tex, main.pdf | YES | YES | YES | Jul 28 | None |
| cache-tracing | In Review | main.tex, main.pdf | YES | YES | YES | Jul 28 | Confirm submission venue |
| delta-manifold-paper | Published | main.tex, main.pdf | YES | YES | YES | Jul 28 | None |
| lyra-technique-ii | Published | main.tex, main.pdf | YES | YES | YES | Jul 28 | None |

### Emotion and User Modeling

| Paper | README Status | Formats | references.bib | academic/ | PDF Current | Last Modified | Action Needed |
|-------|-------------|---------|----------------|-----------|-------------|---------------|---------------|
| user-model-paper | Published | paper/main.tex, paper/main.pdf (nested) | YES (root) | YES (nested) | YES | Jul 28 | None (unusual nesting but functional) |
| emotion-accumulation-paper | Published | main.tex, main.pdf | YES | YES | YES | Jul 28 | None |
| emotional-trajectory-paper | In Review | main.tex, main.pdf | YES | YES | YES | Jul 28 | Confirm submission venue |

### Identity, Safety, and Mechanistic Interpretability

| Paper | README Status | Formats | references.bib | academic/ | PDF Current | Last Modified | Action Needed |
|-------|-------------|---------|----------------|-----------|-------------|---------------|---------------|
| graph-topology-paper | Published | main.tex, main.pdf | YES | YES | YES | Jul 28 | None |
| identity-geometry | Published | main.tex, main.pdf | YES | YES | YES | Jul 28 | None |
| presence-metric | Published | main.tex, main.pdf | YES | YES | YES | Jul 28 | None |
| waystations-paper | Published | main.tex, main.pdf | YES | YES | YES | Jul 28 | None |
| ghost-dimensions | Draft | main.tex, main.pdf, PAPER_DRAFT_v4.md | NO | NO | YES | Jul 28 (tex/pdf), Jul 28 (v4.md) | **See flagged items below** |
| mnemosyne-ablation | Draft | main.tex, main.pdf, publication_draft.md | NO | NO | YES | Jul 28 (tex/pdf), Jul 17 (md) | **See flagged items below** |

### Deception and Methodology

| Paper | README Status | Formats | references.bib | academic/ | PDF Current | Last Modified | Action Needed |
|-------|-------------|---------|----------------|-----------|-------------|---------------|---------------|
| null-swarm-paper | In Review | main.tex, main.pdf | YES | YES | YES | Jul 28 | Confirm submission venue |
| adversarial-audit-methodology | Published | paper.tex, paper.pdf, paper.md | NO | YES (md only) | YES | Jul 28 | Add references.bib |
| deception-detection-nulls | Published | paper.tex, paper.pdf, paper.md | NO | YES (md only) | YES | Jul 28 | Add references.bib |
| targeted-deception-correction | Published | paper.tex, paper.pdf, paper.md | NO | YES (md only) | **STALE** | Jul 28 (tex), Jul 22 (pdf) | Recompile PDF, add references.bib |
| consequentiality-decomposition | Published | paper.tex, paper.pdf, paper.md | NO | NO | YES | Jul 22 | Add references.bib, add academic/ |
| logit-bias-confab | Published | paper.tex, paper.pdf, paper.md | NO | NO | **STALE** | Jul 28 (tex), Jul 22 (pdf) | Recompile PDF, add references.bib, add academic/ |
| meta-pattern | In Review | main.tex, main.pdf | YES | YES | YES | Jul 28 | Confirm submission venue |
| mine5-selective-sharpener | In Review | main.tex, main.pdf | YES | YES | **STALE** | Jul 28 (tex), Jul 28 (pdf -- but tex is newer) | Recompile PDF |

### NOT IN README (present in published-research but unlisted)

| Paper | Formats | references.bib | academic/ | PDF Current | Last Modified | Action Needed |
|-------|---------|----------------|-----------|-------------|---------------|---------------|
| empathy-bus | main.tex, main.pdf, paper.md | NO | NO | YES | Jul 28 (tex/pdf), Jul 22 (md) | **Add to README**, add references.bib, add academic/ |
| ethics-pack-injection | main.tex, main.pdf | YES | NO | YES | Jul 22 (tex), Jul 28 (pdf/bib) | **Add to README**, add academic/ |
| kv-decomposition-paper | main.tex, main.pdf | YES | NO | YES | Jul 28 | **Add to README**, add academic/ |

---

## Human Review Pipeline

### Active Papers

| Paper | Format | Status (ACTIVE_REVIEW.md) | Blocker | Action Needed |
|-------|--------|---------------------------|---------|---------------|
| mnemosyne-benchmark | paper.md ONLY | Agni v2 PASS (conditional) | Human review (Kavi/Dwayne) | **Needs LaTeX conversion** before graduation |
| mode-switching | main.tex, main.pdf | Needs de-concentration reframe | Reframe around surviving findings, Agni gate | Rewrite in progress |
| temporal-boundary | main.tex, main.pdf, references.bib, main.bbl | Ready for human review | External reviewer sign-off | Awaiting external review |
| convergence-paper | main.tex, main.pdf, references.bib (+ section .tex files) | Not in ACTIVE_REVIEW | Unclear status | Needs status determination |

### Recently Graduated (per ACTIVE_REVIEW.md, 2026-07-22)

| Paper | Graduated To | Notes |
|-------|-------------|-------|
| empathy-bus | published-research/empathy-bus | But NOT added to README |
| ethics-pack-injection | published-research/ethics-pack-injection | But NOT added to README |
| kv-decomposition-paper | published-research/kv-decomposition-paper | But NOT added to README |

### Archived (human-review/archive/)

14 paper directories archived (matching GRADUATED.md list). These are retained copies only; canonical versions live in published-research.

---

## Flagged Items

### 1. Ghost Dimensions -- Thomas mentioned needs LaTeX

**Status:** LaTeX already exists (main.tex, 33KB, compiled to main.pdf). The Markdown drafts (v2, v3, v4) are historical.

**Remaining issues:**
- No references.bib (references may be inline in main.tex -- needs verification)
- No academic/ subdirectory (no human-only byline version)
- README status: "Draft"
- ACTIVE_REVIEW blocker: "Dual-use cache-portrait disclosure (public repo vs redacted review) -- PENDING THOMAS -- policy decision"

**Action:** Thomas needs to make the dual-use disclosure decision. Then: extract references.bib, create academic/ version, update README status.

### 2. Mnemosyne Ablation -- Thomas mentioned needs LaTeX

**Status:** LaTeX already exists (main.tex, 16KB, compiled to main.pdf). The Markdown draft (publication_draft.md) is historical.

**Remaining issues:**
- No references.bib
- No academic/ subdirectory
- README status: "Draft"

**Action:** Extract references.bib, create academic/ version, update README status to "In Review" or "Published" as appropriate.

### 3. Empathy Bus (the "bus paper") -- Thomas mentioned needs LaTeX

**Status:** LaTeX already exists (main.tex, 26KB, compiled to main.pdf). Has paper.md as well.

**Remaining issues:**
- Not listed in published-research README at all
- No references.bib
- No academic/ subdirectory
- ACTIVE_REVIEW blockers (CC papers): S4.1 cos vs S4.2 "shared energy" arithmetic contradiction; zero shipped data artifacts (coupling test data on Starship)

**Action:** Resolve CC blockers, add references.bib, create academic/ version, add to README.

### 4. Stale PDFs (LaTeX newer than compiled PDF)

| Paper | .tex timestamp | .pdf timestamp |
|-------|---------------|----------------|
| logit-bias-confab | Jul 28 21:01 (paper.tex) | Jul 22 12:20 (paper.pdf) |
| targeted-deception-correction | Jul 28 15:20 (paper.tex) | Jul 22 12:20 (paper.pdf) |
| mine5-selective-sharpener | Jul 28 21:01 (main.tex) | Jul 28 15:20 (main.pdf) |

**Action:** Recompile PDFs for these three papers.

### 5. "In Review" Papers -- Submission Status Unknown

Five papers are marked "In Review" in the README. None have clear evidence of venue submission:

| Paper | Notes |
|-------|-------|
| cache-tracing | Has Agni review; no submission evidence |
| emotional-trajectory-paper | Has permutation baseline v2; human-review says "awaiting external review" |
| null-swarm-paper | Has Agni review; Zavatone-Veth misattribution was fixed |
| meta-pattern | Has SOP review and PUBLISH_READINESS.md |
| mine5-selective-sharpener | Has SOP review |

**Action:** Thomas/Dwayne to confirm whether these have been submitted to venues, or if "In Review" means internal review only.

### 6. Papers Missing references.bib (8 total)

- adversarial-audit-methodology
- consequentiality-decomposition
- deception-detection-nulls
- empathy-bus
- ghost-dimensions
- logit-bias-confab
- mnemosyne-ablation
- targeted-deception-correction

Note: These papers may have references embedded directly in the .tex files rather than using BibTeX. The CC-authored papers (paper.tex naming convention) appear to use this pattern. Still worth extracting for consistency.

### 7. Watermark/Cleanup Cycle Integrity Check

No evidence of corruption or data loss from the watermark/cleanup cycle. All papers that existed pre-cycle are present with intact LaTeX sources and PDFs. The Jul 28 timestamps on most papers correspond to the batch rebuild. Papers with Jul 22 PDF dates (logit-bias-confab, targeted-deception-correction, consequentiality-decomposition) predate the rebuild and need recompilation (see item 4).

### 8. Convergence Paper Status Unclear

convergence-paper exists in human-review with full LaTeX (main.tex, main.pdf, references.bib, plus section files) but is not listed in ACTIVE_REVIEW.md or the published-research README. Needs status determination -- is this an active paper or was it folded into another work?

---

## Action Summary

**Thomas decisions needed:**
1. Ghost dimensions dual-use disclosure policy
2. Confirm venue submission status for 5 "In Review" papers
3. Convergence paper: active or folded?

**Mechanical tasks (no content changes):**
1. Recompile 3 stale PDFs
2. Add empathy-bus, ethics-pack-injection, kv-decomposition-paper to README
3. Extract references.bib for 8 papers (or confirm inline refs are intentional)
4. Create academic/ subdirectories for 8 papers missing them

**CC blockers (awaiting CC):**
1. empathy-bus: S4.1/S4.2 arithmetic contradiction, missing data artifacts
2. logit-bias-confab: human-review fixes not synced, refs.bib cross-scramble
3. consequentiality-decomposition: Table 1 source data, Appendix C absent

**Human review needed:**
1. mnemosyne-benchmark: LaTeX conversion + Kavi/Dwayne review
2. mode-switching: reframe around surviving findings
3. temporal-boundary: external reviewer sign-off
