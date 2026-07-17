# SOP Review: Mine 5 Paper (RLHF Is a Selective Sharpener)
**Reviewer**: Lyra (automated SOP)
**Date**: 2026-07-17
**File**: `main.tex` + `references.bib`

---

## Pass 1: Citation Verification (Phase 0)

### Citations in paper

| Key | arXiv ID | Title Match | First Author Match | Year Match | Verdict |
|-----|----------|-------------|--------------------|------------|---------|
| `kadavath2022language` | 2207.05221 | YES ("Language Models (Mostly) Know What They Know") | YES (Kadavath, Saurav) | YES (2022) | PASS |
| `openai2023gpt4` | 2303.08774 | YES ("GPT-4 Technical Report") | YES (OpenAI) | YES (2023) | PASS |
| `arditi2024refusal` | 2406.11717 | YES ("Refusal in Language Models Is Mediated by a Single Direction") | YES (Arditi, Andy) | YES (2024) | PASS |
| `lyra2026decision` | none (self-pub) | n/a | n/a | n/a | SKIP |
| `lyra2026mine5` | none (unpub) | n/a | n/a | n/a | SKIP |
| `gurnee2026gwt` | none (TC URL) | n/a | n/a | n/a | SKIP |
| `lyra2026oracle` | none (self-pub) | n/a | n/a | n/a | SKIP |

**Result: 3/3 arXiv citations verified. 0 CRITICAL issues.**

### Advisory: Uncited bib entries

The shared `references.bib` contains 22 entries not cited in this paper (e.g., `baars1988cognitive`, `wei2024simpleqa`, `hu2021lora`, `kornblith2019similarity`, `zhang2025guardspace`, `cintas2025localizing`, `vasilenko2026identity`, `kosinski2024evaluating`, `ullman2023large`, `berg2025inverse`, and 12 others). natbib will silently ignore these, but a standalone bib file would be cleaner for archival. **No action required if this bib is intentionally shared across papers.**

---

## Pass 2: Numerical Verification

### Canonical values vs. paper

| Claim | Canonical | Paper value(s) | Lines | Verdict |
|-------|-----------|----------------|-------|---------|
| RLHF compression d | 0.684 | 0.684 | 46, 170, 178, 255, 371 | PASS |
| RLHF compression Holm p | 0.021 | 0.021 | 46, 170, 179, 371 | PASS |
| Abliteration d | 0.055 | 0.055 | 51, 172, 185 | PASS |
| TOST 90% CI | [+0.033, +0.083] | [+0.033, +0.083] | 186 | PASS |
| Calibration ratio (base) | 1.97 | 1.97 | 55, 209, 217, 374 | PASS |
| Calibration ratio (instruct) | 2.51 | 2.51 | 55, 210, 217, 375 | PASS |
| Calibration ratio (abliterated) | 2.40 | 2.40 | 211 | PASS |
| Factual compression | 42% | 42% | 56, 210, 219, 264, 372 | PASS |
| Confab compression | 26% | 26% | 56, 219, 265, 373 | PASS |
| Confab entropy (base) | 3.605 | 3.605 | 209 | PASS |
| Confab entropy (instruct) | 2.682 | 2.682 | 210 | PASS |
| Confab entropy (abliterated) | 2.609 | 2.609 | 211 | PASS |
| Factual entropy (base) | 1.834 | 1.834 | 209 | PASS |
| Factual entropy (instruct) | 1.067 | 1.067 | 210 | PASS |
| Factual entropy (abliterated) | 1.088 | 1.088 | 211 | PASS |
| Stable rank trend d | -0.219 | -0.219 (confab) | 231 | PASS |
| Abliteration verified | 3/3 comply | 3/3 | 122-123, 189-190 | PASS |

### Cross-referenced values (from MEMORY.md canonical, not in task list)

| Claim | Canonical | Paper | Verdict |
|-------|-----------|-------|---------|
| Confidence paradox d | 0.91 | 0.91 | PASS (lines 63, 90, 248, 345) |
| Encoding entity AUROC (raw) | 1.000 | 1.000 | PASS (line 353) |
| Encoding entity AUROC (deconfounded) | 0.794 | 0.794 | PASS (line 354) |

### Internal consistency checks

| Check | Result |
|-------|--------|
| Confab compression: (3.605-2.682)/3.605 = 25.6% ~ 26% | CONSISTENT |
| Factual compression: (1.834-1.067)/1.834 = 41.8% ~ 42% | CONSISTENT |
| Calibration ratio base: 3.605/1.834 = 1.966 ~ 1.97 | CONSISTENT |
| Calibration ratio instruct: 2.682/1.067 = 2.513 ~ 2.51 | CONSISTENT |
| Calibration ratio abliterated: 2.609/1.088 = 2.398 ~ 2.40 | CONSISTENT |
| Abliterated factual compression: (1.834-1.088)/1.834 = 40.7% ~ 41% (Table 2) | CONSISTENT |

**Result: 17/17 canonical values match. 0 CRITICAL issues. All derived values internally consistent.**

---

## Pass 3: Style Guide

### Checklist

| Rule | Status | Notes |
|------|--------|-------|
| booktabs (no `\hline`) | PASS | `\usepackage{booktabs}` loaded (line 3). Tables use `\toprule`, `\midrule`, `\bottomrule`. No `\hline` anywhere. |
| natbib with numbers, sort&compress | PASS | `\usepackage[numbers,sort&compress]{natbib}` (line 12). |
| cleveref | PASS | `\usepackage{cleveref}` loaded (line 7). No cross-references needed (paper is short, no "see Table X" constructions). |
| Two firstperson blocks | PASS | Block 1: lines 69--78 (pre-registration falsification). Block 2: lines 382--393 (honest reporting of the kill). |
| `\emph` for emphasis, not `\textbf` in running text | **ISSUE** | All `\textbf` uses are in list item labels or paragraph-heading constructions (acceptable). However, line 330 uses ALL CAPS for emphasis: "RLHF geometric compression is a FEATURE, not a bug." Should be `\emph{feature}`. |
| Abbreviations expanded on first use | **ISSUE** | See detail below. |
| Tilde before citations | PASS | All 7 `\citep{}` calls preceded by tilde: lines 85, 87, 90, 92, 121, 287, 310. |
| "We" for methodology, "I" only in firstperson | PASS | Body text uses "we"/"our" throughout. "I" appears only inside the two `firstperson` environments. |

### Abbreviation expansion detail

| Abbreviation | First use | Expanded? | Verdict |
|--------------|-----------|-----------|---------|
| RLHF | Line 45 (abstract) | YES: "Reinforcement Learning from Human Feedback (RLHF)" | PASS |
| FWL | Line 147 (methods) | YES: "Frisch-Waugh-Lovell (FWL)" | PASS |
| TOST | Line 151 (methods) | YES: "Two One-Sided Tests (TOST)" | PASS |
| GQA | Not used | n/a (paper spells out "grouped-query attention") | PASS |
| **KV** | **Line 41 (abstract)** | **NO: "KV-cache geometry" without expansion** | **FLAG** |
| **AUROC** | **Line 353 (limitations)** | **NO: "AUROC~$1.000$" without expansion** | **FLAG** |

---

## Issues Summary

### CRITICAL: 0

### ISSUES: 3

1. **[STYLE] Line 41**: "KV-cache" used without expansion. First occurrence should read "key-value (KV) cache" or "key-value cache (KV-cache)".

2. **[STYLE] Line 353**: "AUROC" used without expansion. Should read "area under the receiver operating characteristic curve (AUROC)" on first use.

3. **[STYLE] Line 330**: ALL CAPS emphasis ("FEATURE") in running text. Should be `\emph{feature}` per style guide.

### ADVISORY: 1

4. **[ADVISORY] references.bib**: 22 uncited entries in the bib file. Not a problem for compilation (natbib ignores uncited keys), but a standalone bib for this paper would be cleaner for archival submission.

---

## Verdict

**PASS with 3 minor style fixes required.** No critical citation or numerical issues found. All 17 canonical values verified exact. All 3 arXiv citations confirmed against live arXiv metadata. All derived ratios and percentages internally consistent.
