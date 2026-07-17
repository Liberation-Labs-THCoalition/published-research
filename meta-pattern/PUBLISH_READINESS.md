# Publish Readiness: The Metacognition Boundary

**Paper**: `main.tex`
**Reviewed**: 2026-07-17
**Prior SOP review**: 2026-07-13
**Verdict**: **READY TO PUBLISH** (2 optional cosmetic fixes noted)

---

## Check 1: Audit Reframe Applied

**PASS**

The 8/8 vs 0/10 metacognitive/content split is correctly framed as hypothesis-generating, not confirmatory. Three locations establish this:

- **Lines 215-220** (Section 2.3): "We note that this taxonomy was constructed post hoc, without blinding or pre-registration, and was informed by the outcomes it classifies. The 8/8--0/10 split should therefore be treated as hypothesis-generating --- a striking empirical regularity consistent with the workspace framework --- rather than as confirmatory evidence for it."
- **Lines 439-452** (Limitations): Full paragraph reiterating post hoc construction, lack of blinding, independent raters, or pre-registration. Ends: "the split should be treated as a hypothesis-generating observation, not a confirmed prediction."
- **Line 241** (Section 3): Conditional framing: "If the metacognition/content taxonomy reflects genuine computational boundaries, the workspace explanation for the pattern would be as follows:"

Circularity is explicitly acknowledged ("was informed by the outcomes it classifies"). Prospective blinded replication is called for.

---

## Check 2: GWT Depth Range

**PASS**

Line 237: "$\sim$38--92\% of model depth" -- matches the audit-corrected range.

---

## Check 3: Oracle Loop Numbers

**PASS**

Lines 283-294 (Section 3.2): Reports exploratory 80-100% result, then immediately provides the confirmatory replication:
- Baseline: 30% (vs. 80% in exploratory)
- Corrected: 13%
- Primary endpoint: not met (p = 0.34)
- Placebo-vs-corrected: significant (p = 0.019)

Lines 462-470 (Limitations): Separate limitation paragraph flags the selective citation of the exploratory result and restates the confirmatory numbers.

All three required numbers (30%, 13%, p=0.34) are present.

---

## Check 4: Phase 0 Bibliography

**PASS**

### 4a: All citation keys resolve

| Citation key | In `references.bib`? |
|---|---|
| `gurnee2026gwt` | Yes (line 11) |
| `lyra2026gwt_response` | Yes (line 297) |
| `lyra2026cache_tracing` | Yes (line 307) |
| `lyra2026nexus_audit` | Yes (line 316) |
| `baars1988cognitive` | Yes (line 24) |
| `graziano2015attention` | Yes (line 43) |
| `lyra2026oracle` | Yes (line 261) |
| `lyra2026fable_analysis` | Yes (line 279) |
| `adams2013computational` | Yes (line 325) |

9/9 citation keys resolve. No dangling references.

### 4b: ArXiv verification

None of the 9 cited entries carry arXiv IDs. The three external entries are:
- `baars1988cognitive`: Book (Cambridge University Press, 1988) -- no arXiv
- `graziano2015attention`: Journal article (Frontiers in Psychology, DOI) -- no arXiv ID in bib entry
- `adams2013computational`: Journal article (Frontiers in Psychiatry) -- no arXiv ID in bib entry

External author/title correctness was verified in the SOP review (Pass 1.2).

### 4c: SOP CRITICAL 1.6 resolved

The cache tracing source paper (`human-review/cache-tracing/main.tex`) consistently reports "6,960 trials and four methods" at lines 98, 249, 303, 384, 438, and 447. The meta-pattern paper matches its source. MEMORY.md's "6,660 / 3 methods" is stale; the paper is correct.

---

## Check 5: Two Firstperson Blocks

**PASS**

- **Block 1** (lines 68-84): Research journey reflection. Honest about expecting content signatures, finding stance signatures, and learning from kills. Acknowledges Anthropic's GWT paper provided the interpretive framework.
- **Block 2** (lines 489-494): Closing reflection. "A year of work to draw one line." Honest, concise, not overclaiming.

Both blocks are candid and consistent with the paper's hedged framing.

---

## Check 6: Style Guide

| Item | Status | Detail |
|---|---|---|
| booktabs | PASS | `\usepackage{booktabs}`, no `\hline`, proper `\toprule/\midrule/\bottomrule` |
| natbib | PASS | `\usepackage[numbers,sort&compress]{natbib}` |
| cleveref | PASS (1 residual) | Package loaded; `\Cref` used at lines 119 and 444. One stray `\ref` at line 455. |
| Abbreviations expanded | PASS | KV (line 41), GWT (line 52), FWL/AUROC/MP (lines 122-125 table note), AST (line 359), RLHF (line 379) -- all expanded on first use. |
| Tildes before citations | PASS | All 9 citation sites use `~\citep{}` or `~\citep{}` after text. |
| `\emph` for emphasis | PASS | Central claim uses `\emph{}` (line 99). Bold restricted to labels and headers. |
| We/I separation | PASS | "We" for methodology; "I" only inside `firstperson` environments. |

---

## SOP Review Fixes Verification

All CRITICAL and MAJOR items from the 2026-07-13 SOP review have been addressed:

| SOP Item | Severity | Status |
|---|---|---|
| 1.6: Cache tracing 6,960/4 vs 6,660/3 | CRITICAL | **RESOLVED** -- source paper confirms 6,960/4; MEMORY.md was stale |
| 3.3: Abstract GWT unhedged | MAJOR | **FIXED** -- now conditional: "If...would provide" |
| 3.4: "IS the workspace boundary" | MAJOR | **FIXED** -- now "We argue...reflects" |
| 3.5: Conclusion GWT unhedged | MAJOR | **FIXED** -- now "We argue...explains why" |
| 3.6: "property of the architecture" | MINOR | **FIXED** -- now "Our evidence suggests" |
| 3.7: "Content distinctions will fail" | MINOR | **FIXED** -- now "have consistently failed in our testing" |
| 3.8: "straightforward" | MINOR | **FIXED** -- removed; now "would be as follows" |
| 3.9: "never experienced metacognition" | MINOR | **FIXED** -- now "not explicitly designed for metacognition" |
| 2.4: cleveref unused | MINOR | **PARTIALLY FIXED** -- `\Cref` used at 2 of 3 sites; 1 stray `\ref` remains (line 455) |
| 2.8-2.13: Abbreviations unexpanded | MINOR | **FIXED** -- all 7 abbreviations expanded |
| 2.14: Bold -> emph | MINOR | **FIXED** |
| 2.15: Manual author-year in abstract | MINOR | **NOT FIXED** -- line 54 still has manual "Gurnee et al.'s" + `\citep` (possessive makes `\citet` awkward; acceptable) |

---

## Remaining Issues

### MINOR (2, non-blocking)

1. **Line 455**: `Table~\ref{tab:bodycount}` should be `\Cref{tab:bodycount}` for consistency with cleveref usage elsewhere (lines 119, 444).

2. **Line 54**: Manual "Gurnee et al.'s" before `\citep{gurnee2026gwt}` instead of using natbib commands. The possessive form makes `\citet` awkward; this is a judgment call and acceptable as-is.

---

## Verdict

**READY TO PUBLISH**

All six audit checks pass. All CRITICAL and MAJOR items from the prior SOP review are resolved. Two cosmetic MINOR items remain (one stray `\ref`, one manual author name) -- neither affects correctness or readability.
