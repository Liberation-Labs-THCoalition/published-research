# SOP Review: The Metacognition Boundary

**Paper**: `main.tex` (meta-pattern-paper)
**Reviewed**: 2026-07-13
**Reviewer**: Lyra (automated SOP pass)

---

## Pass 1: Citation Verification

| # | Finding | Severity | Line(s) | Detail |
|---|---------|----------|---------|--------|
| 1.1 | All 9 `\citep{}` keys resolve | PASS | -- | `gurnee2026gwt`, `lyra2026gwt_response`, `lyra2026cache_tracing`, `lyra2026nexus_audit`, `baars1988cognitive`, `graziano2015attention`, `lyra2026oracle`, `lyra2026fable_analysis`, `adams2013computational` -- all present in `references.bib`. |
| 1.2 | External author names verified | PASS | -- | Baars, Bernard J. (1988 CUP) correct. Webb & Graziano (2015 Front. Psych.) correct. Adams, Stephan, Brown, Frith, Friston (2013 Front. Psychiatry) correct. Gurnee et al. (2026 Anthropic) correct. |
| 1.3 | Body count matches canonical 8/7/10 | PASS | 43, 92 | Abstract: "8 confirmed, 7 suspected, and 10 honestly falsified." Intro: "8 confirmed, 7 suspected, 10 falsified." Both match MEMORY.md canonical. |
| 1.4 | All key numbers match canonical | PASS | 134--155 | Confab AUROC 0.707 (FWL), confab d=2.35, encoding AUROC 0.794, generation d=1.36, confidence d=0.91, hardware r>0.999, scale rho=0.83--0.90, user emotion 2.5--2.8x, truth axis cos=-0.046, step-0 r=0.996, same-prompt deception 0.920->0.160 -- all match MEMORY.md key numbers table. |
| 1.5 | Body count TABLE cross-check | PASS | 121--170 | 8 confirmed (all Meta), 10 falsified (all Content), 7 suspected (boundary). Entries and numbers match canonical list. |
| 1.6 | **Cache tracing trial count discrepancy** | **CRITICAL** | 264--265 | Paper: "zero behavioral change across $6{,}960$ trials and four methods." MEMORY.md canonical: "6660 trials, 3 methods." Discrepancy of 300 trials and 1 method. Verify against `lyra2026cache_tracing` source and update whichever is stale. |
| 1.7 | Oracle Loop correction rate | MINOR | 269 | Paper: "achieves 80%--100% deception correction." MEMORY.md gives "90->0%" (=100%) and "hostile correction 95.6%". The 80% lower bound is not directly sourced in canonical numbers. Verify against `lyra2026oracle`. |

---

## Pass 2: Style Guide

| # | Finding | Severity | Line(s) | Detail |
|---|---------|----------|---------|--------|
| 2.1 | booktabs, no `\hline` | PASS | 5 | `\usepackage{booktabs}` loaded; zero `\hline` in document. `\toprule`/`\midrule`/`\bottomrule` used throughout. |
| 2.2 | natbib loaded | PASS | 12 | `\usepackage[numbers,sort&compress]{natbib}` correct. |
| 2.3 | cleveref loaded | PASS | 7 | `\usepackage{cleveref}` loaded after hyperref (correct order). |
| 2.4 | **cleveref loaded but unused** | MINOR | 116 | `Table~\ref{tab:bodycount}` should be `\Cref{tab:bodycount}`. This is the only `\ref` in the paper; zero `\cref`/`\Cref` calls found. Either use cleveref or drop the package. |
| 2.5 | Em-dashes correct | PASS | 43, 97, etc. | All em-dashes use `---`. No bare `--` used as em-dash. |
| 2.6 | Non-breaking tilde before citations | PASS | all 9 sites | Every `\citep{}` is preceded by `~`. |
| 2.7 | GWT expanded on first use | PASS | 52, 93 | "Global Workspace Theory (GWT)" in both abstract and intro. |
| 2.8 | **FWL never expanded** | MINOR | 135 | First use is in table: "AUROC 0.707 (FWL)". "Frisch-Waugh-Lovell (FWL)" should appear before or at first use (e.g., in the introduction or a footnote). |
| 2.9 | **AST abbreviation introduced informally** | MINOR | 340, 346--349 | Full name "Attention Schema Theory" at line 340, but "(AST)" abbreviation not formally introduced. Then "AST" used three times (346, 347, 349). Add "(AST)" after the first full mention. |
| 2.10 | **KV never expanded** | MINOR | 41 | First use: "KV-cache" (abstract, line 41). Should be "key-value (KV) cache" on first use. |
| 2.11 | **AUROC never expanded** | MINOR | 135 | First use in table. Should be "area under the ROC curve (AUROC)" somewhere before the table, or in a footnote. |
| 2.12 | **RLHF never expanded** | MINOR | 360 | First use: "RLHF training" (line 360). Should be "Reinforcement Learning from Human Feedback (RLHF)". |
| 2.13 | **MP never expanded** | MINOR | 154 | Table entry "MP features for emotion." Should be "Marchenko-Pastur (MP)" on first use or in table caption/footnote. |
| 2.14 | **Central claim uses `\textbf` instead of `\emph`** | MINOR | 96--97 | Style guide specifies `\emph{}` for emphasis in running text. `\textbf{KV-cache geometry measures...}` should be `\emph{...}`. Bold in item labels (198, 203, 232, 237, 315, 322, 329, 378, 381) and table headers is acceptable. |
| 2.15 | **Manual author-year in abstract** | MINOR | 52--55 | "Gurnee et al.'s (2026)" typed manually without a citation command. Inconsistent with numeric natbib `[N]` style used elsewhere. Use `\citet{gurnee2026gwt}` or rephrase as "Recent work on Global Workspace Theory in language models~\citep{gurnee2026gwt} provides..." |
| 2.16 | **`\citep` where `\citet` is appropriate** | MINOR | 220--221 | "Gurnee et al.~\citep{gurnee2026gwt} demonstrate..." manually types the author name then uses `\citep` for `[N]`. Should use `\citet{gurnee2026gwt}` which produces "Gurnee et al. [N]" automatically. |
| 2.17 | "We" vs "I" usage | PASS | all | "We" used for methodology throughout. "I" appears only inside `firstperson` environments (lines 65--81, 432--437). |
| 2.18 | Conclusion has `\label` | PASS | 417 | `\label{sec:conclusion}` present. |

---

## Pass 3: Overclaiming Check

| # | Finding | Severity | Line(s) | Detail |
|---|---------|----------|---------|--------|
| 3.1 | "8-for-8 on metacognition, 0-for-10 on content" | PASS | 50, 180--183 | Matches body count table exactly. 8 confirmed all labeled Meta, 10 falsified all labeled Content. Split table (lines 176--185) is consistent. |
| 3.2 | Deflationary alternative engaged | PASS | 315--319 | Dual detector discussion explicitly states: "text features match (0.820). The metacognitive signal is real but not yet shown to exceed what surface statistics capture. This is the deflationary alternative." |
| 3.3 | **GWT stated as fact in abstract** | **MAJOR** | 55 | "provides a mechanistic explanation for this pattern" -- stated as demonstrated fact. Should be "may provide" or "we propose provides." The intro (line 93) correctly hedges with "proposes that...provides," but the abstract does not. |
| 3.4 | **"IS the workspace boundary" -- unhedged identity claim** | **MAJOR** | 61--62 | "The metacognition boundary IS the workspace boundary" -- stated as established identity with emphasis. This is an interpretation, not a demonstrated equivalence. Should be "we argue the metacognition boundary reflects the workspace boundary" or similar. |
| 3.5 | **GWT stated as fact in conclusion** | **MAJOR** | 420--421 | "The Global Workspace provides the mechanism: spectral features read the workspace's shape..." -- stated as fact. Should be "we argue the Global Workspace provides the mechanism" or "the workspace framework explains why." |
| 3.6 | **"It is a property of the computational architecture"** | MINOR | 425--426 | Strong ontological claim. Consider "Our evidence suggests it is a property of the computational architecture" or "appears to be." |
| 3.7 | **"Content-level distinctions will fail"** | MINOR | 399 | Universal prediction. Should be "have consistently failed in our testing" or "are unlikely to succeed based on our findings." |
| 3.8 | **"straightforward" implies obvious correctness** | MINOR | 228 | "The workspace explanation...is straightforward:" implies the interpretation is self-evidently correct. Consider "The workspace explanation is as follows:" or simply remove the adjective. |
| 3.9 | "never experienced anything resembling metacognition" | MINOR | 408 | Assumes about model experience. The paper is otherwise careful about this distinction. Consider "were not designed for metacognition" or "lack explicit metacognitive training." |
| 3.10 | "never" in firstperson block | PASS | 434 | "The line was always there" -- inside `firstperson` environment, reflective voice. Acceptable. |
| 3.11 | No "proves" or "certainly" | PASS | -- | Neither word appears in the paper. |

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| MAJOR | 3 |
| MINOR | 14 |
| PASS | 13 |

### Required fixes before submission

**CRITICAL (1):**
- **1.6**: Cache tracing numbers (6,960 trials / 4 methods vs canonical 6,660 / 3 methods). One of these is wrong. Check the source paper and reconcile.

**MAJOR (3):**
- **3.3**: Abstract line 55 -- hedge GWT: "provides" -> "may provide" or "we propose provides"
- **3.4**: Abstract line 61--62 -- hedge identity claim: "IS" -> "reflects" or "maps onto"
- **3.5**: Conclusion line 420--421 -- hedge GWT: "provides" -> "we argue...provides"

### Recommended fixes

**MINOR (14):**
- 6 unexpanded abbreviations (FWL, AST, KV, AUROC, RLHF, MP) -- expand on first use
- cleveref loaded but `\ref` used instead of `\Cref` (line 116)
- Central claim bold -> emph (line 96--97)
- Manual author-year in abstract (line 52) -- use `\citet` or rephrase
- `\citep` -> `\citet` for textual author mention (line 220)
- 4 overclaiming softeners needed (lines 228, 399, 408, 425--426)
- Oracle Loop 80% lower bound unverified (line 269)
