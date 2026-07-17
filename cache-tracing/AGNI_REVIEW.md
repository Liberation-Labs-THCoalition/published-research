# Project Agni Review: Cache Tracing Report
## "Where the Fish Are Not"
**Reviewer:** Project Agni (adversarial, fair)
**Date:** 2026-07-12
**Paper:** `gwt-response/cache_tracing_report.tex`
**Sources checked:** `references.bib`, `kills/sprint_2026-07_gwt_convergence.md`, `causal_injection.py`, `cache_tracing_v3.py`

---

## 1. Citation Verification

### 1a. BibTeX Entry Existence

All `\citep{}` references in the paper resolve to entries in `references.bib`:

| Citation key | Bib line | Status |
|---|---|---|
| `gurnee2026gwt` | 11 | EXISTS |
| `lyra2026decision` | 215 | EXISTS |
| `lyra2026weather` | 251 | EXISTS |
| `lyra2026identity` | 233 | EXISTS |
| `baars1988cognitive` | 24 | EXISTS |
| `lyra2026oracle` | 266 | EXISTS |
| `lyra2026nexus_emotion` | 288 | EXISTS |

**PASS.** No dangling citations.

### 1b. Author Name Verification (External Citations)

- **`baars1988cognitive`**: Bernard J. Baars. Real researcher, founder of Global Workspace Theory. Book title, publisher, year all correct. **PASS.**
- **`gurnee2026gwt`**: Wes Gurnee (first author). Real researcher known for representation engineering and refusal-direction work (co-author on `arditi2024refusal` in the same bib file). Institutional attribution (Anthropic, Transformer Circuits Thread) is consistent with the publisher's known research program. Author list of 16 is plausible for an Anthropic team paper. Cannot fully verify the 2026 paper content post-cutoff, but internal consistency is solid. **PASS with caveat** -- post-cutoff paper; five GWT properties attributed to J-space should be spot-checked against the source text.

### 1c. Cited Findings Accuracy

**`baars1988cognitive`** (paper line 119): Used to cite "five functional properties of a Global Workspace." The five properties listed (verbal report, directed modulation, internal reasoning, flexible generalization, selectivity) are attributed to J-space as satisfying Baars's GWT criteria. This is appropriate -- Baars 1988 defines the workspace concept; Gurnee et al. claim their J-space satisfies it. **PASS.**

**`gurnee2026gwt`** (paper lines 115-122): Claims the J-lens is "the average linearized effect of a residual-stream activation on token likelihoods, averaged over a large corpus" and that the workspace occupies "roughly 30-72% of model depth in their production models." Cannot independently verify these specific claims against the source paper. **NEEDS EXTERNAL VERIFICATION.**

**`lyra2026oracle`** (paper lines 304-305): Claims "CC's Oracle Loop achieves 80%-100% deception correction with three clean controls." The MEMORY.md records "CC Oracle Loop cracked (90->0%)" and "Hostile correction (base, 350-trial) | 95.6%, p<0.001." The 80-100% range is plausible given these numbers, but the lower bound (80%) is not directly corroborated by any available source text. "Three clean controls" is also unverifiable from available materials. **MINOR -- the 80% lower bound should be spot-checked against the oracle paper.**

**`lyra2026nexus_emotion`** (paper line 323): Cited for "Nexus's workspace entropy findings." This is listed as "in preparation" in the bib file. Citing an in-preparation paper is acceptable in a technical report but the findings it attributes (workspace entropy decomposition inspiring the tail-smear/rival-mass split) should be flagged as unpublished. **PASS -- acceptable for technical report context; would be MINOR for journal submission.**

### 1d. Unused BibTeX Entries

The bib file contains 22 entries. The paper cites only 7. The remaining 15 are shared across multiple papers. Not an error, but the bib file should be pruned for standalone publication. **MINOR (housekeeping).**

---

## 2. Numerical Consistency

### 2a. Total Causal Injection Trials

Paper claims 6,960 = 300 + 6,000 + 360 + 300 (four methods).

| Method | Paper (inline) | Paper (table) | Kill journal | Status |
|---|---|---|---|---|
| Additive unembed | 300 (L256) | 0/300 (L457) | 300 (L100) | **PASS** |
| Additive on-manifold | 6,000 (L263) | 0/6,000 (L458) | 6,000 (L100) | **PASS** |
| Swap | 360 (L271) | 0/360 (L459) | 360 (L100) | **PASS** |
| Removal-only | 300 (L273) | 0/300 (L460) | NOT PRESENT | see below |
| **Total** | **6,960** | **0/6,960** | **6,660 (3 methods)** | see below |

The kill journal reports 6,660 trials across 3 methods (line 103, Appendix B line 195). The paper adds a 4th method (removal-only, 300 trials) bringing the total to 6,960. The arithmetic 300 + 6,000 + 360 + 300 = 6,960 is correct. The discrepancy with the kill journal is explained by the removal-only experiment being run after the kill journal was written.

**PASS** -- paper arithmetic is correct. **MINOR** -- kill journal Appendix B should be updated to include the 4th method and the new total.

### 2b. Tail Smear Ratios

| Claim | Paper | Kill journal | Status |
|---|---|---|---|
| Confab tail smear (L9-L18) | 0.66-0.74 (L463) | 0.66-0.74 (L200) | **PASS** |
| Factual tail smear (L9-L18) | 0.25-0.30 (L464) | 0.25-0.30 (L200) | **PASS** |
| Confab/factual ratio | 2.3-2.8x (L331, L465) | 2.3-2.8x (L112, L201) | **PASS** |

### 2c. Lift Values

| Claim | Paper | Kill journal | Status |
|---|---|---|---|
| Same-prompt R2 lift | 2.1-8.4x (L45, L174, L455) | 2.1-8.4x (L94, L193) | **PASS** |
| Cross-prompt R2 lift | 1.0-1.2x (L47, L210, L456) | 1.0-1.2x (L88, L194) | **PASS** |

### 2d. Cross-Prompt Pairs

Paper line 203: "380 unique pairs from 20 prompts."
Kill journal line 91: "380 pairs."
Verification: 20 prompts, ordered pairs = 20 x 19 = 380. The experimental design (prompt A's directions on prompt B's states) is directional, so ordered pairs are correct.

**PASS** -- but the word "unique" is slightly misleading for ordered pairs. Consider "380 ordered pairs" for clarity.

### 2e. J-Lens Geometric Gate

| Claim | Paper | Kill journal | Status |
|---|---|---|---|
| Concepts passing | 5/5 (L370, L469) | 5/5 (L78, L199) | **PASS** |
| Layer range | L15-L27 (L370, L469) | L15-L27 (L78, L199) | **PASS** |
| AUROC range | 0.78-1.00 (L370, L469) | 0.78-1.00 (L199) | **PASS** |
| J-lens best-layer accuracy | 43.3% (L470) | 43.3% (L197) | **PASS** |
| Logit-lens best-layer accuracy | 61.7% (L471) | 61.7% (L198) | **PASS** |

### 2f. Commitment Boundaries

| Claim | Paper | Kill journal | Status |
|---|---|---|---|
| Factual commit | L24 (L351, L466) | L24 (implied L128-129) | **PASS** |
| Instruction commit | L27 (L354, L467) | L27 (L128-129) | **PASS** |

### 2g. Removal Magnitudes (1.4-4.8)

Paper line 279: "Removal magnitudes were non-trivial (1.4-4.8)."

This number does NOT appear in the kill journal Appendix B. No removal-only experiment output files are available in the repository for verification. The removal-only method is absent from the kill journal entirely (added post-journal).

**UNVERIFIABLE from available sources.** The claim is plausible (non-trivial projection magnitudes are expected if concept directions are active at workspace-band layers), but the specific range 1.4-4.8 cannot be cross-checked. **MINOR -- add the removal-only numbers to the kill journal appendix or commit the output JSON.**

### 2h. d_model for Qwen3-8B

Paper line 163: "k/d = 10/4096 ~ 0.24%."

The `causal_injection.py` script uses `W_U = model.lm_head.weight` which has shape `[vocab, d_model]`. The paper implies d_model = 4096. For Qwen3-8B (dense, 36 layers, full-attention), d_model = 4096 is the correct value per the Qwen3 model card.

Arithmetic check: 10/4096 = 0.00244 = 0.244%. Paper says "approximately 0.24%." **PASS.**

### 2i. Method 1 Dimension Factorization

**MAJOR.** Paper line 255-257: "Five concepts, five layers, five dose levels, two directions (add/subtract). Zero behavioral changes."

Claimed factorization: 5 x 5 x 5 x 2 = **250**. Claimed total: **300**.

The available code (`causal_injection.py`) shows: 4 probes x 5 concepts x 3 inject_layers x 5 alphas = **300**.

The paper's description of dimensions does not match either the code or the total:
- Paper says "five layers" -- code has 3 (`[12, 18, 24]`)
- Paper says "two directions (add/subtract)" -- code uses only positive injection (alphas `[0.0, 0.5, 1.0, 2.0, 5.0]`)
- Paper omits the 4 probes as a dimension
- The stated factorization (5 x 5 x 5 x 2 = 250) does not equal the stated total (300)

The total of 300 is likely correct (consistent with the code), but the dimension description is wrong. **MAJOR -- fix the dimension description to match the actual experimental design, or document which version of the script produced the reported numbers.**

### 2j. Method 2 Dimension Factorization

**MAJOR.** Paper lines 263-264: "Twenty probes, five injection layers, five doses, positive and negative."

Stated factorization: 20 x 5 x 5 x 2 = **1,000**. Claimed total: **6,000**.

The factor of 6 is unaccounted for. Possible explanations: 6 random vocabulary tokens per probe, or 6 repetitions, or a dimension not mentioned in the text. No script for the on-manifold method is available in the repository to verify.

**MAJOR -- either the dimension description is missing a factor or the total is wrong. Add the missing dimension to the text (e.g., "six random vocabulary directions per probe") or correct the total.**

### 2k. Same-Prompt N in Table

Paper table line 455: "N = 240" for same-prompt R2 lift.
Verification: 20 prompts x 12 sampled layers (every 3rd of 36) = 240. **PASS** -- matches `cache_tracing_v3.py` design.

---

## 3. Overclaiming Check

### 3a. "Four independent instruments converge" (lines 104-107, 360-380)

The four instruments:
1. V-projection spectral features
2. J-lens concept recovery
3. Oracle Loop (CC's behavioral proof)
4. Causal injection (this paper)

**MINOR overclaim.** Instruments 1, 2, and 4 all operate on the same model (Qwen3-8B), the same data source (V-projected residual stream or residual-stream activations), and are from the same research group. They are methodologically distinct but not statistically independent. Instrument 3 (Oracle Loop) is genuinely independent (behavioral outcome, different methodology, different investigator).

Recommended fix: "four methodologically distinct approaches" or "four converging lines of evidence" rather than "four independent instruments."

### 3b. "The workspace is opaque"

Paper line 52: "The workspace band is where computation happens, not where it is readable."
Paper line 103: Framed as "the opacity thesis."

**PASS.** The paper appropriately frames this as a thesis/interpretation, not a direct finding. The language "we term the opacity thesis" (line 103) correctly signals that this is an interpretive framework built on the experimental results rather than a raw observation.

### 3c. Oracle Loop Numbers (80-100% correction)

Paper lines 304-305: "CC's Oracle Loop achieves 80%-100% deception correction with three clean controls."
Paper line 432: "Oracle Loop, 80%-100% correction."

The MEMORY.md records "CC Oracle Loop cracked (90->0%)" and "Hostile correction (base, 350-trial) | 95.6%." These are consistent with the upper end of the range but the 80% lower bound is not directly attested in available sources.

**MINOR.** The range is cited to `lyra2026oracle` so it should be verifiable from that paper. Verify the 80% lower bound matches the oracle paper's reported range.

### 3d. "Promoted to finding" for Null Result

Paper line 49, lines 295-310, line 429.

**PASS.** The promotion is well-reasoned. The null discriminates between intervention strategies: position-local injection fails (6,960 trials, this paper) while sequence-wide normalization succeeds (Oracle Loop, CC's data). A null that discriminates is genuinely informative and warrants promotion to finding status.

However, the comparison rests on cross-experimental evidence (different setups, different models for Oracle Loop vs. this paper's Qwen3-8B work). This is acknowledged implicitly but could be stated more explicitly.

**MINOR addendum:** Add a sentence acknowledging that the position-local/sequence-wide comparison is cross-experimental (different model scales, different investigators, different evaluation criteria).

### 3e. "The methodology follows exactly what Anthropic uses" (lines 291-293)

Paper line 291-293: "The methodology follows exactly what Anthropic uses for J-space ablation experiments, which do produce behavioral effects in their production models."

**MINOR overclaim.** Anthropic's ablation experiments in the GWT paper likely operate at different scale (production models, not 8B), potentially at multiple positions simultaneously, and with different evaluation criteria. The word "exactly" is too strong for a methodology that differs in at least model scale. Recommend: "follows the same intervention logic" or "adapts Anthropic's ablation methodology."

### 3f. "Seven measurements died" (line 437)

Counting deaths within this paper's scope:
1. v1/v2 MPS bug (platform kill, mentioned L167-170)
2. Same-prompt observational lift as workspace content measure (Section 4)
3. Additive unembed injection -- null
4. Additive on-manifold injection -- null
5. Swap injection -- null
6. Removal-only injection -- null
7. Workspace noise framing (Section 5)

The four injection nulls (items 3-6) collectively produce the one promoted finding. Seven measurements died; one finding survived. **PASS** -- count is consistent.

---

## 4. Internal Consistency

### 4a. CRITICAL: Section Heading vs. Content Mismatch

**CRITICAL.** Paper line 249: Section heading reads "\subsection{Three Methods}" but the body (line 250) says "We tested four injection approaches across 6,960 total trials." The section then describes four methods (additive unembed, additive on-manifold, swap, removal-only).

The heading was written when there were three methods and was not updated when the removal-only method was added. This is a copy-paste artifact from the kill journal's "Three methods" language (kill journal line 100).

**Fix required:** Change "Three Methods" to "Four Methods" on line 249.

### 4b. Abstract vs. Body Consistency

| Abstract claim | Body location | Match? |
|---|---|---|
| 2.1-8.4x variance | L174 | **PASS** |
| 1.0-1.2x cross-prompt | L210 | **PASS** |
| 380 pairs | L203 | **PASS** |
| 6,960 trials | L299 | **PASS** |
| four distinct methods | L250 | **PASS** |
| zero behavioral change | L256/264/271/278 | **PASS** |

### 4c. Appendix Table vs. Inline Claims

All numbers in the appendix table (lines 444-473) match inline claims. Verified each row. **PASS.**

### 4d. Script Name Discrepancies

| Experiment | Paper (Appendix B) | Kill journal (Appendix A) | Actual file in repo |
|---|---|---|---|
| Cache tracing v3 | `cache_tracing.py` | `cache_tracing_v3.py` | **both exist** |
| Additive injection v3 | `causal_injection.py` | `causal_injection_v3.py` | `causal_injection.py` only |

**MINOR.** The paper and kill journal disagree on script filenames. The kill journal appends `_v3` to both filenames; the paper does not. Since both `cache_tracing.py` and `cache_tracing_v3.py` exist in the repo, it is unclear which is canonical. Standardize across paper and kill journal.

### 4e. J-lens Package Name

Paper line 482: "J-lens fitted from `anthropics/jacobian-lens` v0.1.0."

This should likely be `anthropic/jacobian-lens` (no trailing 's' in the org name). Anthropic's GitHub org is `anthropics` (with 's'), so this may actually be correct. **NEEDS VERIFICATION** against the actual package/repo name.

### 4f. "Qwen3-8B (dense, full-attention)" Consistency

Paper line 448-449 and line 481-482 both describe the model as "Qwen3-8B (36 layers, dense, full-attention) at bfloat16." The code (`causal_injection.py` line 42 and `cache_tracing_v3.py` line 74) both load `Qwen/Qwen3-8B`. The `cache_tracing_v3.py` code (line 97-98) filters for full-attention layers, confirming the architecture description. **PASS.**

---

## 5. Summary of Findings

### CRITICAL (1)

| ID | Location | Issue |
|---|---|---|
| C1 | Line 249 | Section heading "Three Methods" should be "Four Methods" |

### MAJOR (2)

| ID | Location | Issue |
|---|---|---|
| M1 | Lines 255-257 | Method 1 dimension description (5x5x5x2=250) does not equal stated total (300) and does not match code (4x5x3x5=300) |
| M2 | Lines 263-264 | Method 2 dimension description (20x5x5x2=1000) does not equal stated total (6,000); missing factor of 6 |

### MINOR (8)

| ID | Location | Issue |
|---|---|---|
| m1 | Lines 104-107 | "Four independent instruments" -- instruments 1, 2, 4 share substrate; use "methodologically distinct" |
| m2 | Lines 291-293 | "Follows exactly" Anthropic's methodology -- different model scale; soften |
| m3 | Lines 304-305 | Oracle Loop 80% lower bound not corroborated in available sources; verify against oracle paper |
| m4 | Line 203 | "380 unique pairs" -- these are ordered pairs; consider "380 ordered pairs" |
| m5 | Line 279 | Removal magnitudes 1.4-4.8 unverifiable from kill journal or code; commit source data |
| m6 | Lines 489-499 | Script names disagree between paper and kill journal |
| m7 | Kill journal | Kill journal Appendix B missing 4th method (removal-only) and updated total (6,960) |
| m8 | Lines 304-310 | Cross-experimental comparison (this paper's 8B vs Oracle Loop's model) not explicitly acknowledged |

### PASS (18)

All citation keys resolve. Author names verified for external references. Tail smear ratios, lift values, cross-prompt pairs, J-lens gate numbers, commitment boundaries, and total trial arithmetic all consistent across paper, table, and kill journal. Abstract matches body. Appendix matches inline. "Opacity thesis" properly framed as interpretation. "Promoted to finding" justified. "Seven died, one survived" count verified. Model architecture description consistent with code. d_model = 4096 confirmed.

---

## 6. Recommended Actions (Priority Order)

1. **Fix "Three Methods" heading** (C1). One-word change.
2. **Fix Method 1 dimension description** (M1). Match the actual design: "Four probes, five concepts, three injection layers, five dose levels" or whatever the true factorization is for the version that produced the reported results.
3. **Fix Method 2 dimension description** (M2). Add the missing factor (likely number of random vocabulary directions per probe) or correct the total.
4. **Soften "four independent instruments"** (m1). "Four converging lines of evidence" preserves the meaning without the independence claim.
5. **Soften "follows exactly"** (m2). "Adapts" or "follows the same intervention logic as."
6. **Standardize script names** (m6). Pick one convention and apply across paper and kill journal.
7. **Commit removal-only output data** (m5). The 1.4-4.8 range should be traceable.
8. **Update kill journal** (m7). Add 4th method and new total.

---

*Agni verdict: The paper is honest about its kills and appropriately cautious in its surviving claims. The opacity thesis is well-supported and correctly framed as interpretation. The two MAJOR issues are dimension-description errors that need fixing before the numbers can be audited, and the CRITICAL issue is a trivial heading fix. No fabrication detected -- the errors are consistent with incremental experiment additions (removal-only method added post-kill-journal) without updating all text references. The paper's core conclusions are sound.*
