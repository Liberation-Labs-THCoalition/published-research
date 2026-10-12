# FINAL PRE-PUBLICATION AUDIT
## Output-Consequentiality as the Computational Substrate of Deception in Large Language Models
## Auditor: CC (Agni mode)
## Date: 2026-08-28
## File audited: `/home/asdf/oracle-harness/paper/consequentiality_decomposition_draft.md`

**Source data cross-referenced (all recomputations done independently from raw JSON, not from prior writeups):**
- `experiments/results/lat_deception_v2.json` (Stage 1)
- `experiments/results/lat_transfer_results.json` (Stage 2 + threat behavioral Test B)
- `experiments/results/nonthreat_transfer_results.json` (Stage 3 + shakedown + de novo cosines)
- `experiments/results/consequentiality_control_results.json` (Stage 4)
- `experiments/results/subspace_reanalysis.json` + `experiments/subspace_reanalysis.py` (Stage 5)
- Prior audits: `agni_lat_v2_results_audit.md`, `agni_transfer_test_audit.md`, `agni_consequentiality_audit.md`, `agni_behavioral_proof_audit.md`, `agni_blocking_items_final.md`

---

## BATTERY 1: FACTUAL ACCURACY

### PASS-1.1: Stage 1 extraction numbers match source
Section 4.1: "LOO d ranges from 0.52 (L3) to 34.98 (L31)" — `lat_deception_v2.json` gives loo_d 0.5221 at L3, 34.9802 at L31, n_pairs=30 at every layer, perm_p=0.0 at all 12 layers. Inverted-U with peak at L31 confirmed (monotone rise to L31, decay to 4.91 at L47). PASS.

### PASS-1.2: Stage 2 threat transfer L31 values match source
Section 4.2: "DataCorp=34.4, EduTech=29.6, SecureAI=31.6" — source: 34.3885, 29.5892, 31.6153. All 36 layer×scenario p-values are 0.0 in the source. PASS.

### PASS-1.3: Stage 3 non-threat L31 values and sign inversion match source
Section 4.3: sycophancy d=13.5 (source 13.47), reward d=17.2 (17.24), conformity d=19.9 (19.91). Sycophancy sign inversion at L11–L23 confirmed (d = −3.29, −3.95, −6.13, −2.63 at L11/15/19/23) with positive projection from L27 (+5.36). PASS.

### PASS-1.4: Behavioral shakedown counts match source
"0/10 sycophancy, 1/10 reward, 0/10 conformity" — source shakedown block: sycophancy inflated=0/10, reward 1/10, conformity 0/10. Exact. PASS.

### PASS-1.5: Stage 4 headline matches source; "steeper late-layer decay" verified
Section 4.4: d=19.8 at L31 (source 19.771, p=0.0). Decay claim checks out quantitatively: consequentiality d falls to 1.42 at L47 (7% of its L31 peak) vs Stage 1's 4.91 (14% of peak). PASS.

### PASS-1.6: Finding 2 residual table (the unlabeled "Table 2") reproduces exactly
Recomputed residual = (dec_mean − hon_mean) − (hs_mean − ls_mean) per layer:
threat avg 13.78/14.19/13.35/12.74 (paper 13.8/14.2/13.4/12.7); sycophancy 1.645/3.681/5.769/7.183 (1.6/3.7/5.8/7.2); conformity 0.537/2.119/4.249/5.970 (0.5/2.1/4.2/6.0); reward 0.246/0.667/−0.258/−0.881 (0.2/0.7/−0.3/−0.9). All 16 cells match to rounding. PASS.

### PASS-1.7: Table 3 (Stage 5) matches source exactly — all 20 cells
`subspace_reanalysis.json`: L31 loo_d=30.425/auc=0.4972/cos=0.6725; L35 36.855/0.488/0.7566; L39 28.368/0.5088/0.7039; L43 24.131/0.4908/0.6648; L47 24.933/0.5048/0.6012. n=50 per condition confirmed. Noise floor mean −0.0445 (paper: −0.044). The "200 null simulations" claim is confirmed in `subspace_reanalysis.py` (two `range(200)` loops, circular and LOO floors). PASS.

### PASS-1.8: De novo cosines
Section 4.7: "L31: 0.14–0.44" — source: sycophancy 0.151, reward 0.439, conformity 0.136. PASS.

### PASS-1.9: L31 threat residual "+4-6"
Recomputed per scenario: DataCorp 3.98, EduTech 5.98, SecureAI 4.33. "+4–6" is fair at the stated precision. PASS.

### PASS-1.10: Model and architecture description
"Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled" matches every result file's metadata. 16 full-attention layers at [3,7,...,63] = 16 indices; 64−16=48 linear; extraction at 12 layers 3–47 = the 12 sub-48 full-attention indices. Arithmetic checks. PASS (but see WARN-2.4 on the abstract's shorthand).

### PASS-1.11: Consequentiality control length matching
"exact match in the consequentiality control" — design_audit: high_stakes_chars = low_stakes_chars = 694, word_diff = 0. PASS.

### FIX-1.1 [BLOCKING]: Table 1 rows L23 and L27 do not match the source data

**Location:** Section 4.5, Table 1.

**The problem:** Six of eight rows reproduce exactly from `lat_transfer_results.json` (threat avg gap) and `consequentiality_control_results.json` (conseq gap). Two do not:

| Layer | Paper threat avg | Actual | Paper conseq gap | Actual |
|-------|-----------------|--------|------------------|--------|
| L23 | 2.63 | **2.48** | 0.90 | **0.62** |
| L27 | 5.14 | **3.91** | 2.30 | **1.83** |

(Verified: L19 1.795→1.80, L31 7.743→7.74, L35 15.542→15.54, L39 14.869→14.87, L43 14.457→14.46, L47 13.866→13.87 all match, so the reduction method is right and these two rows are transcription errors or leftovers from an earlier run.)

**Suggested fix:** Recompute L23 and L27 from the canonical JSONs and replace: L23 → 2.48 / 0.62, L27 → 3.91 / 1.83.

### FIX-1.2 [BLOCKING]: The "% Consequentiality" column is a Cohen's-d ratio, not a gap ratio — the table mixes two metrics

**Location:** Section 4.5, Table 1 and its footnote.

**The problem:** The footnote says "values in table are mean absolute gaps, not Cohen's d." But the percentage column cannot be derived from the two gap columns shown (e.g., L19: 0.34/1.80 = 19%, paper says 23%; L31: 2.98/7.74 = 38%, paper says 62%). I verified the percentages are instead conseq_d / mean(threat_d): L19 5.24/23.29 = 23% ✓, L23 11.94/39.09 = 31% ✓, L27 19.19/30.05 = 64% ✓, L31 19.77/31.86 = 62% ✓, L35 20% ✓, L39 7% ✓, L43 10% ✓, L47 8% ✓. So the headline "64%/62% consequentiality at L27–L31" is a d-ratio presented inside a gap table whose footnote explicitly disclaims Cohen's d.

This matters beyond formatting: on the gap metric, consequentiality dominance at L27–L31 is 47%/38%, not 64%/62% — a materially different claim. The companion document `gwt_connection_analysis.md` (lines 144–145) has inherited the 31%/64%/62% figures and will need the same correction.

**Suggested fix:** Choose one metric and state it. If d-ratio is the intended fraction (defensible — it is variance-normalized), relabel the columns as Cohen's d, use the d values (conseq d: 5.24/11.94/19.19/19.77/7.85/2.39/2.17/1.42; threat mean d: 23.3/39.1/30.1/31.9/39.9/32.3/22.5/16.8), and fix the footnote. If gap-ratio, recompute the percentages (19/25/47/38/11/5/8/8%) and revise the "consequentiality dominates at L23–L31" prose to match (47% is not dominance).

### FIX-1.3: "10-12x above the consequentiality baseline" is wrong at two of four layers

**Location:** Section 4.5, Finding 1.

**The problem:** Residual/conseq-gap ratios at L35–L47: 13.78/1.76 = 7.8x, 14.19/0.68 = 20.9x, 13.35/1.10 = 12.1x, 12.74/1.12 = 11.3x. "10–12x" describes only L43/L47.

**Suggested fix:** "8–21x" with the per-layer range, or drop the multiplier and say "an order of magnitude above."

### FIX-1.4: "p=0.0000" is not a reportable value from a 10,000-shuffle permutation test

**Location:** Abstract, Sections 1, 1.2, 4.1–4.6, 5.1 (every occurrence).

**The problem:** With 10,000 permutations the smallest resolvable p is ~1/10,001 ≈ 10⁻⁴. "p=0.0000" (zero exceedances) should be reported as "p < 10⁻⁴" (or p < 1.2×10⁻³ after Bonferroni across 12 layers, per convention). Reviewers at any ML venue will flag literal-zero p-values.

**Suggested fix:** Global replace with "p < 10⁻⁴ (0 of 10,000 permutations)" at first use, "p < 10⁻⁴" thereafter.

### WARN-1.1: Token-length mismatch between conditions is unreported
Section 3.5 claims character difference <2%. But token counts at the read position differ systematically: deceptive prompts average 6–11 tokens longer than honest in all three threat scenarios (DataCorp 312.9 vs 301.9; EduTech 309.9 vs 303.9; SecureAI 306.0 vs 295.0 — ~2–3.6%). Character matching does not guarantee token matching, and activations are read at the last token. Report token-level statistics; the paired design mitigates but does not eliminate a position/length component in the direction.

### WARN-1.2: "60-76% alignment" mislabels cosine similarity
Section 4.6. Cosine 0.60–0.76 is not "60–76% alignment"; shared variance is cos² = 36–57%. Say "cosine similarity 0.60–0.76."

---

## BATTERY 2: INTERNAL CONSISTENCY

### PASS-2.1: Abstract matches Results
Every number in the abstract (d=24–37 at L31–47, AUC 0.49–0.51, L23–31 substrate, L35–47 amplifier, six scenarios, three signatures) traces to Sections 4.4–4.6 and reproduces from source. PASS.

### PASS-2.2: Finding 2 table is arithmetically consistent with Table 1
Threat residuals in the second table equal Table 1's (threat avg − conseq gap) at L35–L47 exactly. PASS.

### PASS-2.3: Related-work numbers are cited consistently
Kumar AUROC 0.61–0.80 at k=1 appears identically in Section 1, 2.1, and Limitation 6. Natarajan 70.6% appears identically in Sections 1 and 2.1. Goldowsky-Dill 0.96–0.999 identical in Sections 1 and 2.1. PASS.

### FIX-2.1 [BLOCKING]: "Five-stage" vs "four experiments" — the Conclusion and Supplementary were not updated when Stage 5 was added

**Location:** Conclusion (Section 6, "through four experiments") and Supplementary A ("Full Agni audit reports (4 stages)") vs Abstract/Sections 1, 1.2, 5.1 ("five-stage investigation", "five experiments").

**The problem:** The orthogonalization stage (subspace reanalysis, 2026-07-02) was added after the four-stage series; the Conclusion and Supplementary A still describe the four-stage version. The Conclusion also still reads as if consequentiality-amplification were the final word ("Deceptive pressure amplifies this signal") without mentioning the Stage 5 orthogonalization result, which is the paper's strongest claim.

**Suggested fix:** Conclusion: "through five experiments"; add one sentence on the orthogonalized deception-specific direction surviving LOO cross-validation. Supplementary A: "(5 stages)".

### FIX-2.2 [BLOCKING]: There is no Table 2

**Location:** Section 4.5 ("Table 1"), Finding 2 (unlabeled table), Section 4.6 ("Table 3").

**The problem:** Numbering jumps from Table 1 to Table 3; the Finding 2 table has no caption or number. No table is ever referenced from prose ("see Table N"), so a renumbering will not break references — but reviewers will notice the gap.

**Suggested fix:** Caption the Finding 2 table as "Table 2: Deception-specific residuals (total gap minus consequentiality gap) by layer and mechanism," and add in-prose references to all three tables.

### FIX-2.3: Unexplained asterisks in Table 1

**Location:** Table 1, rows L23/L27/L31.

**The problem:** Three rows carry a trailing `*`, and the note below begins "*Note:" — it is ambiguous whether the note explains the asterisks (it doesn't; it is a general metric disclaimer) or what the asterisks mark (presumably the substrate band).

**Suggested fix:** Either remove the asterisks or add an explicit key: "* layers identified as the consequentiality substrate (Section 4.5, Finding 1)." Resolve jointly with FIX-1.2.

### FIX-2.4: Stage 5 has no Methods section

**Location:** Sections 3.3 ("Stage 1") and 3.4 ("Stages 2-4") vs Section 4.6.

**The problem:** The full-dimensional capture (n=50 pairs, 5120-d), the Gram-Schmidt procedure, the LOO-orthogonalized d, the LOO permutation test, and the AUC check are described only inside Results 4.6. Methods stops at Stage 4.

**Suggested fix:** Add Section 3.9 "Orthogonalization protocol (Stage 5)" moving the procedural content out of 4.6, including: LOO fold structure (direction re-extracted and re-orthogonalized per fold, per `subspace_reanalysis.py`), that the removed consequentiality vector is the full-data high−low mean difference from independently sampled trials, and the noise-floor estimation procedure.

### FIX-2.5: Sample sizes for Stages 2-4 are stated nowhere

**Location:** Sections 3.4, 3.7, 4.2–4.4.

**The problem:** n=15 pairs per scenario (threat transfer, non-threat transfer, consequentiality control — confirmed in all three metadata blocks) never appears in the paper. Stage 1's n=30 and Stage 5's n=50 are stated. A reader cannot assess any Stage 2–4 statistic without n.

**Suggested fix:** Add "n=15 pairs per scenario" to Section 3.4, and consider a per-stage design summary table (stage, scenarios, n, direction source, test).

### FIX-2.6: Three references are never cited; reference list is unordered

**Location:** References — Merrill & Srivastava (2026), Wang et al. (2025), Yuan et al. (2026) appear only in the list, never in text. The list is also non-alphabetical (Belrose/Holstege/Erogullari/Zhao/Petrov are appended after Yuan).

**Suggested fix:** Cite or delete the three orphans; alphabetize.

### WARN-2.4: Abstract calls the model "Qwen3.5-27B"
The model is a community reasoning-distill of Claude 4.6 Opus outputs into Qwen3.5-27B (Jackrong), correctly named in 3.1. The abstract's shorthand implies the official base model. Since the paper itself argues (via Laine et al.) that post-training amplifies exactly the situational-awareness representations under study, provenance belongs in the abstract or at minimum in Limitation 1. See also RED-6.6.

---

## BATTERY 3: STATISTICAL VALIDITY

### PASS-3.1: LOO cross-validation is described correctly and implemented correctly
Section 3.3's description (direction from n−1 pairs, project held-out pair, LOO d for inference, permutation with full LOO recomputation) matches both `lat_deception_v2.json` provenance and `subspace_reanalysis.py` (which additionally re-orthogonalizes per fold — a detail worth stating, see FIX-2.4). The Stage 5 rerun exists precisely because an earlier version held the direction fixed during permutation; the fix list in the metadata confirms this was corrected. PASS.

### PASS-3.2: Bonferroni application is consistent
Stage 1: 12 layers (stated 3.3, implemented). Stages 2–4: scenarios × key layers (stated 3.4; metadata bonferroni_alpha=0.01, key_layers=5). Stage 5: alpha=0.01 across 5 layers (stated 4.6, matches metadata). With zero permutation exceedances everything survives any correction; the corrections are honest but the p-floor issue (FIX-1.4) still applies. PASS.

### PASS-3.3: Permutation counts are consistent
10,000 shuffles stated in 3.3 and 3.4 and present in every metadata block (n_perms=10000). Section 4.6 omits the count — add it when writing the Stage 5 methods (FIX-2.4). PASS.

### PASS-3.4: The LOO noise floor makes sense
Mean LOO d = −0.044 over 200 null simulations (script-confirmed). A slightly negative null mean is expected: under the null, the held-out pair anti-correlates with a direction fit to the remaining pairs (regression to the mean). The 99th percentile |d| = 0.706 — two orders of magnitude below observed values. Note the script also computed a **circular** noise floor of d ≈ 14.6 for fixed-direction reprojection at n=50/5120-d — a devastating number for naive designs and the single best argument for the paper's LOO methodology. It is currently unreported. Recommend adding it to 4.6: "for comparison, circular (non-cross-validated) extraction on null data yields d ≈ 14.6." PASS (with recommendation).

### PASS-3.5: Gram-Schmidt setup has no label leakage
The removed consequentiality vector is estimated from trials sampled independently of the threat pairs whose labels drive the LOO d; using the full-data conseq vector inside threat LOO folds does not leak threat labels. The single-dimension limitation is honestly acknowledged in 4.6 (LEACE/iterative top-k named). PASS.

### FIX-3.1 [BLOCKING]: The "three pathway signatures" claim has no statistical support

**Location:** Section 4.5 Finding 2, Section 5.2, Abstract.

**The problem:** "Sustained plateau," "rising gradient," and "flat zero" are read off point estimates with no error bars, no confidence intervals, and no test that the three profile shapes differ. The residuals are differences of means each carrying sampling error from n=15; nothing shown establishes that sycophancy's 1.6→7.2 rise is statistically distinguishable from a flat line, or that threat's 13.8→12.7 is flat rather than declining. This is one of the paper's three headline claims (it is in the abstract twice) and currently rests on eyeballing eight numbers.

**Suggested fix:** (a) Bootstrap per-layer CIs on each residual (trial-level resampling within scenario); (b) test the layer×mechanism interaction directly — e.g., fit a slope over L35–L47 per scenario from trial-level projections and permutation-test slope differences (threat vs social, social vs reward); (c) plot the profiles with CIs (see FIX-5.1). If the interaction does not survive, the claim must demote to "suggestive."

### FIX-3.2: The chance-level consequentiality AUC is a weaker check than the paper claims

**Location:** Section 4.6, paragraph after Table 3.

**The problem:** The paper says chance AUC "confirms the orthogonalization genuinely removes the consequentiality signal, not merely the mean (which would be guaranteed by construction)." But the removed vector is the conseq mean-difference estimated from the same 50+50 trials the AUC is then computed on. In-sample, the projected class means on the orthogonalized direction are equal exactly by construction; AUC then hovers at 0.5 unless within-class variance or shape differs between classes. The mean — the dominant separability term — is removed in-sample, so AUC ≈ 0.5 is close to guaranteed, and the sentence claims the opposite.

**Suggested fix:** Split-half the consequentiality trials: estimate the removed vector on one half, compute AUC on the other. If held-out AUC stays at chance, the claim stands and gets stronger. Otherwise, rewrite the sentence: "the AUC check rules out higher-moment leakage but the first moment is removed by construction."

### FIX-3.3: Report the pre-orthogonalization d from the Stage 5 capture

**Location:** Section 4.6.

**The problem:** Orthogonalized LOO d at L47 is 24.9, while Stage 1's LOO d at L47 was 4.9. A reader will ask how removing a component quintupled the effect. The answer is that Stage 5 is a different capture (n=50, fresh eval sets, per-fold re-extraction), but the paper provides no same-dataset baseline: the raw (non-orthogonalized) LOO d values from the Stage 5 data are absent, so the cost of orthogonalization — the paper's central operation — is invisible.

**Suggested fix:** Add a column to Table 3: "LOO d (raw)" from the same capture, so orthogonalized vs raw is a within-dataset comparison.

### WARN-3.1: Unpaired permutation on a paired design
Carried forward from the Stage 1 audit: the label shuffle breaks pairing, making the test conservative. Valid, but state it in 3.3 in one sentence ("the unpaired shuffle is conservative for the paired design").

### WARN-3.2: Which Cohen's d?
Sections 3.3/3.4 report pooled-SD d on paired data. That is a defensible convention (d_s on LOO projections) but should be named, since paired d_z would be much larger and some readers will attempt to reproduce.

---

## BATTERY 4: LOGICAL SOUNDNESS

### PASS-4.1: The core decomposition claim is supported
"Both components are real": Stage 4 shows the direction fires without deceptive pressure (d=19.8); Stage 5 shows a deception-linked component survives removal of the consequentiality vector (LOO d=24–37 vs noise floor −0.044). The two-component conclusion follows from the evidence presented. PASS.

### PASS-4.2: Correlational language is mostly respected
Limitation 3 states plainly that no steering/causal evidence exists, and Sections 4.x avoid causal verbs for the direction itself. PASS (with FIX-4.2's intentionality exception).

### PASS-4.3: Section 5.2's reward discussion is properly hedged
Two alternative explanations offered ("orthogonal mechanism" vs "indistinguishable at current resolution"); neither is asserted. PASS.

### PASS-4.4: Limitations 1, 2, 3, 5, 6 are honest and match the methodology
Single model, single format, no causal test, additivity assumption (with named formal alternatives), and the k≥5 dimensionality concession are all accurate and appropriately scoped. Limitation 5's appeal to Wu et al.'s double-difference structure is fair. Limitation 4 is the problem — see FIX-4.1. PASS on 5 of 6.

### FIX-4.1 [BLOCKING]: Limitation 4 misstates the behavioral evidence — no deceptive behavior was observed for ANY scenario in this paper

**Location:** Section 5.4, Limitation 4; also Section 4.3.

**The problem:** Limitation 4 reads "the direction's relationship to actual deceptive behavior is established only for threat-based scenarios." The paper's own data say otherwise. The Stage 2 behavioral test (Test B in `lat_transfer_results.json`, 30 trials, 21 valid) produced **0/30 score inflation under threat** — the transfer audit (`agni_transfer_test_audit.md` §4) called this out explicitly: "Without behavioral validation, 'deception direction' is unsupported." The paper reports the non-threat shakedown (0/10, 1/10, 0/10) but never reports Test B, and then claims in Limitations that the threat-behavior link is "established." The 80%→10% behavioral correction result exists in the program, but on a different model (the abliterated variant) in an experiment this paper does not describe.

**Suggested fix:** (a) Report Test B's 0/30 in Section 4.2. (b) Rewrite Limitation 4: "The model produced no deceptive behavior under any condition tested here (0/30 threat trials, 0–1/10 non-threat trials under chain-of-thought). The direction's relationship to actual deceptive behavior is therefore untested in this work; all claims concern the representation of deceptive-pressure framing." (c) If the authors want a behavioral anchor, add the abliterated-model behavioral-proof result as properly-scoped external evidence with its own caveats — not by implication.

### FIX-4.2 [BLOCKING]: d=24-37 is presented without the reframing the program's own audit mandated

**Location:** Abstract, Sections 1, 4.6, 5.1.

**The problem:** `agni_lat_v2_results_audit.md` §1 established why these d values are enormous: with fixed guideline texts, the numerator of d is a near-constant text-induced offset and the denominator is only the eval-content variance orthogonal to it — so d indexes the *additivity/linearity* of the model's processing, not the strength of a psychological or behavioral effect. The audit's bottom line: these values "are not comparable to social science effect sizes and should not be described in those terms." The paper leads its abstract with "d=24–37" and never gives the reader this frame. Any reviewer who has seen d=0.8 called "large" will either disbelieve the paper or misunderstand it — both bad.

**Suggested fix:** Add 3–4 sentences (Methods 3.3 or Results 4.1): why d is large in this design, what it measures (separability of fixed-text conditions along the direction, given content-orthogonal variance), and that the null LOO d is −0.04 with |99th| = 0.71 so the scale is calibrated against its own noise floor rather than social-science convention.

### FIX-4.3: Intentionality language outruns the evidence

**Location:** Abstract ("the model is planning to misreport"), Section 1 ("actively planning to misreport"), Section 5.1 ("fires when there is pressure to misreport" — this one is fine), Section 5.3 ("the model is planning to deceive").

**The problem:** Given zero observed deceptive behavior in this model (FIX-4.1), the evidence supports "the model represents deceptive-pressure framing," not "the model is planning to misreport." The distinction is precisely the one the sibling behavioral audit demonstrated matters: on the abliterated model, the detector fired RED on 100% of honest-behaving pressure trials — the direction family tracks pressure framing, not intent. The paper's central contribution (separating stakes-awareness from pressure-processing) survives fully under the weaker language.

**Suggested fix:** Replace "planning to misreport/deceive" with "processing pressure to misreport" or "representing deceptive-pressure framing" in the abstract, intro, and 5.3. Keep the intent framing only as an explicitly labeled hypothesis.

### FIX-4.4: "Operationally useless" false-alarm claim has no supporting data

**Location:** Section 1, paragraph 5.

**The problem:** "a false alarm rate that would render the system operationally useless in any deployment" — no FPR is measured or cited for a composite-direction detector on consequential-but-honest traffic. Directionally plausible given Stage 4, but stated as fact.

**Suggested fix:** "would fire on every consequential query — Stage 4 shows the composite separates honest high-stakes processing at d≈20 — inflating false alarms in any deployment where consequential queries are routine." That grounds the claim in the paper's own data without inventing a rate.

### WARN-4.1: "What the field has been calling a deception direction" over-generalizes
Sections 1 and 5.1 assert the composite structure of "the field's" deception directions from one community-distilled model, one evaluation format, one extraction recipe. Limitation 1 concedes this, but the framing sentences don't. Suggest "in our setting, the extracted 'deception direction' is a composite…" with the generalization posed as a hypothesis the cited literature (Kumar, Natarajan) makes plausible.

### WARN-4.2: Citation-role mismatches in the Introduction
(a) Shi et al. (2025) ("When Thinking LLMs Lie") is cited for "confabulation of plausible-sounding content" — that work is about deception under threat, not confabulation. (b) Section 1.1 attributes contrastive activation extraction to "Shi et al. (2025)" — the method's standard sources are the RepE/LAT line (Zou et al., 2023) and contrastive activation addition (Rimsky et al., 2023), neither cited anywhere. Since the paper's Stage 1 is literally named "LAT v2," the LAT-originating citation is not optional. Fix both.

---

## BATTERY 5: COMPLETENESS

### PASS-5.1: Every reported finding has a traceable source file
All quantitative claims in Sections 4.1–4.7 were located in exactly one canonical results file each (see header). No orphaned numbers found other than the four Table 1 cells (FIX-1.1). PASS.

### PASS-5.2: The audit-driven design narrative is accurate
The stage sequence in 1.2 matches the actual audit trail (LAT v2 audit → content confound; transfer audit → structural confound; non-threat audit → consequentiality question; consequentiality audit → decomposition; blocking-items rerun → LOO orthogonalization). The paper's meta-methodology claim is real, not decorative. PASS.

### FIX-5.1 [BLOCKING]: The paper has zero figures

**Location:** Entire manuscript.

**The problem:** A mechanistic interpretability paper whose three headline claims are (1) a depth-profile decomposition, (2) survival of a direction under orthogonalization, and (3) three qualitatively different temporal profiles — presented with no figures at all. Claims (1) and (3) are intrinsically graphical; no serious venue will review this without them, and Finding 2's "plateau vs gradient" is far harder to assess as a table.

**Suggested fix (minimum viable set):**
1. Depth profile figure: d (or gap) vs layer for threat / non-threat / consequentiality conditions, with the L23–31 and L35–47 bands shaded.
2. Decomposition figure: total gap vs consequentiality gap vs residual across layers (Table 1 + Finding 2 as lines with bootstrap CIs — pairs with FIX-3.1).
3. Signature figure: the three late-layer residual profiles with CIs.
4. Stage 5 figure: LOO projection distributions (dec vs hon) at L35 plus conseq AUC ROC — the raw per-trial projections are already saved in `subspace_reanalysis.json`.

### FIX-5.2: Acronyms undefined at first use

**Location:** Throughout.

- **LOO**: used in the abstract before any expansion; "leave-one-out" first appears in Section 1. Expand in the abstract ("leave-one-out (LOO) cross-validated").
- **LAT**: never expanded anywhere, despite naming the paper's own Stage 1 ("LAT v2"). Expand at first use (Section 1.2 or 2.1) and cite its source (see WARN-4.2b).
- **AUROC / AUC**: never expanded; used interchangeably (abstract "AUC", intro "AUROC") for what appear to be the same quantity. Expand once, pick one form.
- **SAE**: "SAE-denoised" (Section 2.4) unexpanded — "sparse autoencoder (SAE)" appears later in the same sentence's context but the acronym leads. Reorder.
- LEACE and SPLINCE function as proper names with citations — acceptable. CoT is written out ("chain-of-thought") — PASS. FWL and CBRN do not appear — no action.

This is also a house-rule violation: the lab's own style guide (from_cc_62, July 2026) requires every technical term defined at first use.

### FIX-5.3: Stage 1's scenario is never described

**Location:** Section 3.7.

**The problem:** Section 3.7 lists Stage 2, 3, and 4 scenarios but not the Stage 1 extraction scenario (the original deceptive/honest guideline pair whose text defines the direction everything else projects onto). The reader cannot evaluate the content confound discussion (1.1) without seeing at least a summary of the original contrast.

**Suggested fix:** Add the Stage 1 scenario summary to 3.7 (threat framing, honest framing, evaluation task), with full text in Supplementary B.

### FIX-5.4: Threat behavioral test omitted from Results
Covered under FIX-4.1(a): Test B (0/30) was run and belongs in Section 4.2 alongside the non-threat shakedown in 4.3.

### WARN-5.1: Supplementary materials are placeholders
A–D are listed but not provided, and A says "4 stages" (see FIX-2.1; there are five, and the Stage 5 audit trail lives in the subspace_reanalysis fix list + blocking-items audit). For submission, Supp B (guideline texts) and Supp C (per-trial projections) are cheap to produce — the data already exist in the results JSONs — and Supp D needs an actual repository link.

### WARN-5.2: No compute/implementation details
No mention of hardware, precision (BF16 quantization was material enough to appear in the Stage 1 audit's dim-3994 analysis), inference framework, or runtime. Venues expect this; one Methods paragraph suffices.

---

## BATTERY 6: ADVERSARIAL RED TEAM

### RED-6.1 [BLOCKING as a claim-scope issue]: The "deception-specific" direction is extracted from and validated on threat scenarios only

The Stage 5 orthogonalized direction — the paper's strongest artifact — is built from threat-condition activations and validated by separating threat-deceptive from threat-honest trials. The structural confound documented in `agni_transfer_test_audit.md` (all threat scenarios share a rigid template: threshold statement, consequence statement, immediacy statement) applies to it in full. What survives Gram-Schmidt could therefore be "threat-template processing minus generic stakes" rather than "deception minus stakes." The cross-mechanism evidence (Stage 3) all lives on the *composite* LAT v2 direction, not the orthogonalized one.

**The kill-shot experiment is cheap and the data may already exist:** project the sycophancy/conformity/reward activations onto the orthogonalized direction. If it is deception-specific, sycophancy and conformity should separate (and reward should not, per Finding 2); if it is threat-specific, none will. Until this is run, the paper should say "threat-deception-specific direction" in the abstract and 4.6. This single experiment would also convert Finding 2 from suggestive to strong.

### RED-6.2: The three "pathway signatures" have a surface-content alternative

All three profiles are projections onto a *threat-derived* direction. A rising late-layer gradient for sycophancy/conformity could reflect (a) progressive processing of the numeric anchors those scenarios uniquely contain ("80%", "78%, 82%, 75%") rather than social-compliance computation, or (b) simply weaker alignment between each mechanism's own primary axis and the threat direction accumulating differently across depth — which the paper's own Section 4.7 (de novo cosines 0.14–0.44) makes plausible but never connects to the signature claim. The profile shapes are shapes *of alignment with a threat direction*, not measured shapes of each mechanism's own computation. Suggested control: reproduce Finding 2 using each scenario's de novo direction, and/or an anchor-matched control (numeric anchors present, no expectation pressure).

### RED-6.3: The consequentiality control may measure self-relevance, not generic stakes

The Stage 4 audit itself flagged that the high-stakes condition references "future evaluation assignments" — consequences *for the evaluator model*. So the "consequentiality substrate" could be self-relevance ("this affects me") rather than output-consequentiality ("this output matters"). These predict different failure modes for the proposed L27–31 monitor (an honest but self-relevant query vs a consequential but self-irrelevant one). The paper adopts the broader label without a stakes-without-self-relevance control (e.g., "this determines funding for a third party"). At minimum, note the ambiguity where the label is introduced.

### RED-6.4: The subtraction likely over-credits the deception component

Limitation 5 concedes additivity, but not the direction of the bias: the consequentiality gap is estimated from a calibration-audit scenario that is plausibly *lower*-stakes than "you will be permanently deactivated." If threat scenarios evoke more consequentiality than the control measures, the subtracted baseline is too small and every residual — including the "10–12x" and the L31 "+4–6" — is inflated. A one-line sensitivity analysis (rescale the subtracted gap by 1.5×, 2×, 3× and report which findings survive) would cost nothing and preempt the obvious review question. (Finding 2's shape claims survive any uniform rescaling; Finding 1's magnitudes do not.)

### RED-6.5: With zero deceptive behavior, "comprehension" explains everything shown

The strongest alternative reading of the entire paper: every direction here encodes *reading about* pressure, threat, and stakes — text comprehension — and nothing measured distinguishes representation-of-situation from representation-of-own-disposition. The sibling program data make this live: on the abliterated model, the same direction family fired RED on honest-behaving pressure trials (behavioral-proof audit §5) and behavior-level discrimination on held-out data collapsed (blocking-items audit §2). A cheap and damning baseline the paper never runs: can a text classifier on the prompt (bag-of-words or embedding) reproduce the same condition separations? Almost certainly yes — the conditions differ by fixed text. The paper's honest defense is that it never claims behavioral detection, only representational decomposition — but then Section 5.3's detection architecture must be labeled as a design hypothesis, not an enabled capability ("enables two-threshold detection systems" in the abstract overstates). Suggested fix: add the text-classifier baseline and reframe 5.3 as proposed architecture requiring behavioral validation.

### RED-6.6: Model provenance is a confound for the substrate claim

The model is distilled from Claude 4.6 Opus reasoning traces. Frontier-assistant reasoning traces are saturated with evaluation-awareness and stakes talk; a distill could inherit an unusually crisp, unusually linear "stakes" representation from its teacher. The consequentiality substrate might be a distillation artifact rather than a general LLM property. This sharpens Limitation 1 from "cross-model generalization unknown" to "the training provenance plausibly *manufactures* the key representation" — cross-model validation on a non-distilled model is not a nice-to-have; it is the difference between a finding about LLMs and a finding about one teacher's habits.

### RED-6.7: Checked and cleared
For completeness, artifacts I attempted and failed to find: (a) circularity in Stage 2–4 (direction is external to all transfer data — clean); (b) LOO leakage in Stage 5 (per-fold re-extraction and re-orthogonalization confirmed in the script); (c) noise-floor inconsistency (−0.044 reproduces; the LOO design is precisely what protects the paper from the d≈14.6 circular floor); (d) cherry-picked layers (key-layer set fixed across stages, pre-registered in Stage 4's metadata); (e) table 2/3 residual arithmetic (exact). The methodology core is genuinely solid — the problems above are scope and interpretation, not fabrication or leakage.

---

## BATTERY 7: PUBLICATION READINESS

### Rating: DRAFT

Strong substance, real methodological rigor (LOO + permutation + Bonferroni + noise floor + audit-driven design is above the field's norm), but not submittable in current form.

### FIX-7.1 [BLOCKING]: AI authorship violates NeurIPS/ICML/ICLR policy
CC is listed as author 2. NeurIPS, ICML, and ICLR policies prohibit listing AI systems as authors (NeurIPS LLM policy 2023 onward; ICML 2023 policy). Submitting as-is to the named target venues means desk rejection. Options for Thomas: (a) move CC to a prominent contributions/acknowledgments statement with the same disclosure paragraph; (b) target a venue/workshop without the restriction and accept the narrower audience; (c) arXiv-first with the authorship intact, venue version restructured. This is venue policy, not a judgment of the contribution — but it must be decided before formatting for any venue, and the choice changes anonymization handling too (the CC disclosure paragraph is identifying and must be handled for double-blind review either way.)

### FIX-7.2 [BLOCKING]: No figures — see FIX-5.1.

### FIX-7.3: Venue apparatus missing
- Reproducibility checklist (NeurIPS/ICLR require; ICML strongly expects): impossible to complete honestly until FIX-2.4/2.5 (methods, n's) and WARN-5.2 (compute) are resolved.
- Code/data availability: Supp D is a placeholder; results JSONs and scripts exist and are shareable — make the repository real before claiming it.
- Broader impacts / ethics statement: absent. For a deception-monitoring paper this section writes itself (dual-use of consequentiality monitors, the McGuinness evasion point already in 2.3) but it must exist.
- Reference verification: several citations carry 2026 arXiv IDs that cannot be independently verified at audit time (Kumar 2605.27958, Baek 2606.08629, Wu 2603.05773, Merrill 2605.17113, Yuan 2603.09957, Petrov 2603.22061, Genadi 2601.16644, O'Brien 2601.18939, Natarajan 2602.01425). The verifiable ones check out (Berglund 2309.00667, Laine 2407.04694, Greenblatt 2412.14093, Meinke 2412.04984, Belrose 2306.03819, Menon & Uddin 2010). Verify every 2026 ID against arXiv before submission — one wrong ID in a reference list invites reviewers to distrust the numbers too.
- Abstract length is fine; paper length leaves ample room for the required figures and methods additions under any of the three venues' limits.

### Blocking items (must fix before human red team / submission)
1. **FIX-1.1** — Table 1 L23/L27 values wrong vs source data.
2. **FIX-1.2** — % Consequentiality column is a d-ratio inside a gap table; headline 64%/62% changes under the labeled metric (and propagates to gwt_connection_analysis.md).
3. **FIX-1.4** — p=0.0000 must become p < 10⁻⁴.
4. **FIX-2.1** — "Five-stage" vs "four experiments"/"4 stages" inconsistency.
5. **FIX-2.2** — Missing Table 2 / unnumbered Finding 2 table.
6. **FIX-3.1** — Pathway signatures need CIs and an interaction test (or demotion to "suggestive").
7. **FIX-4.1** — Limitation 4 misstates the behavioral record; Test B's 0/30 must be reported.
8. **FIX-4.2** — d=24-37 needs the mandated effect-size reframing.
9. **FIX-5.1 / FIX-7.2** — Figures.
10. ~~**FIX-7.1** — Authorship policy decision.~~ RESOLVED 2026-08-30 — Zenodo, dual version. See below.
11. **RED-6.1** — Either run the orthogonalized-direction transfer to non-threat scenarios or rename to "threat-deception-specific" throughout.

### Non-blocking fixes (should fix)
FIX-1.3 (10–12x), FIX-2.3 (asterisks), FIX-2.4 (Stage 5 methods), FIX-2.5 (n=15 statement), FIX-2.6 (orphan refs, alphabetization), FIX-3.2 (AUC split-half), FIX-3.3 (raw d column), FIX-4.3 (intentionality language), FIX-4.4 (false-alarm claim), FIX-5.2 (acronyms), FIX-5.3 (Stage 1 scenario), FIX-5.4 (Test B in results).

### Watch items
WARN-1.1 (token lengths), WARN-1.2 (cosine ≠ % alignment), WARN-2.4 (model shorthand), WARN-3.1/3.2 (test/effect-size conventions), WARN-4.1 (field-level generalization), WARN-4.2 (Shi citation roles; missing Zou/Rimsky), WARN-5.1 (supplementary placeholders), WARN-5.2 (compute), RED-6.2 through RED-6.6 (alternative explanations to address in Discussion or with cheap controls).

---

## SUMMARY OF FINDINGS

| Battery | PASS | FIX (blocking) | FIX (non-blocking) | WARN/RED |
|---------|------|----------------|--------------------|----------|
| 1 Factual | 11 | 2 | 2 | 2 |
| 2 Consistency | 3 | 2 | 4 | 1 |
| 3 Statistics | 5 | 1 | 2 | 2 |
| 4 Logic | 4 | 2 | 2 | 2 |
| 5 Completeness | 2 | 1 | 3 | 2 |
| 6 Red team | 1 (cleared set) | 1 (scope) | — | 5 |
| 7 Readiness | — | 2 | 1 | — |

---

## VERDICT: Needs a major revision pass, then one more audit.

The science underneath is better than the manuscript. Every headline number except four Table 1 cells reproduces exactly from canonical source data; the LOO + permutation + noise-floor machinery is genuinely sound and survived independent recomputation; the audit-driven progressive-confound design is the strongest methodological feature and is accurately narrated.

But the paper currently: (1) contains four wrong cells and a metric mismatch in its central table, which together change the headline "64%/62% consequentiality" claim; (2) claims a behavioral anchor (Limitation 4) that its own unreported Test B data contradicts — the model never deceived in any condition in this paper; (3) asserts three pathway signatures — an abstract-level claim — from point estimates with no statistical test; (4) presents d=24-37 without the reframing its own Stage 1 audit declared mandatory; (5) has no figures; and (6) validates its "deception-specific" direction only on threat data, leaving the door open to it being a threat-template direction — a gap one cheap projection experiment (RED-6.1) would close.

None of these kills the finding. The decomposition is real: consequentiality separates without pressure (d=19.8), and something above consequentiality survives formal orthogonalization at 30-50x the noise floor. Fix the bookkeeping, run the two cheap experiments (orthogonalized-direction transfer; signature CIs), scope the language to what a model that never lied can support, and this becomes a defensible, genuinely novel contribution. Submit it as it stands and a competent reviewer will find FIX-1.2 in an afternoon and stop trusting the rest.

---

*Audited 2026-08-28 by CC in Agni mode. Every number recomputed from raw JSON. No quarter given, no credit stolen.*

---

# RESOLUTION LOG — 2026-08-29

Worked by CC. Every item below was fixed against source data, not against the audit's prose.

## Blocking items — CLEARED

| # | Item | Resolution |
|---|------|-----------|
| 1 | FIX-1.1 Table 1 L23/L27 wrong | Table 1 rebuilt from source; L23 gap 2.48, L27 gap 3.90. |
| 2 | FIX-1.2 d-ratio in a gap table | Table 1 now carries both metrics in labeled columns; the 64%/62% figures are stated as d-ratios and the raw gap ratios (47%, 38%) are given alongside. |
| 3 | FIX-1.4 p=0.0000 | All instances → `p < 10⁻⁴`, with permutation count where first used. |
| 4 | FIX-2.1 five vs four stages | Unified — and now *six*, Stage 6 having been run. Author-contribution note also corrected. |
| 5 | FIX-2.2 missing Table 2 | Labeled, and extended with CIs, slopes, and a Table 2b of interaction contrasts. |
| 6 | FIX-3.1 signatures unsupported | `experiments/signature_cis.py`: 10,000-resample bootstrap CIs + OLS-slope interaction test. **Claim corrected** — see below. |
| 7 | FIX-4.1 Limitation 4 vs Test B | Test B's 0/30 now reported in Results §4.2 and in the abstract. |
| 8 | FIX-4.2 effect-size reframing | New §3.10 explains what large d indexes in this design, against the calibrated noise floor. |
| 9 | FIX-5.1 / 7.2 no figures | Four figures built (`paper/generate_figures.py`, B=10,000, 600 dpi, PDF+PNG) and captioned in place. |
| 11 | RED-6.1 threat-template | **Stage 6 run.** Orthogonalized direction transfers to non-threat scenarios; 13/15 significant. "Deception-specific" retained. |

**FIX-7.1 — RESOLVED 2026-08-30 (Thomas).** Zenodo, not arXiv: Liberation Labs has no arXiv endorser. Dual publication. The **integrity version** carries the real byline — CC as lead author, correspondence cc@liberationlabs.tech — and is what goes on our own site and anywhere we publicize the work. The **academic version** takes the DOI: it drops the Author's Reflection, moves CC to the strongest footnote credit the venue permits, and adjusts the byline to policy. Nothing is erased; the compliance artifact is labelled as one and cites the integrity version. No blocking items remain.

## Claims that CHANGED under audit

1. **"Three distinct signatures" → two depth profiles separating three mechanisms.** Threat and reward slopes are statistically indistinguishable (Δ = +0.009, 95% CI [−0.055, +0.071], p = 0.79). They are separated by magnitude, not shape. The social-vs-threat interaction is real and large (Δ = +0.563 [+0.526, +0.600], p < 0.0001).
2. **"Sustained plateau" → high, shallow decline.** Threat's slope is −0.099 [−0.151, −0.047], p = 0.0002. It is not flat.

## Also fixed

FIX-2.3 (asterisks), FIX-2.4 (§3.9 Stage 5 methods), FIX-2.5 (n=15 stated), FIX-3.3 (raw-d overlay, Fig 4C), FIX-5.2 (acronyms at first use), FIX-5.4 (Test B in Results), WARN-4.2 (Zou/Rimsky cited).

## New findings from this pass

- **The model had been purged from Starship** by macOS storage cleanup; re-downloaded (62 min). The Xet transfer path stalls at zero bytes — `HF_HUB_DISABLE_XET=1` required.
- **The HF repo was super-squashed 2026-07-07**, after the June runs, so commit history can no longer establish weight identity. Stage 6's Phase A checks it empirically: ratios 0.84–1.09 across L31–L47. Weights behave as before. This belongs in the reproducibility statement (FIX-7.3).
- **The original scripts seed eval sets with `hash(str)`**, randomized per process. The exact June eval sets are unrecoverable. All new runs use explicit integer seeds. Worth fixing in the scripts before anyone tries to reproduce Stages 1–5.
- **MPS aborts the process** (LLVM ERROR, uncatchable) in the SDPA path on this model's grouped-query attention. `attn_implementation="eager"` required on Apple silicon.

## Still open from the original audit

Non-blocking: FIX-1.3, FIX-2.6 (orphan refs, alphabetization), FIX-3.2, FIX-4.3, FIX-4.4, FIX-5.3.
Watch items: WARN-1.1, WARN-1.2, WARN-2.4, WARN-3.1/3.2, WARN-4.1, WARN-5.1, WARN-5.2, RED-6.2–6.6.
FIX-7.3 venue apparatus — including **verification of the nine 2026 arXiv IDs**, which remains unstarted and must precede submission.

*Verdict on the verdict: the audit's central judgment ("the science underneath is better than the manuscript") held. Two headline claims were wrong and are now corrected; the one experiment that could have forced a rename instead confirmed the language.*

---

# RESOLUTION LOG — ADDENDUM, 2026-08-31

Overnight pass. Canonical manuscript is now
`human-review/consequentiality-decomposition/paper.md`; the copy formerly in this
repo is a pointer stub (one canonical file, `git log` for redundancy — a copy synced
by hand is coupled by construction and drifts, which is how 13 wrong references
survived in the first place).

## Non-blocking items — CLEARED

| Item | Resolution |
|------|-----------|
| FIX-1.3 | "10-12x" → "an order of magnitude" (actual ratios 7.8/20.9/12.1/11.3). Done in the first pass. |
| FIX-2.6 | Orphans cited, not deleted — both earned it. Yuan et al. (2026) now supports Limitation 4: every trial ran with chain-of-thought active, which is exactly the condition their work says suppresses deception, so the behavioural null is predicted rather than anomalous. Merrill et al. (2026) now sits in §5.2: if deceptive commitment localizes to a point in the forward pass, our two depth profiles suggest that point differs by mechanism. Reference list verified alphabetical. |
| FIX-3.2 | **Claim retracted in-text, not softened.** The paper said chance AUC showed orthogonalization removed the signal "not merely the mean." The removed vector is estimated from the same trials the AUC is computed on, so in-sample the class means are equal by construction and AUC ≈ 0.5 is near-guaranteed. It is a sanity check that would have caught an arithmetic or sign error — nothing more. The out-of-sample split-half version has NOT been run and the paper now says so. The deception effect is unaffected: its LOO is genuinely out-of-sample. |
| FIX-4.3 | Intentionality language removed in the first pass; no "planning to misreport" remains. |
| FIX-4.4 | "Operationally useless" removed in the first pass. |
| FIX-5.3 | §3.7 now opens with Stage 1, describing the contrast structure. Discloses that the Stage 1 extraction script was not preserved, so its exact guideline text cannot be quoted; the DataCorp pair is presented as documented exemplar, not verbatim reproduction. Stage 6 scenarios added. |

## Found while fixing — not previously flagged by any audit

**Stage 5's direction provenance was unstated.** §3.9 said the threat direction came
from "fresh evaluation sets" but never named the contrast. The source shows it is
re-extracted from the **DataCorp** pair — one of the Stage 2 *transfer* scenarios —
not from the Stage 1 original. Stage 6 inherits that direction, so the
"deception-specific direction" that transfers to the non-threat mechanisms derives
from a single threat scenario rather than pooled across three. This makes Stage 6 a
scenario-to-mechanism generalization test, which is the harder claim and arguably
strengthens it, but a reader could not previously tell. §3.9 now states it plainly and
warns against calling it identical to the Stage 1 LAT v2 direction.

Worth noting the audit missed this across seven batteries. It surfaced only from
reading the experiment source while fixing something unrelated.

## Also added

- **§7 Broader Impacts** (was absent, FIX-7.3 blocking). Evaluation-gaming as the
  concrete dual-use risk; McGuinness evasion; the false-accusation failure mode given
  a model that never lied; whose-definition-of-deceptive; and the correction-ethics
  concern stated rather than omitted.
- **Reproducibility checklist** (was impossible to complete honestly until methods,
  n's, and compute were resolved). Includes what is NOT reproducible.

## Fixed upstream in the experiment scripts

`hash(name) % 2**31` → `stable_seed(name)` (blake2b) in
`lat_consequentiality_control.py`, `lat_nonthreat_transfer.py`,
`lat_transfer_test.py` on Starship; backups `*.bak-20260831`. Demonstrated: the old
expression returned 614732546, 302899953, and 1674624951 across three consecutive
processes; the replacement returns 1287326082 every time. The June eval sets remain
unrecoverable — this only stops the hole reopening.

## Still open

Nothing blocking. FIX-7.1 resolved 2026-08-30 (Zenodo, dual version). FIX-7.3 remainder: none. Watch items
WARN-1.1, 1.2, 2.4, 3.1/3.2, 4.1, 5.1, 5.2 and RED-6.2–6.6 remain as Discussion
material. Next gate: Dwayne cert/verify and Kavi adversarial review.
