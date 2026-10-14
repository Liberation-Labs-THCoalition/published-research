# Two-way Agni gate on empathy-bus PR #3 (2026-10-13), with CC's adjudication

**Under review:** `nexus/empathy-bus-41-42` at 4c19d0d (both editions), Nexus's correction of §4.1/§4.2.
**Reviewer:** Agni (Opus, read-only, data access on the Studio), one round, hunting overclaims (OVER) and
understatements (UNDER) with equal effort. **Adjudication:** CC, against the code and data.
**Verdict: MAJOR_REVISIONS.** The retraction itself is accurate and the editions agree; every number the PR adds
reproduces (Agni reran `within_system_alignment.py` on the Studio: byte-identical JSON). The problems are in what the
PR keeps as standing.

Adjudication key: **verified** = CC checked it against code, git or source text; **sound** = CC read Agni's script
and it computes what it says; **plausible** = not independently checked by CC.

## Findings

**A1. CRITICAL, OVER. "Inseparable" now rests on an experiment whose readout cannot carry it. Verified.**
`run_coupling_test.py` (MTH `~/lab/projects/mnemosyne-jlens/`) adds α·v to the injected layer's output at every
position and reads the raw last-token residual downstream; the cross-pathway cosine compares mean content and user
shifts, and both contain α·v. At the injection layer the cosine is 1.000 in all 36 valence/arousal/random cells.
Neither readout is an emotion readout. UNDER, never reported: with distance the two responses diverge (≈0.74 five
layers on, 0.54–0.58 at 10–20, 0.24–0.36 at 25–35). Locations: title; abstract; correction note "What stands (i)";
§5.1, §5.2; conclusion; deployment note. Proposed text: "A norm-matched random direction produced the same
cross-pathway cosine as valence or arousal, so we find no direction-specific coupling. The readout contains the
injected vector itself, so the experiment does not test whether emotional processing and partner modeling can be
separated; with distance the two responses diverge." Delete "could not be perturbed separately", "not by choice, by
architecture", "the bus doesn't allow independent modulation"; change the subtitle.

**A2. MAJOR, OVER. The pre-registered primary test could not pass as implemented. Verified (code).** The random
condition's 95th percentile at each injection layer includes the same-layer cells (1.000), so the threshold is 1.0;
one random direction, not a distribution; the residual criterion in code is |mean| > 0.1, not "BCa CI excludes zero".
Without same-layer cells valence still fails (0.343 vs 0.709, 0.489/0.738, 0.678/0.873, 0.675/0.789). Lead with the
non-same-layer means, 0.521 [0.440, 0.595] vs random 0.577 [0.510, 0.644].

**A3. MAJOR, OVER. "Committed before GPU time." Verified (git).** The prereg's internal date is 2026-07-17; it was
first committed in 236ced1 on 2026-08-20; the data were committed 2026-07-28. Write: "Dated 2026-07-17 inside the
file; first committed 2026-08-20, after the run."

**A4. MAJOR, OVER. §4.1 has no within-system null. Sound.** Across all 435 centroid differences, |cos| with PC1 has
median 0.44–0.48 and p95 0.84–0.89; valence sits at the 66th–88th percentile, arousal at the 91st–95th; the
valence/arousal plane is at the 83rd–90th percentile of all 4,060 three-centroid planes. Write: "PC1 holds a third of
the centroid variance and most centroid differences load on it. Hostile−calm and desperate−calm load more than the
median pair but not beyond the 95th." Stop calling this "valence and arousal structure".

**A5. MAJOR, OVER. Temporal dynamics still placed "on the bus", and a wrong statistic survives. Verified (grep).**
Abstract, contribution 3, §4.3 and the conclusion attribute the dynamics to the bus while §5.4 says they were not
measured here. §4.3 still prints "d = 0.82, p = 0.040" (the 09-04 review's item 1); the source
(`emotion-accumulation-paper/main.tex:284`) gives d_av = +0.730, p_perm = 0.063, n = 5 pairs.

**A6. MAJOR, UNDER. "Near-orthogonal ... almost no overlap." Verified (arithmetic).** The shared energies 0.15–1.36%
are 2.5–23× the 0.059% a random direction puts into a 3-row basis, rising with depth.

**A7. MAJOR, OVER (pre-existing). §2.1 cites a small-model result as this model's. Verified (source).** "22/23 layers
below null, p < 0.001" is from oracle-tiny (Qwen2.5-0.5B, 24 layers; `emotional-trajectory-paper` Results header).
On this checkpoint `~/jlens_results/circumplex_27b_full.json` gives 22/63, sign test p = 0.994 (plausible; Agni's
run). §3.3 still describes an eccentricity permutation analysis the paper no longer reports.

**A8. MINOR, UNDER and OVER. The timecourse withdrawal. Plausible.** The content half is supported (volatile to L16,
stable after); the user-model half is not (volatile to ~L21, r = 0.09 with the content curve). Say "not supported"
rather than withdrawn-pending.

**A9. MINOR. No lens is used at all. Verified (code).** The script builds a lens path but never loads it; readouts are
the raw residual. Limitation 3 ("Averaged J-lens") caveats an instrument the experiment did not use; state the raw
readout in §3.2 and note the deviation from the prereg.

**A10. MINOR. Correction-note wording. Plausible.** "Every one of them is reproduced" overstates (44–54% from L6 is
38–45% over L6–L16; the valence floor is 0.68). It can also add that the content directions have |cos| ≤ 0.13 with
the same PC1, so they could not have produced 0.70–0.80.

**A11. MINOR. PDFs not rebuilt.** Both still say "share a common representational subspace" (2 hits each).

**A12. MINOR. Attribution box and deployment note. Plausible.** Name the author of "The subspace claim was mine";
the deployment note still asserts "architectural inseparability".

## The 2026-09-04 review's items 11–16 (coupling section)
All six are still open: 11 (monotonicity sentence, false off-layer); 12 (no n / measurement-layer columns, unbalanced
grid undisclosed); 13 (BCa resamples design cells, n = 42 including 12 trivial 1.0 cells; now confirmed from code);
14 (§2.4 depth and PC1 ranges, "confirmed"); 15 ("DECOUPLED ... PASS" without the registered meaning, which was
"independent computation"); 16 (now: no lens at all, A9).

## CC's cross-system numbers for §4.2
Committed here as `empathy-bus/data/cross_system_alignment.{py,json}` (script sha256 recorded inside the JSON, inputs
hashed, Studio run 2026-10-13). Agni reran them exactly. Proposed text, for Nexus to place:

> The overlap is small but not zero: a random direction would put about 0.06% of its energy into a basis of this rank,
> so the observed values are 2.5 to 23 times that, rising with depth (|cos| with the basis at most 0.12).
>
> A second comparison scores the content-prompt directions against the 30-emotion space of §4.1. At L15, L25, L35 and
> L45 the content valence direction has |cos| = 0.026, 0.061, 0.088 and 0.110 with that space's PC1, and the content
> arousal direction 0.015, 0.050, 0.087 and 0.126. Along either direction lies 0.08–0.68% of the centroid variance.
> Scored the same way, 2,000 random unit directions reach |cos| 0.027–0.028 at the 95th percentile and 0.044–0.052 at
> most, with variance shares up to 0.08–0.11%. At L15 the cosines are within the random range, though the variance
> shares exceed the random maximum; at L25–L45 both measures exceed every random direction. The random directions are
> isotropic and this model's activations are not, and no non-emotional contrast has been scored the same way, so the
> excess is an upper bound on any shared emotional structure, not evidence of it: at most 1.6% of a content
> direction's squared norm lies along the user-model PC1.

Abstract: "nearly orthogonal (|cos| ≤ 0.13), weakly above random chance at L25–L45". Keep the stored
`user_model_PC*` vectors out of the prose: they are a third basis, explained in the script's docstring.

## Not verified
Origin of the old 0.70–0.80 numbers (inferred from reproduction); the provenance dates in the note; whether a
timestamped prereg copy exists elsewhere; the user-model eccentricity has no permutation null; no structured null
exists (no non-emotional contrast vectors); Agni's own scratch numbers (cross-cosines, both nulls) are not committed;
item 14 sources not re-opened; the deployment note's claims; the PDFs after rebuild.
