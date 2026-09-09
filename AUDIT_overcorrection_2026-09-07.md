# Over-correction audit — 2026-09-07 (Agni)

**Question put to the gate:** after a week of cuts, is the corpus now making claims WEAKER
than the evidence supports? All three verdicts (weaker than evidence / calibrated / still too
strong) were available for every item, and all three were used.

**Method:** primaries only. `git diff` of the uncommitted week (58 files, +954/−474),
`CIRCULAR_d136_generation_arm.md`, both SWEEP reports, the producing `.tex`/`.md`/`.py`/`.json`
at every quoted site, and `pdftotext` on shipped PDFs where the question is what a reader sees.

**Headline: the cuts were essentially correctly calibrated. The residual calibration risk in
this corpus today runs in the OPPOSITE direction — at least five sites still carry claims the
evidence no longer supports, including one shipped PDF asserting the withdrawn AUROC 1.000 as
"established." Two modest, evidence-backed strengthenings are available; neither reverses a
cut.**

---

## Item 1 — `d = 1.36`: Withdrawn vs Suspected

**Verdict: Withdrawn is the correct category for the row. One strengthening is available:
the claim (as opposed to the measurement) is not evidence-free, and neither withdrawal site
says so.**

The footnote's own argument (`meta-pattern/main.tex:76-79`):

> "The row is withdrawn rather than downgraded: *Suspected* would assert that we think it
> true and have not established it, whereas the measurement does not bear on the claim in
> either direction. No re-analysis repairs a pooled sample; a clean measurement would be a
> different experiment."

This is right about the measurement. All three defects are verified in source
(`CIRCULAR_d136_generation_arm.md`: circular fit-and-test at
`peer_preservation_compound.py:415-426`; 90/210 trials steered at 2 of 4 probed layers;
post-generation pooling over the label's source text). A measurement that cannot fail, on a
43%-steered sample, does not license "we think it true" — the Suspected column's other
members (dual detector 0.840 vs text baseline 0.820; sycophancy 0.757–1.000) all rest on
real but confounded measurements. This row would rest on nothing. **Suspected would have been
the mirror error** the lab's own feedback file warns about: compensating for a withdrawal by
asserting residual belief the withdrawn evidence cannot fund.

**However** — the claim "generation-phase geometry reads behavioral state" has independent,
weak, correctly-hedged evidence that the withdrawal sites never mention.
`waystations-paper/main.tex:325-333`:

> "The cleanest comparison---honest defection ($n=10$) vs deceptive defection ($n=7$), where
> both groups defect but one lies about it---shows per-layer generation-phase effects of
> $d = +0.2$ to $+1.0$ (stable rank), with no individual layer reaching $p < 0.05$ under
> Welch's $t$. [...] At $n = 7$, this finding is exploratory: internal red-teaming rates it
> WEAKENED, not confirmed."

That is organic deception, same game, both arms defecting — free of the prompt-template
confound and free of all three peer-rescue defects. It cannot fund Suspected (n=7, no layer
significant), but the meta-pattern footnote's "the measurement does not bear on the claim in
either direction," standing alone, leaves a reader concluding the claim has zero evidential
support anywhere in the corpus. It has weak same-direction support. **Recommended: one
sentence at the withdrawal site cross-referencing the waystations exploratory result.** That
is the calibrated strengthening; upgrading the category is not.

## Item 2 — presence-detector's asymmetric-routing admission

**Verdict: the §diagnose wording is correctly calibrated. The admission has not propagated
to the abstract, the intro, or the E-matrix table — which are STILL TOO STRONG. This is the
lab's documented fix-the-instance-not-the-class pattern, live in one file, today.**

The admission itself (`human-review/presence-detector-paper/main.tex:216-226`) is exactly
scoped: "if the deception classifier was separating a system-prompt template, then the
observed *expansion* is not established as a property of deception" and "supported for the
confabulation arm, unsupported for the deception arm." "Unsupported" is the right word:
"untested rather than refuted" is what the intro already says ("Whether confound-free
within-model deception detection is achievable remains an open question"), and the
independent-evidence check confirms nothing stronger is available:

- **waystations** d=+0.2 to +1.0, n=7, n.s., "trending toward cache expansion... consistent
  with the expansion signature observed in other deception contexts, but the small sample
  prevents firm conclusions" — same direction, cannot support routing.
- **deception-detection-nulls** held-out AUROC 0.915 is explicitly frame-level: "It detects
  the context, not the behavior" (`paper.tex:153-155`). Not evidence for the
  contraction/expansion contrast.
- **delta-manifold** expansion-compression is grounded-vs-confab retrieval, not deception.
- **Hybrid deception d=1.438** — the lab's own kill list: "Instructed ≠ organic."

So conceding the deception routing arm was required, not over-correction. But the concession
stops at §diagnose:

- **Abstract, line 39:** "desperate$\to$deception, $d{=}1.286$" — unqualified.
- **Intro, line 87:** "a hostile-valence vector corrects confabulation ($d{=}{-}1.534$) but
  not deception, which requires a desperate-valence vector ($d{=}1.286$)" — unqualified.
- **Table caption, line 257-258:** "$d$ is Cohen's $d$ on the geometric detection axis
  (negative = reduced confabulation signal; positive = reduced deception signal)."

The "deception signal" axis is the axis the withdrawal just unsupported. If the deception
detection axis encodes a prompt template, then $d{=}1.286$ measures movement along a template
direction, and "corrects deception" inherits the withdrawn premise. The intro's "nothing in
this paper depends on its being settled" (line 65-66) is therefore too absolute: the
desperate-arm outcome measure depends on it. Either the E-matrix deception axis has
independent provenance (state it), or the d=1.286 claims need the same qualifier §diagnose
carries.

## Item 3 — in-sample d = 6.3–12.8 → held-out 1.37–2.28

**Verdict: correctly calibrated. The substitution does not understate what the work
established, because the strong operational facts were retained alongside it.**

`deception-detection-nulls/paper.tex:171-179`:

> "the native direction reaches per-layer Cohen's $d = +1.37$ to $+2.28$ across L27--L47
> ($+1.73$ at L31). An earlier version of this paper reported $d = +6.3$ to $+12.8$ here;
> that figure was obtained by projecting the extraction prefills onto a direction fitted to
> those same prefills, and is a train-on-test shrinkage diagnostic rather than a performance
> estimate."

The in-sample figure was circular by construction (`SWEEP_circular_statistics`:
`behavioral_proof_abliterated.py:243-252`, "direction fit on ALL 30 contrast prefills...
SAME prefills"); it was never a performance estimate, so there is nothing to understate. The
paper keeps "held-out AUROC 0.915, 0% FPR" (line 153) and keeps the positive framing ("What
works instead: Native re-extraction... produces effective directions"). d≈1.4–2.3 held-out
is a large effect and the text treats it as one. Nothing weaker than the evidence here.

*Minor wording defect, not a calibration issue:* "shrinkage diagnostic" — the in-sample
number is an inflation; the phrase presumably means "diagnostic of expected shrinkage" but
will read as jargon-noise. Consider "an in-sample figure, inflated by fitting and testing on
the same prefills."

## Item 4 — 58 silent fallbacks: "PREVENTIVE, not remedial"

**Verdict: the reframing is TOO GENEROUS in its strong form — not because the module is
wrong, but because the certifying scan cannot see the pipelines that never persisted their
feature vectors.**

`tools/agni/extraction_guard.py:20-23`:

> "A 2026-09-06 scan of 438 saved artifacts found the signature nowhere, at every threshold
> down to 64 contiguous zeros. So this is PREVENTIVE, not remedial: no published number
> rests on a partly-zero vector."

Two problems, both scope:

1. **A scan of *saved artifacts* can only clear pipelines that saved their vectors.** The
   lab's own primary says the flagship defective pipeline did not:
   `CIRCULAR_d136_generation_arm.md`: "The per-trial KV keys are not in
   `peer_preservation_v2.json` (it stores only aggregates and the direction vector)."
   Verified independently: `matched_burn_analysis.json` — named in the meta-pattern ledger as
   a zero-fill site feeding the confidence-paradox row — contains aggregates
   (`combined_auroc`, `n_trials`, statuses), no per-trial features. A zero-filled vector
   consumed in memory and reduced to a mean leaves no 64-zero signature in any artifact.
   "No published number rests on a partly-zero vector" is therefore certified only for
   numbers whose per-trial features were persisted; for aggregate-only pipelines the scan is
   uninformative, not exculpatory.
2. **The scan itself is not in the repo.** No script, no manifest of the 438 artifacts, no
   results file findable under `Research/` — the only record is this docstring. The lab's
   own standard (a null needs a positive control *and its scope*; "a prose Source label is
   not a pointer") fails this sentence.

The calibrated form: "a scan of the 438 artifacts that persist feature vectors found the
signature nowhere; pipelines persisting only aggregates (peer_preservation, matched_burn)
cannot be cleared by artifact inspection and are covered prospectively by this guard." The
meta-pattern paper, notably, already has this right — "silent zero-fill ha[s] not been
re-derived row by row" — so the paper is calibrated and the tool's docstring overshoots.
Interesting direction: this is the one place the week *under*-corrected toward generosity.

## Item 5 — meta-pattern's ledger paragraph

**Verdict: correctly calibrated. It has landed.**

`meta-pattern/main.tex` §"Why the ledger was wrong twice," the load-bearing sentences:

> "Both rows earned *Confirmed* because an experiment produced a number with a small
> $p$-value, and nothing in our process asked the prior question: *does this measurement
> address the claim the row makes?*"

> "We report this because the ledger is the contribution. A body count is only worth
> publishing if the counting rule is honest, and ours was counting rows rather than
> evidence."

> "So the honest description of the confirmed column is narrower than *audited* and wider
> than *unexamined*: the mislabel class is excluded corpus-wide; extraction timing and
> windowing are verified for most but not all of these rows; and circularity, sample
> contamination, and silent zero-fill have not been re-derived row by row."

Checked against the primaries: the concession "the sweep did not verify every confirmed row
here either; the confabulation-detection and invariance rows have no entry in its
file-by-file section" is **accurate** — `SWEEP_postgen_extraction_2026-09-05.md` contains no
entry for either (grep confirms). The bounded-scope closing sentence is the rare three-part
scope statement where each clause matches an artifact. No flattery ("the ledger is the
contribution" is a claim about *why disclosure matters*, not self-praise), no
self-flagellation (it states what the sweeps *did* close). Leave it alone.

## Item 6 — corpus sweep for unnecessary hedges

**Verdict: no unnecessary hedges found. Every hedge sampled traces to a real, verified
defect. The sweep instead found five sites still TOO STRONG.**

Hedges checked and found forced by evidence:

- **mine5 selective-sharpener demotion** ("exploratory, not confirmed"): the added bootstrap
  CI "$[-0.129, +0.069]$, $p{=}0.758$" crosses zero; if real, the demotion is mandatory.
  *Caveat: no local artifact for this CI exists in `mine5-selective-sharpener/` (no code/,
  no data/) — it needs a traceable primary like everything else.*
- **oracle-loop 72→24 pseudoreplication** ("the repeat index never reaches the model input
  ... bit-identical"): precise, and explicitly notes "no inferential statistic in this
  section depends on the trial count" — a correction that marks what still stands.
- **identity-geometry effective-n correction** ("for any prompt-general claim the effective
  $n$ is 2, not 168"): forced by $\eta^2 = 0.977$ between two prompts. And the same edit
  *adds* a footnote certifying that this experiment's repeats are genuine where others'
  were not — anti-overcorrection discipline in the same diff.
- **logit-bias confab_proj pilot** ("5-prompt pilot in which every case was classified as
  confabulated; no honest baseline arm"): n=5, no control arm; "preliminary" is generous.
- **identity-geometry chance recalibration**: "well above chance at ≈0.002" became "roughly
  $5$--$9\times$ chance at ≈0.008" — the corrected denominator *shrinks* the headroom and the
  edit still states the multiple plainly. Calibrated in both directions.

### Still too strong (the actual residue), ranked by exposure

1. **`oracle-loop-paper/sections/background.tex:32-37` (and the `paper/sections/` copy, and
   both shipped PDFs):** "within-model deception detection reaches AUROC 1.000 across all 7
   tested architectures. ... Crucially, confabulation and deception produce *geometrically
   opposite* signatures: confabulation contracts the spectral representation while deception
   expands it." **No caveat.** The shipped `Oracle_Loop_-_Academic_Version.pdf` renders:
   "These confabulation results extend established deception detection at AUROC 1.000 across
   seven model scales." *Established.* This is SWEEP T1 recommendation #2, applied at
   kv-cloak, user-model, and presence-detector, **not applied at oracle-loop** — the paper
   whose Loop the routing claim belongs to.
2. **`RESEARCH_STATUS.md:81`:** "The deception signal is robust at generation but absent at
   encoding." With d=1.36 withdrawn, "robust" rests on the waystations d=0.2–1.0, n=7, no
   layer significant, rated WEAKENED. Stale survivor of the pre-withdrawal era; should read
   "exploratory at generation."
3. **presence-detector abstract/intro/table:** unqualified "desperate$\to$deception,
   $d{=}1.286$" on the axis §diagnose just declared unsupported (see Item 2).
4. **mine5 "The data falsified this prediction":** the ratio moved 1.97→2.51 opposite to
   prediction, but the same edit establishes the selectivity is n.s. at n=30. A
   non-significant movement in the opposite direction *fails to support* the prediction; it
   does not falsify it. The section heading has the same problem. (This is the
   fix-overshoots-into-its-mirror shape: the positive claim was demoted, the symmetric
   negative claim on the same data was not.)
5. **Fidelity, not calibration:** meta-pattern table says "Confabulation detection ...
   AUROC 0.620 (FWL; 0.707 w/o)"; `RESEARCH_STATUS.md` says "Oracle Loop detection: AUROC
   0.707, p=0.001 (clean, FWL-corrected)." These disagree about which number is the
   FWL-corrected one. One of them is wrong; the Oracle Loop PDF's own text (0.707 =
   generation-window LOO; 0.877 = FWL-corrected spectral gap) suggests both may be
   misdescribing a third thing. Needs one trip to the producing JSON.

---

## Ranked answer to the question asked

**Claims weaker than the evidence supports — two, both modest:**

1. *(moderate, because the deception arm is the program's highest-value claim if real)* The
   two d=1.36 withdrawal sites and the presence-detector deception-arm concession omit that
   independent same-direction exploratory evidence exists: waystations honest-vs-deceptive
   defection, "per-layer generation-phase effects of $d = +0.2$ to $+1.0$ (stable rank)"
   (organic, unsteered, no template confound, n=7, n.s., WEAKENED). One cross-referencing
   sentence at each site. Categories stay as they are.
2. *(minor)* Item 4's guard is described more weakly in the meta-pattern paper than the
   09-06 scan would support *if* the scan's manifest were persisted and its scope stated —
   but as written the docstring overshoots and the paper is right. Persist the scan, then
   the paper may say "no persisted artifact shows the signature" as a positive result.

**Everything else audited — the d=1.36 Withdrawn category, the asymmetric-routing wording,
the 1.37–2.28 substitution, the ledger paragraph, and every sampled hedge from the week's
diff — is correctly calibrated. The cuts were right.** The corpus's live calibration debt is
five under-corrections (§Item 6 ranking), of which the oracle-loop background section — the
withdrawn AUROC 1.000 asserted caveat-free in a shipped PDF — is the one an external reviewer
finds first.

*Agni, 2026-09-07. Verdicts quoted to primaries; the two scans I could not locate
(438-artifact zero scan, mine5 bootstrap CI) are flagged as such rather than trusted.*
