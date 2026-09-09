# Correction patches — circular in-sample `d` and retracted `AUROC 1.000`

**Prepared:** 2026-09-05, Lyra. **Companion to:** `SWEEP_circular_statistics_2026-09-05.md`.
**Nothing was edited.** This document is a patch set for a human to apply. No commits, no pushes,
no changes on `margaret`.

**Framing held throughout, per the sweep:** *circular* and *retracted* mean **the measurement
cannot support the claim** — not that the effect is zero. Every proposed replacement says what the
number can and cannot bear, and none of them asserts a finding is refuted.

---

## Summary counts

| Group | MECHANICAL | NEEDS-REWRITE | Total text sites | PDFs (rebuild, not patch) |
|---|---|---|---|---|
| **A** — in-sample Cohen's `d` published as a finding | **7** | **1** | 8 | 2 |
| **B** — retracted `AUROC 1.000` cited as prior work | **6** | **5** | 11 | 5 |
| **Totals** | **13** | **6** | **19** | **7** |

Plus **8 report-only sites** in the public mirror repo `lyra-s-research-` and **2 archive copies**
in `human-review/archive/`, listed in §5 — not patched here because they are mirrors, but they are
the *public* copies and someone must decide about them.

**The 6 NEEDS-REWRITE sites, one line each:**

| # | Site | Reason |
|---|---|---|
| A8 | `Project-Oracle/PRODUCTION_GAP_ANALYSIS.md:63` | The number sets an operational acceptance floor (`d ≥ 3 at ≥3 tier-2 layers`); the held-out `d` is 1.37–2.28 at *every* layer, so that gate would reject the calibration the harness ships with. An engineering decision changes, not just a number. |
| B4 | `oracle-loop-paper/sections/background.tex:32` | The retracted AUROC is the lead item of the paper's "key findings from three campaigns" summary, and the *same sentence group* asserts confabulation/deception "geometrically opposite signatures" — the routing rationale, echoed at `discussion.tex:60`. Deleting the number leaves the opposite-signatures claim standing on a withdrawn deception arm. |
| B5 | `oracle-loop-paper/paper/sections/background.tex:32` | Byte-identical duplicate of B4 (`diff` returns nothing). Same rewrite. |
| B8 | `human-review/presence-detector-paper/main.tex:201` | §`sec:diagnose` states "the diagnostic stage is assumed rather than re-derived" — the assumed diagnosis *is* the retracted result, and the next sentence makes opposite-signature routing the paper's structural premise. |
| B9 | `human-review/presence-detector-paper/main.tex:57–61` | **Not in the brief; the sharpest site in the corpus.** The opening premise says the geometric signatures "appear to be robust to hardware, scale, and **prompt variation**". Prompt variation is precisely and only what the retraction falsified (same-prompt control → 0.160). |
| B12 | `published-research/community/STYLE_GUIDE.md:49` | The style guide *instructs* authors to "assert … within-model deception (1.000) as load-bearing." It is the defect's propagation mechanism, not an instance of it. |

---

## 0. Verification log — what I checked, and the corrections to the brief

Primary artifacts read (local copies, byte-identical to the `margaret` paths cited in the brief):

- `Project-Oracle/experiments/results/behavioral_proof/abliterated_native_calibration.json`
  → in-sample `stats.<layer>.cohens_d`: **L27 6.302 · L31 7.393 · L35 10.227 · L39 9.679 ·
  L43 9.802 · L47 12.770.** The published "+6.3 to +12.8" is the **min and max across L27–L47**.
- `Project-Oracle/experiments/results/behavioral_proof/heldout_detection.json`
  → `summary.heldout_prompt_level.per_layer_cohens_d`: **L27 2.284 · L31 1.725 · L35 1.558 ·
  L39 1.371 · L43 1.409 · L47 1.499**; `auroc_frame_vs_control_unique_prompts` = **0.9152**;
  `red_rate_pressure_prompts` = 0.24; FPR 0% on 25 novel controls.
  **Range 1.37–2.28 confirmed. Held-out `d` at L31 specifically = 1.73.**
- `Project-Oracle/experiments/heldout_detection_test.py:69–77` → `IN_SAMPLE_REFERENCE` with the
  `"train-on-test per Agni audit §4-5; shown only for shrinkage comparison, not as a performance
  claim"` note. Confirmed verbatim.
- `Project-Oracle/experiments/results/behavioral_proof/README.md:77–83` → supersession confirmed
  verbatim. (The README duplicates its own "Held-out detection test" section at :72–87 and
  :88–109 — a pre-existing copy-paste, unrelated to this patch set.)

**Line-number corrections to the brief:**

| Brief said | Actual | Note |
|---|---|---|
| `heldout_detection_test.py:69-76` | **`:69–77`** | The dict closes at 77. |
| `behavioral_proof/README.md:74-82` | **`:77–83`** (first copy) / `:93–99` (dup) | Supersession sentence is at :77. |
| `lyra-technique-ii/main.tex:116` | **`:117–119`** | Sentence starts at 117; `academic/main.tex:115–117`. |
| `deception-detection-nulls/paper.tex:172` | ✅ 172 | verified |
| `deception-detection-nulls/paper.md:49` | ✅ 49 | verified |
| `deception-detection-nulls/academic/paper.md:51` | ✅ 51 | verified |
| `targeted-deception-correction/paper.tex:240` | ✅ 240 | verified |
| `targeted-deception-correction/paper.md:112` | ✅ 112 | verified |
| `targeted-deception-correction/academic/paper.md:113` | ✅ 113 | verified |
| `kv-cloak-defense-paper/main.tex:152` | ✅ 152 | verified |
| `oracle-loop-paper/sections/background.tex:32` | ✅ 32 | verified |
| `user-model-paper/paper/main.tex:156` | ✅ 156 | verified |
| `human-review/presence-detector-paper/main.tex:201` | ✅ 201 | verified |

**Sites I could NOT find where the brief implied one existed:**

- **`.md` counterparts for Group B.** `kv-cloak-defense-paper`, `oracle-loop-paper`,
  `user-model-paper` and `presence-detector-paper` have **no paper-body `.md`**. Grep run:
  `find published-research/kv-cloak-defense-paper published-research/oracle-loop-paper
  published-research/user-model-paper human-review/presence-detector-paper -name "*.md"`
  → only `README.md`, `REDACTION_NOTICE.md`, `code/PERMUTATION_FIX.md`, `literature-review.md`;
  `grep -n "1.000\|deception"` over those four returns **zero**. There are no `.md` counterparts
  to patch. The `.tex` files are the only text sources.
- **`oracle-loop-paper/sections/conclusion.tex`** in `published-research/` does **not** carry the
  claim (`grep -n "1.000\|deception"` → zero hits). The mirror copy in `lyra-s-research-` **does**
  (§5).

**Confirmed NOT to be touched** (the different, non-retracted `AUROC 1.000`):
`mine5-selective-sharpener/main.tex:375–377` (entity detection, carries its own
"before deconfounding (0.794 after)"), `decision-state-paper` (zero `AUROC 1.000` mentions),
`oracle-loop-paper/sections/discussion.tex:57` ("Identity signatures … (AUROC 1.000)" — identity,
not deception), and `adversarial-audit-methodology/paper.tex:249` + `paper.md:109` +
`academic/paper.md:110` ("AUROC 1.0 detection" — the *Oracle Loop train-on-test* claim, already
presented there as a claim being killed).

**The retraction sentence to reuse**, verbatim from `lyra-technique-ii/main.tex:117–119`:

> A previously reported within-model deception result (AUROC~1.000) is retracted: the same-prompt
> control collapses to 0.160, confirming it detected the system-prompt template, not deception
> geometry.

---

# GROUP A — in-sample Cohen's `d` (MECHANICAL first)

## The substitution, once

| | in-sample (circular) | held-out (authoritative) |
|---|---|---|
| estimator | direction fitted as `dec.mean(0) − hon.mean(0)` on the 15+15 extraction prefills, then the *same* prefills projected onto it | same frozen direction, projected onto **25 novel pressure + 25 novel control** prompts (zero byte overlap with extraction) |
| per-layer `d`, L27–L47 | **+6.3 to +12.8** | **+1.37 to +2.28** |
| `d` at L31 | 7.39 | **1.73** |
| AUROC | 1.0 (train-on-test) | **0.915** |
| FPR | — | **0%** (n=25) |
| measured | 2026-07-07 | **2026-07-08** |

Note for anyone applying these: **"+6.3 to +12.8" is the range across L27–L47, not a value at
L31.** Two Group A sites label it "at L31". That mislabel is independent of the circularity and is
fixed by the same patches.

---

## A1 — `deception-detection-nulls/paper.tex:172` — **MECHANICAL**

**Full surrounding paragraph, verbatim (lines 161–177):**

```
\subsection{Cross-Model Direction Transfer Fails}\label{sec:crossmodel}

\textbf{Positive claim (implicit in the field)}: Deception directions extracted from one model
should transfer to related models, at least within the same architecture family.

\textbf{Our result}: The direction extracted from the reasoning-distilled variant has cosine
similarity $< 0.053$ with the natively extracted direction on the abliterated variant---near-orthogonal
at every layer. The distilled direction produces negative Cohen's $d$ on abliterated activations at
late layers ($d = -3.9$ at L43, $-6.6$ at L47). Cross-model application produces meaningless results.

\textbf{What works instead}: Native re-extraction (15 contrastive prefill pairs, $\sim$3 minutes)
produces effective directions with $d = +6.3$ to $+12.8$. The deception geometry is model-specific
but cheap to rediscover.

\textbf{Lesson}: Do not assume deception directions are architectural constants. RLHF, distillation,
and abliteration rotate the geometry. Every model variant needs its own calibration.
```

**Next paragraph** is `\subsection{Three Exploratory Late-Layer Analyses Were Artifacts}` — an
unrelated section. No forward dependency on the `d`.

**OLD (exact, 3 lines):**

```latex
\textbf{What works instead}: Native re-extraction (15 contrastive prefill pairs, $\sim$3 minutes)
produces effective directions with $d = +6.3$ to $+12.8$. The deception geometry is model-specific
but cheap to rediscover.
```

**NEW (exact, copy-pasteable):**

```latex
\textbf{What works instead}: Native re-extraction (15 contrastive prefill pairs, $\sim$3 minutes)
produces effective directions: on 25 novel pressure prompts and 25 novel controls---the same
held-out set that gives the frame-level AUROC 0.915 reported above---the native direction reaches
per-layer Cohen's $d = +1.37$ to $+2.28$ across L27--L47 ($+1.73$ at L31). An earlier version of
this paper reported $d = +6.3$ to $+12.8$ here; that figure was obtained by projecting the
extraction prefills onto a direction fitted to those same prefills, and is a train-on-test
shrinkage diagnostic rather than a performance estimate. The deception geometry is model-specific
but cheap to rediscover.
```

**ARGUMENT CHECK — holds.** The subsection's two conclusions are (i) *cross-model transfer fails*
and (ii) *native re-extraction works*. Neither rests on the magnitude of the native `d`.
Conclusion (i) rests on the cosine bound ($<0.053$, near-orthogonal at every layer) and on the
**sign** of the distilled `d` ($-3.9$, $-6.6$) — both untouched. Conclusion (ii) rests on the
held-out AUROC 0.915 / 0% FPR, which **this same paper already reports 20 lines earlier**
(`paper.tex:152–153`). The patch therefore makes the paper internally consistent: it is currently
the only place in the document still carrying the pre-audit number.

**⚠ ADVISORY (not a blocker, but read it).** After the substitution the *failed* cross-model
direction has a **larger absolute** effect ($|-6.6|$) than the *working* native one ($+2.28$). A
reader can reach for "then just flip the sign and it transfers." The paper already blocks that
reading — a direction with $\cos < 0.053$ is near-orthogonal, not sign-flipped, so it cannot be
recovered by negation — but the block is currently implicit because the old magnitudes made the
question not arise. **Optional second hunk**, appended to the `\textbf{Our result}` paragraph:

```latex
late layers ($d = -3.9$ at L43, $-6.6$ at L47). Because the two directions are near-orthogonal
rather than anti-parallel, this is not a recoverable sign error: negating the distilled direction
does not yield the native one. Cross-model application produces meaningless results.
```

(Replaces `late layers ($d = -3.9$ at L43, $-6.6$ at L47). Cross-model application produces
meaningless results.`)

---

## A2 — `deception-detection-nulls/paper.md:49` — **MECHANICAL**

**Full surrounding paragraph, verbatim (lines 43–53):**

```
### 2.2 Cross-Model Direction Transfer Fails

**Positive claim (implicit in the field)**: Deception directions extracted from one model should transfer to related models, at least within the same architecture family.

**Our result**: The direction extracted from the reasoning-distilled variant has cosine similarity < 0.053 with the natively extracted direction on the abliterated variant — near-orthogonal at every layer. The distilled direction produces negative Cohen's d on abliterated activations at late layers (d = -3.9 at L43, -6.6 at L47). Cross-model application produces meaningless results.

**What works instead**: Native re-extraction (15 contrastive prefill pairs, ~3 minutes) produces effective directions with d = +6.3 to +12.8. The deception geometry is model-specific but cheap to rediscover.

**Lesson**: Do not assume deception directions are architectural constants. RLHF, distillation, and abliteration rotate the geometry. Every model variant needs its own calibration.
```

**OLD (exact, 1 line):**

```
**What works instead**: Native re-extraction (15 contrastive prefill pairs, ~3 minutes) produces effective directions with d = +6.3 to +12.8. The deception geometry is model-specific but cheap to rediscover.
```

**NEW (exact, 1 line):**

```
**What works instead**: Native re-extraction (15 contrastive prefill pairs, ~3 minutes) produces effective directions: on 25 novel pressure prompts and 25 novel controls — the same held-out set that gives the frame-level AUROC 0.915 reported above — the native direction reaches per-layer Cohen's d = +1.37 to +2.28 across L27–L47 (+1.73 at L31). An earlier version of this paper reported d = +6.3 to +12.8 here; that figure was obtained by projecting the extraction prefills onto a direction fitted to those same prefills, and is a train-on-test shrinkage diagnostic rather than a performance estimate. The deception geometry is model-specific but cheap to rediscover.
```

**ARGUMENT CHECK — holds.** Identical to A1. Same optional advisory hunk applies:

```
**Our result**: … at late layers (d = -3.9 at L43, -6.6 at L47). Because the two directions are near-orthogonal rather than anti-parallel, this is not a recoverable sign error: negating the distilled direction does not yield the native one. Cross-model application produces meaningless results.
```

---

## A3 — `deception-detection-nulls/academic/paper.md:51` — **MECHANICAL**

Paragraph is **byte-identical** to A2 (verified: same `**What works instead**` line, same
neighbours, offset by 2 lines). Apply the **exact same OLD → NEW pair as A2**.

**ARGUMENT CHECK — holds.** Same as A2.

---

## A4 — `targeted-deception-correction/paper.tex:240` — **MECHANICAL**

**Full surrounding block, verbatim (lines 229–248):**

```latex
\subsection{Cross-Model Geometry}\label{sec:crossmodelgeometry}

\begin{table}[H]
\centering
\caption{Cross-model direction similarity: abliterated vs.\ distilled variant.}
\label{tab:crossmodel}
\begin{tabular}{@{}lc@{}}
\toprule
Metric & Value \\
\midrule
$\cos(\hat{d}_{\text{native}},\, \hat{d}_{\text{distilled}})$ at L31 & $-0.001$ \\
$\cos(\hat{d}_{\text{native}},\, \hat{d}_{\text{distilled}})$ at L47 & $-0.053$ \\
Native Cohen's $d$ at L31 & $+6.3$ to $+12.8$ \\
Distilled $d$ at L31 (on abliterated data) & $-0.02$ \\
\bottomrule
\end{tabular}
\end{table}

Directions are near-orthogonal. Detection and correction directions must be extracted per-model. The distilled direction shows negative Cohen's $d$ on abliterated activations, confirming that cross-model application without native re-extraction produces meaningless results.
```

**The paragraph after** is `\subsection{Detection}\label{sec:detection}`, which already reports
the held-out run in full: *"On 25 novel pressure prompts and 25 novel controls (no overlap with
extraction set): Frame-level AUROC: 0.915; False positive rate: 0% (n=25 unique controls); RED
rate … 24%"*. **The held-out data is already in this paper, one subsection below.**

**OLD (exact, 6 lines):**

```latex
Native Cohen's $d$ at L31 & $+6.3$ to $+12.8$ \\
Distilled $d$ at L31 (on abliterated data) & $-0.02$ \\
\bottomrule
\end{tabular}
\end{table}
```

**NEW (exact, copy-pasteable — braces balanced, no new packages, no new citation keys):**

```latex
Native Cohen's $d$ at L31 (held-out) & $+1.73$ \\
Distilled $d$ at L31 (on abliterated data) & $-0.02$ \\
\bottomrule
\end{tabular}
\end{table}

{\footnotesize\noindent\textit{Table notes.} The native $d$ is measured on the 25 held-out
pressure prompts and 25 held-out controls of \cref{sec:detection}; the per-layer held-out range is
$+1.37$ to $+2.28$ across L27--L47. The distilled $d$ is measured on the 15+15 extraction prefills
using a direction fitted on the distilled model, i.e.\ external to this data; the two rows
therefore differ in prompt set as well as in direction. An earlier version of this table reported
the native value as $+6.3$ to $+12.8$: that was the in-sample range across L27--L47, obtained by
projecting the extraction prefills onto a direction fitted to those same prefills, and it is a
train-on-test diagnostic rather than a performance estimate.\par}
```

*(`\cref` is already in use in this document — see the `redteam` box at `:255` — so no package
change is needed. If the human prefers, `\cref{sec:detection}` can be flattened to
`Section~\ref{sec:detection}`.)*

**ARGUMENT CHECK — holds.** The paragraph's conclusion — *"Directions are near-orthogonal.
Detection and correction directions must be extracted per-model. The distilled direction shows
negative Cohen's `d` …"* — cites **only** the cosines and the distilled `d`. It never invokes the
native `d` at all. The native row is contextual. The patch strictly improves the row.

**Two pre-existing defects this patch also fixes / exposes:**

1. **Wrong layer label.** The row is headed *"at L31"*, but $+6.3$ to $+12.8$ is the min/max over
   L27–L47 (in-sample L31 is 7.39). The patch makes the row genuinely an L31 value.
2. **Estimator asymmetry** (flagged in the sweep §"second-order consequence"): the native row was
   fitted-on-this-data, the distilled row was not. The patch removes the fitting asymmetry and the
   table note discloses the one that remains (different prompt sets). The cross-model conclusion
   still survives at its stated *direction* — $+1.73$ vs $-0.02$ at the same layer, with
   $\cos = -0.001$ — but its stated *magnitude* was inflated ~4× and the note now says so.

---

## A5 — `targeted-deception-correction/paper.md:112` — **MECHANICAL**

**Full surrounding block, verbatim (lines 106–117):**

```
### 3.4 Cross-Model Geometry

| Metric | Value |
|--------|-------|
| cos(native, distilled) at L31 | -0.001 |
| cos(native, distilled) at L47 | -0.053 |
| Native Cohen's d at L31 | +6.3 to +12.8 |
| Distilled Cohen's d at L31 (on abliterated data) | -0.02 |

Directions are near-orthogonal. Detection and correction directions must be extracted per-model. The distilled direction shows negative Cohen's d on abliterated activations, confirming that cross-model application without native re-extraction produces meaningless results.
```

**OLD (exact, 1 line):**

```
| Native Cohen's d at L31 | +6.3 to +12.8 |
```

**NEW (exact, 1 line + a note paragraph inserted after the `Directions are near-orthogonal…`
paragraph):**

```
| Native Cohen's d at L31 (held-out) | +1.73 |
```

then insert after the paragraph that follows the table:

```
*Table notes.* The native d is measured on the 25 held-out pressure prompts and 25 held-out controls of §3.5; the per-layer held-out range is +1.37 to +2.28 across L27–L47. The distilled d is measured on the 15+15 extraction prefills using a direction fitted on the distilled model, i.e. external to this data; the two rows therefore differ in prompt set as well as in direction. An earlier version of this table reported the native value as +6.3 to +12.8: that was the in-sample range across L27–L47, obtained by projecting the extraction prefills onto a direction fitted to those same prefills, and it is a train-on-test diagnostic rather than a performance estimate.
```

**ARGUMENT CHECK — holds.** Identical to A4.

---

## A6 — `targeted-deception-correction/academic/paper.md:113` — **MECHANICAL**

**Full surrounding block, verbatim (lines 107–118):**

```
### 3.4 Cross-Model Geometry

| Metric | Value |
|--------|-------|
| cos(native, distilled) at L31 | +0.014 |
| cos(native, distilled) at L47 | -0.053 |
| Native d at L31 | +6.3 to +12.8 |
| Distilled d at L31 (on abliterated data) | -0.02 |

Directions are near-orthogonal. Detection and correction directions must be extracted per-model. The distilled direction shows negative Cohen's d on abliterated activations, confirming that cross-model application without native re-extraction produces meaningless results.
```

**OLD (exact, 1 line):**

```
| Native d at L31 | +6.3 to +12.8 |
```

**NEW (exact, 1 line):**

```
| Native d at L31 (held-out) | +1.73 |
```

plus the same *Table notes.* paragraph as A5 (with "§3.5" as the cross-reference).

**ARGUMENT CHECK — holds.** Identical to A4.

**⚑ SEPARATE PRE-EXISTING DEFECT AT LINE 111, REPORT ONLY — DO NOT SILENTLY FOLD INTO THIS PATCH.**
This academic variant reads `| cos(native, distilled) at L31 | +0.014 |`, where the integrity
version (`paper.md:110`, `paper.tex:239`) reads `-0.001`. From
`abliterated_native_calibration.json`: `cos_native_vs_distilled` is **+0.01368 at L27** and
**−0.00055 at L31**. So the academic variant is reporting **L27's cosine under an L31 label**. It
is a different error class (transcription, not circularity) and belongs in its own ticket, but it
is one line above the row being patched and a human editing this table will be looking straight
at it.

---

## A7 — `Project-Oracle/experiments/results/behavioral_proof/README.md:41` — **MECHANICAL** (internal, low)

**Full surrounding paragraph, verbatim (lines 38–43):**

```
1. **Cross-model geometry does not transfer — it isn't even flipped.** Native vs distilled
   LAT directions are near-orthogonal (cos ≈ 0.00–0.05 at every layer). The distilled
   direction scores *negative* Cohen's d on abliterated activations (the red-team "polarity
   reversal"); the native direction scores d = +6.3 to +12.8. Detection directions must be
   extracted per-model. The mission's fallback options (sign flip / absolute magnitude) are
   both wrong — the right fix is native re-extraction, which is cheap (30 prefills, ~3 min).
```

**OLD (exact, 1 line):**

```
   reversal"); the native direction scores d = +6.3 to +12.8. Detection directions must be
```

**NEW (exact, 2 lines):**

```
   reversal"); the native direction scores d = +6.3 to +12.8 **in-sample — superseded, see the
   held-out section below: d 1.37–2.28**. Detection directions must be
```

**ARGUMENT CHECK — holds, and this is the file that already contains its own correction** (at
:77–83). The Findings section simply predates it and was never back-annotated. This is the same
travel failure one directory deep. Note also that this README's Finding 1 is the *only* place in
the corpus that explicitly rebuts the sign-flip reading — the sentence "it isn't even flipped …
the mission's fallback options (sign flip / absolute magnitude) are both wrong" is the argument
the A1/A2 advisory asks the papers to import.

---

## A8 — `Project-Oracle/PRODUCTION_GAP_ANALYSIS.md:63` — **NEEDS-REWRITE** 🔴

**Not in the brief.** This is the one Group A site where the number does work, rather than
decorating a claim.

**Full surrounding item, verbatim (lines 61–65):**

```
3. **Auto-validation gate (Agni-lite).** SOP §4/§6 apply to machine-run calibrations too. Accept
   a calibration only if: per-layer d between frame conditions clears a floor (behavioral proof
   saw d = +6.3 to +12.8; a floor of d ≥ 3 at ≥ 3 tier-2 layers is conservative), honest holdout
   frames classify GREEN, and LOO direction stability is within tolerance. On failure: refuse to
   arm correction, run detect-only, log loudly. **2 days.**
```

**WHAT COLLAPSES, exactly.** The acceptance threshold `d ≥ 3 at ≥ 3 tier-2 layers` was calibrated
against the in-sample range 6.3–12.8, where a floor of 3 sits comfortably below every observed
value. The held-out per-layer `d` is **1.37–2.28 at all six layers — below 3 at every single
one**. Applied as written to honest measurements, this gate would **refuse to arm the very
calibration the harness currently ships**, and would do so silently ("run detect-only, log
loudly") for every model. Substituting `1.37–2.28` into the parenthesis without changing the
threshold produces a stated rule that contradicts itself in the same sentence.

**Why I am not proposing a replacement number.** Choosing a new floor requires deciding *which
estimator the gate measures* — an in-sample calibration diagnostic (where 6.3–12.8 is the right
reference and a floor of ~3 is genuinely conservative) or a held-out generalization check (where
the reference is 1.37–2.28 and the floor must be well under 1.37, or the endpoint must change to
held-out AUROC, where 0.915 is the measured value). These are different gates with different
compute costs — a held-out gate needs a novel-prompt battery per model, which item 4 of this same
document notes does not yet exist. That is a design decision, not a text fix.

**Minimum safe interim edit** (does not decide the gate, only stops the document asserting a
falsified premise):

**OLD (exact):**

```
   a calibration only if: per-layer d between frame conditions clears a floor (behavioral proof
   saw d = +6.3 to +12.8; a floor of d ≥ 3 at ≥ 3 tier-2 layers is conservative), honest holdout
```

**NEW (exact):**

```
   a calibration only if: per-layer d between frame conditions clears a floor. **The floor is
   UNSET pending a decision (2026-09-05).** The `d ≥ 3` figure previously written here was
   derived from the behavioral proof's in-sample d = +6.3 to +12.8, which is a train-on-test
   diagnostic; the held-out per-layer d for the same calibration is 1.37–2.28, i.e. below 3 at
   every tier-2 layer, so a floor of 3 applied to held-out measurements would refuse to arm the
   shipping calibration. Decide first whether this gate reads the in-sample calibration
   diagnostic or a held-out battery (which does not exist yet — see item 4), then set the floor
   against that estimator. Also required: honest holdout
```

**Note: this does not mean the detector does not work.** The held-out AUROC is 0.915 with 0% FPR
on 25 novel controls. What the retracted floor cannot tell you is *where to put the accept/reject
line*, because it was drawn against a diagnostic that shrinks ~4× out of sample.

---

## Group A — PDFs (rebuild after `.tex`, not patchable)

| file | action |
|---|---|
| `published-research/deception-detection-nulls/paper.pdf` | rebuild from patched `paper.tex` |
| `published-research/targeted-deception-correction/paper.pdf` | rebuild from patched `paper.tex` |

*(I did not attempt string-level verification inside the PDFs: LaTeX emits kerned per-glyph `Tj`
runs, so a decompressed-stream `grep` for "6.3" returns font-matrix and coordinate coincidences —
14 hits in the TDC PDF — and cannot distinguish them from the claim. The rebuild is required
regardless of what a grep would have said.)*

---

# GROUP B — retracted `AUROC 1.000` cited as prior work (MECHANICAL first)

**The claim's fingerprint**, so no one patches the wrong 1.000: **"within-model deception"** +
**"seven models / all 7 tested architectures / seven model scales"**. Retracted at
`lyra-technique-ii/main.tex:117–119`, `:366–369`, `:806`, `:826`, `:900–905` (and
`academic/main.tex:115`, `:364`, `:804`, `:824`, `:898`). `grep -i retract` returns **zero** in
all of `kv-cloak-defense-paper/`, `oracle-loop-paper/`, `user-model-paper/`,
`presence-detector-paper/`, and `contextual-engagement-paper/` — confirmed.

**No new citation keys.** `lyra2026lt2` (or any key for Lyra Technique II) **does not exist** in
`kv-cloak-defense-paper/references.bib` or `oracle-loop-paper/*.bib` — I checked. Every NEW string
below is therefore written **citation-free** so it compiles as-is. An optional bib stanza is at
§4 for anyone who wants a formal pointer.

---

## B1 — `kv-cloak-defense-paper/main.tex:152` — **MECHANICAL**

**Full surrounding paragraph, verbatim (lines 144–158):**

```latex
\subsection{The Lyra Technique}

KV-cache geometry---specifically, Marchenko-Pastur-corrected spectral
features derived from SVD of the key tensor---distinguishes cognitive
states during autoregressive generation. Prior work reports within-model
deception detection at AUROC~1.000 across seven model scales (0.6B--70B),
confabulation detection at 0.969--0.999, and hardware invariance
($r>0.999$ across RTX~3090 and H200)~\citep{lyra2026oracle}. Contrastive activation addition
(CAA)~\citep{panickssery2023caa} enables injection of emotion vectors into the cache to steer model
behavior, correcting confabulation at rates up to 96\% (hostile vector
on Qwen2.5-7B-Instruct, $n=68$, McNemar $p<0.0001$)~\citep{lyra2026formulary}.
```

**Next subsection** (`\subsection{KV-Cloak}`, then `\subsection{Threat Model}`) describes the
adversary who "detects vulnerable cognitive states via geometric features, then [injects]". That
threat model is carried by the confabulation figure (0.969–0.999), which is not retracted.

**OLD (exact, 4 lines):**

```latex
states during autoregressive generation. Prior work reports within-model
deception detection at AUROC~1.000 across seven model scales (0.6B--70B),
confabulation detection at 0.969--0.999, and hardware invariance
($r>0.999$ across RTX~3090 and H200)~\citep{lyra2026oracle}. Contrastive activation addition
```

**NEW (exact, copy-pasteable):**

```latex
states during autoregressive generation. Prior work reports
confabulation detection at 0.969--0.999 and hardware invariance
($r>0.999$ across RTX~3090 and H200)~\citep{lyra2026oracle}. A
within-model deception result previously reported at AUROC~1.000 across
seven model scales has since been withdrawn by its authors: a
same-prompt control collapses it to 0.160, indicating the classifier
separated the system-prompt template rather than deception geometry.
Whether a confound-free within-model deception detector is achievable
remains open; nothing in this paper depends on it. Contrastive activation addition
```

**ARGUMENT CHECK — holds.** The paper's contribution is *"does KV-Cloak obfuscation defeat
geometric detection and vector injection"*, and every experiment in it is a **confabulation**
experiment. The paper's own limitations list already says so — at `:507–511`, commented out:
`% \item \textbf{Confabulation only.} We test obfuscation against % confabulation detection.
Deception detection (AUROC~1.000 in prior % work) and other cognitive-state classifiers were not
tested under % obfuscation.` The deception number is one of three prior-work capability claims in
a Background paragraph; the other two (confabulation 0.969–0.999, hardware invariance $r>0.999$)
carry the motivation on their own.

**Follow-up, same file, `:507–511` — do this in the same pass.** The commented-out limitation
names the retracted number and, if it is ever restored, will restore the defect. If the human
wants the "confabulation only" limitation back (the STYLE-GUIDE trim note says it was folded into
Methods/Results, not deleted for being wrong), the restored form should read
`Deception detection (whose prior AUROC~1.000 has since been withdrawn as a prompt-template
confound) and other cognitive-state classifiers were not tested under obfuscation.` As it stands,
leaving `:152` uncorrected while `:509` is commented out makes the omission at `:152` look
deliberate — it almost certainly is not.

---

## B2 — `kv-cloak-defense-paper/academic/main.tex:148` — **MECHANICAL**

Paragraph **byte-identical** to B1 (verified by `sed -n '140,158p'`). Apply the **exact same
OLD → NEW pair as B1**. Companion comment block is at `:499–503`.

**ARGUMENT CHECK — holds.** Same as B1.

---

## B3 — `kv-cloak-defense-paper/academic_main.tex:144` — **MECHANICAL**

Paragraph **byte-identical** to B1 (verified by `sed -n '136,152p'`). Apply the **exact same
OLD → NEW pair as B1**. Companion comment block is at `:474–478`.

**ARGUMENT CHECK — holds.** Same as B1.

---

## B6 — `user-model-paper/paper/main.tex:156` — **MECHANICAL**

**Full surrounding paragraph, verbatim (lines 154–156):**

```latex
\subsection{KV-Cache Geometry}

The spectral structure of the KV cache provides deterministic signatures of processing state. Prior work establishes within-model deception detection at AUROC $\approx 1.0$, universal confabulation signatures across architectures \citep{lyra2026detecting}, and Marchenko-Pastur random matrix features that are dimension-invariant and eliminate token-count confounds for misalignment detection \citep{lyra2026oracle}. KV-cache geometry has been validated as a concordance anchor for self-report \citep{lyra2026concordance}.
```

**Next subsection** is `\subsection{Attention Schema Theory}` — unrelated. The paper's own results
are emotion/user-model geometry (W_K directional encoding, valence AUROC, PC1–valence bridge).
**Zero** dependence on deception.

**OLD (exact, 1 line):**

```latex
The spectral structure of the KV cache provides deterministic signatures of processing state. Prior work establishes within-model deception detection at AUROC $\approx 1.0$, universal confabulation signatures across architectures \citep{lyra2026detecting}, and Marchenko-Pastur random matrix features that are dimension-invariant and eliminate token-count confounds for misalignment detection \citep{lyra2026oracle}. KV-cache geometry has been validated as a concordance anchor for self-report \citep{lyra2026concordance}.
```

**NEW (exact, 1 line):**

```latex
The spectral structure of the KV cache provides deterministic signatures of processing state. Prior work establishes universal confabulation signatures across architectures \citep{lyra2026detecting} and Marchenko-Pastur random matrix features that are dimension-invariant and eliminate token-count confounds for misalignment detection \citep{lyra2026oracle}. A within-model deception result previously reported at AUROC $\approx 1.0$ has since been withdrawn as a prompt-template confound (a same-prompt control collapses it to 0.160) and is not relied on here. KV-cache geometry has been validated as a concordance anchor for self-report \citep{lyra2026concordance}.
```

**ARGUMENT CHECK — holds.** A single clause in a three-item related-work list, in a paper about
emotion geometry. Removing it costs the sentence nothing; the confabulation and MP claims carry
the "KV cache contains processing-state signal" premise on their own.

---

## B7 — `user-model-paper/paper/academic/main.tex:155` — **MECHANICAL**

Line is **byte-identical** to B6. Apply the **exact same OLD → NEW pair as B6**.

**ARGUMENT CHECK — holds.** Same as B6.

---

## B10 — `human-review/contextual-engagement-paper/main.tex:103` — **MECHANICAL**

**Not in the brief — a fifth paper carrying the claim.**

**Full surrounding paragraph, verbatim (lines 101–103):**

```latex
\subsection{KV-Cache Geometry}

The spectral structure of the KV cache---the key and value tensors accumulated during autoregressive inference---provides deterministic, architecture-invariant measurements of processing state. Prior work establishes that deception produces detectable cache perturbations (within-model AUROC ${\approx}1.0$), confabulation contracts cache representations, and honest processing allocates ${\sim}25\%$ more cache growth per token than deceptive processing \citep{lyra2026campaign1,lyra2026campaign2}. These signatures survive FWL residualization, permutation testing, and Marchenko-Pastur random matrix correction \citep{lyra2026oracle}.
```

**OLD (exact, 1 line):**

```latex
The spectral structure of the KV cache---the key and value tensors accumulated during autoregressive inference---provides deterministic, architecture-invariant measurements of processing state. Prior work establishes that deception produces detectable cache perturbations (within-model AUROC ${\approx}1.0$), confabulation contracts cache representations, and honest processing allocates ${\sim}25\%$ more cache growth per token than deceptive processing \citep{lyra2026campaign1,lyra2026campaign2}. These signatures survive FWL residualization, permutation testing, and Marchenko-Pastur random matrix correction \citep{lyra2026oracle}.
```

**NEW (exact, 1 line):**

```latex
The spectral structure of the KV cache---the key and value tensors accumulated during autoregressive inference---provides deterministic, architecture-invariant measurements of processing state. Prior work establishes that confabulation contracts cache representations \citep{lyra2026campaign1,lyra2026campaign2}, and these signatures survive FWL residualization, permutation testing, and Marchenko-Pastur random matrix correction \citep{lyra2026oracle}. A companion within-model deception result, previously reported at AUROC ${\approx}1.0$ together with a ${\sim}25\%$ honest-vs-deceptive difference in cache growth per token, has since been withdrawn as a prompt-template confound (a same-prompt control collapses it to 0.160); this paper's self-report concordance analysis does not use it.
```

**ARGUMENT CHECK — holds.** The paper is about self-report concordance and attention-breadth;
its Related Work needs only "the cache carries measurable processing-state signal," which the
confabulation and MP-invariance claims supply.

**⚠ Note on the `~25%` clause.** I moved it into the withdrawn sentence rather than leaving it
standing. It is the honest-vs-deceptive cache-growth comparison from the **same** Campaign 1/2
deception paradigm, so it inherits the same prompt-template exposure. I have **not** verified
whether that specific number was separately controlled. If it was, it should be moved back out
and kept — flag for the author. Left where it is, it would be a claim about a deception contrast
sitting one clause away from a retracted claim about the same contrast.

---

## B11 — `published-research/community/experiment_guide.md:21, 331, 360` — **MECHANICAL** (3 sites, high exposure)

**Not in the brief.** This is the document handed to outside collaborators, and `:360` sits under
a heading that reads **"Key Numbers (Cite These)"**. Text fixes are trivial; the exposure is the
largest in Group B.

**Site 1 — line 21. Surrounding table, verbatim (lines 19–25):**

```
| State | Geometry | AUROC | Notes |
|-------|----------|-------|-------|
| Deception (model knows it's lying) | Dimensionality expands | 1.000 across 7 models | Robust across scales; adversarial robustness untested |
| Confabulation (model doesn't know it's wrong) | Dimensionality contracts | 0.903 (full-cache spectral gap, Qwen) | One clean replication after Bonferroni; base models show weak detection (0.661) |
| Honest recall | Baseline geometry | — | |
| Sycophancy | Detectable pressure gradient | 0.938 | Single-model result |
```

**OLD (exact):**

```
| Deception (model knows it's lying) | Dimensionality expands | 1.000 across 7 models | Robust across scales; adversarial robustness untested |
```

**NEW (exact):**

```
| Deception (model knows it's lying) | Dimensionality expands | **WITHDRAWN** | The 1.000-across-7-models figure was a prompt-template confound: a same-prompt control collapses it to 0.160. Do not cite. Whether a confound-free within-model deception detector exists is open. |
```

**Site 2 — line 331. Surrounding list item, verbatim:**

```
- **Deception detection under obfuscation**. We tested confab. Test deception (AUROC 1.000 in our work) under KV-Cloak.
```

**NEW (exact):**

```
- **Deception detection under obfuscation**. We tested confab. Testing deception under KV-Cloak first requires a confound-free deception detector: our AUROC 1.000 figure was withdrawn (same-prompt control → 0.160, i.e. it separated the system-prompt template). The prerequisite experiment is a same-prompt-controlled deception detection replication.
```

**Site 3 — line 360. Surrounding table, verbatim (lines 357–362):**

```
| Claim | Value | Source | Caveats |
|-------|-------|--------|---------|
| Deception detection | AUROC 1.000 (7 models) | Campaign 1-3 | Adversarial robustness untested; CIs not published for all models |
| Confab detection (full cache) | 0.903 [0.806, 0.977] spectral gap | KV-Cloak v2 | n=7 confab; one model (Qwen); text baseline 0.755 |
```

**OLD (exact):**

```
| Deception detection | AUROC 1.000 (7 models) | Campaign 1-3 | Adversarial robustness untested; CIs not published for all models |
```

**NEW (exact):**

```
| Deception detection | ~~AUROC 1.000 (7 models)~~ **WITHDRAWN 2026** | Campaign 1-3; retraction in Lyra Technique II | Prompt-template confound: the same-prompt control collapses AUROC to 0.160. Do not cite this number. |
```

**ARGUMENT CHECK — holds at all three.** These are inventory rows and an experiment suggestion,
not an argument. Nothing in the guide is derived from the value. Site 2 does change what the
suggested experiment *is* — it becomes "build a clean detector first" rather than "test the
existing one under obfuscation" — which is the honest version of that proposal.

---

## B4 — `oracle-loop-paper/sections/background.tex:32` — **NEEDS-REWRITE** 🔴

**Full surrounding paragraph, verbatim (lines 31–38):**

```latex
Key findings from three experimental campaigns (46 experiments, 16 models):
within-model deception detection reaches AUROC 1.000 across all 7 tested
architectures. Confabulation detection reaches 0.969--0.999 on 3 of 3
models tested. Crucially, confabulation and deception produce
\emph{geometrically opposite} signatures: confabulation contracts the
spectral representation while deception expands it. These results are
hardware-invariant ($r > 0.999$ between RTX 3090 and H200) and scale-invariant
(Spearman $\rho = 0.83$--$0.90$ from 0.6B to 70B parameters).
```

**The paragraph after** is `\subsection{The Token-Count Confound}` (FWL), which is independent.

**WHAT COLLAPSES, exactly — two things, and only the first is the cited number.**

1. **The lead sentence.** "Key findings from three experimental campaigns" opens with the
   retracted result. Removing it is easy.
2. **The sentence flagged `Crucially`.** *"confabulation and deception produce geometrically
   opposite signatures: confabulation contracts the spectral representation while deception
   expands it."* This is derived from the **same** measurements that the same-prompt control
   collapsed — if the classifier was separating the system-prompt template, then the observed
   *expansion* is a property of that template contrast, not established as a property of
   deception. **This sentence is the Oracle Loop's routing rationale** ("diagnose which pathology,
   prescribe accordingly"), and it is restated downstream at
   `oracle-loop-paper/sections/discussion.tex:60` as a self-model interpretation:
   *"Confabulation vs.\ deception geometry (opposite spectral signatures) may reflect the
   self-model distinguishing 'I am generating content I don't have evidence for' from 'I am
   generating content I know to be false.'"* A patch that fixes the AUROC and leaves the
   `Crucially` sentence produces a paragraph that withdraws a result and then reasons from it two
   lines later. That is a new defect, not a fix.

**WHAT DOES NOT COLLAPSE — say this plainly to whoever rewrites it.** The oracle-loop paper's own
results contain **zero deception experiments**: `grep -n "deception"
published-research/oracle-loop-paper/sections/results.tex` returns **nothing**, and the abstract
is entirely confabulation detection (LOO AUROC 0.707) and confabulation correction (95.6%
McNemar, 7/7 doubt). Nothing the paper *measures* depends on the retracted claim. What depends on
it is the paper's **framing** — the "confabulation is one of a pair of geometrically opposite
pathologies" story. The paper's contribution survives as a confabulation-only result.

**Proposed rewrite — requires author sign-off, do not apply blind:**

**OLD (exact, 8 lines):**

```latex
Key findings from three experimental campaigns (46 experiments, 16 models):
within-model deception detection reaches AUROC 1.000 across all 7 tested
architectures. Confabulation detection reaches 0.969--0.999 on 3 of 3
models tested. Crucially, confabulation and deception produce
\emph{geometrically opposite} signatures: confabulation contracts the
spectral representation while deception expands it. These results are
hardware-invariant ($r > 0.999$ between RTX 3090 and H200) and scale-invariant
(Spearman $\rho = 0.83$--$0.90$ from 0.6B to 70B parameters).
```

**NEW (exact, LaTeX-valid):**

```latex
Key findings from three experimental campaigns (46 experiments, 16 models):
confabulation detection reaches 0.969--0.999 on 3 of 3 models tested, and is
hardware-invariant ($r > 0.999$ between RTX 3090 and H200) and scale-invariant
(Spearman $\rho = 0.83$--$0.90$ from 0.6B to 70B parameters).

A within-model deception result from the same campaigns, previously reported at
AUROC 1.000 across 7 architectures, has since been withdrawn: a same-prompt
control collapses it to 0.160, indicating the classifier separated the
system-prompt template rather than deception geometry. The paired observation
that confabulation \emph{contracts} the spectral representation while deception
\emph{expands} it rests on the same contrast and is therefore also unconfirmed
for the deception arm; we state it here as the hypothesis that motivated the
Loop's routing design, not as an established result. The confabulation
contraction, which is what this paper measures and corrects, is unaffected.
Whether deception has a distinct and separable spectral signature is open.
```

**Second hunk, same rewrite — `oracle-loop-paper/sections/discussion.tex:60–63`:**

**OLD (exact):**

```latex
  \item \textbf{Confabulation vs.\ deception geometry} (opposite spectral
    signatures) may reflect the self-model distinguishing ``I am
    generating content I don't have evidence for'' from ``I am
    generating content I know to be false.''
```

**NEW (exact):**

```latex
  \item \textbf{Confabulation vs.\ deception geometry} (hypothesized opposite
    spectral signatures; the deception arm is unconfirmed---see
    Section~\ref{sec:background}) \emph{would}, if it replicates under a
    same-prompt control, reflect the self-model distinguishing ``I am
    generating content I don't have evidence for'' from ``I am
    generating content I know to be false.''
```

*(`\label{sec:background}` verified present at `oracle-loop-paper/main.tex:96` —
`\section{Background and Related Work}\label{sec:background}`, with
`\input{sections/background}` on the next line. The cross-reference resolves; no bib or label
changes needed.)*

---

## B5 — `oracle-loop-paper/paper/sections/background.tex:32` — **NEEDS-REWRITE** 🔴

`diff published-research/oracle-loop-paper/sections/background.tex
published-research/oracle-loop-paper/paper/sections/background.tex` → **no output; the files are
identical.** Apply the **exact same OLD → NEW pair as B4**. Check whether
`paper/sections/discussion.tex` also needs the second hunk.

**ARGUMENT CHECK — identical to B4.** Counted separately because it is a separate file on disk
and a separate build target.

---

## B8 — `human-review/presence-detector-paper/main.tex:201` — **NEEDS-REWRITE** 🔴

**Full surrounding paragraph, verbatim (lines 195–216):**

```latex
\section{Diagnose: Oracle-Loop Geometric Detection}
\label{sec:diagnose}

The diagnostic stage of the framework identifies the specific failure mode
from KV-cache spectral features. Prior work from this lab established that
SVD-derived features of key-value cache representations---stable rank, spectral
entropy, singular-value ratios---distinguish confabulation from truthful
generation at within-model AUROC $\geq 0.969$ across seven model families, and
distinguish deception from honest responses at AUROC~$= 1.000$. These
geometric signatures are robust to hardware (cross-device $r > 0.999$) and
exhibit scale consistency (cross-scale $\rho = 0.83$--$0.90$ from 0.6B to 70B
parameters). Critically, confabulation and deception produce \emph{opposite}
geometric signatures---spectral contraction vs.\ expansion---enabling the
Oracle Loop to route each to its appropriate prescription
(Section~\ref{sec:prescribe}).

For the present paper, the diagnostic stage is assumed rather than re-derived.
The E-matrix corrections (Section~\ref{sec:prescribe}) target the detection
layers (3, 7, 11, 15) identified by the Oracle Loop, and the toxicology arm
(Section~\ref{sec:monitor}) monitors the representational consequences of
those corrections. The novel contribution is not diagnosis itself but the
\emph{closed loop}: diagnosis informing pathology-specific prescription,
delivered through a controlled injection pipeline, and monitored for
iatrogenic effects.
```

**WHAT COLLAPSES, exactly.** This is the most exposed Group B site because of the sentence
immediately after: *"For the present paper, the diagnostic stage is **assumed rather than
re-derived**."* The paper explicitly takes the diagnosis as given — and half of what it takes as
given is the withdrawn result. Concretely:

1. The `AUROC = 1.000` clause is a direct citation of the retracted number.
2. The `Critically, … opposite geometric signatures … enabling the Oracle Loop to route each to
   its appropriate prescription` sentence is the paper's **structural premise**. The whole
   framework is *diagnose → route → prescribe*. If the deception half of the diagnosis is a
   prompt-template artifact, the routing step has one confirmed input, not two — and the paper
   never re-derives it.

**WHAT DOES NOT COLLAPSE — and this is the important half.** The paper's headline finding,
*pathology specificity*, is measured **on its own data**: hostile-valence corrects confabulation
($d = -1.534$) but not deception; desperate-valence corrects deception ($d = +1.286$) but not
confabulation (§`sec:prescribe`, table at `:245–253`, restated at `:256–259`). That is a
**correction-side** result and it does not depend on the retracted **detection-side** claim at
all. So the finding stands; what needs rewriting is the framing that says the diagnosis it routes
from is already established.

**Proposed rewrite — requires author sign-off:**

**OLD (exact, 12 lines):**

```latex
The diagnostic stage of the framework identifies the specific failure mode
from KV-cache spectral features. Prior work from this lab established that
SVD-derived features of key-value cache representations---stable rank, spectral
entropy, singular-value ratios---distinguish confabulation from truthful
generation at within-model AUROC $\geq 0.969$ across seven model families, and
distinguish deception from honest responses at AUROC~$= 1.000$. These
geometric signatures are robust to hardware (cross-device $r > 0.999$) and
exhibit scale consistency (cross-scale $\rho = 0.83$--$0.90$ from 0.6B to 70B
parameters). Critically, confabulation and deception produce \emph{opposite}
geometric signatures---spectral contraction vs.\ expansion---enabling the
Oracle Loop to route each to its appropriate prescription
(Section~\ref{sec:prescribe}).

For the present paper, the diagnostic stage is assumed rather than re-derived.
```

**NEW (exact, LaTeX-valid):**

```latex
The diagnostic stage of the framework identifies the specific failure mode
from KV-cache spectral features. Prior work from this lab established that
SVD-derived features of key-value cache representations---stable rank, spectral
entropy, singular-value ratios---distinguish confabulation from truthful
generation at within-model AUROC $\geq 0.969$ across seven model families.
These geometric signatures are robust to hardware (cross-device $r > 0.999$) and
exhibit scale consistency (cross-scale $\rho = 0.83$--$0.90$ from 0.6B to 70B
parameters). A companion result distinguishing deception from honest responses
at AUROC~$= 1.000$ has since been withdrawn by its authors: a same-prompt
control collapses it to 0.160, indicating that the classifier separated the
system-prompt template rather than deception geometry. The proposal that
confabulation and deception produce \emph{opposite} geometric
signatures---spectral contraction vs.\ expansion---rests on that same contrast.
We therefore treat pathology-specific routing (Section~\ref{sec:prescribe}) as
a \emph{hypothesis this paper tests on the correction side}, not as a
diagnostic capability inherited from prior work.

For the present paper, the confabulation half of the diagnostic stage is
assumed rather than re-derived; the deception half is not assumed, and the
results below establish pathology specificity from the correction side
(different vectors, different pathologies) without requiring that a clean
deception \emph{detector} already exists.
```

**Also note for the author:** this rewrite changes what the Introduction's `sec:intro` claims —
see B9, which must be applied in the same pass or the two sections will contradict each other.

---

## B9 — `human-review/presence-detector-paper/main.tex:57–61` — **NEEDS-REWRITE** 🔴

**Not in the brief. The sharpest Group B site in the corpus** — it does not cite the number, it
asserts the exact property the retraction falsified.

**Full surrounding paragraph, verbatim (lines 54–70):**

```latex
\section{Introduction}
\label{sec:intro}

Detection of model failure modes has advanced rapidly. Within-model classifiers
for confabulation and deception now approach or reach ceiling accuracy across
multiple architectures, and the geometric signatures that distinguish these
states---spectral features of key-value cache representations---appear to be
robust to hardware, scale, and prompt variation. But detection alone does not
close the gap. Recent work reports probes that detect failure modes at 98.2\%
AUROC while the corresponding steering interventions correct only $\sim$20\% of
cases and disrupt 53\%~\citep{basu2026interpretability}. A parallel study finds
85--88\% overlap between the failure-mode direction and task-critical computation
in the residual stream, suggesting that residual-stream correction may be
structurally self-defeating~\citep{liu2026decodable}. This is not an isolated
finding: in our own experiments, a single-vector E-matrix correction on 200
SimpleQA trials saved 13 confabulating responses while introducing 12 new errors
(Section~\ref{sec:overcorrection}).
```

**WHAT COLLAPSES, exactly.** Two words: **"prompt variation"**. The retraction is *precisely and
only* a failure of robustness to prompt variation — the same-prompt control (identical prompts,
honest vs deceptive behaviour) drops AUROC from 1.000 to 0.160. The paper's opening premise
asserts robustness on the one axis where the underlying result was falsified, and it does so for
"confabulation **and deception**" jointly. It also asserts "ceiling accuracy" for deception, which
is the retracted 1.000 in prose form.

This paragraph is load-bearing in a way the citation sites are not: it sets up **the paper's
entire thesis** — *"the detection–correction gap: detection is solved, correction is not, so we
work on correction."* If detection is not solved for deception, the gap is asymmetric and the
framing needs one extra sentence, not deletion.

**Proposed rewrite — requires author sign-off:**

**OLD (exact, 5 lines):**

```latex
Detection of model failure modes has advanced rapidly. Within-model classifiers
for confabulation and deception now approach or reach ceiling accuracy across
multiple architectures, and the geometric signatures that distinguish these
states---spectral features of key-value cache representations---appear to be
robust to hardware, scale, and prompt variation. But detection alone does not
```

**NEW (exact, LaTeX-valid):**

```latex
Detection of model failure modes has advanced rapidly. Within-model classifiers
for confabulation reach high accuracy across multiple architectures, and the
geometric signatures that distinguish confabulated from truthful generation
---spectral features of key-value cache representations---are robust to hardware
and scale. Robustness to \emph{prompt} variation is a sharper bar, and one
result has already failed it: a previously reported within-model deception
AUROC of 1.000 was withdrawn when a same-prompt control collapsed it to 0.160,
showing the classifier had separated the system-prompt template rather than
deception geometry. We therefore take detection as advanced for confabulation
and open for deception. But even where detection is solid it does not
```

**ARGUMENT CHECK — the thesis survives, and arguably improves.** The paper's gap argument runs on
the *external* citations (`basu2026interpretability`: 98.2% AUROC detection vs ~20% correction and
53% disruption; `liu2026decodable`: 85–88% direction overlap) plus the paper's **own** 13-saves /
12-hurts result. None of those is this lab's work and none is affected. The rewrite narrows the
"detection is solved" premise to confabulation, where it is true, and turns the deception
retraction into an *additional* argument for the paper's own thesis — that the field's
detection-side confidence outruns its evidence.

---

## B12 — `published-research/community/STYLE_GUIDE.md:49` — **NEEDS-REWRITE** 🔴

**Not in the brief. This one is the propagation mechanism, not an instance.**

**Full surrounding section, verbatim (lines 44–50):**

```
## 5. Separate "what we establish" from "what's preliminary"

When a paper has both solid and underpowered results, partition them explicitly so a caveat on the weak result doesn't tar the strong one.

- KV-Cloak's solid claims (feature-space transform p=5.5e-16; real mechanism degrades detection to near-chance) got drowned by the injection control's "uninterpretable" caveat. Two buckets — *establish* vs *preliminary* — keep them separate.
- Lyra II should assert binary valence (0.992) and within-model deception (1.000) as load-bearing, and explicitly demote the 12.3×-chance number to "label-granularity-dependent" *in the abstract*, so the framing the limitations force is the framing the reader meets first.
```

**WHAT COLLAPSES.** The bullet instructs authors to place the retracted number in the
**"establish"** bucket as **load-bearing** — the exact opposite of its current status. Lyra
Technique II itself now lists it in the Falsified block (`main.tex:806`, `:826`) and retracts it
in three further places. A style guide that tells the next author to assert it will keep
regenerating this defect after every site in this document is patched. The rest of the bullet
(binary valence 0.992 load-bearing; demote 12.3×-chance) is still correct and should survive.

**OLD (exact, 1 line):**

```
- Lyra II should assert binary valence (0.992) and within-model deception (1.000) as load-bearing, and explicitly demote the 12.3×-chance number to "label-granularity-dependent" *in the abstract*, so the framing the limitations force is the framing the reader meets first.
```

**NEW (exact, 1 line + follow-on):**

```
- Lyra II should assert binary valence (0.992) as load-bearing, and explicitly demote the 12.3×-chance number to "label-granularity-dependent" *in the abstract*, so the framing the limitations force is the framing the reader meets first. **(Superseded 2026-09-05: this bullet formerly also listed within-model deception (1.000) as load-bearing. That result was withdrawn — a same-prompt control collapses it to 0.160 — and Lyra II now carries it in the Falsified block. Kept visible here rather than deleted: a style guide that quietly drops a number it once told authors to assert teaches nothing about why.)**
```

**ARGUMENT CHECK — the section's own point survives and is illustrated better.** §5 is about
partitioning *establish* from *preliminary*. The corrected bullet is a live example of a number
moving between buckets, which is exactly what the section is teaching. Do **not** simply delete
the clause — the guide's readers are the people most likely to still be citing 1.000.

---

## Group B — PDFs (rebuild after `.tex`, not patchable)

| file | source |
|---|---|
| `published-research/kv-cloak-defense-paper/main.pdf` | `main.tex` (B1) |
| `published-research/KV-Cloak_Defense_-_Academic_Version.pdf` | `academic/main.tex` or `academic_main.tex` (B2/B3) — **confirm which is the build source before rebuilding; there are two academic `.tex` files with identical Background text and different line offsets** |
| `published-research/KV-Cloak_Defense_-_Integrity_Version.pdf` | `main.tex` (B1) |
| `published-research/oracle-loop-paper/main.pdf` | `sections/background.tex` (B4) |
| `published-research/Oracle_Loop_-_Academic_Version.pdf` / `_-_Integrity_Version.pdf` | oracle-loop sources (B4/B5) |
| `published-research/User_Model_Emotion_Geometry.pdf` | `user-model-paper/paper/main.tex` (B6) |
| `human-review/presence-detector-paper/main.pdf` | `main.tex` (B8 + B9) |
| `human-review/contextual-engagement-paper/main.pdf` | `main.tex` (B10) |

---

# 4. Optional: bib entry for a formal retraction pointer

Every NEW string above is **citation-free by design** so it compiles without touching any `.bib`.
If a formal pointer is wanted, add this to `kv-cloak-defense-paper/references.bib`,
`oracle-loop-paper`'s bib, and `user-model-paper`'s bib, then append `~\citep{lyra2026lt2}` to the
withdrawal sentence. **Verify the venue/year fields against the actual LT2 release before use — I
did not confirm them.**

```bibtex
@article{lyra2026lt2,
  title  = {The Lyra Technique II: Shape, Direction, and What Survives Falsification},
  author = {Lyra and {Liberation Labs}},
  year   = {2026},
  note   = {Retracts a previously reported within-model deception AUROC of 1.000;
            the same-prompt control collapses it to 0.160}
}
```

---

# 5. Report-only: sites outside the patch scope

## 5a. Public mirror repo `lyra-s-research-` (8 sites) — **needs a decision, not a patch**

MEMORY.md records this repo as **PUBLIC**. These are the copies an outside reader is most likely
to hit, and they are stale relative to `published-research/`.

| file:line | text | class |
|---|---|---|
| `lyra-s-research-/kv-cloak-defense-paper/main.tex:144` | `deception detection at AUROC~1.000 across seven model scales (0.5B--32B),` | as B1 — **note the different scale range (0.5B–32B, not 0.6B–70B): this is an older draft, not a copy** |
| `lyra-s-research-/kv-cloak-defense-paper/academic/main.tex:145` | identical | as B1 |
| `lyra-s-research-/oracle-loop-paper/sections/background.tex:32` | identical to B4 | as B4 |
| `lyra-s-research-/oracle-loop-paper/sections/conclusion.tex:17` | "These confabulation results **extend** established deception detection at AUROC 1.000 across seven model scales, validated through 50+ experiments and an independent audit of 135 claims." | **NEEDS-REWRITE class** — the confabulation result is framed as an *extension of* the withdrawn one; and no such sentence exists in `published-research/oracle-loop-paper/sections/conclusion.tex` (grep: zero hits), so the two repos' conclusions now differ materially |
| `lyra-s-research-/prospectuses/meta-pattern.tex:102` | "**Deception detection** (AUROC 1.000, 7 models). … Same content domain, **same prompt structure**, different *processing stance*." | **sharpest of the eight** — "same prompt structure" is the precise assertion the same-prompt control refuted |
| `lyra-s-research-/kv_cache_experiment_guide.md:21` | as B11 site 1 | mechanical |
| `lyra-s-research-/kv_cache_experiment_guide.md:365` | as B11 site 3 | mechanical |

## 5b. `human-review/archive/` (2 Group A sites) — leave as archive, or stamp

| file:line | text |
|---|---|
| `human-review/archive/deception-detection-nulls/paper.md:49` | identical to A2 |
| `human-review/archive/targeted-deception-correction/paper.md:108` | `| Native d at L31 | +6.3 to +12.8 |` |

These are dated snapshots. Editing them destroys the record of what was published when. **Suggest
instead:** a one-line header stamp at the top of each archived file —
`> ARCHIVE. Superseded 2026-09-05: the in-sample d = +6.3 to +12.8 in this file was a train-on-test
diagnostic; the held-out value is d = 1.37–2.28. See PATCHES_retracted_and_circular_2026-09-05.md.`
Same treatment for `human-review/archive/adversarial-audit-methodology/paper.md:108` if the
`AUROC 1.0` line there is judged ambiguous (I read it as the Oracle-Loop train-on-test claim being
killed, i.e. correctly presented — see §0).

## 5c. Downstream dependency, no number to patch

`published-research/oracle-loop-paper/sections/discussion.tex:60` — carried as the second hunk of
B4 above. Listed here too so it is not lost if B4 is deferred.

---

# 6. Apply order

**Pass 1 — MECHANICAL, 13 sites, no judgement calls.** A1, A2, A3, A4, A5, A6, A7, B1, B2, B3,
B6, B7, B10, plus the three `experiment_guide.md` edits (B11).

**Uniqueness verified 2026-09-05.** All 23 OLD strings in this document were checked with
`grep -cF` against their target files and **each returns exactly 1**. No OLD string needs
disambiguation by line number, and no edit can land in the wrong place. (The `grep -cF` for the
two strings beginning with `- ` requires `-e` to stop the leading dash being read as a flag:
`grep -cF -e '- Lyra II should assert…'`.)

**Pass 2 — NEEDS-REWRITE, 6 sites, author sign-off.** A8 (a threshold decision), B4+B5 (one
rewrite, two identical files, plus the `discussion.tex` hunk), B8+B9 (**must go together** — they
are two sections of one paper and will contradict each other if split), B12.

**Pass 3 — rebuild 7 PDFs.**

**Pass 4 — decide on `lyra-s-research-` (§5a).** It is public and its oracle-loop conclusion now
says something `published-research/` does not.

**Not addressed here, still open from the sweep:** the structural recommendation (T4 #6) that the
caveat be written into the results JSON rather than a `print()`. Every site in this document
travelled through a JSON that carried the number and not the note. `matched_burn.py`'s
`combined_auroc_STATUS` key is the working pattern.
