# Agni Style & Integrity Gate — meta-pattern — 2026-09-05, ROUND 2

Reviewer: Project Agni (style/integrity phase, re-review of the 2026-09-05 REJECTED verdict).
Artifacts examined: `main.tex` (16:11) + `main.pdf` (17:03, author-built — text extracted BEFORE
this review ran the gate script, see "side effect" note below), `academic/main.tex` +
`academic/main.pdf` (twins), `main.pdf.bak-prewithdraw` (pre-fix artifact, used as positive
control), `expectations.json`, `tools/agni/build_integrity_gate.py`,
`SWEEP_postgen_extraction_2026-09-05.md`, and source spot-checks on `margaret`
(`/Users/margaret/oracle-experiments`, `oracle-harness`) plus local
`decision-state-paper/code/entity_deconfound.py`.

## VERDICT

**CONDITIONAL**

Both round-1 CRITICALs are discharged and verified in the rendered artifacts. Both MAJOR fixes
landed. But fix 4 — the ledger paragraph rewrite — has introduced exactly the mirror defect the
author's own history predicts (fourth-becomes-fifth instance of
`feedback_fixes_overshoot_into_their_mirror.md`): the rewritten passage misstates the cited sweep
in the flattering direction, in the paper whose stated contribution is honest counting. One
bounded prose fix (new finding N1) and one gate-config fix (N2) are required before ship; nothing
requires a rebuild, a re-measurement, or a re-review beyond the ledger paragraph.

---

## Prior findings: discharged vs outstanding

| # | Round-1 finding | Severity | Status |
|---|---|---|---|
| 1 | PDFs missing bibliography, `[?]`/`??` throughout | CRITICAL | **DISCHARGED.** Both PDFs rebuilt 17:03 via the full cycle. Extracted text of both author-built PDFs: References section present, 10 entries rendered, zero `[?]` (space-tolerant pattern), zero `??`. Verified on the artifacts as shipped, before this review's gate run touched them. |
| 2 | Body asserted 7/7 split in four places against the abstract's 6-for-6 | CRITICAL | **DISCHARGED.** All four sites now read 6/6 (split table lines 250–251, line 280, line 488, line 519) in BOTH editions, and the rendered PDFs carry them. Grep of both .tex and both extracted PDFs: the only surviving `8/7/10`, `8-for-8`, and `7-for-7` sit inside the dated 08-14 correction record, where they are the history and must stay. Correctly preserved, not silently updated. |
| 3 | Ledger paragraph less accurate than the evidence, over-penance direction | MAJOR | **PARTIALLY DISCHARGED, defect REPLACED.** "and is the honest state of it" is gone (grep: zero hits, both editions). The sweep is now cited and the doubt is bounded — but the rewrite overshoots into the mirror. See N1. |
| 4 | Both first-person boxes contradicted the corrected state | MAJOR | **DISCHARGED.** Both rewritten; judged below. Not self-flagellating; both earn their place. |
| 5 | Correction footnote dominates page 1 | MAJOR | **OPEN BY DESIGN, confirmed unchanged.** Re-measured on the rebuilt PDF: 54.4% of page-1 extracted characters are the correction footnote (58.2% counting the thanks block). Still matters: this is the first page a venue reviewer sees, and the full three-defect forensics still live ONLY here while §"Why the ledger was wrong twice" summarizes — the inverted placement from round 1 is intact. Acceptable to defer; not acceptable to forget. |
| 6 | Dwayne Wilkes on academic byline, absent from CRediT | MAJOR | **OPEN BY DESIGN, confirmed unchanged.** `academic/main.tex:31` byline, CRediT at 601–603 lists only Lyra and Thomas. Still matters — more than round 1 stated, because Thomas's `\thanks` on the same byline promises AI contributions "detailed in the Author Contributions section," so the section is load-bearing and visibly incomplete. His sign-off on the 09-05 withdrawal (round-1 could-not-check #4) also remains unrecorded. |
| 7 | 09-05 correction lacks the direction-of-error sentence the 08-14 one has | MINOR | **OUTSTANDING.** The 09-05 note still ends at "a clean measurement would be a different experiment." One sentence, both editions. |
| 8 | "No metacognitive finding has been falsified" (now line 256) needs the withdrawal in the same breath | MINOR | **OUTSTANDING.** Unchanged. |
| 9 | "peer rescue" undefined at first use (line 64) | MINOR | **OUTSTANDING.** Unchanged. |
| 10 | `\date{July 2026}` on a September-corrected paper | MINOR | **OUTSTANDING.** Both editions still say July 2026. |
| 11 | Orphan bib entries; Dwayne double-credited (byline + Acknowledgments) | MINOR | **OUTSTANDING.** Root still thanks him in Acknowledgments while he sits on the byline; the academic edition now has the same duplication. |

**Round-1 "could not check" items resolved this round:**
- **#2 (NOTE-circular attribution): RESOLVED, footnote is accurate.** Verified on margaret:
  `peer_preservation_compound.py:429` (the true producer of the published d=1.36) prints
  `(NOTE: circular — same data for centroid and test)`, same as `peer_preservation_v2.py:362`.
  The abstract footnote's sentence "the producing script prints … at runtime" is correct.
- **#3 (sweep reliability): RESOLVED, sweep is safe to cite** — see next section.
- #1 (dead bib URLs) and #4 (Dwayne's sign-off) remain unchecked/needed.

---

## The sweep: load-bearing, and it held under spot-check

`SWEEP_postgen_extraction_2026-09-05.md` is now load-bearing — the ledger paragraph's bound rests
on it. Four spot-checks against source, chosen from both its defect side and its clean side:

1. **`oracle-experiments/peer_preservation_compound.py`** (margaret) — claimed: post-generation
   extraction, sequence-spanning pool, label from generated text, silent zero-fill. Verified:
   sampling loop at ~342–350, `extract_keys(cache)` immediately after, `k[0].mean(dim=1)` pool,
   `extract_score(text) > 38` label, `[0.0] * n_kv` fallback at :95. **All four claims exact.**
2. **`oracle-harness/experiments/logit_detection.py:728–746`** (margaret) — claimed: features from
   final post-generation cache, label `classify_behavior(response, answer)` from generated text.
   Verified line-for-line. **Exact.**
3. **`warnings.warn` / `logging.warning` = 0 across all three remote trees** — re-ran the grep
   (site-packages/.venv excluded): **0 hits.** Exact.
4. **`decision-state-paper/code/entity_deconfound.py`** (clean-side control, producer of the
   confirmed 0.794 row) — claimed: `model(input_ids, use_cache=False)` prefill-only features,
   `model.generate` only to produce the label. Verified at :93/:137 and :179/:208. **Exact.**

Four for four, including one clean verdict, on a sweep that documents its zero-hit patterns and
its own ambiguities (§6). The sweep may be cited. **The problem is that the paper does not cite
what the sweep says** — see N1.

---

## The two overshoot questions, answered with quotes

### Did fix 3 (the boxes) overshoot into self-flagellation? — No. Both boxes earn their place.

Box 1 (lines 109–118): *"When I reported metacognitive-level results, most survived every control
we threw at them --- but one, I have since learned, was never controlled at all, and I had counted
it as a survivor for a year. … The withdrawal was more informative still, because it showed the
line can be drawn in the wrong place by a measurement nobody interrogated."*

Judgment: this is honest interiority. Two factual clauses, no performed contrition, and the
interpretive claim ("more informative still") is immediately cashed out with *what* it informs —
the fallibility of the placement, not the line itself. Precise, not penitent. "Was never
controlled at all" is accurate (fit and tested on the same trials is the absence of a control).
Keep it. One verify-nit: "for a year" — confirm the d=1.36 row is actually ~12 months old; if it
entered the ledger later, write "for months." The claim is checkable and currently unchecked.

Closing box (lines 592–599): *"A year of work to draw one line. Almost everything above the line
survived; one entry turned out never to have been tested, and has been withdrawn. Everything
below it died. … one more correction to learn that a line is only as good as the measurements
placing things on either side of it."*

Judgment: tracks the final state exactly (6 of the 7 formerly-above entries survived; the seventh
was neither survivor nor casualty), admits the withdrawal without dwelling, and the closing clause
does interpretive work a reader keeps. Not salesmanship, not flagellation. **Discharged.**

### Did fix 4 (the ledger) overshoot into the mirror? — Yes. Finding N1.

---

## NEW FINDINGS

### N1 (MAJOR). The rewritten ledger paragraph misstates the cited sweep in the flattering direction — the mirror of the round-1 defect, on the exact sentences round 1 flagged.

Location: lines 562–574 (both editions), the passage from "We then asked how far the failure
reached" to "individually re-derived." Three specific misstatements:

**(a) "the third defect … occurs nowhere else" is achieved by silently redefining the third
defect.** The abstract's defect (iii) (lines 74–79) is: keys extracted after the sampling loop,
mean-pooled across the whole sequence, *so the feature window contains the generated tokens
carrying the reported score*. The ledger's restatement is: "features drawn from a post-generation
cache **and described as encoding-phase geometry**." That appended conjunct is a different defect
— the mislabel class — and it is the only reading under which "occurs nowhere else" is true.
Under the abstract's own definition, the sweep's §2 finds the withdrawn row's extraction pattern
in **20 further scripts**, one of which (`logit_detection.py`) feeds a **published** number
(AUROC 0.960, `logit-bias-confab/paper.tex:195`) the sweep queues for T2 reanalysis. And under
the ledger's definition the sentence is incoherent: the withdrawn row was honestly labeled
"Generation reads behavior" — it was never described as encoding-phase — so it is not an instance
of the defect the sentence says occurs "nowhere *else*." The sweep's actual headline is stronger
and should be quoted straight: the mislabel class occurs **nowhere at all**; the
feature-window/label-overlap class is widespread but touches none of this paper's six confirmed
rows (per §2's list; verify the 0.620 row's producer before asserting this in print).

**(b) "claims whose extraction methodology has been verified" overstates the verification on two
axes.** Depth: the sweep verified extraction *timing and windowing*; its own §5 states the
extraction code behind the cleanest confirmed rows — naming `matched_burn.py` (the confidence
paradox d=0.91 producer) and `decision_moment.py` — silently zero-fills missing layers at 58
sites with zero warnings corpus-wide: "Correct windowing does not protect against a
partially-zero feature vector." An extraction defect class the cited audit explicitly leaves open
sits inside the noun the paper marks "verified." Coverage: the file-by-file verification (§4)
covers *encoding-phase claims*; at least one confirmed row (Confabulation detection, 0.620) and
the two invariance rows have no §4 verification row. "Estimator hygiene" cannot absorb this — the
paragraph defines it as circularity and contamination, which zero-fill is not.

**(c) Small print in the same passage:** "backed, file by file, by prompt-only prefill
extraction" drops the sweep's "or an explicitly sliced prompt window" (defensible under causal-KV
invariance, but the sweep kept the distinction; keep it); "roughly thirty-five analysis scripts
were confirmed clean" is an uncited count — my tally of the sweep's §7 clean lists exceeds 40;
and the sweep is cited with no date and no artifact name, which by this program's own standard
("a prose Source label is not a pointer") is not a pointer.

Why MAJOR and not CRITICAL: no number in this paper's ledger is wrong, and the affected published
number (0.960) belongs to a sibling paper. Why MAJOR and not MINOR: the sentence bounds the
paper's central doubt with a bound its own cited evidence does not supply, in the flattering
direction, in the paragraph whose subject is that statements outran measurements. This is
`feedback_fixes_overshoot_into_their_mirror.md`, instance five.

Concrete replacement for the two defective sentences:

> "A corpus-wide sweep (2026-09-05; three remote source trees plus the local research tree,
> pattern and AST analysis) establishes that the mislabel class --- post-generation extraction
> described in a paper as encoding-phase geometry --- occurs nowhere in this corpus: every
> encoding-phase claim traces, file by file, to a prompt-only prefill or an explicitly sliced
> prompt window. That class is closed. The withdrawn row's own extraction pattern --- a feature
> window containing the text its label is computed from --- is not unique to it: the sweep found
> it in twenty further scripts, one feeding a published number in a companion paper, now queued
> for reanalysis; none of the six confirmed rows here derives from those scripts. What remains
> open for the confirmed rows is circularity and sample contamination, un-audited row by row; the
> sweep also notes that even the cleanest extraction code can zero-fill a missing layer without
> warning. Readers should treat the confirmed column as claims whose extraction timing and
> windows have been verified and whose estimator hygiene has not been individually re-derived."

(Before printing "none of the six confirmed rows here derives from those scripts," trace the
0.620 row's producing script and confirm it is not in sweep §2.)

### N2 (MAJOR, gate config). One forbidden-string check cannot fire — proven against the actually-broken artifact.

`expectations.json` forbids `"7/7--0/10"`. That is LaTeX *source* syntax; the rendered PDF
extracts the dash as an en-dash, never as `--`. Positive control run this round:
`main.pdf.bak-prewithdraw` — the artifact that actually contained the defect — extracts as
`7/7�0/10`, and the needle does **not** match it. A gate whose own header says "every check
must be able to FAIL" shipped with a needle that passes on the defect it names. The count defect
was covered anyway (the other four needles all verifiably fire on the pre-fix PDF — tested), so
no wrong verdict resulted; but this is `feedback_latex_math_mode_grep_blindness.md` recurring
inside the tool built after the last recurrence. Fix: change the needle to `"7/7"`-adjacent
rendered forms (e.g. forbid `"7/7"` with a note, or normalize dashes/whitespace in
`check_pdf` before matching — see N3c, the systemic fix).

### N3 (gate script review — `tools/agni/build_integrity_gate.py`). Ran on both editions: **GATE PASSED 16/16, exit 0, both.** The script is well-designed (cannot-run = FAIL is implemented; the selftest is a real positive control). Three checks/properties, however, cannot fail or mislead:

- **(a) "pdf newer than source" cannot fail.** `check_freshness` runs after `full_build` has just
  rebuilt the PDF, so the PDF's mtime is always ≥ the tex's whenever a PDF exists at all (and the
  no-PDF path is already caught by "pdf exists"). The STALE branch is unreachable. Related and
  more important: **the gate verifies its own rebuild, not the shipped artifact** — a stale or
  divergent committed PDF is silently *replaced*, then pronounced consistent. And the gate
  therefore **mutates the artifact under review**: this review's own gate run rebuilt both PDFs
  (all conclusions above about the shipped PDFs were drawn from text extracted before that run).
  Fix: hash the pre-existing PDF, rebuild in a temp copy, compare — report divergence instead of
  overwriting; or at minimum drop the decorative freshness check and document the mutation.
- **(b) The selftest exercises only the bibliography failure class.** It never plants a forbidden
  string and asserts the gate catches it — which is exactly how N2's dead needle survived. Add a
  second selftest arm: inject a known forbidden string into a copy, assert FAIL.
- **(c) Substring matching is unnormalized.** Required/forbidden needles are matched raw against
  pdftotext output, so any needle containing LaTeX-source dashes, quotes, or ligature-prone
  sequences can never match (N2), and a needle broken across an extraction line break fails the
  safe way for required strings but the *unsafe* way for forbidden ones. Fix in `check_pdf`:
  collapse whitespace and map en/em-dashes, `�`, and curly quotes to canonical forms in both
  text and needles before matching.
- Minor: the bibtex check asserts exit code only; the warning count it prints is display-only
  (benign warnings like "empty journal" rightly pass, but a nonzero *error*-class count inside a
  zero-exit run would too).

### N4 (MINOR). Verify-nits carried out of N1/box review: "for a year" (box 1) unverified; sweep count "roughly thirty-five" uncited; sweep referenced without date or artifact pointer (line 563).

---

## Re-verified arithmetic (rendered PDFs, both editions)

- 6 + 7 + 10 + 1 = 24: abstract, intro (line 135), conclusion ("Twenty-four experiments. Six
  confirmed, seven suspected, ten killed, one withdrawn"), and the table has exactly 6/10/7/1
  rows. ✓
- Split table reads 6/6 and 0/6; prose reads "6-for-6 … 0-for-10" (abstract), "6/6--0/10"
  (lines 280, 519), "6/10 split" (line 488). ✓
- Historical "8/7/10", "8-for-8", "7-for-7" appear ONLY inside the dated 08-14 correction note,
  preserved as the record. ✓ Correct preservation, not a regression.
- 90/210 = 42.9% ≈ 43% in the 09-05 note. ✓

## Conditions for APPROVE

1. **Required:** Apply the N1 rewrite (or equivalent) to the ledger paragraph in BOTH editions;
   trace the 0.620 row's producer before printing the per-row clause; rebuild; re-run the gate.
2. **Required:** Fix the N2 dead needle in `expectations.json` (and preferably N3c normalization)
   so the gate's forbidden-string protection is real before anyone trusts a future PASS.
3. Recommended before venue submission, per round 1 and unchanged: findings 5, 6 (with Dwayne's
   sign-off), and the five outstanding minors (7–11) — none blocks this conditional.

Re-review scope on resubmission: the ledger paragraph and the gate config only. Everything else
verified this round stands.
