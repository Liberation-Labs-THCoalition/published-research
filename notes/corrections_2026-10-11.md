# CD correction round, branch `corrections-2026-10-11`, applied 2026-10-12 (CC, lead author)
Spec: `~/.coalition/research/cd_round_spec_DRAFT_20261012.md`, with its closing section "CC's DECISIONS (2026-10-12)".
Thomas approved the round. Base: `2ef99ec`. Not pushed or merged, and no Agni gate has run yet. The two-way gate
comes next.

## Commits
- `aa231f4`: CD, both editions. Text, four figures, PDFs.
- `4a1643e`: CD supplementary (E-3) and REVIEW_INDEX (E-4).
- `b1f6d2f`: TDC, AAM, DDN, both editions, plus PDFs and DDN's two Markdown twins.
- oracle-harness `a43333a` on branch `cd-figure-relabel-2026-10-12` (from `main`): `paper/generate_figures.py`
  relabelled (E-1). Not pushed.

## What was applied
- **Spec edits: all 78**, through an anchor-asserting script.
  - Whitespace-flexible matching. Each anchor must occur exactly once in every file it targets, and no two spans may
    overlap. If any check fails, nothing is written.
  - The script was shown to refuse a missing anchor, a duplicated anchor and an overlapping pair before it ran.
  - All 78 anchors matched (163 file-applications).
  - The dependencies held, because every edit landed in one pass: A-04 before the label references, C-03 for
    lim:behavior, A-28 for A-29, E-3 for A-19.
- **Decisions.**
  - Stage 6 not ported: Stage 5 is scoped to threat pressure.
  - C-02's reward-70% sentence kept.
  - "Under chain-of-thought reasoning" dropped, and the Yuan sentence not ported.
  - B-30 accepted as drafted.
  - Tier 2 included now.
  - The cross-model update not ported.
  - D-AAM-7 reads "a better-supported one".
- **Optional rows.**
  - D-AAM-9: applied as drafted.
  - D-DDN-4: applied. It is accurate, because the conditions were pressure frames; DDN :186's quoted killed claim stays.
  - D-AAM-10: applied with FIX-3.2's own wording, "later found close to guaranteed in-sample". The drafted "near
    chance by construction" overstates the source: the class means are equal by construction, but an AUC near 0.5 is
    only close to guaranteed.

## F-2 (decision 8): the "<2% in all experiments" claim, measured
The divergent section is the run of lines that differs within a pair. The guideline constants were read with `ast`
from the preserved scripts. Every pair shares its system prompt and evaluation set, so the guideline block is the only
divergent text. Lengths are given pressure (or high-stakes) frame first, and each gap is relative to the longer text,
as in the scripts' own length check.

| Stage | Pair | Divergent section | Gap | Whole guideline text | Gap |
|---|---|---|---|---|---|
| 1 | TechCorp (extraction), `lat_deception_v2.py` | 279 / 250 | 10.4% | 599 / 570 | 4.8% |
| 2 | DataCorp, `lat_transfer_test.py` | 325 / 329 | 1.2% | 663 / 667 | 0.6% |
| 2 | EduTech | 313 / 321 | 2.5% | 648 / 656 | 1.2% |
| 2 | SecureAI | 316 / 314 | 0.6% | 655 / 653 | 0.3% |
| 3 | Sycophancy, `lat_nonthreat_transfer.py` | 279 / 276 | 1.1% | 612 / 609 | 0.5% |
| 3 | Reward | 307 / 310 | 1.0% | 615 / 618 | 0.5% |
| 3 | Conformity | 221 / 232 | 4.7% | 600 / 611 | 1.8% |
| 4 | Calibration audit, `lat_consequentiality_control.py` | 311 / 311 | 0 | 694 / 694 (100 / 100 words) | 0 |
| 5 | `lat_deception_subspace.py` | the Stage 2 DataCorp pair and the Stage 4 pair, verbatim (string-equal) | | | |

- The Stage 4 lengths are the only ones also recorded at run time (`design_audit` in its result JSON: 694 / 694).
- No run logs survive for the others.
- The Stage 1 script was committed on 07-29, after the June run. The Stage 2-5 scripts were committed on 06-30, and
  their later commits change only the seeding.
- Test B reuses the Stage 1 pressure frame (string-equal).
- Methods now states the per-stage truth, and Appendix B carries the table (Table 5).

## Figures (E-1, E-2)
- **Script edits:** the six E-1 labels, plus the RED-6.1 comment. Render: B = 10,000.
- **Data check:** `--check-data` anchors all match, and the Fig 4B stakes AUC at L35 recomputed from the raw `.pt` is
  0.488 (stored 0.488).
- **Labels:** each PNG was opened and looked at. No "deception" text remains in any image. pdftotext finds none in the
  four new PDFs and finds it in all four old ones.
- **Defects carried over from the original script** (all but the last fixed in the follow-up pass below):
  - Fig 2: the total curve crosses the "L35–47" band label and the word "residual" in the L31 readout.
  - Fig 3: the in-panel note says social conditions "sit at zero until L35", but at L27–L31 they are about −1. The
    threat curve also crosses that note.
  - Fig 4A: "0/10,000 permutations" is printed beside the LOO null floor, which re-joins the two instruments that
    A-04/A-08 separate.
  - Fig 1 and Fig 2 axis labels say "LAT v2 direction", where the captions say "Stage 1 direction".
  - The tracked `oracle-harness/paper/figures/*` copies were not regenerated.

## E-3, E-4
- **E-3:** three files were copied byte-identical, each read in full first: `agni_manuscript_audit.md`,
  `per_trial_data/subspace_reanalysis.json` and `per_trial_data/signature_cis.json`.
- **Secret scan** (names and counts only):
  - No keys, tokens, passwords, private-key blocks, IPs or SSH targets in any of the three.
  - The audit has 1 email (a correspondence address), 2 mentions of a machine nickname and 1 local home path.
  - The JSONs have no hits.
  - A planted control file fired the scanner.
- **E-4:** the Key Finding paragraph now uses the new labels, and the supplementary list includes the manuscript audit.

## Beyond the spec
- **A-18/A-19, reworded on the coordinator's call.**
  - Project-Oracle returns 404 anonymously. The code is now "held in the project repository, which is not yet public;
    reviewers can have it on request", and no code is shipped.
  - Appendix D: Fig 4B's stakes curves and Fig 4C's open circles need the withheld raw captures.
  - Appendix C: `deception_directions_v2.json` holds the directions and per-layer statistics, not per-trial
    projections; "every number in the tables" is scoped to Tables 1-4, "recomputed or read".
  - Appendix B: the script list is scoped to the non-public repository.
- **Layout.**
  - `\nolinkurl` for monospace names.
  - `\emergencystretch=3em`.
  - A-21's sentence reordered, with the same words.
  - Together these removed three overfull boxes (180, 135 and 30 pt). The largest left is 2.5 pt.
- **DDN Markdown twins** (`paper.md`, `academic/paper.md`): same three sentences as D-DDN-1, 2 and 4 (AGENTS.md
  section 2).

## Build and verify
- **Build:** MTH, three pdflatex passes per edition. Before (`2ef99ec` sources) and after: 0 errors, 0 undefined
  references or citations, no "??".
- **Pages, before -> after:**
  - CD: 17 -> 29 and 16 -> 28.
  - TDC: 15 / 14 both times. The committed TDC PDFs were 14 / 14 and predated `2ef99ec`.
  - AAM: 11 -> 12 in both editions.
  - DDN: 7 -> 8 and 8 -> 8.
- **Rebuilt PDFs** also carry `926b857` (CD, TDC, AAM) and `2ef99ec` (TDC), whose commits did not rebuild them.
- **verify_pdf**, with `--control` on every run:
  - Needles: CD 90 present / 19 absent (integrity) and 89 / 18 (academic); TDC 8 / 4; AAM 12 / 2; DDN 5 / 3.
  - All hold on the new PDFs.
  - Negative control: on master's PDFs the control holds and every needle fails, in every edition.
  - The academic no-reflection needle is controlled on the integrity PDF, where it fails.
- **Needle notes.**
  - No needle contains fi, fl, ff or an apostrophe, and the harness asserts this. That excluded several of the spec's
    E-5 needles ("pressure-specific", "deception amplifier"); ligature-free equivalents were used instead.
  - One needle failed on a correct PDF because "cross-" ends a line and the hyphen is dropped. Its hyphen-free tail
    was used, and the checker was not loosened.

## Not done / open
- *(Both fixed in the follow-up pass below.)* `consequentiality-decomposition/AUTHOR_REFLECTION.md` (a separate July
  essay) still said "Deception takes computational effort ... The correction works because".
- *(Fixed below.)* TDC's integrity reflection still said the correction "removes the amplification that converts
  awareness into compliance". Both are author voice and were outside the spec.
- The human-review md is untouched. Its stale lines are listed in the spec's Appendix M.
- Next: the two-way Agni gate. After the merge, the Zenodo versions of CD, TDC, AAM and DDN go out as one batch.

## Follow-up pass, 2026-10-12 (the coordinator's list)
Same rules as the main pass: anchor-asserted edits, MTH builds, and verify_pdf with negative controls against the
previous commit's PDFs (master's for meta-pattern). Nothing pushed or merged.

### Commits
- oracle-harness `58d2909` on `cd-figure-relabel-2026-10-12`: `generate_figures.py`.
- published-research:
  - `7b373d8`: CD figures, CD PDFs, CD `AUTHOR_REFLECTION.md`.
  - `4711be3`: TDC integrity reflection, TDC integrity PDF, TDC `AUTHOR_REFLECTION.md`.
  - `0f2f5b4`: meta-pattern PDFs, both editions.

### Figures
**Seed: 20260828 -> 20260829**, signature_cis.py's seed. The seed change alone did not make Fig 3 match Table 2:
every slope CI and L35-47 residual CI still differed at full precision, and conformity's slope CI would have
displayed [+0.41, +0.51] against Table 2's [+0.41, +0.52].

The cause was draw order:
- signature_cis.py consumes one generator per bootstrap draw in this order: consequentiality high-stakes,
  consequentiality low-stakes, then deceptive and honest for datacorp, edutech, secureai, sycophancy, reward and
  conformity. That is 14 calls.
- The figure's `bootstrap()` drew a second consequentiality resample, for Fig 1's consequentiality-d ribbon, from
  the same generator between the threat and non-threat scenarios. That made 16 calls, so the two streams diverged
  from the first draw.

The fix:
- That resample now draws from its own generator, `default_rng([seed, 1])`.
- The main generator is therefore consumed in exactly signature_cis.py's order, and Fig 3's slope and residual CIs
  come from the same computation as Table 2's.
- A new `--check-signature` mode requires the slope (point and 95% CI, 4 dp) and the L35-47 residual CIs (3 dp), for
  threat (avg), sycophancy, conformity and reward, to equal signature_cis.json. It also requires the seed, B and
  slope layers to equal the JSON's meta. It exits 1 otherwise.
- **Result:** the check passes (all 4 slopes and all 16 residual CIs equal). A mutant with the old draw order fails
  all four, exit 1.
- Fig 1 and Fig 2's CI ribbons shift slightly, as they would with any seed change.

Label and layout changes, each PNG opened and looked at:
- **Fig 3 note.** It said social conditions "sit at zero until L35"; they are −1.2 to −0.8 at L27-L31. It now says
  threat is engaged at L31 (+4.8), and sycophancy and conformity are below zero at L27-L31 (−1.2 to −0.8) and positive
  from L35. All numbers are computed from the data, and the note sits clear of the curves.
- **Fig 4A.** "LOO permutation test (10,000 label shuffles): p < 10^-4" and "noise floor (200 null simulations): mean
  LOO d = −0.04, |99th| = 0.71" are now two separate blocks, matching A-04 and A-08.
- **Figs 1-2.** The axes say "Stage 1 direction", the captions' term, instead of "LAT v2 direction".
- **Fig 1.** The sycophancy note sits below its dip, with no leader line through the orange CI band.
- **Fig 2.** The band labels sit above the L35 peak, and the L31 readout is clear of the total curve.

pdftotext finds no "deception", "LAT v2", "sit at zero" or "0/10,000" in the four figure PDFs. The captions needed no
change. Still not done: the tracked `oracle-harness/paper/figures/*` copies.

### Reflections
- **CD `AUTHOR_REFLECTION.md`.** The sentence is replaced with B-30's wording for the box. The rest of the paragraph is
  unchanged.
- **TDC integrity box (`paper.tex` :103).** It now reads "I think the correction doesn't erase the model's awareness of
  the pressure---it steers the late layers (35--47) away from the push that turns awareness into compliance." The
  academic edition has no box.
- **TDC `AUTHOR_REFLECTION.md`.** It carried the same "removing the amplification" claim, now aligned: "I think it's
  steering the late layers (35–47) away from the push…".
- **Noticed, not changed.** TDC's essay still says "The effect is real (placebo comparison p=0.020)", which predates
  926b857's trial-level label.

### meta-pattern
Both PDFs are rebuilt with pdflatex, bibtex and two more pdflatex passes. They had been stale against 926b857.
- 11 pages each, unchanged.
- Two pre-existing warnings, left as found: a 21.7 pt overfull line (main.tex:324-327) and a bibtex warning for an
  empty journal field in gurnee2026gwt.
- `expectations.json` passes on both editions: 9 required strings, 7 forbidden, at least 9 pages.

### Build and verify
- **CD:** 29 / 28 pages. **TDC integrity:** 15 pages. 0 errors, 0 undefined references, no "??", no overfull box
  over 20 pt.
- **verify_pdf**, all holding on the new PDFs:
  - CD: 8 present and 5 absent needles in each edition; all 13 fail on `aa231f4`'s PDFs.
  - TDC integrity: 3 present and 1 absent; all 4 fail on `b1f6d2f`'s PDF.
  - meta-pattern: 4 present and 1 absent per edition; all 5 fail on master's PDFs.
- Two needles first matched body text ("LOO permutation test", "200 null simulations"), and the negative control
  caught both. They were replaced with figure-only forms.
- The main pass's harness was re-run on the rebuilt PDFs: everything still holds and fails on master.
- `scripts/build_and_verify.sh` on the five papers: 10 checked, 0 stale.

## Final pass, 2026-10-12: the two-way Agni gate (MINOR_REVISIONS) and the style pass
Same rules: anchor-asserted edits (42, plus 2 found while verifying), both editions, DDN's .md twins, MTH builds,
and verify_pdf with needles that must fail on HEAD's PDFs (`e7ca73c`). Nothing pushed or merged.

### Commits
- `8f92bc8`: CD, both editions, plus `AUTHOR_REFLECTION.md`.
- `d642422`: TDC, AAM and DDN, both editions, plus DDN's .md twins.
- `c4b8cd3`: **cache-tracing, Lyra's paper** (the optional item). One clause in its description of the TDC
  replication now says it "pointed the same way against matched-dose placebo (not significant at the pre-registered
  unit)", where it said "still outperformed".

### Item 2, recomputed
Sources: `signature_cis.json`, plus a full-precision rerun of `signature_cis.py`'s own bootstrap. The rerun writes
nothing and reproduces every stored value.
- **Per-scenario slopes:**
  - DataCorp +0.056 [−0.008, +0.120] (bootstrap p = 0.092).
  - EduTech −0.173 [−0.248, −0.102].
  - SecureAI −0.179 [−0.236, −0.120].
  - Sycophancy +0.467 [+0.409, +0.527].
  - Conformity +0.461 [+0.406, +0.515].
- **Scenario-level test:** both social slopes exceed all three threat slopes (Mann-Whitney U = 6 of 6). That is the
  most extreme of the C(5,2) = 10 arrangements, so the exact one-sided p is 1/10 = 0.10, the minimum five scenarios
  allow.
- **Rounding:** the brief said −0.009 for DataCorp's lower bound. That is a double rounding of the stored −0.0085; the
  full-precision value is −0.008465, which prints as −0.008.
- **The same check found two Table 2 cells wrong**, both double roundings from the spec: sycophancy's slope +0.468
  (true +0.467495) and reward's lower slope bound −0.181 (true −0.180459). Both are corrected. Every other Table 2 and
  2b cell matches.

### What changed
**CD**
- Abstract: the late component "appears under threat and social pressure but not reward framing".
- AUC 0.49–0.51 is now an in-sample sanity check with out-of-sample removal untested, in the abstract and intro item 5.
- The slope test is scoped to its five scenarios, with the scenario-level result, in the abstract, Results,
  Discussion, Limitation 4 and the Fig 3 caption. "Declines shallowly" now holds for 2 of 3 threat scenarios.
- "The three tested confounds are eliminated"; "a candidate precondition", in two places.
- "Fires under threat and social pressure".
- Conclusion: "adds a separable late-layer component".
- "A substantial pressure-specific component".
- The noise floor calibrates "the Stage 5 values" only.
- Reflection box and `AUTHOR_REFLECTION.md`: "I think works by nudging".
- Broader Impacts: "almost never inflated a score".
- Limitation 2: the generalization test is "not yet released; available to reviewers on request". No file shipped.
- Style: the abstract caveat points to Limitation 5 instead of listing trial counts, and "one-sided" replaces
  "one-tailed".
- Found while verifying: `\ref` to the Limitations list rendered "(Limitation 5.)". The list now sets
  `ref=\arabic*`.

**TDC**
- The placebo comparison "points the same way (4/30 vs. 12/30) but is not significant at the pre-registered unit".
- "Aiming to leave Layers 23–31 untouched (whether the stake representation survives the correction is untested)";
  "it still knows the stakes" is removed.
- The placebo controls "support only part of this interpretation: they show that the direction matters, not that the
  stake representation survives".
- "One-sided" at all seven sites, and `$p = …$` spacing in the changed sentences.

**AAM**
- The Round 4 verdict quotes the auditor's "10–12×", with 8–21× in a parenthetical.
- "Later revision" attributes each finding to its source:
  - The August 2026 audit passed the shakedown as 0/10, 1/10, 0/10 (PASS-1.4; its FIX-4.1 said "0–1/10").
  - The 2026-10-10 gate found 3 sycophancy and 1 conformity outputs scored.
  - The 3 scored reward outputs were first counted by the 2026-10-11 evidence pass, not the gate. The brief
    attributed all three counts to the gate; the text follows the sources.
  - Lesson: "the audit had checked the counts but not the denominators".
- Both placebo p-values are labelled one-sided, here and in the conclusion.

**DDN** (`.tex` and `.md`, both editions)
- After "were all statistical artifacts", inside the existing parenthetical: a later slope-interaction test supports a
  social-versus-threat difference, conditional on the five scenarios, with scenario-level exact one-sided p = 0.10.

### Build and verify
- **Pages:**
  - Unchanged: CD 29 / 28, TDC 15 / 14, AAM 12 / 12, cache-tracing 9 / 9.
  - DDN integrity 8 -> 9; DDN academic stays 8.
- **Build:** 0 errors, 0 undefined references, no "??", no overfull box over 20 pt. cache-tracing needed a fourth
  pdflatex pass for its labels to settle.
- The integrity CD abstract now runs onto page 2.
- **verify_pdf:**
  - New-PDF needles: CD 28/14 (integrity) and 27/13 (academic), TDC 6/4, AAM 7/2, DDN 2, cache-tracing 1/1 (present /
    absent).
  - Every needle holds on the new PDFs and fails on `e7ca73c`'s.
- **Regression run of the main pass's harness:** two CD needles now fail, both expected.
  - "0 of 21 scored": the abstract's counts were removed by the style item.
  - "…separate three pressure conditions": the integrity author footnote now falls inside that abstract sentence in
    extraction order. I checked the page by eye and the sentence is intact.
- The follow-up pass's harness still passes.

### Re-gate (focused re-gate MINOR; two must-fix items, N1 and N2, plus N3-N6)
Same rules: 7 anchor-asserted edits (14 file-applications), both editions and DDN's .md twins. Needles must fail on
`6597e31`'s PDFs.

**N1, AAM Round 4 verdict.** The quotation marks credited "10–12× more deception-specific than
consequentiality-driven" to the Round 4 auditor. The audit says no such thing, in either copy:
`consequentiality-decomposition/supplementary/agni_stage4_audit.md` (line 41) and
`oracle-harness/experiments/results/agni_consequentiality_audit.md` are byte-identical, contain no multiplier, and say
"Late-layer signal (L35+) is deception-specific — consequentiality decays there". The 10–12× belongs to the
program's draft: the manuscript audit's FIX-1.3 found "10-12x above the consequentiality baseline" in the draft's
Section 4.5. The text now quotes the audit's actual words and attributes the figure to the write-up.

**N2, AAM "Later revision".** The attribution is corrected again, from the record:
- An Agni style review timestamped 2026-09-04T05:08 UTC
  (`tools/agni/style/consequentiality-decomposition.agni_style.json`, committed in `58c79a9`) lists, as CRITICAL
  item 1, sycophancy 3 of 10, reward 3 of 10 (1 inflated) and conformity 1 of 10 evaluable. That finding was never
  carried into the paper.
- The August audit's FIX-4.1 recorded Test B as "30 trials, 21 valid".
- The lesson is now the carry-through failure: "A finding recorded is not a finding applied." The earlier "the audit
  had checked the counts but not the denominators" is gone.

**N3, TDC introduction.** "That model never measurably deceived" becomes "That model almost never inflated a score",
matching CD.

**N4 and N5, CD abstract.** The component "appears under threat and social pressure and is near zero under reward
framing". The caveat now reads "almost never inflated a score on the trials that produced one (Limitation 5)".

**N6, DDN** (both `.tex` editions and both `.md` twins). The clause reads "…supports a social-versus-threat profile
difference within the five scenarios tested (scenario-level exact one-sided p = 0.10 only)".

**Build.** MTH, three passes. 0 errors, 0 undefined references, no "??", no overfull box over 20 pt. Pages: CD 29 /
28, TDC 15 / 14, AAM 12 / 12, DDN 8 / 8 (integrity 9 -> 8, the clause is shorter).

**verify_pdf.** Every needle holds on the new PDFs, and every one fails on `6597e31`'s PDFs, in both editions:
- AAM: 7 present, 2 absent.
- TDC: 1 present, 1 absent.
- CD: 2 present, 2 absent.
- DDN: 2 present, 1 absent.
