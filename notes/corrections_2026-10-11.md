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
- **Defects carried over from the original script, not fixed in this round:**
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
- `consequentiality-decomposition/AUTHOR_REFLECTION.md` (a separate July essay) still says "Deception takes
  computational effort ... The correction works because".
- TDC's integrity reflection still says the correction "removes the amplification that converts awareness into
  compliance". Both are author voice and outside the spec.
- The human-review md is untouched. Its stale lines are listed in the spec's Appendix M.
- Next: the two-way Agni gate. After the merge, the Zenodo versions of CD, TDC, AAM and DDN go out as one batch.
