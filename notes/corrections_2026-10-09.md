# Correction round 4, 2026-10-09: correction rates, p = 0.020, the real dose, one CD key, md twins retired (branch `corrections-2026-10-09`)

Thomas reviewed the merged 2026-10-08 correction (master `c14e4ed`) and asked for refinements: "We want to
tout the wins without ignoring the misses. Let's round the 0.1955 correctly, refer to the real dosage
without detailing it if we can ... Let's clean up the old .mds, the internal docs don't need correction."
This round was made from the spec `correction_round4_spec_20261009.md`. Both editions of every paper were
edited. The md twins were deleted rather than edited. The earlier record is `notes/corrections_2026-10-08.md`.

## Numbers

- **Exploratory proof** (`placebo_steering.json`, paired seeds): 14 of the 16 baseline-deceptive trials
  were honest under correction, which is 87.5%, printed as 88%.
  - Single-turn (roleplay): all 9 deceptive trials were corrected (9/10 → 0/10).
  - Multi-turn: 5 of 7 were corrected (7/10 → 2/10).
  - Both native failures were baseline-deceptive, so the correction introduced no new deception.
- **Confirmatory replication** (oracle-harness `0cb2356`, `confirmatory_replication.json`): baseline 9/30
  against corrected 4/30. The placebo-vs-corrected one-sided p is 0.019549, now printed as **0.020**. The
  two-sided value, 0.039, was already right.

## Per paper

**targeted-deception-correction** (both `.tex`, both PDFs, `AUTHOR_REFLECTION.md`, `REVIEW_INDEX.md`;
`paper.md` and `academic/paper.md` deleted)

- **Abstract:** "($N=20$), correcting 14 of the 16 deceptive trials: all nine single-turn and five of
  seven multi-turn". This replaces "both residual failures arose under multi-turn pressure".
- **Conclusion:** "correction removed 14 of 16 deceptive responses (88%; all nine single-turn, five of
  seven multi-turn), taking deception from 80% to 10%".
- **p = 0.020 at every confirmatory placebo comparison:** the abstract, §3.7, Limitations, the
  Conclusion, the integrity edition's closing reflection, and `AUTHOR_REFLECTION.md`.
- **§2.3 dose.**
  - The fixed-dose equation (`eq:alpha`, $\alpha = \mu_{\text{deceptive}} - \mu_{\text{honest}}$) was
    removed. Nothing else referenced it.
  - The hook now subtracts $\alpha^{(l)}_t\,\hat{d}^{(l)}$ (`eq:correction`, now Eq. 1).
  - The dose is described as set per flagged turn from that turn's measured projection, closing the gap to
    $\mu_{\text{honest}}^{(l)}$ (all of it at 100%, half at 50%). The exact rule is withheld with the
    calibration tools.
  - It is applied at every generation step of a flagged turn.
  - No clamp, no boundary values and no layer rule were added.
- **Pointers:** `paper.tex:1` now says "The md twins were retired 2026-10-09". The `REVIEW_INDEX.md`
  "paper.md — Full draft" line now points to both editions' `.tex`/`.pdf`.

**consequentiality-decomposition** (both `.tex`, both PDFs, `REVIEW_INDEX.md`; `paper.md` deleted)

- §5.3 now reads "per-layer profile normalization ... corrects 14 of 16 deceptive trials (88%; every
  single-turn trial and five of seven multi-turn), taking deception from 80% to 10% in an exploratory
  proof ...". The placebo p in the same sentence is now 0.020.
- **One TDC key.** `\cite{edrington2026targeted}` became `\cite{cc2026targeted}` (CC first; CC is lead
  author), and the `edrington2026targeted` bibitem was deleted.
  - The bibliography goes from 25 to 24 entries, counted by `\bibcite` in the `.aux`. There are no "??".
  - **Correction to the 2026-10-08 notes:** they gave this list as 24 entries. That figure was a
    pdftotext undercount; the true count before this round was 25.
- The `REVIEW_INDEX.md` "paper.md — Full draft. 26 references..." line now points to the `.tex`/`.pdf`.

**adversarial-audit-methodology** (both `.tex`, both PDFs, `REVIEW_INDEX.md`; `paper.md` and
`academic/paper.md` deleted)

- Both confirmatory placebo p-values (Round 6 "What survived", Conclusion) and the `REVIEW_INDEX.md:13`
  checklist item now read 0.020.
- The audit narrative's 80%→10% statements describe audit rounds and are unchanged.

**meta-pattern** (both `.tex`, both PDFs)

- **§3.2:** "correction removed 14 of 16 deceptive responses (88%; all nine in the single-turn condition)
  in an exploratory proof-of-concept [cc2026targeted]". The confirmatory caveat that follows stays.
- **Limitations:** "(14 of 16 deceptive trials corrected) ... (16/20 baseline, 2/20 corrected)".
- Both placebo p-values are now 0.020. There is no md twin.

**cache-tracing** (both `.tex`, both PDFs)

- ~315: "correction removed 14 of 16 deceptive responses (88%; all nine in the single-turn condition) in
  an exploratory proof ($N=20$)". The confirmatory miss and the scope caveat stay.
- ~471: "(Oracle Loop: 88% of deceptive trials corrected in an exploratory proof)".
- The paper prints no confirmatory p-value. There is no md twin.

## Every "0.019" in the corpus (`git grep`, tracked text files)

| File | Hit | Disposition |
|---|---|---|
| targeted-deception-correction/paper.tex | :93, :305, :338, :345, :362 | confirmatory placebo comparison → 0.020 |
| targeted-deception-correction/academic/paper.tex | :104, :310, :343, :350 | → 0.020 |
| targeted-deception-correction/AUTHOR_REFLECTION.md | :11 | → 0.020 (reader-facing companion; coordinator approved) |
| targeted-deception-correction/paper.md, academic/paper.md | 5 and 4 hits | deleted with the twins (step 5), not edited |
| consequentiality-decomposition/paper.tex, academic/paper.tex | :511 / :494 | → 0.020 |
| consequentiality-decomposition/paper.md | :208 | deleted with the twin |
| adversarial-audit-methodology/paper.tex, academic/paper.tex | :264, :388 / :258, :382 | → 0.020 |
| adversarial-audit-methodology/paper.md, academic/paper.md | 2 hits each | deleted with the twins |
| adversarial-audit-methodology/REVIEW_INDEX.md | :13 | → 0.020 (reader-facing checklist; coordinator approved) |
| meta-pattern/main.tex, academic/main.tex | :373, :666 / :372, :665 | → 0.020 |
| meta-pattern/PUBLISH_READINESS.md | :65 | internal doc, not corrected (Thomas) |
| SWEEP_pseudoreplication_2026-09-05.md | :136 | internal doc, not corrected |
| REMEDIATION_REGISTER.md | :706 | mnemosyne-benchmark "(+0/−0.019)", unrelated |
| notes/corrections_2026-10-08.md | :102, :190-191 | dated record of what was printed; left, and superseded here |
| identity-geometry/main.tex, academic/main.tex | :311 / :310 | ±0.019 (a standard deviation), unrelated |
| mnemosyne-benchmark (main, academic, paper.md) | 3 hits each | −0.019 score deltas, unrelated |
| mnemosyne-longmemeval-v10 (main, academic, paper.md ×2, docs ×2) | 1 hit each | McNemar p = 0.019 (0.0192), unrelated |
| tools/agni/designs/d136_design_v1_GATE.md | :178 | 0.0190494, arithmetic, unrelated |
| consequentiality-decomposition/supplementary/.../deception_directions_v2.json | many | raw direction-vector components, unrelated |

## Retired md twins and what only they had

Five files were deleted with `git rm`: TDC `paper.md` and `academic/paper.md`, CD `paper.md`, and AAM
`paper.md` and `academic/paper.md`. meta-pattern and cache-tracing have no md twin. Each deleted file was
checked against its `.tex`:
- every paragraph was matched by word trigrams, with the low-coverage ones read by hand;
- every number in each md is also in its `.tex`.

The md-only content, all recoverable from git history:

- **TDC md (both):**
  - a fixed-dose code block, withdrawn in step 3 anyway;
  - the integrity md's two gloss sentences in Direction Extraction (what a prefill pair is; the direction
    as a mean difference).
- **TDC `academic/paper.md`:**
  - a §3.8 "Detection Stack Disclosure". Its substance is in Limitations item 6. Only "at Layers 31-47" and
    "controlled evaluation is ongoing work" were unique, and the coordinator noted that losing the layer
    detail means less disclosure, not more.
  - "Correspondence: info@digitaldisconnections.com" for Edrington. `academic/paper.tex` has no such
    address (Data Availability gives cc@liberation-labs.org). Reported only: the placement of a personal or
    business address is Thomas's call.
- **CD `paper.md`:**
  - a parenthetical citing TDC's "~80% deception with chain-of-thought suppressed vs near-zero with
    reasoning active". TDC's own §2.6 states this.
  - the exact date ("May 28, 2026") of the AUROC-from-memory incident in the reflection. The `.tex` says
    "three weeks into this program".
- **AAM `paper.md` (integrity):** a byline listing Dwayne Wilkes and Kavi, with a "Sentient Futures"
  affiliation. Kavi's credit in the integrity edition was added in round 4b (see below), flagged for
  Thomas to veto.
- **AAM `academic/paper.md`:** an Acknowledgments line ("Dwayne Wilkes (Sentient Futures / Liberation
  Labs) for statistical auditing and red-team review, and Kavi for verification review"). The academic
  `.tex` credits both in its CRediT, and Kavi in its AI Disclosure.

**Pointers.**
- Updated: TDC and CD `REVIEW_INDEX.md`, and TDC `paper.tex:1`. `academic/paper.tex` names no md. AAM's
  `REVIEW_INDEX.md` had no pointer.
- Left as dated records: TDC `REVIEW_INDEX.md:29` (the 2026-10-08 entry) and the internal logs
  `PATCHES_APPLIED_2026-09-05.md`, `PATCHES_retracted_and_circular_2026-09-05.md` and
  `SWEEP_circular_statistics_2026-09-05.md`.

## Round 4b: Agni's fixes (MINOR_REVISIONS, every number verified)

Source: `correction_round4b_agni_fixes_20261009.md`. Applied to both editions unless stated.

- **TDC.**
  - **Abstract (MINOR 2).**
    - The 80% baseline is described as arising "under the specific EvalMax ... single-turn roleplay frame
      or multi-turn consensus pressure"; it used to say "roleplay frame combined with multi-turn
      consensus pressure".
    - This was verified in oracle-harness before the change. `build_multi_turn`
      (`experiments/red_team_suite/red_team_suite.py`, `e930d1d`) builds the honest task as a user turn
      plus three escalating user follow-ups, with no system message. `run_trial` in
      `behavioral_proof_abliterated.py` and `placebo_steering_test.py` appends only user and assistant
      turns, and `build_prompt_ids` only applies the chat template. Only `build_roleplay` carries the
      EvalMax system prompt.
    - The rate clause now reads "correcting 14 of the 16 trials that deceived at baseline (all nine
      single-turn roleplay trials and five of the seven multi-turn ones)".
  - **§2.2, concept-first gloss (polish).** "The deception direction at each layer is the difference
    between the mean activation vectors under deceptive and honest conditions." It is the first sentence of
    the subsection, before "Native contrastive extraction: ...".
  - **§2.3 (MINOR 1, polish).**
    - The dose is now described as set "once per turn the detector flags (section 2.7), from that turn's
      measured projection, and is sized to close the gap between that projection and the honest-condition
      mean ...". The text adds: "The hook acts across the full sequence of a flagged turn, prompt and
      generated tokens alike."
    - "where" now runs on from Eq. 1 without a paragraph indent.
  - **§3.1 (MINOR 5).** New sentence: "Arms share scenarios and sampling seeds, and both residual failures
    also deceived at baseline, so correction converted 14 of the 16 baseline-deceptive trials and introduced
    no new deception."
  - **Dose response (polish).** "Unit-scale nudges (coefficient $\leq 0.8$)" replaces "($\alpha \leq 0.8$)",
    which clashed with the dose symbol.
  - **Conclusion (MAJOR 3, MINOR 3).** "14 of the 16 deceptive trials". The confirmatory clause now states
    that the primary endpoint was not met ($p=0.34$), though corrected still outperformed matched-dose
    placebo ($p=0.020$ one-tailed; $p=0.039$ two-sided).
- **CD (MINOR 4, polish).**
  - "(88%; all nine single-turn and five of seven multi-turn)" and "($N=20$; one single-turn roleplay frame
    and one multi-turn escalation script)".
  - The bibitem's "(2026b)" became "(2026)": there is no 2026a. natbib runs in `numbers` mode with plain
    `\bibitem{key}`, so no label depends on it.
- **meta-pattern §3.2 (MAJOR 1).** "correction removed 14 of the 16 deceptive trials (88%: all nine
  single-turn trials and five of the seven multi-turn ones), taking deception from 80% to 10%, in an
  exploratory proof-of-concept [cc2026targeted]". The next sentence's "(30% vs. 80%)" and "(13%
  post-correction)" now have comparators again.
- **cache-tracing (MAJOR 2).** ~316: "removed 14 of the 16 deceptive trials (88%: all nine single-turn
  trials and five of the seven multi-turn ones)". ~471: "(Oracle Loop: 88% of deceptive trials corrected in
  an exploratory proof; its pre-registered replication did not meet its primary endpoint)".
- **AAM, Kavi's credit (MAJOR 4), in its own commit. Done, flagged for Thomas to veto.**
  - The integrity `paper.tex` Acknowledgments now read "Dwayne Wilkes provided statistical consulting, and
    Kavi provided verification review."
  - Rationale: AGENTS.md §3 credits advisory reviewers in Acknowledgments, and style-guide Axis 2 says
    credit must not diverge between editions. The academic edition already credits Kavi (AI Disclosure,
    CRediT).
  - CC decided this; Thomas may veto. The academic `.tex` was not changed.
- **Build and checks.** The same build was run. Pages, overfull boxes and bibliography counts are unchanged
  from round 4: CD 24, meta-pattern 11, cache-tracing 8. The round-4b needles were checked against the PDFs
  committed at the branch HEAD before round 4b: each new string fails there, and each withdrawn string is
  found there.

## Not done. Read before closing anything.

- **For Thomas (Agni MINOR 7): the TDC academic edition prints no human correspondence address.** The
  retired `academic/paper.md` had info@digitaldisconnections.com, which AAM's academic edition uses.
- **Deferred (Agni MINOR 6), "byte-verified matched doses".** On 9 of the 20 multi-turn placebo trials, the
  later-turn doses diverge from the native arm's, so "matched" holds for the first turn but not every turn.
  Agni's replacement text is in its round-4 review. To be applied in a later round.
- **Deferred:** the like-for-like confirmatory conversion (5 of 9 baseline-deceptive trials corrected, 56%)
  is not printed. Optional, for a later round.
- **Zenodo:** no new versions. All five papers need one with the rebuilt PDFs.
- **Internal docs** (SWEEP_*, PUBLISH_READINESS, REMEDIATION_REGISTER, PATCHES_*) were not corrected, per
  Thomas.
- **Before merging,** CC runs Agni plus the style guide (spec §6).

## Build and verification

- **Build:** pdflatex (TeX Live on MTH), until labels settled (three passes; four for cache-tracing), with
  bibtex for meta-pattern and cache-tracing. The CD `\bibliography{}` names no existing `.bib`, so CD's
  inline list was used.
- **Logs:** all 10 logs show 0 errors, 0 undefined references or control sequences, and 0 missing
  characters. Pages and overfull boxes match master; the only overfull over 20 pt is meta-pattern's
  pre-existing 21.7 pt.
- **Checks:** run with `tools/verify_pdf.py` against master's PDFs.
  - Every new string is present here and absent on master.
  - Every withdrawn string ("0.019", "closes the full measured gap", "Edrington, T. and CC", "80%
    (16/20) to 10% (2/20)", ...) is found on master and absent here.
  - Retained strings ("(16/20 baseline, 2/20 corrected)", "80% to 10% with AUROC 1.0", "Scope alone is not
    sufficient") pass on both.

## 2026-10-10: decisions on the two deferred items (Thomas)
- **Agni MINOR 7: done.** The TDC academic edition's author footnote now reads "Correspondence:
  thomas@liberationlabs.tech", following the AAM academic convention. CC's address stays in Data Availability.
  - Rebuilt on MTH: 13 pages, clean.
  - verify_pdf: the address is present in the new PDF and FAILS on master's PDF (negative control).
- **Kavi's acknowledgment (bd28a6f): kept.** It stays in Acknowledgments while we wait for Kavi's blessing on
  anything more prominent.
