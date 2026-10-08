# Mechanical corrections, 2026-09-30 (branch `corrections-2026-09-30`)

This note records the open register items that were **unassigned** and **mechanical**, where the
correct value is settled by a primary (a data file, a verification report, an external record, or
the paper's own table). Each fix was made by one agent and then re-derived from the primary by a
second, independent agent. Where the primary did not support the proposed fix, the item is marked
REFUTED. Where a fix needs a judgement call, it is marked NEEDS_DECISION and was **not** edited.

**Not done in this commit. Read this before closing anything.**

- **`REMEDIATION_REGISTER.md` was NOT edited.** Its owner has to reconcile the rows listed below.
  The IDs are `<paper> #<n>` and refer to that paper's `tools/agni/style/<paper>.agni_style.json`
  finding. For LT-II and emotional-trajectory they are survey IDs.
- **The tracked PDFs were NOT rebuilt.** AGENTS.md §4 applies: until they are, every fix below
  still ships in the PDF a reader downloads. That covers 17 files: `main.pdf` and
  `academic/main.pdf` for lyra-technique-ii, formulary-paper, emotional-trajectory-paper,
  identity-geometry, spectral-shape-paper, null-swarm-paper, emotion-accumulation-paper and
  graph-topology-paper, plus `paper.pdf` and `academic/paper.pdf` for
  adversarial-audit-methodology. Two things to know when rebuilding:
  - Five of these papers (lyra-technique-ii, spectral-shape-paper, adversarial-audit-methodology,
    emotion-accumulation-paper, graph-topology-paper) had source changes in `c5be832`
    (self-citation DOIs) after their last PDF commit, `c9bf68f`/`7375c67`. A rebuild will ship
    those changes too, so name both in the commit (AGENTS.md §3).
  - `emotion-accumulation-paper/academic/main.txt` is a tracked pdftotext extract of the shipped
    PDF. It still reads "available at Liberation-Labs-THCoalition" without the repository name.
    Regenerate it or drop it when the PDF is rebuilt.
- **Zenodo deposits need a new version for every touched paper:** lyra-technique-ii,
  formulary-paper, emotional-trajectory-paper, identity-geometry, spectral-shape-paper,
  null-swarm-paper, adversarial-audit-methodology, emotion-accumulation-paper,
  graph-topology-paper. The new version should be uploaded after the PDFs are rebuilt.

Build check: both editions of all nine papers were compiled with
`latexmk -g -pdf -halt-on-error` in a scratch copy outside the repo, once from `HEAD` (`c5be832`)
and once from the working tree. All 18 builds exited 0 with zero `!` errors, and each edited build
has the same page count as its HEAD build. There are two differences, and both are intended:
graph-topology `main` went from 7 to 6 pages because a duplicate heading was removed, and
emotion-accumulation lost its 158pt overfull box. The warning sets are otherwise identical. In the
rebuilt PDFs, every new string below is present and every old string is absent, and the reverse
holds in the HEAD builds, so the check is able to fail.

## 1. All items

Status counts: **FIXED 15 · NEEDS_DECISION 3 · REFUTED 2 · OUTSIDE_REPO 3** (23 items).

| Register id | Paper | Status | File:line (flight / academic) | Before -> after | Primary evidence | Verifier |
|---|---|---|---|---|---|---|
| LT-II #25 | lyra-technique-ii | FIXED | `main.tex:556` / `academic/main.tex:554` | `$0.42$--$0.46$ accuracy` -> `$0.41$--$0.46$ accuracy` | `data/persona_intensity_results.json` five_level_acc: Qwen2.5-7B 0.412, Llama-3.1-8B 0.456, Mistral-7B 0.452. The 2.1--2.3x claim still holds (0.412/0.2 = 2.06, 0.456/0.2 = 2.28). | CORRECT |
| LT-II #26 | lyra-technique-ii | FIXED | `main.tex:466,608,609` / `academic:464,606,607` | `0.410`->`0.409`; `accuracy $0.056$`->`$0.057$`; `accuracy $0.427$`->`$0.43$` | `data/WK_BACKTRACK_V2_RESULTS.json` probes[0] accuracy 0.40889. `data/emotion_denoising_results.json` raw_52D accuracy_30class 0.057 and denoised rank_10 accuracy 0.43. The file stores 0.43 to two decimals only. | CORRECT |
| LT-II #27 | lyra-technique-ii | FIXED | `main.tex:693` / `academic:691` | `${\sim}0.062$` -> `$0.060$--$0.091$` | `data/selection_bias_results.json` se over 45 rows: min 0.0599, max 0.0906. The same values come out of null_std in all 16 sweep files. 0.062 is only the fallback default in `code/selection_bias_analysis.py:64`, and no row uses it. | CORRECT |
| LT-II #22 | lyra-technique-ii | FIXED | `main.tex:827` / `academic:825` | `$+0.006$--$0.068$` -> `$+0.000$--$0.068$` | Sweep, task "confab vs grounded", 15 models: 8 have delta 0.0000, max 0.0681. The row's raw and denoised ranges also span all 15. 0.006 was Gemma-2-9B's delta. | CORRECT |
| LT-II #21 | lyra-technique-ii | FIXED | `main.tex:785` / `academic:783` | `Qwen-32B` -> `Qwen2.5-32B (q4)` | `data/canonical_scale_sweep_Qwen2.5-32B-q4_results.json` self-ref raw 0.8821. It is the only 32B sweep file. The label now matches Table 3. | CORRECT |
| formulary #13 | formulary-paper | FIXED | `main.tex:63-64` / `academic:59-60` | `zero adverse events above $\alpha = 0.5$` -> `zero adverse events at $0.5 \leq \alpha \leq 1.0$` | `data/dose_response.json` n_adverse: alpha 0.5 = 0, 1.0 = 0, 1.5 = 25, 2.0 = 29 | CORRECT |
| formulary #14 | formulary-paper | FIXED | `main.tex:470` / `academic:466` | `2--43\%` -> `1.6--43.9\%` | adverse_hedged/n_hedged over 24 cells of `data/formulary_{distilled,base}_summary.json`: min 2/124 = 1.6%, max 36/82 = 43.9%. This matches Table tab:confab. | CORRECT |
| formulary #15 | formulary-paper | NEEDS_DECISION | `main.tex:477` / `academic:473` | not edited | No primary exists for the "15%" lower bound. See §2. | CORRECT (decision) |
| delta-manifold #16 | delta-manifold-paper | REFUTED | `main.tex:595` / `academic:593` | not edited: "+31%" is right | `data/honesty_signal.json`: delta g 2.3218 / endpoint g 1.7722 = 1.310. The survey's +30% came from dividing rounded values. | CORRECT |
| delta-manifold #19 | delta-manifold-paper | NEEDS_DECISION | `main.tex:183` / `academic:181` | not edited | condition_number is absent from all stored data but is computed by `spectral-shape-paper/code/lyra_features.py:112`. See §2. | CORRECT (decision) |
| emotional-trajectory #11 | emotional-trajectory-paper | FIXED | `main.tex:75,802` / `academic:73,804` | `33--58\% depth` -> `42--58\% depth` | `data/permutation_results_v2.json` eccentricity: lowest L14 0.0726 and L10 0.0971. Depth is layer/24, so 41.7% to 58.3%. 33% is L8, which ranks 11th. | CORRECT |
| emotional-trajectory #14 | emotional-trajectory-paper | FIXED | `main.tex:524` / `academic:526` | `From L9 onward, arousal overtakes` -> `From L9 onward (except L22), arousal overtakes` | Same file: arousal_d > valence_d at every layer L9-L23 except L22 (valence 0.8342 > arousal 0.8309). **Scope:** the plateau bullet at `main.tex:453-454` / `academic:455-456` ("from L9 onward", under "L8--L20") is true within L9-L20 and was left alone on purpose. Closing #14 means accepting that reading. | CORRECT |
| identity-geometry #4 | identity-geometry | FIXED | `main.tex:521` / `academic:520` | `$0.56$--$0.83$` -> `$0.554$--$0.733$` | `data/fingerprint_clean.json` within-condition presence, bare/persona/constructed at L35 and L47: 0.5537-0.7331, which matches the paper's :255. The old range came from the superseded `fingerprint_original.json`. | CORRECT |
| spectral-shape #17 | spectral-shape-paper | FIXED | `main.tex:371-372` / `academic:370-371` | `above the permutation null (95th percentile: 0.570)` -> `above their permutation nulls (95th percentiles: 0.570 and 0.566, respectively)` | `verification/verification_report.md:13,17`: MP null95 0.566, shape null95 0.570. The verifier reproduced both by re-running `compute_paper_stats.py` with seed 42. | CORRECT |
| null-swarm #7 | null-swarm-paper | FIXED (revised) | `main.tex:291` / `academic:290` | `(\emph{below} random)` -> `(within the $\pm$ spread of random)` | The paper's own numbers: random baseline 0.085 ± 0.065, so the band is [0.020, 0.150], and 0.032 lies inside it. **The first edit read "within 1~SD of random". The verifier marked it WRONG** because no source defines ±0.065 as an SD (`audit-2026-07-15/.../null-swarm/data-analyst.md:58`). It was revised to wording that holds under any reading of ±, which is also the register row's own title. `\emph` was dropped per AGENTS.md cause 5. | WRONG -> revised; re-checked in PDF |
| adversarial-audit #5 | adversarial-audit-methodology | FIXED | `paper.tex:449` / `academic/paper.tex:454` | `Casper, S., Lin, J., Kwon, J., Culp, G., and Hadfield-Menell, D.` -> the 21 arXiv authors (Casper, Ezell, Siegmann, ... Krueger, Hadfield-Menell) | arXiv 2401.14446 API record: 21 authors, DOI 10.1145/3630106.3659037, FAccT '24. It matches `paper.md:198` and `academic/paper.md:213`. | CORRECT |
| contextual-engagement #12 | human-review/contextual-engagement-paper | OUTSIDE_REPO | `main.tex:142` | proposed only | `KV-Experiments/code/concordance/experiment.py:53` "mistralai/Mistral-7B-Instruct-v0.3". All 241 phase_a JSONs agree. See §3. | CORRECT |
| temporal-boundary #14 | human-review/temporal-boundary | OUTSIDE_REPO | `references.bib:58` | proposed only | The transformer-circuits.pub "Scaling Monosemanticity" byline gives Trenton Bricken. Tristan Hume is a different author. See §3. | CORRECT |
| convergence #14 | human-review/convergence-paper | REFUTED (defect open) | `circumplex_subsection.tex:13-15` | not edited | The survey read 0.987 as a cosine. It is `arousal_d.real` at L20 in `emotional-trajectory-paper/data/permutation_results_v2.json`, a d-type direction strength from the 24-layer validation model, so relabelling it `\cos` would make it more wrong. See §2. | CORRECT |
| gwt-response #9 | human-review/gwt-response | OUTSIDE_REPO | `main.tex:839` | proposed only | `:619-626` Experiment 2 maps the champion features; `:627` Experiment 3 does not. See §3. | CORRECT |
| emotion-accumulation #4 | emotion-accumulation-paper | FIXED | `main.tex:582` / `academic:580` | `\texttt{Liberation-Labs-THCoalition/Project-Oracle}` -> `\texttt{Liberation-\allowbreak Labs-\allowbreak THCoalition/\allowbreak Project-Oracle}` | The build logs of the shipped PDFs show `Overfull \hbox (158.56pt too wide)`, and "/Project-Oracle" was clipped off the page (pdftotext finds 0 "Project-Oracle"). After the edit there is no overfull and the rebuilt PDF shows "Labs-THCoalition/Project-Oracle". | CORRECT |
| deception-nulls #7 | deception-detection-nulls | NEEDS_DECISION | `paper.tex:148,348` / `academic:148,356` | not edited | `\bibitem{cc2026audit}` is never cited. The sentence that would cite it is the subject of the CRITICAL agni #1. See §2. | INCOMPLETE (evidence corrected; status right) |
| graph-topology #142 | graph-topology-paper | FIXED | `main.tex:187` (added), old `:199-201` (removed) | Removed the duplicate `\section*{Acknowledgments}` that the 2026-09-03 sweep inserted before `\bibliography`. The Dwayne Wilkes line moved **verbatim** into the existing `\subsection*{Acknowledgments}` at `:183`. | Diff against `main.tex.bak-byline-20260903` shows the sweep inserted exactly those three lines. The rebuilt PDF has one "Acknowledgments" (HEAD build: two) and all four credits. No credit was added or removed. `academic/main.tex` has no duplicate and was not touched. | CORRECT |

## 2. NEEDS_DECISION, and REFUTED items with an open defect

- **formulary #15.** `main.tex:477` / `academic:473` read "hostile has 15--45\% adverse rate on
  overconfidence." No artifact on this machine sources 15%. The only overconfidence figure is
  :301, 45.5% adverse (base model, n=55). The distilled 350-trial overconfidence checkpoint is on
  Starship at `/Users/margaret/oracle-experiments/results/formulary_350/overconfidence/checkpoint.json`
  (see `Project-Oracle/experiments/agni_formulary350_full.py:21,40`). Options:
  (A) "hostile has 45.5\% adverse rate on overconfidence (base model)."
  (B) Pull the distilled rate from that checkpoint, commit the artifact, and keep a range only if
  both ends are sourced.
  (C) Delete the clause.
  The same string is in the STALE `formulary-paper/academic_main.tex:352` and in the superseded
  `lyra-s-research-/formulary-paper/`.
- **delta-manifold #19.** "Features include ... condition number" (`main.tex:183` /
  `academic:181`). No stored or analysed dataset has condition_number, but the named extractor
  computes it. `Llama_persona_full.json` also has no stable_rank, participation_ratio or
  sv_kurtosis. Options:
  (A) Delete "condition number, ". The list would then still not describe the Llama dataset.
  (B) Separate what the extractor computes from what was analysed, and name the six endpoint
  features in `verification/compute_delta_paper_stats.py:32-33`.
  (C) Leave it as a true statement about `lyra_features.py`.
- **deception-nulls #7**, which depends on agni #1 (CRITICAL). The Note at `:148` (both editions)
  says the audit-methodology registry "reports this value as 0.497; that is an error". The
  published companion paper reads 0.238 (`adversarial-audit-methodology/paper.tex:300`), so the
  present tense is false. **Correction to the first survey:** 0.497 *was* in the companion's
  registry. It is in human-review `85d2461` (2026-07-10), `paper.md:136`, and was corrected to 0.238
  in `5d36069` (2026-07-17). So the `.md` wording "originally reported ... since corrected" is
  historically accurate, but only for a pre-publication draft. The DOI version never carried 0.497.
  Options:
  (A) Add `\cite{cc2026audit}` to the current sentence. Not recommended: it makes a false
  present-tense claim look sourced.
  (B) Rewrite or delete the Note under #1 first, then cite wherever the audit paper is still named.
  (C) If the Note goes, cite at `:274` or `:314`, or delete the orphan bibitem.
  (D) Replace the `.tex` Note with the `.md` wording plus `\cite{cc2026audit}`. This closes #1 and
  #7 together, but "originally reported" points at a draft the reader cannot see.
- **convergence #14** is REFUTED as proposed, but the defect is open. In
  `human-review/convergence-paper/circumplex_subsection.tex:13-15`, "peaking at near-identity
  ($d = 0.987$) at L20" presents a direction-strength value from the 24-layer, 896-dim Qwen2
  validation model as the peak of a Qwen3.5-27B cosine. Options:
  (a) Restore the pre-`26ae286` wording "...(mid-to-late layers), and the valence direction at
  $\cos = 0.70$--$0.80$", which matches `empathy-bus/main.tex:161`.
  (b) Move the number to its own sentence, correctly labelled and attributed.
  (c) Find a real per-layer Qwen3.5-27B cosine series; none exists locally.
  Copies are in `lyra-s-research-/convergence-paper/`, `longview-final/` and
  `longview-submission-papers/convergence-paper/`. The permutation sentence in the same paragraph
  ("23 of 24 layers", "10x ... 59x") is also from the 24-layer model and needs its own register row.
- **delta-manifold #16** is REFUTED, with a residual. The printed endpoint g = 1.78 re-derives to
  1.772 from raw data. 1.78 equals J x the README's rounded d = 1.80. Changing 1.78 to 1.77
  (`main.tex:40,51,108,358,368,596,762` and the academic twins) is a separate decision.

## 3. OUTSIDE_REPO: proposed edits, not applied (these files are outside published-research)

- **contextual-engagement #12**
  `Research/human-review/contextual-engagement-paper/main.tex:142`:
  `Mistral-7B-v0.3-Instruct ($N = 230$)` -> `Mistral-7B-Instruct-v0.3 ($N = 230$)`.
  The same one-token swap is needed in `lyra-s-research-/prospectuses/contextual-engagement.tex:70`.
  Rebuild `main.pdf` afterwards. The abstract's family shorthand at `:66` is fine.
- **temporal-boundary #14**
  `Research/human-review/temporal-boundary/references.bib:58`:
  `Bricken, Tristan and` -> `Bricken, Trenton and`.
  Make the same change in `lyra-s-research-/temporal-boundary-paper/references.bib:58`. Then
  regenerate `main.bbl` with bibtex and rebuild `main.pdf`; do not hand-edit the bbl.
- **gwt-response #9**
  `Research/human-review/gwt-response/main.tex:839`:
  `contain our champion features (Experiment~3)` -> `(Experiment~2)`.
  The table cell at `:285` ("Experiment~3") is correct and should stay. Note that `gwt-response/`
  is **untracked** in the human-review repo, so an edit there is not version-controlled unless the
  directory is added.

## 4. Found in passing, not in any batch, not edited

- `formulary-paper/main.tex:473` / `academic:469` say "Hostile on the base model shows zero adverse
  events at $\alpha \leq 1.0$". That is false: `dose_response.json` has alpha = 0.25 with 1/33
  adverse, which is Table :329 "3.0\% (1/33)". No register row was found. Proposed fix:
  `$\alpha \leq 1.0$` -> `$0.5 \leq \alpha \leq 1.0$`.
- `formulary-paper/academic_main.tex:349` (STALE, git-history only) and the `lyra-s-research-`
  copies still carry "2--43\%". Register #18 proposes removing the stale file.
- `spectral-shape-paper/verification/compute_paper_stats.py` sets `DATA_DIR` to `verification/`,
  so it fails as committed. A re-run gives MP p = 0.0040 against 0.003 in Table 1; both are
  significant.
- `lyra-technique-ii/code/selection_bias_analysis.py:17` docstring says "null_std ~ 0.062", which
  is the fallback, not the data.
- The null-swarm "±0.065" has no source (`audit-2026-07-15` opus-vetting.md:41,80), and upstream
  `ghost-dimensions` reports a single-sample random baseline of 0.072. Re-aligning Case 2.3 with
  upstream is a separate decision.
- adversarial-audit-methodology: all 8 bibitems are uncited (register #6).
  `REVIEW_INDEX.md:19` asks "FAccT confirmed?"; the arXiv record confirms it.
- Scratch builds of the corrected PDFs (not committed) are in the session scratchpad at
  `final_build/wt/`.
