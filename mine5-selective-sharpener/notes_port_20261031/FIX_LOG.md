# Fix log: verifier findings on the mine5 port (PR #16)

Written 2026-10-31 on branch `lyra/mine5-port`, on top of `500fc9a`. Two verifiers read the port. Their reports
(`VERIFY_1_numbers.md`, `VERIFY_2_fidelity.md`) sit beside this file, untracked. Old line numbers below are the
verifiers' (pre-fix `main.tex`); the academic edition's lines are one lower.

## Verifier 1 (numbers and verdicts)

| # | Finding | Outcome |
|---|---|---|
| 1 | Registered H5 neither reported nor withdrawn | **Applied.** Correction paragraph, after "H3 is falsified" (both editions): H5 is listed, was not run, and is untested, because `code/mine5_three_model.py` extracts encoding-phase features only; its one `model.generate` call (`:218`) is the H2 abliteration check. |
| 2 | H1's registered FWL-corrected p never printed | **Applied.** H1 paragraph now gives d = 0.757, label-shuffle p = 0.0041 (token-count slope removed within each model, mean kept). Both values moved from report-only to checked items in `code/recompute_registered.py`. |
| 3 | H1's registered "uniformly ... ~25-35%" | **Applied.** One sentence under H1: the d criterion is met, but compression is not uniform (factual 42%, confabulation 26%). |
| 4 | Checker tests presence, not every printed copy | **Applied (reword).** Data and Code now says the script "checks that each recomputed value is printed in both editions of this paper (presence, not every printed copy)"; the docstring says a wrong second copy is NOT caught. The script was not changed to count occurrences. |
| 5 | "Compression is deeper than safety directions" presumes abliteration removed them | **Applied, to the class.** Rewritten at old :427-432, :52-53, :493-495, and also at :256-258 and :349-353 (same presumption). "safety directions removed" at :43 and :114 is now "targeted for removal". |
| 6 | "spectral compression" for an entropy result (:258) | **Applied** inside the :256-258 rewrite ("entropy compression"). |
| 7 | Logit measure attributed to the KV-cache (:346-347) | **Applied.** Now "measured in the output logits; its KV-cache counterpart is the stable-rank drop of H4 (d = 1.04)". |
| 8 | Exploratory pattern stated as a finding (:410, :439) | **Applied** at both. At :439 the wording is "The selective sharpening, an exploratory pattern, may be specific to this architecture" (a first wording added a fourth overfull box, so it was reflowed). |
| 9 | Convergence paper :378-381 contradicts mine5 | **Rejected for this PR:** the brief limits edits to this paper's folder. Follow-up for PR #13. |
| note | Kavi given a pronoun (:522, academic :527) | **Applied.** "We thank Kavi for verification review of this work." Kavi's pronouns are unknown. |

## Verifier 2 (port fidelity)

| # | Finding | Outcome |
|---|---|---|
| 1 | Academic edition's 2026-08-20 correction history misdated | **Applied, narrowed.** `academic/main.tex` Correction paragraph: "Revisions of 28--29 July 2026 (in this edition, 20 August 2026 for the H2 wording)". Scoped to the H2 wording because `git show d21ac5a` puts the "TOST Inconclusive" heading and the footnote dated 2026-08-20 into the academic edition; when the selectivity-ratio interval reached that edition was not traced. |
| 2 | `SOP_REVIEW.md` still certifies the withdrawn analysis | **Applied.** Two-line "Superseded 2026-10-20" banner at the top. |
| 3 | Three overfull boxes per edition | **Rejected:** pre-existing text, nothing hidden. The rebuild shows the same three (main.tex lines 112-115, 131-136, 394-402). |
| 4 | bibtex "empty journal in gurnee2026gwt" | **Rejected for this PR:** `references.bib` is unchanged; fix in a separate change. |
| 5 | Port notes ship in the public paper folder | **Deferred to Lyra.** This file joins them; the VERIFY reports stay untracked. Keep or drop the folder before the PR leaves draft. |
| 6 | Follow-ups named in the PR body | **No change;** still tracked in the PR body. |

## Consequences and checks

- Neither `.tex` now matches PR #4's head `54c7f9d1` byte for byte. The PR #4 content is still all there; these
  edits add to and rescope it.
- Recompute (`notes_port_20261031/recompute_output.txt`, run from an empty folder): real checks 57, failures 0,
  planted false value caught, data hashes OK, exit 0. Before: 55.
- The two new checks were made to fail once: with `0.0041` changed to `0.0014` in a scratch copy of `main.tex`,
  the run gave failures 1, exit 1.
- PDFs rebuilt (MiKTeX pdflatex, bibtex, pdflatex twice): 8 pages each, 0 errors, 0 LaTeX warnings. Text pulled
  from the PDFs, whitespace removed, shows 0.684, 0.055, 0.746, 2.51 and 1.97 the same number of times as before
  the fixes (5, 6, 4, 5, 5), and 0.757 and 0.0041 once each, in both editions.
- No CR bytes in the files this change writes; `git diff --check` is clean.
