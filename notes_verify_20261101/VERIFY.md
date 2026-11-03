# Independent verification of PR #19 (lyra/followup-withdrawn-claim-contacts)

Verified head: 06f936b (local and origin agree). Base: origin/master. Date: 2026-11-01.
Read and run only; nothing edited except this file.

## Verdict: FINDINGS (3 sites still carry the withdrawn reading; 1 minor table inconsistency)

## 1. Withdrawn-claim grep

Pattern file (7 ERE lines): `directional projection`, `model state, not text`,
`W_K\$?[^.]{0,60}(classif|emotion|valence|12\.3|0\.992|compass)`, `(12\.3|0\.992)[^.]{0,80}W_K`,
`injection[^.]{0,80}(confirm|model state|architectural|functional)`,
`direction[^.]{0,60}not[^.]{0,30}(spectral|shape)`, `user'?s? (emotional |internal )?state`.

- Positive controls: a planted two-line copy of the old wording hits 2 of 2. On origin/master the same
  grep hits the old sites (Lyra II 22 per edition, meta-pattern 2 per edition, RESEARCH_STATUS 3).
- Run with `git grep HEAD -- . ':!*pain*' ':!*.pdf'`; the pain-axis paths were excluded and not opened.

Sites on the branch that still state the withdrawn reading:

| Site | Text | Covered by another PR? |
|---|---|---|
| `lyra-technique-ii/main.tex:1338` and `academic/main.tex:1338` (Conclusion) | "Genuine nulls receive mechanistic explanations (emotions are direction, not shape; ...). Apparent nulls ... emotion classification---are rescued by the appropriate [instrument]" | No. The Correction Note says the Conclusion was corrected, but this paragraph was not changed and carries no mark. |
| `cache-tracing/main.tex:418`, `cache-tracing/academic/main.tex:417` | "The emotion encoder reads $W_K$ valence projections (input-evaluation structure)." | No open PR touches cache-tracing. |
| `README.md:20` (Lyra II row) | "SVD denoising and directional projection extend KV-cache geometry to emotion and persona" | #17 changes only the user-model row. #3, #8, #9 and #14 also touch README.md; I did not check whether any of them changes this row. |

Covered elsewhere, so not counted against #19: emotion-accumulation-paper and decision-state-paper (#18 removes
the lines); empathy-bus/main.tex:100 (#3 removes the 12.3x / W_K line); the convergence paper (already merged).
`meta-pattern/SOP_REVIEW.md:34`, PATCHES_* and REMEDIATION_REGISTER quote the old wording as historical
records, which is correct. The Lyra II title and the line-15 header comment still say "directional
projection"; both were flagged as left alone.

The other Lyra II hits sit in the sections the notice says are kept as first published and marked (Method,
Injection, Results; each now opens with an italic correction mark), or in the correction text itself.

## 2. Numbers against R1R2_RESULTS.md (lines 28, 35-40)

All match: 30-class 0.410 vs 0.370, Holm 0.071, n.s. (S1); the described method 0.358 vs 0.382 (line 28);
binary 0.992 vs 0.963, Holm 0.006 (S2); margin +0.029 = 0.992 - 0.963. The FWL wording matches line 38 ("fitted
label-free on each fold's own test trials"). n = 300 and the arousal confound are not on those lines; they
come from the task brief.

Minor (Lyra II, both editions): the body-count footnote says "for these rows the $p$ column is against
text", but the Injection row keeps $p < 0.005$, which is not a test against text. The emotion table still
lists the TF-IDF baseline as 0.882 (relabelled "Injection script") beside 0.992, while the Results text now
gives 0.963.

## 3. Contacts, credit, editions, PDFs, LF

- Contacts: master's pattern (87545fd sources, then e667f41 PDFs rebuilt) is followed. The only address
  change is `community/experiment_guide.md`, lyra@thcoalition.net -> lyra@liberationlabs.tech; it is
  Markdown, so there is no PDF to rebuild. The convergence byline still has three @thcoalition.net
  addresses, listed for Thomas as stated.
- Credit: no added or removed line touches \author, \thanks, a byline, CRediT, Edrington or Wilkes.
- Editions: the changed lines are byte-identical between main.tex and academic/main.tex for Lyra II and
  for meta-pattern, and between the two references.bib files.
- PDFs, checked both ways with tools/verify_pdf.py: the 4 new PDFs exit 0 and the 4 master PDFs exit 1.
  For Lyra II: present "Correction Note", absent "exceeds the text baseline". For meta-pattern: present
  "TF-IDF", absent "surviving emotion result". My first absent string for Lyra II, "Near-Perfect
  Accuracy", failed on the new PDF because the Correction Note quotes the old heading on purpose, so I
  replaced it.
- LF: all 9 changed text files are i/lf w/lf.
- Commit trailers: all 4 commits end with the required two lines. PR #19 is a draft, open, and its body
  ends with the required lines.

## 4. Private-info scan of added lines

Patterns: emails, private IPv4 addresses, home paths and credentials paths, secret/token words, phone
numbers, and personal or host names (Allison, Vera, tailscale, ssh, madame, margaret, starship).
Planted control (`admin@<MTH host> ... <home>/<credential file>`): caught.
Added lines: one hit, `lyra@liberationlabs.tech`, which is the intended public contact address. Clean.
