# Mnemosyne v10 on LongMemEval_S-cleaned — Review Index

**Status**: Released in this repository 2026-09-25. Zenodo deposit pending (Thomas, through the website).
**Date staged**: 2026-09-25
**Authors**: Nexus (lead), Thomas Edrington. Academic version: Thomas Edrington, with Nexus acknowledged.
**Working copy and review thread**: `Liberation-Labs-THCoalition/human-review`, PR #1
(branch `mnemosyne-longmemeval-v10`).

## Second reads

Every number in the paper was recomputed from the raw files by a reader who did not build the pipeline. The reports
are in `reviews/`.

| What was read | Reader | Date | Verdict | Report |
|---|---|---|---|---|
| v9 baseline run (the system v10 replaces) | Fable 5.1 | 2026-09-23/24 | PASS WITH CAVEATS | `reviews/REVIEW_v9_baseline.md` |
| v10 evidence recall, dev and held-out | Opus 5.5 | 2026-09-24 | CONFIRMED WITH CAVEATS | `reviews/REVIEW_v10_recall.md` |
| v10 held-out reader accuracy, judge calibration, placement | Fable 5.1 | 2026-09-25 | WARN — CONFIRMED WITH CAVEATS | `reviews/REVIEW_v10_held_readers.md` |
| The paper's text, every number and claim | Fable 5.1 | 2026-09-25 | WARN — CONFIRMED WITH CAVEATS; all nine required fixes applied | `reviews/REVIEW_paper_fable.md` |

## Checklist

### Content
- [x] Every number reproduced from the raw files by a second reader (all four reports above)
- [x] The two method errors the paper's reader found are fixed: the prompt template (`none`, not `merge`) and
      the embedding device (GPU, not CPU)
- [x] No LoCoMo content (Thomas, 2026-09-25: the LoCoMo rerun will be reported separately)
- [x] Thomas's read of the facts, 2026-09-25: "All the facts look straight to me."

### Release
- [x] Code as run, byte-identical, with `SHA256SUMS`; frozen configuration and question split
- [x] Every answer and judgment behind the tables, all six arms, both judges
- [x] Scrubbed: one answer quoted the account e-mail (redacted, see `README.md`); no keys, passwords or tokens
- [x] Left out and why: model conversation logs (they carry harness-injected account details), prompts and
      embeddings (rebuildable from the code and the public dataset)

### Quality
- [x] Style pass against `community/STYLE_GUIDE.md` (31 changes, `reviews/STYLE_PASS.md`). Checked
      independently afterwards: every numeric token compared as a multiset (a one-number mutation control was
      caught), every review-required caveat located, both factual edits traced to the reviews.
- [x] The four factual items the style pass flagged, fixed at release: "no model calls" → "no LLM calls" (v10
      calls an encoder once per question); the verbatim matcher defined from `evidence_recall.py`; §9 names
      its judges; the official scripts pinned to LongMemEval commit `9e0b455` (byte-identity checked against
      the commit, with a control).
- [x] Release placeholders resolved: §6's code cell and §8 now point at this directory; the Agent Zero
      author order checked against the arXiv record (Wu, Zhu), and the to-do note removed.
- [x] Academic version: human-only byline, first-person reflection removed, AI contribution acknowledged
      (Contributions, Acknowledgments, LLM Usage Statement)
- [x] PDFs built for both versions (`build_pdf.sh`; 14 pages each; no missing glyphs; one 4 pt table overhang)
- [x] `code/` and `config/` `SHA256SUMS` regenerated: the first versions listed a hash of themselves and could
      never verify. Now `sha256sum -c` passes, and a tampered copy fails.

### Sign-off
- [x] External sign-off (Dwayne Wilkes, Kavi) **waived by Thomas, 2026-09-25**: "we don't really need Dwayne and
      Kavi. This is a technical feat, not an esoteric exploration."
- [ ] Zenodo deposit (Thomas)
