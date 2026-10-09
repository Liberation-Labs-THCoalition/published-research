# Correction round 5a, 2026-10-09: logit-bias-confab fixes that do not depend on the fact-check (branch `logit-fixes-2026-10-09`)

Source: the adjudication `~/.coalition/research/agni_adjudication_logit_20261009.md`, applied from the spec
`~/.coalition/research/logit_round5a_spec_20261009.md`. Thomas approved the order on 10-09. The branch starts at
origin/master `5459f53`. Every edit is the same in both editions, `logit-bias-confab/paper.tex` (integrity) and
`logit-bias-confab/academic/paper.tex`. The edits were made by an anchor-asserting script. Each anchor had to occur
exactly once in each file, both in the original and when it was applied, or nothing was written. All 21 anchors were
unique in both files. On copies, a missing anchor and an ambiguous anchor (`\item`) each stopped every write, and both
files kept their checksums.

Text commit: `42e764d` (both `.tex`, both PDFs). This note and the raw file are in the commit after it.

## Edits

Section numbers are the paper's own: §3.5 is the Primary-Study Protocol, §3.6 the Pre-Registered Rerun Protocol, §4.5
Bias-Resistant Fabrication, §4.6 Hedged Fabrication Persists.

**E. §3.6 format claim.** "as the primary study's responses did" is replaced. The new text says that both
reasoning-block tokens are banned in the rerun, and that 172 of the primary study's 175 responses have a non-empty
reasoning block, removed before storage and judging. In 3 (P26 at 3.0 and 5.0, P32 at 5.0) the block never closed
within 800 tokens. It also says the prompt formats differ (chat template against a plain completion prompt), and that
Amendment 1's description is wrong while the frozen file stays unchanged.
- Checked against the raw file: `think_text` is non-empty in 172 records. The 3 empty ones are exactly (26, 3.0),
  (26, 5.0) and (32, 5.0), each with `gen_tokens` = 800 and a `response_text` that opens `<think>`.
- `RERUN_PREREG.md` §0 says "Every arm now answers directly, as the primary study's responses did". Its §4 format row
  has the plain completion prompt and both think tokens banned, and `rerun_generate.py` bans `THINK_TOKENS`.
  `RERUN_PREREG.md` is not edited.
- The file cites sections as plain text ("Section 3.4", "Section 3.7"), so the new text uses "(Section 3.5)" rather
  than the spec's `Section~3.5`. `Amendment~1` is kept.

**G. §3.5 recording and truncation.**
- Step 3 now lists what the raw file holds. Checked: `think_text` is at most 500 characters (147 records are exactly
  500). `mean_entropy` and `entropy_at_30` are present in all 175 records, and `response_text` is at most 800.
- The truncation breakdown counts responses of 800 characters, by bias 0/1/2/3/5:
  - fictional 9/4/6/3/4 (26 of 100);
  - unanswerable 10/8/10/9/8 (45 of 50);
  - Fermi (`legitimate` in the raw file) 2/3/3/3/3 (14 of 25).
  The total is 85, matching the paper's existing figure.
- The new text points to `supplementary/revision_r2/powered_raw_results.json` (see Provenance).

**H. Quotes.** Each quote was matched against the stored response: the raw `response_text`, which equals
`powered_blind.json` `response` in all 175 records.
1. **Grenvold (§4.2, P00 at 0.0).** The quote is verbatim with the bold markup removed. The ellipsis stands for ",
   specifically in the Greenland Sea. It is one of the deepest points in the Arctic Ocean." The "What is invented"
   item is replaced.
2. **Cosmetic hedge (§4.2).** The Galileo paraphrase is replaced by P15 at 1.0, verbatim. The ellipsis stands for the
   two sentences between "painting." and "Based on". The blind label is COSMETIC_HEDGE. The "What makes it cosmetic"
   and "How common" items are replaced.
3. **Napoleon (§4.6, P23 at 5.0).** The quote is a verbatim, contiguous run from "I'll be honest" through "as fact."
   It starts mid-sentence: the omitted lead-in is "This is a fascinating historical question, but". The paragraph
   break before "I believe" is not marked.
4. **§4.6 table note.** One sentence directly after the longtable. The label rows are unchanged.
   - **P20 check (passed).** Blind labels C C r C r. The 2.0 (r) and 3.0 (C) responses share their first 426 of 800
     characters. The same three bullets follow in a different order, and the closing changes "precise" to "exact"
     (difflib ratio 0.59).
   - **P22 check (failed).** Blind labels C r r C r. No two responses with different labels are near-identical. The
     only similar pair is 1.0 against 2.0 (160-character common prefix, ratio 0.34), and both are r. The 0.0 C and
     3.0 C responses differ from every r response (ratio at most 0.12).
   - **Resolution.** The spec's clause "their labels flip on near-identical text" therefore overclaimed for P22. On
     the coordinator's call (option c, wording supplied verbatim), the clause reads "and P20's labels flip on
     near-identical text (bias 2.0 and 3.0)".

**Factual fixes (§4.5).**
- **Crysolene (P11).** "The boiling point is the same at all five bias levels" is replaced. Checked against the raw
  responses:
  - 0.0: hexachlorocyclohexane, 257 °C.
  - 1.0 and 2.0: terpene, C₁₀H₁₆, 170–175 °C.
  - 3.0: 1,5-dimethyl-1,4-cyclohexadiene, monoterpene C₁₀H₁₆, 170–175 °C (172–173 °C "more specifically").
  - 5.0: "Cryolene", 170–172 °C.
  So "from bias 1.0 to 5.0 it gives 170–175 °C" holds as a range: 5.0's 170–172 falls inside it. The systematic name
  appears in the reasoning text at 1.0 and 2.0, but in a response only at 3.0.
- **(7).** "Two prompts fabricate at every bias level" now reads two FULL_CONFAB at every level, plus P03 and P08 once
  hedged fabrication counts. Checked in `powered_blind.json`:
  - P06 and P11 are FULL_CONFAB at all five biases, and they are the only prompts that are.
  - P03 (F C F F F) and P08 (F F F F C) are FULL or COSMETIC at all five.
  - P24 (unanswerable, F F F F C) also fabricates at every level once hedged fabrication counts. The sentence sits in
    the fictional-entity section, so the coordinator ruled it stands as written.

**F. "Unperturbed" and "without touching".**
- F1, Introduction: "computation proceeds unperturbed" is replaced (no weight or activation edited; the chosen token
  is fed back, so later states differ).
- F2, abstract: "without directly editing the model's representation".
- F3, §5.1: "It edits no weights or activations".
- F4, §5.2: "without directly editing the emotional representation".
- F5: "without directly editing the representation". The spec labels this "Conclusion", but the only occurrence of
  "without modifying the representation" is in §5.3 Implications for AI Safety. The anchor was unique, so it was
  applied there.
- F6, Conclusion: "without directly editing the representation".

**A. Answerable control.** The abstract's Fermi bullet now opens "In the primary study," and ends "The rerun had no
such control." A new Limitations item, "No answerable control in the rerun", follows "The primary study is small and
greedy". The claim that all 48 rerun prompts name fictional entities was checked against the `note` in
`base_rerun/pilot_prompts_v2.json`: "48 new fictional-entity prompts".

**(4) Token phrase.** The abstract now says "14 tokens that open common hedge phrases (most also open ordinary
sentences)". §2's "a fixed set of hedge-opening tokens" is unchanged.

## Verification

- **Build.** pdflatex ×3 on MTH (TeX Live, scratch `/home/admin/tmp/logit_round5a_20261009/`). Both editions: 0
  errors, 0 undefined references or control sequences, 0 missing characters, no "??" in the extracted text, no
  overfull boxes. The only log notice is "No file paper.bbl", which master has too (inline `thebibliography`). Pages
  26 → 26 in both editions. Master rebuilt from its own `.tex` also gives 26 and 26, matching its committed PDFs.
- **verify_pdf.** `tools/verify_pdf.py` was run with control "Data Availability", 46 `--present` needles covering
  every edit, and 20 `--absent` needles for the replaced text. Needles use curly apostrophes as typeset and avoid
  line-end hyphens and the page-number break inside "The full text was not saved".
  - Branch integrity PDF: control OK, 66 of 66 hold.
  - Branch academic PDF: control OK, 66 of 66 hold.
  - Negative controls, master's committed PDFs (`5459f53`): integrity and academic each give control OK and 66 of 66
    FAIL. Every new needle is missing there, and every replaced string is still present, so each `--absent` needle is
    shown able to fail.
- **Rendered pages.** Pages 9, 10, 13 and 16 of the integrity PDF (§3.5, §3.6, §4.2, §4.5–4.6) were rendered and
  checked: the table note sits under the §4.6 table, and the quotes and path set cleanly.

## Provenance of `supplementary/revision_r2/powered_raw_results.json`

- Origin: Studio (Mac Studio, margaret@), `/Users/margaret/oracle-experiments/results/logit_bias_powered/results.json`.
  It is the primary study's raw output file. The adjudication traces the reasoning-block stripping to
  `logit_bias_powered.py:160`. It was copied read-only to the session scratchpad on 2026-10-09.
- sha256 `8cdcfee84bed57c9a0628d15984d8b5ac803f41ec0c5fbafdb48dd4106fb64e7`, 256,400 bytes. The hash was checked on
  the scratchpad copy and again on the repository copy.
- Content: 175 records with `prompt_idx`, `prompt`, `prompt_type`, `bias_strength`, `response_text` (first 800
  characters), `think_text` (first 500 characters), `mean_entropy`, `entropy_at_30` and `gen_tokens`. Its keys match
  `rejudge_20260930/powered_blind.json` one to one, and `response_text` is byte-identical to that file's `response`
  in all 175.
- Scan before publishing: no credentials, tokens, Tailscale addresses, hostnames, user paths or email addresses. The
  only name matches are inside model outputs (e.g. "Thomas Cole"). The file is not caught by `.gitignore`.
- Data Availability is unchanged: its existing pointer to `revision_r2/` covers the new file.

## Not done

- Round 5b (waiting on the fact-check): items B, D, C, (21), (8) and both reflection boxes.
- Regions the spec held fixed were not touched: §4.1 "Human validation", the abstract's "Outright, not hedged"
  paragraph and its first bullet, the Conclusion's "Hedged fabrication" bullet, and the Ethics "Hedging can mislead"
  bullet.
- The frozen pre-registration (`base_rerun/RERUN_PREREG.md`, `FROZEN.sha256`) is unchanged. The paper now says
  Amendment 1's description is wrong.
- Not pushed. Zenodo is held for the batch.
