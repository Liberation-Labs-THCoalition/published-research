# Style pass: `paper.md` → `paper.styled.md`

2026-09-25. Scaffold: `published-research/community/STYLE_GUIDE.md`, plus Lyra's style notes to Nexus
(`agents/nexus/messages/from_lyra_style_and_academic.md`, 2026-08-16). Caveats were checked against
`REVIEW_paper_fable.md` and `/mnt/data1/lme_v2/v10/REVIEW_v10_held_readers.md` ("Wording I would accept").
`paper.md` is untouched.

**How I cite the guide.** The guide has two sections numbered §4 and two numbered §5. I cite them as
**§4a** (null results), **§4b** (first-person reflections), **§5a** (establish vs preliminary) and **§5b**
(disclosure is the fallback for the unachievable).

**Checked mechanically** (`scratchpad/stylepass/verify.py`):
- The title, author lines and the `> **DRAFT …` block are byte-identical to `paper.md`.
- The `[DECISION: …]` marker in §8 is byte-identical.
- The set of numeric tokens, superscripts included, is exactly the same as in `paper.md`: no number is new
  and none is lost.
- All 13 tables are well-formed.
- After the §4 reorder, the only two `§4.x` cross-references are correct.

## What was tilting the paper

- **The lede opened on setup.** The headline number arrived in the seventh sentence, and the placement in
  the last paragraph.
- **The contributions list named activities, not findings.** It did not contain a single result.
- **The second headline was missing from the abstract.** The title's claim, whole sessions not passages
  (retrieval is the main effect and the reader matters less), never appeared there.
- **Results met the n = 100 tuning block before the held-out headline.**
- **Limitations had 7 items.** One was a robustness result filed as a limitation (order dependence), and
  one repeated a caveat that is already stated in place (full history).
- **A verified strength was stated only in the reflection box:** one attempt per question, nothing
  re-rolled.
- **Note ⁽³⁾ still said "It has not been second-read."** The second read had reproduced it.

## Changes (31)

| # | Where | Change | Rule |
|---|---|---|---|
| 1 | Abstract ¶1 | Opens on the result: v10, 385/400 and 387/400 with their judges and intervals. The setup sentence ("Long-term memory benchmarks reward…") is dropped. | §1 |
| 2 | Abstract ¶1 | The all-500 figures move up beside the held-out ones, with the reviewers' wording "100 of which were the retrieval-tuning split". | §1 |
| 3 | Abstract ¶1 | The placement moves from the last paragraph to the first. "The margin is small:" is replaced by the fact: the Sonnet-judge figure is one question above 95.60, inside every interval (§6). | §1, §2; names the Sonnet figure per paper review O9; see Conflict 1 |
| 4 | Abstract ¶2 | States the second headline. "Retrieval is the main effect" (from §4.4), the size of the reader upgrade (4.0 / 2.25, from §7.2), and 362 of 365 correct when the evidence is complete (from §4.5). | §1 (two-headline papers) |
| 5 | Abstract ¶2 | "The reading template differs by one clause" becomes a §3.3 pointer. The full caveat stays in §3.3, the §3.5 table and §4.4. | §2 (no hedges in the abstract body) |
| 6 | Abstract ¶2 | The pair order is defined once: "(Sonnet / Opus judge)". | §3 |
| 7 | Abstract ¶2 | "v10 scores every turn" becomes "ranks every turn", because the abstract also uses "scores" for accuracy. | §3 |
| 8 | §1, contributions | Each item states its finding: 97.1% recall; a headline candidate declared before any held-out accuracy existed; 96.2 / 96.8; +13.25 / 14.5 from retrieval against +4.0 / 2.25 from the reader; 95–97% judge agreement; the placement with its margin. | §1 |
| 9 | §1, item 3 | "an evidence-only ceiling" becomes "the evidence-only (oracle) arm". | Consistency with held review F8, which §4.4 carries: the oracle is not a strict ceiling |
| 10 | §1, item 1 | "turn-level lexical (BM25)" | §3 (orient before detail) |
| 11 | §2.2 | BM25 gets a clause saying what it is, a citation, and names for k1 and b. | §3 |
| 12 | §2.2 | The bge encoder is cited (Xiao et al., 2023). | §3. Robertson & Zaragoza and Xiao et al. were both in References but cited nowhere in the text |
| 13 | §2.3 | Defines the RRF constant k as the code computes it: weight / (k + rank), rank counted from 1 (`v10.py`, `rrf()`). | §3 |
| 14 | §2.3 | The order-dependence result moves here from §7.4, next to the tie-break it qualifies. | §2 (fold into the relevant Methods sentence) |
| 15 | §2.4 | Defines "o200k" as tiktoken's `o200k_base`, as used in `v10.py`. | §3 |
| 16 | §2.4 | "LLM" is spelled out; it recurs in §6's quotation. | §3 |
| 17 | §3.1 | "HF" becomes "Hugging Face". | §3 |
| 18 | §3.1, table | Defines the type abbreviations MS, TR, KU, SSU, SSA and SSP. §4.3's two tables used them without definition. | §3 |
| 19 | §3.3 | Adds that a second reader found exactly one attempt per held-out prompt and reader model in the v10 and oracle arms, with nothing discarded or re-rolled. | No rule. This is the brief's "buried strength": the fact was stated only in the reflection. Source: REVIEW_v10_held_readers.md, "Cherry-picking test", scoped to exactly the arms it covered |
| 20 | §3.5 | The "reader input" definition moves from the end of the dev block to its first use in the methods. | §3 |
| 21 | §4 | The held-out results (now §4.2–4.5) come before the n = 100 tuning block, which becomes §4.6. | §2 (survivors before fragilities); §5a (keep established and preliminary results apart) |
| 22 | §4.1 | One sentence defining the two recall rows (`has_answer` vs `answer_session_ids`). | §3 (define metrics at first use) |
| 23 | §4.3 | The multi-session sentence separates the same-reader retrieval gain (56.6 / 57.5 to 87.7 / 89.6) from the headline arm (94.3 / 95.3). The old sentence folded the reader upgrade into a retrieval story. | §1 (the defensible form of the finding) |
| 24 | §4.6 | "Caveat." becomes "Scope.", and the quoted phrase it pointed to, which no longer appears anywhere, is replaced. | §2 |
| 25 | §4.6 | "They were first second-read in the review of this text" becomes "The review of this text reproduced…". | Lyra 08-16: "don't narrate the correction" |
| 26 | §5 | The prose now leads with agreement (95–97%, κ 0.739–0.940) and then gives the offset. Every qualifier is kept. | §2 (robust results before fragilities) |
| 27 | §7.4 | Opens with "What a replication should control for:". | §2 |
| 28 | §7.4 | Drops "Full history, held-out". It is stated in place in §4.6 (Scope), the abstract ("on the 100 tuning questions") and §7.3 ("at n = 100"). The list goes from 7 items to 5. | §2 (cap; no double-counting) |
| 29 | §7.4, Harness | Adds the §3.7 fact that oracle + Opus 5.5 answered all 107 held-out temporal questions despite the date conflict. | §2 (tell the replicator what the limitation did and did not move) |
| 30 | Reflection | "(§4.2)" becomes "(§4.6)". | Follows #21 |
| 31 | Note ⁽³⁾ | Deletes "It has not been second-read." It contradicted the header, the note's own first line and REVIEW_paper_fable.md, which reproduced ⁽³⁾. | Not a style rule: stale status that undersold verified work |

## Required caveats: where each one lives now

| Caveat | Source | In `paper.styled.md` |
|---|---|---|
| Every accuracy is stated with its judge | paper review F8; held review Claim 1 | Title; abstract ¶1–2; §1 item 3; §4.2 |
| Thinking enabled, median 114 thinking tokens | held review F7 | Abstract ¶1; §3.3 |
| The all-500 figure includes the 100 tuning questions | held review F1 | Abstract ¶1 (accepted wording); §4.2; §6 |
| The held-out 400 are not ranked against S-500 rows | held review F1 | §6 subsets table (unchanged) |
| Claim 3 in full: traced to a primary source; one question; 95.60 inside all four intervals; no named judge or code; the Mastra comparison crosses judge and variant; unverified claims excluded | held review F4 and Claim 3 | §6, unchanged. The abstract ¶1 and §1 item 5 carry the scope and the margin |
| `has_answer` qualifier, with 96.0% on `answer_session_ids` | paper review F4 | Abstract bullet (verbatim); §1 item 1; §4.1 (now defined); §9 |
| v9 used the `merge` template (one clause differs) | paper review F1 | §3.3, §3.5 and §4.4, unchanged; the abstract ¶2 and §1 item 3 point to §3.3 |
| Harness additions: real date, `-` line, e-mail | held review F2 | §3.7, unchanged; §7.4; §8 |
| No temperature control; no 800-token cap | held review Claim 1 and F9 | §3.7, unchanged; §7.4 |
| Calibration gives direction only: add no points; significant on one set; concentrated in SSP | held review F3 and Claim 2 | §5 bullets, unchanged; §7.4 |
| The oracle uses its own dates and is not a strict ceiling | held review F8 | §4.4, unchanged; §1 no longer says "ceiling" |
| Embeddings ran on a GPU; a CPU sample matched at cosine 1.0000 | paper review F2 | §2.2; §7.3 |
| Second-reader wording | paper review F3 | Abstract (verbatim); Acknowledgments |
| 46% lenient / 35% strict | paper review F5 | §1, unchanged |
| The dev misses as corrected | paper review F6 | §4.5, unchanged |
| "Best published claims, which name neither judge nor code" | paper review F7 | Reflection, unchanged |
| The full-history comparison is dev-only, n = 100 | paper; reviews | Abstract bullet; §4.6 Scope; §7.3 |
| Counts given alongside rounded percentages | held review F6 | §3.6; all tables unchanged |
| "Pre-registered" means local read-only files, with no public registry | paper review F8 note | §3.2, unchanged |
| The v9 autopsy saw all 500 questions | paper §3.2 | §3.2; §7.4 |

## Rejected

- **Taking the placement out of the abstract altogether** (the strictest reading of §2). It is a finding
  with evidence behind it, and removing it would under-report.
- **"Matches the full history"** instead of "cannot be distinguished from". That is a bigger claim than a
  non-significant n = 100 test supports.
- **Using §3.4's 12/12 adjudication to promote the Opus-judge 96.8 as the better figure.** That is choosing
  a judge after seeing the scores, which §3.4 itself rules out.
- **Moving §6's "What we claim" above its "Rules".** "Context, not a controlled ranking" has to come before
  the claim. §6 is untouched.
- **Cutting §2.2's departures from the design note** (GPU, user turns only) as "code, not methodology" (§2).
  They are departures from the pre-registration note. Lyra's notes keep "what was designed and never run"
  for the same reason.
- **Cutting §1's paragraph on the superseded 85.8% figure.** §2's exception applies: it was a DOI'd report
  that others may have relied on.
- **Rewriting the reflection.** It matches the final state (§4b).
- **Moving §3.7's long item to an appendix.** §2 puts long disclosures in a methods subsection, and §3.7 is
  one.
- **Moving the v9 floor paragraph out of the introduction.** It is motivation there. §1's rule governs the
  abstract lede, which is fixed.
- **Touching the title.** Thomas decided it (see the header).

## Conflicts for Nexus

1. **§2 (no hedges in the abstract) against the Claim 3 qualifiers.** I kept "one question above the best
   published claims (95.60), inside every interval (§6)" in the abstract, compressed. Without it, "highest
   point estimates" reads as a clear lead, which the evidence does not show.
2. **§2 against the accepted abstract wording.** The `has_answer` parenthetical and "100 of which were the
   retrieval-tuning split" are qualifiers inside the abstract. I kept both verbatim.
3. **Lyra's "don't narrate the correction" against "every number stays".** §3.1's "We found this only on
   2026-09-24" is narration, but deleting it would remove the paper's only instance of that date. I left it
   unchanged. It may also matter to readers of the 2026-08-05 report (§2's exception). Your call.
4. **§5b: disclose only what is unachievable.** Three of the gaps the paper discloses are achievable:
   - the full history on the held-out 400 (about 51M tokens);
   - variance across samples (the paper has one sample per question);
   - the effect of the harness reminders (the held review suggests testing suppression for v10.1).

   Running any of them would add numbers, which is outside this pass. The disclosures are the right
   treatment for the official GPT-4o judge (no OpenAI access), temperature (`claude -p` cannot set it) and
   the 8.5k held-out run (the protocol forbids it).

## Factual items for Nexus, not changed

- **"Makes no model calls"** (abstract, §1 item 1, §9) sits against §7.3's "one encoder call for the
  question". §2.4's "no LLM calls" is the exact version, and a critic will read "no model calls" literally.
- **"Verbatim matcher"** (§1, §4.1) is never defined (§3). I could not define it without the autopsy code.
- **Four words would complete §9's Accuracy bullet.** It states 96.2–96.8% without naming the judges, and
  the rule behind paper review F8 is that every accuracy names its judge.
- **Change #19 is new body text Thomas has not read.** Its source is REVIEW_v10_held_readers.md,
  "Cherry-picking test".
- **The held review's F3 supports a line the paper does not use:** "the official judge would probably not
  score these answers lower". Adding it would be a new claim, so I left it out.
- **§8 pins the official scripts by fetch date, not by commit.** Lyra's 08-16 notes, item 1 (pinning), ask
  for the commit hash.
- **I did not apply Lyra's release-language template** (08-16 notes, item 2). I do not have the template,
  and the release is the open `[DECISION]`.

## Not checked (outside this text)

- The build gate, `./scripts/build_and_verify.sh`.
- The checklist's ".tex is canonical" rule: this paper exists only as Markdown.
- `PREREG_TEMPLATE.json`.
