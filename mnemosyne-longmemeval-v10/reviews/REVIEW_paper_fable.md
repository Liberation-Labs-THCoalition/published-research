# paper.md (Mnemosyne v10 on LongMemEval_S-cleaned): adversarial second read of the draft text

2026-09-25. Second reader: a Fable 5.1 agent. Zero model tokens, CPU only, no GPU, no `claude -p`, no ollama. Read-only
everywhere except this file and the scratchpad `paperreview/`. Every number below was recomputed from the raw judgment,
answer, prompt, recall and dataset files with my own code (`recompute_held.py`, `recompute_dev_stats.py`, `calib2.py`,
`autopsy_calib.py`, `dev7.out` in the scratchpad; outputs saved as `*.out`). The builder's `compare_arms.py`,
`calib_judges.py` and `v10.py` were read, never run. The earlier second reads (`REVIEW_v10_recall.md`, Opus 5.5 agent;
`REVIEW_v10_held_readers.md`, Fable 5.1 agent) were treated as binding on wording. Scaffold: `agni/meridian/reviewer.py`
(FAIL / WARN / PASS, evidence quote + source ref on anything filed above PASS).

## Verdict: WARN — CONFIRMED WITH CAVEATS. Not depositable as written; every fix is to text.

**Every number in the paper traces and reproduces.** All held-out and all-500 counts, intervals, task-averages,
McNemar pairs, per-type figures, judge agreement, self-preference lifts, dev-block tables, the calibration table, the
retrieval-recall table, the stage sweeps and budget curve, the dataset statistics, the thinking-token and >800-token
figures, and the three numbers first computed for this draft ⁽¹⁾⁽²⁾⁽³⁾ all come out of the raw files exactly as printed.
No FAIL criterion is met: no value contradicts its source, no citation is fabricated, no headline contradicts the data.

**Nine things must change before deposit** (all WARN; none moves a number):

1. **§3.3 describes the wrong reader template.** v10, oracle and full-history prompts use the official *plain-history*
   template (`merge_key_expansion_into_value="none"`); only the v9 arm used the history-plus-facts `merge` template. That
   also means the v9-vs-v10 "reader held fixed" comparison carries an undisclosed instruction difference. (F1)
2. **§2.2 and §7.3 say the embeddings were computed on CPU. They were computed on a GPU** (`device: cuda Quadro K2200`),
   contrary to both the paper and the pre-registration note. (F2)
3. **The abstract and acknowledgments say a second reader "reproduced every count".** The dev-block reader numbers were
   never second-read (the paper says so itself in §4.2), the ⁽ⁿ⁾ numbers had no second read until this one, and there were
   two second readers on two models, not one. (F3)
4. **The abstract and conclusion drop the `has_answer` qualifier** the recall review made a condition of publishing 97.1%.
   (F4)
5. **§1's "46%" is the lenient-matcher figure; §4.1's "0.35" for the same quantity is strict.** Say which. (F5)
6. **§4.6's dev breakdown misdescribes two of the seven misses**: one "hard counting" question is a preference question
   whose only evidence session was not retrieved, and one is an abstention. There are two retrieval misses, not one. (F6)
7. **The reflection says "one question above the best unverifiable claims".** The 95.60s are traceable published claims;
   the claims the paper *excludes as unverifiable* are higher than ours. (F7)
8. **Two references are incomplete** (the paper flags one). Author lists and titles supplied below, affiliations verified
   from the PDFs. (F9)
9. **The title carries a bare "96%"** with no judge. Recommended, Thomas's call. (F8)

---

## Findings, by severity

No FAIL findings.

### F1 (WARN, source misrepresentation of our own harness): §3.3 names the `merge` template; the v10/oracle/full arms used the `none` template, and only v9 used `merge`

**Claim.** `paper.md:202-204`: *"The prompt comes from the official LongMemEval prompt builder, using its history-plus-facts
template (`merge`) with chain-of-thought, and giving the dataset's question date as `Current Date:`."* §2.4 (`:134-135`)
adds that v10 prompts go through *"the same one used for the oracle and full-history arms."*

**Evidence.**
- `/mnt/data1/lme_v2/v10/build_v10_prompts.py:54-56`: `prepare_prompt(entry, "orig-session", 10**6, False, "json", True,
  tokenizer=enc, tokenizer_backend="openai", max_retrieval_length=180_000, merge_key_expansion_into_value="none")`
- `/mnt/data1/lme_v2/baseline/build_baseline_prompts.py:9-10` (docstring): *"Settings are the official CoT reading setup
  without fact merging: history_format=json, useronly=false, cot=true, merge_key_expansion_into_value=none"*; `:57-59` passes
  `merge_key_expansion_into_value="none"`.
- `/mnt/data1/lme_v2/official/run_generation.py:55` (`none` + CoT): `'I will give you several history chats between you and
  a user. Please answer the question based on the relevant chat history. Answer the question step by step: ...'`; `:60`
  (`merge` + CoT): `'I will give you several history chats between you and a user, as well as the relevant user facts
  extracted from the chat history. Please answer the question based on the relevant chat history and the user facts. ...'`
- `/mnt/data1/lme_v2/baseline/v10_o55t/prompts/q000.txt:1` begins *"I will give you several history chats between you and
  a user. Please answer the question based on the relevant chat history. Answer the question step by step"* (the `none`
  text); so do `baseline/oracle_o55t/prompts/q000.txt:1` and `baseline/full/prompts/q001.txt:1`.
- `/mnt/data1/lme_v2/prompts/q000.txt:1` (the v9 arm) begins *"I will give you several history chats between you and a user,
  as well as the relevant user facts extracted from the chat history."* (the `merge` text). `PROTOCOL.md:104-107`: *"Reading
  template: the official `merge_key_expansion_into_value == 'merge'` + CoT variant ... That is what a Mnemosyne v9 context
  is."*

**Effect.** No number changes. "Two readers read identical prompts" (Opus 4.6 vs Opus 5.5 on v10) stays true: the 400
held v10 and v10_o55t prompt files are byte-identical (checked). But §3.3 misdescribes the prompt the headline reader saw,
and §4.5's *"With the reader held fixed"* (`:373`) and the abstract's *"with the same reader"* (`:27`) present the v9-vs-v10
comparison as controlled while the instruction sentence differs between the two arms. The `merge` template is the
official one for a fact-augmented context and is arguably the right choice for v9; it still has to be said.

**Why WARN and not FAIL.** The scaffold files a label that does not match the data as WARN unless it changes a
conclusion. A 13–14.5-point paired gap does not turn on one instruction clause, and the choice favours v9 if anything.
Contest it if you disagree.

**Fix.** §3.3: *"using its plain-history chain-of-thought template (`merge_key_expansion_into_value = none`)"*. Add to §3.5
or §4.5: *"The v9 arm, whose context contains extracted facts, used the official history-plus-facts (`merge`) variant of
the same template; v10, oracle and full history used the plain-history variant. The two differ by one clause."*

### F2 (WARN, context drift + presentation): the turn embeddings were computed on a GPU, not on CPU

**Claim.** `paper.md:116`: *"Encoder: BAAI/bge-base-en-v1.5 (revision `a5beb1e3`), run on CPU."* `paper.md:494-495`: *"The
turn embeddings are computed ahead of time, on CPU."*

**Evidence.** `/mnt/data1/lme_v2/v10/emb/embed.log:2`: `device: cuda Quadro K2200`. `/mnt/data1/lme_v2/v10/embed_turns.py:2`:
*"Embed every unique LongMemEval_S-cleaned USER turn, plus the 500 questions, on GPU1."*; `:5-6`: *"(A CPU run over all
189,522 turns managed 10 turns/s on this loaded machine, >5 h: see emb/old_cpu_partial.)"*; `:9`: *"Run with
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"*. `emb/meta.json` records no device. The pre-registration note said
CPU: `/mnt/data1/lme_v2/v10/DESIGN.md:74-75`: *"Dense cosine over turns, using bge-base-en-v1.5 on CPU."*

**Effect.** None on any result: the recall review recomputed 24 sampled turn rows and 16 question rows on CPU and matched
the stored rows at cosine 1.0000 (`REVIEW_v10_recall.md:228-230`). But the paper states a fact about the run that is false,
in the section a reproducer reads, and §7.3 uses it as a cost claim.

**Fix.** §2.2: *"run once on a Quadro K2200 (fp32 model, fp16 storage); a second reader recomputed a sample on CPU and
matched at cosine 1.0000."* §7.3: *"The turn embeddings are computed ahead of time (one pass over 93,931 unique user turns;
GPU here, CPU-feasible)."* Optionally note in §2.5 that the design note planned CPU and assistant-turn embeddings, and
the run used a GPU and user turns only (`SELF_LIMITS_AUDIT.md:25` lists the latter as a compute limit).

### F3 (WARN, overstated verification + misattribution): "reproduced every count" and a single second reader

**Claim.** `paper.md:34`: *"An adversarial second reader reproduced every count from the raw files."* `paper.md:547-548`:
*"A Fable 5.1 agent served as adversarial second reader. It reproduced every count from the raw files, corrected our
placement wording, and found the harness disclosure in §3.7."*

**Evidence.**
- `REVIEW_v10_held_readers.md:10-11`: *"Every count in `COMPARISON_held_readers.md` reproduces exactly from the raw
  judgment files"* — that file is the 400-question held table, not the paper.
- `RESULTS_v10.md:10-11`: *"Reader results, dev: unchanged from 09-24 and not second-read on their own."* The paper
  agrees with itself at `:318-320`: *"They have not been second-read on their own"*.
- `paper.md:8-9`: *"Numbers first computed for this draft are marked ⁽ⁿ⁾ and still need their own second read."*
- `REVIEW_v10_recall.md:3`: *"Second reader: an Opus 5.5 agent."* The retrieval-recall numbers in §4.1 were second-read by
  that agent, not by a Fable agent.

**Effect.** The abstract asserts a completeness of verification the paper's own §4.2 and footnotes deny. As of this
review the ⁽ⁿ⁾ numbers *are* reproduced (below) and the dev-block tables reproduce too (below), so the sentence can be made
true rather than deleted.

**Fix.** Abstract: *"Adversarial second readers reproduced every held-out count, the retrieval-recall figures and the
numbers first computed for this report from the raw files."* Acknowledgments: *"An Opus 5.5 agent second-read the
retrieval recall; a Fable 5.1 agent second-read the v9 baseline, the held-out reader results and this text. Between them
they reproduced every count from the raw files, corrected our placement wording, and found the harness disclosure in
§3.7."*

### F4 (WARN, required caveat dropped): "every evidence session" without the `has_answer` qualifier

**Claim.** `paper.md:26`: *"**Retrieval:** v10 puts every evidence session into the context for 97.1% of held-out
questions."* `paper.md:535-537`: *"puts all the evidence in front of the reader for 97% of held-out LongMemEval_S-cleaned
questions."*

**Evidence.** `REVIEW_v10_recall.md:13-15`: *"**The wording.** "All evidence sessions" is measured on `has_answer`-labelled
sessions. On the benchmark's own `answer_session_ids` it is **96.0%** held-out, not 97.1% (F1)."*; `:60`: *"**Fix.** Say "all
`has_answer`-labelled evidence sessions", or report both figures (97.1 / 96.0)."* §4.1 (`:280-281`) does report both; the
abstract and §9 report only 97.1 with the unqualified phrase the review struck.

**Fix.** Abstract: *"v10 puts every `has_answer`-labelled evidence session into the context for 97.1% of held-out questions
(96.0% when every official `answer_session_ids` session is required)."* §9: *"puts every labelled evidence session in front
of the reader for 97% (96% on the official session list) of held-out ..."*

### F5 (WARN, internal inconsistency): "46%" in §1 is the lenient-matcher figure; §4.1 reports the strict one (0.35)

**Claim.** `paper.md:62-64`: *"But on multi-session questions, by our autopsy's measure, all the evidence reached v9's
context only 46% of the time."* `paper.md:288-289`: *"Held-out recall by type, v10 against v9 strict: multi-session: 0.97
against 0.35"*.

**Evidence.** Recount from `/mnt/data1/lme_v2/review/evidence_recall.json` (`autopsy_calib.out`): 121 non-abstention MS
questions with evidence; all sessions covered **strict 43/121 = 35.5%**, **lenient 56/121 = 46.3%**.
`MULTISESSION_BRIEF.md:35-41` table row *"all of it | 56"* and *"Only **46%** of multi-session questions get all their
evidence"* — the 56 is the lenient count. Both figures are correct; the paper uses one definition in §1 and the other in
§4.1 for the same quantity without saying so.

**Fix.** §1: *"only 46% of the time by the lenient verbatim matcher (35% strict; §4.1 uses strict)."*

### F6 (WARN, presentation, not second-read): §4.6's dev breakdown misdescribes two of the seven misses

**Claim.** `paper.md:393-398`: *"Seven questions were missed under both judges. Five of them were also missed by the
full-history reader; these are hard counting and ordering questions. One was a retrieval miss: two of three
magazine-subscription sessions were found. One had all four charity sessions in context and still summed wrong."*

**Evidence** (`dev7.out`; rows from `v10/runs/20260924T122332_eval_dev.json`, labels from `judge_v10_*` and
`judge_full_*`, question text from the dataset). The seven, with evidence covered/needed and full-history outcome:

| qid | type | ev | full S/O | question |
|---|---|---|---|---|
| d851d5ba | MS | 4/4 | right/right | "How much money did I raise for charity in total?" |
| 1a8a66a6 | MS | 2/3 | right/right | "How many magazine subscriptions do I currently have?" |
| bf659f65 | MS | 3/3 | wrong/wrong | "How many music albums or EPs have I purchased or downloaded?" |
| 09d032c9 | **SSP** | **0/1** | wrong/wrong | "I've been having trouble with the battery life on my phone lately. Any tips?" |
| a96c20ee_abs | **abstention** | 0/0 | wrong/wrong | "At which university did I present a poster ..." |
| gpt4_7abb270c | TR | 6/6 | wrong/wrong | "What is the order of the six museums I visited ..." |
| 370a8ff4 | TR | 2/2 | wrong/wrong | "How many weeks had passed since I recovered from the flu ..." |

Five were indeed also missed by the full-history reader, but only three of those are counting/ordering questions; one is
a preference question whose single evidence session v10 did not retrieve (a second retrieval miss), and one is an
abstention question. "86–87 of 92 with every evidence session in context" (`:394`) reproduces (86/92 Sonnet, 87/92 Opus;
the 92 include one abstention question that carries `has_answer` turns).

**Fix.** *"Seven questions were missed under both judges. Two were retrieval misses (two of three magazine-subscription
sessions found; a preference question whose one evidence session was not retrieved), one was an abstention question, and
four had every evidence session in context: three counting or ordering questions that the full-history reader also
missed, and one that had all four charity sessions and still summed wrong."*

### F7 (WARN, internal inconsistency): the reflection's "best unverifiable claims" contradicts §6

**Claim.** `paper.md:566-567`: *"the highest result we could trace to a source, one question above the best unverifiable
claims, with every deviation written down."*

**Evidence.** `paper.md:445-446`: *"the Sonnet-judge figure exceeds the top published claims (95.60, Chronos and Agent
Zero) by one question."* Those two are traceable to primary sources (`COMPETITORS.md:511` S18 arXiv:2603.16862; `:521` S20
arXiv:2608.29606). The claims the paper excludes as *unverifiable* (`paper.md:453-457`; `COMPETITORS.md:407-411`) are
Supermemory "~99%", OMEGA 95.4 and agentmemory 96.2 — all **above** our 95.8. So the sentence is false on its own terms
and hands a hostile reader the line "they say they beat the unverifiable claims, but excluded a 99 and a 96.2".

**Fix.** *"one question above the best published claims, which name neither their judge nor their code, with every
deviation written down."*

### F8 (WARN, presentation; recommended, Thomas's call): the title states a bare "96%"

**Claim.** `paper.md:1`: *"Mnemosyne v10 Reaches 96% on Pre-Registered Held-Out LongMemEval_S-cleaned Questions"*.

**Evidence.** The accepted wording never states an accuracy without its judge (`REVIEW_v10_held_readers.md:277-279`:
*"scores **385/400 = 96.2%** under a claude-sonnet-5 judge and **387/400 = 96.8%** under a claude-opus-5-5 judge"*), and §5
of the paper itself shows the official judge passing 0.4–2.2 points fewer of another system's answers. The abstract's first
bullet does carry the judges, so the exposure is one headline.

**Fix (optional).** *"... Reaches 96% Under Claude Judges on Pre-Registered Held-Out LongMemEval_S-cleaned Questions"*, or
keep the title and accept the attack. Also consider whether "Pre-Registered" (read-only local files with hashes and a
log; no public registry) should read "pre-specified" — both prior reviews accepted the term, so this is a judgment call.

### F9 (WARN, incomplete deliverable): two references lack authors and titles; everything else verified

**Claim.** `paper.md:579`: *"PwC (2026). Chronos. arXiv:2603.16862v1. [author list to be taken from the arXiv record before
deposit]"*; `paper.md:587`: *"Zero Labs (2026). Agent Zero Memory. arXiv:2608.29606."*

**Verified from the arXiv record and PDF first pages** (fetched read-only; PDFs at
`~/.claude/projects/-home-admin/5a81960d-.../tool-results/webfetch-1790377303872-3vg1c1.pdf` and `...-1a5fou.pdf`):
- **Chronos:** Sahil Sen, Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah — all *"Commercial Technology and Innovation
  Office, PricewaterhouseCoopers, U.S."* (pwc.com e-mails). Title: *"Chronos: Temporal-Aware Conversational Agents with
  Structured Event Retrieval for Long-Term Memory."* arXiv:2603.16862v1, 17 Mar 2026, *"Preprint. Under review."* The
  abstract states *"Chronos High scores 95.60% accuracy"*. The judge model is never named; footnotes contrast other systems
  as *"not directly comparable to GPT-4o-judged systems"* and *"systems evaluated with the official benchmark judge"*, which
  supports the survey's "judge model not named" (`COMPETITORS.md:87`).
- **Agent Zero Memory:** Pengyuan Zhu and Ming Wu, *"Zero Labs"* (meetzero.ai e-mails); the arXiv abstract page lists the
  order as *"Ming Wu, Pengyuan Zhu"* while the PDF byline puts Zhu first — take the order from the arXiv metadata and check
  once more at deposit. Title: *"Agent Zero Memory: Provenance-Aware Long-Term Memory for LLM Agents."* arXiv:2608.29606v1,
  30 Aug 2026. Abstract: *"95.60% on LongMemEval"*; dataset variant not named anywhere I grepped; no code link.
- **Wu et al.:** title, authors (Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu), *"ICLR 2025"*,
  arXiv:2410.10813 (v2, 4 Mar 2025) — all match `paper.md:582-584`.
- **C-Pack:** Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, Jian-Yun Nie; arXiv:2309.07597 (2023)
  — matches `paper.md:585-586`.
- **Zenodo 10.5281/zenodo.21801643:** *"Character Profiles Are All You Need ..."*, Thomas Edrington, 5 Aug 2026; abstract
  states *"0.943 F1 on the full 10-conversation LoCoMo benchmark"* and *"On LongMemEval, Mnemosyne scores 85.8% with LLM
  judge scoring"* — §1.1's two figures (`paper.md:79-80`) are what the record says.
- **Mastra, Plastic Labs, TAM URLs and dates** match `COMPETITORS.md` S9 (`:467`), S21 (`:524`), S25 (`:541`). Cormack et al.
  2009 (SIGIR) and Robertson & Zaragoza 2009 (FnTIR 3(4)) are domain-recall entries; nothing to check them against here.

**Fix.** Replace the two entries with the author lists and titles above.

---

## PASS-level observations (audit trail; no change required unless noted)

- **O1, ⁽¹⁾ phrasing.** `median_tokens` is 23,918.5 (low/high medians 23,918 / 23,919); the paper rounds half up to 23,919 —
  fine, but say "median 23.9k" or give the .5. The three medians are of three distributions: only 102/400 (25.5%) held
  contexts have every session whole; 415 windowed sessions across the 400. "9 sessions, 8 of them whole" is a fair summary.
- **O2, ⁽³⁾ precision.** The three complete-evidence failures are not the same three under both judges: 0a995998 and
  7024f17c (both MS) are shared; Sonnet adds caf03d32 (SSP), Opus adds 195a1a1b (SSP). "Reading, 3" is right as a count;
  "3 under each judge, two in common" is the exact version. The 10 retrieval failures are the same 10 under both judges.
- **O3, §3.7(5) "up to 25.3k".** From the 400 held v10 prompt files (o200k): history portion max 25,208 (25.2k), whole prompt
  max 25,296 (25.3k), median history 23,982; 168/400 held histories exceed 24,000 (RESULTS_v10.md's 211 is over all 500).
  Say "history up to 25.2k, prompt up to 25.3k".
- **O4, §2.1 "Nothing else reaches the retriever".** `v10.py:178` `retrieve(v, qidx, cfg, dense)` also receives the question's
  dataset index, used only to fetch its precomputed embedding row (`:189`) and to seed the random control (`:191`). The
  recall review verified the stored question rows equal the embedding of the question text alone. One clause would
  pre-empt a code-reading critic: *"plus the question's row index into the precomputed question embeddings."*
- **O5, §3.2 "before v10 existed".** `split.json` mtime 10:33:58; the first dev evaluation file is stamped 10:37:49. That
  supports "before any v10 result existed"; it cannot show `v10.py` did not exist at 10:33. Prefer the weaker phrase.
- **O6, §4.1 "Abstention questions have no evidence to recall".** 9 of the 30 carry `has_answer` turns (8 held, 1 dev), all
  covered by v10; they are excluded from the recall metric by rule. Say "are excluded from the recall metric".
- **O7, §5 significance sentence.** Accepted wording, but the exact version: Honcho set p = 0.043 (Sonnet) and 0.052 (Opus);
  full-context set p = 0.79 and 0.42. Under the Opus judge the offset is not significant on either set.
- **O8, §4.2 / §7.3 "~27.5k".** Median reader input (usage input + cache tokens) for v10 on dev is 27,022 (held 27,101);
  full history 126,710; ratio 21.3%. `SELF_LIMITS_AUDIT.md:57` has 27.0k. "~27k, 21%" is the raw-file figure; 27.5k / 22%
  comes from `RESULTS_v10.md`.
- **O9, abstract "one question above".** True of the Sonnet-judge figure only (the Opus figure is six questions above);
  the accepted wording says "the Sonnet-judge figure". Conservative, but name it.
- **O10, DESIGN.md deviations not mentioned.** The design note planned to embed assistant turns truncated to 512 tokens
  (`DESIGN.md:75-76`); the run embedded user turns only (`embed_turns.py:30`, `meta.json` `"roles": ["user"]`). The paper
  describes the code correctly; a one-clause disclosure of the change from the design note would close the gap.
- **O11, Chronos judge.** The Chronos paper's footnotes imply its judge is *not* GPT-4o-judged-comparable in the way it
  criticises others for — consistent with the paper's "judge model not named", nothing to change.

---

## The three numbers first computed for this draft: independent recomputation

All three reproduce exactly. Scripts: `recompute_held.py` (held/all-500/⁽¹⁾/⁽³⁾), `recompute_dev_stats.py` (dev, dataset,
harness stats). No builder code imported; judge labels re-derived as `"yes" in judge_raw.lower()` and checked against the
stored `label` on all 5,000 headline-relevant files.

**⁽¹⁾ Median held-out context (paper `:133`, `:593-594`): "23,919 tokens: 9 sessions, 8 of them whole".**
From the 400 rows of `v10/runs/20260924T122426_eval_held.json` (hash `7bc75f509aa0`, row order equals `held_qids`):
median `tokens` = **23,918.5** (the file's `median_tokens` field), median `n_sessions` = **9.0**, median `whole` = **8.0**.
Range of `tokens` 23,425–24,000; mean sessions 9.37, mean whole 8.34. Reproduced (see O1 on the .5).

**⁽²⁾ Headline arm on all 500 by type, and abstentions (paper `:347-357`, `:595-597`).**
From `judge_v10_o55t_sonnet5/*.json` and `judge_v10_o55t_opus55/*.json` (500 files each, filename = `question_id`,
`judge_model` correct, `question_type` and abstention flag equal the dataset's), joined to the dataset:

| type | n | Sonnet-5 | Opus-5.5 | paper |
|---|---|---|---|---|
| KU | 78 | 77 = 98.7 | 77 = 98.7 | 98.7 / 98.7 ✓ |
| MS | 133 | 124 = 93.2 | 127 = 95.5 | 93.2 / 95.5 ✓ |
| SSA | 56 | 55 = 98.2 | 55 = 98.2 | 98.2 / 98.2 ✓ |
| SSP | 30 | 26 = 86.7 | 27 = 90.0 | 86.7 / 90.0 ✓ |
| SSU | 70 | 70 = 100.0 | 70 = 100.0 | 100.0 / 100.0 ✓ |
| TR | 133 | 127 = 95.5 | 128 = 96.2 | 95.5 / 96.2 ✓ |
| abstentions | 30 | 27/30 | 30/30 | 27/30 / 30/30 ✓ |
| total | 500 | **479 = 95.80%** [93.66, 97.24], task-avg **95.39** | **484 = 96.80%** [94.87, 98.02], task-avg **96.44** | ✓ |

**⁽³⁾ Error attribution, headline arm, held-out (paper `:384-391`, `:598-601`).**
Held failures of `v10_o55t` under each judge, joined by `question_id` to the recall record's `all_sessions`, `n_ev` and
`abstention` fields:
- Sonnet: **15 = 10 incomplete-evidence + 3 complete-evidence + 2 abstentions** ✓.
- Opus: **13 = 10 + 3 + 0** ✓.
- Held non-abstention questions with incomplete evidence: **11**; failed under each judge: **10** (the same 10 under both:
  6d550036, ba358f49, bc149d6b, b46e15ed, gpt4_e061b84f, 6e984302, gpt4_8279ba03, 0977f2af, ceb54acb, d6233ab6); passed: 1 ✓.
- Held non-abstention questions with complete evidence: **365**; right under each judge: **362/365** ✓ (see O2 for which 3).
- Held abstentions: 24; failed under Sonnet: 88432d0a_abs, gpt4_93159ced_abs; under Opus: none ✓.

---

## What was attacked and held up

**Every held-out and all-500 number** (`recompute_held.out`). All ten judge directories: 500 files each, filename =
`question_id`, `judge_model` as labelled, `label == ('yes' in judge_raw.lower())` on every file, `question_type` and
abstention flag equal the dataset's. Held table (5 arms × 2 judges), Wilson intervals (own implementation, z = 1.96),
task-averages (96.25 / 96.56 headline), every McNemar pair in §4.5 and §4.2 (own exact two-sided binomial), judge
agreement 396/400 (Sonnet-yes/Opus-no 1, Opus-yes/Sonnet-no 3), self-preference lifts +1.00 / +2.25 / +0.50 / +1.50 /
+1.25, all-500 headline 479 / 484 with both intervals containing 95.60 and 94.87 and neither containing 93.6, oracle +
Opus 4.6 on all 500 = 475 / 483 (95.0 / 96.6, §1), v9 on all 500 = 393 / 401 (78.6 / 80.2, §1). Every figure equals the
paper, `COMPARISON_held_readers.md` and `REVIEW_v10_held_readers.md`.

**Dev block** (`recompute_dev_stats.out`). All twelve dev accuracies and intervals in §4.2 and §4.2's "Opus 5.5" line
(v10_o55t 94 / 97, oracle_o55t 97 / 98), every McNemar pair, and the "86–87 of 92" and "seven missed under both judges,
five also missed by full" counts. The full-history arm is judged on exactly the 100 dev questions.

**Retrieval recall and sweeps.** Held 365/376 = 0.9707 and 24 unscorable rows; `v9_by_type` strict MS 0.351, TR 0.510,
KU 0.667, SSU 0.902, SSP 0.542, SSA 0.889 (paper's 0.35 / 0.51 / 0.67 / 0.54–0.90); v10 by type 0.982 / 0.969 / 0.978 /
0.958 / 1.000 / 0.951; `tm_all_sessions_strict` 0.955; v9 strict/lenient 0.593 / 0.670. Stages: four stages at budget
16,000; stage-1 winner bm25+dense 0.936 (only maximum); stage-2 winner `second_weight` 1.0 × `w_dense` 2.0 = 0.947 (only
maximum); stage-3 three-way tie at 0.947 → `whole_top` 99, `window` 2; stage-4 tie → time off; curve 0.798 / 0.894 /
0.947 / 0.968 / 0.979 at 8.5k / 12k / 16k / 24k / 32k → 24k is the smallest ≥ 0.95; BM25 alone at 24k = 0.957
(`20260924T104017_sweep_dev.json`); random control 0.085. Frozen config on dev 0.968, hash `7bc75f509aa0` equals
`frozen_v10.json` and the held record. Every §2.5 and §4.1 statement checks.

**Dataset and protocol facts.** sha256 of `longmemeval_s.json` = `d6f21ea9d60a0d56…` (equals `split.json` and
`emb/meta.json`); `split.json` sha256 prefix `a0bd99c6afe13c17`; dev 100 / held 400, disjoint, `*_qids` consistent;
types 133 / 133 / 78 / 70 / 56 / 30 with 30 abstentions; sessions per haystack 38–62, mean 47.73; 896 `has_answer` turns.
Official scripts: reader `temperature: 0`, CoT `max_tokens` 800 (`run_generation.py:342, 366-367`); judge
`gpt-4o-2024-08-06`, `temperature: 0`, `max_tokens: 10`, `label = 'yes' in eval_response.lower()` (`evaluate_qa.py:14, 108-113`).
Judge controls: 4/4 pilot "Yes", 2/2 deliberately wrong "No" including the abstention (`judge_pilot/`, `judge_control/`).
The v9 review adjudicated 12 disagreements, Opus 12/12, 5 marginal (`REVIEW_fable.md:100-102, 231`).

**Harness statistics.** v10_o55t held thinking tokens median 114, max 1,727, min 18, a thinking block in 400/400, model
`claude-opus-5-5` in 400/400; oracle_o55t median 113.5, max 3,817. Visible output > 800 tokens: v10_o55t 23 (21 / 21 judged
correct), v10 26 (18 / 20), oracle_o55t 19 (16 / 19). Reader input medians: v9 8,549 (all 500); v10 27,022 dev; full
126,710 dev; oracle 7,574 dev. v10 and v10_o55t held prompts byte-identical on 400/400.

**Calibration** (`calib2.out`). Published files sha256 prefixes `af0727c3f5a52001` (Haiku full context, 313/500 = 62.6)
and `4aafec663a9457a0` (Honcho + Haiku, 452/500 = 90.4), 500 `detailed_results` each with `passed` labels. Our four
`calib/judge_*` directories re-scored: offsets −2.2 / −2.0 / −0.4 / −0.8, agreement 95.0 / 95.6 / 97.2 / 97.2, kappa
0.739 / 0.768 / 0.940 / 0.940, discordant counts 7/18, 6/16, 6/8, 5/9 — every cell equals `CALIBRATION.md` and §5.
Sonnet's Honcho deficit sits in SSP (18/30 vs 27/30, −30 points; MS 0.0), as §5 says.

**v9 autopsy claims** (`autopsy_calib.out`). "When all the evidence reached v9's context, v9 and the oracle arm tied"
(`paper.md:62-63`): on the 279 non-abstention questions where v9's strict matcher found every evidence session, v9 scores
270 / 272 and oracle 271 / 273 (Sonnet / Opus), paired 3 vs 4 and 4 vs 5, p = 1. Holds.

**Method against code** (`v10.py`, `embed_turns.py`, `meta.json`, `frozen_v10.json`). BM25 k1 = 1.2, b = 0.75 over every
turn; dense cosine on user turns only (`roles: ["user"]`), bge-base-en-v1.5 @ `a5beb1e3…`, CLS + L2, max_length 512, query
prefix on the question only; RRF k = 60 with `w_bm25` 1.0 / `w_dense` 2.0; session score = best + 1.0 × second-best;
tie-break by content hash (`rrf`) and by session hash (`order`); whole sessions first (`whole_top` 99), else ±2 turns
around the two best turns; budget 24,000 counted on `json.dumps(turn, ensure_ascii=False)` + 1 per turn + 16 per session
header; render chronological with dates; no LLM call, no type routing; `view()` asserts its key set. §2.1–§2.4 describe
the code, with the two exceptions in F2 and O4. `frozen_v10.json` is read-only (`-r--r--r--`).

**Placement and competitor figures** against `COMPETITORS.md` and its Sources. Chronos High 95.60 / claude-opus-4-6 /
S / no code / judge unnamed (`:67`, S18 `:511-513`); Agent Zero 95.60 / gpt-5.5 / "500 questions" / no code / "the official
LLM judge" (`:74`, S20 `:521-522`); Mastra 93.6 [D] (468/500) / 94.87 / gpt-4o / S / code public (`:39`, S9 `:467-473`); TAM
92.25 (369/400) / gpt-5 / S-cleaned 400 held-out / gpt-4o-2024-08-06 (`:185`, S25 `:541-544`); Supermemory "~99%" 8-variant
ensemble, OMEGA 95.4 task-weighted not opened, agentmemory 96.20 on the oracle file with `question_type` hints
(`:407-411`, `:182`, S28 `:553-554`); paper baseline 60.6 / 64.0 (S1 `:424`); cleaned re-release 2025/09 (S2 `:430`);
"23k to 100k tokens" (`:25`). The intervals of all four of our figures contain 95.60; "95.4–96.6 task-averaged" spans
95.39–96.56. All as the paper states, and all within the accepted Claim 3 wording.

**Accepted wording** (`REVIEW_v10_held_readers.md:276-303`, `REVIEW_v10_recall.md:277-283`). The abstract, §4.3, §5 and §6
follow Claims 1–3 clause by clause, including "one question", "lack a named judge and public code", "crosses judge and
dataset variant", the four harness deviations, "median 114 thinking tokens", the F8 oracle-dates caveat (`:379-380`), the
F3 "add no points" caveat (`:417`), the F1 500-question ranking (§6 ranks the 500 figure and files the 400 with TAM), and
the recall review's F2 budget framing (§7.1 gives the 8.5k comparison and 3× budget). The only departures from accepted
wording are F4 (the `has_answer` qualifier) and the title (F8).

**§1.1 and the record.** The Zenodo record exists, is Thomas's, is dated 2026-08-05, and states both figures the
subsection withdraws or supersedes.

---

## Wording I would accept for the changed passages

- **Abstract, retrieval bullet:** "v10 puts every `has_answer`-labelled evidence session into the context for 97.1% of
  held-out questions (96.0% when every official `answer_session_ids` session is required)."
- **Abstract, last paragraph:** "Adversarial second readers reproduced every held-out count, the retrieval-recall figures
  and the numbers first computed for this report from the raw files."
- **§2.2:** "Encoder: BAAI/bge-base-en-v1.5 (revision `a5beb1e3`), run once on a Quadro K2200; a second reader recomputed a
  sample on CPU and matched the stored rows at cosine 1.0000."
- **§3.3:** "The prompt comes from the official LongMemEval prompt builder, using its plain-history chain-of-thought
  template (`merge_key_expansion_into_value = none`) and giving the dataset's question date as `Current Date:`. The v9
  arm, whose context contains extracted facts, used the official history-plus-facts (`merge`) variant, which differs by one
  clause; v10, oracle and full history share the plain-history variant."
- **§1:** "only 46% of the time by the lenient verbatim matcher (35% strict; §4.1 reports strict)."
- **§4.6 dev paragraph:** as given under F6.
- **Reflection:** "one question above the best published claims, which name neither judge nor code".
- **References:** Sen, S., Lumer, E., Gulati, A., & Subbiah, V. K. (2026). Chronos: Temporal-aware conversational agents
  with structured event retrieval for long-term memory. arXiv:2603.16862v1. (PricewaterhouseCoopers.) — Wu, M., & Zhu, P.
  (2026). Agent Zero Memory: Provenance-aware long-term memory for LLM agents. arXiv:2608.29606v1. (Zero Labs; confirm
  author order against the arXiv record at deposit.)

## Pushback

Every finding above is arguable. F1 and F2 are the two I would defend hardest: both are verified, quoted contradictions
between the method section and the files that ran. F8 is a recommendation. If a finding is wrong, cite the path:line I
should have read and it travels with the review.
