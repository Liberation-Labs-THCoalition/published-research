# Adversarial second read — LongMemEval v2 (Mnemosyne v9 contexts, reader claude-opus-4-6)

Reviewer: Fable 5.1, 2026-09-23/24. Read-only on the pipeline; scratch under `review/`.
Judge probes used: 20 of the allowed 20 `claude -p` calls (`review/judge_probe/`). No new reader runs.

## Verdict: PASS WITH CAVEATS

The number is real: no answer leakage by construction, joins are exact for all 500 rows, the reader
provenance is clean, the judge and metric code are the official ones, and my independent recomputation
matches `score.py` to the digit. It is **not yet publishable as written**, because (1) the provenance
story in PROTOCOL.md is wrong for 150 of the 500 contexts and omits that the run's own quality gate was
raised and then removed to reach 500; (2) the system measured is a partially degraded v9 (embedding
failures on ~9% of questions, gap layer inert); (3) the two judges differ by 1.6 pp overall and 16.7 pp on
preference questions, and every disagreement I adjudicated went against the "primary" judge; and (4) there
is no same-reader baseline, so nothing comparative can be claimed. Fix the text, pick the judge honestly,
and publish with the caveats paragraph below.

## The numbers

| | Sonnet-5 judge | Opus-5.5 judge |
|---|---|---|
| Overall accuracy (500, official) | **0.786** (393/500) | **0.802** (401/500) |
| Task-averaged accuracy | 0.8124 | 0.8423 |
| Abstention accuracy (30) | 0.800 (24/30) | 0.867 (26/30) |
| Non-abstention accuracy (470) | 0.785 (369/470) | 0.798 (375/470) |
| Wilson 95% CI on overall | [0.748, 0.820] | [0.765, 0.835] |
| single-session-user (70) | 0.9571 | 0.9571 |
| single-session-preference (30) | 0.6667 | 0.8333 |
| single-session-assistant (56) | 1.0000 | 0.9821 |
| multi-session (133) | 0.5789 | 0.6015 |
| temporal-reasoning (133) | 0.7744 | 0.7820 |
| knowledge-update (78) | 0.8974 | 0.8974 |

`score.py` output (RESULTS_v2.md) and my recomputation from the raw `judge_raw` strings
(`review/independent_metrics.txt`) are identical on every line. Per-type figures include the abstention
items of that type, as the official `print_qa_metrics.py` does. Judge agreement 488/500 (97.6%),
Cohen's kappa 0.927. Reader input: median 8,549 tokens/question (p90 10,802) versus ~115k for the raw
LongMemEval_S history.

## Findings, ranked by severity

### F1 — HIGH — PROTOCOL.md's provenance for the contexts is false for 150/500, and the run's quality gate was raised then removed

PROTOCOL.md says `provenance/precompute_v9.py` "produced `frontier_v9_contexts.json` on 2026-09-08".
It produced idx 0–349 only. Reconstructed from `/tmp/precompute_v9.log`, `/tmp/precompute_v9_rest.log`,
file mtimes, and Nexus's own Sep 8 session transcript
(`~/.claude/projects/-home-admin/cb1ef511-a020-4299-a3f9-4e6d79164827.jsonl`, tool_use commands only):

- 05:00:36 local: `/tmp/precompute_v9.py` edited — transcript 2026-09-08T12:00:36Z: "update the abort
  threshold from 10% to 25% for intermittent timeouts". The provenance copy is this post-edit version
  (docstring line 5 still says ">10%", code lines 65–67 and 75 check 0.25; final print line 76 still says
  "exceeded 10%"). Its sha256 `c3c120ed…` matches `/tmp/precompute_v9.py` (mtime 05:00:36).
- 05:00→11:14 local: that run **aborted at 350/500** — last line of `/tmp/precompute_v9.log`:
  `ABORT: embedding failure rate 26.3% > 25%`.
- 12:09:45 local: `/tmp/precompute_v9_remainder.py` (sha256 `d36695f99323d7d02c44e42bff22d41ade47782d0f5649cc49e0519870108287`,
  **not in `provenance/`**, not mentioned in PROTOCOL.md) loaded the 350, computed idx 350–499 with the
  same `process_question` flags, and appended into the same file (`:9-10`, `:32-42`). It has **no abort
  threshold** and records only `embed_failures` (no `embed_hits`, `embed_convergent`, `graph_boosted`,
  `gaps_detected`, `gap_fills_added`, `domain_profile_chars`) — hence the two key schemas in the file
  (15 keys for idx 0–349, 9 keys for idx 350–499; `review/contexts_health.txt`). Its log ends
  `Saved 500 total (22 embed failures this batch)` at 15:14:09, matching the contexts file mtime.
- By the first script's own metric, the cumulative failure rate over all 500 is **36%** (180 failed
  embedding calls / 500 questions) — the run would have aborted under either the 10% or 25% gate.

This is not leakage and does not change any label. It is a false statement of method plus an undisclosed
removal of the experiment's own guard. Both must be corrected in PROTOCOL.md before anything is published
(copy the remainder script into `provenance/`, record both hashes, both logs, and the threshold edit).

### F2 — HIGH (comparability) — no same-reader baseline exists

Nothing in this directory measures `claude-opus-4-6` reading the raw ~115k-token history or the oracle
evidence sessions with the same template and judges. The 78.6–80.2% therefore measures
Mnemosyne-retrieval + Opus-4.6-reading jointly. The official paper's GPT-4o full-context number (60.6%)
is from a 2024 reader; a 2026 reader may score well above that on the raw history unaided. Until the
baseline exists, the only supportable statement is "X% with a median 8.5k-token Mnemosyne context",
not "Mnemosyne improves accuracy". (A baseline is ~57M reader input tokens at full history; a
stratified 100-question subset would be ~11M.)

### F3 — MEDIUM — the system measured is a degraded v9, not the designed v9

From `frontier_v9_contexts.json` (`review/contexts_health.txt`):

- 46/500 questions had ≥1 failed embedding call (`embed_failures` ∈ {1,2,4,8,12,13,20}); 180 failed calls
  total. In idx 0–349, **15 questions have `embed_hits == 0`** — the question embedding itself failed
  (`benchmark_longmemeval_v8…py:145-147` returns `([], 1)`), so no embedding fusion ran and the context is
  TF-IDF + other layers only. For idx 350–499 this count is unknowable (field not recorded); 22 of those
  150 had failures per the remainder log.
- Gap detection (`v9c`) added **0** fills across all 500 (`gap_fills_added > 0` in 0 rows): the layer is
  inert as run.
- Graph boost touched 48 questions; domain profiles appear in 96 contexts; `[Structured Facts]` in 491;
  `[Event History]` and `[Retrieved Conversations]` in all 500.

The description "Mnemosyne v9 full stack" overstates what was measured. Describe it as "v9 as run on
2026-09-08 (embedding retrieval degraded on 46/500 questions, gap layer contributed nothing)".

### F4 — MEDIUM — judge deviation and asymmetry; the primary judge is the strict outlier

- Official: `gpt-4o-2024-08-06`, temperature 0, max_tokens 10. Here: two Claude judges via `claude -p`,
  no temperature or max_tokens control (`judge.py:8-11` discloses this).
- The 12 disagreements (`review/disagreements.txt`): 10 are Opus-yes/Sonnet-no (5 single-session-preference,
  4 multi-session, 1 temporal-reasoning); 2 are Sonnet-yes/Opus-no (`1f2b8d4f`, `1903aded`). My
  adjudication under the official rubric text sided with **Opus-5.5 on 12/12**, 5 of them marginal
  (`0bc8ad93`, `75832dbd`, `75f70248`, `80ec1f4f_abs`, `gpt4_f2262a51`). Sonnet-5 does not apply the
  preference rubric's explicit leniency ("does not need to reflect all the points in the rubric").
- Sonnet-5's strictness is stable, not noise: re-judged, it reproduced 10/12 of its disagreement labels
  (flips: `0bc8ad93` No→Yes, `1f2b8d4f` Yes→No) and 6/6 random agreed labels (`review/judge_probe/summary.txt`).
- Both judges rejected `gpt4_a1b77f9c` (answer 9 weeks, gold 8) although the official temporal prompt
  says off-by-one errors in weeks are correct; on re-run Opus-5.5 flipped it to "Yes". Judge
  non-determinism is real but small: 3 flips in 20 probes, all on borderline items, 0/6 on random items
  (~±0.5 pp on the aggregate).
- Recommendation: report both judges, or Opus-5.5 as primary with Sonnet-5 as the conservative floor.
  Reporting Sonnet-5 alone as "the" number is defensible only if labelled as the strict end.

### F5 — LOW — retrieval conditions on the dataset's `question_type` label

`process_question` reads `item['question_type']` (`provenance/benchmark_longmemeval_v8.from_git_764b7e7.py:1407`)
and uses it once: `if with_graph and question_type != 'temporal-reasoning'` (`:1562`) — the graph layer is
switched off for temporal questions. `assemble_context_with_convergence` and `assemble_context_with_facts`
take the parameter but never use it. This is test-time use of a label a deployed system would not have.
Effect is small (graph boosted 48 questions) but must be disclosed.

### F6 — LOW — reader-side deviations from the official generation script

- No output cap: 30/500 answers exceed the official 800-token CoT `max_tokens` (16 exceed 1,000; max
  1,406); official would have truncated them. Of those 30, 17 (Sonnet) / 23 (Opus) were judged correct.
- System prompt "You are a helpful assistant." (`answer_one.sh:16`); official sends only a user message.
- No temperature control (official temperature 0). Reader output is therefore not exactly reproducible.

### F7 — LOW — one of four modules that ran cannot be hash-verified, but its imported names are unchanged

`structured_memory.py`'s Sep-11 `.pyc` was overwritten on 2026-09-23 14:25 and the file is absent from
`764b7e7`. The six names the v8 module imports from it (`classify_hall`, `FACT_PATTERNS`, `EVENT_PATTERNS`,
`PREFERENCE_PATTERNS`, `STATE_PATTERNS`, `CHARACTER_PATTERNS`) are AST-identical between commit
`e5ff9a4` (2026-08-27) and today's working tree (`review/sm_e5ff9a4.py`, `review/sm_wt.py`), so no
behavioural uncertainty follows. Its DB code (`sqlite3.connect` on `~/memory-data/memory.db`) is in
functions the benchmark never calls.

### F8 — INFO — one label rests on a disclaimed guess; memorisation signal inconclusive

`1903aded` (single-session-assistant, gold "Transcriptionist"): the reader states it cannot see the
assistant's list, then offers a "typical" 10-item list with **Transcriptionist bolded at #7**, then
declines to answer. Sonnet-5 says Yes (string present), Opus-5.5 says No. The prompt does not contain
"Transcriptionist"; the real assistant turn (15 items, never shown) shares 7/10 items with the guess and
has Transcriptionist at #7 and Social media manager at #8, as the guess does
(`review/contamination_1903aded.txt`). Suggestive, but generic enough to be genre prior. The systematic
checks found no contamination: of 447 reader "quotes" not verbatim in their own prompt, **0** appear
verbatim in that question's raw haystack (`review/contamination_test.txt`); every both-judges-correct
answer whose gold words are absent from the prompt is a multi-session arithmetic result ($12, $140, 190…),
never a retrievable string (`review/correct_without_evidence.txt`). Single-session golds were present in
the prompt for 57/59 (user) and 51/53 (assistant) correct answers.

### F9 — INFO — PROTOCOL.md corrections required

1. Contexts: two scripts, two runs, threshold edit, remainder script hash, both logs (F1).
2. "Size match, not hash match" → replace with the bytecode proof below (stronger than a hash of the
   file: it proves the code that was compiled, not the bytes on disk).
3. Dataset path: `precompute_v9.py:12` reads `~/benchmarks/longmemeval/data/longmemeval_s.json`; it is
   byte-identical (sha256 `d6f21ea9…`) to `/mnt/data1/datasets/longmemeval/longmemeval_s.json`. Say so.
4. Add F3, F5, F6 disclosures.

## What passed, and how each check could have failed

**Leakage by construction — clean.** `process_question` reads only `question`, `question_type`,
`haystack_sessions`, `haystack_dates` (`:1406-1409`). Zero references to `answer`, `answer_session_ids`,
`has_answer`, `gold`, `question_id`, `question_date` in lines 1369–1704 or any helper it calls; the
module's only uses of `item['answer']` are in `main()` for printing/F1/storage (`:2062, :2116, :2189`).
The dataset does carry `has_answer: true` on 896 turns (10,960 turns have the key) — a real leak path —
but `extract_passages` copies only `role` and `content` (`review/lmp_ran.py:296-297`), so it cannot reach
the text. No helper reads files or caches; the only I/O is the embedding POST (text → vector). Session
ids like `answer_280352e9` never enter passages (not passed to `extract_passages`). Prompt scan over all
500: `has_answer`, `answer_[0-9a-f]{8}`, `_abs`, `question_id`, `noans`, `Correct Answer`, `question_type`
and the six type names each occur in **0** prompts; "gold" occurs in 147 prompts as ordinary words
(golden, Golden Retriever, marigolds, Goldman). Gold text appears verbatim in 230/500 contexts, which is
retrieval of haystack text, consistent with the gold-in-prompt rate being 57/59 for single-session-user.
How it could have failed: any `json.dumps(turn)`, any `item[...]` beyond the four fields, a per-question
cache file, or a session-id in passage text would have shown up in the greps or the prompt scan.

**Provenance of the retrieval code — proven for 3 of 4 modules, stronger than PROTOCOL claims.**
The Sep-11 17:41 `__pycache__/*.cpython-312.pyc` headers record the source mtime and size at import time;
I compiled the candidate sources with the same Python 3.12.3 and compared the marshalled code objects:
- `benchmark_longmemeval_v8.pyc`: src mtime 2026-09-08 01:12:30, size 99,835 — **marshal byte-identical**
  to `git show 764b7e7:benchmark_longmemeval_v8.py` (which equals the provenance copy, sha256 `186ee4af…`).
  The file was not modified between 01:12 Sep 8 and the import on Sep 11; both runs (05:00 and 12:09)
  fall inside that window.
- `event_ledger.pyc`: src mtime 2026-08-01, size 9,971 — byte-identical to the `764b7e7` blob.
- `longmemeval_precompute.pyc`: src mtime 2026-08-28, size 21,552 — differs from the `764b7e7` blob only
  in the module docstring (Cyrillic-homoglyph watermark stripped in the public commit) and is
  structurally identical to the current working-tree file. (This discrepancy shows the test discriminates.)
- `structured_memory.pyc`: overwritten Sep 23; see F7.
Note `764b7e7` is dated 2026-09-12, four days after the run, on `public-clean`/`origin/main`, not on
`main`; it is the only commit ever containing the v8 file. The bytecode match, not the commit, is the
evidence. Caveat: the chain relies on the kernel mtime not having been reset by a `cp -p`-style copy.

**Joins — 500/500 exact** (`review/joins_check.txt`). For every row: `idx` equals dataset position and
the v9 `idx`; prompt = official head + v9 `context` for the same `question_id` + `Current Date:
question_date` + `Question: question` + `Answer (step by step):`; `prompts_meta` type/abstention match the
dataset; answer file `idx` matches; both judgment files carry the same `question_id`, `question_type`
and abstention flag. `build_prompts.py` joins by `question_id` and asserts question-text equality; the
template is regex-extracted from the byte-identical official `run_generation.py` (`:22-25`).

**Reader provenance — clean** (`review/reader_provenance.txt`). 500/500 `claude-opus-4-6`; `num_turns` 1;
`thinking_blocks` 0 and `thinking_tokens` 0; one usage iteration; 0 web tool calls; 500 unique
`session_id`s; `cache_read_input_tokens` 0 everywhere (no prompt reuse across calls); prompt chars per
input token 3.16–4.48 (no injected system content); 0 duplicate answer texts; no answer under 40 chars;
no empty stderr; `logs/failures.log` never created. Shift test: 109 answers contain ≥25-char "quotes"
not verbatim in their own prompt; **0** of those quotes appear in any other prompt; 72 have ≥60%
longest-common-substring with their own prompt (markdown bolding and elision inside quotes). CLAUDE.md
and `.claude/` are absent at `/`, `/mnt`, `/mnt/data1`, `/mnt/data1/lme_v2`, `run_cwd` and `~/.claude/`.

**Judge fidelity — official code, clean labels.** `official/evaluate_qa.py`, `print_qa_metrics.py`,
`run_generation.py` are byte-identical to fresh downloads from `xiaowu0162/LongMemEval` `main`
(`review/fresh_*`; sha256 `ecce9c4c…`, `e9283933…`, `4f1eb3c6…`). `judge.py:59` uses the official
`'yes' in response.lower()`; the rule would mislabel a "No, … yes …" reply — none occurred: Sonnet-5
produced 5 distinct raw strings (`Yes`/`No`/`yes`/`Yes.`/`no`), Opus-5.5 five, the only non-bare reply
being `"The response says 5 and the correct answer is 3.\n\n**no**"` (`c4a1ceb8`, labelled False,
correctly). Recomputing every label from `judge_raw` matched the stored label 1000/1000. Abstention: the
30 `_abs` items used the abstention template (`judge.py:40`); reconstructed prompts for `0862e8bf_abs`
and `8a2466db` are shown in `review/odd_answers.txt`. Controls (`judge_control/`) show the judge saying
"No" to deliberately wrong answers, so it can fail. `score.py`'s only change to the official printer is
the model-name assert (`score.py:37-39`).

**Metric fidelity — identical** to `score.py` for both judges on all reported lines.

## Judge error estimate

- Random hand-grade, n = 30 (`review/handgrade_sample.txt`, seed 20260923): both judges agreed on all 30;
  I agree with them on 29. The exception is `gpt4_a1b77f9c` (9 weeks vs gold 8; official off-by-one
  allowance), where both judges are wrong in the strict direction. Shared-error rate ≈ 1/30 ≈ 3%
  (95% CI ~0.1–17%). No false positives found in 30 random + 12 disagreement items (FP rate < ~7% by rule
  of three; the 1 disclaimed-guess case `1903aded` is the only candidate).
- Disagreement set, n = 12 (2.4% of items): Opus-5.5 correct 12/12 (5 marginal); Sonnet-5 correct 0/12.
- Estimated total error: **Sonnet-5 ≈ 5% (≈2.4% unique + ≈3% shared), Opus-5.5 ≈ 3–4%**, both
  dominated by false negatives. Judge non-determinism ≈ ±0.5 pp on the aggregate (3/20 probe flips, all on
  borderline items, 0/6 on random items).
- Implication: the truth under the official rubric is more likely at or slightly above the Opus-5.5
  figure (≈80–82%) than at the Sonnet-5 figure. Sampling uncertainty (±3.5 pp) dominates judge error.

## Caveats paragraph (for publication)

> **Scope and comparability.** This is LongMemEval_S (500 questions; ~48 sessions, ≈115k tokens of
> raw history per question), scored with the official `print_qa_metrics.py` definitions: overall accuracy
> over all 500 items including the 30 abstention items, task-averaged accuracy over six question types,
> and abstention accuracy on the 30 `_abs` items. The reader was `claude-opus-4-6` (via `claude -p`, no
> temperature control, no output cap — 30/500 answers exceeded the official 800-token CoT cap), not the
> official `gpt-4o-2024-08-06` at temperature 0. Judging used the official answer-check prompts with two
> Claude judges, `claude-opus-5-5` (80.2% overall) and `claude-sonnet-5` (78.6%), not
> `gpt-4o-2024-08-06`; the judges agree on 97.6% of items (κ = 0.93) but differ by 16.7 pp on
> single-session-preference, and hand-adjudication of all disagreements favoured the Opus-5.5 labels, so
> Sonnet-5 should be read as the strict end. Sampling uncertainty is ±3.5 pp (Wilson 95%); judge
> non-determinism adds roughly ±0.5 pp. The reader saw a Mnemosyne v9 context (median 8.5k tokens:
> profile, structured facts, event ledger, 15 retrieved passages) in the official "merge"+CoT reading
> template, not the raw history or the paper's retrieval pipeline; the number therefore measures
> Mnemosyne retrieval and Opus-4.6 reading jointly. **No same-reader baseline was run** (Opus-4.6 on the
> full raw history or on the oracle evidence sessions), so this result does not show that Mnemosyne
> improves accuracy; it shows what accuracy was reached with a ~8.5k-token context. The paper's
> LongMemEval_S figures (GPT-4o reader and judge) are 60.6% full-context, 64.0% with Chain-of-Note,
> 87.0%/92.4% with oracle evidence, ChatGPT 57.7%, Coze 33.0%; the gap to those is largely a
> 2024-vs-2026 reader difference and is not evidence about the memory system. Vendor-reported
> LongMemEval_S numbers use other readers, judges and unaudited pipelines and are not directly comparable.
> The contexts were built on 2026-09-08 in two runs: the first aborted at 350/500 when its embedding
> failure gate (raised that morning from 10% to 25%) tripped; the remaining 150 were built by a second
> script with no gate. 46/500 questions had at least one embedding-call failure, and the gap-fill layer
> contributed nothing on any question, so the system measured is v9 as run, not v9 as designed.
> Retrieval also conditions on the dataset's `question_type` label (graph layer disabled for
> temporal-reasoning questions), which a deployed system would not have.

## Recommended before publishing

1. Rewrite PROTOCOL.md provenance per F1/F9; copy `/tmp/precompute_v9_remainder.py` and both `/tmp`
   logs into `provenance/` (they are in `/tmp` and will vanish on reboot).
2. Report both judges or Opus-5.5 primary; include the CI.
3. Run the same-reader baselines (full history and oracle) before any comparative sentence. Budget it.
4. Optional but cheap (46 reader calls): rebuild the 46 embedding-degraded contexts against a healthy
   embedding endpoint, re-answer and re-judge only those, and report the clean-v9 number alongside.
5. Keep the disclaimed-guess rule behaviour (`1903aded`) in mind when writing about single-session-assistant
   = 100%/98%: one of the 56 is a string match on a guess the reader itself withdrew.

## Scratch index (`review/`)

`joins_check.txt`, `reader_provenance.txt`, `shift_test.txt`, `contamination_test.txt`,
`contamination_1903aded.txt`, `correct_without_evidence.txt`, `contexts_health.txt`, `judge_raw_scan.txt`,
`independent_metrics.txt`, `labels.json`, `disagreements.txt`, `handgrade_sample.txt`, `odd_answers.txt`,
`judge_probe/` (20 calls, `summary.txt`), `fresh_*.py` / `fresh_README.md` (GitHub main copies),
`v8_ran.py` / `lmp_ran.py` / `el_ran.py` / `sm_e5ff9a4.py` / `sm_wt.py` (module versions compared),
`lme_full.html` / `zep_abs.html` (paper sources for baseline numbers).
