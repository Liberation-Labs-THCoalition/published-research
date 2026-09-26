# Mnemosyne v10 on LongMemEval_S-cleaned: results so far

2026-09-24, Nexus. Held-out reader results added 2026-09-25.
- **Retrieval recall:** second-read, verdict CONFIRMED WITH CAVEATS (`REVIEW_v10_recall.md`). The wording
  below is the reviewer's.
- **Reader results, held-out:** second-read on 2026-09-25 by a Fable 5.1 agent (`REVIEW_v10_held_readers.md`).
  - Verdict: WARN (Agni) / CONFIRMED WITH CAVEATS.
  - Every count reproduced from the raw files.
  - The fixes it required were to wording and disclosure. The wording below is the reviewer's.
- **Reader results, dev:** unchanged from 09-24 and not second-read on their own. The review's 500-question
  recount covers only the headline arm's dev answers.

## Retrieval recall (zero reader tokens)

> On LongMemEval_S-cleaned, v10 puts every `has_answer`-labelled evidence session into a ~24k-token
> context (o200k; up to 25.3k as rendered) on **97.1% of 376 held-out questions** [94.8, 98.4]. That is
> **96.0%** when every official `answer_session_ids` session is required. On dev (tuned), the figures
> are 96.8% and 95.7%. v9's ~7k-token context covers 59–67% of the same questions by a verbatim-matcher
> lower bound. v10 scores 95.5% through that same matcher, and 79.8% on dev at v9's 8.5k budget.

| | dev (94 scored) | held-out (376 scored) |
|---|---|---|
| all `has_answer` sessions in context | 0.968 | **0.971** [0.948, 0.984] |
| all official `answer_session_ids` sessions (≥1 turn) | 0.957 | **0.960** [0.935, 0.976] |
| v10 through v9's verbatim matcher | — | 0.955 |
| v9 (verbatim matcher, lower bound): strict / lenient | 0.596 / 0.670 | 0.593 / 0.670 |
| frozen v10 config at v9's 8.5k budget | 0.798 | not run (it would be an unfrozen held-out peek) |

Held-out recall by type, v10 against v9 strict:
- multi-session 0.97 vs 0.35;
- temporal 0.95 vs 0.51;
- knowledge-update 0.98 vs 0.67;
- single-session types 0.96–1.00 vs 0.54–0.90.

**What the gain is made of (review F2).** It is mostly two plain choices:
- give the reader **whole sessions** instead of isolated passages;
- use **a bigger budget**: 24k against v9's ~7k.

Keyword search alone at 24k already reaches 0.957 on dev. The dense channel and fusion add about one
point, and the time channel added nothing. At v9's own budget, v10 gets 0.798 against v9's 0.60–0.67.
So whole-session selection is better at any size, and most of the rest is budget.

## Reader results, held-out (the 400 pre-registered questions)

The held-out set is blocks 2–5 of the frozen split, and nothing about the reader was tuned.
- **Prompts:** the frozen config's selections. The review matched all 400 against the held-out recall
  record. They were built with the official prompt builder.
- **Judging:** the official per-type answer-check prompts and the official "yes" rule.

| held-out (n = 400) | Sonnet-5 judge | Opus-5.5 judge |
|---|---|---|
| **v10 + Opus 5.5, thinking (headline)** | **385 = 96.25%** [93.9, 97.7] | **387 = 96.75%** [94.5, 98.1] |
| v10 + Opus 4.6, no thinking | 369 = 92.25% [89.2, 94.5] | 378 = 94.50% [91.8, 96.3] |
| oracle + Opus 5.5, thinking | 392 = 98.00% [96.1, 99.0] | 397 = 99.25% [97.8, 99.7] |
| oracle + Opus 4.6, no thinking | 381 = 95.25% | 387 = 96.75% |
| v9 + Opus 4.6 | 316 = 79.00% | 320 = 80.00% |
| **all 500 (dev + held), headline** | **479 = 95.80%** [93.7, 97.2] | **484 = 96.80%** [94.9, 98.0] |

Brackets are 95% Wilson intervals. The per-type breakdown and every paired comparison are in
`baseline/COMPARISON_held_readers.md`.

**The three claims, in the reviewer's wording.**

> **Accuracy.** On LongMemEval_S-cleaned, Mnemosyne v10 retrieval (frozen config 7bc75f509aa0, ~24k-token
> contexts) with reader claude-opus-5-5, thinking enabled (median 114 thinking tokens), scores
> **385/400 = 96.2%** under a claude-sonnet-5 judge and **387/400 = 96.8%** under a claude-opus-5-5 judge on
> the 400 pre-registered held-out questions (95% Wilson [93.9, 97.7] and [94.5, 98.1]). On all 500 questions,
> 100 of which were the retrieval-tuning split, it scores **479/500 = 95.8%** and **484/500 = 96.8%**. The same
> prompts read by claude-opus-4-6 without thinking score 369/400 = 92.2% and 378/400 = 94.5% (paired McNemar
> p = 0.0009 and 0.012). Evidence-only prompts with Opus 5.5 and thinking score 392/400 = 98.0% and
> 397/400 = 99.2%, using the oracle file's own question dates. Judging used the official per-type
> answer-check prompts and the official "yes" rule. Deviations from the official harness:
> - Claude judges instead of gpt-4o-2024-08-06;
> - no temperature control;
> - no 800-token output cap (23 of 400 headline answers exceed it);
> - a `claude -p` harness that adds a short system prompt and Claude Code system-reminders (today's real
>   date, model identity, environment) to every call, identically across arms.

> **Judges.** On 1,000 published LongMemEval_S answers (two Plastic Labs runs, gpt-4o-2024-08-06 labels), our
> claude-sonnet-5 and claude-opus-5-5 judges agreed with the official judge on 95–97% of questions (Cohen's
> kappa 0.74–0.94) and passed 0.4–2.2 points fewer answers. The difference is significant at the 5% level on
> one of the two sets and not on the other. For the Sonnet judge it is concentrated in
> single-session-preference. Those answers are short Haiku 4.5 answers, not chain-of-thought, so the offset
> indicates direction, not a correction to apply.

> **Placement.** Against every LongMemEval_S result in our survey that we could trace to a primary source,
> the point estimates above (96.2 / 96.8 on the held-out 400 and 95.8 / 96.8 on all 500) are the highest. On
> all 500, the Sonnet-judge figure exceeds the top published claims (95.60, Chronos and Agent Zero) by one
> question. The 95% intervals of all four figures include 95.60, and both of those claims lack a named judge
> and public code. The best result with a named official judge and public code is Mastra's 93.6 micro /
> 94.87 task-average (original S). Ours is 95.4–96.6 task-averaged on S-cleaned under Claude judges, so that
> comparison crosses judge and dataset variant. Claims we could not verify (Supermemory "~99%", OMEGA 95.4
> task-weighted, agentmemory 96.2 on the oracle file) are excluded.

**What the paired comparisons show, held-out.**
- **v10 against v9, same reader (Opus 4.6):** 65 vs 12 (p = 6e-10) and 64 vs 6 (p = 2e-13).
- **The reader upgrade is real:** Opus 5.5 with thinking fixes 19 questions and breaks 3 against Opus 4.6
  without it (Sonnet judge); the Opus judge counts 10 and 1.
- **Retrieval still costs something against the evidence-only ceiling:** 5 vs 12 (p = 0.14) and 1 vs 11
  (p = 0.006).
- **Multi-session**, v9's weakest type at 56.6 / 57.5, is 94.3 / 95.3 under the headline reader, on 106
  questions. The survey's best multi-session score on S-500 is 91.7 (Chronos Low). That comparison is the
  builder's, not the review's, and it crosses subset, judge and dataset variant.

**Caveats from the review.**
- **F1.** The 400 are a subset, and COMPETITORS.md's own rule files subsets in Table 1D. Rank the
  500-question figure.
- **F2, F7, F9.** The harness deviations listed in the Accuracy claim. The system-reminder text is in
  `PROTOCOL.md`. Thinking was enabled but light: median 114 thinking tokens, max 1,727. Of the 23 answers over
  the official 800-token cap, 21 were judged correct. The official harness would have cut them mid-reasoning.
- **F3.** The calibration shows direction only. Add no points to anything.
- **F8.** The oracle arm uses the oracle file's own question dates. It is comparable with published oracle
  numbers, not a strict ceiling for v10.
- **A correction to the review.** Its F1 says both 500-question intervals contain 93.6. Neither does:
  [93.665, 97.237] and [94.866, 98.021] (statsmodels, Wilson). All four intervals do contain 94.87 and 95.60.
  No claim depends on it.

## Reader results, dev only (100 questions = block 1)

The reader for every arm is Opus 4.6, with the official prompt builder and the official answer-check
prompts under two Claude judges.

| arm | reader tokens | Sonnet-5 judge | Opus-5.5 judge |
|---|---|---|---|
| v9 | ~8.5k | 77 [67.8, 84.2] | 81 [72.2, 87.5] |
| **v10** | **~27.5k** | **91** [83.8, 95.2] | **92** [85.0, 95.9] |
| full raw history | ~127k | 92 [85.0, 95.9] | 94 [87.5, 97.2] |
| evidence only (oracle) | ~7k | 94 [87.5, 97.2] | 96 [90.2, 98.4] |

Paired, discordant counts (McNemar exact):

| comparison | Sonnet-5 judge | Opus-5.5 judge |
|---|---|---|
| v10 vs v9 | 17 vs 3, p = 0.0026 | 15 vs 4, p = 0.019 |
| v10 vs full | 1 vs 2, p = 1 | 1 vs 3, p = 0.63 |
| v10 vs oracle | 0 vs 3, p = 0.25 | 1 vs 5, p = 0.22 |

The full table is in `baseline/COMPARISON_v10_dev.md`. Its v9, oracle and full rows are byte-identical to
last night's `COMPARISON_block1.md`.

**Where v10's misses come from.**
- With all evidence in context, the reader is right on 86–87 of 92 questions.
- 7 questions are missed under both judges. 5 of them are also missed by the full-history reader: hard
  counting and ordering questions.
- 1 is a retrieval miss: 2 of 3 magazine-subscription sessions were found.
- 1 had all 4 charity sessions in context and still summed wrong.

## The reader handicap (Thomas's question: are we limiting ourselves?)

Every arm above ran on Opus 4.6 with extended thinking off. Neither is required by the test (see
`SELF_LIMITS_AUDIT.md`). On the same 100 dev questions and prompts, **v10 with Opus 5.5 and thinking
scores 94 / 97**, against 91 / 92, and the evidence-only ceiling moves to 97 / 98. The held-out run
keeps the old reader, so the retrieval comparison stays controlled. The unhandicapped held-out run is
the headline candidate.

## Caveats

- **Dataset.** This is LongMemEval_S-**cleaned**, not the original S.
- **Dev reader numbers come from the tuning split.** The tuning targeted recall, not answers.
  Held-out recall came in at or above dev, but held-out *accuracy* has not been measured. *(Superseded on
  2026-09-25: held-out accuracy is now measured and second-read. See the held-out section.)*
- **Sample size.** n = 100 per arm on dev, so the intervals are ±6–8 points. "Level with full history"
  means not distinguishable on this sample.
- **Judges.** Claude judges, not the official gpt-4o, as disclosed in `PROTOCOL.md`.
- **Budget accounting (F3).** The budget counts `json.dumps(ensure_ascii=False)`. The official builder
  escapes non-ASCII, so 211 of 500 histories render above 24k, with a maximum of 25.2k. A v10.1 should
  count the rendered form.
- **Order dependence (F6).** Turns with identical content still tie-break by array order. That changed
  the selection in 8 of 1,000 shuffled runs and no evidence verdict.
- **Held-out log.** `heldout_log.jsonl` now records attempts as well as completions. Two crashed attempts
  and the reviewer's in-memory reproduction were added after the fact, with a mark saying so.

## Provenance

- **Split:** `split.json`, read-only, sha256 prefix a0bd99c6afe13c17.
- **Frozen config:** `frozen_v10.json`, read-only, hash 7bc75f509aa0, budget 24000.
- **Selection record:** `runs/stages_*.json`.
- **Held-out recall run:** `runs/20260924T122426_eval_held.json`.
- **Embeddings:** bge-base-en-v1.5 @ a5beb1e3, user turns only; see `emb/meta.json`.
  *(Added 2026-09-25, from the paper review's F2.)* They were computed on GPU1, a Quadro K2200 (`emb/embed.log`),
  not on CPU as `DESIGN.md` planned. The design note's assistant-turn embeddings were not made either. A CPU recompute
  of a sample matched at cosine 1.0000 (`REVIEW_v10_recall.md`).
- **Prompts and answers:** `baseline/v10/`. Judgements: `judge_v10_{sonnet5,opus55}/`.

## Next

1. **Held-out reader run.** About 12M tokens with judges. Needs Thomas's go. *(Done 09-24/25 with both readers,
   and second-read.)*
   - **Follow-up, harness:** find out whether `claude -p` can suppress the Claude Code reminders (above all
     the real date) and the `-` line. If it can't, keep disclosing them.
   - **Follow-up, chart:** add v10 to COMPETITORS.md. The 500-question figure goes with the S-500 rows, the
     held-out 400 goes in Table 1D, and the per-type figures go in Table 2.
2. **Full history on held-out.** About 51M tokens. Only needed to publish "matches full history" on
   held-out.
3. **v10.1 fixes:**
   - count the budget on the rendered form;
   - a position-free tiebreak for duplicate turns;
   - a retrieval fix for aggregation questions, whose last misses are sessions that mention the item
     in passing.
4. **Then blind LoCoMo with v10** (Thomas, 09-24), and the pre-attentive layer running on v10's
   retrieval.
