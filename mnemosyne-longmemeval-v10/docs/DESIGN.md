# Mnemosyne v10 retrieval: design and pre-registration

2026-09-24, Nexus. Written before any v10 number existed.

## Why v10

Last night's arms, same reader (Opus 4.6), same official prompts:
- v9 context (~8.5k tokens) scored 78.6–80.2.
- Evidence sessions only ("oracle", ~7k tokens) scored 95.0–96.6.
- The whole raw history (~115k tokens) scored 92–94 on block 1.

The autopsy puts the cause in retrieval, not reading. When all the evidence reached v9's context, the
arms tie. v9 gets all of it there on only 46% of multi-session questions. v10's single job is
**evidence recall at a token budget**. Reading stays identical: v10 hands the reader whole sessions,
or turn windows, through the same official prompt builder the oracle and full-history arms used.

## Inputs: an allow-list, enforced in code

The retriever sees a `view()` built field by field:
- the question text;
- the question date;
- for each haystack session, its date and its turns as (role, content).

Nothing else reaches it. It never sees:
- session IDs (evidence sessions are prefixed `answer_`);
- `has_answer`, `question_type`, `question_id` or the answer.

`view()` builds new objects and asserts the key set. This is the LoCoMo lesson: slicing the full
record put the answer key in front of the reader.

## Metrics: zero reader tokens

Per question:
- **Evidence session covered:** at least one of its `has_answer` turns is in the context. This is the
  same definition the v9 autopsy used.
- **All evidence turns included:** the stricter check.
- **Tokens used,** counted in o200k.

The headline is the share of non-abstention questions with **all** evidence sessions covered, by
type, at budgets of 8.5k (v9's size), 16k, 24k and 32k tokens. It is reported beside v9 on the same
questions. v9's figure comes from the autopsy's verbatim matcher, which is a lower bound for v9, so
v10 is also scored through that same matcher on its rendered text as a like-for-like check.

Questions with no `has_answer` turns (21) can't be scored on recall. They are counted and reported,
not dropped silently.

## Split: tuning never sees the test

The split lives in `split.json`, which is read-only (sha256 prefix a0bd99c6afe13c17).
- **dev** is block 1 of `baseline/full_order.json` (100 questions). We have already examined block
  1 in detail, so it can't serve as a clean test set.
- **held** is blocks 2–5 (400 questions).

Rules:
- Every parameter is chosen on dev.
- held-out recall can only be computed for a **frozen** config, meaning a read-only config file.
- Every held-out evaluation appends to `heldout_log.jsonl` with the config's hash. Peeking is
  possible but never invisible.

## Disclosed prior knowledge

The design choices below come from two sources:
- **The v9 autopsy, which covered all 500 questions.** That includes the held-out 400.
- **The published systems in `COMPETITORS.md`:**
  - retrieve whole sessions or neighbouring turns;
  - fuse several search channels;
  - use a larger budget.

These choices are design-level, not tuned parameters. The claim v10 can make is limited by this: it
is a retrieval design motivated by an analysis that saw every question, tuned on 100 and tested on
400.

## v10.0

- **Two channels.** BM25 over turns, with the corpus being each question's own haystack. Dense
  cosine over turns, using bge-base-en-v1.5 on CPU. Assistant turns are truncated to 512 tokens for
  the embedding; BM25 sees the full text.
- **Fusion.** Reciprocal-rank fusion of the two channels per turn. A session's score is its best
  turn plus a weighted second-best.
- **Selection.** Fill the budget greedily with whole sessions. When a session doesn't fit, take a
  ±w-turn window around its best turns instead.
- **Rendering.** Sessions in chronological order with their dates.
- **What it doesn't use:** no question-type routing and no LLM calls in retrieval.

## Selection rule (written 10:49, before any dense-channel result)

*(This heading said 11:05. The second reader found it was written at 10:49, so the error was in the harmless
direction. The embeddings finished at 12:16, and the first dense-channel result came after that.)*

The dev sweeps run in four stages, each at a 16k budget and each keeping the previous stage's winner:
1. **Channels:** bm25; dense; bm25+dense.
2. **Session scoring:** `second_weight` ∈ {0, 0.5, 1} × `w_dense` ∈ {0.5, 1, 2}.
3. **Packing:** `whole_top` ∈ {99, 5, 3} × `window` ∈ {2, 3}.
4. **Time channel:** off, or on with `w_time` ∈ {0.5, 1}.

- **Winner:** highest dev all-evidence recall. Ties go to the simpler config (fewer channels, default
  values).
- **Budget:** the smallest of {8.5k, 12k, 16k, 24k, 32k} at which dev all-evidence recall ≥ 0.95. If
  none reaches it, 32k.
- **Why 0.95:** with complete evidence the reader scored 61/63 on block 1. So ≥0.95 recall should
  land near the full-history arm's accuracy, if recall carries through to reading. The dev reader
  run tests exactly that.
- **Freezing:** the frozen config is written read-only as `frozen_v10.json` before any held-out
  evaluation.

Stage 0, already run on dev with keyword search alone (so the reader can see where the rule started):
- 0.809 at 8.5k, 0.894 at 12k, 0.915 at 16k, 0.957 at 24k, 0.968 at 32k.
- v9 scored 0.596 strict / 0.670 lenient on the same questions.
- The random-order control scored 0.085.
- The time channel was tried once on 9 dev questions: +1 question at 16k, −2 at 8.5k.

## After recall

Paired reader runs on dev: v10 against v9, full history and oracle, with both judges. Then held-out
with the frozen config. Every number that leaves the lab goes through an adversarial second reader
first.
