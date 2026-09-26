# v10 held-out reader accuracy, judge calibration and chart placement: adversarial second read

2026-09-25. Second reader: a Fable 5.1 agent. Zero model tokens, CPU only, no GPU touched. The only file written
outside the session scratchpad is this one. Every number below was recomputed from the raw answer, judgment and
prompt files with my own code, not the builder's scripts (`compare_arms.py`, `calib_judges.py` and `v10.py` were
read, never run). v10's retrieval recall was verified in `REVIEW_v10_recall.md` and is not redone here.

## Verdict: WARN (Agni) — CONFIRMED WITH CAVEATS

**The numbers are real, and the run is clean.** Every count in `COMPARISON_held_readers.md` reproduces exactly from
the raw judgment files; every held-out prompt the reader saw is the frozen config's selection; all 2,000 held
reader transcripts (5 arms × 400) contain exactly the prompt file, the right model, one turn, no tools, and the
recorded answer; and of the 10,010 `claude -p` transcripts on this machine, exactly one attempt exists per held
prompt per reader model. Nothing was discarded, re-rolled or peeked at. No FAIL criterion is met.

**Three things must change before the chart placement goes out**, all presentation and context, none changing a
number:

1. **F1.** The chart ranks a **400-question held-out subset** against S-500 tables. `COMPETITORS.md`'s own rule puts
   subsets in Table 1D, "do not rank these against Tables 1A to 1C", and that is where the builder filed TAM's
   400-question result. Report the 500-question figure beside it: **479/500 = 95.8%** (Sonnet-5 judge),
   **484/500 = 96.8%** (Opus-5.5 judge), with the disclosure that 100 of those questions were the retrieval-tuning
   split. On 500, the Sonnet-judge figure clears the top published claim (95.60) by **one question**.
2. **F2.** Every reader call carried Claude Code system-reminders the official protocol does not send, including
   **"Today's date is 2026-09-24"** beside the prompt's `Current Date: 2023/...`, the model's identity, an environment
   snapshot and the account e-mail, plus a stray `-` first line. Identical in kind across all arms, so every paired
   comparison stands. It is undisclosed: `PROTOCOL.md` names only the system prompt.
3. **F3/F4.** "Above every published score" and "0.4–2.2 points stricter" both need the qualifiers given below. The
   calibration is measured on another system's short answers and is not distinguishable from zero on one of its
   two sets; the competitor list contains higher claims the builder could not verify, and the comparison crosses
   judges and dataset variants.

| held-out (n = 400) | Sonnet-5 judge | Opus-5.5 judge |
|---|---|---|
| **v10 + Opus 5.5, thinking (headline)** | **385 = 96.25%** [93.9, 97.7] | **387 = 96.75%** [94.5, 98.1] |
| v10 + Opus 4.6, no thinking | 369 = 92.25% [89.2, 94.5] | 378 = 94.50% [91.8, 96.3] |
| oracle + Opus 5.5, thinking | 392 = 98.00% [96.1, 99.0] | 397 = 99.25% [97.8, 99.7] |
| oracle + Opus 4.6, no thinking | 381 = 95.25% | 387 = 96.75% |
| v9 + Opus 4.6 | 316 = 79.00% | 320 = 80.00% |
| headline, task-averaged | 96.25 | 96.56 |
| **all 500 (dev + held), headline** | **479 = 95.80%** [93.7, 97.2]; task-avg 95.39 | **484 = 96.80%** [94.9, 98.0]; task-avg 96.44 |

Brackets are 95% Wilson intervals (checked against `statsmodels`). Paired discordants on held, headline vs Opus 4.6:
19/3 (p = 0.00086) and 10/1 (p = 0.0117); headline vs its oracle: 5/12 (p = 0.14) and 1/11 (p = 0.0063). All McNemar
p-values match `scipy.stats.binomtest`.

In the commands below, `S=/tmp/claude-1001/-home-admin/5a81960d-e981-44a5-b6a5-d661f19e37c0/scratchpad/heldreview`.
That is the session scratchpad and may be cleaned; each script's output is saved beside it as `*.out`.
`python3 $S/recompute.py` must run first (it writes `labels.json` in `$S` only). The reader transcripts referred to
are under `~/.claude/projects/-mnt-data1-lme-v2-run-cwd/`, one per `session_id` recorded in the answer files.

---

## Findings, by severity

### F1 (MEDIUM, CONTEXT DRIFT): a 400-question subset is ranked against the S-500 tables, against the file's own rule

**What the file says.** `COMPETITORS.md` line 172: *"Table 1D. Not S-500: oracle variant, subsets, M (do not rank
these against Tables 1A to 1C)."* TAM's "S-cleaned, 400 held-out questions (100 dev questions excluded), 92.25
(369/400)" is filed there (line 185), not in Table 1A, for exactly this reason. Claim 3 places our 385/400 and
387/400 above Tables 1A–1C.

**Why it matters.** The held split is stratified by type × abstention, so the 400 are representative, and the
pre-registered protocol (DESIGN.md) makes held-400 the *right* number for "did the tuning generalise". But the chart
question is a different one, and the rule the builder wrote for competitors must apply to us.

**The 500-question figures** (the reader was never tuned; retrieval was tuned on the 100 dev questions, for recall):

| headline arm, all 500 | correct | micro | Wilson 95% | task-avg | vs. top published claim (95.60) |
|---|---|---|---|---|---|
| Sonnet-5 judge | 479/500 | **95.8%** | [93.7, 97.2] | 95.39 | +0.2 pts = **one question** |
| Opus-5.5 judge | 484/500 | **96.8%** | [94.9, 98.0] | 96.44 | +1.2 pts = 6 questions |

The 500-question Sonnet task-average (95.39) is above Mastra's headline task-average (94.87). Both 500-question
intervals contain 95.60, 94.87 and 93.6.

**Fix.** Give both: "96.2 / 96.8 on the 400 pre-registered held-out questions; 95.8 / 96.8 on all 500, of which 100
were the retrieval-tuning split". Or file the held-400 row in Table 1D with TAM's and rank only the 500 figure.

```bash
python3 $S/recompute.py | sed -n '/=== DEV/,/^$/p'      # all-500 counts and intervals
sed -n 172p /mnt/data1/lme_v2/COMPETITORS.md; sed -n 185p /mnt/data1/lme_v2/COMPETITORS.md
```

### F2 (MEDIUM, PRESENTATION / undisclosed protocol deviation): the reader received Claude Code system-reminders, including today's date

**What the reader actually saw.** The `claude -p` transcripts render five `<system-reminder>` blocks into every
reader call, before the model answers:

- `# Environment` (working directory `/mnt/data1/lme_v2/run_cwd`, platform, shell, OS version);
- `You are powered by the model named Opus 5.5. The exact model ID is claude-opus-5-5. Assistant knowledge cutoff is June 2026.`;
- `<total_tokens>15000000 tokens left</total_tokens>`;
- the account's e-mail address ("use it only to identify the user ...");
- **`Today's date is 2026-09-24.`** — 2026-09-25 on 215 of the 400 oracle_o55t held calls, which ran past midnight.

The system prompt is `You are a Claude agent, built on Anthropic's Claude Agent SDK.` + `You are a helpful assistant.`
(`prompt_snapshot.cliPrefix`). And because the script passes `-` as the prompt argument and pipes the file on stdin,
the user message is `"-\n" + prompt_file` in 2,000 of 2,000 held transcripts checked (see "What held up").

**Effect.** The official generation script sends one user message and nothing else. The date reminder contradicts
the prompt's `Current Date:` on every temporal-reasoning question. This is the same in every arm (v9 through
oracle_o55t; the v9 arm's reminders say 2026-09-23), so no paired comparison moves. On the absolute number the
direction is unknown; oracle_o55t scores 107/107 on temporal reasoning under both judges, so any damage is small.
The extra tokens are the ~400-token overhead PROTOCOL.md already measured with a one-word prompt; nobody read what
was in them.

**Disclosure gap.** `PROTOCOL.md` › Protocol choices: *"there is a system prompt ('You are a helpful assistant.'),
where the official script sends only a user message."* That sentence must grow.

**Fix.** Disclose the reminders and the `-` line as harness deviations. For v10.1, test whether `claude -p` can
suppress them; if not, keep them and keep saying so. Nothing in them carries labels: `has_answer`,
`answer_session_ids`, `question_type`, `CLAUDE.md` and `MEMORY.md` occur 0 times outside the prompt text in all 2,000
transcripts.

```bash
python3 $S/transcripts_check.py                      # att:date / date:2026-09-2x counts per arm
F=~/.claude/projects/-mnt-data1-lme-v2-run-cwd/$(python3 -c "import json;print(json.load(open('/mnt/data1/lme_v2/baseline/v10_o55t/answers/q000.json'))['session_id'])").jsonl
python3 -c "import json,sys;[print(json.dumps(b)[:400]) for l in open('$F') for e in [json.loads(l)] if e.get('type')=='attachment' for b in e.get('rendered') or []]"
```

### F3 (LOW, UNVERIFIABLE transfer): the judge calibration reproduces, but it does not transfer as a number

**What reproduces.** From the two published Plastic Labs files (byte-identical to GitHub, sha256 prefixes
`4aafec663a9457a0` and `af0727c3f5a52001`, 500 questions each, question text equals the dataset's after the
`[date] ` prefix is stripped, `expected_answer` equals the gold) and our four `calib/judge_*` directories (500 files
each, label = `'yes' in raw`):

| published answers | our judge | official pass | ours | offset | agree | kappa | McNemar p |
|---|---|---|---|---|---|---|---|
| Honcho + Haiku 4.5 | Sonnet-5 | 90.4 | 88.2 | −2.2 | 95.0 | 0.739 | 0.043 |
| Honcho + Haiku 4.5 | Opus-5.5 | 90.4 | 88.4 | −2.0 | 95.6 | 0.768 | 0.053 |
| Haiku 4.5 full context | Sonnet-5 | 62.6 | 62.2 | −0.4 | 97.2 | 0.940 | 0.79 |
| Haiku 4.5 full context | Opus-5.5 | 62.6 | 61.8 | −0.8 | 97.2 | 0.940 | 0.42 |

The published labels are gpt-4o-2024-08-06 labels: the result files do not name the judge, but the Honcho bench code
at commit `a1d689b` (the directory the results sit in; 2025-12-18, six days after the runs) has
`model="gpt-4o-2024-08-06", max_tokens=10, temperature=0` and `passed = "yes" in eval_response.lower()`. I fetched
and read it. The S21 excerpt in COMPETITORS.md is accurate.

**What does not transfer.**
- **Different answers.** The calibration answers are Claude Haiku 4.5's, median 481–568 characters. The headline
  arm's answers are Opus 5.5 chain-of-thought, median 774 characters (v10 with Opus 4.6: 1,119). A judge's
  strictness on short direct answers is not its strictness on long step-by-step ones.
- **The offset is not uniform.** On the Honcho set the Sonnet judge's entire deficit sits in single-session-preference:
  **−30.0 points on 30 questions** (9 flips); multi-session is +0.0. The Opus judge's is spread (−2 to −4 on MS, TR,
  KU). Our held SSP is 22/24 under both judges, so the official judge could plausibly move that type, and only that
  type.
- **Half of it is indistinguishable from zero.** The full-context offsets (−0.4, −0.8) have McNemar p = 0.79 / 0.42.
  The Honcho offsets are borderline (p = 0.043 / 0.053). "0.4–2.2 points stricter" states a precision the data do
  not have.
- **Direction only.** The calibration supports "the official judge would probably not score these answers lower".
  It does not support adding 0.4–2.2 points to anything, and the builder has not done so; the wording must keep it
  that way.

```bash
python3 $S/calib.py                                  # table above, per-type offsets, McNemar
sha256sum /mnt/data1/lme_v2/calib/published/*.json | cut -c1-16
grep -n -E 'gpt-4o-2024-08-06|"yes" in' $S/fetch/longmem_common.py.a1d689b
```

### F4 (LOW, wording): "above every published LongMemEval_S score" needs its exclusions stated

`COMPETITORS.md` is the chart, and it lists what claim 3 is silently excluding:

- **Higher claims the builder could not verify** (its own section "Numbers seen claimed but NOT verified"):
  Supermemory "~99%" (possibly a parody); OMEGA "95.4% task-weighted / 93.2% raw" (source not opened);
  agentmemory **96.20** (481/500) with Opus 4.6, labelled "LongMemEval_S" in its README but computed on the oracle
  file (Table 1D, caveat 3).
- **The two 95.60s** (Chronos High, Agent Zero) do not name their judge; Agent Zero does not name the dataset
  variant; neither has code. Both held intervals contain 95.6, as claimed. So does the 500-question interval.
- **Variant and judge cross.** Ours is S-cleaned with Claude judges. Mastra's 93.6 (micro, derived from its
  per-type counts: 75+116+53+30+67+127 = 468) is original S with a gpt-4o judge. The file's own caveat 1 says
  *"Judges differ, and the difference is large"*, and its bottom line 8 refuses to call an equal task-average "a
  tie" for that reason. The same discipline applies upward.

Everything claim 3 asserts is true of the numbers in the tables. It is the phrase "every published score" that
overreaches: the accurate version is "every score we could trace to a primary source".

```bash
sed -n 407,420p /mnt/data1/lme_v2/COMPETITORS.md    # the unverified list
sed -n 182p /mnt/data1/lme_v2/COMPETITORS.md        # agentmemory 96.20 on the oracle file
```

### F5 (LOW, stale deliverable): RESULTS_v10.md does not carry the held-out reader results

`RESULTS_v10.md` (written 19:55Z on 09-24) still says "Reader results: dev split only; held-out is pending" and, under
Caveats, "held-out accuracy has not been measured". `COMPARISON_held_readers.md` is a generated table with no
reader, judge, dataset or protocol caveats on it. The claims under review exist only in the builder's message and
that table. Update RESULTS_v10.md with the table above and the caveats from this review before anything is quoted.

### F6 (INFO): rounding sits on the half

385/400 = **96.25** and 369/400 = **92.25** are printed as 96.2 and 92.2 (`.1f` rounds half to even); 387/400 =
96.75 → 96.8 and 397/400 = 99.25 → 99.2. Not wrong, but state the counts, as the comparison file does.

### F7 (INFO): "extended thinking" was on, and light

`alwaysThinkingEnabled:true` produced a thinking block in 400/400 held answers of both o55t arms, with a **median of
114 thinking tokens** (max 1,727 in v10_o55t, 3,817 in oracle_o55t). The reader thought briefly, not at a large
budget. Say "thinking enabled (median 114 thinking tokens)".

### F8 (INFO): the oracle ceiling is the standard oracle number, not a same-timeline ceiling

Carried forward from the baseline review: `longmemeval_oracle.json` is used as distributed, and its `question_date`
differs from the S file's on all 500 questions, so the oracle arm's temporal questions are not the same temporal
questions. 98.0 / 99.2 is comparable with published oracle numbers, not a strict upper bound for v10 on this file.

### F9 (INFO): the official output cap was not applied

Disclosed for v9 in PROTOCOL.md; quantified here for held: 23 of 400 headline answers exceed the official 800-token
CoT `max_tokens` (visible tokens, thinking excluded); 21 were judged correct by each judge. oracle_o55t: 19 (16 / 19
correct); v10 with Opus 4.6: 26 (18 / 20). The official harness would have truncated them mid-reasoning.

---

## What was attacked and held up

**The counts.**
- All 10 judgment directories (5 arms × 2 judges) hold 500 files; filename = `question_id`; `judge_model` correct;
  `question_type` and abstention flag agree with the dataset; `label == ('yes' in judge_raw.lower())` on every file.
- All 2,000 headline-relevant judge replies (`judge_v10_o55t_*`) are a bare yes/no. Six replies elsewhere are
  verbose; none contains the substring "yes" in a "no" verdict, so the official rule labelled all six as the judge
  meant (`$S/recompute.py`, "judge_raw values other than yes/no").
- Held recount, per arm, per judge, per type, and every discordant pair, equals `COMPARISON_held_readers.md`.
- Judge agreement on the headline arm: 396/400 (Sonnet-yes/Opus-no 1; Opus-yes/Sonnet-no 3).
- **Self-preference check** (the Opus-5.5 judge scoring Opus-5.5 answers): its lift over the Sonnet judge on held is
  +1.00 (v9), +2.25 (v10), **+0.50 (v10_o55t)**, +1.50 (oracle), **+1.25 (oracle_o55t)**. Smallest on the arms it
  could favour. No sign of it.

**The prompts the reader saw** (`$S/held_prompts.py`, no v10 code imported).
- 400/400 held v10 prompts parse; each `### Session` block is a JSON list of `{role, content}` turns forming an
  ordered subsequence of a real haystack session with that date; sessions are chronological; `Current Date` and
  `Question` equal the dataset's; 0 occurrences of `has_answer`, `question_type`, `question_id`,
  `answer_session_ids`, `_abs`, any type name or any `answer_/noans_` session id.
- Evidence coverage from the prompt files: **365/376 = 0.9707** on `has_answer` sessions, **361/376 = 0.9601** on
  official `answer_session_ids` — both identical to the recall review. The builder's meta says 373/384 because 8
  abstention questions carry `has_answer` turns; all 8 are covered.
- Per question, the number of sessions, whole sessions and budget tokens in the prompt files equal the frozen
  config's logged held-out record (`runs/20260924T122426_eval_held.json`, hash 7bc75f509aa0) on **400/400**. So the
  one edit to `v10.py` after that record (19:55Z; it adds a "started" line and a `status` field to the held-out log,
  inside `main()`, and I read the edit in the transcript) changed no selection. The held prompts were built at 20:02Z
  with `--config frozen_v10.json --split held`, which asserts the config is read-only.
- `diff -rq`: v10 and v10_o55t prompts byte-identical; oracle and oracle_o55t byte-identical.

**The reader calls** (`$S/transcripts_check.py`, `$S/attempts_scan.py`).
- For all 5 arms × 400 held answers, the transcript named by the answer's `session_id` exists; its single user
  message equals `"-\n"` + the prompt file; every assistant turn has the recorded model; there is no `tool_use`
  block; a thinking block is present iff the arm is an o55t arm; the concatenated text equals the recorded `answer`.
  (Two v9 transcripts first showed as unparseable; that was my parser splitting on U+2028 inside JSON strings. The
  second scan, splitting on newlines only, parses all 10,010 transcripts with 0 bad lines.)
- **Cherry-picking test.** Indexing all 10,010 transcripts by (first user message, model): for every held prompt of
  v10, v10_o55t, oracle and oracle_o55t there is **exactly one** attempt with that prompt and that model. No
  discarded or repeated answers exist.
- Answer records: 0 empty, `num_turns` = 1, `web_search_requests` = `web_fetch_requests` = 0, 400 distinct
  `session_id`s per arm, no identical answer text across arms.

**Timeline** (builder transcript, this session's own file).
- 19:55Z RESULTS_v10.md pre-declares "the unhandicapped held-out run is the headline candidate" with no held
  accuracy in existence. 20:04Z v10 held answers start (Opus 4.6). 20:46Z v10_o55t and oracle_o55t held answers
  start; the first held judgments of any arm (`judge_v10_*`) start 20:49Z. The headline configuration was chosen
  before any held-out accuracy was seen, and both configurations are reported.
- Failures: 6 `session limit` failures in oracle_o55t (q232–237), logged, never written as answers, retried after
  the reset; 91 judge failures, all Opus-5.5 (`is_error`), logged, none scored, all retried. The only `rm` on any
  benchmark path is the scratch `v10_scratch_bm25` prompt directory at 19:26Z, before any answer existed.
- `compare_arms.py`'s `--split held` filter intersects the 500 judged qids with `held_qids` (400); the header
  "400 questions judged in every arm by both judges (v10 split: held)" is exact.

**Inputs.**
- Dataset sha256 `d6f21ea9…` equals `split.json`; held idx → qid mapping equals `held_qids`; dev ∩ held = ∅.
- Published calibration files byte-identical to `plastic-labs/honcho-benchmarks/main/a1d689b/` (hash and size).

---

## Wording I would accept

**Claim 1.**
> On LongMemEval_S-cleaned, Mnemosyne v10 retrieval (frozen config 7bc75f509aa0, ~24k-token contexts) with reader
> claude-opus-5-5, thinking enabled (median 114 thinking tokens), scores **385/400 = 96.2%** under a claude-sonnet-5
> judge and **387/400 = 96.8%** under a claude-opus-5-5 judge on the 400 pre-registered held-out questions
> (95% Wilson [93.9, 97.7] and [94.5, 98.1]); on all 500 questions, 100 of which were the retrieval-tuning split,
> **479/500 = 95.8%** and **484/500 = 96.8%**. The same prompts read by claude-opus-4-6 without thinking score
> 369/400 = 92.2% and 378/400 = 94.5% (paired McNemar p = 0.0009 and 0.012). Evidence-only prompts with Opus 5.5 and
> thinking score 392/400 = 98.0% and 397/400 = 99.2%, using the oracle file's own question dates. Judging used the
> official per-type answer-check prompts and the official "yes" rule. Deviations from the official harness: Claude
> judges instead of gpt-4o-2024-08-06; no temperature control and no 800-token output cap (23 of 400 headline
> answers exceed it); a `claude -p` harness that adds a short system prompt and Claude Code system-reminders
> (today's real date, model identity, environment) to every call, identically across arms.

**Claim 2.**
> On 1,000 published LongMemEval_S answers (two Plastic Labs runs, gpt-4o-2024-08-06 labels), our claude-sonnet-5
> and claude-opus-5-5 judges agreed with the official judge on 95–97% of questions (Cohen's kappa 0.74–0.94) and
> passed 0.4–2.2 points fewer answers; the difference is significant at the 5% level on one of the two sets and not
> on the other, and for the Sonnet judge it is concentrated in single-session-preference. Those answers are short
> Haiku 4.5 answers, not chain-of-thought, so the offset indicates direction, not a correction to apply.

**Claim 3.**
> Against every LongMemEval_S result in our survey that we could trace to a primary source, the point estimates
> above — 96.2 / 96.8 on the held-out 400 and 95.8 / 96.8 on all 500 — are the highest; on all 500 the Sonnet-judge
> figure exceeds the top published claims (95.60, Chronos and Agent Zero) by one question. The 95% intervals of all
> four figures include 95.60, and both of those claims lack a named judge and public code. The best result with a
> named official judge and public code is Mastra's 93.6 micro / 94.87 task-average (original S); ours is 95.4–96.6
> task-averaged on S-cleaned under Claude judges, so that comparison crosses judge and dataset variant. Claims we
> could not verify (Supermemory "~99%", OMEGA 95.4 task-weighted, agentmemory 96.2 on the oracle file) are excluded.
