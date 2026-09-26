# v10 evidence recall: adversarial second read

2026-09-24. Second reader: an Opus 5.5 agent. Zero model tokens, CPU only. The only file written outside the scratchpad
is this one.

## Verdict: CONFIRMED WITH CAVEATS

The number is real. Retrieval cannot see the labels. The prompt files the reader actually receives contain every
`has_answer` evidence session on 91/94 dev questions: I checked this with my own parser, not v10's code. The held-out
365/376 reproduces exactly. The dev stage selection reproduces 26/26 with the post-freeze code. No leak found.

Two things must change before the number goes out:
1. **The wording.** "All evidence sessions" is measured on `has_answer`-labelled sessions. On the benchmark's own
   `answer_session_ids` it is **96.0%** held-out, not 97.1% (F1).
2. **The framing of the v9 comparison.** v10 runs at 3.3× v9's context size and is compared against a lower-bound
   matcher. At v9's budget, the same config gets 79.8% on dev (F2).

| | dev (94 scored) | held (376 scored) |
|---|---|---|
| Claim metric (`has_answer` sessions) | 91 = 0.968 | 365 = **0.971** [0.948, 0.984] |
| Official `answer_session_ids`, ≥1 turn in context | 90 = 0.957 | 361 = **0.960** [0.935, 0.976] |
| Official, whole session in context | 89 = 0.947 | 360 = 0.957 |
| v10 through v9's matcher (`tm`, strict) | – | 0.955 |
| v9 strict / lenient (matcher, lower bound) | 0.596 / 0.670 | 0.593 [0.543, 0.642] / 0.670 [0.621, 0.716] |
| Frozen config at 8.5k (v9-sized budget) | 0.798 | not computed: it would be an unfrozen held-out peek |

Brackets are 95% Wilson intervals.

In the commands below, `S=/tmp/claude-1001/-home-admin/5a81960d-e981-44a5-b6a5-d661f19e37c0/scratchpad/v10review`.
That is the session scratchpad and may be cleaned; the key commands are also given inline. Run
`cd $S && python3 rerun.py` first. It takes about 4 minutes on CPU and writes `rerun.pkl` in `$S` only. It never touches
`heldout_log.jsonl`.

---

## Findings, by severity

### F1 (MEDIUM): "ALL evidence sessions" means all `has_answer`-labelled sessions, which is a subset

**What happens.** 32 of the 470 non-abstention questions list `answer_session_ids` sessions that contain no
`has_answer` turn:
- by split: 25 held, 7 dev;
- by type: 20 temporal-reasoning, 10 multi-session, 2 knowledge-update.

`evidence()` cannot see those sessions. The reverse never happens: every `has_answer` session is in
`answer_session_ids`.

**Effect.** Under the official definition, with every `answer_session_ids` session in context:
- held: 361/376 = 0.960;
- dev: 90/94 = 0.957.

Five questions flip: held 289, 290, 295, 300, and dev 299. All are temporal-reasoning. In each, the missed session
is a same-kind event on another date. Example: 290 asks "Who did I go with to the music event last Saturday?" v10 has
the labelled 04/15 Queen concert session and misses the unlabelled 03/18 Billie Eilish concert. These sessions are
there to make the question ambiguous; they don't carry the answer. So answerability is probably unaffected, but the
sentence "gets ALL of a question's evidence sessions" is not what is measured.

v9's autopsy uses the same `has_answer` definition, so the v9 comparison is like-for-like on this axis.

**Fix.** Say "all `has_answer`-labelled evidence sessions", or report both figures (97.1 / 96.0).

```bash
# 32 questions (labels only, no retrieval)
python3 -c "import json;d=json.load(open('/mnt/data1/datasets/longmemeval/longmemeval_s.json'));print(sum(1 for q in d if not q['question_id'].endswith('_abs') and {s for s,x in zip(q['haystack_session_ids'],q['haystack_sessions']) if any(t.get('has_answer') for t in x)}!=set(q['answer_session_ids'])))"
# both definitions, frozen config, in memory (does not append to heldout_log.jsonl); pass dev and/or held
python3 - dev held <<'EOF'
import sys, json; sys.path.insert(0, '/mnt/data1/lme_v2/v10'); import v10
cfg = {**v10.DEFAULT, **json.load(open('/mnt/data1/lme_v2/v10/frozen_v10.json'))}
data = json.load(open(v10.DATA)); split = json.load(open(v10.ROOT + '/split.json')); dense = v10.Dense()
for name in sys.argv[1:]:
    a = b = n = 0
    for qi in split[name]:
        it = data[qi]; ev = v10.evidence(it)
        if it['question_id'].endswith('_abs') or not ev: continue
        ch, _, _ = v10.retrieve(v10.view(it), qi, cfg, dense)
        asi = [si for si, s in enumerate(it['haystack_session_ids']) if s in it['answer_session_ids']]
        n += 1
        a += all(any(si in ch and (ch[si] is None or ti in ch[si]) for ti in tis) for si, tis in ev.items())
        b += all(si in ch for si in asi)
    print(f'{name}: n={n}  has_answer definition {a}/{n}={a/n:.4f}  answer_session_ids definition {b}/{n}={b/n:.4f}')
EOF
```

### F2 (MEDIUM): the v9 gap is not budget-matched and sets exact membership against a lower bound

**Budget.**
- v9 contexts are a median 7.2k o200k tokens (max 24.3k). v10 uses 24k.
- At 8.5k, the frozen config scores 0.798 on dev (stage curve), against v9's 0.596/0.670 on the same 94 questions. At a
  matched budget the dev gap is about 13–20 points, not 30–37.
- BM25 alone at 24k already scores 0.957 on dev (Stage 0), against 0.968 for the frozen config. Most of the headline
  gain is whole sessions plus 3× the budget, not the dense channel or the tuning.

**Lower bound.** v9's 59.3 / 67.0 is read correctly:
- the counts are 223/376 and 252/376;
- all 470 qids join and `n_ev` agrees on every question;
- v9's contexts come from the same cleaned haystack: 99.5% of their dated snippets appear verbatim in it.

But the figure is a verbatim-matcher lower bound, and v9 contexts are extracted sentences. The strict test (≥50% of a
turn's 50-character windows must match) penalises sentence extraction structurally. v10 contexts are whole turns.

**The matcher also cuts against v10.** 6 held questions are exact hits but matcher misses (251, 451, 465, 473, 475,
494). `render()` JSON-escapes newlines and quotes. Against the JSON render the windows match at 0.00–0.43; against raw
text they match at 1.00. v9 contexts are raw text. So the like-for-like pair is v10 `tm` 0.955 against v9 strict 0.593,
and even that is conservative for v10.

**Fix.** Headline the comparison at a matched budget: freeze an 8.5k config and log it on held. Otherwise, say "at
about 3× v9's context size".

```bash
python3 -c "import json;[print(c['budget'],round(c['all_sessions'],3)) for c in json.load(open('/mnt/data1/lme_v2/v10/runs/stages_20260924T121904.json'))['curve']]"
python3 -c "import json,os,statistics,tiktoken;e=tiktoken.get_encoding('o200k_base');c=json.load(open(os.path.expanduser('~/benchmark_results/longmemeval/frontier_v9_contexts.json')));print(statistics.median(len(e.encode(x['context'],disallowed_special=())) for x in c))"
cd $S && python3 tm_gap.py && python3 v9_haystack_check.py
```

### F3 (LOW): the budget is counted on a different serialisation from the one the reader sees

**Cause.** `turn_tokens()` counts `json.dumps(turn, ensure_ascii=False)`. The official builder calls
`json.dumps(session)` with the default `ensure_ascii=True`, so every non-ASCII character becomes `\uXXXX`, and each
emoji becomes a surrogate pair.

**Size.** Escaping alone puts the real history up to 1,196 tokens over the counted budget. The header and separator
estimate is close: median +12, range −72 to +49.
- Of 500 histories, 211 exceed 24,000 real tokens, 8 exceed 24,500 and 2 exceed 25,000.
- The largest is held idx 367: a 25,207-token history and a 25,296-token full prompt.
- The dev prompt files have a median of 24,082 and a max of 24,942 (idx 439). That agrees with the meta, and "~24k"
  holds.

**Effect.** None on recall. It matters for any budget-matched comparison.

**Fix.** Count with `ensure_ascii=True`, or count the rendered session string.

```bash
cd $S && python3 token_decomp.py      # needs rerun.pkl
```

### F4 (LOW): the held-out log records only successful runs

**What happened.** `heldout_log.jsonl` has one entry. The builder session's transcript shows three
`v10.py eval --split held` calls:
- **19:19:18Z and 19:21:14Z:** both crashed with tiktoken's `ValueError` on `<|endoftext|>`. The crash happens inside
  `evaluate()`, before any number exists.
- **19:24:14Z:** succeeded; this is the logged run.

**Why it matters.** Nothing leaked this time. But a crash or Ctrl-C leaves no trace, and anything that imports `v10`
computes held-out numbers with no log entry. That includes this review.

**Fix.** Append a "started" line (config hash and time) before evaluating, and a "finished" line after.

**Disclosure.** This review re-ran the frozen config on held in memory. It reproduced 365/376 and added the
`answer_session_ids` metric. It computed no held-out number for any other config.

```bash
cd $S && python3 transcript_held.py
```

### F5 (LOW): DESIGN.md's provenance text is wrong, in the safe direction

**The timestamp.** The section headed "Selection rule (written 11:05, before any dense-channel result)" was added by an
Edit at 17:49:34Z, which is 10:49:34 PDT.
- The file's birth time and mtime are both 10:49:34.61; its ctime is 44 ms later.
- Nothing has written to it since.

So the rule existed about 15 minutes earlier than it says. That is still before:
- the stage grids (11:12:28);
- `run_stages.py` (11:12:49);
- the embeddings (finished 12:16:16).

**The header.** "Written before any v10 number existed" is stale. The same edit added the Stage 0 dev results (runs
from 10:37 to 10:43), though those are labelled as prior runs.

**Fix.** Correct the timestamp. Nothing in the selection is affected.

```bash
stat -c '%w | %y | %z' /mnt/data1/lme_v2/v10/DESIGN.md
```

### F6 (INFO): identical-content turns leave a residual order dependence

**What happens.** `rrf()`'s comment says ties are "never [broken] by haystack position". But turns with identical
content share a hash, and `np.lexsort` is stable, so those ties fall back to array order. 276 of 500 haystacks contain
duplicated turns.

**Effect.** I shuffled session order with 2 seeds for each of the 500 questions. The selection changed in 8 of 1,000
runs (questions 32, 38, 135, 308, 424, 462). No evidence verdict changed.

**Why it can't carry labels.** No evidence turn is duplicated in any haystack (0/500). Answer sessions are spread evenly
through the haystack: their position deciles run from 0.11 to 0.92.

```bash
cd $S && python3 invariance.py        # needs rerun.pkl; also runs the label-scramble test
```

### F7 (INFO): the post-freeze tiktoken change cannot change a result

**Edit history.** The transcript shows v10.py had four edits before the freeze: the initial write, then the tie-break,
the time channel and the dense NaN handling. It had exactly one edit after: `disallowed_special=()` at 19:23:20Z. The
file's mtime (12:23:20 local) matches that edit. Between the freeze and the logged held-out run, the only other tool
calls are the two crashed held-out runs.

**Why it's safe.**
- tiktoken's output is identical unless the text contains an o200k special-token literal.
- Only held idx 351 (gpt4_78cf46a3) contains one: `<|endoftext|>`, not in an evidence turn, and `all_sessions` is True.
  The old code raised there, so no earlier value existed.
- 17 other haystacks contain Llama-3 tokens such as `<|end_header_id|>`, but those are ordinary text to o200k.

**Checked.** With the current code:
- all 26 stage evaluations reproduce `stages_20260924T121904.json` exactly;
- all 100 dev rows match `runs/20260924T122332_eval_dev.json`.

One side note: the budget counts that literal as about 7 ordinary tokens, while the builder counts it as 1 special
token. That over-counts, which is the safe direction.

```bash
cd $S && python3 restage.py           # ends with "TOTAL DIFFS 0"
```

---

## What was attacked and held up

**Leakage.**
- **Labels.** `view()` reads only the question, the question date, the session dates and each turn's (role, content),
  and it asserts that key set.
  - The label-scramble test stripped `has_answer`, renamed every session id and blanked
    `question_id`/type/answer/`answer_session_ids`. It changed 0/500 retrievals (`invariance.py`).
- **Caches.** `_tf_cache`, `_len_cache` and `Dense.row` are keyed only on sha1(role\0content). The embedding rows are
  ordered by (length, hash).
- **Embeddings** (`emb_check.py`). I recomputed 24 sampled turn rows and 16 question rows on CPU, with the GPUs hidden.
  - They match the stored rows at cosine 1.0000.
  - The question plus answer text gives 0.913 and the neighbouring row gives 0.384. So queries embed the question
    alone, and the rows are aligned.
- **Question-agnostic controls, dev only** (`controls_dev.py`).
  - Packing the shortest sessions first gets 10/94 = 0.106. Random order at 24k gets 16/94 = 0.170.
  - Answer sessions are larger than distractors (median 3,300 against 2,336 budget tokens), so the greedy packing isn't
    exploiting a size artefact.
  - v10 fills 19% of sessions and 21% of haystack tokens (dev medians).
- **Dataset.** The local file is 277,383,467 bytes. That is the size of `xiaowu0162/longmemeval-cleaned`'s
  `longmemeval_s_cleaned.json`; the original `longmemeval_s` is 278,025,796. Its sha256 matches `split.json` and
  `emb/meta.json`.

**Evaluation.**
- **Coverage logic.** In `evaluate()`, a session is covered if at least one of its `has_answer` turns is in the chosen
  whole session or window. `all_sessions` = `bool(ev) and all(covered)`. `all_turns` equals `all_sessions` in every run,
  because sessions go in whole.
- **Denominators.** Held: 376 = 400 − 24 abstention. Dev: 94 = 100 − 6. All 21 questions without `has_answer` turns
  are abstention questions, so no non-abstention question is dropped.

**Reproduction from the reader's prompts (dev).** `repro_prompts.py` imports no v10 code. It parses the 100 prompt
files, JSON-decodes each session block, and looks for each `has_answer` turn's (role, content) in a block carrying
that session's date. It agrees with a raw `json.dumps(content)` substring search in every case.
- 91/94 = 0.9681.
- (`n_ev`, `ev_covered`) equals `prompts_meta_v10.json` on 100/100 questions.
- Every block is a subsequence of a real haystack session with that date.
- The blocks equal `retrieve()`'s chosen sets on 100/100 questions.

**Held-out.** An in-memory re-run with `v10.retrieve` reproduces 365/376 exactly (`rerun.py`).

**Pre-registration.**
- **Frozen config.** `frozen_v10.json` equals the stage record's frozen config (hash 7bc75f509aa0).
- **Rule applied as written:**
  - stage 1: bm25+dense, 0.936, the only maximum;
  - stage 2: `second_weight` 1.0 × `w_dense` 2.0, 0.947, the only maximum;
  - stage 3: a three-way tie at 0.947, resolved to `whole_top` 99 (the default) by simplicity;
  - stage 4: a tie, resolved to time off (fewer channels);
  - budget: 16k gives 89/94 = 0.9468 < 0.95, so 24k (0.968).
- **Timeline:**
  - 10:49: rule written;
  - 11:12: grids written;
  - 12:16:16: embeddings finished;
  - 12:16:36: a dev-only dense sanity check (dense top-10 turns reach an evidence session on 93/100 questions; 23/100
    with shuffled questions);
  - 12:17:08–12:19:04: stages;
  - 12:19:04: frozen;
  - 12:19:18: first held-out call.
- Nothing was tuned on held. The disclosed prior knowledge (the v9 autopsy over all 500 questions) stands as disclosed.

## Suggested claim wording

> On LongMemEval_S-cleaned, v10 puts every `has_answer`-labelled evidence session into a ~24k-token context (o200k;
> up to 25.3k as rendered) on 97.1% of 376 held-out questions [94.8, 98.4]. That is 96.0% when every official
> `answer_session_ids` session is required. On dev (tuned), the figures are 96.8% and 95.7%. v9's roughly 7k-token
> context covers 59–67% of the same questions by a verbatim-matcher lower bound. v10 scores 95.5% through that same
> matcher and 79.8% on dev at v9's 8.5k budget.
