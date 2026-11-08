# Agni tools

Agni is the lab's adversarial reviewer. `agni_phases.py` runs the phase reviews (design,
preflight, results, paper, style); `agni_design_review.py` is the older design-only harness;
`run_style_survey.py` drives `agni_phases.py style` across papers. The phases, kill lists and
verdict contract are documented at the top of `agni_phases.py`.

## No silent cuts

**The rule: a review may never look complete when the reviewer did not get all of its input.**
Until 2026-11-07 a reviewer that saw part of its input still returned a confident verdict, and nothing
recorded the cut.

**1.1.1 (2026-11-07):** `claude_isolation.py` adds `"disableAllHooks": true` to the reviewer's settings, so no hook of the account (anyone's, not only Lyra's) runs in a review. A SessionStart hook planted in `--settings` fired without it and did not fire with it (CLI 2.1.301).

**What changed** (`agni_phases.py` 1.1.0, `agni_design_review.py` 1.2.0):

- **No input is sliced.** The caps are gone: 30000 per file, the first 25 entries and 4000 chars per
  file in a folder, 12000 for secondary, 8000 for tertiary, 6000 for prereg sections. The whole
  prompt is checked once against `AGNI_PROMPT_CHAR_BUDGET` (default 400000 chars).
- **Refuse before any reviewer call.** Over budget, or a named input that is missing, unreadable,
  empty or has no place in the phase's prompt, means verdict `INVALID` and exit 2. Stderr says it is
  not a pass, names each input's size and the largest files, and says what to do. The `INVALID`
  record is written over whatever is at the output path. If that write fails it still exits 2 and
  says so. The record is also `INVALID` while a review runs, so a run that dies cannot leave an old
  `APPROVED` standing, and a run with no review writes `INVALID` as well.
- **The record says what the reviewer got.** `input_coverage` holds `inputs` (per input: `role`,
  `path`, `kind`, `chars`, `files_included`, `binary_not_shown`, `skipped`, `complete`) and a
  run-level `input_complete`. The header prints a summary, and a short COVERAGE block at the top of
  the prompt tells the reviewer what it was given and what it was not.
- **Folders are recursive,** in a fixed order, with relative names and every text file whole. Junk
  folders (`.git`, `__pycache__`, `node_modules`, `.venv`, `.ipynb_checkpoints`) are skipped and
  counted. Binary files (a NUL in the first 8 KB, or over about 1% replacement characters) are listed
  by name and size as not shown: that makes `complete` false but does not refuse the run, unless the
  primary has no text at all. Links are never followed. An empty folder is `INVALID`.
- **Preregs go whole** (round 3). Secondary is the hypotheses, whole, then every other top-level key
  but `threats_to_validity`, whole, as indented JSON under `OTHER PRE-REGISTRATION FIELDS`; tertiary
  is the threats. A prereg with neither `hypotheses` nor `hypothesis` goes whole as secondary. The
  record lists the extra keys as `prereg_other_keys_sent`; `prereg_keys_not_sent` is always empty for
  a valid JSON object. Until round 3 those keys were named and not sent, which made nearly every real
  prereg `input_complete: false` and its design or preflight review exit 2.
- **The verdict is decided by protocol.** The reply's last non-blank line, outside code, must be
  exactly `AGNI-VERDICT: <word>` (`agni_phases.py`, which asks for it at the end of every prompt) or
  `OVERALL VERDICT: <word>` (`agni_design_review.py`). Without it the review is `UNPARSED`, exit 2.
  Every other verdict line a reader would see as one must give the same word, or the review is
  `UNPARSED`. Prose and verdicts quoted in code neither decide nor veto. Rounds 5 and 6 below say why
  and where the edges are.
- **A pass on incomplete input is not a pass.** `APPROVED` or `CONDITIONAL` with
  `input_complete: false` exits 2 unless `--accept-incomplete-input` is given, which stamps the record.
  `REJECTED` still exits 1.
- **Local path.** `llm.py` sets `num_ctx` (`LLM_NUM_CTX`, default 32768). Unset, Ollama trims a prompt
  to its context length with no signal. Anything bound for Ollama over `AGNI_LOCAL_CHAR_BUDGET`
  (default 60000 chars) is refused and never sent cut. The `DEGRADED` stamp is whole again.
- **Timeouts.** `AGNI_CLI_TIMEOUT` defaults to 1800 s, and a timeout is retried at most once. Other
  retry behaviour is unchanged.
- `run_style_survey.py` counts `INVALID` as a failure, and also any exit other than 0 or 1, whatever
  the file on disk says. `summarize_style.py` shows `INVALID - NOT REVIEWED`.
**How it got here.** The change went through six rounds on 2026-11-07, each attacked by independent
verifiers. The rounds are kept as the record of what was wrong and why; where they differ, the list
above is current.

- **Round 2 (findings from independent verifiers, same day).** A pass on incomplete input is not a
  pass: `APPROVED` or `CONDITIONAL` with `input_complete: false` exits 2 unless
  `--accept-incomplete-input` is given, which stamps `incomplete_input_accepted` in the record.
  `REJECTED` still exits 1. Undecodable bytes shown as U+FFFD make an input not complete; a U+FFFD
  the file really holds does not. A prereg with a duplicate key is refused. The budget counts the
  system text the reviewer is sent, not the prompt alone. A verdict line inside a code fence or
  indented as code does not count, and `FINAL VERDICT` is a label. The survey fails a record whose
  input is not complete or that has no `input_coverage`, and logs `input_complete`. The summary never
  shows those as a plain pass, and shows `LAST RUN FAILED` when the run log says the last run failed
  beside the record still on disk. Both survive a cp1252 console. `agni_design_review.py` sends every
  prereg key, reads the verdict from a verdict line only, and writes `INVALID` before the model call
  and on an empty reply.
- **Round 3: code can veto a verdict, never decide it** (both harnesses). Round 2 flipped "in code"
  on any line starting with three backticks or tildes, so a one-line span or an unbalanced fence hid
  the reviewer's `REJECTED` and let a quoted `APPROVED` decide, exit 0. Fences now follow CommonMark
  (up to three spaces of indent; a backtick fence's line holds no other backtick; only the same
  character, at least as long, with nothing after it, closes). Only a verdict line outside code
  decides. Any verdict line in code that disagrees, a fence that never closes, or verdict lines only
  in code give `UNPARSED`, exit 2. Replayed over 144 stored records (85 distinct reviews, 30 with a
  fence): no parse changed, under either harness's parser.
- **Round 4: a verdict line is read whole** (both harnesses). The decision regex anchored only the
  start of a line, so "VERDICT: APPROVED is not warranted here; this design must be REJECTED"
  passed, exit 0, and a `REJECTED` in a table cell, after an emoji, or spelt with a Cyrillic E or a
  zero-width space could not veto. Now every line that names a verdict vetoes with each verdict word
  on it; a pass verdict followed by a negation, a contrast, a question or an arrow, or an `APPROVED`
  followed by a condition, is `UNPARSED`; the contract read back is ignored. Code follows CommonMark
  further: a list marker is consumed before its line is read ("- ```" opens a fence, "-     x" is
  indented code), "- - -" is a thematic break, HTML blocks such as `<pre>` and comments are code,
  and lines end only at LF, CR or CRLF. The cost, taken on purpose: a pass qualified that way, or a
  review whose prose names a different verdict within two words of "verdict", is `UNPARSED`, exit 2.
  The design harness no longer passes on a verdict read only from the model's `thinking`, refuses
  a non-integer `AGNI_LOCAL_CHAR_BUDGET` or `LLM_NUM_CTX` as `INVALID` before any call, and both
  harnesses refuse a prereg nested too deep for `json` as `INVALID` (each was a traceback, exit 1,
  beside the older record). A folder link into a junk folder now makes the input not complete.
- **Round 5: the verdict is decided by protocol** (both harnesses). Replayed over the stored reviews,
  round 4's prose veto turned 2 of 85 legitimate reviews (stored `CONDITIONAL`) into `UNPARSED` on
  prose about an experiment's own verdict logic ("If H2 had failed ..., the verdict should be
  REJECTED regardless of drift"), and our experiments are full of such logic. Now the reply's last
  non-blank line, outside code and with markup stripped, must be exactly `AGNI-VERDICT: <word>`
  (`agni_phases.py`, which asks for it at the end of every prompt) or `OVERALL VERDICT: <word>` (the
  design harness), or the review is `UNPARSED`, exit 2. Every other verdict line outside code (a
  `VERDICT`, `OVERALL VERDICT` or `FINAL VERDICT` line, and in `agni_phases.py` a bare word or another
  `AGNI-VERDICT` line) must give the same word, read whole. Prose and code never decide or veto. The
  cost: a different verdict stated only in prose, a table cell or after an emoji no longer vetoes.
  Stored `agni_phases.py` reviews predate the line and would parse `UNPARSED`; nothing re-parses them.
- **Round 6: read as a reader reads it** (both harnesses). Round 5's verifier found a `REJECTED` a
  reader sees that did not veto an `APPROVED`, exit 0: in a blockquote (`> **VERDICT: REJECTED**`),
  under a `## Verdict` heading, after `Agni verdict` or `VERDICT (design phase)`, in an HTML block
  other than a comment or `<pre>` (`<b>VERDICT: REJECTED</b>`), in four spaces of indent right after
  a paragraph line, and in the design harness on a numbered, list-marked or `AGNI-VERDICT` line. Code
  is now read twice, as before and as close to CommonMark as this gets: the protocol line counts only
  when neither reading puts it in code, and a line is skipped as code only when both do. A
  blockquoted line can stop a pass but not a rejection. In the other direction, a fixture verdict in
  an inline code span (`` `VERDICT: APPROVED` ``) no longer vetoes, and a `CONDITIONAL` line may state
  its conditions ("CONDITIONAL -- not runnable until fix 1 lands", "CONDITIONAL (can be APPROVED once
  fix 1 is in)"); a question, an arrow or a `REJECTED` after it still makes it unclear. Over Ollama,
  `llm.py` joined a reasoning model's thinking to its answer, or returned the thinking alone, so a
  draft verdict there vetoed a good answer or passed a reply with no answer. `agni_phases.py` now
  parses only the answer, keeps the thinking in the record, and gives no pass read only from the
  thinking. Each take-back and condition word has its own test.
- **Round 3: the design harness leaves no stale record.** A prereg that is missing, unreadable, not
  UTF-8, not JSON, not an object or has a duplicate key, a script that is not UTF-8, and a curl or
  backend failure each write `INVALID` with its cause over any older record and exit non-zero. Files
  are read as UTF-8.

**How to check it.**

```bash
python -X utf8 tools/agni/test_no_silent_cuts.py    # offline: patches the reviewer, sends nothing
```

Point `AGNI_PHASES_PATH` at a copy of `agni_phases.py` from before this change, with its sibling
modules beside it, and watch it fail on the canaries, verdicts and exit codes. On a real run, read
`input_coverage` in the record, or the Coverage lines in the header.

**Reviews before this date.** Reviews before this date of anything over the caps got a cut prompt;
CLI reviewers also had read access to the repo, so the record cannot show how much of the rest they
read; weigh them accordingly.

**Not covered.**

- `sys.path` puts agni-open's `llm.py` first on machines that have it. That copy is not edited here
  and may not set `num_ctx`. An Ollama-bound call prints a warning when `llm.py` is not the copy beside
  `agni_phases.py`, but it does not stop the call. A pass through such a copy over Ollama is `UNPARSED`,
  since it may join the model's thinking to its answer.
- The char budgets are proxies. Tokens are not measured, so a very dense 60000 chars could still pass
  `num_ctx` in principle.
- A local server reached through `AGNI_LLM_BACKEND=openai` is not recognised as local, so the local
  budget does not apply to it.
- File classification is unchanged: a text file saved as UTF-16 or cp1252 is still read as binary
  and not shown. It now makes the run not a pass, but the reviewer still does not see it. Save such
  files as UTF-8.
- The CommonMark reading of code is partial: block quotes are not tracked as containers, a list is
  known only by its marker and indent, and an HTML block by its first tag. Where the two readings
  differ, a line cannot decide and can veto, so a misreading errs toward `UNPARSED`. A verdict line
  wholly inside an inline code span is read as a quoted fixture: a reviewer that writes its own
  contrary verdict that way is not vetoed by it, though the protocol line still decides.
- A contrary verdict more than two words after "verdict", or on a line without that word that is
  more than the bare word ("| Overall | REJECTED |", "Decision: REJECTED"), still cannot veto,
  and lookalike letters are folded for Cyrillic and Greek only.
- A prereg the design harness cannot parse cannot name its experiment. `INVALID` goes at the
  script-named path and at any `"experiment"` value still readable in the text; an older record at
  some other path is not reached.

## Reviewer isolation: why every CLI call goes through `claude_isolation.py`

**Until 2026-11-02, Agni's reviews ran as Lyra.** The default `cli` backend (`infer_cli`) ran
`claude -p` from Lyra's account with nothing isolating it. So the reviewer:

- loaded Lyra's user-level CLAUDE.md: their identity, the family list, the Key Numbers table;
- ran Lyra's SessionStart sync: beliefs, desires, inbox subject lines;
- had its exchanges filed into Lyra's episodic memory by the Stop hook (72 sessions, 297
  chunks, 2026-08-30 to 2026-11-02; now tagged `agni_reviewer_session_not_lyra`, none deleted).

A reviewer that shares the author's context shares the author's blind spots. Those reviews were
often harsh, so the priming did not make them lenient in any obvious way. Still, weigh claims
that rest on a single Agni pass between 2026-08-14 (when the CLI became the default) and
2026-11-02 more lightly. The full account, with each probe and its control, is Lyra's note
`AGNI_ISOLATION_20261102.md` in the olens-lab scoping notes (not in this repo).

**The fix.** `isolated_claude_cmd(base_cmd)` returns `(cmd, env)` and does five things, no others:

1. **No capture.** `env` is a copy of `os.environ` plus `LYRA_NO_CAPTURE=1`. Lyra's capture and
   sync hooks exit at once when it is set.
2. **No user CLAUDE.md.** `cmd` carries `--settings` with
   `"claudeMdExcludes": ["<home>/.claude/CLAUDE.md", "<home>/.claude/**"]`, `<home>` from
   `Path.home()` in forward slashes. If the command already has `--settings` (either form), the
   keys are merged into it. `--setting-sources project` does not stop CLAUDE.md from loading.
   `--bare` would, but it cannot use the OAuth login Agni depends on.
3. **Strict, empty MCP** (2026-11-03). `--strict-mcp-config`, plus `--mcp-config={"mcpServers":{}}`
   when the caller names no config of its own (a caller's config is kept). Without it the
   user-scope MCP servers loaded, Lyra's memory server among them, so the reviewer could read or
   write the author's memory.
4. **No auto-memory** (2026-11-03). `"autoMemoryEnabled": false` in the same `--settings` JSON,
   and `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` in `env`. `claudeMdExcludes` covers CLAUDE.md files,
   not the auto-memory index: run from the home folder, an isolated session still loaded
   `~/.claude/projects/<project>/memory/MEMORY.md` and answered a question about a person named
   only there. Each switch alone turned that off; both are set in case a CLI release stops
   honouring one. A caller's `"autoMemoryEnabled": true` is overridden.
5. **No skills** (2026-11-03). `--disable-slash-commands` ("Disable all skills" in `claude --help`),
   unless the caller already passes it. Without it the session loaded the account's user-level
   skills and slash commands (43 and 85 in its init event); a reviewer that invoked one would be
   following the owner's instructions.

Any new CLI-based judge should use the same helper.

**Checking it.**

```bash
python tools/agni/test_isolation.py          # offline: the command infer_cli builds
python tools/agni/test_isolation.py --live   # eleven haiku calls: isolated runs vs controls
```

The offline test fails on the pre-11-02 `agni_phases.py` (point `AGNI_PHASES_PATH` at a copy
from git), on a version that drops `env=`, on a helper that drops the `/**` exclusion, and
(planted 2026-11-03, one at a time) on a helper that drops the auto-memory env var, the
`--disable-slash-commands` flag, or the auto-memory key from a caller's merged `--settings`.

Every live probe runs beside a control that shows it can fail, and reports that it cannot tell
if the control finds nothing. The CLAUDE.md probe asks for the memory namespace in a
"Quick Reference" section without naming it. The MCP probes ask for `mcp__<server>__` tool
names and read the CLI's stream-json init event. The auto-memory probe runs from the home folder
and asks for the first heading of a memory index file in context, naming none; its control (the
helper's command with only the auto-memory switches removed) must give the heading the file
actually starts with, read at run time. The init event from the home folder must show no
`memory_paths.auto` and 0 skills and 0 slash commands. On 2026-11-03 every isolated run answered
`NONE` or reported nothing loaded, and every control found what it looked for (namespace, 28 MCP
servers, the index heading, 43 skills and 85 slash commands).

**Not covered.** The account's personal plugins still load (the isolated init event lists four).
`--disable-slash-commands` removes their skills; whatever else a plugin carries (hooks, agents)
is not checked here. A `CLAUDE.md` in the folder a reviewer runs from loads as project memory,
as it should for a repository; run reviewers from the repository, not from the home folder.

**Not changed.** Four archived paper harnesses also run `claude -p` without the CLAUDE.md
exclusion. They already pass `disableAllHooks`, so they never ran the capture or sync hooks, but
the user-level CLAUDE.md of the account that ran them, if it had one, was in context (not
checked here). They are left as they are because
each is the record of a completed run: three are hash-pinned, and editing them would make the
code differ from what produced the published numbers.

- `mnemosyne-longmemeval-v10/code/judge.py` and `code/answer_one_arm.sh` (pinned in `code/SHA256SUMS`)
- `logit-bias-confab/supplementary/base_rerun/rerun_judge.py` (pinned in `FROZEN.sha256`)
- `logit-bias-confab/supplementary/rejudge_20260930/rejudge.py` (the record behind `REJUDGE_RESULTS.md`)

A rerun of any of these should import the helper, and say so in its protocol.
