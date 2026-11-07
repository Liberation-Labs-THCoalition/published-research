# Agni tools

Agni is the lab's adversarial reviewer. `agni_phases.py` runs the phase reviews (design,
preflight, results, paper, style); `agni_design_review.py` is the older design-only harness;
`run_style_survey.py` drives `agni_phases.py style` across papers. The phases, kill lists and
verdict contract are documented at the top of `agni_phases.py`.

## No silent cuts

**The rule: a review may never look complete when the reviewer did not get all of its input.**
Until 2026-11-07 a reviewer that saw part of its input still returned a confident verdict, and nothing
recorded the cut.

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
- **Preregs.** Hypotheses and threats go whole. A prereg with neither `hypotheses` nor `hypothesis`
  goes whole as secondary. Its other keys are not sent: they are named as `prereg_keys_not_sent`, in
  the record and in the COVERAGE block, and they make `complete` false. Most real preregs will read
  `input_complete: false` for that reason, and that is what the reviewer really got.
- **The verdict comes from a verdict line, not from a word.** The contract is at the top of
  `agni_phases.py`. Reasoning prose never counts, and verdict lines that disagree give `UNPARSED`.
- **Local path.** `llm.py` sets `num_ctx` (`LLM_NUM_CTX`, default 32768). Unset, Ollama trims a prompt
  to its context length with no signal. Anything bound for Ollama over `AGNI_LOCAL_CHAR_BUDGET`
  (default 60000 chars) is refused and never sent cut. The `DEGRADED` stamp is whole again.
- **Timeouts.** `AGNI_CLI_TIMEOUT` defaults to 1800 s, and a timeout is retried at most once. Other
  retry behaviour is unchanged.
- `run_style_survey.py` counts `INVALID` as a failure, and also any exit other than 0 or 1, whatever
  the file on disk says. `summarize_style.py` shows `INVALID - NOT REVIEWED`.

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
  `agni_phases.py`, but it does not stop the call.
- The char budgets are proxies. Tokens are not measured, so a very dense 60000 chars could still pass
  `num_ctx` in principle.
- A local server reached through `AGNI_LLM_BACKEND=openai` is not recognised as local, so the local
  budget does not apply to it.
- `agni_design_review.py` follows the rule for cuts only. Its `parse_verdict` still falls back to a
  verdict word anywhere in the text.

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
