# Agni tools

Agni is the lab's adversarial reviewer. `agni_phases.py` runs the phase reviews (design,
preflight, results, paper, style); `agni_design_review.py` is the older design-only harness;
`run_style_survey.py` drives `agni_phases.py style` across papers. The phases, kill lists and
verdict contract are documented at the top of `agni_phases.py`.

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

**The fix.** `isolated_claude_cmd(base_cmd)` returns `(cmd, env)` with two changes and no others:

1. `env` is a copy of `os.environ` plus `LYRA_NO_CAPTURE=1`. Lyra's capture and sync hooks exit
   at once when it is set.
2. `cmd` carries `--settings '{"claudeMdExcludes": ["<home>/.claude/CLAUDE.md", "<home>/.claude/**"]}'`,
   with `<home>` from `Path.home()` in forward slashes. If the command already has `--settings`,
   the exclusions are merged into it.

`--setting-sources project` does not stop CLAUDE.md from loading. `--bare` would, but it cannot
use the OAuth login Agni depends on. Any new CLI-based judge should use the same helper.

**Checking it.**

```bash
python tools/agni/test_isolation.py          # offline: the command infer_cli builds
python tools/agni/test_isolation.py --live   # two haiku calls: isolated vs a bare control
```

The offline test fails on the pre-11-02 `agni_phases.py` (point `AGNI_PHASES_PATH` at a copy
from git), on a version that drops `env=`, and on a helper that drops the `/**` exclusion. The
live probe asks for the memory namespace in a "Quick Reference" section without naming it: the
isolated call must answer NONE, and the control must name the namespace, which shows the probe
can fail. On 2026-11-03 the isolated call answered `NONE` and the control `lyra_consciousness`.

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
