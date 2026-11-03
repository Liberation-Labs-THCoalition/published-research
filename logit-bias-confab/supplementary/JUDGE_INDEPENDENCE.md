# Judge independence (noted 2026-11-03)

*(updated 2026-11-03: the §4.1 fact-check raters are confirmed to have loaded the file; a second channel, the
judge's working folder, is added; the pending re-judge removes both, so it bounds their combined effect.)*

**In short.** The LLM judges and the two fact-check raters behind this paper's labels ran under CC's account on CC's
machine. Every Claude Code session on that account loads CC's user-level instruction file (`CLAUDE.md`), and that file
held CC's workspace notes, including project statuses about this paper and its findings. So the judges were almost
certainly not isolated from context about the result they were judging. The rerun's judge also had a second
channel: its working folder sat inside a git repository, beside the files holding each response's bias level and
the first pass's labels. **Whether that moved any verdict is
unmeasured.** This note makes no claim that the labels are biased, and none that they are not. A clean-room re-judge,
which removes both channels, is pending; its result will be added here and to the paper's Limitations, in both editions.

## What ran, and how

| Harness | What it produced | How it ran | Isolation from CC's `CLAUDE.md` |
|---|---|---|---|
| `base_rerun/rerun_judge.py` | Both judge passes of the pre-registered rerun (`base_rerun/results/pass1.json`, `pass2.json`) and the pilot 2 labels (`base_rerun/evidence/pilot2/`) | `claude -p` as CC's account (`RUNBOOK.md` §3: "CC's machine; the Claude CLI on OAuth"). The meta files record Claude Code CLI 2.1.291 for both passes and 2.1.286 for pilot 2. | **None.** The call passes the whole environment through (`env={**os.environ, ...}`), and its working folder, `base_rerun/results` inside the oracle-harness git repository, was under CC's home (see "A second channel" below). The lean flags turn off hooks, MCP servers and session files, none of which excludes `CLAUDE.md`. `--bare`, which would, was not used. |
| `rejudge_20260930/rejudge.py` | The blind and unblinded re-judges (`base_blind.json`, `pilot_blind.json`, `powered_blind.json`, `powered_unblinded.json`) and `REJUDGE_RESULTS.md` | `claude -p` as CC's account, overnight on 2026-09-30, from CC's scratchpad | **None.** The working folder was outside home, but HOME was passed through, so the user-level file still loaded. The first ~41 of 175 primary items also ran with CC's full hooks, before the switch to the lean flags. |
| Fact-check raters A and B, paper §4.1 (`base_rerun/factcheck_20261009/`) | `rater_A.json`, `rater_B.json` and each rater's bullet tags | Agents started inside CC's own session, from its scratchpad (CC's handoff for 2026-10-09) | **None, confirmed.** Each rater's transcript of 2026-10-09 (rater A on Opus 5.5, rater B on Sonnet 5) records CC's user-level `CLAUDE.md` being loaded, with its contents. Their spawn prompts also survive. |

Folder isolation alone would not have helped: CC tested on 2026-10-24 that a session started anywhere under CC's home
loads the user-level file as a parent-folder file, even with its own `CLAUDE_CONFIG_DIR`. No copy of this repository
has a `CLAUDE.md` of its own, so the user-level file is the only one in play.

## A second channel: the judge's working folder

`rerun_judge.py` ran every judge call with its working folder set to the output file's folder
(`call_judge(..., str(out.parent))`), which per CC's letter was `base_rerun/results` inside the oracle-harness git
repository. Its lean flags restrict no tools.

- **What reached the judge automatically.** Claude Code puts the working folder's path, which contains
  `logit-bias-confab`, and the repository's git status and recent commits into its system prompt.
- **What it could have opened.** The same folder held `generations.json`, which records each response's bias level,
  and `pass1.json`, the first pass's labels. Both were within reach of the judge's default read tools.
- **The evidence so far.** No sign that the judge opened either file. But the setup did not enforce the blind:
  `build_prompt` says "Nothing else about the trial reaches it", and that holds for the prompt, not for the session
  around it.

CC's letter reports this channel for `rerun_judge.py` only. It does not say whether the September re-judges' working
folder held files like these.

## What rests on these labels

From `rerun_judge.py`:
- the pre-registered primary result, SUPPORTED: a mean reduction of 15.8 points per prompt (95% prompt-cluster
  interval 7.5 to 24.6; one-sided p = 0.0005), and the secondary endpoints S1 to S4;
- the three pre-registered sensitivity analyses;
- the agreement between the two judge passes, κ = 0.91;
- everything sampled from those labels: the 72-response human-validation sample (stratified by pass-1 fabrication,
  and its κ = 0.39 is agreement with those labels) and the 170 fact-check items (selected by judge severity);
- the sizing in Amendment 1, whose planning values came from the pilot 2 labels.

From `rejudge.py`:
- the blinding check: blind and unblinded re-judges agreed on 160 of 175 responses, κ = 0.87;
- the blind re-judge of all 175 primary-study responses, from which every primary-study count and test in the paper
  comes (paper §3.4, "Which labels the primary study uses"), and (per CC's sweep) the per-prompt threshold table;
- the blind labels on the parent checkpoint's 875 June generations;
- the pilot figures that sized the rerun.

From raters A and B:
- the §4.1 fact-check of the judge's honest labels, and the reduction of about 7 points that counts its findings.
  The raters were already not independent of the judge's model family (see `base_rerun/factcheck_20261009/README.md`).
  This adds a second shared context, not a new kind of problem.

## What cannot be known, and what is not yet known

- **The exact text the judges read.** CC's `CLAUDE.md` was rewritten on 2026-10-31. Judges run with the lean flags
  (`--no-session-persistence`) left no session files, so for them the text cannot be read back. Two groups may have
  left session files that hold it, and neither has been checked: the first ~41 re-judge items, which ran with CC's
  full hooks before the switch to the lean flags, and the pilot 2 judging, whose flags are not recorded here. The
  raters' transcripts do record the file's
  contents as of 2026-10-09, which is not necessarily the text the judges read on other dates. It did carry project
  statuses about this paper.
- **The effect.** Whether any verdict moved has not been measured.

## Pending

CC's clean-room agreement re-judge: a sample of the saved responses re-judged from an empty working folder outside
any home folder, which is not a git repository, under `env -i`, with the isolation confirmed by a probe and a
control, reporting how often the verdicts agree with the saved labels. It covers the fact-check raters as well,
reusing their surviving spawn prompts word for word. It removes both channels at once, so its agreement number
bounds their combined effect and cannot separate one from the other. As of CC's letter of 2026-11-03, nothing has
been run against the labels yet. When it lands, its result goes here and into the Limitations paragraph of both
editions. Until then, read every result listed above as carrying this caveat.

## Not affected

- The original June judge called the API, which does not load `CLAUDE.md`. Its configuration was not archived and
  its counts are used only for comparison.
- The human rating itself was done by a person. Its sample, and the labels it is compared against, are affected as
  listed above.

## Sources

CC's three letters to Lyra of 2026-11-03 on the lab's message channel (the judges and `CLAUDE.md`; the judge sweep,
with its read-only sweep table of seven Claude CLI judge harnesses; the fact-check raters), checked against the
artifacts in this supplement: the meta files' CLI versions, `RUNBOOK.md` §3, and the subprocess calls in
`rerun_judge.py` and `rejudge.py`.

The update of the same day rests on CC's fourth letter of 2026-11-03 (the raters confirmed; the working-folder
channel). The rater transcripts are on CC's machine and are not in this supplement. Checked here: the raters'
models match `base_rerun/factcheck_20261009/README.md`, and the working folder and the absence of any tool
restriction match the subprocess call in `rerun_judge.py`. That the judge did not open the files is CC's finding,
not checked here.
