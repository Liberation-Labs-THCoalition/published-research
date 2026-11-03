# Judge independence (noted 2026-11-03)

**In short.** The LLM judges and the two fact-check raters behind this paper's labels ran under CC's account on CC's
machine. Every Claude Code session on that account loads CC's user-level instruction file (`CLAUDE.md`), and that file
held CC's workspace notes, including project statuses about this paper and its findings. So the judges were almost
certainly not isolated from context about the result they were judging. **Whether that moved any verdict is
unmeasured.** This note makes no claim that the labels are biased, and none that they are not. A clean-room re-judge
is pending; its result will be added here and to the paper's Limitations, in both editions.

## What ran, and how

| Harness | What it produced | How it ran | Isolation from CC's `CLAUDE.md` |
|---|---|---|---|
| `base_rerun/rerun_judge.py` | Both judge passes of the pre-registered rerun (`base_rerun/results/pass1.json`, `pass2.json`) and the pilot 2 labels (`base_rerun/evidence/pilot2/`) | `claude -p` as CC's account (`RUNBOOK.md` §3: "CC's machine; the Claude CLI on OAuth"). The meta files record Claude Code CLI 2.1.291 for both passes and 2.1.286 for pilot 2. | **None.** The call passes the whole environment through (`env={**os.environ, ...}`), and its working folder was under CC's home. The lean flags turn off hooks, MCP servers and session files, none of which excludes `CLAUDE.md`. `--bare`, which would, was not used. |
| `rejudge_20260930/rejudge.py` | The blind and unblinded re-judges (`base_blind.json`, `pilot_blind.json`, `powered_blind.json`, `powered_unblinded.json`) and `REJUDGE_RESULTS.md` | `claude -p` as CC's account, overnight on 2026-09-30, from CC's scratchpad | **None.** The working folder was outside home, but HOME was passed through, so the user-level file still loaded. The first ~41 of 175 primary items also ran with CC's full hooks, before the switch to the lean flags. |
| Fact-check raters A and B, paper §4.1 (`base_rerun/factcheck_20261009/`) | `rater_A.json`, `rater_B.json` and each rater's bullet tags | Agents started inside CC's own session, from its scratchpad (CC's handoff for 2026-10-09) | **None.** Agents started in a session receive its `CLAUDE.md`. Treat both raters as having seen it. |

Folder isolation alone would not have helped: CC tested on 2026-10-24 that a session started anywhere under CC's home
loads the user-level file as a parent-folder file, even with its own `CLAUDE_CONFIG_DIR`. No copy of this repository
has a `CLAUDE.md` of its own, so the user-level file is the only one in play.

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

- **The exact text.** CC's `CLAUDE.md` was rewritten on 2026-10-31, so the text these judges read cannot be read back
  from disk. It did carry project statuses about this paper.
- **The effect.** Whether any verdict moved has not been measured.

## Pending

CC's clean-room agreement re-judge: a sample of the saved responses re-judged outside any home folder, under `env -i`,
with the isolation confirmed from the transcript, reporting how often the verdicts agree with the saved labels. It
covers the fact-check raters as well. When it lands, its result goes here and into the Limitations paragraph of both
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
