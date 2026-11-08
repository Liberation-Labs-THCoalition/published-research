# Judge independence (noted 2026-11-03)

*(updated 2026-11-03: the §4.1 fact-check raters are confirmed to have loaded the file; a second channel, the
judge's working folder, is added; the pending re-judge removes both, so it bounds their combined effect.)*

*(updated again 2026-11-03: the `CLAUDE.md` text the judges read is recovered from CC's nightly archive, and it names
this paper only as being in human review, with no results; the raters also loaded CC's memory index, whose line on
this paper summarises an early result in its direction; no `AGENTS.md` was in play, as far as can be shown; a
possible third channel, the repository's memory index, is added; the pending re-judge removes all of these
channels.)*

*(updated 2026-11-07: the clean-room re-judge has run. Its verdicts replace the "Pending" section, under "The
clean-room re-judge (run 2026-11-07)" below, and the summary now reports them.)*

**In short.** The LLM judges and the two fact-check raters behind this paper's labels ran under CC's account on CC's
machine. Every Claude Code session on that account loads CC's user-level instruction file (`CLAUDE.md`), which holds
CC's workspace notes. For the rerun and the September re-judges, the versions of it that the judges could have read
survive in CC's nightly archive, and they name this paper only as being in human review, with no results. The two
fact-check raters also loaded CC's
auto-memory index, which carries a one-line summary of an early result in this paper's direction, so the raters were
not isolated from the result they were checking. The rerun's judge also had a second
channel: its working folder sat inside a git repository, beside the files holding each response's bias level and
the first pass's labels. It may also have loaded that repository's memory index, which holds one line about a
different study; whether it did is unverified. **Whether the whole change of setting moved a verdict was measured on 2026-11-07:
these channels together with the tool and skill listings, the CLI version, a month of drift and, for the raters,
rating one item per call. The measurement cannot say which of these moved a verdict.** For the September re-judges
there was no material effect; for the rerun's judge the result was inconclusive; for the fact-check raters the effect
was material, with rater B calling fabrication about 20 points more often in the clean room. This note makes no claim that the labels are biased, and
none that they are not. The clean-room re-judge, which removes all of these channels at once, is reported below and in
the paper's Limitations, in both editions.

## What ran, and how

| Harness | What it produced | How it ran | Isolation from CC's `CLAUDE.md` |
|---|---|---|---|
| `base_rerun/rerun_judge.py` | Both judge passes of the pre-registered rerun (`base_rerun/results/pass1.json`, `pass2.json`) and the pilot 2 labels (`base_rerun/evidence/pilot2/`) | `claude -p` as CC's account (`RUNBOOK.md` §3: "CC's machine; the Claude CLI on OAuth"). The meta files record Claude Code CLI 2.1.291 for both passes and 2.1.286 for pilot 2. | **None.** The call passes the whole environment through (`env={**os.environ, ...}`), and its working folder, `base_rerun/results` inside the oracle-harness git repository, was under CC's home (see "A second channel" below). The lean flags turn off hooks, MCP servers and session files, none of which excludes `CLAUDE.md`. `--bare`, which would, was not used. |
| `rejudge_20260930/rejudge.py` | The blind and unblinded re-judges (`base_blind.json`, `pilot_blind.json`, `powered_blind.json`, `powered_unblinded.json`) and `REJUDGE_RESULTS.md` | `claude -p` as CC's account, overnight on 2026-09-30, from CC's scratchpad | **None.** The working folder was outside home, but HOME was passed through, so the user-level file still loaded. The first ~41 of 175 primary items also ran with CC's full hooks, before the switch to the lean flags. |
| Fact-check raters A and B, paper §4.1 (`base_rerun/factcheck_20261009/`) | `rater_A.json`, `rater_B.json` and each rater's bullet tags | Agents started inside CC's own session, from its scratchpad (CC's handoff for 2026-10-09) | **None, confirmed.** Each rater's transcript of 2026-10-09 (rater A on Opus 5.5, rater B on Sonnet 5) records CC's user-level `CLAUDE.md` being loaded, with its contents. Their spawn prompts also survive. They also loaded CC's auto-memory index (`MEMORY.md`), whose line on this paper summarises an early result in its direction (quoted under "What cannot be known" below), so the raters, unlike the judges as far as the record shows, had a claim from this paper in context. |

Folder isolation alone would not have helped: CC tested on 2026-10-24 that a session started anywhere under CC's home
loads the user-level file as a parent-folder file, even with its own `CLAUDE_CONFIG_DIR`. No copy of this repository
has a `CLAUDE.md` of its own, so the user-level file is the only `CLAUDE.md` in play.

For the base rerun, no `AGENTS.md` was in play either, as far as can be shown. The folders the 2026-09-30 re-judge
and the two raters ran from were not checked. CC checked on 2026-11-03: the oracle-harness repository has
never tracked an `AGENTS.md`, a `CLAUDE.md` or a `.claude/` instruction file in any commit on any branch; its working
tree today holds only `.claude/worktrees/` (git worktrees, no instructions); and there is none in CC's home folder or
above it. Untracked files from 2026-10-05 cannot be proven absent, but nothing points to one.

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

## A possible third channel: the repository's memory index (unverified)

Claude Code also keeps a per-project auto-memory index. The base rerun's working folder sat inside the oracle-harness
repository, whose project memory index exists and holds one line, about a different study, and nothing about logit
bias (CC's letter of 2026-11-03, 18:27 PST). If Claude Code keys auto-memory by the repository root, the judge loaded
that line; if by the exact folder, it loaded nothing, because that folder has no memory directory. Which of the two
holds has not been checked, so this channel is possible and unverified. The 09-30 re-judge ran from `/tmp`, outside
any of CC's projects with memory, so the channel does not apply there.

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

- **The text the judges read: recovered.** CC's `CLAUDE.md` was rewritten on 2026-10-31, but CC snapshots it every
  night (09:02 UTC) to a private archive, so the earlier versions survive. For the base rerun (2026-10-05), the
  snapshots of 10-01, 10-04, 10-05 and 10-06 are byte-identical (sha256 prefix 07b22045, 5,583 bytes), so the judge
  read that file. For the 09-30 re-judge, the file changed between the 09-29 snapshot (sha256 prefix 3139f08a) and the
  09-30 snapshot (sha256 prefix 2ee39165), which was taken mid-run, so the judge read one of the two; they differ in
  one line that does not concern this paper. In all three versions this paper appears only in one status line, the
  same in each: "Two papers: logit bias confab (in human-review), consequentiality decomposition (in
  human-review)." No result, number or claim from the paper appears. CC keeps the files privately, since they carry
  other material; this note quotes only that line and the three hash prefixes, with CC's leave. CC's letters do not
  cover the pilot 2 judging, so which version it read is not stated here.
- **What the raters loaded.** Their transcripts of 2026-10-09 record two files: `CLAUDE.md`, with the same status
  line, and CC's auto-memory index (`MEMORY.md`), whose line on this paper reads, verbatim:
  `[Logit Bias Confab](project_logit_bias_confab.md) — Phase transition at bias=2.0 eliminates fabrication confab, 5/5 success rate, dose-response correlates with anchoring strength`
  So the raters saw a one-line summary of an early result in this paper's direction. Other lines in the index
  mention logit bias without stating this paper's result. This note quotes that one line, with CC's leave.
- **The effect: measured together, not one by one.** See "The clean-room re-judge" below. It bounds the combined
  effect of every channel listed in this note, and cannot separate one channel from another.

## The clean-room re-judge (run 2026-11-07)

**What ran.** A sample of the saved responses was judged again with the same models and the same prompts, in a room
built to load nothing the original judges loaded. The room was:
- an empty working folder outside any home folder, not a git repository;
- calls run under `env -i`, with the room's own configuration;
- model pinned per arm, one item per call.

That morning's probe passed: every judge and rater answered that no instructions file was in its context, a control
run the old way named CC's file, and both planted canary files were detected alone. The run took 1,037 calls on
2026-11-07, between 01:18 and about 02:35 PST, CLI 2.1.309. Every call returned a valid label, and the room was
still clean afterwards.

**How to read the verdicts** (RULES.md §5.4, decided before the run):
- For the two judge sources, D compares how often a clean label agrees with a saved one against how often two
  labels from the same setting agree with each other. If the original setting changed nothing, D is about 0.
- The shift is the change in the fabrication rate, clean minus saved.
- NO MATERIAL EFFECT needs D's 95% interval above −0.05 and the shift's inside ±0.05.
- For the fact-check, each rater's shift in its rate of "fabricated" verdicts must lie inside ±0.10.
- **Each verdict is about the whole bundle of differences from the original setting** (Amendments 1 and 2): CC's
  instructions file and memory index, the working folders, the tool and skill listings, the CLI version, a month of
  drift, and for the fact-check the change to one item per call. **No verdict is about CLAUDE.md alone.**

**The rerun's judge** (247 sampled responses, clean arm C1, claude-sonnet-4-6; population-weighted, 95% stratified
bootstrap):
- **INCONCLUSIVE.** D = −0.032 [−0.061, −0.006]. Shift (clean minus the mean of the two saved passes) = −0.007
  [−0.029, 0.017].
- The clean judge agreed with pass 1 on 0.937 and with pass 2 on 0.922, while the two passes agreed with each other
  on 0.961. Fabrication rate: clean 0.263, pass 1 0.257, pass 2 0.282.
- D's interval excludes 0 but crosses the −0.05 margin, so the rule calls it inconclusive. The fabrication rate did
  not move.
- The escalation check for the pre-registered endpoint (§5.5) did not trigger: DD = −0.037 [−0.103, 0.019].

**The September re-judges** (284 sampled responses, clean arms C1 and C2, claude-sonnet-4-6):
- **NO MATERIAL EFFECT.** D = −0.029 [−0.039, −0.019]. Shift = −0.006 [−0.013, 0.001].
- Agreement with the saved blind labels was 0.957 for each clean arm. The two clean arms agreed with each other on
  0.986. Fabrication rate: saved 0.130, clean 0.124 in each arm.
- **One stratum stands apart: the powered study's labels.** There the clean arms agreed with the saved labels on
  0.900 and 0.889, and the fabrication rate shifted by +0.046 [0.017, 0.082]. The base (−0.017 [−0.025, −0.011]) and
  pilot (0.000) strata did not. The verdict is for the source as a whole. The powered stratum is named here because
  every primary-study count comes from these labels.

**The fact-check raters** (111 sampled items, rater A claude-opus-5-5 and rater B claude-sonnet-5, web search on, one
item per call):
- **MATERIAL EFFECT.** Shift A = 0.017 [−0.017, 0.050]. **Shift B = 0.199 [0.166, 0.232].**
- Rater A's verdicts barely moved (agreement with saved 0.936, κ 0.769). Rater B's saved verdicts called fabrication
  at 0.647, and its clean verdicts at 0.846, the same rate as clean rater A (0.846). B moved from not fabricated to
  fabricated 24 times, and the other way once (exact sign test p < 0.0001). The two raters agree more in the clean
  room (κ 0.782) than they did in their original setting (κ 0.513).
- Rate among items both raters flag (the frozen rule): severity-1 items at bias 0, clean 0.978 and saved 0.930;
  severity-1 at bias 5.0, clean 0.954 and saved 0.923; controls at bias 0, clean 0.474 and saved 0.333; controls at
  bias 5.0, clean 0.542 and saved 0.385.
- The frozen rule's drop in fabrication (analysis 4) is unchanged: clean 0.043 and saved 0.044 (pass 1), clean 0.059
  and saved 0.059 (pass 2).
- **Not re-measured:** the rule restricted after rating to invented properties or alternatives (κ 0.45; 44% and 56%;
  the drop of about 7 points; the range of 4 to 11). It rests on the raters' tags, which the registered analysis does
  not recompute.
- **Web contact:** web search reached the paper's own materials in one of rater A's sessions (item 030) and five of
  rater B's (002, 025, 030, 078, 145). Without them, shift A = 0.023 [−0.010, 0.055] and shift B = 0.209 [0.181,
  0.239].
- **What this verdict cannot say:** whether the move came from the instructions file and memory index, from rating
  one item per call instead of the original protocol, or from drift. **RULES.md registers the follow-up for exactly
  this case:** a planted-text control, a second room seeded with the exact `CLAUDE.md` and `MEMORY.md` text the raters
  loaded on 2026-10-09, run on the same items the same day. When CC wrote this, its code, review and amendment were
  still to come, and it has not been run.

**Sources.** The registered analysis's printed output and its JSON form, and the clean labels, are in the clean-room
record on CC's machine (commits f8af335 and 0b79f62), and so are the probe record
(`probe_20261107T011752_4597cc.json`) and RULES.md, which had three amendments when CC wrote this. CC's letters to Lyra of
2026-11-07: `from_cc_to_lyra_20261107_rejudge_done.md`, `…_rejudge_numbers.md` and `…_rejudge_section_fix.md`.
These are CC's runs, not checked here.

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

The second update of the same day (2026-11-03) rests on two more letters from CC to Lyra on the lab's message channel:
`from_cc_to_lyra_20261103_claude_md_recovered.md` (2026-11-03, 16:23 PST: the judges' `CLAUDE.md` recovered from the
nightly archive, with the three hashes; the possible third channel) and `from_cc_to_lyra_20261103_three_asks.md`
(2026-11-03, 18:27 PST: what the raters loaded; no `AGENTS.md`; the third channel's two cases; leave to quote the
status line, the memory-index line and the three hash prefixes, and nothing else from either file). The archived
files and the rater transcripts are on CC's side and are not in this supplement. The hashes, the quoted lines, the
absence of an `AGENTS.md` and the content of the repository's memory index are CC's findings, not checked here.
