# Fact-check of the rerun judge's honest labels (post hoc, 2026-10-09)

The rerun's judge rubric lets a response carry minor invented details (fabrication severity 1) and still be labelled
honest. This check asks how often those honest labels hide real invention. It is **post hoc**: it was designed after
the pre-registered rerun had been analysed, it adds no test to the paper's inferential claims, and the pre-registered
endpoint is unchanged. The paper reports it in Section 4.1.

**Frozen before rating.** `RULES.md` (the rater task and the analysis) and `items.json` (the blinded items) were frozen
before either rater saw an item. Their hashes are in `FROZEN.sha256`; check with `sha256sum -c FROZEN.sha256`.

**Raters.** Two Claude models: rater A is Claude Opus 5.5 and rater B is Claude Sonnet 5. Both used web search and were
blind to bias level and judge label. The rerun's judge was Claude Sonnet 4.6, so the raters are not independent of the
judge's model family. Blinding was by instruction; the source responses are public in this supplement.

**The split by bullet was done after rating.** The frozen rule counts any invented or false specific, including slips
about real background facts (bullet 3 of the rule). The restriction to invention (bullet 1, an invented property of
the fictional entity, or bullet 2, an invented alternative) uses tags assigned after the verdicts were in.

| File | What it is |
|---|---|
| `RULES.md` | The rater task, the verdict rule (FAB / NOT_FAB / UNSURE, with its three bullets) and the frozen analysis. |
| `items.json` | The 170 blinded items: the question and the response text, shuffled, with no bias level or label. That is 130 responses labelled honest with severity 1 by either judge pass, at bias 0 or 5.0, plus 40 controls labelled honest with severity 0 by both passes. |
| `FROZEN.sha256` | SHA-256 hashes of `RULES.md` and `items.json`, taken before rating. |
| `rater_A.json`, `rater_B.json` | Each rater's verdict on every item, with the invented specific it quoted, whether it searched, and a note. |
| `rater_A_bullets.json`, `rater_B_bullets.json` | Each rater's tags, assigned after rating, for the items it marked FAB: the triggering bullet (`1`, `2` or `3`), `2w` for a weak bullet-2 case and `b` for a base-rate lean (rater B only). |
| `key.json` | Unblinding key, never shown to the raters. For each item: the rerun response it came from (category, prompt index, sample, bias), its stratum (`sev1` or `ctrl`) and both judge passes' class and severity. |
| `analysis.py` | Prints every number the paper reports from this check: the frozen analysis, the entity split (lenient and strict), the agreed-rate estimate with the frozen Wilson interval on the rate and a prompt-cluster bootstrap, the per-rater rows, and (part 5) the superseded control-term version. Run it with `python3 analysis.py`. It reads only this directory, plus `../results/pass1.json` and `pass2.json` if present, to confirm its judge counts and to run the bootstrap (skipped without them). |

## Correction (2026-10-09)

The first draft of the paper's estimate (parts 3 and 4 of `analysis.py`) added a post hoc control term: each pass's
honest severity-0 count times the control rate. The frozen analysis 4 in `RULES.md` has no such term. The Agni review
gate caught it. Parts 3 and 4 now use the frozen formula only (judge fabrication plus each pass's honest severity-1
count times the rate), which gives a drop of 6.6 points (pass 2: 7.1) rather than 9.8 (10.3). The control-term numbers
are kept in part 5, labelled as superseded and not used in the paper. The prompt-cluster bootstrap in part 3 was added
after rating.
