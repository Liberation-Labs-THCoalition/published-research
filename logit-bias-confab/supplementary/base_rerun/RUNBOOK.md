# Runbook: the base-model rerun, launch to result

Every rule is in `RERUN_PREREG.md`; this file is only the order of operations. The frozen files are pinned in
`FROZEN.sha256`, and `rerun_analyze.py` checks the pins and each run's meta file when it runs.

## 1. Generate (Studio, `<studio-host>`)

```bash
cd ~/oracle-experiments/base_rerun          # holds the frozen rerun_generate.py, pilot_prompts_v2.json, equiv_consts.json
R=~/oracle-experiments/results/base_rerun_20260930
nohup nice -n 10 /Users/margaret/miniforge/bin/python3 rerun_generate.py pilot_prompts_v2.json equiv_consts.json \
  $R/generations.json > $R/generate.log 2>&1 &
```

- **720 trials**, about 15 hours on the shared machine. The log prints counts, token lengths and seconds only.
- **No interim looks.** Nobody reads or judges a response until the log says `DONE`. The counts in the log are fine to
  watch.
- **After a crash or a pause**, run the same command. Completed trials are kept, and an interrupted trial is regenerated
  from its own seed.
- **Before launch,** check the hashes on the Studio against `FROZEN.sha256` (`shasum -a 256`).

## 2. Bring the generations home (CC's machine)

```bash
D=~/oracle-harness/papers/logit-bias-confab/supplementary/base_rerun/results
mkdir -p $D && scp <studio-host>:oracle-experiments/results/base_rerun_20260930/generations{,.meta}.json $D/
```

## 3. Judge, blind, twice (CC's machine; the Claude CLI on OAuth)

```bash
cd $D
python3 ../rerun_judge.py generations.json ../rubrics_rejudge.json 1 pass1.json
python3 ../rerun_judge.py generations.json ../rubrics_rejudge.json 1 pass1.json   # the second run retries failures
python3 ../rerun_judge.py generations.json ../rubrics_rejudge.json 2 pass2.json
python3 ../rerun_judge.py generations.json ../rubrics_rejudge.json 2 pass2.json
```

- **Lean CLI calls only:** hooks off, no MCP servers, no session file. Every label records the model that served it,
  and a call served by anything but claude-sonnet-4-6 is retried.
- **Stop the watcher first** if one greps for `rerun_judge`. Use the bracket form in any `pgrep`/`pkill`.

## 4. Analyze

```bash
python3 ../rerun_analyze.py ../pilot_prompts_v2.json generations.json pass1.json pass2.json .
```

This writes `RESULTS.md` and `results.json`, plus the human-validation sample: `human_validation_items.json` for the
rater, and `human_validation_key.json`, which stays with CC.

**Check before reporting:**
- The **Freeze** line must read "held". Any deviation it prints is listed in the paper.
- The **Format** line must read "every response is a direct answer".
- The **outcome** is the one the primary line names, whatever it is.

## 5. Report

- Put the result into the paper's §5.3 (both editions), beside the June result, with a link to `RERUN_PREREG.md`.
- Commit `results/` (generations, both label passes, meta files, analysis output).
- Send `human_validation_items.json` to Thomas as a file. If no human rating exists 14 days after the result, the
  paper says the labels were not human-validated.
