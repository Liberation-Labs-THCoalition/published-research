# Re-run of the judged follow-up, 20 October 2026

The record behind the paper's paragraph "Two defects in the script explain much of this" (Larger Follow-Ups) and the
abstract's sentence on the one-token difference.

| File | What it is |
|---|---|
| `PREREG.md` | The preregistration, last edited 14:44 PDT on 20 October 2026, before either attempt ran. |
| `dig.py` | The re-run's script. |
| `inputs.sha256` | SHA-256 of the script, the prereg and the seven input files. |
| `guard_mth.log`, `dig_mth_stopped_by_guard.log` | A first attempt on another machine (21:38 PDT), stopped by a memory-pressure guard before its first condition finished. |
| `dig_studio.log` | The run that produced the results (21:41 to 21:49 PDT, CPU, fp32). |
| `results_studio.json` | Its results. |
| `RESULTS.md` | The summary written that night, after a second reader (a separate Claude instance) recomputed every comparison from the raw strings. Its "not yet acted on" section is as of that night: version 1.1 of the paper acts on the items about this paper; the shipped MLX path is outside it and untested. |

**Inputs.** The seven input files (`src/ethics_packs/...`, `src/experiment_results/llm_judged_1780088341.json`) are in
the lab's KV-Experiments repository, folder `graph-topology-injection/`, at commit `a439e2b` (on `main`). Their hashes
in `inputs.sha256` (listed there under `./src/`) match that commit. To re-run, set `DIG_SRC` to that folder;
the script's default is `~/KV-Experiments/graph-topology-injection`.

**One edit to the record.** The first two lines of `dig_studio.log` were a urllib3 `NotOpenSSLWarning` naming a local
path on the machine that ran it. They are removed. Nothing else is changed.

**Scope.** One model (Qwen/Qwen2.5-1.5B), CPU fp32, Hugging Face transformers. It covers FULL_KV and TEXT_CONTEXT,
not K_ONLY or V_ONLY, and it does not re-run the pilot.
