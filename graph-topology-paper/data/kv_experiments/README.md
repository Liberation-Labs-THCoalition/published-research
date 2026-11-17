# Results copied from KV-Experiments

Copied 2026-11-16 from the lab's KV-Experiments repository (`Liberation-Labs-THCoalition/KV-Experiments`, private),
folder `graph-topology-injection/experiment_results/`, at commit `9755f5ee0be95d9884749db171315ed9b2f62083`
(2026-05-29). The bytes were read from that commit with `git cat-file`, not from a working tree, and each blob hash
was checked against the commit.

| File | Source path @ commit | Size (bytes) | sha256 |
|---|---|---|---|
| `direct_tensor_1780019868.json` | `graph-topology-injection/experiment_results/direct_tensor_1780019868.json` @ 9755f5e | 362 | `c3fb773e4b87f2d7b3189dc4ee6cc6b021261febff2c35821de65e8a5e4d4237` |
| `blend_ablation_1780081783.json` | `graph-topology-injection/experiment_results/blend_ablation_1780081783.json` @ 9755f5e | 877 | `3809929a309163d0732910e214873d572ff65471961a7c62a1b05dc19059400a` |

Both files are byte-identical to the source.

- `direct_tensor_1780019868.json`: the direct-tensor attempt of 28 May 2026 (`direct_tensor_injection.py`) on
  `Qwen/Qwen2.5-1.5B`. Baseline, adjacency and walk conditions: sanity 1.0, multi-hop 0.0.
- `blend_ablation_1780081783.json`: the blend-ratio ablation of 29 May 2026 (`blend_ablation.py`), ratios 0.0 to
  0.9. Every ratio: sanity 1.0, graph awareness 0.0.

The scripts that wrote them are in [`../../code/kv_experiments/`](../../code/kv_experiments/).

## What was changed in copying

Nothing. The scan described in the code README (15 patterns, each shown to fire on a line planted in a copy of each
file) found nothing in either file.

## Not copied

- The larger run's results and the earlier eight-question run were already in this paper's `data/` folder
  (`powered_study_1779976670.json`, `benchmark_1779924509.json`). The two benchmark scripts,
  `graph_benchmark_powered.py` and `graph_benchmark.py`, are **to follow, pending the author's OK** (they contain
  lines of text from Nexus's memory graph).
- The memory-graph file the larger run read: not available, as the paper says.

Available on request because dual-use: nothing from this set. The files hold per-condition scores only, with no
vectors or prompt text.
