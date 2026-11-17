# Scripts copied from KV-Experiments

Copied 2026-11-16 from the lab's KV-Experiments repository (`Liberation-Labs-THCoalition/KV-Experiments`, private),
folder `graph-topology-injection/`, at commit `9755f5ee0be95d9884749db171315ed9b2f62083` (2026-05-29). The bytes were
read from that commit with `git cat-file`, not from a working tree, and each blob hash was checked against the commit.
Nexus wrote these scripts (see the paper's Author Contributions).

| File | Source path @ commit | Size (bytes) | sha256 |
|---|---|---|---|
| `graph_encoder.py` | `graph-topology-injection/graph_encoder.py` @ 9755f5e | 10,686 | `4dba0133b95349ff87f33fa075878d413008f6cc064abdcb1867581aba1e1490` |
| `graph_experiment.py` | `graph-topology-injection/graph_experiment.py` @ 9755f5e | 11,279 | `83612851eccdddf7703ede59cfad8b6c519f554c0ad28c6593da97413f33d160` |
| `graph_experiment_phase2.py` | `graph-topology-injection/graph_experiment_phase2.py` @ 9755f5e | 11,630 (source 11,635) | `93975059125d66e55879efa587e85068308774e35f65c3a3855f7862c9d0973e` (source `ece698daa93c66a29fae0c22e7dd5b90512668d9c24d0fdcfa12282f8b66d94b`) |
| `direct_tensor_injection.py` | `graph-topology-injection/direct_tensor_injection.py` @ 9755f5e | 13,198 | `bb5b22c8246be0a84646663b6a49491132fa6efb47d65ca9c68bb4126eab5a69` |
| `blend_ablation.py` | `graph-topology-injection/blend_ablation.py` @ 9755f5e | 8,436 | `32dce0fdda359a8fd1b15c723f096f378b8d6d94febe9288c2ff07f67850fff4` |
| `graph_benchmark_powered.py` | `graph-topology-injection/graph_benchmark_powered.py` @ 9755f5e | 17,529 (source 17,533) | `e3cc42e4ea1214d61877f18206f9a20d6bc86bb59691a7fcd63e64e04e01c0f5` (source `f9525284c952b03b7b17d0c276eddeab0c2b71998314ccb658647ca4895678e7`) |
| `graph_benchmark.py` | `graph-topology-injection/graph_benchmark.py` @ 9755f5e | 9,858 (source 9,865) | `449447cb5a8b3651ed26093b60d26801b6630a2a77584ff82544a2f6f952dc08` (source `8bdcf0f7ffb6cea3673278da09c7d7f27f50d3ad832a9144a9e98f3e3b870790`) |

The two benchmark scripts were added later the same day, read from the same commit in the same way. Four of the seven
files are byte-identical to the source. The results of the two cache-injection attempts are in
[`../../data/kv_experiments/`](../../data/kv_experiments/).

## What each script is

- `graph_encoder.py`: the three encodings (adjacency, spectral, walk) and the 21-node pilot graph. It mentions
  `kv_packs.py` in its docstring but does not import it.
- `graph_experiment.py`: Phase 1 (imports `graph_encoder`; queries a model through Ollama at `OLLAMA_URL`).
- `graph_experiment_phase2.py`: Phase 2 with the scrambled controls (imports `graph_encoder`).
- `direct_tensor_injection.py` and `blend_ablation.py`: the two cache-injection attempts on `Qwen/Qwen2.5-1.5B`
  (28 and 29 May 2026), both of which scored 0 on the graph questions.
- `graph_benchmark_powered.py`: the larger run (results in
  [`../../data/powered_study_1779976670.json`](../../data/powered_study_1779976670.json)). Four conditions (no
  context, the walk-encoded memory graph, the same facts as flat text, and unrelated prose cut to the graph's length),
  10 sanity and 36 multi-hop questions, three runs, through Ollama at `OLLAMA_URL`.
- `graph_benchmark.py`: the earlier eight-question run on the memory graph, baseline against injected (results in
  [`../../data/benchmark_1779924509.json`](../../data/benchmark_1779924509.json)).

Both benchmark scripts read the walk-encoded memory graph from a text file, `lab_graph_walk_encoding.txt`, in `/tmp`
or the home directory. That file is not available, as the paper says.

## What was changed in copying

Four substitutions, in three files:

| File | Line | Before | After | Why |
|---|---|---|---|---|
| `graph_experiment_phase2.py` | 15 (docstring) | `Runs on the [lab machine's name] via MLX` | `Runs on a Mac via MLX` | it named a lab machine; the name is left out here too |
| `graph_benchmark.py` | 198 (comment) | `# Try [lab machine's name] path` | `# Try Mac path` | the same machine name |
| `graph_benchmark.py` | 199 | `"~/[folder]/lab_graph_walk_encoding.txt"` | `"~/lab_graph_walk_encoding.txt"` | it named a folder on that machine; the script still looks in the home directory |
| `graph_benchmark_powered.py` | 242 | `"~/[folder]/lab_graph_walk_encoding.txt"` | `"~/lab_graph_walk_encoding.txt"` | the same folder; the script still falls back to `/tmp` |

Nothing else needed changing. A scan of all nine copied files (the seven here and the two result files) looked for
private host names, Tailscale and other IP addresses, home paths (macOS, Linux, Windows and `~`-relative, and
`expanduser`), the home-directory folder above, e-mail addresses, private names, text from Nexus's memory graph,
internal headers, and keys or tokens. Before it ran, a line for each of its 15 patterns was planted in a copy of every
file, and each pattern fired on its planted line. Its other hits, all kept:

- two Ollama URLs, `http://localhost:11434` and `http://localhost:11436`: Ollama's default port and a local
  alternative on the machine running the script, not private addresses, and both can be overridden with `OLLAMA_URL`;
- in the two benchmark scripts, the multi-hop questions and the flat-text control (`FLAT_TEXT_KNOWLEDGE`), which are
  built from a small part of Nexus's memory graph and are copied unchanged;
- the lab's review tool, Agni, named in the docstring of `graph_benchmark_powered.py`;
- the graph file's name and its two locations, which name no machine or user.

The scan's host-name pattern missed the machine name in the comment at line 198 of `graph_benchmark.py`, where it
stands alone; a separate check found it there and in no other copied file. No e-mail header or patent header is
present in these files.

## Not copied

- The memory-graph file the larger run read: not available, as the paper says.

Available on request because dual-use: nothing from this set. These scripts build a prefix cache from text or a graph
and decode over it; nothing is added to the cache during generation, and no direction, layer or alpha is stored.
