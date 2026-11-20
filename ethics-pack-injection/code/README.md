# code/

`morebench_eval.py` is the evaluation script as used for the v4 run (30 May 2026). It is kept byte-for-byte as run;
v3 differed only in its 300-second request timeout.

Its docstring says it "Tests whether ethics KV packs make models reason better". The packs are not injected as a KV
cache. The script puts each framework's walk-encoding text into the system prompt (see the paper's Methods).

The packs themselves are in `../data/kv_experiments/ethics_packs/`: byte-for-byte copies of the lab's KV-Experiments
repository, `graph-topology-injection/ethics_packs/`, at commit 9755f5e (29 May 2026), with each file's SHA-256 in
`../data/kv_experiments/README.md`. Set `PACKS_DIR` to that folder to run the script on them. The script reads MoReBench's theory file from `morebench/morebench_theory.jsonl` relative to
where it runs; that copy is not redistributed here.
