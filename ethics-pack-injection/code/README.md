# code/

`morebench_eval.py` is the evaluation script as used for the v4 run (30 May 2026). It is kept byte-for-byte as run;
v3 differed only in its 300-second request timeout.

Its docstring says it "Tests whether ethics KV packs make models reason better". The packs are not injected as a KV
cache. The script puts each framework's walk-encoding text into the system prompt (see the paper's Methods).

The packs themselves are in the lab's KV-Experiments repository, `graph-topology-injection/ethics_packs/`, at commit
9755f5e (29 May 2026). The script reads MoReBench's theory file from `morebench/morebench_theory.jsonl` relative to
where it runs; that copy is not redistributed here.
