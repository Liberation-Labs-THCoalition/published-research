# Follow-up scripts, copied from KV-Experiments (2026-11-16)

These are the scripts behind the follow-up runs of 29 May 2026 and the re-judging of 14 October 2026
(Section "Follow-ups" of the paper). They come from the lab's KV-Experiments repository
(`Liberation-Labs-THCoalition/KV-Experiments`, private), folder `graph-topology-injection/`. Each file was
taken from the git blob at the commit named below, not from a working tree. They are published here under
Thomas Edrington's decision of 2026-11-16: KV-Experiments files that are not dual-use are copied into
published-research, and dual-use ones are available on request.

## Files

| File | Source path @ commit | Source bytes | Source sha256 | Published bytes | Published sha256 |
|---|---|---|---|---|---|
| `powered_decomp_study.py` | `graph-topology-injection/powered_decomp_study.py` @ 9755f5e | 13,616 | `e16a1a9cb31d1f4b384ebb5fbfeac34ad65808810c900870faef44f19e831527` | 13,580 | `6a43507fd2c3d6f82acece07295e7ce05da808e446ee2234b25118530099f762` |
| `llm_rescorer.py` | `graph-topology-injection/llm_rescorer.py` @ 9755f5e | 14,660 | `75e95e9c03b8793e67c964f1516f1125d95ba3c3b1ab65b9ac33660d38592cc5` | 14,624 | `3d78ff54e3d90cfb4230ae9344f4aea8cd258d4396ed7c9622b7fce215ad7345` |
| `rejudge_llm_judged.py` | `graph-topology-injection/rejudge_llm_judged.py` @ 82ac524 (same blob at a439e2b) | 5,333 | `c88e019a30265b4c6b6c79dcad6f6ecaf465d255f817d8286eed355cdb594d71` | 5,333 | unchanged |

The third follow-up script, `graph-topology-injection/value_only_injection.py` @ 9755f5e (14,556 bytes,
sha256 `473e3d8b86a46ab6d1f43f0c0ead26821de4be5ab1894b40f6585f48e4a66de0`), is not copied again: it is the
same git blob (598915b) as `../value_only_injection.py`, the pilot's script, already in this paper's folder.

## What was scrubbed

One substitution, the same line in two files:

| File, line | Before | After |
|---|---|---|
| `llm_rescorer.py` line 24; `powered_decomp_study.py` line 32 | `PACKS_DIR = Path.home() / "<two folders>/ethics_packs"` | `PACKS_DIR = Path(__file__).parent / "ethics_packs"` |

Reason: the original pointed into the author's home directory; the two folder names under it are left out
here too. The new value is the layout
`rejudge_llm_judged.py` already uses. Nothing else in any file was changed; the source sha256 above lets
anyone with access to KV-Experiments check that.

**Kept on purpose:** `http://localhost:11434` (`llm_rescorer.py` line 21, `rejudge_llm_judged.py` line 28).
It is Ollama's standard local default port on whatever machine runs the script, not a private host or
address. `llm_rescorer.py` also takes `OLLAMA_URL` from the environment.

**Not present:** private host names, other home paths, e-mail addresses, phone numbers, keys or tokens,
names of people other than public authors, and the patent-and-contact header that the KV-Experiments
`code/*.py` files carry (so no address needed replacing).

## Running them

- Models: `Qwen/Qwen2.5-1.5B` from Hugging Face (fp32); the judge is `deepseek-v2:16b` served by Ollama.
- The scripts expect, beside them, `ethics_packs/<pack>/triples.json` and `walk_encoding.txt`, and they
  write to `experiment_results/`. `rejudge_llm_judged.py` reads
  `experiment_results/llm_judged_1780088341.json` (here `../../data/kv_experiments/`) and imports
  `JUDGE_PROMPT` from `llm_rescorer.py` in the same folder.
- The ethics packs are not in this folder. They are KV-Experiments `graph-topology-injection/ethics_packs/`
  @ 9755f5e, which are not dual-use; the lab's inventory of 2026-11-16 assigns them to the
  ethics-pack-injection paper (PR #68).

## Available on request

Nothing from this set. None of it is dual-use: the scripts prefill a cache from benign text (K and V taken
from different encodings) and decode greedily; nothing is added to the cache during generation, and no
direction, chosen layer or strength is stored or applied.

## How the files were checked

- Dual-use: the lab's inventory of 2026-11-16 (rows 11 and 12) called these NOT dual-use, with a borderline
  note against the formulary's withheld "cache-injection implementation". A second reader closed the
  borderline: that entry covers modifying the cache during generation, which these scripts do not do.
- Hygiene: a pattern scan (e-mail, Unix/Windows/tilde home paths, home-directory APIs, private host names,
  IPv4, localhost ports, key and token shapes, credential paths, phone numbers, patent header, private names,
  memory-graph text) over the source blobs and again over the published copies. Controls: a planted line
  fired every pattern in a copy of each file (and, for JSON, a JSON-escaped plant too); a near-miss line
  fired none. The home-directory API pattern was added after the first run missed the `Path.home()` line
  above; it fires on that line in the source blobs and not in the published copies.
