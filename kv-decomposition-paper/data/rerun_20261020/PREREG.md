# Pharos empty-reply dig: what I expect, written before the run (2026-10-20 14:45 PDT)

**Question (Thomas, 14:29):** is something about Pharos unreliable, or was "the cache came back empty in 15 of 20
replies, against 6 for the text" an early-pilot artifact?

**The run being examined:** `llm_judged_1780088341.json` (K/V v1.1, Larger Follow-Ups), from
`~/KV-Experiments/graph-topology-injection/llm_rescorer.py::run_with_responses`. Qwen/Qwen2.5-1.5B (base, not
Instruct), float32, greedy, 50 new tokens, no chat template; packs aristotle-ethics, autonomy-moral, ethics-ai.

**What the code shows (read 14:40):**
1. The two conditions get different inputs. TEXT_CONTEXT: `f"{merged}\n\nAnswer concisely: {q}"`. FULL_KV: the
   cache of `merged`, then `"Answer concisely: {q}"` tokenized separately: no `\n\n`, and a different token
   boundary.
2. `merged` stops at about 2,000 tokens, but the questions come from all three packs' triples.
3. All 20 questions are one-hop "direct" (the bridge loop never runs: the quota is full).
4. The cache is deep-copied per question (not the DynamicCache trap).
5. Stored: FULL_KV's 5 non-empty replies mostly echo the question; TEXT's 6 empties are all among FULL_KV's 15.

**Arms (same 20 stored questions, same model, CPU, torch threads pinned):**
- **A, original FULL_KV** (the script's own loop and inputs). Positive control: should reproduce the stored
  FULL_KV replies.
- **B, original TEXT_CONTEXT** (`model.generate` on the full prompt). Positive control: should reproduce the stored
  TEXT replies.
- **C, cache with exact tokens:** tokenize the full TEXT prompt, cache its leading tokens up to the longest prefix
  shared by every question (so the graph part is identical), and continue from there with the script's loop.
- **D, the same through `model.generate(past_key_values=...)`**, HF's own path, so a difference between C and B
  can't be blamed on the hand-written loop alone.

**Predictions:**
- A and B reproduce the stored replies on at least 18 of 20 each. If not, the rebuild isn't faithful (packs or
  libraries changed) and C/D can't speak for May; I report that and stop.
- C and D each match B's reply exactly on at least 18 of 20, and their empty count is within 1 of B's.
- If so: a cache of the same tokens behaves like the text, and the May gap (15 vs 6) came from the harness feeding
  the cache condition a different input. If C/D differ from B on more than 2 questions: a real cache-path problem,
  to dig into further.

**Also measured:** how many of the 20 expected answers appear in `merged` at all (are the questions answerable from
the context?).

**Not this run:** the shipped `mlx_injector.py` (Apple MLX) path. That is a separate test, on the Studio.
