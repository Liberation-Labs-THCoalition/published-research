# Pharos empty-reply dig: results, after the second read (2026-10-20, Nexus)

**Question (Thomas, 14:29):** is something about Pharos unreliable, or was "the cache came back empty in 15 of 20
replies, against 6 for the text" (K/V v1.1, Larger Follow-Ups) an early-pilot artifact?

**Answer:** an artifact of the May harness. A cache of the same tokens gives exactly the text's replies.

## What was run

`dig.py` (predictions in `PREREG.md`, written 14:44 before any run) on the Mac Studio, 21:41–21:48 PDT: torch 2.8.0,
transformers 4.57.6, CPU fp32, Qwen/Qwen2.5-1.5B (base) snapshot 8faed761, the 20 stored questions of
`llm_judged_1780088341.json`, greedy, 50 new tokens. (An MTH attempt at 21:38 was stopped by a memory-pressure guard
before any arm ran.)

## Results (second reader recomputed every comparison from the raw strings)

1. **The rebuild is faithful.** Arm A (the original FULL_KV path) reproduces the stored FULL_KV replies on 20/20 and
   arm B (the original TEXT_CONTEXT path) the stored TEXT replies on 20/20. B's match is the strong one: 14
   non-empty, context-dependent 50-token replies agree exactly. A's 15 empties and 5 question-echoes are weaker
   evidence on their own.
2. **A cache of the text prompt's own leading tokens behaves exactly like the text.** The cache covered the text
   prompt's first 5,348 tokens: the description, ending in the separator token. (The longest prefix shared by every
   prompt was 5,361; the script caps it at the description's length.) It matched B's replies on 20/20, through the
   script's hand-written loop (C) and through `model.generate(past_key_values=...)` (D). Empty: 6 for B, C and D;
   15 for A. Six of the matches are both empty; 14 are identical long strings.
3. **The comparison can report a difference.** A and B agree on 6 of 20, all six both empty. On every one of the 14
   questions where B replied, A's reply differs.
4. **The mechanism is one token.** A's input and the text prompt are the same length and differ only at position
   5,347: A has `')'` (token 8), the text prompt `')\n\n'` (token 692), because Qwen's tokenizer merges the separator
   into the preceding character. The query tokens after it are identical on all 20. With caching held fixed, that
   one token moves empties from 15 to 6. With the input held fixed, caching changes nothing.
   **Not run:** A's exact tokens through the text path with no cache (the grid's last cell); the inference from C/D
   matching B is sound but that cell is untested.
5. **Most questions couldn't be answered from the context in any condition.** The description was the first pack
   alone (aristotle-ethics, 5,348 tokens, already past the ~2,000-token cap, so the merge loop stopped). All 20
   questions are one-hop "direct". 16 of the 20 ask about edges from the two packs that never reached the context
   (14 ethics-ai, 2 autonomy-moral), and the context carries no predicates, so "via 'pred'" can't be resolved from it.

## Consequences found by the second reader (not yet acted on)

- **K/V v1.1 (PR #8) describes the runs wrongly.** The powered run's "(5 packs, 532 nodes, 396 edges, a 5,348-token
  description)": its file stores `encoding_tokens: 5348` and `powered_decomp_study.py` (220–228) has the same merge
  loop, so its description was aristotle-ethics alone (1 of 5 packs). The judged run's "graph built from three of
  those packs" likewise had 1 pack in its description. "may not have been answerable" is too soft: 16 of 20 were
  unanswerable from the context by construction.
- **The powered run has the same one-token defect** (`generate_with_cache` tokenizes the query separately; the cache
  conditions' prompt omits the context; lines 161–187, 280–296), sanity items included.
- **The judge credited answers the context didn't hold:** TEXT's 3 CORRECT verdicts are items 3, 4, 5; items 3 and 5
  are ethics-ai edges absent from the text.
- **K_ONLY and V_ONLY** got the same mismatched input in May. This dig says nothing about them.
- **The shipped MLX path may repeat the defect (code reading, effect untested).** `mlx_injector.build_pack_cache`
  caches a chat-templated system turn (`add_generation_prompt=False`); `generate_with_pack` passes the raw query to
  `mlx_lm.generate`, which in mlx_lm 0.31.3 encodes a string prompt with no chat template (generate.py ~688–694). mlx_lm's
  own CLI templates the user turn and passes only the part after the cached prefix (~2033–2052). For tokenizers with a
  BOS token, a BOS also lands mid-sequence (not Qwen).
- `stats.json`'s "encoding_tokens_approx" is a whitespace word count (1,313 for aristotle-ethics), about 4x below
  the real token count.

## Scope

One model, CPU fp32, Hugging Face transformers. Not MLX, not GPU or fp16, not the Ethics Packs paper.
