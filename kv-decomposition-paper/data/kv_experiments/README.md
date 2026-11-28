# Follow-up results, copied from KV-Experiments (2026-11-16)

These are the result files of the follow-up runs of 29 May 2026 (the powered run and the LLM-judged run)
and of the re-judging of 14 October 2026 (Section "Follow-ups" of the paper). They come from the lab's
KV-Experiments repository (`Liberation-Labs-THCoalition/KV-Experiments`, private), folder
`graph-topology-injection/experiment_results/`. Each file was taken from the git blob at the commit named
below, not from a working tree, and is published byte for byte. They are published under Thomas Edrington's
decision of 2026-11-16: KV-Experiments files that are not dual-use are copied into published-research, and
dual-use ones are available on request. The scripts that wrote them are in `../../code/kv_experiments/`.

## Files

| File | Source path @ commit | Bytes | sha256 |
|---|---|---|---|
| `powered_decomp_1780057645.json` | `graph-topology-injection/experiment_results/powered_decomp_1780057645.json` @ 9755f5e | 14,235 | `fbad3c3fea2dccd2c7e7330e9b4e1239e3aea60612ed02e2fa227465b6edab1f` |
| `llm_judged_1780088341.json` | `graph-topology-injection/experiment_results/llm_judged_1780088341.json` @ 9755f5e (same blob at a439e2b) | 36,314 | `df839af2bf1ddbc8832a39f3d6f1f484228a664102946a6e5b85c96bf4fb04d6` |
| `rejudge_llm_judged_1791980451.json` (TEXT_CONTEXT) | `graph-topology-injection/experiment_results/rejudge_llm_judged_1791980451.json` @ 82ac524 (same blob at a439e2b) | 12,109 | `42db6bb408d04ae68b3a0b998d7d09b35dbd5f0ced9de9ea769df079bcde4625` |
| `rejudge_llm_judged_1791980882.json` (the other four conditions) | `graph-topology-injection/experiment_results/rejudge_llm_judged_1791980882.json` @ 82ac524 (same blob at a439e2b) | 32,954 | `5fef2b281026149b280d963d9d2d94174a138396e9d0ed6a9a6c855867ba3bf2` |

`llm_judged_1780088341.json` is also one of the inputs of the re-run of 20 October 2026: its sha256 is the
one listed for `experiment_results/llm_judged_1780088341.json` in `../rerun_20261020/inputs.sha256`.

The powered run stored no replies (the paper says so); its file holds scores, questions and summaries only.

## What was scrubbed

Nothing. A pattern scan of all four files (e-mail, home paths, home-directory APIs, private host names, IPv4,
localhost ports, key and token shapes, credential paths, phone numbers, patent header, private names,
memory-graph text) found no hits. Controls: a planted line, raw and JSON-escaped, fired every pattern in a
copy of each file; a near-miss line fired none. The files hold no numeric arrays longer than 20 (the
per-question scores) and no file paths.

## Available on request

Nothing from this set. None of it is dual-use: these are keyword scores and LLM-judge verdicts on questions
about Stanford Encyclopedia of Philosophy ethics packs, with the model's replies; no vector, direction or
prompt pair is stored.

Not copied here, though not dual-use: `graph-topology-injection/FINDINGS.md` (cited for the May gates) and
`graph-topology-injection/ethics_packs/` (the other inputs of the October re-run). The lab's inventory of
2026-11-16 assigns both to the ethics-pack-injection paper (PR #68).
