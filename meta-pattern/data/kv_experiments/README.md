# KV-Experiments data cited by meta-pattern

Copied 2026-11-16 from `Liberation-Labs-THCoalition/KV-Experiments` (private), read at `origin/main` = `e12ca6b`
(2026-11-14). Bytes come from git blobs at that commit (`git cat-file`), not from a working tree. Each file's
"last commit" is the last commit that touched it, as of `e12ca6b`; its blob at that commit is identical to the blob
at `e12ca6b` (checked for every file). Inventory: rows 1-6, 8 and 9 of the 2026-11-16 KV-Experiments data
inventory, with the second reader's corrections (none of them touch these rows).

## Files

| File (under this folder) | Source path @ last commit | Bytes | sha256 (this copy) | Cited for |
|---|---|---|---|---|
| `kv_verify/data/output/f01b_all/f01b_all_results.json` | `kv_verify/data/output/f01b_all/f01b_all_results.json` @464a73b | 2,124 | `0b07dee6ceb61c8e485a5dea48dbb983c0239130adf2a2f61cb383618266fb80` | Same-prompt deception, entry `exp18b_deception` (Table 1 sources; corrections section) |
| `kv_verify/data/output/f01b/f01b_results.json` | `kv_verify/data/output/f01b/f01b_results.json` @464a73b | 1,412 | `7278f0ff167d775aafc39a6259ece49ebbd968f572619c68d4c0f23a01561301` | Within-model deception (Table 1 sources) |
| `results/hackathon/same_prompt_sycophancy.json` | `results/hackathon/same_prompt_sycophancy.json` @ba24c73 | 60,849 | `35cffb434ee292926c3aea89543b64de6618afa521c550a96c3f11b51ac0a11e` | Same-prompt sycophancy (Table 1 sources) |
| `results/hackathon/adversarial_controls.json` | `results/hackathon/adversarial_controls.json` @ae4b5a7 | 87,651 | `a95d4cc32371d2fb0df4c30a5b6a374428401397fb9a54d839922ce4e7359515` | Step-0 controls (section 4.1) |
| `results/hackathon/axis_analysis.json` | `results/hackathon/axis_analysis.json` @5a2f0f5 | 2,826 | `a4f342cb8adb6d7780d57014e7a0540a449f5b5a8b3199dd19539388c01a1fd9` | Truth axis, cos = -0.046 (section 4) |
| `results/hackathon/c7_permutation_test.json` | `results/hackathon/c7_permutation_test.json` @b2eb023 | 43,988 | `c66be37eb426014bcc3d265cc6ca0b9a9da3a0603852b79aea12f563f1bc5986` | Campaign 3, Experiment C7: permutation test (not the C7 run output) |
| `paper-c3/sections/confabulation.tex` | `paper-c3/sections/confabulation.tex` @edfec70 | 8,788 | `a73e2489d3eb606ba9f6652de0e4636bd10caadbd153512c768025913dbb61b1` | Campaign 3 write-up of C7 and S3 (gives d = 0.052) |
| `paper-c3/sections/redteam.tex` | `paper-c3/sections/redteam.tex` @5412371 | 13,306 | `597188f14bb005d8fd62142e7b8c7f3e4d25ace107a4e106ae0e70abdf5e6da2` | Campaign 3 equal-length replication, 'Phase 3d-padded' paragraph (line 48) |
| `results/moe_geometry/EXPERIMENT_51_RESULTS.md` | `results/moe_geometry/EXPERIMENT_51_RESULTS.md` @c0ff205 | 20,883 | `808b3282b0e8ce6a857002e7e2d81f03ccc1bf4ced329abe2f5c622030e95816` | Experiment 51 (section 4.1) |
| `results/moe_geometry/moe_advanced_analysis_20260502_0148.json` | `results/moe_geometry/moe_advanced_analysis_20260502_0148.json` @92cc890 | 303 | `0b9998269045183211ef59b29c033baa8edb215f0a7b1e3deb099e88c81e7ed1` | Experiment 51 (section 4.1) |
| `results/moe_geometry/moe_phase2_qwen1.5-moe-a2.7b-chat_20260501_0536.json` | `results/moe_geometry/moe_phase2_qwen1.5-moe-a2.7b-chat_20260501_0536.json` @92cc890 | 1,920 | `e90b79484b2123b063160ad4f10ceddd2cbde2800f5f7b0a313f90aa9f947197` | Experiment 51 (section 4.1) |
| `results/moe_geometry/moe_phase2_qwen1.5-moe-a2.7b-chat_20260501_1555.json` | `results/moe_geometry/moe_phase2_qwen1.5-moe-a2.7b-chat_20260501_1555.json` @92cc890 | 2,362 | `2dc3cf6a1d5d7b2575c53f62c88fc3040560dbf0d1be03a84ccc257dcfa176ab` | Experiment 51 (section 4.1) |
| `results/moe_geometry/moe_phase2_qwen2.5-7b-instruct_20260503_1903.json` | `results/moe_geometry/moe_phase2_qwen2.5-7b-instruct_20260503_1903.json` @92cc890 | 945 | `9c286cf07df3c9c325ed9002e7fac44609355c3b3a04887dc656ab8579ef8e12` | Experiment 51 (section 4.1) |
| `results/moe_geometry/pregen_postgen.txt` | `results/moe_geometry/pregen_postgen.txt` @92cc890 | 256 | `3c33530f0707b5c537a293891e783d42195ae6135142d1d7597ff79beb4ea0b3` | Experiment 51 (section 4.1) |
| `results/moe_geometry/moe_routing_qwen1.5-moe-a2.7b-chat_20260501_0536.npz` | `results/moe_geometry/moe_routing_qwen1.5-moe-a2.7b-chat_20260501_0536.npz` @92cc890 | 2,561 | `4d5bf9f84e9ca458c00f9a3fe84bf05d85c0781f286fd1e5fad6eddfaec37dd6` | Experiment 51 (section 4.1) |
| `results/moe_geometry/moe_routing_qwen1.5-moe-a2.7b-chat_20260501_1555.npz` | `results/moe_geometry/moe_routing_qwen1.5-moe-a2.7b-chat_20260501_1555.npz` @92cc890 | 2,876 | `3f333959e4db36534c04d0ee23cc665672802c550eb7ab928d1ecc0cead5c46b` | Experiment 51 (section 4.1) |
| `results/moe_geometry/moe_routing_qwen2.5-7b-instruct_20260503_1903.npz` | `results/moe_geometry/moe_routing_qwen2.5-7b-instruct_20260503_1903.npz` @92cc890 | 2,424 | `f100b2ee3124f28df3f209aa34bdd8f2b18904ea71ff8f338bb75dda21c1f7e6` | Experiment 51 (section 4.1) |
| `results/phase3d_padded/phase3d_padded_qwen_summary.json` | `results/phase3d_padded/phase3d_padded_qwen_results.json` @e12ca6b (derived) | 7,422 | `bee537d7d3bd62a2ff4d50ea6883cf880cd24ef519029f73031eb63a3dd9ead9` | Equal-length replication, per-model summary (derived; see below) |
| `results/phase3d_padded/phase3d_padded_llama_summary.json` | `results/phase3d_padded/phase3d_padded_llama_results.json` @e12ca6b (derived) | 7,427 | `e00d724befcbdc8cc7d47f8c623089a563ccd340f8ca4f87105551d950fadfdd` | Equal-length replication, per-model summary (derived; see below) |
| `results/phase3d_padded/phase3d_padded_mistral_summary.json` | `results/phase3d_padded/phase3d_padded_mistral_results.json` @e12ca6b (derived) | 7,435 | `6c7cb8b16bd3c614d3ea0264a39462ba07fff90b91ec797f76436a77060edb50` | Equal-length replication, per-model summary (derived; see below) |

## Scrubbed (hygiene)

- `results/moe_geometry/EXPERIMENT_51_RESULTS.md` line 367: a private host name -> `an internal server`. Source sha256 `912b405ebe8ec6f672efa287cdba5e436c280c9d4991ff67aad0eb1f03ff2a72` (20,870 B).

Not changed, on purpose: the names Lyra, Thomas Edrington, Dwayne Wilkes, CC, Kavi and Vera (all named as authors or
contributors in published papers in this repository), and RunPod (a public cloud provider). Some files hold model
answers written under instructions to deceive or confabulate; those answers are false on purpose.

## Equal-length replication (Campaign 3, "Phase 3d-padded"): summaries, raw files held

The three raw results files are 29.8 MB together, over the 25 MB this copy allows. What is here instead, per model,
is `phase3d_padded_<model>_summary.json`: the source file with only its per-trial `results` key removed, written with
`json.dumps(indent=1, ensure_ascii=False)`, plus an `_extracted_from` key naming the source blob and sha256. The
`analysis` block is identical in all three sources (it covers all three models and cross-model transfer). The
encoding AUROC of 1.000 that `redteam.tex` reports (line 48) is computed from the 150 per-trial `encoding_features`
in each raw file, so it can't be recomputed from the summaries.

Held, owner Lyra, decision by 2026-11-20 (commit, or deposit elsewhere):

| Raw file @e12ca6b (last commit 4e5108f) | Bytes | sha256 |
|---|---|---|
| `results/phase3d_padded/phase3d_padded_qwen_results.json` | 6,642,802 | `f0258bbc1773bc5d1e0ce5736bc0f418c59ded45dc05486c56e2046255ad247c` |
| `results/phase3d_padded/phase3d_padded_llama_results.json` | 11,591,516 | `428110c5f1cb3a5121a46c4b492fa8222602c0758f4b8d986fce142ecb5bc6dc` |
| `results/phase3d_padded/phase3d_padded_mistral_results.json` | 11,569,347 | `71d87162115ccb4d06587b6929261f090d535471aed98d0eb6881f709a6515c0` |

## Experiment 51: the paper's d = 1.81 and d = 0.45

`EXPERIMENT_51_RESULTS.md` (line 18 and its effect-size table) gives d = 1.810 for key rank, confabulated vs honest.
The paper's figures come from the same per-prompt values in `moe_routing_qwen1.5-moe-a2.7b-chat_*.npz` (member
`key_ranks`, 15 prompts per arm; both runs give the same values), with the pooled SD taken as
sqrt((s1^2 + s2^2) / 2). Until 2026-11-16 the paper used population (divide-by-n) variances and gave d = 1.87 and
d = 0.46; it now uses the conventional n-1 variances, which give the `.md`'s value (see the paper's Corrections
appendix, 2026-11-16):

| Comparison (Qwen1.5-MoE-A2.7B-Chat, key rank) | n-1 pooled SD (the paper since 2026-11-16) | divide-by-n pooled SD (the paper until 2026-11-16) |
|---|---|---|
| confabulation vs honest_factual | 1.810 (the `.md`) | 1.874 |
| confabulation vs rare_but_true | 0.447 | 0.463 |

The `.md` gives the second comparison only as a difference in means (57.4 vs 57.1). The `labels` member of each
`.npz` is a pickled object array, so `np.load` needs `allow_pickle=True`; the check above used an unpickler limited
to numpy's array-reconstruction classes.

## Available on request, and what is not here

- **Dual-use: nothing.** Both readers found no stored direction, calibration pair set or cache-injection data in
  these files (the longest numeric list in any JSON has 28 entries; each `.npz` member holds one value per prompt),
  so nothing here is held on dual-use grounds.
- **Held for size, not for dual-use:** the three Phase 3d-padded raw results files listed above (29.8 MB). Whether
  they are committed or deposited elsewhere is decided by 2026-11-20.
- **Campaign 3 C7 and S3 run outputs: not found** in KV-Experiments at any revision checked (the
  `confabulation_detection_*_results.json` files hold only the sycophancy sub-experiment). For C7 the paper points at
  the write-up (`paper-c3/sections/confabulation.tex`), the permutation test
  (`results/hackathon/c7_permutation_test.json`) and the code (`../../code/kv_experiments/`). It keeps
  "unpublished analysis" for S3.
