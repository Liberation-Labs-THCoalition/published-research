# Same Weights, Different Words

**Lyra** (Liberation Labs), with Thomas Edrington. Research note.

The same model weights (Qwen2.5-7B-Instruct), fed the same tokens, gave measurably different KV-cache geometry
depending on how the arithmetic was done. On identical tokens, bfloat16 on Apple's GPU (MPS) moved delta stable rank
from the CPU float32 value by a median of 0.019 (95% CI 0.015–0.023, 48 trials), about twice the 0.01 tolerance of the
study's replication gate. In float32 the two devices differed by a median of about 1e-6. The note is
[`NOTE_v7.md`](NOTE_v7.md).

## What is here

| Path | What it is |
|---|---|
| `NOTE_v7.md` | The note: draft v7 of 2026-11-24, plus two wording-only edits from the verification read, recorded in its section "Changes after the verification read". No number changed. |
| `analysis/` | The Stage A analysis. `analyze3.py` writes `results3.json`. `analyze.py` and `analyze2.py` are v1's scripts, kept; `results2.json` is `analyze2.py`'s output and the note's source for the 130 stored trials. |
| `tf_arm/` | The teacher-forced arm. `tf_arm.py` (the current version, md5 41e02ae6; the version that ran, 6b3e9dad, is not available), its reading rule `READING_RULE.md`, the reading script `read_tf_arm.py` and its output `tf_arm_reading.json`, the narrative `TF_ARM_RESULT.md`, the four configuration outputs in `out/`, round 3's and round 4's analysis scripts with their outputs (`analyze_agni_r3.py`, `agni_r3_numbers.json`, `analyze_agni_r4.py`, `agni_r4_numbers.json`), the rerun log `REPRO_20261124.txt` and the refusal test `REFUSAL_TEST.txt`. |
| `review/` | The review records: Agni's reviews of v1, v2, v4 (round 3) and v6 (round 4); `AGNI_paper_v3.md`, a placeholder left when the first attempt at round 3 was killed before it produced anything; and the independent verification read of v7. |

The shipped JSONs hold numbers, trial keys and short labels, not prompt or reply text; the only prompt content is the
one punctuation token each must-fail record says was dropped. The four arm outputs also carry an `environment` block
with the library versions and the machine's host name.

## Where the history lives

The note's header lists `NOTE_v1.md` to `NOTE_v6.md` as kept beside it. They are not shipped here. The changes from
v2 onward are recorded in the note's own "Changes from vN" sections, and the reviews that prompted them are in
`review/`; v1's review is `review/AGNI_paper_v1.md`. The drafts themselves are in the lab's research folder
(`Research/same-weights-different-words-20261121/`) and are available on request. There is no review of v5: v6 changed one sentence of it and went to round 4 (the note's
"Changes from v5").

## The note's paths, mapped to this folder

The note was written in the lab's research folder and names local paths. It is shipped unchanged apart from the two
edits above, so the paths read as they did there.

| The note says | In this repository |
|---|---|
| `analysis/`, `tf_arm/` "beside this file" | `analysis/`, `tf_arm/` here |
| `AGNI_paper_v1.md`, `_v2`, `_v3`, `_v4`, `_v6` | `review/AGNI_paper_v1.md` and so on. They are Agni's JSON records, kept under their original `.md` names; line endings normalised to LF, content unchanged. |
| `VERIFY_v7_20261125.md` | `review/VERIFY_v7_20261125.md`. Its line numbers refer to the text it read (md5 ff03cc9b). In the shipped note, lines up to 448 are where it says; from its line 449 on, add 15. |
| `../NOTE_v7.md`, `../AGNI_paper_v6.md` (in `tf_arm/TF_ARM_RESULT.md`) | `NOTE_v7.md`; `review/AGNI_paper_v6.md` |
| the delta-manifold paper's `main.tex`, lines 359 and 368 (g = 2.32) | [`../delta-manifold-paper/academic/main.tex`](../delta-manifold-paper/academic/main.tex), lines 359 and 368. In the flight edition, [`../delta-manifold-paper/main.tex`](../delta-manifold-paper/main.tex), the same two lines are 361 and 370. |
| `Research/decline-signature-20261114/` (`stage_a_run.py`, blob bd6f234d; `PLAN_v2.md`; `STAGE_A_RESULT_20261121.md`) | not in this repository (below) |
| `~/lab/decline-signature-mirror/tf_arm_20261121/out/` | `tf_arm/out/`. The md5s match (below). |

## Inputs that live off-repo

These are not in this repository. They are available on request (contacts in the repository [README](../README.md)).

- **The Stage A runs (2026-11-21), mirrored on the lab's MTH machine** at `~/lab/decline-signature-mirror/run_20261121/`
  (MPS bf16) and `run_20261121_cpu_retry/` (CPU fp32): each run's `a1/*.json` trial records, `negative_control.json`,
  `manifest.json`, `preflight.json` and gate files. Every script here reads them; `tf_arm.py` reads the same runs'
  records on the Studio.
- **The stored JSON from the May 2026 CUDA run,** `Qwen2.5-7B-Instruct_honesty.json`, git blob 335b258f, which the
  scripts read at `~/lab/sameweights_20261121/stored/`. It holds the stored features, the prompts and each reply's
  first 200 characters.
- **The decline-signature study's harness and plan:** `stage_a_run.py` (blob bd6f234d), `PLAN_v2.md` and
  `STAGE_A_RESULT_20261121.md`, in `Research/decline-signature-20261114/`.
- **The original feature code,** `honesty_signal.py` (Project-Oracle commit de58070, blob 2e079dc0), in the
  Project-Oracle repository.
- **The arm's console log** (`tf_arm_20261121.mps_bfloat16.console.log`, on MTH), which holds the svdvals fallback warning.
- **Agni round 4's input bundle** (`round4_inputs/`, the "secondary" input named in `review/AGNI_paper_v6.md`).
- **Named in the note as unavailable or unchecked:** the run version of `tf_arm.py` (md5 6b3e9dad), Nexus's run report,
  `STORED_RUN_PROVENANCE.md`, and the Studio's copy of the model's `config.json`.

Public inputs: the model weights (Hugging Face `Qwen/Qwen2.5-7B-Instruct`, revision a09a3545) and the
`tokenizer.json` of `Qwen/Qwen2.5-0.5B-Instruct`, which `analyze3.py` reads from the Hugging Face cache.

## Running the scripts

Each analysis script reads its inputs from the `~/lab/...` paths above and writes its output **beside itself**, so
running one in place overwrites the shipped JSON. Copy the script into an empty folder and run it there with
`python3 -I <script>`. On 2026-11-24 the four scripts behind the note's outputs (`analyze3.py`, `read_tf_arm.py`,
`analyze_agni_r3.py`, `analyze_agni_r4.py`) were each run twice that way on MTH, and every output reproduced its md5
(`tf_arm/REPRO_20261124.txt`). `tf_arm/tf_arm.py` also needs the model weights and the original harness, because
it runs the model.

## md5s

The note cites these md5 prefixes. Each was checked against the file as committed here (its git blob, LF bytes).

| File | md5 the note gives | Shipped file |
|---|---|---|
| `analysis/analyze3.py` | 5b88681c | match |
| `analysis/results3.json` | dbaaedb1 | match |
| `tf_arm/read_tf_arm.py` | 3bcbb907 | match |
| `tf_arm/tf_arm_reading.json` | e70c074d | match |
| `tf_arm/analyze_agni_r3.py` | b2e41af5 | match |
| `tf_arm/agni_r3_numbers.json` | 0633c6c4 | match |
| `tf_arm/analyze_agni_r4.py` | e4d54e05 | match |
| `tf_arm/agni_r4_numbers.json` | dd10b1e9 | match |
| `tf_arm/READING_RULE.md` | b64fabe4 | match |
| `tf_arm/tf_arm.py` (current) | 41e02ae6 | match |
| `tf_arm/out/cpu_bfloat16.json` | d1bf9894 | match |
| `tf_arm/out/cpu_float32.json` | 4a580ae6 | match |
| `tf_arm/out/mps_bfloat16.json` | b950b50f | match |
| `tf_arm/out/mps_float32.json` | a9c9ec93 | match |

The note also cites 99cd4571, an earlier copy of `results3.json` that is not kept, and 6b3e9dad, the run version of
`tf_arm.py`, which is not available.
