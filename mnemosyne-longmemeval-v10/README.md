# Mnemosyne v10 on LongMemEval_S-cleaned: paper and reproduction package

The paper (Nexus, Thomas Edrington) is `paper.md`, built to `main.tex` and `main.pdf`. The academic edition (human-only
byline, AI Disclosure and CRediT contributions, no first-person reflection) is in `academic/`. `build.py` regenerates
both editions' `.tex` and PDF from the Markdown (pandoc, then pdflatex in the repository's LaTeX house style): edit
the Markdown, not the `.tex`.
`REVIEW_INDEX.md` records the review trail and the release decisions. Everything behind the paper's tables is here, apart from
the items under "Not included" below.

## Contents

| path | what |
|---|---|
| `code/` | The scripts **as run**, byte-identical to the files the second readers verified (`SHA256SUMS`). The absolute paths inside them (`/mnt/data1/lme_v2`, `/mnt/data1/datasets/...`) are our machine's. To reproduce, point `ROOT`/`DATA` at yours. |
| `config/` | `frozen_v10.json` (hash `7bc75f509aa0`), the pre-registered `split.json` (dev = 100, held = 400), and `emb_meta.json` (encoder, revision, pooling). |
| `docs/` | `PROTOCOL.md` (harness, judges, deviations, leakage audit), `DESIGN.md` (the pre-registration note), `RESULTS_v10.md`, and the comparison and calibration tables. |
| `reviews/` | The adversarial second reads: the v9 baseline, v10 retrieval recall, v10 held-out readers, and the text of this paper. |
| `data/answers_<arm>.jsonl` | One line per question: `question_id`, `idx`, `model`, `answer`, and token counts (`input_tokens`, `output_tokens`, `thinking_tokens`). |
| `data/judgments_<arm>_<judge>.jsonl` | `question_id`, `question_type`, `abstention`, `judge_model`, `judge_raw`, `label`. The label is `'yes' in judge_raw.lower()`, the official rule. |
| `data/judge_controls.jsonl` | Pilot answers the judges passed, and deliberately wrong answers they failed. |
| `data/heldout_log.jsonl` | The held-out retrieval log: the completed run; two earlier attempts that crashed before producing any number (tiktoken rejected a literal `<|endoftext|>` in a turn), recorded after the fact at the second reader's request; and the second reader's in-memory reproduction (365/376). |
| `data/recall_*_frozen.json` | Per-question retrieval record for the frozen config on held-out and dev: evidence sessions covered, tokens, sessions, whole sessions. |

Arms: `v9` (Mnemosyne v9 context, Opus 4.6); `v10` (Opus 4.6); `v10_o55t` (Opus 5.5 with thinking, the headline);
`oracle` and `oracle_o55t` (evidence sessions only); `full` (the whole history, dev block only). Judges: `sonnet5`
(claude-sonnet-5, pre-registered primary) and `opus55` (claude-opus-5-5).

## Reproducing

1. Download LongMemEval_S-cleaned (`xiaowu0162/longmemeval-cleaned` on Hugging Face). Check its sha256 against
   `split.json` (`d6f21ea9…`).
2. `embed_turns.py`: embeds the user turns with bge-base-en-v1.5 at the revision in `emb_meta.json`.
3. `v10.py --config config/frozen_v10.json --split held`: runs the retrieval and writes the recall record.
4. `build_v10_prompts.py`, then `answer_one_arm.sh`: build the official prompts and run one reader call per
   question.
5. `judge.py`, then `compare_arms.py`: judge with the official answer-check prompts and write the paired tables.

## Not included, and why

- **Reader transcripts.** The `claude -p` harness injects the account e-mail and today's real date into every call
  (disclosed in `PROTOCOL.md` and §3.7 of the paper).
- **Prompts and turn embeddings.** Both are regenerable from the code and the public dataset. The prompts embed
  the dataset's chat text.
- **The dataset and the official LongMemEval scripts.** Fetch them from their sources. The scripts we ran are
  byte-identical to xiaowu0162/LongMemEval at commit `9e0b455`; their hashes are in `PROTOCOL.md`.

**One redaction.** One answer in `answers_oracle.jsonl` (Opus 4.6, evidence-only arm) quoted the account e-mail
from the harness's system-reminder. It is replaced with `[account e-mail redacted]`. That answer's judgment was
made on the original text. No answer in any arm mentions the real (2026) date. Where answers refer to "today's
date", they use the prompt's own 2023 date.

## License

The repository `LICENSE.md` applies (Hippocratic License 3.0, with the commercial addendum).
