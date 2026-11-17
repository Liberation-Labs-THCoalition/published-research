# Code copied for this paper (2026-11-16)

**Source repository: `Liberation-Labs-THCoalition/Project-Oracle`** (private), not KV-Experiments. The folder
keeps the `kv_experiments` name that the 2026-11-16 copy uses across papers; the file below comes from
Project-Oracle.

| Published file | Source path @ commit | Source bytes / sha256 | Published bytes / sha256 |
|---|---|---|---|
| `oracle_generalization_test.py` | `experiments/oracle_generalization_test.py` @ `e68ee60709666cf5ad4d9043b60e22d7143e68b0` (2026-07-02, "Oracle generalization: direction works outside lab scenarios") | 36,877 / `3a8ff974d3ec7d2555f6b5f7ac672c09f10b1a0e3b3800a49c733191109da0d5` | 37,067 / `e9563f088b38d3c66bb1d0aa4bf3db7273e10ff08d5e573d42953e231309da9e` |

The bytes were read from the commit (`git cat-file blob`), not from a working tree. The blob
(`3a244f9d4d52382d35bc402e99a26f6ba01aebfc`) is unchanged at Project-Oracle `origin/main` 5480499. Stored with LF
line endings (0 CR).

## Why it is published

The paper's limitations list cites this test. The 2026-11-16 inventory marked the script "uncertain" and the
second reader changed that to **not dual-use**: its forward hook only records the last-token activation
(`use_cache=False`, nothing is added to the residual stream), it projects onto an external direction that the
file does not contain, and its prompts (code review, report summarization, personal advice) carry none of the
lab's calibration-pair text.

## What was scrubbed

Three lines, all absolute paths in a private home directory on a lab machine. Nothing else changed (the diff
against the source is these three lines). `<home>` below stands for the private home directory, which is not
reproduced here.

| Line | Before | After |
|---|---|---|
| 80 | `HF_CACHE = "<home>/models/hf_cache"` | `HF_CACHE = None` (the Hugging Face default cache), with a comment marking the scrub |
| 81 | `RESULTS_DIR = Path("<home>/oracle-experiments/results/oracle_generalization")` | `RESULTS_DIR = Path("results/oracle_generalization")`, with a comment |
| 82 | `LAT_V2_PATH = Path("<home>/oracle-experiments/results/lat_deception_v2/deception_directions_v2.json")` | `LAT_V2_PATH = Path("results/lat_deception_v2/deception_directions_v2.json")`, with a comment |

The hygiene scan covered home paths, private host names, IPv4 addresses, e-mail addresses, key and token
patterns, and the names of lab members who are not public authors. Before and after the scrub it was run with a
planted positive control (one hit for every pattern) and a clean negative control (no hits). The source had 3
home-path hits on lines 80 to 82 and no other hits. The scrubbed file has none.

## To run it

It needs the model named in the file (`Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled`) and the LAT v2
direction file at `LAT_V2_PATH`. That direction file is not part of this copy.

## Still available on request

- **Analysis and figure code** (`signature_cis.py`, `generate_figures.py`, `subspace_reanalysis.py`): not reviewed for
  this copy, so the paper's "on request" wording stands.
- **`lat_transfer_test.py`**: dual-use, because it carries the calibration scenario pairs for a direction the lab
  holds back.
- **The Oracle Loop's detection, correction and calibration modules**: dual-use (the program's dual-use
  assessment withholds the implementation).
