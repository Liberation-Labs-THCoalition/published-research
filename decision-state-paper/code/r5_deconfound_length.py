"""Length-only baselines beside the deconfound table, and what the deconfound script reads (review round 5,
findings 7 and 8).

CPU only, no model run. Collects from data/entity_deconfound/text_baseline_r4.json the prompt-length-only
AUROC of every comparison and the mean prompt length per condition, cross-checks those means against
phase1_ground_truth.json, records the recorded label of each comparison in phase3_results.json, and lists
the lines of code/entity_deconfound.py that read the preregistration file, set the spectral window from the
question length, and add the log prompt-length covariate. It also records whether the preregistration file
the script opens exists anywhere in this paper's directory.

Every input hash is the sha256 of the LF form (CRLF -> LF), which is the git blob.
Output: data/entity_deconfound/r5_deconfound_length.json
"""
import hashlib
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
E = "decision-state-paper/data/entity_deconfound/"
TB, P1, P3 = E + "text_baseline_r4.json", E + "phase1_ground_truth.json", E + "phase3_results.json"
CODE = "decision-state-paper/code/entity_deconfound.py"
PREREG_NAME = "entity_deconfound_preregistration.json"
OUT = PAPER / "data" / "entity_deconfound" / "r5_deconfound_length.json"


def lf_bytes(rel):
    return (REPO / rel).read_bytes().replace(b"\r\n", b"\n")


def load(rel):
    return json.loads(lf_bytes(rel).decode("utf-8"))


def main():
    tb, p1, p3 = load(TB), load(P1), load(P3)
    comps = {k: {"recorded_label": v.get("shipped_label"),
                 "cache_features_auroc_with_3_raw_key_coords": v.get("shipped_features_auroc_with_wk"),
                 "question_text_charngram": v.get("charngram_questions"),
                 "prompt_len_only": v.get("prompt_len_only")}
             for k, v in tb["comparisons"].items()}
    pl = defaultdict(list)
    for r in p1:
        pl[r["condition"]].append(r["prompt_len"])
    p1_means = {k: round(st.mean(v), 2) for k, v in sorted(pl.items())}
    qf = {k: v.get("mean_prompt_len") for k, v in tb["question_form"].items()}
    src = lf_bytes(CODE).decode("utf-8").split("\n")
    pats = {"preregistration": r"PREREG", "question_length_window": r"q_len",
            "log_prompt_length_covariate": r"log1p"}
    code_lines = {name: [{"line": i + 1, "code": l.strip()} for i, l in enumerate(src) if re.search(p, l)]
                  for name, p in pats.items()}
    found = sorted(p.relative_to(REPO).as_posix() for p in PAPER.rglob(PREREG_NAME))
    result = {
        "LABEL": "Review round 5, findings 7 and 8. Collected from stored outputs and the shipped script; nothing re-fitted.",
        "inputs": {r: hashlib.sha256(lf_bytes(r)).hexdigest() for r in (TB, P1, P3, CODE)},
        "hash_note": "sha256 of the LF form (CRLF replaced by LF), i.e. of the git blob",
        "comparisons": comps,
        "phase3_recorded_labels": {k: v.get("label") for k, v in p3["comparisons"].items()},
        "mean_prompt_len_text_baseline": qf,
        "mean_prompt_len_phase1_recomputed": p1_means,
        "phase1_has_correctness_field": tb.get("phase1_has_correctness_field"),
        "entity_deconfound_py_lines": code_lines,
        "preregistration_file_name": PREREG_NAME,
        "preregistration_file_found_in_paper_dir": found,
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(REPO).as_posix())
    for k, v in comps.items():
        print(f"  {k}: len-only {v['prompt_len_only']}")
    print("  prompt len", qf, p1_means)
    print("  phase3 labels", result["phase3_recorded_labels"])
    for k, v in code_lines.items():
        print("  ", k, [x["line"] for x in v])
    print("  prereg found:", found)


if __name__ == "__main__":
    main()
