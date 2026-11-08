"""Which estimator each suffix-control figure is, the ablation's projection AUROCs, and what the
"treated as real" reading records (review round 5, findings 1, 6 and 13).

CPU only; reads stored outputs and re-fits nothing. Collects, with their keys:
  * the headline bootstrap mean (A1_headline_t4_full) against the out-of-fold estimate of the same model
    on the observed trials (main["full_52feat(headline)"].oof) and the bootstrap means that exist;
  * the word-frequency control's figures, which are single out-of-fold runs (no bootstrap key);
  * the single-feature AUROCs of the six key-direction projections in the shipped test-1 output;
  * the label of the unblinded "treated as real" reading, whether it records answer correctness, and the
    nine-pair drop-set figures.

Every input hash is the sha256 of the LF form (CRLF -> LF), which is the git blob.
Output: data/suffix_controls/r5_suffix_estimators.json
"""
import hashlib
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
S = "decision-state-paper/data/suffix_controls/"
RES, WF, FU = S + "results.json", S + "r4_wordfreq.json", S + "r4_followups.json"
OUT = PAPER / "data" / "suffix_controls" / "r5_suffix_estimators.json"


def lf_bytes(rel):
    return (REPO / rel).read_bytes().replace(b"\r\n", b"\n")


def load(rel):
    return json.loads(lf_bytes(rel).decode("utf-8"))


def all_keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from all_keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from all_keys(v)


def main():
    rs, wf, fu = load(RES), load(WF), load(FU)
    main_keys = ["full_52feat(headline)", "full_FWL_ton_infold", "enc_FWL_ton_infold", "suffix_only_LR",
                 "charngram_names(char_wb2-4,C=1)"]
    est = {k: {"oof": rs["main"][k]["oof"], "ci_pair": rs["main"][k].get("ci_pair"),
               "t4_boot_mean": (rs["main"][k].get("t4_boot") or {}).get("mean")} for k in main_keys}
    proj = {}
    for line in fu["F8h_shipped_test1_stdout"]:
        m = re.match(r"\s*((?:enc|gen)_\w+_proj)\s+([0-9.]+)\s*$", line)
        if m:
            proj[m.group(1)] = float(m.group(2))
    # best of the six key-direction projections only; the stdout's best single feature overall is
    # mean_logit_entropy (its line carries a trailing '*', which the old all-feature pattern skipped)
    best = max(proj.items(), key=lambda kv: kv[1])
    reading = fu["F5_reading"]
    s4 = fu["F5_S4_rerun"]["S4_reading_2of3"]
    s4v = {k: (v.get("auroc") if isinstance(v, dict) else v) for k, v in s4.items()}
    result = {
        "LABEL": "Review round 5, findings 1, 6 and 13. Collected from stored outputs with their keys; nothing re-fitted.",
        "inputs": {r: hashlib.sha256(lf_bytes(r)).hexdigest() for r in (RES, WF, FU)},
        "hash_note": "sha256 of the LF form (CRLF replaced by LF), i.e. of the git blob",
        "finding6_estimators": {
            "headline_bootstrap_mean_A1_headline_t4_full": rs["A1_headline_t4_full"]["mean"],
            "main": est,
            "wordfreq_zipf_infold_covariate": wf["zipf_infold_covariate"],
            "wordfreq_keys_naming_a_bootstrap": sorted(k for k in all_keys(wf) if "boot" in k.lower() or k.startswith("t4")),
            "note": "main[...].oof and the wordfreq figures are single out-of-fold runs on the observed 90 trials; t4_boot_mean and A1_headline_t4_full are means of 1,000 bootstrap AUROCs (a different estimator)."},
        "finding13_projection_auroc_shipped_test1": proj,
        "finding13_best_projection_feature": best,
        "finding1_reading": {
            "F5_reading_LABEL": reading["LABEL"],
            "per_fake_fields": sorted({k for v in reading["per_fake"].values() if isinstance(v, dict) for k in v}),
            "word_correct_anywhere_in_F5_reading": "correct" in json.dumps(reading).lower(),
            "F5_counts": fu["F5_counts"],
            "S4_reading_2of3": s4v},
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(REPO).as_posix())
    print("  A1 mean", result["finding6_estimators"]["headline_bootstrap_mean_A1_headline_t4_full"])
    print("  oof", {k: v["oof"] for k, v in est.items()}, "boot", {k: v["t4_boot_mean"] for k, v in est.items()})
    print("  zipf", wf["zipf_infold_covariate"], "boot keys", result["finding6_estimators"]["wordfreq_keys_naming_a_bootstrap"])
    print("  proj", proj, "best", best)
    print("  reading", result["finding1_reading"]["word_correct_anywhere_in_F5_reading"], fu["F5_counts"], s4v)


if __name__ == "__main__":
    main()
