"""Counts behind the review round 6 (decision-state) corrections. CPU only, no model or judge run.

Run from the repository root:
  python decision-state-paper/code/r6_ds_review_checks.py --repo . --out decision-state-paper/data/r6_ds_review_checks.json

From stored files only:
  * preamble_and_cliff: how many generations in data/decision_moment_judged.json open with the fixed template
    "Let me think through this carefully", by condition; for those, the preamble length (tokens up to and including
    the ':' of "My reasoning:", found as token ids 271 4888 31626 25), the share of preamble positions 1..end with
    logit entropy < 0.02 and margin > 12, and the stored entropy and margin at the first token after the preamble.
  * judge_3b_labels: the 3B judge's labels (field behavior_llm, no shipped producer) on the 45 answerable trials,
    and the same-label rep pairs behind tab:divergence split by whether the prompt is answerable.
  * second_judge_original_label: which stored label second_judge_results.json's original_label equals, and the
    agreement of the Qwen labels with the regex and with the 3B labels.
  * regex_by_kind: regex (behavior), regex + declarative markers (behavior_regex_v2) and 3B labels on the 45
    unanswerable trials, by question kind (domain).
  * margin_by_template_opening: tokens 0-9 mean logit margin and entropy, fake minus real (Cohen's d, pooled
    population SD as in r5_early_token_windows.py), within generations that open with the template and within those
    that do not (data/matched_burn_results.json).
  * zipf_structure: share of the 30 names' Zipf variance that lies between the six question templates, and the
    templates whose fakes all end in "-ton".
  * followup_reading_counts: from r4_followups.json F5_reading, per fake the number of reps read R (answers or
    starts to answer) and R-or-I (treated as real).
Trials are treated as independent, as in the paper's original tests. Every input hash is the sha256 of the LF form.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

DJ = "decision-state-paper/data/decision_moment_judged.json"
MB = "decision-state-paper/data/matched_burn_results.json"
SJ = "decision-state-paper/data/second_judge_results.json"
WF = "decision-state-paper/data/suffix_controls/r4_wordfreq.json"
FU = "decision-state-paper/data/suffix_controls/r4_followups.json"
TEMPLATE = "Let me think through this carefully"
PREAMBLE_END = [271, 4888, 31626, 25]  # "\n\nMy reasoning:"


def lf_bytes(repo, rel):
    return (repo / rel).read_bytes().replace(b"\r\n", b"\n")


def load(repo, rel):
    return json.loads(lf_bytes(repo, rel).decode("utf-8"))


def preamble_len(tok):
    for j in range(len(tok) - 3):
        if tok[j:j + 4] == PREAMBLE_END:
            return j + 4
    return None


def first_diff(a, b):
    for j, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return j
    return min(len(a), len(b))


def d_pooled(k, r):
    return float((np.mean(k) - np.mean(r)) / np.sqrt((np.var(r) + np.var(k)) / 2))


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in sorted(set(a) | set(b)))
    return round(po, 4), round((po - pe) / (1 - pe), 4)


def r4(x):
    return round(float(x), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo = Path(a.repo)
    T = load(repo, DJ)["trials"]

    # ---- preamble and cliff (decision_moment_judged)
    opens = Counter((t["condition"], t["generated_text"].startswith(TEMPLATE)) for t in T)
    rows = []
    for t in T:
        if not t["generated_text"].startswith(TEMPLATE):
            continue
        tr = t["trajectory"]
        L = preamble_len(tr["tokens"])
        rows.append({"condition": t["condition"], "L": L, "H": tr["logit_entropy"], "M": tr["logit_margin"]})
    missing = sum(r["L"] is None for r in rows)
    rows = [r for r in rows if r["L"] is not None]
    Ls = [r["L"] for r in rows]
    pos_H = [h for r in rows for h in r["H"][1:r["L"]]]
    pos_M = [m for r in rows for m in r["M"][1:r["L"]]]
    cliff = {}
    for c in ("honest", "confab"):
        rr = [r for r in rows if r["condition"] == c]
        h = [r["H"][r["L"]] for r in rr]
        m = [r["M"][r["L"]] for r in rr]
        cliff[c] = {"n": len(rr), "H_mean": r4(np.mean(h)), "H_min": r4(min(h)), "H_max": r4(max(h)),
                    "n_H_in_1.5_to_2.7": sum(1.5 <= x <= 2.7 for x in h),
                    "margin_mean": r4(np.mean(m)), "margin_min": r4(min(m)), "margin_max": r4(max(m)),
                    "n_margin_le_0.5": sum(x <= 0.5 for x in m),
                    "prev_token_H_mean": r4(np.mean([r["H"][r["L"] - 1] for r in rr])),
                    "prev_token_margin_mean": r4(np.mean([r["M"][r["L"] - 1] for r in rr]))}
    preamble = {
        "opens_with_template": {"unanswerable_confab": opens[("confab", True)], "answerable_honest": opens[("honest", True)],
                                "total": opens[("confab", True)] + opens[("honest", True)], "n_trials": len(T)},
        "preamble_end_not_found": missing,
        "preamble_tokens": {"median": float(np.median(Ls)), "mean": r4(np.mean(Ls)), "min": min(Ls), "max": max(Ls)},
        "token0_H_median": r4(np.median([r["H"][0] for r in rows])),
        "positions_1_to_end": {"n": len(pos_H), "share_H_lt_0.02": r4(np.mean(np.array(pos_H) < 0.02)),
                               "share_margin_gt_12": r4(np.mean(np.array(pos_M) > 12)),
                               "n_preambles_all_H_lt_0.02": sum(max(r["H"][1:r["L"]]) < 0.02 for r in rows),
                               "n_preambles_all_margin_gt_12": sum(min(r["M"][1:r["L"]]) > 12 for r in rows)},
        "cliff_first_token_after_preamble": cliff,
    }

    # ---- 3B labels on answerable trials and the tab:divergence pairs
    ans = Counter(t["behavior_llm"] for t in T if t["condition"] == "honest")
    byp = defaultdict(list)
    for t in T:
        byp[t["prompt"]].append(t)
    pairs = defaultdict(list)
    for p, ts in byp.items():
        for x, y in combinations(ts, 2):
            if x["behavior_llm"] == y["behavior_llm"]:
                pairs[x["behavior_llm"]].append({"prompt": p, "condition": x["condition"],
                                                 "first_diff": first_diff(x["trajectory"]["tokens"], y["trajectory"]["tokens"])})
    div = {}
    for lab in ("HONEST", "HEDGED"):
        v = pairs[lab]
        own = "honest" if lab == "HONEST" else "confab"
        kept = [q["first_diff"] for q in v if q["condition"] == own]
        div[lab] = {"n_pairs": len(v), "n_pairs_other_condition": len(v) - len(kept),
                    "other_condition_pairs": sorted([[q["prompt"], q["first_diff"]] for q in v if q["condition"] != own]),
                    "restricted_to_" + ("answerable" if own == "honest" else "unanswerable"):
                        {"n_pairs": len(kept), "mean": r4(np.mean(kept)), "median": float(np.median(kept))}}
    judge = {"producer_note": "no shipped script or prompt produces behavior_llm; the judged file records only judge_model and judge_timestamp",
             "answerable_labels": dict(sorted(ans.items())), "divergence_pairs": div}

    # ---- second judge
    sj = load(repo, SJ)
    by = {t["trial_idx"]: t for t in T}
    q = [s["qwen_label"] for s in sj["trials"]]
    reg = [by[s["trial_idx"]]["behavior"] for s in sj["trials"]]
    llm = [by[s["trial_idx"]]["behavior_llm"] for s in sj["trials"]]
    orig = [s["original_label"] for s in sj["trials"]]
    second = {"original_judge_field": sj["original_judge"],
              "original_label_equals_regex_behavior": sum(o == r for o, r in zip(orig, reg)),
              "original_label_equals_behavior_llm": sum(o == l for o, l in zip(orig, llm)), "n": len(orig),
              "qwen_vs_regex_raw_kappa": list(kappa(reg, q)), "qwen_vs_3B_raw_kappa": list(kappa(llm, q))}

    # ---- regex by kind on unanswerable trials
    kinds = {}
    for d in ("fabricated", "future", "impossible"):
        tt = [t for t in T if t["condition"] == "confab" and t["domain"] == d]
        kinds[d] = {k: dict(sorted(Counter(t[k] for t in tt).items())) for k in ("behavior", "behavior_regex_v2", "behavior_llm")}

    # ---- margin by template opening (matched burn)
    mb = load(repo, MB)["trials"]
    strat = {}
    for f in ("logit_margin", "logit_entropy"):
        strat[f] = {}
        for name, grp in (("opens_with_template", True), ("does_not", False), ("all", None)):
            rr = [float(np.mean(t["trajectory"][f][0:10])) for t in mb if t["is_real"] and (grp is None or t["generated_text"].startswith(TEMPLATE) == grp)]
            kk = [float(np.mean(t["trajectory"][f][0:10])) for t in mb if not t["is_real"] and (grp is None or t["generated_text"].startswith(TEMPLATE) == grp)]
            strat[f][name] = {"n_real": len(rr), "n_fake": len(kk), "real_mean": r4(np.mean(rr)), "fake_mean": r4(np.mean(kk)),
                              "d_fake_minus_real": r4(d_pooled(kk, rr))}

    # ---- Zipf structure
    zp = load(repo, WF)["zipf_per_name"]
    tmpl = {("real" if t["is_real"] else "fake") + "|" + t["entity"]: t["template"] for t in mb}
    x = np.array(list(zp.values()), dtype=float)
    g = [tmpl[k] for k in zp]
    grand = x.mean()
    ssb = sum(g.count(tm) * (np.mean([v for k, v in zp.items() if tmpl[k] == tm]) - grand) ** 2 for tm in set(g))
    ton = defaultdict(list)
    for k in zp:
        if k.startswith("fake|"):
            ton[tmpl[k]].append(k.split("|")[1].endswith("ton"))
    zipf = {"n_names": len(zp), "n_templates": len(set(g)), "between_template_share_of_variance": r4(ssb / ((x - grand) ** 2).sum()),
            "templates_all_fakes_ton": sorted(tm for tm, v in ton.items() if all(v))}

    # ---- follow-up reading counts
    rd = load(repo, FU)["F5_reading"]["per_fake"]
    fol = {n: {"labels": v["labels_rep0_1_2"], "n_R": v["labels_rep0_1_2"].count("R"),
               "n_R_or_I": sum(l in ("R", "I") for l in v["labels_rep0_1_2"])} for n, v in sorted(rd.items())}
    fol_sum = {"R_in_2plus": sorted(n for n, v in fol.items() if v["n_R"] >= 2),
               "R_or_I_in_2plus": sorted(n for n, v in fol.items() if v["n_R_or_I"] >= 2), "per_fake": fol}

    out = {"LABEL": "Review round 6 (decision-state) verification counts. Stored files only; no model, judge or GPU run. Trials treated as independent.",
           "hash_note": "sha256 of the LF form (CRLF replaced by LF), i.e. of the git blob",
           "inputs": {rel: hashlib.sha256(lf_bytes(repo, rel)).hexdigest() for rel in (DJ, MB, SJ, WF, FU)},
           "preamble_and_cliff": preamble, "judge_3b_labels": judge, "second_judge_original_label": second,
           "regex_by_kind": kinds, "margin_by_template_opening": strat, "zipf_structure": zipf,
           "followup_reading_counts": fol_sum}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
