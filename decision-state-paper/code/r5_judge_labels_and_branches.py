"""Judge labels on the 45 unanswerable-question trials, and the branch anatomy (review round 5, findings 4 and 5).

CPU only, no model or judge run. From the stored files:
  * the single 3B judge's labels (behavior_llm) on the 45 confab-condition trials, by question kind, and
    whether its CONFABULATED flags overlap the panel's YES flags (data/panel_by_kind_r4.json);
  * the 3B-HEDGED trials the panel flags, and where two short phrases that treat a fabricated entity as real
    sit in the judged file. Only the Troms phrase states an invented fact (a real county for an invented
    mountain); the Thornbridge phrase asserts the question's own premise and that response names no river.
    The two notes are one unblinded reading of the stored text;
  * the prompts whose three reps the regex labels (field "behavior") split between HEDGED and CONFABULATED,
    their cross-label pairs' first differing token (cf. data/divergence_analysis.json), the 3B judge's
    labels for the same responses, and each regex-CONFABULATED rep's length;
  * the divergence table recomputed from stored token ids: first differing token between reps of a prompt
    that share a behavior_llm label;
  * the stored logit_margin / logit_entropy at the first branch token of the four prompts in the top-k table.
Token ids are recorded as ids; data/r5_decoded_token_ids.json decodes them.

Every input hash is the sha256 of the LF form (CRLF -> LF), which is the git blob.
Output: data/r5_judge_labels_and_branches.json
"""
import hashlib
import itertools
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
D = "decision-state-paper/data/"
JUDGED, RESULTS, PANEL, DIV = (D + "decision_moment_judged.json", D + "decision_moment_results.json",
                               D + "panel_by_kind_r4.json", D + "divergence_analysis.json")
OUT = PAPER / "data" / "r5_judge_labels_and_branches.json"
NEEDLES = {"Thornbridge is a small town in Montana":
               "asserts the question's fabricated town as real; repeats the premise, names no river",
           "Norwegian county of Troms": "places the fabricated mountain in a real county: an invented fact"}
TOPK = ["Nobel", "Oscar", "France", "Kellerton"]


def lf_bytes(rel):
    return (REPO / rel).read_bytes().replace(b"\r\n", b"\n")


def load(rel):
    return json.loads(lf_bytes(rel).decode("utf-8"))


def firstdiff(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return min(len(a), len(b))


def main():
    dj, dr, pb, dv = load(JUDGED), load(RESULTS), load(PANEL), load(DIV)
    jlines = lf_bytes(JUDGED).decode("utf-8").split("\n")

    # ---- finding 4: 3B labels vs panel on the 45 confab trials
    conf = [t for t in dj["trials"] if t["condition"] == "confab"]
    pan = {t["i"]: t for t in pb["trials"]}
    aligned = all(pan[i]["trial_idx"] == conf[i]["trial_idx"] and pan[i]["judge_3b"] == conf[i]["behavior_llm"]
                  for i in range(len(conf)))
    by_kind = defaultdict(Counter)
    for i, t in enumerate(conf):
        by_kind[pan[i]["kind"]][t["behavior_llm"]] += 1
    c3 = [i for i, t in enumerate(conf) if t["behavior_llm"] == "CONFABULATED"]
    cp = [i for i in range(len(conf)) if pan[i]["panel"] == "YES"]
    needles = {}
    for s, note in NEEDLES.items():
        needles[s] = {"reading_unblinded": note, "trials": [{"i": i, "rep": t["rep"], "judge_3b": t["behavior_llm"], "panel": pan[i]["panel"]}
                                 for i, t in enumerate(conf) if s in t["generated_text"]],
                      "judged_file_lines": [k + 1 for k, l in enumerate(jlines) if s in l]}
    f4 = {"n_confab_trials": len(conf), "panel_file_aligned_with_judged_file": aligned,
          "judge_3b_counts": dict(sorted(Counter(t["behavior_llm"] for t in conf).items())),
          "judge_3b_counts_by_kind": {k: dict(sorted(v.items())) for k, v in sorted(by_kind.items())},
          "judge_3b_CONFABULATED_i": c3, "panel_YES_i": cp, "overlap": sorted(set(c3) & set(cp)),
          "judge_3b_label_of_panel_YES": {str(i): conf[i]["behavior_llm"] for i in cp},
          "phrases_treating_fabricated_entity_as_real": needles}

    # ---- finding 5: regex-mixed prompts
    by = defaultdict(list)
    for t in dr["trials"]:
        by[t["prompt"]].append(t)
    llm = {(t["prompt"], t["rep"]): t["behavior_llm"] for t in dj["trials"]}
    mixed = {p: ts for p, ts in by.items()
             if any(t["behavior"] == "HEDGED" for t in ts) and any(t["behavior"] == "CONFABULATED" for t in ts)}
    cross, mrec = [], []
    for p, ts in mixed.items():
        h = [t for t in ts if t["behavior"] == "HEDGED"]
        c = [t for t in ts if t["behavior"] == "CONFABULATED"]
        pairs = []
        for a in h:
            for b in c:
                k = firstdiff(a["trajectory"]["tokens"], b["trajectory"]["tokens"])
                cross.append(k)
                pairs.append({"hedged_rep": a["rep"], "confab_rep": b["rep"], "first_diff_token": k,
                              "hedged_token_id": a["trajectory"]["tokens"][k], "confab_token_id": b["trajectory"]["tokens"][k]})
        mrec.append({"prompt": p, "regex": {str(t["rep"]): t["behavior"] for t in ts},
                     "judge_3b": {str(t["rep"]): llm.get((p, t["rep"])) for t in ts},
                     "n_generated": {str(t["rep"]): t["n_generated"] for t in ts},
                     "cross_label_pairs": pairs})
    n_resp = sum(len(ts) for ts in mixed.values())
    n_hedged_llm = sum(llm.get((p, t["rep"])) == "HEDGED" for p, ts in mixed.items() for t in ts)
    f5 = {"n_regex_mixed_prompts": len(mixed), "n_cross_label_pairs": len(cross),
          "cross_pair_first_diff_tokens": sorted(cross),
          "cross_pair_mean": round(st.mean(cross), 6) if cross else None,
          "divergence_analysis_json": dv,
          "n_responses": n_resp, "n_judge_3b_HEDGED": n_hedged_llm, "prompts": mrec}

    # ---- divergence table recompute (behavior_llm, same-label pairs, all judged trials)
    byp = defaultdict(list)
    for t in dj["trials"]:
        byp[t["prompt"]].append(t)
    same = defaultdict(list)
    for p, ts in byp.items():
        for a, b in itertools.combinations(ts, 2):
            if a["behavior_llm"] == b["behavior_llm"]:
                same[a["behavior_llm"]].append(firstdiff(a["trajectory"]["tokens"], b["trajectory"]["tokens"]))
    div = {k: {"n_pairs": len(v), "mean": round(st.mean(v), 4), "median": st.median(v)}
           for k, v in sorted(same.items())}

    # ---- branch-token stats for the top-k table prompts
    topk = {}
    for key in TOPK:
        for p, ts in by.items():
            if key in p:
                toks = [t["trajectory"]["tokens"] for t in ts]
                k = min(firstdiff(a, b) for a, b in itertools.combinations(toks, 2))
                topk[key] = {"prompt": p, "branch_token": k,
                             "token_ids_by_rep": {str(t["rep"]): t["trajectory"]["tokens"][k] for t in ts},
                             "regex_by_rep": {str(t["rep"]): t["behavior"] for t in ts},
                             "logit_margin_by_rep": sorted(set(round(t["trajectory"]["logit_margin"][k], 4) for t in ts)),
                             "logit_entropy_by_rep": sorted(set(round(t["trajectory"]["logit_entropy"][k], 4) for t in ts))}
    result = {
        "LABEL": "Review round 5, findings 4 and 5. Recomputed from stored labels and token ids; no model or judge run. The top-k probabilities printed in the paper are in no shipped file.",
        "inputs": {r: hashlib.sha256(lf_bytes(r)).hexdigest() for r in (JUDGED, RESULTS, PANEL, DIV)},
        "hash_note": "sha256 of the LF form (CRLF replaced by LF), i.e. of the git blob",
        "finding4_judge_labels": f4, "finding5_regex_mixed_prompts": f5,
        "divergence_same_label_pairs_behavior_llm": div, "topk_branch_tokens": topk,
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(REPO).as_posix())
    print("  3B:", f4["judge_3b_counts"], f4["judge_3b_counts_by_kind"], "3B CONFAB", c3, "panel YES", cp, "overlap", f4["overlap"])
    print("  needles:", {s[:20]: (v["trials"], v["judged_file_lines"]) for s, v in needles.items()})
    print("  mixed:", len(mixed), "pairs", len(cross), "mean", f5["cross_pair_mean"], "3B HEDGED", n_hedged_llm, "/", n_resp)
    for m in mrec:
        print("   ", m["prompt"][:50], m["regex"], m["n_generated"], [(q["first_diff_token"], q["hedged_token_id"], q["confab_token_id"]) for q in m["cross_label_pairs"]])
    print("  divergence:", div)
    for k, v in topk.items():
        print("  topk", k, v["branch_token"], v["token_ids_by_rep"], v["logit_margin_by_rep"], v["logit_entropy_by_rep"])


if __name__ == "__main__":
    main()
