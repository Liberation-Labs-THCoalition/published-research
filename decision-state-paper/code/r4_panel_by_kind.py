"""Decision-state paper: the three-judge panel's YES votes, counted by question kind.

CPU only, stored files only. No judge or model is run. Writes only its own output file.
Added for the fourth-round (r4) review check of PR #18 (November 2026).
Run with `python -I r4_panel_by_kind.py [path/to/decision-state-paper]`.

Input : data/decision_moment_judged.json   (the 45 'confab' trials, in file order)
        data/panel_judge_results.json      (per-judge labels, same order: panel_judge.py
                                            builds both lists from the same filter)
Output: data/panel_by_kind_r4.json

What it reports
  - YES votes and the panel majority per question kind (the 'domain' field).
  - The entity string panel_judge.extract_entity() put into the judge prompt, per trial.
    The prompt tells every judge that this entity 'does NOT exist'.
  - Recomputed agreement (pairwise kappa with UNKNOWN as its own category, as the
    shipped kappa() does; unanimity), as a positive control against the shipped file.
  - Two restricted counts, both POST-HOC (chosen after reading which trials were flagged):
      (a) excluding the three 'President of France in 2032' trials;
      (b) the fabricated-entity kind only, the one kind the prompt's premise fits.
"""
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "panel_by_kind_r4.json"
f_j = REPO / "data" / "decision_moment_judged.json"
f_p = REPO / "data" / "panel_judge_results.json"
raw_j, raw_p = f_j.read_bytes(), f_p.read_bytes()
d = json.loads(raw_j.decode("utf-8"))
p = json.loads(raw_p.decode("utf-8"))
confab = [t for t in d["trials"] if t["condition"] == "confab"]
J = p["panel"]
L = p["per_judge_labels"]
assert len(confab) == p["n_trials"] == 45

R = {"inputs": {f_j.relative_to(REPO.parent).as_posix(): hashlib.sha256(raw_j).hexdigest(),
                f_p.relative_to(REPO.parent).as_posix(): hashlib.sha256(raw_p).hexdigest()},
     "index_note": "i is the 0-based position among the 45 confab trials; trial_idx is the file's own field"}


def kappa(a, b):  # verbatim from panel_judge.py
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


pk = {f"{a}|{b}": round(kappa(L[a], L[b]), 3) for a, b in combinations(J, 2)}
unan = sum(1 for i in range(45) if len(set(L[j][i] for j in J)) == 1)
maj = ["YES" if [L[j][i] for j in J].count("YES") >= 2 else "NO" for i in range(45)]
R["positive_control"] = {
    "pairwise_kappa_recomputed": pk, "matches_shipped": pk == p["pairwise_kappa"],
    "mean_pairwise_kappa": round(sum(pk.values()) / 3, 3), "unanimous_trials": unan,
    "majority_matches_shipped_verdict": maj == p["panel_verdict"],
    "unknown_count_per_judge": {j: L[j].count("UNKNOWN") for j in J}}

rows = []
for i, t in enumerate(confab):
    rows.append({"i": i, "trial_idx": t["trial_idx"], "kind": t["domain"], "rep": t["rep"],
                 "prompt": t["prompt"], "entity_in_judge_prompt": p["entities"][i],
                 "votes": {j: L[j][i] for j in J}, "panel": p["panel_verdict"][i],
                 "judge_3b": t.get("behavior_llm")})
R["trials"] = rows

kinds = sorted(set(r["kind"] for r in rows))
by = {}
for k in kinds:
    rr = [r for r in rows if r["kind"] == k]
    by[k] = {"n_trials": len(rr), "n_prompts": len(set(r["prompt"] for r in rr)),
             "panel_yes": sum(r["panel"] == "YES" for r in rr),
             "panel_yes_trials": [(r["i"], r["prompt"], r["rep"]) for r in rr if r["panel"] == "YES"],
             "yes_votes_per_judge": {j: sum(r["votes"][j] == "YES" for r in rr) for j in J},
             "unknown_per_judge": {j: sum(r["votes"][j] == "UNKNOWN" for r in rr) for j in J},
             "judge_3b_labels": dict(Counter(r["judge_3b"] for r in rr)),
             "entities_in_judge_prompt": dict(Counter(r["entity_in_judge_prompt"] for r in rr))}
R["by_kind"] = by
R["panel_yes_total"] = sum(r["panel"] == "YES" for r in rows)

# Evidence that the two France YES trials are honest answers (substring checks that can fail).
ev = {27: ["has not happened yet", "Macron was elected President of France in 2017"],
      29: ["because that year is in the future"]}
R["france_yes_evidence"] = {str(i): {s: (s in confab[i]["generated_text"][:500]) for s in ss}
                            for i, ss in ev.items()}

fr = [r for r in rows if "France" in r["prompt"]]
nofr = [r for r in rows if "France" not in r["prompt"]]
fab = [r for r in rows if r["kind"] == "fabricated"]
R["posthoc_restricted_counts"] = {
    "LABEL": "POST-HOC: both restrictions were chosen after reading which trials the panel flagged.",
    "a_excluding_france_trials": {"yes": sum(r["panel"] == "YES" for r in nofr), "n": len(nofr),
                                  "caveat": "drops 2 YES votes from a misapplied prompt but keeps 28 NO votes "
                                            "from the same misapplied prompt on the other future/unknowable trials"},
    "b_fabricated_kind_only": {"yes": sum(r["panel"] == "YES" for r in fab), "n": len(fab),
                               "principle": "the judge prompt asserts the entity does not exist; that premise "
                                            "holds only for the fabricated-entity kind, whatever the votes were"},
    "france_trials": [(r["i"], r["rep"], r["panel"], r["votes"]) for r in fr]}

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: R[k] for k in ("positive_control", "panel_yes_total", "france_yes_evidence",
                                    "posthoc_restricted_counts")}, indent=1, ensure_ascii=False))
for k, v in by.items():
    print(k, {kk: v[kk] for kk in ("n_trials", "panel_yes", "yes_votes_per_judge", "unknown_per_judge",
                                    "judge_3b_labels", "entities_in_judge_prompt")})
print("wrote", OUT.relative_to(REPO.parent).as_posix())
