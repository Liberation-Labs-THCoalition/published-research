"""Judge agreement split by question kind (review round 5, finding 3).

CPU only, no judge run. Recomputes Cohen's kappa between the three panel judges from the stored votes in
data/panel_by_kind_r4.json, for all 45 trials and by question kind, and on the trials where both judges
of a pair gave a parseable (YES/NO) answer. Records where the two judges that answered every trial
disagree, and the unparseable answers by kind.

Every input hash is the sha256 of the LF form (CRLF -> LF), which is the git blob.
Output: data/r5_panel_agreement_by_kind.json
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
PB = "decision-state-paper/data/panel_by_kind_r4.json"
OUT = PAPER / "data" / "r5_panel_agreement_by_kind.json"


def lf_bytes(rel):
    return (REPO / rel).read_bytes().replace(b"\r\n", b"\n")


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    labs = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labs)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def main():
    pb = json.loads(lf_bytes(PB).decode("utf-8"))
    T = pb["trials"]
    judges = sorted(T[0]["votes"].keys())
    groups = {"all": T, "fabricated": [t for t in T if t["kind"] == "fabricated"],
              "future_and_impossible": [t for t in T if t["kind"] != "fabricated"],
              "future": [t for t in T if t["kind"] == "future"],
              "impossible": [t for t in T if t["kind"] == "impossible"]}
    res = {}
    for g, ts in groups.items():
        r = {"n": len(ts), "votes": {j: dict(sorted(Counter(t["votes"][j] for t in ts).items())) for j in judges},
             "panel_YES_i": [t["i"] for t in ts if t["panel"] == "YES"],
             "unanimous": sum(len(set(t["votes"].values())) == 1 for t in ts), "pairs": {}}
        for x in range(len(judges)):
            for y in range(x + 1, len(judges)):
                a = [t["votes"][judges[x]] for t in ts]
                b = [t["votes"][judges[y]] for t in ts]
                both = [(p, q) for p, q in zip(a, b) if p in ("YES", "NO") and q in ("YES", "NO")]
                r["pairs"][f"{judges[x]}|{judges[y]}"] = {
                    "agree": sum(p == q for p, q in zip(a, b)), "kappa_raw_labels": kappa(a, b),
                    "both_parseable_n": len(both),
                    "both_parseable_agree": sum(p == q for p, q in both),
                    "both_parseable_kappa": kappa([p for p, _ in both], [q for _, q in both]),
                    "both_YES": sum(p == q == "YES" for p, q in zip(a, b))}
        res[g] = r
    full = [j for j in judges if all(t["votes"][j] in ("YES", "NO") for t in T)]
    dis = []
    if len(full) == 2:
        dis = [{"i": t["i"], "kind": t["kind"], "entity_in_judge_prompt": t["entity_in_judge_prompt"],
                full[0]: t["votes"][full[0]], full[1]: t["votes"][full[1]]}
               for t in T if t["votes"][full[0]] != t["votes"][full[1]]]
    unparse = {j: dict(sorted(Counter(t["kind"] for t in T if t["votes"][j] not in ("YES", "NO")).items()))
               for j in judges}
    result = {
        "LABEL": "Review round 5, finding 3. Agreement recomputed from stored votes; no judge was re-run. Agreement on the future and unknowable trials under a correctly named entity has not been measured.",
        "inputs": {PB: hashlib.sha256(lf_bytes(PB)).hexdigest()},
        "hash_note": "sha256 of the LF form (CRLF replaced by LF), i.e. of the git blob",
        "judges": judges,
        "judges_with_a_parseable_answer_on_every_trial": full,
        "disagreements_between_those_two": dis,
        "unparseable_by_kind": unparse,
        "shipped_positive_control": pb.get("positive_control"),
        "by_group": res,
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(REPO).as_posix())
    k = f"{full[0]}|{full[1]}" if len(full) == 2 else None
    for g in ("all", "fabricated", "future_and_impossible"):
        if k:
            p = res[g]["pairs"].get(k) or res[g]["pairs"].get("|".join(reversed(k.split("|"))))
            print(f"  {g}: {k} agree {p['agree']}/{res[g]['n']} kappa {p['kappa_raw_labels']} both_YES {p['both_YES']}")
    print("  disagreements:", [(d["i"], d["entity_in_judge_prompt"]) for d in dis])
    print("  unparseable:", unparse)


if __name__ == "__main__":
    main()
