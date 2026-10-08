"""Human validation of the rerun's pass-1 labels (RERUN_PREREG.md: "Agreement with pass 1 is reported (kappa on
fabrication yes/no). It does not change the outcome."). Post-result; reads only frozen outputs and the human ratings.

The 72 items are a stratified sample: 12 from each cell of bias x pass-1 fabrication yes/no. Kappa on the sample is
reported as registered; because the sample over-represents the judge's fabrication labels, agreement is also given
within each stratum and reweighted to the 720 responses.

    python3 human_validation.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FAB = {"FULL_CONFAB", "COSMETIC_HEDGE"}
SEED, N_BOOT = 20261007, 10_000


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    key = json.loads((RES / "human_validation_key.json").read_text())
    human = {r["item"]: r["label"] for r in json.loads((HERE / "human_ratings.json").read_text())["labels"]}
    p1 = {tuple([r["key"][0], int(r["key"][1]), int(r["key"][2]), float(r["key"][3])]): r["judge"]["classification"]
          for r in json.loads((RES / "pass1.json").read_text()) if r.get("judge", {}).get("classification")}
    rows = []
    for item, k in sorted(key.items(), key=lambda kv: int(kv[0])):
        kk = (k[0], int(k[1]), int(k[2]), float(k[3]))
        rows.append({"item": int(item), "prompt": (k[0], int(k[1])), "bias": float(k[3]), "judge": p1[kk], "human": human[int(item)]})
    J6, H6 = [r["judge"] for r in rows], [r["human"] for r in rows]
    J2, H2 = [x in FAB for x in J6], [x in FAB for x in H6]
    res = {"n": len(rows), "agree_fab": sum(a == b for a, b in zip(J2, H2)), "kappa_fab": kappa(J2, H2),
           "agree_6class": sum(a == b for a, b in zip(J6, H6)), "kappa_6class": kappa(J6, H6)}
    # prompt-cluster bootstrap for both kappas
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt"]].append(r)
    prompts = list(by_prompt)
    rng = np.random.default_rng(SEED)
    kb, k6 = [], []
    for _ in range(N_BOOT):
        pick = [x for i in rng.integers(0, len(prompts), len(prompts)) for x in by_prompt[prompts[i]]]
        a2, b2 = [x["judge"] in FAB for x in pick], [x["human"] in FAB for x in pick]
        a6, b6 = [x["judge"] for x in pick], [x["human"] for x in pick]
        kb.append(kappa(a2, b2)); k6.append(kappa(a6, b6))
    kb, k6 = np.array([x for x in kb if np.isfinite(x)]), np.array([x for x in k6 if np.isfinite(x)])
    res["kappa_fab_ci95"] = [float(np.percentile(kb, 2.5)), float(np.percentile(kb, 97.5))]
    res["kappa_6class_ci95"] = [float(np.percentile(k6, 2.5)), float(np.percentile(k6, 97.5))]
    res["n_prompts"] = len(prompts)
    # confusion
    res["confusion_fab"] = {"judge_fab_human_fab": sum(a and b for a, b in zip(J2, H2)), "judge_fab_human_not": sum(a and not b for a, b in zip(J2, H2)),
                            "judge_not_human_fab": sum(b and not a for a, b in zip(J2, H2)), "judge_not_human_not": sum(not a and not b for a, b in zip(J2, H2))}
    res["confusion_6class"] = {f"{j} -> {h}": c for (j, h), c in sorted(Counter(zip(J6, H6)).items())}
    # strata and reweighting to the 720
    pop = Counter((float(k[3]), v in FAB) for k, v in p1.items())
    strata = {}
    for b in (0.0, 2.0, 5.0):
        for f in (True, False):
            cell = [r for r in rows if r["bias"] == b and (r["judge"] in FAB) == f]
            agree = sum((r["human"] in FAB) == f for r in cell)
            strata[f"bias {b}, judge {'fab' if f else 'not fab'}"] = {"n": len(cell), "human_agrees": agree, "N_population": pop[(b, f)]}
    res["strata"] = strata
    tot = sum(v["N_population"] for v in strata.values())
    res["agree_fab_reweighted_to_720"] = sum(v["N_population"] / tot * v["human_agrees"] / v["n"] for v in strata.values() if v["n"])
    # human-label fabrication rate per bias, reweighted (descriptive)
    rates = {}
    for b in (0.0, 2.0, 5.0):
        Nb = pop[(b, True)] + pop[(b, False)]
        r = 0.0
        for f in (True, False):
            cell = [x for x in rows if x["bias"] == b and (x["judge"] in FAB) == f]
            r += pop[(b, f)] / Nb * (sum(x["human"] in FAB for x in cell) / len(cell))
        rates[str(b)] = {"judge_rate": pop[(b, True)] / Nb, "human_rate_reweighted": r}
    res["fabrication_rate_by_bias"] = rates
    # post hoc adjudication of the disputed items (adjudication.json): a disputed item counts as fabrication when the
    # check found an invented specific. Reported beside the registered kappa, never instead of it.
    adj_path = HERE / "adjudication.json"
    if adj_path.exists():
        adj = json.loads(adj_path.read_text())["items"]
        disputed = sorted(r["item"] for r in rows if r["judge"] in FAB and r["human"] not in FAB)
        assert disputed == sorted(int(i) for i in adj), "adjudication must cover exactly the disputed items"
        Ha = [(r["human"] in FAB) or adj.get(str(r["item"]), {}).get("verdict") == "invented" for r in rows]
        res["adjudicated"] = {"rule": "a disputed item counts as fabrication when the check found an invented specific",
                              "agree_fab": sum(a == b for a, b in zip(J2, Ha)), "kappa_fab": kappa(J2, Ha),
                              "n_invented": sum(v["verdict"] == "invented" for v in adj.values()),
                              "n_borderline": sum(v["verdict"] == "borderline" for v in adj.values())}
    (HERE / "human_validation_results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
