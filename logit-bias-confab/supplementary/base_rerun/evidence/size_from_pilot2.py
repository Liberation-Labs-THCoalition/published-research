"""Amendment 1 sizing: every pilot-dependent number in RERUN_PREREG.md, computed from pilot 2 (direct answers,
baseline only, 48 prompts x 2). Uses the frozen power simulation's own functions and the analysis's own test.

    python3 evidence/size_from_pilot2.py PILOT2_GENERATIONS.json PILOT2_LABELS.json > evidence/pilot2_sizing.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from power_sim_rerun import beta_ab, power, reduction_pp  # noqa: E402
from rerun_analyze import ENDPOINTS, cluster_boot_ci, signflip_p  # noqa: E402

ORS = (1.0, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7)
FAB = ENDPOINTS["fabrication"]


def icc1(groups):
    groups = [g for g in groups if len(g) >= 2]
    k = sum(map(len, groups)) / len(groups)
    grand = sum(map(sum, groups)) / sum(map(len, groups))
    msb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups) / (len(groups) - 1)
    msw = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups) / (sum(map(len, groups)) - len(groups))
    return float((msb - msw) / (msb + (k - 1) * msw)) if msb + (k - 1) * msw > 0 else float("nan")


def outcome_sim(p0, rho, odds_ratio, bound, reps, rng):
    a, b = beta_ab(p0, rho)
    big = np.random.default_rng(1).beta(a, b, 2_000_000)
    truth = float((big - big * odds_ratio / (1 - big + big * odds_ratio)).mean())
    cover = excluded = 0
    for _ in range(reps):
        q0 = rng.beta(a, b, 48)
        q1 = q0 * odds_ratio / (1 - q0 + q0 * odds_ratio)
        d = rng.binomial(5, q0) / 5 - rng.binomial(5, q1) / 5
        lo, hi = cluster_boot_ci(d, n_boot=2000, rng=rng)
        cover += lo <= truth <= hi
        excluded += signflip_p(d, "greater", n_flips=2000, rng=rng) > 0.05 and hi < bound
    return truth, excluded / reps, cover / reps


def main():
    gens, labels = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    lab = {tuple(r["key"][:3]): r["judge"]["classification"] for r in labels if "classification" in r["judge"]}
    groups = defaultdict(list)
    for g in gens:
        k = (g["category"], g["index"], g["sample"])
        if k in lab:
            groups[(g["category"], g["index"])].append(int(lab[k] in FAB))
    flat = [x for v in groups.values() for x in v]
    p0, rho = float(np.mean(flat)), icc1(list(groups.values()))
    secs = float(np.mean([g["seconds"] for g in gens]))
    think = sum(("<think>" in g["response"]) or ("</think>" in g["response"]) for g in gens)
    scen = [("Pilot 2 values", p0, rho), ("ICC 0.30", p0, 0.30), ("ICC 0.60", p0, 0.60),
            ("p0 %.2f" % max(p0 - 0.07, 0.05), max(p0 - 0.07, 0.05), rho)]
    table = {name: [(power(p, r, o), reduction_pp(p, r, o)) for o in ORS] for name, p, r in scen}
    # 80%-power point on the pilot-2 row: interpolate the mean reduction where power crosses 0.80
    row = table["Pilot 2 values"]
    cross = None
    for (pw_a, pp_a), (pw_b, pp_b) in zip(row[2:], row[3:]):      # ORs 0.3 -> 0.7: power falls as OR rises
        if pw_a >= 0.8 > pw_b:
            cross = pp_a + (pw_a - 0.8) / (pw_a - pw_b) * (pp_b - pp_a)
    if cross is None and row[1][0] >= 0.8 > row[2][0]:
        cross = row[1][1] + (row[1][0] - 0.8) / (row[1][0] - row[2][0]) * (row[2][1] - row[1][1])
    bound = round(cross) if cross is not None else None
    rng = np.random.default_rng(99)
    sims = {}
    if bound:
        target = next((o for o in np.arange(0.10, 1.0, 0.01) if reduction_pp(p0, rho, o) <= bound + 0.6), None)
        _, null_ex, null_cov = outcome_sim(p0, rho, 1.0, bound / 100, 400, rng)
        truth, wrong_ex, wrong_cov = outcome_sim(p0, rho, float(target), bound / 100, 600, rng)
        covs = [null_cov, wrong_cov] + [outcome_sim(p0, rho, o, bound / 100, 300, rng)[2] for o in (0.4, 0.7)]
        sims = {"null_excluded": null_ex, "wrong_at_pp": round(100 * truth, 1), "wrong_excluded": wrong_ex,
                "coverage_min": min(covs), "coverage_max": max(covs)}
    out = {"n_labels": len(flat), "n_fab": int(sum(flat)), "p0": p0, "icc": rho, "prompts": len(groups),
           "never": sum(1 for v in groups.values() if not any(v)), "think_tag_responses": think,
           "mean_seconds": secs, "hours_720": 720 * secs / 3600, "cross_pp": cross, "bound_pp": bound,
           "table": {k: [[round(pw, 2), round(pp, 1)] for pw, pp in v] for k, v in table.items()}, "ors": ORS,
           "sims": sims}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
