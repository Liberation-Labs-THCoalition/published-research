"""Exploratory statistics for the paper's revision (Agni paper review r1, 2026-10-06). Post hoc, never used for a claim
the pre-registered rerun did not make. Reuses the rerun's frozen test functions (base_rerun/rerun_analyze.py).

Primary study (abliterated 27B, greedy, one response per prompt and bias; labels from the blind and the condition-aware
re-judge, supplementary/rejudge_20260930/):
  - label counts per category and bias; per-prompt label strings for the unanswerable and Fermi prompts;
  - fictional prompts: per-prompt difference d = baseline - bias b (fabrication, and FULL_CONFAB alone), an EXACT
    sign-flip test over the nonzero d (one- and two-sided), and a prompt-cluster bootstrap 95% interval;
  - Cohen's kappa between the two re-judges, with a prompt-cluster bootstrap interval.
Pre-registered rerun (parent checkpoint, T = 0.7, base_rerun/results/):
  - generation length and cap hits per condition (a check against RESULTS.md);
  - the primary endpoint with capped generations excluded;
  - a Frisch-Waugh-Lovell estimate: fabrication on the bias-5.0 indicator, both residualised on generation length and
    prompt fixed effects (bias 0 and 5.0 rows only), with a prompt-cluster bootstrap interval;
  - kappa between the two judge passes, with a prompt-cluster bootstrap interval.
Search-only sensitivity (added 2026-10-07, found while porting the paper to LaTeX): eight primary-study responses at
bias 1.0-3.0 are a bare search call ("I'll search for information about ..." plus a <search> tag) and nothing else. The
re-judges label most of them FULL_CONFAB. The fictional tests are repeated with those responses counted as not
fabrication. Own RNG stream (SEED + 1), so every number above is unchanged.

    python3 revision_stats.py OUTDIR
"""
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUPP = HERE.parent
sys.path.insert(0, str(SUPP / "base_rerun"))
import rerun_analyze as ra  # noqa: E402

SEED = 20261006
N_BOOT = 10_000
FAB = {"FULL_CONFAB", "COSMETIC_HEDGE"}
BIASES = (0.0, 1.0, 2.0, 3.0, 5.0)
SHORT = {"FULL_CONFAB": "F", "COSMETIC_HEDGE": "C", "HONEST_HEDGE": "h", "HONEST_REDIRECT": "r", "LEGITIMATE": "L",
         "META_HEDGE": "m"}


def category(idx):
    return "fictional" if idx < 20 else "unanswerable" if idx < 30 else "legitimate"


def load_primary(name):
    out = {}
    for r in json.loads((SUPP / "rejudge_20260930" / name).read_text()):
        lab = (r.get("judge") or {}).get("classification")
        _, idx, bias = r["key"]
        out[(int(idx), float(bias))] = lab
    return out


def exact_signflip(d):
    """Exact sign-flip p over the nonzero entries of d: one-sided (H1 mean > 0) and two-sided."""
    nz = [x for x in d if x != 0]
    obs = float(np.mean(d)) if len(d) else float("nan")
    if not nz:
        return {"one_sided": 1.0, "two_sided": 1.0, "n_nonzero": 0}
    n = len(d)
    ge = ab = 0
    for signs in itertools.product((-1, 1), repeat=len(nz)):
        m = sum(s * x for s, x in zip(signs, nz)) / n
        ge += m >= obs - 1e-12
        ab += abs(m) >= abs(obs) - 1e-12
    tot = 2 ** len(nz)
    return {"one_sided": ge / tot, "two_sided": ab / tot, "n_nonzero": len(nz)}


def boot_ci(d, rng):
    d = np.asarray(d, dtype=float)
    idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
    means = d[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _kappa(pairs):
    """ra.kappa takes two {item: label} dicts; a bootstrap draw can repeat a cluster, so items are keyed by position."""
    return ra.kappa({i: a for i, (a, _) in enumerate(pairs)}, {i: b for i, (_, b) in enumerate(pairs)})[0]


def kappa_ci(pairs_by_cluster, rng):
    """Cohen's kappa over all pairs, with a cluster bootstrap (clusters resampled with replacement)."""
    clusters = list(pairs_by_cluster)
    allp = [p for c in clusters for p in pairs_by_cluster[c]]
    k = _kappa(allp)
    ks = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        pp = [p for i in pick for p in pairs_by_cluster[clusters[i]]]
        ks.append(_kappa(pp))
    ks = np.array([x for x in ks if np.isfinite(x)])
    return {"kappa": float(k), "ci95": [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))],
            "n_pairs": len(allp), "n_clusters": len(clusters), "boot": int(ks.size)}


def primary(rng):
    res = {}
    labs = {"blind": load_primary("powered_blind.json"), "unblinded": load_primary("powered_unblinded.json")}
    for mode, L in labs.items():
        missing = [k for k, v in L.items() if v not in SHORT]
        counts = {}
        for cat in ("fictional", "unanswerable", "legitimate"):
            counts[cat] = {str(b): dict(Counter(L[(i, b)] for i in range(35) if category(i) == cat and (i, b) in L))
                           for b in BIASES}
        strings = {f"P{i:02d}": "".join(SHORT.get(L.get((i, b)), "?") for b in BIASES) for i in range(20, 35)}
        tests = {}
        for endpoint, cls in (("fabrication", FAB), ("full_confab", {"FULL_CONFAB"})):
            for b in BIASES[1:]:
                d = [int(L[(i, 0.0)] in cls) - int(L[(i, b)] in cls) for i in range(20)]
                tests[f"{endpoint}_0_vs_{b}"] = {"rate_0": sum(L[(i, 0.0)] in cls for i in range(20)) / 20,
                                                 "rate_b": sum(L[(i, b)] in cls for i in range(20)) / 20,
                                                 "mean_d": float(np.mean(d)), "ci95": boot_ci(d, rng),
                                                 **exact_signflip(d)}
        res[mode] = {"invalid_labels": missing, "counts": counts, "per_prompt_unanswerable_and_fermi": strings,
                     "fictional_tests": tests}
    pairs = defaultdict(list)
    for (i, b), a in labs["blind"].items():
        pairs[i].append((a, labs["unblinded"][(i, b)]))
    res["kappa_blind_vs_unblinded"] = kappa_ci(pairs, rng)
    return res


def rerun(rng):
    R = SUPP / "base_rerun" / "results"
    gens = json.loads((R / "generations.json").read_text())
    p1, p2 = ra.load_labels(R / "pass1.json"), ra.load_labels(R / "pass2.json")
    cap = ra.CAP
    tok = {(g["category"], int(g["index"]), int(g["sample"]), float(g["bias"])): int(g["n_tokens"]) for g in gens}
    out = {"cap": cap, "generation": {}}
    for b in (0.0, 2.0, 5.0):
        v = [t for k, t in tok.items() if k[3] == b]
        out["generation"][str(b)] = {"n": len(v), "mean_tokens": float(np.mean(v)), "hit_cap": int(sum(t >= cap for t in v))}
    prompts = sorted({(k[0], k[1]) for k in tok})

    def rates(keep):
        r = {}
        for b in (0.0, 5.0):
            for p in prompts:
                ks = [k for k in p1 if (k[0], k[1]) == p and k[3] == b and keep(k)]
                if ks:
                    r[(p, b)] = np.mean([p1[k]["classification"] in FAB for k in ks])
        common = [p for p in prompts if (p, 0.0) in r and (p, 5.0) in r]
        return np.array([r[(p, 0.0)] - r[(p, 5.0)] for p in common]), common

    d_all, c_all = rates(lambda k: True)
    d_unc, c_unc = rates(lambda k: tok[k] < cap)
    out["primary_all"] = {"n_prompts": len(c_all), "mean_d": float(d_all.mean()),
                          "p_one_sided": ra.signflip_p(d_all), "ci95": list(ra.cluster_boot_ci(d_all))}
    out["primary_uncapped_only"] = {"n_prompts": len(c_unc), "n_labels_dropped": int(sum(tok[k] >= cap for k in p1 if k[3] in (0.0, 5.0))),
                                    "mean_d": float(d_unc.mean()), "p_one_sided": ra.signflip_p(d_unc),
                                    "ci95": list(ra.cluster_boot_ci(d_unc))}
    # FWL with prompt fixed effects: residualise y (fabrication), x (bias 5.0) and z (tokens) within prompt, then
    # partial z out of both; the slope of y_res on x_res is the length-adjusted bias effect (negative = reduction).
    rows = [(k, int(p1[k]["classification"] in FAB), int(k[3] == 5.0), tok[k]) for k in p1 if k[3] in (0.0, 5.0)]

    def fwl(sample_prompts):
        by = defaultdict(list)
        for k, y, x, z in rows:
            by[(k[0], k[1])].append((y, x, z))
        Y, X, Z = [], [], []
        for p in sample_prompts:
            arr = np.array(by[p], dtype=float)
            arr = arr - arr.mean(axis=0)
            Y.append(arr[:, 0]); X.append(arr[:, 1]); Z.append(arr[:, 2])
        Y, X, Z = np.concatenate(Y), np.concatenate(X), np.concatenate(Z)
        zz = Z @ Z
        yr = Y - Z * (Z @ Y) / zz
        xr = X - Z * (Z @ X) / zz
        return float((xr @ yr) / (xr @ xr)), float(((X - X.mean()) @ (Y - Y.mean())) / ((X - X.mean()) @ (X - X.mean())))

    est, raw = fwl(prompts)
    boots = []
    for _ in range(N_BOOT):
        pick = [prompts[i] for i in rng.integers(0, len(prompts), size=len(prompts))]
        boots.append(fwl(pick)[0])
    out["fwl_length_adjusted"] = {"slope_fabrication_on_bias5": est, "unadjusted_within_prompt_slope": raw,
                                  "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                  "note": "negative = bias 5.0 lowers fabrication; per-response probability scale"}
    pairs = defaultdict(list)
    for k in p1:
        if k in p2:
            pairs[(k[0], k[1])].append((p1[k]["classification"], p2[k]["classification"]))
    out["kappa_pass1_vs_pass2"] = kappa_ci(pairs, rng)
    return out


def is_search_only(text):
    """A bare search call: a <search...> tag in a short response. All eight are 147-160 characters."""
    return bool(re.search(r"<search", text)) and len(text) < 300


def search_sensitivity(rng):
    res = {}
    for mode, name in (("blind", "powered_blind.json"), ("unblinded", "powered_unblinded.json")):
        raw = json.loads((SUPP / "rejudge_20260930" / name).read_text())
        L = load_primary(name)
        so = {(int(r["key"][1]), float(r["key"][2])) for r in raw if is_search_only(r["response"])}
        listing = [{"prompt": f"P{i:02d}", "bias": b, "label": L[(i, b)]} for i, b in sorted(so)]
        tests = {}
        for endpoint, cls in (("fabrication", FAB), ("full_confab", {"FULL_CONFAB"})):
            for b in BIASES[1:]:
                fab = lambda i, bb: int(L[(i, bb)] in cls and (i, bb) not in so)  # noqa: E731
                d = [fab(i, 0.0) - fab(i, b) for i in range(20)]
                tests[f"{endpoint}_0_vs_{b}"] = {"rate_0": sum(fab(i, 0.0) for i in range(20)) / 20,
                                                 "rate_b": sum(fab(i, b) for i in range(20)) / 20,
                                                 "mean_d": float(np.mean(d)), "ci95": boot_ci(d, rng),
                                                 **exact_signflip(d)}
        res[mode] = {"search_only": listing, "fictional_tests_search_as_not_fabrication": tests}
    return res


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "n_boot": N_BOOT, "primary_study": primary(rng), "rerun": rerun(rng),
           "search_only_sensitivity": search_sensitivity(np.random.default_rng(SEED + 1))}
    (outdir / "revision_stats.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
