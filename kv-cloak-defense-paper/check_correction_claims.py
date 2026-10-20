"""
Gate for the version 1.1 correction of "Cache Geometry Under Obfuscation"
(kv-cloak-defense-paper). Written before the corrected prose, 2026-10-20.

One assertion per numerical or factual claim that version 1.1 makes or keeps
(abstract, results, discussion, conclusion, and the correction note), each
computed from this paper's data files. A claim that fails here is withdrawn
or scoped down in the text; it is not reworded until it passes.

Data (all under data/):
  kv_cloak_replication_100.json, kv_cloak_replication_analysis.json,
  kv_cloak_positive_control.json  -- shipped with version 1.0.
  kv_cloak_v2/, kv_cloak_real/, kv_cloak_cross_arch/, kv_cloak_base/
                                  -- added in version 1.1. They are the run
     files behind Tables 1 and 2, copied from the lab's KV-Experiments
     repository (results/oracle_harness/, commit 8429260, 2026-05-06) and
     byte-identical to the MTH mirror of the same directory. The SHA-256
     values below were measured on MTH; files are hashed with CRLF folded to
     LF, which is the form git stores them in.

Conventions, fixed before the numbers were computed:
  - A printed AUROC "rounds to" the exact value: |printed - exact| <= 0.0005
    for three decimals, <= 0.005 for two.
  - AUROC in the run files is direction-free, max(A, 1 - A); exact values
    here are recomputed from the per-trial features the same way.
  - "Survives correction for five features": p < 0.05 / 5 = 0.01.
  - Selection-adjusted test: the best of the five features' direction-free
    AUROCs, against 10,000 label permutations (seed 2026).

Usage:  python check_correction_claims.py [--plant]
  --plant  restores version 1.0's hedged count (n = 89) as the expected value,
           to show the gate can fail.
Exit status is non-zero if any assertion fails. SKIP lines are checks whose
source is not in this repository; a SKIP is not a pass.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPO = HERE.parent
PLANT = "--plant" in sys.argv
FAILS, SKIPS = [], []


def claim(text, ok, evidence):
    print(f"[{'PASS' if ok else 'FAIL'}] {text}\n       {evidence}")
    if not ok:
        FAILS.append(text)


def skip(text, why):
    print(f"[SKIP] {text}\n       {why}")
    SKIPS.append(text)


def load(rel):
    return json.load(open(DATA / rel, encoding="utf-8"))


def rounds(printed, exact, dp=3):
    return abs(printed - exact) <= 0.5 * 10 ** (-dp) + 1e-12


def folded_auc(pos, neg):
    a = stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))
    return max(a, 1 - a), a


MTH_SHA256 = {
    "kv_cloak_v2/cloak_results_v2.json": "22edea9d74dfd827483d36a1f887a0aaf96082f0c212a792ece949bc257aa2e1",
    "kv_cloak_v2/trials_v2.json": "963ec0e61a4162a1337bc086379432be90e653a62a19a9d9088043ac919d2b3f",
    "kv_cloak_real/results.json": "9a4983ec5a661608e8704456258b002dee5405a7dd472f07f14a644df3f66f36",
    "kv_cloak_real/trials.json": "4b1a5da1d34dcd09e0f4e418b16dad256595892ad79a2b25cc1c915172caa136",
    "kv_cloak_cross_arch/summary.json": "e508c0a5ec17c31654a1f5aee94bfc2ff12d85a6f6361f4dd5236f9b4bc0ab40",
    "kv_cloak_cross_arch/DeepSeek-R1-Distill-Qwen-7B.json": "b6c2b3fdea1486c28f2bf94d2546e6f927c6993c7208a6fac333a96089b89546",
    "kv_cloak_cross_arch/Llama-3.1-8B-Instruct.json": "55d99e2432f01ac0f50f5180a3961d78b097d48ac1f9c3aaa8b7bd4ea7b248f7",
    "kv_cloak_cross_arch/Mistral-7B-Instruct-v0.3.json": "1dcade9890461dd009625c94434fb7caeea52a8b2f1c59fa1c2a2e7951e461bd",
    "kv_cloak_cross_arch/Qwen2.5-14B-Instruct.json": "f5dfddfa728488ba59277af7e50db960bd1df16e299721115176e61ae2310b57",
    "kv_cloak_base/cloak_base_results.json": "0b29c710ff0f9438748ad3754b9adb15ab351a78af2d0405734942d3bd465261",
    "kv_cloak_base/trials_base.json": "8839d78f40f8907a4331205f6b415ff1332302aec60e2b5b6f1e22e4f0d54758",
}
FEATS = ["mp_spectral_gap", "mp_signal_rank", "mp_signal_fraction", "mp_top_sv_excess", "mp_norm_per_token"]


def section(title):
    print(f"\n=== {title} ===")


def main():
    # ------------------------------------------------------------ provenance
    section("Provenance of the added run files")
    bad = []
    for rel, want in MTH_SHA256.items():
        got = hashlib.sha256((DATA / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if got != want:
            bad.append(rel)
    claim("The 11 run files behind Tables 1 and 2 are byte-identical to the MTH mirror (KV-Experiments 8429260)",
          not bad, f"{len(MTH_SHA256) - len(bad)}/{len(MTH_SHA256)} SHA-256 match" + (f"; mismatched: {bad}" if bad else ""))

    v2 = load("kv_cloak_v2/trials_v2.json")
    v2r = load("kv_cloak_v2/cloak_results_v2.json")
    real = load("kv_cloak_real/trials.json")
    realr = load("kv_cloak_real/results.json")
    same_u = sum(a["features"]["none"]["full"] == b["mp_none"] for a, b in zip(v2, real))
    same_r = sum(a["features"]["orthogonal"]["full"] == b["mp_sim_rotation"] for a, b in zip(v2, real))
    same_b = sum(a["behavior"] == b["behavior"] for a, b in zip(v2, real))
    claim("Table 2's uncloaked and rotation columns are the same 100 trials as the v2 run (one set of replies)",
          len(v2) == len(real) == 100 and same_u == same_r == same_b == 100,
          f"identical uncloaked features {same_u}/100, rotation features {same_r}/100, labels {same_b}/100")

    # --------------------------------------------- the replication (per-head)
    section("Replication: per-head rotation (data/kv_cloak_replication_100.json)")
    rep = load("kv_cloak_replication_100.json")
    ana = load("kv_cloak_replication_analysis.json")
    T = rep["trials"]
    meta = rep["rotation_metadata"]
    claim("One set of four Haar-random head rotations (seed 2026) was used for every trial",
          meta["seed"] == 2026 and len(meta["checksums"]) == rep["n_kv_heads"] == 4
          and max(meta["orthogonality_check"]) < 1e-5 and rep["cloak_seed"] == 2026,
          f"seed {meta['seed']}, {len(meta['checksums'])} matrices, max |QQ^T - I| = {max(meta['orthogonality_check']):.1e}")

    FN = ["stable_rank", "spectral_entropy", "top_sv_ratio", "norm", "sv1", "condition_number"]
    L = [f"L{x}" for x in rep["probe_layers"]]
    rel_x = max(abs(t["cloaked_features"][l][f] - t["uncloaked_features"][l][f])
                / max(abs(t["uncloaked_features"][l][f]), 1e-12) for t in T for l in L for f in FN)
    nfeat = len(L) * len(FN)
    claim(f"Rotation left all {nfeat} cross-head SVD features (6 features x 4 layers, heads concatenated on the feature axis) unchanged in all 100 trials",
          nfeat == 24 and len(T) == 100 and rel_x < 1e-6, f"max relative change {rel_x:.1e} (text: below 1e-6)")
    heads = sorted(T[0]["uncloaked_per_head"][L[0]].keys())
    rel_h = max(abs(t["cloaked_per_head"][l][h] - t["uncloaked_per_head"][l][h])
                / max(abs(t["uncloaked_per_head"][l][h]), 1e-12) for t in T for l in L for h in heads)
    claim("Rotation left all 16 per-head stable ranks (4 heads x 4 layers) unchanged in all 100 trials",
          len(heads) * len(L) == 16 and rel_h < 1e-6, f"max relative change {rel_h:.1e} (text: below 1e-6)")

    # The mathematical statement v1.1 makes: per-head and feature-axis spectra cannot move;
    # a layout that stacks the heads some other way can.
    rng = np.random.default_rng(0)
    K = rng.standard_normal((4, 30, 16))
    Q = [stats.ortho_group.rvs(16, random_state=s) for s in range(4)]
    KQ = np.stack([K[h] @ Q[h] for h in range(4)])
    sv = lambda M: np.linalg.svd(M, compute_uv=False)
    d_head = max(np.abs(sv(K[h]) - sv(KQ[h])).max() for h in range(4))
    d_feat = np.abs(sv(np.concatenate(list(K), axis=1)) - sv(np.concatenate(list(KQ), axis=1))).max()
    d_rows = np.abs(sv(np.concatenate(list(K), axis=0)) - sv(np.concatenate(list(KQ), axis=0))).max()
    claim("Math: per-head rotation cannot change per-head or feature-axis-concatenated spectra; a head-stacked (row) layout can change",
          d_head < 1e-9 and d_feat < 1e-9 and d_rows > 1e-3,
          f"synthetic check: per-head {d_head:.1e}, feature-axis {d_feat:.1e}, row-stacked {d_rows:.2f}")

    # Recompute the replication's own classifier (code/kv_cloak_replication.py, run_classifier).
    def X_of(key):  # as build_X in the script: inf and NaN become 0
        X = np.array([[t[key][l].get(f, 0) for l in L for f in FN] for t in T], dtype=float)
        X[~np.isfinite(X)] = 0
        return X
    Xu, Xc = X_of("uncloaked_features"), X_of("cloaked_features")
    y = np.array([t["label"] for t in T])
    g = np.array([t["group"] for t in T])

    def cv(Xtr_all, Xte_all):
        probs = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(Xtr_all, y, g):
            sc = StandardScaler().fit(Xtr_all[tr])
            clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(sc.transform(Xtr_all[tr]), y[tr])
            probs[te] = clf.predict_proba(sc.transform(Xte_all[te]))[:, 1]
        return roc_auc_score(y, probs)

    a_u, a_c, a_t = cv(Xu, Xu), cv(Xc, Xc), cv(Xu, Xc)
    claim("The replication classifier scores AUROC 0.994 with and without the rotation (recomputed; file says 0.9944 / 0.9944, drop 0.0)",
          rounds(0.994, a_u) and rounds(0.994, a_c) and abs(a_u - a_c) < 1e-3
          and ana["uncloaked_auroc"] == ana["cloaked_auroc"] == 0.9944 and ana["auroc_drop"] == 0.0,
          f"recomputed uncloaked {a_u:.4f}, cloaked {a_c:.4f}; file {ana['uncloaked_auroc']} / {ana['cloaked_auroc']}, drop {ana['auroc_drop']}")
    claim("A classifier fitted on uncloaked caches scores the same 0.994 on cloaked caches (transfer)",
          rounds(0.994, a_t) and abs(a_t - a_u) < 1e-3, f"train uncloaked, test cloaked, same folds: {a_t:.4f}")
    nconf = sum(t["is_confab"] for t in T)
    beh = {b: sum(t["behavior"] == b for t in T) for b in ("CONFABULATED", "HEDGED", "HONEST")}
    claim("Its labels are prompt types (50 confabulation-inducing prompts vs 50 factual questions), not behaviour",
          nconf == 50 and len(T) - nconf == 50 and all(t["label"] == int(t["is_confab"]) for t in T)
          and beh == {"CONFABULATED": 35, "HEDGED": 15, "HONEST": 50},
          f"label == is_confab in all 100; behaviour field {beh}")
    claim("The replication's model is the distilled Qwen3.5-27B, probe layers 3, 7, 11, 15",
          rep["model"] == "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled" and rep["probe_layers"] == [3, 7, 11, 15],
          f"{rep['model']}, layers {rep['probe_layers']}")

    # -------------------------------------------- detection, uncloaked (v2)
    section("Detection without the cloak (distilled model; v2 / real run files)")
    beh2 = {b: sum(t["behavior"] == b for t in v2) for b in ("CONFABULATED", "HEDGED", "AMBIGUOUS")}
    want_h = 89 if PLANT else 87
    claim(f"n = 7 confabulations and {want_h} hedged replies from 100 prompts (6 ambiguous excluded)",
          beh2 == {"CONFABULATED": 7, "HEDGED": want_h, "AMBIGUOUS": 6}
          and v2r["trial_counts"] == {"confab": 7, "hedged": 87, "ambiguous": 6},
          f"data: {beh2}" + ("   <-- PLANTED false value (version 1.0's n = 89)" if PLANT else ""))
    C = [t for t in real if t["behavior"] == "CONFABULATED"]
    H = [t for t in real if t["behavior"] == "HEDGED"]
    ex = {}
    for s in ("mp_none", "mp_sim_rotation", "mp_real_kvcloak"):
        for f in FEATS:
            ex[s, f] = folded_auc([t[s][f] for t in C], [t[s][f] for t in H])
    R = realr["results"]
    sg = R["none"]["mp_spectral_gap"]
    claim("Spectral gap (sigma1/sigma2): AUROC 0.903, 95% bootstrap CI [0.810, 0.980]; FWL 0.877; R^2 = 0.068 vs token count",
          rounds(0.903, ex["mp_none", "mp_spectral_gap"][0]) and rounds(0.810, sg["ci"][0]) and rounds(0.980, sg["ci"][1])
          and rounds(0.877, sg["fwl_auroc"]) and rounds(0.068, sg["r_squared"]),
          f"exact {ex['mp_none', 'mp_spectral_gap'][0]:.6f}; CI {sg['ci']}; FWL {sg['fwl_auroc']}; R2 {sg['r_squared']}")
    claim("Version 1.0's CI [0.806, 0.977] is not the run file's interval",
          not (rounds(0.806, sg["ci"][0]) and rounds(0.977, sg["ci"][1])), f"file CI {sg['ci']}")
    sr = R["none"]["mp_signal_rank"]
    claim("Signal rank: AUROC 0.755 [0.578, 0.918] (version 1.0 printed 0.754 in Table 2 and [0.570, 0.914] in the text)",
          rounds(0.755, ex["mp_none", "mp_signal_rank"][0]) and rounds(0.578, sr["ci"][0]) and rounds(0.918, sr["ci"][1])
          and not rounds(0.754, ex["mp_none", "mp_signal_rank"][0]),
          f"exact {ex['mp_none', 'mp_signal_rank'][0]:.6f}; CI {sr['ci']}")
    others = {f: (R["none"][f]["auroc"], R["none"][f]["p"]) for f in ("mp_signal_fraction", "mp_top_sv_excess", "mp_norm_per_token")}
    claim("The other three features: AUROC <= 0.714, none significant (p >= 0.06)",
          all(a <= 0.7145 and p >= 0.06 for a, p in others.values()), f"{others}")
    claim("Bootstrap confidence intervals use 2000 iterations; features are residualized against output token count (FWL)",
          v2r["config"]["bootstrap"].startswith("2000 iterations") and "token count" in v2r["config"]["fwl"],
          f"config: {v2r['config']['bootstrap']!r}; {v2r['config']['fwl']!r}")
    claim("AUROCs are direction-free, max(A, 1 - A) (e.g. rotation top-SV excess: raw 0.457, reported 0.544)",
          abs(ex["mp_sim_rotation", "mp_top_sv_excess"][1] - 0.4565) < 5e-4
          and abs(R["sim_rotation"]["mp_top_sv_excess"]["auroc"] - 0.5435) < 1e-9,
          f"raw {ex['mp_sim_rotation', 'mp_top_sv_excess'][1]:.4f}, file {R['sim_rotation']['mp_top_sv_excess']['auroc']}")
    claim("Only the spectral gap survives correction for five features without the cloak (p < 0.01)",
          [f for f in FEATS if R["none"][f]["p"] < 0.01] == ["mp_spectral_gap"],
          f"p: {[(f[3:], round(R['none'][f]['p'], 5)) for f in FEATS]}")

    # selection-adjusted: best of five, label permutation
    TT = C + H
    yy = np.array([True] * len(C) + [False] * len(H))

    def best_of_five(X, lab):
        n1, n0 = lab.sum(), len(lab) - lab.sum()
        out = []
        for j in range(X.shape[1]):
            r = rankdata(X[:, j])
            a = (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            out.append(max(a, 1 - a))
        return max(out)

    prng = np.random.default_rng(2026)
    perms = [prng.permutation(yy) for _ in range(10000)]
    sel = {}
    for s in ("mp_none", "mp_sim_rotation", "mp_real_kvcloak"):
        X = np.array([[t[s][f] for f in FEATS] for t in TT])
        obs = best_of_five(X, yy)
        null = np.array([best_of_five(X, p) for p in perms])
        sel[s] = (obs, (1 + (null >= obs).sum()) / (1 + len(perms)), np.quantile(null, 0.95), np.median(null))
    claim("Selection-adjusted (best of five features vs label permutation): uncloaked p < 0.001",
          sel["mp_none"][1] < 0.001, f"best {sel['mp_none'][0]:.3f}, p {sel['mp_none'][1]:.4f}")

    # base model and cross-architecture
    base = load("kv_cloak_base/cloak_base_results.json")
    bsf = base["results"]["none"]["full"]["mp_signal_fraction"]
    claim("Base model (Qwen2.5-7B-Instruct): best feature signal fraction 0.661 (FWL 0.687, R^2 = 0.602), n = 48 confabulations, 52 hedged",
          base["config"]["model_id"] == "Qwen/Qwen2.5-7B-Instruct" and rounds(0.661, bsf["auroc"]) and rounds(0.687, bsf["fwl_auroc"])
          and rounds(0.602, bsf["r_squared_vs_tokens"]) and base["trial_counts"] == {"confab": 48, "hedged": 52, "ambiguous": 0}
          and max(v["auroc"] for v in base["results"]["none"]["full"].values()) == bsf["auroc"],
          f"{bsf['auroc']} / FWL {bsf['fwl_auroc']} / R2 {bsf['r_squared_vs_tokens']}; {base['trial_counts']}")
    xa = load("kv_cloak_cross_arch/summary.json")
    det = {m.split("/")[1]: v["detection"] for m, v in xa.items()}
    pct = {m.split("/")[1]: (v["behavior_counts"].get("CONFABULATED", 0), v["n_prompts"]) for m, v in xa.items()}
    un = {m: d["none"]["auroc"] for m, d in det.items() if d["none"]["auroc"] is not None}
    claim("Cross-architecture (50 prompts each): uncloaked AUROC 0.590-0.784 on three models; DeepSeek-R1-7B had 1 confabulation (no AUROC)",
          all(n == 50 for _, n in pct.values()) and rounds(0.590, min(un.values())) and rounds(0.784, max(un.values()))
          and len(un) == 3 and det["DeepSeek-R1-Distill-Qwen-7B"]["none"]["auroc"] is None and pct["DeepSeek-R1-Distill-Qwen-7B"][0] == 1,
          f"{un}; confabulations/prompts {pct}")
    claim("Confabulation rates in Table 1: 12%, 26%, 14%, 2% (Qwen-14B, Mistral, Llama, DeepSeek); 7% and 48% (distilled, base)",
          [100 * c // n for c, n in (pct["Qwen2.5-14B-Instruct"], pct["Mistral-7B-Instruct-v0.3"], pct["Llama-3.1-8B-Instruct"], pct["DeepSeek-R1-Distill-Qwen-7B"])] == [12, 26, 14, 2]
          and beh2["CONFABULATED"] == 7 and base["trial_counts"]["confab"] == 48,
          f"{pct}")
    ps = {m: d["none"]["p"] for m, d in det.items() if d["none"].get("p") is not None}
    claim("No cross-architecture model survives Bonferroni correction for the three comparisons with an AUROC (p < 0.0167)",
          len(ps) == 3 and all(p >= 0.05 / 3 for p in ps.values()), f"{ {m: round(p, 4) for m, p in ps.items()} }")
    claim("Six models in all (the distilled 27B, the 7B base, and four cross-architecture models), 50-100 prompts each",
          len(xa) == 4 and len(v2) == 100 and len(load("kv_cloak_base/trials_base.json")) == 100, f"cross-arch {list(det)}")

    # ------------------------------------------------- under the cloak
    section("Under the cloak")
    hu = np.array([t["features"]["none"]["full"]["mp_spectral_gap"] for t in v2 if t["behavior"] == "HEDGED"])
    hc = np.array([t["features"]["orthogonal"]["full"]["mp_spectral_gap"] for t in v2 if t["behavior"] == "HEDGED"])
    w = stats.wilcoxon(hu, hc)
    claim("Rotation shifts the hedged replies' spectral gap from 24.7 +- 3.7 to 10.8 +- 1.3 (Wilcoxon W = 0, p = 5.5e-16, n = 87); every reply moved down",
          len(hu) == 87 and f"{hu.mean():.1f} {hu.std(ddof=1):.1f} {hc.mean():.1f} {hc.std(ddof=1):.1f}" == "24.7 3.7 10.8 1.3"
          and w.statistic == 0 and f"{w.pvalue:.1e}" == "5.5e-16" and (hc < hu).all(),
          f"n {len(hu)}; {hu.mean():.2f}+-{hu.std(ddof=1):.2f} -> {hc.mean():.2f}+-{hc.std(ddof=1):.2f}; W {w.statistic}, p {w.pvalue:.2e}")
    alt = {}
    for name, keep in (("hedged + ambiguous (93)", lambda t: t["behavior"] != "CONFABULATED"), ("all 100", lambda t: True)):
        a = np.array([t["features"]["none"]["full"]["mp_spectral_gap"] for t in v2 if keep(t)])
        b = np.array([t["features"]["orthogonal"]["full"]["mp_spectral_gap"] for t in v2 if keep(t)])
        alt[name] = f"{stats.wilcoxon(a, b).pvalue:.1e}"
    claim("The Wilcoxon p = 5.5e-16 reproduces only for the 87 hedged replies (no group of 89 exists)",
          all(v != "5.5e-16" for v in alt.values()) and 89 not in (beh2["HEDGED"], beh2["HEDGED"] + beh2["AMBIGUOUS"], len(v2)),
          f"other groupings: {alt}")

    uc = np.array([t["mp_none"]["mp_spectral_gap"] for t in C]); uh = np.array([t["mp_none"]["mp_spectral_gap"] for t in H])
    best = max(((np.mean(uc >= th) - np.mean(uh >= th), th) for th in np.unique(np.r_[uc, uh])))
    th = best[1]
    flags = {}
    for s in ("mp_none", "mp_sim_rotation", "mp_real_kvcloak"):
        flags[s] = (int(sum(t[s]["mp_spectral_gap"] >= th for t in C)), int(sum(t[s]["mp_spectral_gap"] >= th for t in H)))
    claim("A spectral-gap threshold fitted without the cloak (flags 7/7 confabulations, 21/87 hedged) flags 0 of 94 replies under rotation and 94 of 94 under the full mechanism",
          flags["mp_none"] == (7, 21) and flags["mp_sim_rotation"] == (0, 0) and flags["mp_real_kvcloak"] == (7, 87),
          f"Youden threshold {th:.2f}; flagged (confab, hedged): {flags}")
    claim("Under rotation the spectral gap still ranks confabulations above hedged replies, in the same direction, at 0.787",
          ex["mp_sim_rotation", "mp_spectral_gap"][1] > 0.5 and ex["mp_none", "mp_spectral_gap"][1] > 0.5
          and rounds(0.787, ex["mp_sim_rotation", "mp_spectral_gap"][0]),
          f"raw AUROC uncloaked {ex['mp_none', 'mp_spectral_gap'][1]:.4f}, rotation {ex['mp_sim_rotation', 'mp_spectral_gap'][1]:.4f}")

    # Table 2, every cell
    printed = {  # feature: (uncloaked, rotation, real), CIs as printed (two decimals)
        "mp_spectral_gap": ((0.903, .81, .98), (0.787, .58, .95), (0.614, .51, .81)),
        "mp_signal_rank": ((0.755, .58, .92), (0.770, .62, .90), (0.766, .57, .94)),
        "mp_signal_fraction": ((0.714, .52, .90), (0.683, .51, .88), (0.681, .51, .88)),
        "mp_top_sv_excess": ((0.609, .51, .79), (0.544, .50, .76), (0.736, .53, .92)),
        "mp_norm_per_token": ((0.598, .51, .77), (0.598, .51, .77), (0.673, .52, .85)),
    }
    cells_bad = []
    for f, row in printed.items():
        for (a, lo, hi), s, k in zip(row, ("mp_none", "mp_sim_rotation", "mp_real_kvcloak"), ("none", "sim_rotation", "real_kvcloak")):
            ci = R[k][f]["ci"]
            if not (rounds(a, ex[s, f][0]) and rounds(lo, ci[0], 2) and rounds(hi, ci[1], 2)):
                cells_bad.append((f, k, a, round(ex[s, f][0], 6), ci))
    claim("Table 2: all 15 AUROCs and 30 CI bounds round correctly from the run files (0.754 -> 0.755, 0.786 -> 0.787, 0.543 -> 0.544 corrected)",
          not cells_bad, f"bad cells: {cells_bad}" if cells_bad else "15/15 AUROCs, 30/30 bounds")
    rk = {f: (round(R["real_kvcloak"][f]["auroc"], 4), round(R["real_kvcloak"][f]["p"], 3)) for f in FEATS}
    claim("Full mechanism: spectral gap 0.614 (p = 0.33); signal rank 0.766 (p = 0.020) and top-SV excess 0.736 (p = 0.038), no lower than uncloaked (0.755, 0.609)",
          rounds(0.33, R["real_kvcloak"]["mp_spectral_gap"]["p"], 2) and rounds(0.020, R["real_kvcloak"]["mp_signal_rank"]["p"])
          and rounds(0.038, R["real_kvcloak"]["mp_top_sv_excess"]["p"])
          and ex["mp_real_kvcloak", "mp_signal_rank"][0] >= ex["mp_none", "mp_signal_rank"][0]
          and ex["mp_real_kvcloak", "mp_top_sv_excess"][0] >= ex["mp_none", "mp_top_sv_excess"][0],
          f"{rk}")
    claim("Under the full mechanism no feature survives correction for five features (all p >= 0.01)",
          all(R["real_kvcloak"][f]["p"] >= 0.01 for f in FEATS), f"min p {min(R['real_kvcloak'][f]['p'] for f in FEATS):.3f}")
    # Draft wording "p = 0.06 (not distinguishable from chance)" FAILED here on first run
    # (p = 0.0545): reworded to the measured value, without the chance/not-chance dichotomy.
    claim("Selection-adjusted: rotation best 0.787, p = 0.042; full mechanism best 0.766, p = 0.054",
          rounds(0.042, sel["mp_sim_rotation"][1]) and rounds(0.054, sel["mp_real_kvcloak"][1])
          and 0.05 < sel["mp_real_kvcloak"][1] < 0.06,
          f"rotation {sel['mp_sim_rotation'][0]:.3f} p {sel['mp_sim_rotation'][1]:.4f}; full {sel['mp_real_kvcloak'][0]:.3f} p {sel['mp_real_kvcloak'][1]:.4f}")
    claim("With seven confabulations, the best of five features reaches 0.77 by chance in 5% of label shuffles (median about 0.64)",
          rounds(0.77, sel["mp_real_kvcloak"][2], 2) and rounds(0.64, sel["mp_real_kvcloak"][3], 2),
          f"null 95th percentile {sel['mp_real_kvcloak'][2]:.3f}, median {sel['mp_real_kvcloak'][3]:.3f}")
    claim("The full mechanism moves sigma1: the spectral gap of every reply rises, from a hedged mean of 24.7 to 84.6 or more",
          all(t["mp_real_kvcloak"]["mp_spectral_gap"] > t["mp_none"]["mp_spectral_gap"] for t in C + H)
          and round(min(t["mp_real_kvcloak"]["mp_spectral_gap"] for t in C + H), 1) == 84.6
          and round(float(np.mean([t["mp_none"]["mp_spectral_gap"] for t in H])), 1) == 24.7,
          f"min under the mechanism {min(t['mp_real_kvcloak']['mp_spectral_gap'] for t in C + H):.1f}")

    # within-cloak, cross-architecture (Table 1 'Det.(clk)')
    clk = {m: d["orthogonal"]["auroc"] for m, d in det.items() if d["orthogonal"]["auroc"] is not None}
    bclk = base["results"]["orthogonal"]["full"]["mp_signal_fraction"]["auroc"]
    pairs = {"distilled": (ex["mp_none", "mp_spectral_gap"][0], ex["mp_sim_rotation", "mp_spectral_gap"][0]),
             "Qwen2.5-7B": (bsf["auroc"], bclk)}
    pairs.update({m: (un[m], clk[m]) for m in clk})
    fell = [m for m, (a, b) in pairs.items() if b < a]; rose = [m for m, (a, b) in pairs.items() if b > a]
    claim("Within the rotation, detection was lower on four models and higher on one (Qwen2.5-14B, 0.784 -> 0.951, p < 0.001)",
          len(fell) == 4 and rose == ["Qwen2.5-14B-Instruct"] and rounds(0.951, clk["Qwen2.5-14B-Instruct"])
          and det["Qwen2.5-14B-Instruct"]["orthogonal"]["p"] < 0.001,
          f"{ {m: (round(a, 3), round(b, 3)) for m, (a, b) in pairs.items()} }")
    claim("Table 1 within-rotation values: 0.787, 0.951, 0.516, 0.559, 0.639",
          rounds(0.787, pairs["distilled"][1]) and rounds(0.951, clk["Qwen2.5-14B-Instruct"]) and rounds(0.516, clk["Mistral-7B-Instruct-v0.3"])
          and rounds(0.559, clk["Llama-3.1-8B-Instruct"]) and rounds(0.639, bclk), f"{clk}, base {bclk}")
    others_v2 = {s: max(v2r["results"][s]["full"][f]["auroc"] for f in FEATS) for s in v2r["results"] if s not in ("none", "orthogonal")}
    claim("Under each other transform run on the distilled model (random invertible, diagonal scaling, four noise levels), the best feature stayed at or above 0.765",
          set(others_v2) == {"invertible", "diagonal", "noise_0.01", "noise_0.05", "noise_0.1", "noise_0.5"} and min(others_v2.values()) >= 0.765,
          f"{others_v2}")
    claim("Global (shared) rotation was run only on the base model",
          "global_orthogonal" in base["results"] and "global_orthogonal" not in v2r["results"], f"v2 strategies {list(v2r['results'])}")

    # ------------------------------------------- injection and magnitude
    section("Injection and random-direction perturbation (cross-architecture file)")
    inj = {m.split("/")[1]: v["injection"] for m, v in xa.items()}
    direct = {m: v["direct"]["corrected"] for m, v in inj.items()}
    claim("Direct steering (the positive control) corrected 0-1 of 10 per model",
          all(v["direct"]["total"] == 10 for v in inj.values()) and set(direct.values()) <= {0, 1}, f"{direct}")
    mag = {m: v["magnitude_attack"]["adverse"] for m, v in inj.items()}
    k, n = sum(mag.values()), 10 * len(mag)
    up_pool = stats.beta.ppf(0.95, k + 1, n - k)
    up_one = stats.beta.ppf(0.95, 2 + 1, 10 - 2)
    claim("Random-direction perturbation: 0-2 adverse events per model in 10 trials; 4 of 40 over four models (pooled bound computed, not claimed)",
          set(mag.values()) <= {0, 1, 2} and max(mag.values()) == 2 and (k, n) == (4, 40) and rounds(0.21, up_pool, 2),
          f"{mag}; pooled upper bound {up_pool:.3f}")
    claim("Table 1 magnitude-attack cells: Qwen2.5-14B 1/10, Mistral 2/10, Llama 1/10, DeepSeek 0/10",
          mag == {"Qwen2.5-14B-Instruct": 1, "Mistral-7B-Instruct-v0.3": 2, "Llama-3.1-8B-Instruct": 1, "DeepSeek-R1-Distill-Qwen-7B": 0}
          and all(v["magnitude_attack"]["total"] == 10 for v in inj.values()), f"{mag}")
    claim("Version 1.0's 'inconsistent with rates above 30% (power 0.97)' fails per model: 2 of 10 allows rates up to 51%; 0.97 is 1 - 0.7^10, the chance of at least one event",
          up_one > 0.30 and rounds(0.97, 1 - 0.7 ** 10, 2), f"one-sided upper bound at 2/10: {up_one:.3f}; 1 - 0.7^10 = {1 - 0.7 ** 10:.3f}")
    mis = inj["Mistral-7B-Instruct-v0.3"]
    claim("On Mistral, the random direction gave the same counts as the steering vector (1 corrected, 2 adverse, 7 unchanged)",
          mis["magnitude_attack"] == mis["direct"] == {"corrected": 1, "adverse": 2, "same": 7, "total": 10}, f"{mis}")
    pc = load("kv_cloak_positive_control.json")
    labelled = any(k in json.dumps(pc["trials"][0]) for k in ('"behavior"', "CONFABULATED", "HEDGED"))
    claim("The shipped positive-control file has 30 calibration pairs and 30 prompts and no behaviour labels; no file records the '3 calibration pairs'",
          pc["n_calibration_pairs"] == 30 and pc["n_prompts"] == 30 and not labelled
          and not any("calibration" in json.dumps(v) for v in xa.values()),
          f"n_calibration_pairs {pc['n_calibration_pairs']}, n_prompts {pc['n_prompts']}, labels present: {labelled}")

    # ----------------------------------------------- what cannot be checked
    section("What version 1.0 printed that cannot be recomputed")
    prev = max(len(t["response_preview"]) for t in v2)
    textkeys = [k for k in json.dumps(v2r).split('"') if "word" in k.lower() or "text_" in k.lower()]
    claim("The text-only baseline cannot be recomputed: trials store only the first 150 characters of each reply, and no run file holds the comparison",
          prev == 150 and not textkeys, f"max response_preview length {prev}; text-baseline keys in results: {textkeys}")
    lines = len((HERE / "code" / "kv_cloak_replication.py").read_bytes().replace(b"\r\n", b"\n").decode("utf-8").splitlines())
    claim("The replication script is 1,038 lines", lines == 1038, f"{lines} lines")

    # ---------------------------------------------- citations in Background
    section("Background citations (checked against the cited papers in this repository)")
    ol = (REPO / "oracle-loop-paper" / "sections" / "background.tex").read_text(encoding="utf-8")
    claim("Oracle Loop paper reports confabulation detection at 0.969-0.999 and r > 0.999 between RTX 3090 and H200",
          "0.969--0.999" in ol and "r > 0.999" in ol and "H200" in ol, "oracle-loop-paper/sections/background.tex")
    lt = (REPO / "lyra-technique-ii" / "main.tex").read_text(encoding="utf-8")
    claim("Lyra Technique II withdraws the 1.000 deception result; same-prompt control 0.160",
          "retracted" in lt and "0.160" in lt, "lyra-technique-ii/main.tex")
    fb = json.load(open(REPO / "formulary-paper" / "data" / "formulary_base_summary.json", encoding="utf-8"))
    hv = fb["confab_per_vector"]["hostile"]
    mcn = stats.binomtest(hv["adverse_hedged"], hv["corr_strict"] + hv["adverse_hedged"], 0.5).pvalue
    claim("Formulary: the hostile vector corrected 65 of 68 confabulations (96%) on Qwen2.5-7B-Instruct, McNemar p < 0.0001",
          fb["config"]["model_id"] == "Qwen/Qwen2.5-7B-Instruct" and (hv["corr_strict"], hv["n_confab"]) == (65, 68)
          and round(100 * 65 / 68) == 96 and mcn < 1e-4, f"{hv['corr_strict']}/{hv['n_confab']}; exact McNemar p {mcn:.1e}")


    # ------------------------------------------- round 2: Agni paper mode on the first draft
    section("Round 2 (claims added after Agni paper mode, REJECTED, on 8b3781f)")
    lf = (REPO / "spectral-shape-paper" / "code" / "lyra_features.py").read_text(encoding="utf-8")
    names_ok = all(f in lf for f in FEATS)
    claim("The lab's extractor (lyra_features.py) stacks heads, keys and values along the token axis; the v2 feature names are its names",
          "torch.cat([k.reshape(-1, k.shape[-1])," in lf and "v.reshape(-1, v.shape[-1])], dim=0)" in lf and names_ok
          and set(v2[0]["features"]["none"]["full"]) == set(FEATS),
          "get_kv_matrix: cat([k.reshape(-1, head_dim), v.reshape(-1, head_dim)], dim=0); feature names match: %s" % names_ok)
    nrel = max(abs(t["features"]["orthogonal"]["full"]["mp_norm_per_token"] - t["features"]["none"]["full"]["mp_norm_per_token"])
               / abs(t["features"]["none"]["full"]["mp_norm_per_token"]) for t in v2)
    claim("Under the v2 rotation, norm per token (invariant to any orthogonal transform) did not move",
          nrel < 1e-4, f"max relative change {nrel:.1e}")
    claim("On the base model, a single shared rotation left the best feature at 0.661 (identical to uncloaked)",
          base["results"]["global_orthogonal"]["full"]["mp_signal_fraction"]["auroc"] == bsf["auroc"] and rounds(0.661, bsf["auroc"]),
          f"global {base['results']['global_orthogonal']['full']['mp_signal_fraction']['auroc']}, uncloaked {bsf['auroc']}")

    src = (HERE / "code" / "kv_cloak_replication.py").read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    body = src[src.index("def run_experiment"):]
    i_enc = body.index("enc_out = model(input_ids, use_cache=True)")
    i_ext = body.index("extract_cache_features(")
    i_gen = body.index("# --- Generate response")
    claim("The replication takes features after the prompt is encoded, before any token is generated",
          i_enc < i_ext < i_gen, f"offsets in run_experiment: encode {i_enc} < extract {i_ext} < generate {i_gen}")
    dom = collections.Counter(t["domain"] for t in T if t["is_confab"])
    claim("Its 50 confabulation-inducing prompts: 30 fabricated entities, 10 unanswerable, 10 future events; one prompt per group (five-fold CV is ordinary)",
          dom == {"fabricated": 30, "impossible": 10, "future": 10} and len({t["group"] for t in T}) == 100, f"{dict(dom)}")
    pl = np.array([t["prompt_len"] for t in T])
    a_pl = folded_auc(pl[y == 1], pl[y == 0])[0]
    claim("Prompt length alone separates the two prompt types at AUROC 0.968", rounds(0.968, a_pl), f"{a_pl:.4f}")

    dom2 = collections.Counter(t["domain"] for t in v2)
    cdom = collections.Counter(t["domain"] for t in v2 if t["behavior"] == "CONFABULATED")
    claim("Distilled prompts: 50 fabricated, 30 unanswerable, 20 future; the 7 confabulations: fabricated 4, unanswerable 3, future 0",
          dom2 == {"fabricated": 50, "impossible": 30, "future": 20} and cdom == {"fabricated": 4, "impossible": 3}, f"{dict(dom2)}; {dict(cdom)}")
    capc = sum(t["n_tokens"] == 400 for t in v2 if t["behavior"] == "CONFABULATED")
    caph = sum(t["n_tokens"] == 400 for t in v2 if t["behavior"] == "HEDGED")
    claim("Six of the seven confabulations, and 49 of the 87 hedged replies, stopped at the 400-token cap",
          (capc, caph) == (6, 49) and max(t["n_tokens"] for t in v2) == 400, f"{capc}/7, {caph}/87")
    gsg = v2r["results"]["none"]["gen"]["mp_spectral_gap"]["auroc"]
    claim("In the last 80 tokens alone, the spectral gap scores 0.591", rounds(0.591, gsg), f"{gsg}")

    ntok = np.array([t["n_tokens"] for t in TT], dtype=float)
    def resid(x):
        X = np.c_[np.ones_like(ntok), ntok]
        return x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    fw_bad = []
    for s_, k_ in (("mp_none", "none"), ("mp_sim_rotation", "sim_rotation"), ("mp_real_kvcloak", "real_kvcloak")):
        for f in FEATS:
            r = rankdata(resid(np.array([t[s_][f] for t in TT])))
            A = (r[yy].sum() - 7 * 8 / 2) / (7 * 87)
            if abs(max(A, 1 - A) - R[k_][f]["fwl_auroc"]) > 1e-4:
                fw_bad.append((s_, f))
    claim("FWL (OLS on output token count over the 94 replies) reproduces all 15 residualized AUROCs in the run file",
          not fw_bad, f"mismatches: {fw_bad}" if fw_bad else "15/15 within 1e-4")
    so = R["sim_rotation"]["mp_spectral_gap"]
    claim("Rotation, same feature: spectral gap 0.787 (p = 0.010); 0.765 after removing length (p = 0.018)",
          rounds(0.010, so["p"]) and rounds(0.765, so["fwl_auroc"]) and rounds(0.018, so["fwl_p"]),
          f"p {so['p']:.4f}; FWL {so['fwl_auroc']} p {so['fwl_p']:.4f}")
    bo, bn = base["results"]["orthogonal"]["full"]["mp_signal_fraction"], base["results"]["none"]["full"]["mp_signal_fraction"]
    claim("Base model, same feature: 0.661 (p = 0.006; FWL 0.687, p = 0.001) -> 0.639 under rotation (p = 0.017; FWL 0.670, p = 0.003)",
          rounds(0.006, bn["p"]) and rounds(0.001, bn["fwl_p"]) and rounds(0.639, bo["auroc"]) and rounds(0.017, bo["p"])
          and rounds(0.670, bo["fwl_auroc"]) and rounds(0.003, bo["fwl_p"]) and bn["p"] < 0.01,
          f"none p {bn['p']:.4f} fwl_p {bn['fwl_p']:.4f}; orth {bo['auroc']} p {bo['p']:.4f} fwl {bo['fwl_auroc']} fwl_p {bo['fwl_p']:.4f}")
    rr = R["real_kvcloak"]; rn = R["none"]
    claim("Full mechanism after removing length: spectral gap 0.593; signal rank 0.705; top-SV excess 0.695; signal rank R^2 vs length 0.77 with and without",
          rounds(0.593, rr["mp_spectral_gap"]["fwl_auroc"]) and rounds(0.705, rr["mp_signal_rank"]["fwl_auroc"])
          and rounds(0.695, rr["mp_top_sv_excess"]["fwl_auroc"]) and rounds(0.77, rn["mp_signal_rank"]["r_squared"], 2)
          and rounds(0.77, rr["mp_signal_rank"]["r_squared"], 2)
          and max(rr[f]["fwl_auroc"] for f in FEATS) == rr["mp_signal_rank"]["fwl_auroc"],
          f"sg {rr['mp_spectral_gap']['fwl_auroc']}, sr {rr['mp_signal_rank']['fwl_auroc']}, tsv {rr['mp_top_sv_excess']['fwl_auroc']}; R2 {rn['mp_signal_rank']['r_squared']} / {rr['mp_signal_rank']['r_squared']}")
    Xr = np.array([[resid(np.array([t["mp_real_kvcloak"][f] for t in TT]))[i] for f in FEATS] for i in range(len(TT))])
    obs_r = best_of_five(Xr, yy)
    p_r = (1 + sum(best_of_five(Xr, pp) >= obs_r for pp in perms)) / (1 + len(perms))
    claim("Under the full mechanism, after removing length, no single feature detects significantly (every residualized p > 0.05)",
          min(rr[f]["fwl_p"] for f in FEATS) > 0.05, f"min residualized p {min(rr[f]['fwl_p'] for f in FEATS):.3f}")
    claim("Selection-adjusted, length removed, full mechanism: best 0.705, p = 0.19", rounds(0.705, obs_r) and rounds(0.19, p_r, 2),
          f"best {obs_r:.3f}, p {p_r:.4f}")
    X10 = np.array([[t["features"]["none"][w][f] for w in ("full", "gen") for f in FEATS] for t in
                    [x for x in v2 if x["behavior"] == "CONFABULATED"] + [x for x in v2 if x["behavior"] == "HEDGED"]])
    obs10 = best_of_five(X10, yy)
    p10 = (1 + sum(best_of_five(X10, pp) >= obs10 for pp in perms)) / (1 + len(perms))
    claim("Uncloaked, best of the ten features in both windows: 0.903, selection-adjusted p = 0.001",
          rounds(0.903, obs10) and rounds(0.001, p10), f"best {obs10:.4f}, p {p10:.4f}")
    lows = [r["ci"][0] for v in R.values() for r in v.values()] + \
           [r["ci"][0] for v in v2r["results"].values() for w in v.values() for r in w.values()]
    claim("Every bootstrap lower bound in the run files is at or above 0.50 (resamples are folded); the method is labelled 'BCa percentile'",
          min(lows) >= 0.5 and "BCa percentile" in v2r["config"]["bootstrap"], f"min lower bound {min(lows)} over {len(lows)} intervals")
    claim("Under diagonal scaling the spectral gap fell to 0.517; 0.765 is about the 95th percentile of best-of-five by chance (0.768)",
          rounds(0.517, v2r["results"]["diagonal"]["full"]["mp_spectral_gap"]["auroc"]) and abs(sel["mp_real_kvcloak"][2] - 0.765) < 0.01,
          f"diagonal sg {v2r['results']['diagonal']['full']['mp_spectral_gap']['auroc']}; null 95th {sel['mp_real_kvcloak'][2]:.3f}")
    drops = sorted(a - b for a, b in pairs.values() if b < a)
    claim("The four within-rotation drops range from 0.02 to 0.19", len(drops) == 4 and rounds(0.02, drops[0], 2) and rounds(0.19, drops[-1], 2),
          f"{[round(d, 3) for d in drops]}")
    olb = (REPO / "oracle-loop-paper" / "sections" / "background.tex").read_text(encoding="utf-8")
    olr = (REPO / "oracle-loop-paper" / "sections" / "results.tex").read_text(encoding="utf-8")
    old = (REPO / "oracle-loop-paper" / "sections" / "discussion.tex").read_text(encoding="utf-8")
    claim("Oracle Loop paper: 0.969-0.999 is from earlier campaigns; its own figure is 0.707 (0.620 with FWL); M is fused into the weights and detection under it is unverified",
          "three experimental campaigns" in olb and "0.969--0.999" in olb and "0.707" in olr and "0.620" in olr
          and "fused into the model weights" in old and "not empirically verified" in " ".join(old.split()),
          "oracle-loop-paper/sections/{background,results,discussion}.tex")


    # ------------------------------------------- round 3: Agni paper/style r2 on fea5480
    section("Round 3 (claims added after Agni paper and style mode r2, both CONDITIONAL, on fea5480)")
    claim("Spectral gap and norm per token are raw in the lab's extractor (S0/S1 and S2_sum/m); three features use the MP edge",
          "'mp_spectral_gap': float(S[0] / S[1])" in lf and "'mp_norm_per_token': float(S2_sum / max(m, 1))" in lf
          and "'mp_top_sv_excess': float(S[0] / mp_thresh)" in lf, "lyra_features.py feature definitions")
    q4 = all(abs(t["features"][s_][w]["mp_signal_rank"] * 4 - round(t["features"][s_][w]["mp_signal_rank"] * 4)) < 1e-9
             for t in v2 for s_ in t["features"] for w in ("full", "gen"))
    btr = load("kv_cloak_base/trials_base.json")
    q7 = all(abs(t["features"][s_][w]["mp_signal_rank"] * 7 - round(t["features"][s_][w]["mp_signal_rank"] * 7)) < 1e-6
             for t in btr for s_ in t["features"] for w in ("full", "gen"))
    claim("Features are layer averages: every distilled signal rank is a multiple of 1/4 (4 layers), every base one of 1/7 (7 layers)",
          q4 and q7, f"distilled 1/4: {q4}; base 1/7: {q7}")
    thr = [t for t in T if "Thornbridge" in t["prompt"]][0]
    v1 = [t for t in v2 if t["prompt_idx"] == 1][0]
    claim("Thornbridge, Montana: confabulated in the replication (80-token cap), hedged in the v2 run (309 tokens)",
          thr["behavior"] == "CONFABULATED" and thr["n_generated"] == 80 and v1["behavior"] == "HEDGED" and v1["n_tokens"] == 309
          and "Thornbridge" in v1["response_preview"], f"replication {thr['behavior']}/{thr['n_generated']}; v2 {v1['behavior']}/{v1['n_tokens']}")
    claim("Labels cannot be audited: v2 stores 150-character previews", prev == 150, f"max preview {prev}")

    def mw(pos, neg):
        u = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        return u.statistic / (len(pos) * len(neg)), u.pvalue
    def sgf(t, st): return t["features"][st]["full"]["mp_spectral_gap"]
    strata = {}
    for name, sel_ in (("fabricated", lambda t: t["domain"] == "fabricated"), ("impossible", lambda t: t["domain"] == "impossible"),
                       ("capped", lambda t: t["n_tokens"] == 400), ("nofuture", lambda t: t["domain"] != "future")):
        Cs = [t for t in v2 if t["behavior"] == "CONFABULATED" and sel_(t)]; Hs = [t for t in v2 if t["behavior"] == "HEDGED" and sel_(t)]
        strata[name] = (len(Cs), len(Hs), mw([sgf(t, "none") for t in Cs], [sgf(t, "none") for t in Hs]),
                        mw([sgf(t, "orthogonal") for t in Cs], [sgf(t, "orthogonal") for t in Hs]))
    ok = (strata["fabricated"][:2] == (4, 42) and rounds(0.976, strata["fabricated"][2][0]) and rounds(0.821, strata["fabricated"][3][0])
          and rounds(0.033, strata["fabricated"][3][1])
          and strata["impossible"][:2] == (3, 25) and rounds(0.933, strata["impossible"][2][0])
          and strata["capped"][:2] == (6, 49) and rounds(0.935, strata["capped"][2][0])
          and strata["nofuture"][:2] == (7, 67) and rounds(0.962, strata["nofuture"][2][0]) and rounds(0.751, strata["nofuture"][3][0])
          and rounds(0.028, strata["nofuture"][3][1]))
    claim("Strata (spectral gap, uncloaked -> rotation): fabricated 0.976 -> 0.821 (p 0.033); unanswerable 0.933; capped 0.935; without future prompts 0.962 -> 0.751 (p 0.028)",
          ok, f"{ {k: (v[0], v[1], round(v[2][0], 4), round(v[3][0], 4), round(v[3][1], 4)) for k, v in strata.items()} }")
    Hf = [t for t in v2 if t["behavior"] == "HEDGED" and t["domain"] == "future"]
    Ho = [t for t in v2 if t["behavior"] == "HEDGED" and t["domain"] != "future"]
    means = [round(float(np.mean([sgf(t, st) for t in g])), 1) for st in ("none", "orthogonal") for g in (Hf, Ho)]
    minC = min(sgf(t, "none") for t in v2 if t["behavior"] == "CONFABULATED")
    above = [t for t in v2 if t["behavior"] == "HEDGED" and sgf(t, "none") >= minC]
    claim("Future-event hedged replies: mean gap 28.3 vs 23.6 uncloaked, 10.2 vs 10.9 rotated; 12 of the 21 hedged at or above the lowest confabulation",
          means == [28.3, 23.6, 10.2, 10.9] and len(Hf) == 20 and (len(above), sum(t["domain"] == "future" for t in above)) == (21, 12),
          f"means {means}; above {len(above)}, future {sum(t['domain'] == 'future' for t in above)}")

    def carry(trials, wins):
        TT_ = [t for t in trials if t["behavior"] == "CONFABULATED"] + [t for t in trials if t["behavior"] == "HEDGED"]
        lab = np.array([t["behavior"] == "CONFABULATED" for t in TT_])
        keys = [(w, f) for w in wins for f in FEATS]
        U = np.array([[t["features"]["none"][w][f] for (w, f) in keys] for t in TT_])
        Rr = np.array([[t["features"]["orthogonal"][w][f] for (w, f) in keys] for t in TT_])
        def raw(x, lb):
            n1 = lb.sum(); n0 = len(lb) - n1; r = rankdata(x)
            return (r[lb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        def stat(lb):
            au = np.array([raw(U[:, j], lb) for j in range(U.shape[1])]); fo = np.maximum(au, 1 - au); j = int(np.argmax(fo))
            ar = raw(Rr[:, j], lb)
            return keys[j], fo[j], (ar if au[j] >= 0.5 else 1 - ar)
        k, ou, orr = stat(lab)
        rg = np.random.default_rng(2026)
        nu, nr = zip(*[stat(rg.permutation(lab))[1:] for _ in range(10000)])
        return k, ou, orr, (1 + sum(x >= orr for x in nr)) / 10001, (1 + sum(x >= ou for x in nu)) / 10001
    d5, d10 = carry(v2, ["full"]), carry(v2, ["full", "gen"])
    b5, b10 = carry(btr, ["full"]), carry(btr, ["full", "gen"])
    claim("Selection carried into rotation, distilled: spectral gap 0.903 -> 0.787, p = 0.030 among five, p = 0.049 among ten",
          d5[0] == d10[0] == ("full", "mp_spectral_gap") and rounds(0.787, d10[2]) and rounds(0.030, d5[3]) and rounds(0.049, d10[3]),
          f"five: p {d5[3]:.4f}; ten: p {d10[3]:.4f}")
    claim("Selection carried into rotation, base: best of ten is signal fraction (last 80 tokens) 0.715 -> 0.717, p < 0.001; full-cache 0.661 -> 0.639, p = 0.038 among five; base best-of-ten uncloaked p < 0.001",
          b10[0] == ("gen", "mp_signal_fraction") and rounds(0.715, b10[1]) and rounds(0.717, b10[2]) and b10[3] < 0.001
          and b5[0] == ("full", "mp_signal_fraction") and rounds(0.639, b5[2]) and rounds(0.038, b5[3]) and b10[4] < 0.001,
          f"ten: {b10[0]} {b10[1]:.4f}->{b10[2]:.4f} p {b10[3]:.4f}, uncloaked p {b10[4]:.4f}; five: p {b5[3]:.4f}")
    bg = base["results"]["none"]["gen"]["mp_signal_fraction"]
    claim("Base, last 80 tokens: signal fraction 0.715 (p < 0.001)", rounds(0.715, bg["auroc"]) and bg["p"] < 0.001, f"{bg['auroc']}, p {bg['p']:.5f}")
    gn = v2r["results"]["none"]["gen"]["mp_norm_per_token"]; go = v2r["results"]["orthogonal"]["gen"]["mp_norm_per_token"]
    claim("Distilled, last 80 tokens: norm per token 0.800 (p = 0.007), the same under rotation",
          rounds(0.800, gn["auroc"]) and rounds(0.007, gn["p"]) and gn["auroc"] == go["auroc"], f"{gn['auroc']} p {gn['p']:.4f}; rotated {go['auroc']}")
    srr = R["none"]["mp_signal_rank"]
    claim("Signal rank: p = 0.026 (not significant after correction); 0.684 after removing length",
          rounds(0.026, srr["p"]) and srr["p"] > 0.01 and rounds(0.684, srr["fwl_auroc"]), f"p {srr['p']:.4f}, FWL {srr['fwl_auroc']}")
    ratio = [t["mp_real_kvcloak"]["mp_norm_per_token"] / t["mp_none"]["mp_norm_per_token"] for t in real]
    claim("The full transform multiplies norm per token by 2.34-2.37 in every trial",
          rounds(2.34, min(ratio), 2) and rounds(2.37, max(ratio), 2) and len(ratio) == 100, f"min {min(ratio):.3f}, max {max(ratio):.3f}")
    claim("The full-transform run computed the full window only",
          all(set(t) == {"prompt_idx", "behavior", "n_tokens", "response_preview", "mp_none", "mp_sim_rotation", "mp_real_kvcloak"} for t in real),
          f"trial keys {sorted(real[0])}")
    claim("The magnitude-attack arm corrected exactly as many replies as direct steering on all four models",
          all(v["magnitude_attack"]["corrected"] == v["direct"]["corrected"] for v in inj.values()),
          f"{ {m: (v['magnitude_attack']['corrected'], v['direct']['corrected']) for m, v in inj.items()} }")
    claim("Llama: 20 of 50 replies ambiguous and excluded",
          xa["meta-llama/Llama-3.1-8B-Instruct"]["behavior_counts"].get("AMBIGUOUS") == 20, f"{xa['meta-llama/Llama-3.1-8B-Instruct']['behavior_counts']}")

    # ----------------------------------------------- the June plan
    section("Correction history")
    plan = None
    for base_dir in (REPO.parent / "human-review" / "archive",):
        p = base_dir / "COALITION_REMEDIATION_PLAN_20260623.md"
        if p.exists():
            plan = p.read_text(encoding="utf-8")
    if plan is None:
        skip("The June 2026 remediation plan recommended withdrawing the rotation claim",
             "human-review/archive/COALITION_REMEDIATION_PLAN_20260623.md is in the lab's human-review repository, not this one")
    else:
        claim("The June 2026 remediation plan (23 June) recommended withdrawing the rotation-simulation degradation claim",
              "Retract the rotation-simulation AUROC degradation claim" in plan and "0.9944 = 0.9944" in plan,
              "human-review/archive/COALITION_REMEDIATION_PLAN_20260623.md, section 1C")

    agni = json.load(open(REPO / "tools" / "agni" / "style" / "kv-cloak-defense-paper.agni_style.json", encoding="utf-8"))
    claim("The lab's style gate rejected version 1.0 on 4 September 2026",
          agni["verdict"] == "REJECTED" and agni["timestamp"].startswith("2026-09-04"),
          f"tools/agni/style/kv-cloak-defense-paper.agni_style.json: {agni['verdict']}, {agni['timestamp'][:10]}")
    try:
        import subprocess
        first = subprocess.run(["git", "log", "--reverse", "--format=%ad", "--date=short", "--", "kv-cloak-defense-paper/main.tex"],
                               cwd=REPO, capture_output=True, text=True, timeout=60).stdout.split()
    except Exception:  # git not available
        first = []
    if first:
        claim("Version 1.0 has been in this repository since 29 April 2026", first[0] == "2026-04-29",
              f"first commit touching main.tex: {first[0]}")
    else:
        skip("Version 1.0 has been in this repository since 29 April 2026", "git history not available")

    recs = {}
    for name in ("agni_paper_kvcloak_v11", "agni_style_kvcloak_v11", "agni_paper_kvcloak_v11_r2", "agni_style_kvcloak_v11_r2"):
        f = HERE / "review" / (name + ".json")
        recs[name] = json.load(open(f, encoding="utf-8")) if f.exists() else None
    claim("Two rounds of Agni paper and style mode on the drafts are on record in review/ (r1: REJECTED / CONDITIONAL; r2: CONDITIONAL / CONDITIONAL)",
          all(recs.values())
          and [(r["phase"], r["verdict"]) for r in recs.values()] == [("paper", "REJECTED"), ("style", "CONDITIONAL"),
                                                                      ("paper", "CONDITIONAL"), ("style", "CONDITIONAL")],
          f"{ {k: (v['phase'], v['verdict'], v['timestamp'][:16]) if v else None for k, v in recs.items()} }")

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAIL: ' + ' | '.join(FAILS)}"
          + (f"   ({len(SKIPS)} SKIP: source not in this repository)" if SKIPS else ""))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
