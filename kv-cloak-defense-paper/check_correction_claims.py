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
          nfeat == 24 and len(T) == 100 and rel_x < 1e-4, f"max relative change {rel_x:.1e}")
    heads = sorted(T[0]["uncloaked_per_head"][L[0]].keys())
    rel_h = max(abs(t["cloaked_per_head"][l][h] - t["uncloaked_per_head"][l][h])
                / max(abs(t["uncloaked_per_head"][l][h]), 1e-12) for t in T for l in L for h in heads)
    claim("Rotation left all 16 per-head stable ranks (4 heads x 4 layers) unchanged in all 100 trials",
          len(heads) * len(L) == 16 and rel_h < 1e-4, f"max relative change {rel_h:.1e}")

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
    claim("Its labels are prompt types (50 fabricated-entity or future-event prompts vs 50 factual questions), not behaviour",
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
    claim("No cross-architecture model survives Bonferroni correction for four comparisons (p < 0.0125)",
          all(p >= 0.0125 for p in ps.values()), f"{ {m: round(p, 4) for m, p in ps.items()} }")
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
    claim("The full mechanism moves sigma1: the spectral gap of every reply rises (hedged mean 24.7 -> above 84)",
          all(t["mp_real_kvcloak"]["mp_spectral_gap"] > t["mp_none"]["mp_spectral_gap"] for t in C + H)
          and min(t["mp_real_kvcloak"]["mp_spectral_gap"] for t in C + H) > 84,
          f"min under the mechanism {min(t['mp_real_kvcloak']['mp_spectral_gap'] for t in C + H):.1f}")

    # within-cloak, cross-architecture (Table 1 'Det.(clk)')
    clk = {m: d["orthogonal"]["auroc"] for m, d in det.items() if d["orthogonal"]["auroc"] is not None}
    bclk = base["results"]["orthogonal"]["full"]["mp_signal_fraction"]["auroc"]
    pairs = {"distilled": (ex["mp_none", "mp_spectral_gap"][0], ex["mp_sim_rotation", "mp_spectral_gap"][0]),
             "Qwen2.5-7B": (bsf["auroc"], bclk)}
    pairs.update({m: (un[m], clk[m]) for m in clk})
    fell = [m for m, (a, b) in pairs.items() if b < a]; rose = [m for m, (a, b) in pairs.items() if b > a]
    claim("Within the rotation, detection fell on four models and rose on one (Qwen2.5-14B, 0.784 -> 0.951, p < 0.001)",
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
    claim("Random-direction perturbation: 0-2 adverse events per model in 10 trials; 4 of 40 pooled (one-sided 95% upper bound 21%)",
          set(mag.values()) <= {0, 1, 2} and max(mag.values()) == 2 and (k, n) == (4, 40) and rounds(0.21, up_pool, 2),
          f"{mag}; pooled upper bound {up_pool:.3f}")
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

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAIL: ' + ' | '.join(FAILS)}"
          + (f"   ({len(SKIPS)} SKIP: source not in this repository)" if SKIPS else ""))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
