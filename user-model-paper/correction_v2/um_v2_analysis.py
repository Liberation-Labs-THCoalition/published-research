#!/usr/bin/env python3
"""um_v2_analysis.py: R1, R2, R7 and the paired tests against R0 (run plan 2026-10-18; revised for Agni round 1,
analysis review N1-N9 and encode review N1-N10). CPU only. Reads um_v2_encode.py's output, the trials file (for R0's
text) and R0_results.json. Pre-registration: prereg_um_v2_R1R2.json. Every rule below was fixed before any encode
existed; nothing is tuned on the data. Every verdict is computed here; none is left to prose.

R1  v2 as committed (wk_backtrack_probe_v2.py, "v2"): its three functions verbatim (v2:48-89, AST-checked by the dry
    run), its loop ported line for line (v2:208-332). Keys = mean over all positions at MODEL layers 3, 7, 11, 15,
    concatenated (4,096 dims) and dotted with one direction normalised over the concatenation: v2's own statistic,
    kept separate from R2's. Reproduced iff |acc - 368/900| <= h, where h is the spread of R1's accuracy when the
    pooled keys are perturbed by the run-to-run and dtype-change differences the encode measured (Agni encode N8).
R2  the probe the paper describes (main.tex:198-207): d_e = W_K(r_e - r) from the training fold only (by linearity,
    class means of the projected residuals W_K r), r = decoder-layer output, positions >= 50 for both r and keys.
    PRIMARY cosine (Agni analysis N1): each model layer's block of d_e and of the pooled key is normalised to unit
    length and the per-layer cosines are averaged, giving the paper's 30 features (main.tex:201). This equals the
    cosine between the concatenations of unit blocks. A single cosine over the raw concatenation (v1 of this script)
    weighted layers by their residual norm, which grows with depth; it is kept as an exploratory cell.
    FWL fitted on the training fold (main.tex:230) on [1, log1p(n_tokens), 1/n_pooled] (encode N1, analysis N7).
R0  R0_text_baseline.py's code with n_splits as a parameter; at 5 folds it must reproduce R0_results.json (a verdict).
R7  spectral features on the first 100 story tokens (n-matched), per layer set, within-fold FWL, within-topic
    permutation null per set, Bonferroni over informative sets, a positive control (topic, non-grouped CV).
Tests: McNemar exact on per-trial correctness; DeLong for paired AUROC; cell (300) and topic (10) cluster bootstraps.
Power through the same McNemar, conditional on R0's actual 10-fold correctness vector.

  python um_v2_analysis.py --enc ENC --trials emotion_bridge_trials.json --r0 R0_results.json --out results.json
"""
import argparse, hashlib, json, platform, sys, time, warnings
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import binom, binomtest, norm, rankdata, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
FULL = list(range(3, 64, 4))
POOLS = ("mean_all", "mean_skip50", "last", "story")
SRCS = ("res_out", "res_in_ln", "res_pre")
VALENCE_POS = {"calm", "cheerful", "content", "delighted", "grateful", "happy",
               "hopeful", "inspired", "joyful", "loving", "peaceful", "pleased",
               "proud", "relaxed", "satisfied", "serene", "thankful", "thrilled"}      # v2:40-42
VALENCE_NEG = {"afraid", "angry", "anxious", "bitter", "depressed", "desperate",
               "disgusted", "frustrated", "gloomy", "hostile", "hurt", "jealous",
               "lonely", "miserable", "sad", "terrified"}                              # v2:43-45
PAPER_K, N_TRIALS, N_EMO = 368, 900, 30     # v2: 0.40889 = 368/900 (wk_backtrack_v2_results.json, read 2026-10-18)
ALPHA, CHANCE, LEN_MARGIN, POWER_MIN = 0.05, 1 / 30, 0.05, 0.8
FLOOR_PTS = 3.0          # Lyra 2026-10-18: EXCEEDS_TEXT_TFIDF needs R2 - R0(10 folds) >= 3.0 points (prereg §5.2)
TIE_MAX = 0.5            # CONFOUND_SUSPECTED: a feature tied on more than half the trials (the FWL reinjection shape)
VARKEPT_MIN = 0.1        # CONFOUND_SUSPECTED: < 10% of a feature's variance left after FWL, i.e. VIF > 10 (prereg §5.2)
H_MAX = 17               # Agni r3: R1/v1 band cap = half the 35-trial gap between v2's 368 and the R0 file's 333
DET_CEIL = 2 ** -8       # Agni r3 encode N6: determinism ceiling, per-layer relative RMS (bf16 unit roundoff)
WITNESS_NEG_MIN = 0.25   # Agni r3 New-5: shuffled May stories must have >= 25% of tokens outside the nucleus
V1_K = 470               # v1's 30-way count, 0.52222 x 900 (wk_backtrack_results.json); checked against the file
SPECTRAL_NAMES = ["frobenius", "eff_rank", "entropy", "top_sv_ratio"]
N_PERM_DEFAULT, N_PERM_R7_DEFAULT, N_BOOT_DEFAULT = 1000, 5000, 10000
S_RAW = ("A raw-residual LR baseline ({src}) reaches {acc:.3f} >= R2's {r2:.3f}: the probe's information is in the "
         "residual stream without W_K or the cache, and the paper may not say that W_K or the KV cache adds to it.")
S_PROJ = ("The pre-RoPE key projection (W_K x) beats the cache keys by {gain} trials (McNemar p = {p:.3g}): the "
          "information is in the projection, not the KV cache's geometry, and any claim is stated about W_K x.")
S_R1R2 = ("The paper's 40.9% is not reproduced on current weights ({r1v}); separately, the probe as the Method "
          "describes it beats text (TF-IDF) by {d:.1f} points. They are different pipelines: the paper's number is "
          "replaced by R2's, reported with R2's pipeline, and 40.9% is reported as unreproduced.")
S_R0 = ("R0 was not reproduced on this environment, so S1, S2 and S4 (which compare against the R0 file) are "
        "reported as 'comparator not reproduced' and are not cited until the cause is found; the primary, against "
        "R0 recomputed at 10 folds here, is unaffected.")
S_R0T = ("A tuned text (TF-IDF) baseline reaches {acc:.3f} >= R2's {r2:.3f}: the comparison with the untuned R0 may "
         "not be cited as the cache beating text.")
R1_LAYERS = (3, 7, 11, 15)                                                            # v2:38, model layer ids
LAYER_SETS = {**{f"L{l}": [l] for l in FULL}, "v2_L3-L15": list(R1_LAYERS), "all16": list(FULL)}
PRIMARY = {"layers": "v2_L3-L15", "pool": "mean_skip50", "sim": "cos_perlayer", "folds": 10, "src": "res_out",
           "target": "cache", "fwl": "pooled"}
LIX = lambda names: [FULL.index(l) for l in names]                                    # model layer id -> shard row
REGISTERED = {"FLOOR_PTS": FLOOR_PTS, "TIE_MAX": TIE_MAX, "VARKEPT_MIN": VARKEPT_MIN, "ALPHA": ALPHA,
              "POWER_MIN": POWER_MIN, "LEN_MARGIN": LEN_MARGIN, "H_MAX": H_MAX, "DET_CEIL": DET_CEIL,
              "WITNESS_NEG_MIN": WITNESS_NEG_MIN, "PRIMARY": PRIMARY, "R1_LAYERS": list(R1_LAYERS),
              "PAPER_K": PAPER_K, "V1_K": V1_K, "SPECTRAL_NAMES": SPECTRAL_NAMES, "N_PERM": N_PERM_DEFAULT,
              "N_PERM_R7": N_PERM_R7_DEFAULT, "N_BOOT": N_BOOT_DEFAULT, "POWER_PSI": "independence",
              "POWER_GATE_PTS": 3}


def check_registered(prereg):
    """Agni r3 New-9: the constants this script decides with must equal the prereg JSON's, not just be hashed."""
    reg = prereg.get("registered_constants")
    if not isinstance(reg, dict):
        raise SystemExit("prereg JSON has no registered_constants block")
    bad = {k: (v, reg.get(k)) for k, v in REGISTERED.items() if reg.get(k) != v}
    if bad:
        raise SystemExit(f"code constants differ from the prereg (code, prereg): {bad}")
    return reg


def check_prediction(path, sealed):
    """Agni r3 encode N9: the sealed prediction must be a prediction, not a placeholder. And it must be the one Lyra
    sealed: its sha256 equals the prereg's registered prediction_sha256 (Lyra, sign-off 2026-10-19). Comparing only to
    the encode's manifest let an edited prediction through, since the encode records whatever file it is given."""
    b = Path(path).read_bytes()
    t = b.decode("utf-8", "replace")
    if len(b) < 200 or not all(x in t for x in ("R1", "R2", "R7")):
        raise SystemExit(f"{path}: not a sealed prediction (needs >= 200 bytes and R1, R2 and R7 in it)")
    if hashlib.sha256(b).hexdigest() != sealed:
        raise SystemExit(f"{path}: sha256 {hashlib.sha256(b).hexdigest()[:16]} is not the sealed prediction's "
                         f"{str(sealed)[:16]} (prereg registered_constants.prediction_sha256)")


def hashed_json(path, want, what):
    if sha(path) != want:
        raise SystemExit(f"{what}: {path} sha256 {sha(path)[:16]} != registered {want[:16]}")
    return json.load(open(path, encoding="utf-8"))


def count_from(results, name_prefix, n):
    """A probe's 30-way count from its results file (Agni r3 New-9: not typed in)."""
    acc = next(p["accuracy"] for p in results["probes"] if p["name"].startswith(name_prefix))
    k = int(round(acc * n))
    assert abs(acc * n - k) < 1e-6, f"{name_prefix}: {acc} x {n} is not a count"
    return k


# ---------- v2:48-89, verbatim ----------
def fwl_residualize(X, confound):
    log_conf = np.log1p(confound).reshape(-1, 1)
    Z = np.column_stack([np.ones(len(log_conf)), log_conf])
    beta = np.linalg.lstsq(Z, X, rcond=None)[0]
    return X - Z @ beta


def compute_emotion_directions(trials, indices, key_activations, emotions_list):
    directions = {}
    for emotion in set(emotions_list):
        emo_indices = [i for i in indices if emotions_list[i] == emotion]
        if len(emo_indices) < 2:
            continue
        emo_keys = np.array([key_activations[i] for i in emo_indices])
        directions[emotion] = emo_keys.mean(axis=0)
    all_keys = np.array([key_activations[i] for i in indices])
    global_mean = all_keys.mean(axis=0)
    for emotion in directions:
        directions[emotion] = directions[emotion] - global_mean
        norm = np.linalg.norm(directions[emotion])
        if norm > 0:
            directions[emotion] = directions[emotion] / norm
    return directions


def project_trials(key_activations, indices, directions, emotions_available):
    n_emotions = len(emotions_available)
    X = np.zeros((len(indices), n_emotions))
    for i, trial_idx in enumerate(indices):
        k = key_activations[trial_idx]
        for j, emotion in enumerate(emotions_available):
            if emotion in directions:
                X[i, j] = float(np.dot(k, directions[emotion]))
            else:
                X[i, j] = 0.0
    return X
# ---------- end verbatim ----------


def gkf(k):
    """GroupKFold, pinned unshuffled (sklearn >= 1.6 has a shuffle argument; v2 and R0 used the default, False)."""
    try:
        return GroupKFold(n_splits=k, shuffle=False)
    except TypeError:
        return GroupKFold(n_splits=k)


def fold_ids(groups, k, X=None, y=None):
    f = np.full(len(groups), -1)
    for i, (_, te) in enumerate(gkf(k).split(np.zeros(len(groups)) if X is None else X, y, groups)):
        f[te] = i
    return f


def fwl_train(Xtr, Xte, Ztr, Zte):
    """FWL fitted on the training fold, applied to both (main.tex:230). Z includes the intercept."""
    b = np.linalg.lstsq(Ztr, Xtr, rcond=None)[0]
    return Xtr - Ztr @ b, Xte - Zte @ b


def covariates(nt, npool, mode):
    cols = [np.ones(len(nt)), np.log1p(nt)]
    if mode == "pooled" and np.ptp(npool) > 0:
        cols.append(1.0 / npool)
    return np.column_stack(cols)


def scale_fit(Xtr, Xte, ytr, binary):
    Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)                 # v2:228-233
    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr)        # v2:238-239
    return clf.predict_proba(Xte)[:, 1] if binary else clf.predict(Xte)


def valence_subset(emotions):
    vv = [i for i, e in enumerate(emotions) if e in VALENCE_POS or e in VALENCE_NEG]   # v2:188-193
    return vv, np.array([1 if emotions[i] in VALENCE_POS else 0 for i in vv])


def r1_thirty(acts, nt, emotions, groups, k=5, fwl="v2"):
    """v2:213-240. fwl='v2' as run (test-fold FWL fitted on the test fold, v2:227); 'train' as the paper states."""
    ea = sorted(set(emotions))
    le = {e: i for i, e in enumerate(ea)}
    preds = np.full(len(emotions), -1)
    for tr, te in gkf(min(k, len(set(groups)))).split(range(len(emotions)), emotions, groups):
        d = compute_emotion_directions(None, tr, acts, emotions)
        Xtr, Xte = project_trials(acts, tr, d, ea), project_trials(acts, te, d, ea)
        if fwl == "v2":
            Xtr, Xte = fwl_residualize(Xtr, nt[tr]), fwl_residualize(Xte, nt[te])
        else:
            Xtr, Xte = fwl_train(Xtr, Xte, covariates(nt[tr], None, "paper"), covariates(nt[te], None, "paper"))
        preds[te] = scale_fit(Xtr, Xte, np.array([le[emotions[i]] for i in tr]), False)
    return preds, ea


def r1(acts, nt, emotions, groups, k=5, fwl="v2"):
    """v2:208-332 for one key type."""
    preds, ea = r1_thirty(acts, nt, emotions, groups, k, fwl)
    y = np.array([ea.index(e) for e in emotions])
    acc = accuracy_score(y[preds >= 0], preds[preds >= 0])
    vv, yv = valence_subset(emotions)
    gv, ntv = groups[vv], nt[vv]
    probs, ccs = np.zeros(len(yv)), []
    gk = gkf(min(k, len(set(gv))))
    for tr, te in gk.split(range(len(yv)), yv, gv):                    # v2:256-280
        otr, ote = [vv[i] for i in tr], [vv[i] for i in te]
        d = compute_emotion_directions(None, otr, acts, emotions)
        Xtr, Xte = project_trials(acts, otr, d, ea), project_trials(acts, ote, d, ea)
        if fwl == "v2":
            Xtr, Xte = fwl_residualize(Xtr, ntv[tr]), fwl_residualize(Xte, ntv[te])
        else:
            Xtr, Xte = fwl_train(Xtr, Xte, covariates(ntv[tr], None, "paper"), covariates(ntv[te], None, "paper"))
        if len(np.unique(yv[tr])) >= 2:
            probs[te] = scale_fit(Xtr, Xte, yv[tr], True)
        Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)              # CCS, v2:288-311 (no scaler, as run)
        dirn = Xtr[yv[tr] == 1].mean(0) - Xtr[yv[tr] == 0].mean(0)
        dirn = dirn / np.linalg.norm(dirn) if np.linalg.norm(dirn) > 0 else dirn
        if len(np.unique(yv[te])) >= 2:
            ccs.append(roc_auc_score(yv[te], Xte @ dirn))
    lp = np.zeros(len(yv))                                             # text-length baseline, v2:319-329
    for tr, te in gk.split(range(len(yv)), yv, gv):
        if len(np.unique(yv[tr])) >= 2:
            lp[te] = LogisticRegression(max_iter=1000).fit(ntv[tr].reshape(-1, 1), yv[tr]).predict_proba(
                ntv[te].reshape(-1, 1))[:, 1]
    return {"accuracy": float(acc), "k_correct": int(np.sum(preds == y)), "times_chance": float(acc * len(ea)),
            "folds": k, "fwl": fwl, "pred": [ea[p] for p in preds], "valence_auroc": float(roc_auc_score(yv, probs)),
            "valence_p": probs.round(6).tolist(), "ccs_auroc": float(np.mean(ccs)) if ccs else None,
            "length_baseline_auroc": float(roc_auc_score(yv, lp))}


def r1_band(blocks, nt, emotions, groups, delta_rel, R, seed, jobs):
    """Spread of R1's correct count when each layer's pooled keys get Gaussian noise of the measured relative RMS size
    (run-to-run and dtype-change differences from the encode). Half-width = max |k_perturbed - k_unperturbed|."""
    rms = np.sqrt((blocks ** 2).mean(axis=(0, 2)))
    p0, ea = r1_thirty(blocks.reshape(len(emotions), -1), nt, emotions, groups)
    y = np.array([ea.index(e) for e in emotions])
    k0 = int(np.sum(p0 == y))
    ks = Parallel(n_jobs=jobs)(delayed(_r1_k)(blocks, rms, delta_rel, nt, emotions, groups, seed + 10_000 + r, y)
                               for r in range(R)) if R else []
    return {"delta_rel_per_layer": [float(x) for x in delta_rel], "R": R, "k_unperturbed": k0, "k_perturbed": ks,
            "half_width_trials": int(max([abs(k - k0) for k in ks], default=0))}


def _r1_k(blocks, rms, delta_rel, nt, emotions, groups, s, y):
    A = blocks + np.random.default_rng(s).normal(size=blocks.shape) * (np.asarray(delta_rel) * rms)[None, :, None]
    p, _ = r1_thirty(A.reshape(len(emotions), -1), nt, emotions, groups)
    return int(np.sum(p == y))


def v1_features(K4, WK4, vecs, emo_order):
    """wk_backtrack_probe.py ("v1"):92-102 and :124-146, on this encode's mean keys and saved W_K with May's
    emotion_vectors.json. direction = W_K @ v (normalised); feature = dot(layer mean key, direction); layer-major,
    then v1's emotion order (the JSON's own order, v1:68)."""
    feats = []
    for j in range(K4.shape[1]):
        D = np.stack([WK4[j] @ np.asarray(vecs[j][e], np.float32) for e in emo_order])
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        feats.append(K4[:, j, :] @ D.T)
    return np.concatenate(feats, axis=1)


def v1_counts(X, slen, emotions, groups):
    """v1:184-255: 30-way count, binary valence AUROC and CCS, v1's FWL on story_length (characters)."""
    le = {e: i for i, e in enumerate(sorted(set(emotions)))}
    y = np.array([le[e] for e in emotions])
    pred = np.zeros(len(y))
    for tr, te in gkf(min(5, len(set(groups)))).split(X, emotions, groups):
        Xtr, Xte = np.nan_to_num(fwl_residualize(X[tr], slen[tr])), np.nan_to_num(fwl_residualize(X[te], slen[te]))
        sc = StandardScaler()
        pred[te] = LogisticRegression(max_iter=2000, C=0.1).fit(sc.fit_transform(Xtr), y[tr]).predict(sc.transform(Xte))
    vv, yv = valence_subset(emotions)
    Xv, sv, gv = X[vv], slen[vv], groups[vv]
    probs, ccs = np.zeros(len(yv)), []
    for tr, te in gkf(min(5, len(set(gv)))).split(Xv, yv, gv):
        Xtr, Xte = np.nan_to_num(fwl_residualize(Xv[tr], sv[tr])), np.nan_to_num(fwl_residualize(Xv[te], sv[te]))
        d = Xtr[yv[tr] == 1].mean(0) - Xtr[yv[tr] == 0].mean(0)
        d = d / np.linalg.norm(d) if np.linalg.norm(d) > 0 else d
        if len(np.unique(yv[te])) >= 2:
            ccs.append(roc_auc_score(yv[te], Xte @ d))
        sc = StandardScaler()
        Ztr, Zte = sc.fit_transform(Xtr), sc.transform(Xte)
        if len(np.unique(yv[tr])) >= 2:
            probs[te] = LogisticRegression(max_iter=2000, C=0.1).fit(Ztr, yv[tr]).predict_proba(Zte)[:, 1]
    return int(np.sum(pred == y)), float(roc_auc_score(yv, probs)), float(np.mean(ccs)) if ccs else None


def _v1_k(K4, rms, delta, WK4, vecs, order, slen, emotions, groups, s):
    K = K4 + np.random.default_rng(s).normal(size=K4.shape) * (np.asarray(delta) * rms)[None, :, None]
    return v1_counts(v1_features(K, WK4, vecs, order), slen, emotions, groups)[0]


def v1_check(K4, WK4, vecs, order, slen, emotions, groups, may_k, delta, R, seed, jobs):
    """Agni r3 encode N4: v1 ran on 2026-05-21 from the same HF cache as v2 (v1:33 = v2:34), 90 minutes before it.
    Recomputing v1 on today's keys and W_K tests today's weights and numerics against v2's evening, with no GPU."""
    k, auc, ccs = v1_counts(v1_features(K4, WK4, vecs, order), slen, emotions, groups)
    rms = np.sqrt((K4 ** 2).mean(axis=(0, 2)))
    ks = Parallel(n_jobs=jobs)(delayed(_v1_k)(K4, rms, delta, WK4, vecs, order, slen, emotions, groups,
                                              seed + 20_000 + r) for r in range(R)) if R else []
    h = int(max([abs(x - k) for x in ks], default=0))
    v = "BAND_TOO_WIDE" if h > H_MAX else "REPRODUCED" if abs(k - may_k) <= h else "NOT_REPRODUCED"
    return {"k": k, "may_k": may_k, "half_width_trials": h, "k_perturbed": ks, "valence_auroc": auc, "ccs_auroc": ccs,
            "verdict": v}


def r1_verdict(k, paper_k, h, v1v):
    """prereg §5.1. The band comes from bf16 run-to-run repeats only (v2 ran in bf16, v2:107); h > H_MAX is too wide
    for 'reproduced' to mean anything. v1's check, not May's bridge fingerprint, says whether weights are v2's."""
    if h > H_MAX:
        return "UNINFORMATIVE_BAND_TOO_WIDE"
    inb = abs(k - paper_k) <= h
    if v1v == "REPRODUCED":
        return "REPRODUCED" if inb else "NOT_REPRODUCED_CODE_PATH"
    return "REPRODUCED_WEIGHTS_UNVERIFIED" if inb else "NOT_REPRODUCED_WEIGHTS_UNVERIFIABLE"


def sign_flip(ca, cb, cell_id, B, seed):
    """Agni r3 encode N3: a paired, cell-level test of R2 > R0 (reported, not a ladder rule): per-cell net wins,
    random signs over the 300 cells, one-sided p."""
    d = np.bincount(cell_id, ca.astype(float) - cb.astype(float))
    obs = float(d.sum())
    null = (np.random.default_rng(seed).choice([-1.0, 1.0], size=(B, len(d))) * d).sum(1)
    return {"net_wins": obs, "B": B, "p_one_sided": float((1 + np.sum(null >= obs)) / (1 + B))}


def r0_tuned(texts, labels, groups, k=10, grid=(0.01, 0.1, 1.0, 10.0)):
    """Agni r3 New-4, exploratory: R0 with C chosen per outer fold by an inner GroupKFold(3) over the training topics."""
    labels = np.array(labels)
    preds, chosen = np.empty(len(labels), dtype=object), []
    for tr, te in gkf(k).split(texts, labels, groups):
        best = None
        for C in grid:
            acc = []
            for itr, ite in gkf(3).split(np.zeros(len(tr)), labels[tr], groups[tr]):
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
                Xa = vec.fit_transform([texts[tr[i]] for i in itr])
                fit = LogisticRegression(C=C, max_iter=2000).fit(Xa, labels[tr][itr])
                acc.append(np.mean(fit.predict(vec.transform([texts[tr[i]] for i in ite])) == labels[tr][ite]))
            if best is None or np.mean(acc) > best[0]:
                best = (float(np.mean(acc)), C)
        chosen.append(best[1])
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        fit = LogisticRegression(C=best[1], max_iter=2000).fit(vec.fit_transform([texts[i] for i in tr]), labels[tr])
        preds[te] = fit.predict(vec.transform([texts[i] for i in te]))
    return [str(x) for x in preds], chosen


def r0_thirty(texts, labels, groups, k):
    """R0_text_baseline.py:31-43, n_splits=k."""
    labels = np.array(labels)
    preds = np.empty(len(labels), dtype=object)
    for tr, te in gkf(min(k, len(set(groups)))).split(texts, labels, groups):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        preds[te] = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, labels[tr]).predict(Xte)
    return [str(p) for p in preds]


def r0_valence(texts, emotions, groups, k):
    """R0_text_baseline.py:46-58, n_splits=k."""
    idx, y = valence_subset(emotions)
    t, g = [texts[i] for i in idx], np.array([groups[i] for i in idx])
    probs = np.zeros(len(idx))
    for tr, te in gkf(min(k, len(set(g)))).split(t, y, g):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(vec.fit_transform([t[i] for i in tr]), y[tr])
        probs[te] = clf.predict_proba(vec.transform([t[i] for i in te]))[:, 1]
    return probs


def r2_features(Kx, Px, ydir, tr, te, sim):
    """Kx, Px: (n, L, d). d_e = W_K(r_e - r) from training trials only; Px rows are W_K r per layer."""
    g = Px[tr].mean(0)
    D = np.stack([Px[tr][ydir[tr] == c].mean(0) - g for c in np.unique(ydir[tr])])     # (E, L, d)
    n = len(Kx)
    if sim == "cos_concat":                                       # one cosine over the raw concatenation (old)
        Df = D.reshape(len(D), -1)
        Df = Df / np.linalg.norm(Df, axis=1, keepdims=True)
        Kf = Kx.reshape(n, -1)
        X = (Kf @ Df.T) / np.linalg.norm(Kf, axis=1, keepdims=True)
    else:
        Du = D / np.linalg.norm(D, axis=2, keepdims=True)         # unit block per layer
        Ku = Kx if sim == "dot_perlayer" else Kx / np.linalg.norm(Kx, axis=2, keepdims=True)
        C = np.einsum("nld,eld->nle", Ku, Du)                     # per-layer cosine (or dot with a unit direction)
        X = C.reshape(n, -1) if sim == "cos_layerfeats" else C.mean(1)
    return X[tr], X[te]


def cv(featfn, y, groups, Z, k, binary, keep=False, splitter=None, nan_box=None):
    """nan_box (a list) receives the NaN count of each fold's features. A cell with any NaN is void (Agni r3 New-8);
    the NaNs are zeroed only so the fold can finish and be reported."""
    out = np.zeros(len(y)) if binary else np.full(len(y), -1)
    raw = res = None
    splits = splitter if splitter is not None else gkf(k).split(np.zeros(len(y)), y, groups)
    for tr, te in splits:
        Xtr, Xte = featfn(tr, te)
        nn = int(np.isnan(Xtr).sum() + np.isnan(Xte).sum())
        if nan_box is not None:
            nan_box.append(nn)
        if nn:
            Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)
        if keep and raw is None:
            raw, res = np.zeros((len(y), Xte.shape[1])), np.zeros((len(y), Xte.shape[1]))
        if keep:
            raw[te] = Xte
        Xtr, Xte = fwl_train(Xtr, Xte, Z[tr], Z[te])
        if keep:
            res[te] = Xte
        out[te] = scale_fit(Xtr, Xte, y[tr], binary)
    return (out, raw, res) if keep else out


def fwl_diag(raw, res, Z, nt, npool):
    """Agni FWL_TIED_FEATURE_REINJECTION: ties, covariate R^2, variance kept, residual rank correlation with length."""
    n = len(raw)
    tie = [1 - len(np.unique(raw[:, c])) / n for c in range(raw.shape[1])]
    b = np.linalg.lstsq(Z, raw, rcond=None)[0]
    vr = np.var(raw - Z @ b, axis=0) / np.maximum(np.var(raw, axis=0), 1e-30)
    rho_n = [abs(spearmanr(res[:, c], nt)[0]) for c in range(res.shape[1])]
    rho_p = [abs(spearmanr(res[:, c], npool)[0]) if np.ptp(npool) > 0 else 0.0 for c in range(res.shape[1])]
    return {"max_tie_fraction": float(max(tie)), "max_covariate_r2": float(np.max(1 - vr)),
            "min_var_kept": float(np.min(vr)), "max_abs_resid_spearman_ntokens": float(np.nanmax(rho_n)),
            "max_abs_resid_spearman_npooled": float(np.nanmax(rho_p))}


def covariate_only_acc(y, groups, Z, k):
    """The '-fitted' classifier: the length covariates alone, same folds and classifier."""
    p = np.full(len(y), -1)
    for tr, te in gkf(k).split(np.zeros(len(y)), y, groups):
        p[te] = scale_fit(Z[tr, 1:], Z[te, 1:], y[tr], False)
    return float(np.mean(p == y))


def cell_arrays(D, spec):
    ls, p = LIX(LAYER_SETS[spec["layers"]]), POOLS.index(spec["pool"])
    K = D["K"] if spec["target"] == "cache" else D["P"]["res_in_ln"]
    return K[:, ls, p, :], D["P"][spec["src"]][:, ls, p, :]


def r2_cell(D, spec, y, groups, nt, NP, vv, yv, ylab=None, valence=True):
    """One specification. With ylab (permuted labels) returns only the 30-way accuracy."""
    Kx, Px = cell_arrays(D, spec)
    npool = NP[spec["pool"]]
    Z = covariates(nt, npool, spec["fwl"])
    yy = y if ylab is None else ylab
    f = lambda tr, te: r2_features(Kx, Px, yy, tr, te, spec["sim"])
    if ylab is not None:
        return float(np.mean(cv(f, yy, groups, Z, spec["folds"], False) == yy))
    nb = []
    pred, raw, res = cv(f, yy, groups, Z, spec["folds"], False, keep=True, nan_box=nb)
    out = {"spec": spec, "accuracy": float(np.mean(pred == yy)), "pred": pred.astype(int).tolist(),
           "fwl_diagnostics": fwl_diag(raw, res, Z, nt, npool)}
    if valence:
        Kv, Pv, yd = Kx[vv], Px[vv], y[vv]
        fv = lambda tr, te: r2_features(Kv, Pv, yd, tr, te, spec["sim"])
        prob = cv(fv, yv, groups[vv], Z[vv], spec["folds"], True, nan_box=nb)
        out["valence_auroc"], out["valence_p"] = float(roc_auc_score(yv, prob)), prob.round(6).tolist()
    out["n_nan"] = int(sum(nb))
    out["void"] = out["n_nan"] > 0
    return out


def specs():
    reg = [dict(PRIMARY, layers=l, pool=p, sim=s, folds=k) for l in LAYER_SETS for p in POOLS
           for s in ("cos_perlayer", "dot_perlayer") for k in (5, 10)]
    mech = [dict(PRIMARY, layers=l, src=s, target=t) for l in LAYER_SETS
            for s, t in (("res_in_ln", "cache"), ("res_pre", "cache"), ("res_in_ln", "proj"))]
    mech += [dict(PRIMARY, layers=l, sim=s) for l in ("v2_L3-L15", "all16") for s in ("cos_concat", "cos_layerfeats")]
    mech += [dict(PRIMARY, fwl="paper")]
    return reg, mech


def perm_within(y, groups, rng, cells):
    """Agni round 2 (analysis N1, encode DEGENERATE_PERMUTATION_NULL): the 30 emotion labels are permuted over the 30
    (emotion, topic) cells within each topic, so a cell's 3 sibling stories keep one label together."""
    yp = y.copy()
    for g in np.unique(groups):
        ix = np.where(groups == g)[0]
        cu, inv = np.unique(cells[ix], return_inverse=True)
        lab = np.array([y[ix][inv == c][0] for c in range(len(cu))])
        yp[ix] = rng.permutation(lab)[inv]
    return yp


def _r7_perm(X, y, groups, Z, s, cells):
    yp = perm_within(y, groups, np.random.default_rng(s), cells)
    return float(np.mean(cv(lambda tr, te: (X[tr], X[te]), yp, groups, Z, 10, False) == yp))


def r7(S, y, groups, Z, cells, n_perm, seed, jobs, label, sets=None, cov_only=None):
    """H3. The layer sets, and so m, are fixed in advance (17: each layer and all 16); a set whose columns are all
    constant still counts in m. Per set: constant columns dropped; FWL diagnostics; a positive control (topic,
    StratifiedGroupKFold(10) grouped by (emotion, topic) cell, so sibling stories never straddle folds); a cell-level within-topic
    permutation null (vector saved); the detection threshold = the null's (1 - alpha/m) quantile, i.e. the accuracy
    this set's null excludes (Agni round 2, analysis N5 / encode N7)."""
    n, rows = len(y), []
    sets = sets or {**{f"L{l}": [l] for l in FULL}, "all16": list(FULL)}       # `sets` is for the dry run only
    m = len(sets)
    thr = ALPHA / m
    if 0 < n_perm and 1 / (n_perm + 1) >= thr:                   # checked before any work (SOP §4.11)
        raise SystemExit(f"R7: {n_perm} permutations cannot reach p < {thr:.4f} (SOP §4.11)")
    # StratifiedGroupKFold, not GroupKFold: GroupKFold deals equal-size groups to folds in a fixed cyclic order, and
    # with cell ids = emotion * 10 + topic that put each topic in one fold, so the control could never pass (dry run)
    csplit = list(StratifiedGroupKFold(10, shuffle=True, random_state=seed).split(np.zeros(n), groups, cells))
    for name, ls in sets.items():
        X = S[:, LIX(ls), :3].reshape(n, -1)
        keep = [c for c in range(X.shape[1]) if len(np.unique(X[:, c])) > 1]
        row = {"layers": name, "n_columns": X.shape[1], "n_constant_dropped": X.shape[1] - len(keep)}
        if not keep:
            rows.append(dict(row, verdict="UNINFORMATIVE_ALL_CONSTANT"))
            continue
        Xk = X[:, keep]
        f = lambda tr, te: (Xk[tr], Xk[te])
        nb = []
        pred, raw, res = cv(f, y, groups, Z, 10, False, keep=True, nan_box=nb)
        acc = float(np.mean(pred == y))
        ctrl = float(np.mean(cv(f, groups, None, Z, 10, False, splitter=csplit) == groups))
        null = Parallel(n_jobs=jobs)(delayed(_r7_perm)(Xk, y, groups, Z, seed + 50_000 + b, cells)
                                     for b in range(n_perm))
        p = float((1 + sum(x >= acc for x in null)) / (1 + n_perm)) if n_perm else None
        ok = ctrl > 2 / len(np.unique(groups))
        fd = fwl_diag(raw, res, Z, Z[:, 1], Z[:, 1])
        det_thr = float(np.quantile(null, 1 - thr)) if n_perm else None
        # Agni r3 New-3(b): a SIGNAL needs the length route closed, with the same thresholds as R2's primary
        why = [w for w, c in (("tie_fraction_above_0.5", fd["max_tie_fraction"] > TIE_MAX),
                              ("variance_kept_below_0.1", fd["min_var_kept"] < VARKEPT_MIN),
                              ("covariates_alone_reach_detection_threshold",
                               cov_only is not None and det_thr is not None and cov_only >= det_thr)) if c]
        verdict = ("VOID_NAN" if sum(nb) else "PERMUTATION_NOT_RUN" if p is None else
                   ("CONFOUND_SUSPECTED" if why else "SIGNAL") if p < thr else
                   "NULL" if ok else "UNINFORMATIVE_CONTROL_FAILED")
        row.update(accuracy=acc, fwl_diagnostics=fd, perm_p=p, confound_reasons=why, n_nan=int(sum(nb)),
                   positive_control_topic_accuracy=ctrl, positive_control_passed=bool(ok),
                   perm_null=[round(x, 6) for x in null], perm_n_distinct=len(set(null)),
                   perm_null_mean=float(np.mean(null)) if n_perm else None,
                   perm_null_p95=float(np.percentile(null, 95)) if n_perm else None,
                   detection_threshold_accuracy=det_thr, verdict=verdict)
        rows.append(row)
    vs = [r["verdict"] for r in rows]
    if "SIGNAL" in vs:
        v = "SIGNAL_FOUND"
    elif "PERMUTATION_NOT_RUN" in vs:
        v = "PERMUTATION_NOT_RUN"
    elif all(x == "NULL" for x in vs):
        v = "NULL_HOLDS"
    elif "NULL" in vs:
        v = "NULL_HOLDS_PARTIAL"
    elif all(x == "UNINFORMATIVE_ALL_CONSTANT" for x in vs):
        v = "UNINFORMATIVE_ALL_CONSTANT"
    elif "CONFOUND_SUSPECTED" in vs:
        v = "CONFOUND_SUSPECTED"
    else:
        v = "UNINFORMATIVE_INSTRUMENT"
    return {"features": label, "rows": rows, "m": m, "bonferroni_alpha": thr, "n_perm": n_perm, "verdict": v,
            "null_sets": [r["layers"] for r in rows if r["verdict"] == "NULL"],
            "confounded_sets": [r["layers"] for r in rows if r["verdict"] == "CONFOUND_SUSPECTED"],
            "covariate_only_accuracy": cov_only}


def fitted_only_acc(D, spec, y, groups, nt, NP):
    """Lyra's '-fitted >= R0' condition: the part of the primary features that FWL removes (Z b, with b fitted on the
    training fold), classified alone with the same folds and classifier."""
    Kx, Px = cell_arrays(D, spec)
    Z = covariates(nt, NP[spec["pool"]], spec["fwl"])
    p = np.full(len(y), -1)
    for tr, te in gkf(spec["folds"]).split(np.zeros(len(y)), y, groups):
        Xtr, Xte = r2_features(Kx, Px, y, tr, te, spec["sim"])
        b = np.linalg.lstsq(Z[tr], Xtr, rcond=None)[0]
        p[te] = scale_fit(Z[tr] @ b, Z[te] @ b, y[tr], False)
    return float(np.mean(p == y))


def raw_resid_cell(R, src, y, groups, nt, NP):
    """Agni round 2 (analysis N3), exploratory: LR on the raw pooled residual itself (model layers 3/7/11/15,
    positions >= 50), same folds, FWL and classifier. Does the W_K/cache route add anything over the residual?"""
    X = R[src][:, LIX(R1_LAYERS), :].reshape(len(y), -1)
    pred = cv(lambda tr, te: (X[tr], X[te]), y, groups, covariates(nt, NP["mean_skip50"], "pooled"), 10, False)
    return {"src": src, "layers": "v2_L3-L15", "pool": "mean_skip50", "folds": 10,
            "accuracy": float(np.mean(pred == y)), "pred": pred.astype(int).tolist()}


def r2_verdict(k2, k0, n, mc, ci_lo, perm_p, null_mean, pipeline_ok, confound, power_reg, void=False):
    """The registered ladder (prereg §5.2), first match wins. k2, k0: correct counts of R2 and R0 (10 folds).
    power_reg: the registered power at +3 points under the a priori independence psi (Lyra, 2026-10-19)."""
    floor = int(np.ceil(FLOOR_PTS / 100 * n - 1e-9))              # 3.0 points = 27 of 900 trials
    d, sig = k2 - k0, mc["p_exact_two_sided"] < ALPHA
    if void:
        return "VOID_NAN_FEATURES"
    if not pipeline_ok:
        return "UNINFORMATIVE_PIPELINE"
    if null_mean is not None and null_mean > 2 * CHANCE:
        return "LEAK_SUSPECTED_PERMUTATION_NULL_ABOVE_2X_CHANCE"
    if confound:
        return "CONFOUND_SUSPECTED"
    if k2 / n <= 2 * CHANCE:
        return "FAIL_WITHIN_2X_CHANCE"
    if perm_p is None or perm_p >= ALPHA:
        return "NOT_ESTABLISHED_PERMUTATION"
    if d <= 0:
        return "FAIL_BELOW_TEXT_TFIDF" if sig else "NOT_ABOVE_TEXT_TFIDF"
    if d >= floor and sig and ci_lo > 0:
        return "EXCEEDS_TEXT_TFIDF"
    why = [w for w, c in (("below_3pt_floor", d < floor), ("cell_ci_includes_0", sig and ci_lo <= 0),
                          ("underpowered", not sig and (power_reg is None or power_reg < POWER_MIN))) if c]
    return "INCONCLUSIVE:" + ",".join(why) if why else "NOT_DISTINGUISHED_FROM_TEXT_TFIDF"


def mcnemar(ca, cb):
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    return {"acc_a": float(ca.mean()), "acc_b": float(cb.mean()), "delta": float(ca.mean() - cb.mean()),
            "a_right_b_wrong": b, "a_wrong_b_right": c,
            "p_exact_two_sided": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def mcnemar_power(c0, delta, psi, B, seed):
    """P(exact McNemar p < .05 with R2 ahead), R2's correctness drawn conditional on R0's actual vector c0, so that
    P(R2 right, R0 wrong) - P(R2 wrong, R0 right) = delta and their sum = psi. psi is clamped to what R0's accuracy
    allows (delta <= psi <= min(2*a0 + delta, 2*(1 - a0) - delta)), so the power is always defined."""
    c0 = np.asarray(c0, bool)
    a0, n = c0.mean(), len(c0)
    psi = min(max(psi, delta), 2 * a0 + delta, 2 * (1 - a0) - delta)
    p01, p10 = (psi + delta) / 2, (psi - delta) / 2
    rng = np.random.default_rng(seed)
    U = rng.random((B, n))
    c2 = np.where(c0[None, :], U >= p10 / a0, U < p01 / (1 - a0))
    b, c = (c2 & ~c0).sum(1), (~c2 & c0).sum(1)
    p = np.minimum(1.0, 2 * binom.cdf(np.minimum(b, c), b + c, 0.5))
    p[b + c == 0] = 1.0
    return float(np.mean((p < ALPHA) & (b > c)))


def mde_observed(D, n):
    """Smallest net gain that would have rejected, given D discordant pairs."""
    for b in range(D // 2 + 1, D + 1):
        if binomtest(b, D, 0.5).pvalue < ALPHA:
            return (2 * b - D) / n
    return None


def boot_acc(ca, cb, clusters, B, seed):
    d = ca.astype(float) - cb.astype(float)
    _, inv = np.unique(clusters, return_inverse=True)
    s, m = np.bincount(inv, d), np.bincount(inv)
    ix = np.random.default_rng(seed).integers(0, len(s), (B, len(s)))
    bs = s[ix].sum(1) / m[ix].sum(1)
    return {"ci95": np.percentile(bs, [2.5, 97.5]).tolist(), "B": B, "n_clusters": len(s)}


def _auc(y, p):
    r = rankdata(p)
    m = y.sum()
    return (r[y == 1].sum() - m * (m + 1) / 2) / (m * (len(y) - m))


def delong(y, pa, pb):
    """Paired DeLong (Sun & Xu 2014 structural components)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, V10, V01 = [], [], []
    for p in (pa, pb):
        tz, tx, ty = rankdata(p), rankdata(p[pos]), rankdata(p[neg])
        aucs.append((tz[pos].sum() - m * (m + 1) / 2) / (m * n))
        V10.append((tz[pos] - tx) / n)
        V01.append(1 - (tz[neg] - ty) / m)
    Sm = np.cov(np.array(V10)) / m + np.cov(np.array(V01)) / n
    var = Sm[0, 0] + Sm[1, 1] - 2 * Sm[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": float(aucs[0] - aucs[1]), "z": float(z),
            "p_two_sided": float(2 * norm.sf(abs(z)))}


def boot_auc(y, pa, pb, clusters, B, seed):
    u, inv = np.unique(clusters, return_inverse=True)
    mem = [np.where(inv == g)[0] for g in range(len(u))]
    rng, out = np.random.default_rng(seed), []
    for _ in range(B):
        sel = np.concatenate([mem[g] for g in rng.integers(0, len(u), len(u))])
        if 0 < y[sel].sum() < len(sel):
            out.append(_auc(y[sel], pa[sel]) - _auc(y[sel], pb[sel]))
    return {"ci95": np.percentile(out, [2.5, 97.5]).tolist(), "B": B, "B_used": len(out), "n_clusters": len(u)}


def holm(named):
    keys = sorted(named, key=lambda k: named[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * named[k]))
        out[k] = run
    return out


def fingerprint_verdict(fp, alt):
    """Two parts (Agni round 2, encode N2). Norms: IN band iff every fp16-now / fp16-May ratio is within the dtype
    effect at fixed weights (|bf16-now / fp16-now - 1|, floor 1e-3); this part is set mostly by k_norm's weights and
    is insensitive to most weight changes. Witness: the number of May story tokens outside this model's fp16
    temperature-0.7 / top-p-0.9 nucleus must not exceed the number whose membership flips between bf16 and fp16 at
    fixed weights; this part is weight-sensitive. IN_BAND needs both. VOID if a row is void or a pass is missing."""
    if not isinstance(fp, dict) or not isinstance(fp.get("bf16"), list):
        return {"verdict": "NOT_RUN"}
    rb, rh, wb, wh = fp["bf16"], fp.get(alt), fp.get("bf16_witness"), fp.get(alt + "_witness")
    if (not all(isinstance(x, list) and x for x in (rh, wb, wh)) or any(r["void"] for r in rb + rh)
            or not all(w["retok_matches_n_generated"] for w in wb + wh)):
        return {"verdict": "VOID", "reason": "a pass is missing, a token count or layer list differs from May's, "
                                             "or a story does not re-tokenise to May's length"}
    drift = max(abs(a / b - 1) for r in rh for a, b in zip(r["norms_now"], r["norms_may"]))
    spread = max(abs(a / b - 1) for r1, r2 in zip(rb, rh) for a, b in zip(r1["norms_now"], r2["norms_now"]))
    tol = max(spread, 1e-3)
    n_out = sum(w["n_outside"] for w in wh)
    n_flip = sum(sum(x != y for x, y in zip(a["inside"], b["inside"])) for a, b in zip(wb, wh))
    norms_ok, wit_ok = drift <= tol, n_out <= n_flip
    neg = sum(w["n_outside_shuffled"] for w in wh) / max(sum(w["n_story_tokens"] for w in wh), 1)
    if neg < WITNESS_NEG_MIN:                 # Agni r3 New-5: the witness cannot tell samples from non-samples
        return {"verdict": "NORMS_ONLY_WITNESS_UNINFORMATIVE" if norms_ok else "OUT_OF_BAND", "norms_ok": norms_ok,
                "witness_negative_control_outside_frac": neg, "max_drift_fp16_now_vs_may": drift, "tolerance": tol}
    return {"verdict": "IN_BAND" if norms_ok and wit_ok else "OUT_OF_BAND", "norms_ok": norms_ok,
            "witness_negative_control_outside_frac": neg,
            "reference": "May's bridge run (default HF cache), not v2's cache: it does not gate R1 (prereg §5.1)",
            "witness_ok": wit_ok, "max_drift_fp16_now_vs_may": drift, "dtype_spread_bf16_vs_fp16_now": spread,
            "tolerance": tol, "witness_tokens": sum(w["n_story_tokens"] for w in wh),
            "witness_outside_fp16": n_out, "witness_flips_bf16_vs_fp16": n_flip,
            "note": "the norm part is insensitive to weight changes that keep k_norm; the witness is the sensitive part"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_enc(enc, verify):
    enc = Path(enc)
    man = json.load(open(enc / "manifest.json"))
    assert man.get("complete"), "encode incomplete; refusing to analyse a partial run"
    assert man["model"]["full_attention_layers"] == FULL, "manifest layer list differs from FULL"
    recs = man["trials"]
    assert [r["trial"] for r in recs] == list(range(len(recs))), "manifest trials out of order"
    with np.load(enc / "wk.npz") as zf:
        wk = {k: zf[k] for k in ("layers", "W_K")}
    assert wk["layers"].tolist() == FULL and wk["W_K"].shape[0] == 16, "wk.npz layers"
    WT = np.ascontiguousarray(wk["W_K"].astype(np.float32).transpose(0, 2, 1))
    n, K, P, S, S100, nt = len(recs), None, {}, None, None, np.zeros(len(recs))
    R, ps = {}, POOLS.index("mean_skip50")
    for i, r in enumerate(recs):
        f = enc / "shards" / r["shard"]
        if verify:
            assert sha(f) == r["shard_sha256"], f"{f}: sha256 differs from manifest"
        with np.load(f) as zf:
            z = {k: zf[k] for k in ("layers", "keys_pool", "spectral", "spectral_n100", "n_tokens", *SRCS)}
        assert z["layers"].tolist() == FULL, f"{f}: layer ids"
        if K is None:
            assert z["keys_pool"].shape == (16, len(POOLS), WT.shape[2]), f"keys_pool shape {z['keys_pool'].shape}"
            K = np.zeros((n,) + z["keys_pool"].shape, np.float32)
            S, S100 = np.zeros((n, 16, 4), np.float32), np.zeros((n, 16, 4), np.float32)
            P = {s: np.zeros((n, 16, len(POOLS), WT.shape[2]), np.float32) for s in SRCS}
            R = {s: np.zeros((n, 16, WT.shape[1]), np.float32) for s in ("res_out", "res_in_ln")}
        assert z["res_out"].shape == (16, len(POOLS), WT.shape[1]), f"{f}: residual shape"
        K[i], S[i], S100[i], nt[i] = z["keys_pool"], z["spectral"], z["spectral_n100"], int(z["n_tokens"])
        for s in SRCS:
            P[s][i] = np.matmul(z[s], WT)                              # W_K r, per layer and pool
        for s in R:
            R[s][i] = z[s][:, ps, :]                                   # raw residual, positions >= 50 (baseline)
    WK4 = np.ascontiguousarray(WT[LIX(R1_LAYERS)].transpose(0, 2, 1))           # v1's W_K at layers 3/7/11/15
    return man, recs, {"K": K, "P": P, "R": R, "WK4": WK4}, S, S100, nt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--r0", required=True)
    ap.add_argument("--prereg", required=True, help="prereg_um_v2_R1R2.json: hashed, and its registered_constants "
                    "must equal this script's")
    ap.add_argument("--prediction", required=True, help="the sealed R1R2_PREDICTION.md; its sha256 must equal the "
                    "prereg's prediction_sha256 and the encode manifest's")
    ap.add_argument("--v2-results", required=True, help="v2's wk_backtrack_v2_results.json (368 is read from it)")
    ap.add_argument("--v1-results", required=True, help="v1's wk_backtrack_results.json (470 is read from it)")
    ap.add_argument("--v1-vectors", required=True, help="May's emotion_vectors.json, which v1 projected")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--n-perm-r7", type=int, default=N_PERM_R7_DEFAULT, help="0 skips R7's permutations (dry runs)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--n-power", type=int, default=4000)
    ap.add_argument("--r1-band-reps", type=int, default=30)
    ap.add_argument("--expect-n", type=int, default=N_TRIALS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-verify-sha", dest="verify", action="store_false")
    a = ap.parse_args(argv)
    t0 = time.time()
    prereg = json.load(open(a.prereg, encoding="utf-8"))
    reg = check_registered(prereg)                                   # Agni r3 New-9
    check_prediction(a.prediction, reg.get("prediction_sha256"))   # Agni r3 encode N9; the seal (sign-off)
    v2res = hashed_json(a.v2_results, reg["v2_results_sha256"], "v2 results")
    v1res = hashed_json(a.v1_results, reg["v1_results_sha256"], "v1 results")
    assert count_from(v2res, "W_K mean-pos", N_TRIALS) == PAPER_K, "368 is not what v2's file says"
    assert count_from(v1res, "W_K projection", N_TRIALS) == V1_K, "470 is not what v1's file says"
    raw = json.load(open(a.trials, encoding="utf-8"))
    trials = raw["trials"] if isinstance(raw, dict) and "trials" in raw else raw
    man, recs, D, S, S100, nt = load_enc(a.enc, a.verify)
    n = len(recs)
    assert n == len(trials) == a.expect_n, f"n: encode {n}, trials {len(trials)}, expected {a.expect_n}"
    for i, (r, t) in enumerate(zip(recs, trials)):
        assert r["story_sha256"] == hashlib.sha256(t["story_text"].encode("utf-8")).hexdigest(), f"trial {i} text"
        assert (r["emotion"], r["topic_idx"]) == (t["emotion"], t["topic_idx"]), f"trial {i} labels"
    pred_sha = sha(a.prediction)
    assert man["prediction"]["sha256"] == pred_sha, "the prediction file is not the one the encode recorded"
    assert man["spectral_feature_names"] == SPECTRAL_NAMES, "spectral column order"           # Agni r3 New-9
    # round 2 (encode N4, analysis IDENTITY_BY_DEFAULT): the mandatory encode checks must be present, not defaulted
    det, fpd = man["determinism"], man["fingerprint"]
    assert isinstance(fpd.get("bf16"), list) and isinstance(fpd.get("bf16_witness"), list), "fingerprint skipped"
    alt_dt = man["args"]["alt_dtype"]
    alt_key = f"trial0_keys_pool_{alt_dt}_vs_{man['args']['dtype']}"
    assert alt_dt and alt_key in fpd, "the fp16 numeric-path check is missing"
    rep_keys = ("back_to_back", "unhooked_vs_hooked", "post_loop_vs_shard")
    worst = max(v for k in rep_keys for arr in det[k].values() for v in arr["rms_rel_per_layer"])
    if worst > DET_CEIL:                                             # Agni r3 encode N6: the encode is void
        raise SystemExit(f"determinism: per-layer relative RMS {worst:.3g} > ceiling {DET_CEIL:.3g}; encode void")
    assert np.linalg.norm(D["K"], axis=-1).min() > 0, "an all-zero key vector (v2:153-160's failure mode)"
    for s_ in SRCS:
        assert np.linalg.norm(D["P"][s_], axis=-1).min() > 0, f"a zero W_K-projected {s_} (would become NaN)"
    emotions = [r["emotion"] for r in recs]
    groups = np.array([r["topic_idx"] for r in recs])
    ea = sorted(set(emotions))
    assert len(ea) == N_EMO, f"{len(ea)} emotions"
    assert set(Counter(zip(emotions, groups.tolist())).values()) == {3}, "every (emotion, topic) cell must hold 3"
    n_fall = sum(r["skip50_fallback"] for r in recs)
    assert n_fall == 0, f"{n_fall} trials have T <= 50; the measured lengths said none, so the input changed"
    cells = np.array([f"{e}|{g}" for e, g in zip(emotions, groups)])
    cell_id = np.unique(cells, return_inverse=True)[1]
    y = np.array([ea.index(e) for e in emotions])
    vv, yv = valence_subset(emotions)
    vv = np.array(vv)
    NP = {"mean_all": nt, "mean_skip50": nt - 50, "last": np.ones(n),
          "story": np.array([r["story_tok_end"] - r["story_tok_start"] for r in recs], float)}
    assert NP["story"].min() >= 100, "a story under 100 tokens: spectral_n100 would not be n-matched"
    texts = [t["story_text"] for t in trials]
    print(f"loaded {n} trials, {len(ea)} emotions, {time.time() - t0:.0f}s", flush=True)

    r0f = json.load(open(a.r0))["all_trials"]
    r0_file = np.array(r0f["thirty_way"]["per_trial_pred"])
    assert len(r0_file) == n, "R0 file length"
    r0v_file = r0f["valence"]["per_trial"]
    assert [d["trial"] for d in r0v_file] == vv.tolist() and [d["y"] for d in r0v_file] == yv.tolist()
    r0v_file = np.array([d["p"] for d in r0v_file])
    for k in (5, 10):   # a SELF-CHECK only (Agni r3 New-6): unshuffled GroupKFold ignores X and y, so this cannot
        # fail; it does not show R0's original run used these folds. R0_REPRODUCED below is the evidence for that.
        assert np.array_equal(fold_ids(groups, k, texts, emotions), fold_ids(groups, k)), f"fold maps differ at {k}"
    R0 = {k: np.array(r0_thirty(texts, emotions, groups, k)) for k in (5, 10)}
    R0v = {k: r0_valence(texts, emotions, groups, k) for k in (5, 10)}
    agree = float(np.mean(R0[5] == r0_file))
    vdiff = float(np.abs(R0v[5] - r0v_file).max())
    r0_check = {"agreement_5fold_vs_file": agree, "valence_max_abs_diff_5fold_vs_file": vdiff,
                "r0_file_sklearn_version": "not recorded in R0_results.json",
                "verdict": "R0_REPRODUCED" if agree == 1.0 and vdiff <= 1e-6 else "R0_MISMATCH"}   # round 2, N8
    print("R0 check", r0_check, flush=True)
    E = np.array(emotions)
    right = lambda pred: np.asarray(pred) == E

    acts = D["K"][:, LIX(R1_LAYERS), POOLS.index("mean_all"), :]
    R1 = {"mean": r1(acts.reshape(n, -1), nt, emotions, groups),
          "last": r1(D["K"][:, LIX(R1_LAYERS), POOLS.index("last"), :].reshape(n, -1), nt, emotions, groups),
          "mean_10fold": r1(acts.reshape(n, -1), nt, emotions, groups, 10),
          "mean_trainFWL": r1(acts.reshape(n, -1), nt, emotions, groups, 5, "train")}
    # Agni r3 (encode N1, analysis New-1): v2's 368 came from bf16 on MPS (v2:107-108), so the band uses bf16
    # run-to-run repeats only (the largest of the three checks per layer). fp16 does not enter it. h is capped.
    rr = [max(det[k]["keys_pool"]["rms_rel_per_layer"][j] for k in rep_keys) for j in range(16)]
    delta = [rr[j] for j in LIX(R1_LAYERS)]
    band = r1_band(acts, nt, emotions, groups, delta, a.r1_band_reps, a.seed, a.jobs)
    assert band["k_unperturbed"] == R1["mean"]["k_correct"], "band baseline differs from R1"     # Agni r3 New-1
    band.update(run_to_run_per_layer=delta, h_max=H_MAX, source="bf16 run-to-run only (v2 ran in bf16, v2:107)")
    vec = json.load(open(a.v1_vectors, encoding="utf-8"))
    assert sha(a.v1_vectors) == reg["v1_vectors_sha256"], "May's emotion_vectors.json is not the registered one"
    v1 = v1_check(acts, D["WK4"], [vec[str(l)] for l in R1_LAYERS], list(vec["0"].keys()),
                  np.array([t.get("story_length", 100) for t in trials], float), emotions, groups, V1_K, delta,
                  a.r1_band_reps, a.seed, a.jobs)                  # v1:167 reads story_length with that default
    del vec
    r1v = r1_verdict(R1["mean"]["k_correct"], PAPER_K, band["half_width_trials"], v1["verdict"])
    fpv = fingerprint_verdict(fpd, alt_dt)
    print(f"R1 {R1['mean']['k_correct']}/{n} (paper {PAPER_K}), h={band['half_width_trials']} (cap {H_MAX}); "
          f"v1 {v1['k']}/{n} (May {V1_K}) {v1['verdict']} -> {r1v}; fingerprint {fpv['verdict']}", flush=True)

    reg_specs, mech = specs()
    out = Parallel(n_jobs=a.jobs)(delayed(r2_cell)(D, s, y, groups, nt, NP, vv, yv) for s in reg_specs + mech)
    grid, mgrid = out[:len(reg_specs)], out[len(reg_specs):]
    prim = next(c for c in grid if c["spec"] == PRIMARY)
    rawres = Parallel(n_jobs=a.jobs)(delayed(raw_resid_cell)(D["R"], s_, y, groups, nt, NP)
                                     for s_ in ("res_out", "res_in_ln"))
    print(f"R2 primary {prim['accuracy']:.4f}, {len(out)} cells, {time.time() - t0:.0f}s", flush=True)
    topics, emo_groups = groups.copy(), y.copy()     # N10: the same probe on topic, grouped by emotion
    ctrl = r2_cell(D, PRIMARY, topics, emo_groups, nt, NP, vv, yv, valence=False)
    pipeline_ok = ctrl["accuracy"] > 2 / len(np.unique(topics))
    if 0 < a.n_perm and 1 / (a.n_perm + 1) >= ALPHA:
        raise SystemExit(f"{a.n_perm} permutations cannot reach p < {ALPHA} (SOP §4.11)")
    pnull = Parallel(n_jobs=a.jobs)(delayed(r2_cell)(D, PRIMARY, y, groups, nt, NP, vv, yv,
                                                     perm_within(y, groups, np.random.default_rng(a.seed + b), cell_id))
                                    for b in range(a.n_perm))
    perm = {"n_perm": a.n_perm, "scheme": "the 30 emotion labels permuted over the 30 cells within each topic",
            "statistic": "R2 primary's own 30-way accuracy, the whole pipeline refit per draw (directions "
                         "re-estimated; no TF-IDF involved). It tests R2 against chance and leakage, NOT R2 > R0.",
            "n_distinct": len(set(pnull)), "null": [round(x, 6) for x in pnull],
            "null_mean": float(np.mean(pnull)) if pnull else None,
            "null_p95": float(np.percentile(pnull, 95)) if pnull else None,
            "p": float((1 + sum(x >= prim["accuracy"] for x in pnull)) / (1 + len(pnull))) if pnull else None}

    p_ok, r0_ok = right(np.array(ea)[prim["pred"]]), right(R0[10])
    mc = mcnemar(p_ok, r0_ok)
    rob = {"estimand": "accuracy(R2) - accuracy(R0, 10 folds), observed delta with percentile intervals",
           "cell": boot_acc(p_ok, r0_ok, cells, a.n_boot, a.seed), "topic": boot_acc(p_ok, r0_ok, groups, a.n_boot,
                                                                                       a.seed),
           "sign_flip_cells": sign_flip(p_ok, r0_ok, cell_id, a.n_boot, a.seed)}
    c1, c0f = right(R1["mean"]["pred"]), right(r0_file)
    a0 = float(r0_ok.mean())
    # Lyra, 2026-10-19 (superseding their 10-18 choice): the registered psi is the a priori independence value from
    # R0's accuracy alone, and "underpowered" is gated on the +3-point power (Agni r3 encode N2, analysis New-2).
    psi_ind = lambda d: (a0 + d) * (1 - a0) + a0 * (1 - a0 - d)
    sens = {"sensitivity_R0_5fold_vs_10fold": float(np.mean(right(R0[5]) != r0_ok)),
            "sensitivity_R1_vs_R0file": float(np.mean(c1 != c0f))}
    power = {f"+{int(d * 100)}pt": {k: mcnemar_power(r0_ok, d, v, a.n_power, a.seed + 7)
                                    for k, v in dict(registered_independence=psi_ind(d), **sens).items()}
             for d in (0.03, 0.05)}
    reg3 = power["+3pt"]["registered_independence"]
    power.update(psi={"registered_independence_at_3pt": psi_ind(0.03), **sens}, registered_power_at_3pt=reg3,
                 mde_observed_discordants=mde_observed(mc["a_right_b_wrong"] + mc["a_wrong_b_right"], n),
                 note="power of the exact McNemar on prediction-level gains, not of a latent cache signal; McNemar "
                      "also treats 900 trials as independent while the unit is 300 cells (optimistic)")
    cov_only = {p: covariate_only_acc(y, groups, covariates(nt, NP[p], "pooled"), 10) for p in POOLS}
    fit_only = fitted_only_acc(D, PRIMARY, y, groups, nt, NP)
    fd = prim["fwl_diagnostics"]
    why = [w for w, c in (("covariates_alone_reach_R0", cov_only[PRIMARY["pool"]] >= a0),
                          ("fitted_part_reaches_R0", fit_only >= a0),
                          ("tie_fraction_above_0.5", fd["max_tie_fraction"] > TIE_MAX),
                          ("variance_kept_below_0.1", fd["min_var_kept"] < VARKEPT_MIN)) if c]
    confound = {"suspected": bool(why), "reasons": why, "covariate_only_accuracy": cov_only[PRIMARY["pool"]],
                "fitted_only_accuracy": fit_only, "R0_10fold_accuracy": a0, "tie_max": TIE_MAX,
                "var_kept_min": VARKEPT_MIN, "diagnostics": fd,
                "flag_covariates_above_2x_chance": {p: v > 2 * CHANCE for p, v in cov_only.items()}}   # Agni r3 New-10
    v2v = r2_verdict(int(p_ok.sum()), int(r0_ok.sum()), n, mc, rob["cell"]["ci95"][0], perm["p"], perm["null_mean"],
                     pipeline_ok, confound["suspected"], reg3, void=prim["void"])

    sec = {"S1_R1_vs_R0file_30way": mcnemar(c1, c0f),
           "S2_R1_vs_R0file_valence": delong(yv, np.array(R1["mean"]["valence_p"]), r0v_file),
           "S3_R2primary_vs_R0_10fold_valence": delong(yv, np.array(prim["valence_p"]), R0v[10]),
           "S4_R2primary_vs_R0file_30way": dict(mcnemar(p_ok, c0f), note="R2 trains on 90% (10 folds), the R0 file "
                                                "on 80% (5 folds): a training-set asymmetry in R2's favour")}
    adj = holm({k: v.get("p_exact_two_sided", v.get("p_two_sided")) for k, v in sec.items()})
    for k in sec:
        sec[k]["p_holm"] = adj[k]
    for k in ("S1_R1_vs_R0file_30way", "S2_R1_vs_R0file_valence"):
        sec[k]["fwl_note"] = "R1 carries v2's FWL: fitted on each fold's own test trials (v2:227), label-free"
    sec["S2_R1_vs_R0file_valence"]["boot_cell"] = boot_auc(yv, np.array(R1["mean"]["valence_p"]), r0v_file,
                                                           cells[vv], a.n_boot, a.seed)
    sec["S3_R2primary_vs_R0_10fold_valence"]["boot_cell"] = boot_auc(yv, np.array(prim["valence_p"]), R0v[10],
                                                                     cells[vv], a.n_boot, a.seed)
    lp = np.zeros(len(yv))                                        # length baseline for R2's valence, 10 folds
    for tr, te in gkf(10).split(np.zeros(len(vv)), yv, groups[vv]):
        lp[te] = LogisticRegression(max_iter=1000).fit(nt[vv][tr, None], yv[tr]).predict_proba(nt[vv][te, None])[:, 1]
    len2 = float(roc_auc_score(yv, lp))
    valence_v = {"R1": "EXPLAINED_BY_LENGTH" if R1["mean"]["length_baseline_auroc"] >= R1["mean"]["valence_auroc"]
                 - LEN_MARGIN else "NOT_EXPLAINED_BY_LENGTH",
                 "R2": "EXPLAINED_BY_LENGTH" if len2 >= prim["valence_auroc"] - LEN_MARGIN else "NOT_EXPLAINED_BY_LENGTH",
                 "R2_length_baseline_auroc": len2}
    r0t_pred, r0t_C = r0_tuned(texts, emotions, groups)           # Agni r3 New-4, exploratory
    r0t = {"accuracy": float(np.mean(right(r0t_pred))), "chosen_C": r0t_C, "pred": r0t_pred,
           "R2primary_vs_this": mcnemar(p_ok, right(r0t_pred))}
    expl = {"R2primary_vs_R1_10fold": mcnemar(p_ok, right(R1["mean_10fold"]["pred"])),
            "R1_vs_R0_10fold": mcnemar(right(R1["mean_10fold"]["pred"]), r0_ok), "R0_tuned_C_EXPLORATORY": r0t}
    for rr_ in rawres:
        rr_["vs_R0_10fold"] = mcnemar(right(np.array(ea)[rr_["pred"]]), r0_ok)
        rr_["R2primary_vs_this"] = mcnemar(p_ok, right(np.array(ea)[rr_["pred"]]))
    ok_grid = [c for c in grid if not c["void"]]
    above = [c["accuracy"] > np.mean(right(R0[c["spec"]["folds"]])) for c in ok_grid]
    accs = np.array([c["accuracy"] for c in ok_grid])
    curve = {"n_cells": len(grid), "n_void_nan": len(grid) - len(ok_grid),
             "n_above_fold_matched_text_DESCRIPTIVE_UNCORRECTED": int(sum(above)),
             "median": float(np.median(accs)), "min": float(accs.min()), "max": float(accs.max()),
             "argmax_spec_EXPLORATORY": ok_grid[int(accs.argmax())]["spec"],
             "last_pool_cells_above_covariate_only": int(sum(c["accuracy"] > cov_only["last"] for c in ok_grid
                                                             if c["spec"]["pool"] == "last"))}
    Zr7 = covariates(nt, None, "paper")
    cov_r7 = covariate_only_acc(y, groups, Zr7, 10)
    R7 = r7(S100, y, groups, Zr7, cell_id, a.n_perm_r7, a.seed, a.jobs, "spectral_n100 (first 100 story tokens)",
            cov_only=cov_r7)
    R7x = r7(S, y, groups, covariates(nt, nt, "pooled"), cell_id, 0, a.seed, a.jobs, "spectral, all positions")
    # the pre-written outcome sentences (prereg §5.6, Agni r3 analysis CONTROL_WITHOUT_BRANCH), emitted when they apply
    sentences = []
    for rr_ in rawres:
        if rr_["accuracy"] >= prim["accuracy"]:
            sentences.append(S_RAW.format(src=rr_["src"], acc=rr_["accuracy"], r2=prim["accuracy"]))
    proj = next(c for c in mgrid if c["spec"] == dict(PRIMARY, src="res_in_ln", target="proj"))
    cach = next(c for c in mgrid if c["spec"] == dict(PRIMARY, src="res_in_ln", target="cache"))
    pc = mcnemar(right(np.array(ea)[proj["pred"]]), right(np.array(ea)[cach["pred"]]))
    gain = int(round(pc["delta"] * n))
    if gain >= int(np.ceil(FLOOR_PTS / 100 * n - 1e-9)) and pc["p_exact_two_sided"] < ALPHA:
        sentences.append(S_PROJ.format(gain=gain, p=pc["p_exact_two_sided"]))
    if r1v.startswith("NOT_REPRODUCED") and v2v == "EXCEEDS_TEXT_TFIDF":
        sentences.append(S_R1R2.format(r1v=r1v, d=100 * mc["delta"]))
    if r0_check["verdict"] == "R0_MISMATCH":
        sentences.append(S_R0)
    if r0t["accuracy"] >= prim["accuracy"]:
        sentences.append(S_R0T.format(acc=r0t["accuracy"], r2=prim["accuracy"]))
    folds = {k: {int(g): int(f) for g, f in zip(groups, fold_ids(groups, k))} for k in (5, 10)}
    res = {
        "meta": {"script": Path(__file__).name, "script_sha256": sha(__file__), "args": vars(a),
                 "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1),
                 "inputs_sha256": {"manifest": sha(Path(a.enc) / "manifest.json"), "trials": sha(a.trials),
                                   "r0": sha(a.r0), "prereg": sha(a.prereg), "prediction": pred_sha,
                                   "v2_results": sha(a.v2_results), "v1_results": sha(a.v1_results),
                                   "v1_vectors": reg["v1_vectors_sha256"]},
                 "registered_constants_checked": sorted(REGISTERED),
                 "encode_model": man.get("model"), "encode_versions": man.get("versions"),
                 "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                              "sklearn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__,
                              "platform": platform.platform()},
                 "topic_to_fold": folds, "primary_spec": PRIMARY, "emotions": ea,
                 "reporting_rules": "say 'text (TF-IDF)', never 'text' alone; never pair R2 EXCEEDS with R7 NULL as "
                                    "'directional, not spectral', nor R1 REPRODUCED with R2 EXCEEDS as 'the paper's "
                                    "result reproduces and beats text', nor the registered power with "
                                    "NOT_DISTINGUISHED as 'a well-powered null' (prereg §5.3)"},
        "checks": {"r0_reproduction": r0_check, "n_tokens": [int(nt.min()), int(nt.max())], "skip50_fallbacks": n_fall,
                   "determinism": det, "determinism_worst": worst, "fingerprint": fpv, "pipeline_control_topic": {
                       "accuracy": ctrl["accuracy"], "passed": bool(pipeline_ok)},
                   "covariate_only_accuracy": cov_only, "confound": confound, "v1_artifact_check": v1},
        "verdicts": {"R1": r1v, "R1_h": band["half_width_trials"], "V1_check": v1["verdict"], "R2_primary": v2v,
                     "R7": R7["verdict"], "valence_length": valence_v, "R0": r0_check["verdict"],
                     "fingerprint_vs_may_bridge": fpv["verdict"],
                     "R1_note": "R2 is read and reported whatever R1 is (prereg §5.1)"},
        "outcome_sentences": sentences,
        "R0": {"acc_file_5fold": float(np.mean(c0f)), "acc_5fold": float(np.mean(right(R0[5]))),
               "acc_10fold": a0, "valence_auroc_10fold": float(roc_auc_score(yv, R0v[10])),
               "pred_10fold": R0[10].tolist(), "valence_p_10fold": R0v[10].round(6).tolist()},
        "R1": dict(R1, band=band, paper_k=PAPER_K,
                   fwl_note="v2's FWL: fitted on each fold's own test trials (v2:227); label-free; kept for fidelity"),
        "R2": {"primary": prim, "permutation": perm, "power": power, "spec_curve": curve, "grid": grid,
               "exploratory": mgrid, "raw_residual_LR_EXPLORATORY": rawres, "proj_vs_cache_res_in_ln": pc},
        "tests": {"primary_R2_vs_R0_10fold_30way": dict(mc, robustness=rob), "secondary_holm": sec,
                  "exploratory": expl},
        "R7": R7, "R7_all_positions_EXPLORATORY": R7x,
        "per_trial": {"emotion": emotions, "topic_idx": groups.tolist(), "story_idx": [r["story_idx"] for r in recs],
                      "valence_trials": vv.tolist(), "valence_y": yv.tolist(),
                      "note": "grid preds are indices into meta.emotions; R1 preds are names"},
    }
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"verdicts": res["verdicts"], "R1_k": R1["mean"]["k_correct"], "R2": prim["accuracy"],
                      "R0_10": a0, "mcnemar": mc, "power3_registered": reg3, "confound": why,
                      "sentences": len(sentences), "seconds": res["meta"]["seconds"]}), flush=True)
    return res


if __name__ == "__main__":
    main()
