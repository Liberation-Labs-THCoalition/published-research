"""Decision-state paper: is the known/unknown classifier reading the '-ton' suffix?

CPU only, stored features only. No model is run. Writes only its own output file.
Shipped in the repository in November 2026 (fourth review); only the input and output
paths were changed from the working copy, so it runs from any checkout.
Run from an EMPTY folder with `python -I suffix_analysis.py`.

Input : decision-state-paper/data/matched_burn_results.json
Output: decision-state-paper/data/suffix_controls/results.json

Scheme copied from code/shore_up_tests.py test 4 (the source of the paper's
0.9288): StandardScaler + LogisticRegression(C=0.1, max_iter=2000), GroupKFold
(k=5) on pair_group, y = 1 for the 'fake' arm. Every number below is reported
three ways where it makes sense:
  oof      : one out-of-fold run on the observed data (the point estimate)
  ci_pair  : 95% percentile CI from resampling the 15 PAIRS (2000 draws) over
             the fixed out-of-fold scores; the 3 reps of a prompt stay together
  t4_boot  : test 4's own estimator (resample trials, refit, mean of 1000),
             so the figure is comparable to the paper's 0.9288
Nulls: label flips WITHIN pairs (keeps one real + one fake per pair) and
trial-level shuffles, 200 each. Expect about 0.5.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parent.parent  # decision-state-paper/
OUT = REPO / "data" / "suffix_controls"
OUT.mkdir(exist_ok=True)
DATA = REPO / "data" / "matched_burn_results.json"

raw = DATA.read_bytes()
d = json.loads(raw.decode("utf-8"))
T = d["trials"]
y = np.array([0 if t["is_real"] else 1 for t in T])
pair = np.array([t["pair_group"] for t in T])
tmpl = np.array([t["template"] for t in T])
name = np.array([t["entity"] for t in T], dtype=object)
prompt = np.array([t["prompt"] for t in T], dtype=object)
ton = np.array([n.lower().endswith("ton") for n in name], dtype=float)
plen = np.array([t["prompt_len"] for t in T], dtype=float)
N_PAIRS = len(set(pair))
R = {"input": DATA.relative_to(REPO.parent).as_posix(), "input_sha256": hashlib.sha256(raw).hexdigest(),
     "sklearn": sklearn.__version__, "numpy": np.__version__,
     "n_trials": len(T), "n_pairs": N_PAIRS, "n_reps": d.get("n_reps")}


def feats_full(t):  # verbatim order from shore_up_tests.py test_4
    f = []
    for key in ["valence_proj", "uncertainty_proj", "reward_proj"]:
        f.append(t["enc_wk"].get(key, 0))
        f.append(t["gen_wk"].get(key, 0))
    for li in [3, 7, 11, 15]:
        ln = f"L{li}"
        for feat in ["stable_rank", "spectral_entropy", "top_sv_ratio"]:
            f.append(t["enc_svd"].get(ln, {}).get(feat, 0))
            f.append(t["gen_svd"].get(ln, {}).get(feat, 0))
            f.append(t["delta_svd"].get(ln, {}).get(feat, 0))
        for feat in ["skip1_stable_rank", "skip1_spectral_entropy"]:
            f.append(t["enc_svd_skip1"].get(ln, {}).get(feat, 0))
    f.append(np.mean(t["trajectory"]["logit_entropy"][:10]))
    f.append(np.mean(t["trajectory"]["logit_margin"][:10]))
    return f


def feats_enc(t):  # RECONSTRUCTION: test 4 minus every generation-phase feature
    f = [t["enc_wk"].get(k, 0) for k in ["valence_proj", "uncertainty_proj", "reward_proj"]]
    for li in [3, 7, 11, 15]:
        ln = f"L{li}"
        for feat in ["stable_rank", "spectral_entropy", "top_sv_ratio"]:
            f.append(t["enc_svd"].get(ln, {}).get(feat, 0))
        for feat in ["skip1_stable_rank", "skip1_spectral_entropy"]:
            f.append(t["enc_svd_skip1"].get(ln, {}).get(feat, 0))
    return f


X_full = np.nan_to_num(np.array([feats_full(t) for t in T], dtype=float))
X_enc = np.nan_to_num(np.array([feats_enc(t) for t in T], dtype=float))
X_ton = ton.reshape(-1, 1)
X_full_ton = np.column_stack([X_full, ton])
Z_ton = np.column_stack([np.ones(len(T)), ton])
R["n_features"] = {"full": X_full.shape[1], "enc": X_enc.shape[1]}

# Are encoding features identical across the 3 reps? (Agni r3 item 3)
dev = 0.0
for p in range(N_PAIRS):
    for r_ in (0, 1):
        ii = np.where((pair == p) & (y == r_))[0]
        dev = max(dev, float(np.abs(X_enc[ii] - X_enc[ii[0]]).max()))
R["enc_max_abs_dev_across_reps"] = dev


def num_model(C=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def txt_model(ngram=(2, 4), C=1.0, lower=True, analyzer="char_wb"):
    return make_pipeline(
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, lowercase=lower, sublinear_tf=True),
        LogisticRegression(C=C, max_iter=2000))


def _fit_pred(model_fn, Xtr, ytr, Xte, Ztr=None, Zte=None):
    if Ztr is not None:  # in-fold FWL: beta from the training fold only
        b = np.linalg.lstsq(Ztr, Xtr, rcond=None)[0]
        Xtr, Xte = Xtr - Ztr @ b, Xte - Zte @ b
    m = model_fn()
    if Xtr.dtype == object:
        m.fit(list(Xtr), ytr)
        return m.predict_proba(list(Xte))[:, 1]
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def oof(X, yv, g, model_fn, splitter=None, Z=None):
    splitter = splitter or GroupKFold(n_splits=min(5, len(set(g))))
    p = np.full(len(yv), np.nan)
    for tr, te in splitter.split(np.zeros(len(yv)), yv, g):
        zt = (Z[tr], Z[te]) if Z is not None else (None, None)
        p[te] = _fit_pred(model_fn, X[tr], yv[tr], X[te], *zt)
    return p


def auc(yv, s):
    return float(roc_auc_score(yv, s)) if len(set(yv)) > 1 else float("nan")


def pair_ci(yv, s, g, mask=None, B=2000, seed=7):
    mask = np.ones(len(yv), bool) if mask is None else mask
    keys = list(np.unique(g[mask]))
    idx = {k: np.where((g == k) & mask)[0] for k in keys}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        ii = np.concatenate([idx[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        if len(set(yv[ii])) > 1:
            out.append(roc_auc_score(yv[ii], s[ii]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)], len(out)


def t4_boot(X, model_fn, B=1000, Z=None):
    """shore_up_tests.py test 4, verbatim logic; exceptions are COUNTED, not hidden."""
    gkf = GroupKFold(n_splits=min(5, len(set(pair))))
    np.random.seed(42)
    out, n_exc = [], 0
    for _ in range(B):
        idx = np.random.choice(len(y), len(y), replace=True)
        X_b, y_b, g_b = X[idx], y[idx], pair[idx]
        Z_b = Z[idx] if Z is not None else None
        if len(set(y_b)) < 2 or len(set(g_b)) < 3:
            continue
        try:
            probs = np.full(len(y_b), np.nan)
            for tr, te in gkf.split(np.zeros(len(y_b)), y_b, g_b):
                zt = (Z_b[tr], Z_b[te]) if Z_b is not None else (None, None)
                probs[te] = _fit_pred(model_fn, X_b[tr], y_b[tr], X_b[te], *zt)
            v = ~np.isnan(probs)
            if np.sum(v) > 10 and len(set(y_b[v])) > 1:
                out.append(roc_auc_score(y_b[v], probs[v]))
        except Exception:
            n_exc += 1
    b = np.array(out)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": round(float(b.mean()), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "n_boot_ok": len(b), "n_exceptions": n_exc}


def nulls(X, model_fn, obs, N=200, Z=None, splitter=None, g=None):
    g = pair if g is None else g
    rng = np.random.default_rng(11)
    flip, shuf = [], []
    for _ in range(N):
        f = rng.integers(0, 2, size=N_PAIRS)
        yp = np.where(f[pair] == 1, 1 - y, y)
        flip.append(auc(yp, oof(X, yp, g, model_fn, splitter, Z)))
        ys = rng.permutation(y)
        shuf.append(auc(ys, oof(X, ys, g, model_fn, splitter, Z)))
    flip, shuf = np.array(flip), np.array(shuf)
    return {"pairflip_mean": round(float(flip.mean()), 4),
            "pairflip_95range": [round(float(v), 4) for v in np.percentile(flip, [2.5, 97.5])],
            "pairflip_p": round(float((1 + np.sum(flip >= obs)) / (1 + N)), 4),
            "shuffle_mean": round(float(shuf.mean()), 4),
            "shuffle_95range": [round(float(v), 4) for v in np.percentile(shuf, [2.5, 97.5])],
            "n_null": N}


def report(label, X, model_fn, mask_sets=None, Z=None, do_t4=False, do_null=True):
    s = oof(X, y, pair, model_fn, Z=Z)
    a = auc(y, s)
    ci, nb = pair_ci(y, s, pair)
    row = {"oof": round(a, 4), "ci_pair": ci, "n_trials": len(y), "n_pairs": N_PAIRS}
    if do_null:
        row["null"] = nulls(X, model_fn, a, Z=Z)
    if do_t4:
        row["t4_boot"] = t4_boot(X, model_fn, Z=Z)
    row["subsets_rescored"] = {}
    for sname, m in (mask_sets or {}).items():
        sci, _ = pair_ci(y, s, pair, mask=m)
        row["subsets_rescored"][sname] = {"auroc": round(auc(y[m], s[m]), 4), "ci_pair": sci,
                                          "n_trials": int(m.sum()), "n_pairs": int(len(set(pair[m]))),
                                          "n_fake_trials": int(y[m].sum())}
    R.setdefault("main", {})[label] = row
    print(f"{label:34s} oof={a:.4f} ci={ci}" + (f"  t4={row['t4_boot']}" if do_t4 else "")
          + (f"  null={row['null']['pairflip_mean']}/{row['null']['shuffle_mean']} p={row['null']['pairflip_p']}" if do_null else ""))
    return s


# ---------------- subsets ----------------
nonton_pairs = sorted(set(pair[(y == 1) & (ton == 0)]))
knowable_pairs = sorted(set(pair[np.isin(name, ["Garland", "Belmont", "Warwick"])]))
S = {
    "S1_reals_plus_nonton_fakes": (y == 0) | (ton == 0),
    "S2_nonton_pairs_only": np.isin(pair, nonton_pairs),
    "S3_ton_pairs_only": ~np.isin(pair, nonton_pairs),
    "S4_drop_knowable_pairs(Garland,Belmont,Warwick)": ~np.isin(pair, knowable_pairs),
}
R["nonton_fakes"] = sorted(set(name[(y == 1) & (ton == 0)]))
R["ton_fakes"] = sorted(set(name[(y == 1) & (ton == 1)]))
R["reals_ending_ton"] = sorted(set(name[(y == 0) & (ton == 1)]))

# ---------------- A. positive controls: reproduce the paper's numbers ----------------
print("A. reproduction")
R["A1_headline_t4_full"] = t4_boot(X_full, num_model)
R["A2_encoding_only_t4_reconstructed"] = t4_boot(X_enc, num_model)
Zp = np.column_stack([np.ones(len(T)), np.log1p(plen)])
Xf_fwl = np.nan_to_num(X_full - Zp @ np.linalg.lstsq(Zp, X_full, rcond=None)[0])
R["A3_matched_burn_fwl_promptlen_oof"] = round(auc(y, oof(Xf_fwl, y, pair, num_model)), 6)
R["A4_suffix_rule_by_hand"] = round(auc(y, ton), 6)
R["prompt_len_std"] = float(np.std(plen))
for k in ["A1_headline_t4_full", "A2_encoding_only_t4_reconstructed", "A3_matched_burn_fwl_promptlen_oof",
          "A4_suffix_rule_by_hand"]:
    print(" ", k, R[k])

# ---------------- B/C/D. main classifiers ----------------
print("B. main (GroupKFold-5 on pair)")
s_full = report("full_52feat(headline)", X_full, num_model, S, do_t4=False)
s_enc = report("enc_23feat(reconstructed)", X_enc, num_model, S)
s_ton = report("suffix_only_LR", X_ton, num_model, S, do_t4=True)
s_ng = report("charngram_names(char_wb2-4,C=1)", name, txt_model, S, do_t4=True)
report("charngram_prompts(char_wb2-4,C=1)", prompt, txt_model, S)
report("full+ton_feature", X_full_ton, num_model, S)
report("full_FWL_ton_infold", X_full, num_model, S, Z=Z_ton, do_t4=True)
report("enc_FWL_ton_infold", X_enc, num_model, S, Z=Z_ton, do_t4=True)
Xf_tonp = X_full - Z_ton @ np.linalg.lstsq(Z_ton, X_full, rcond=None)[0]
Xe_tonp = X_enc - Z_ton @ np.linalg.lstsq(Z_ton, X_enc, rcond=None)[0]
report("full_FWL_ton_pooled", Xf_tonp, num_model, S)
report("enc_FWL_ton_pooled", Xe_tonp, num_model, S)

# ---------------- D2. retrain inside subsets ----------------
print("D2. retrained inside subsets")
R["retrained"] = {}
for sname in ["S1_reals_plus_nonton_fakes", "S2_nonton_pairs_only", "S4_drop_knowable_pairs(Garland,Belmont,Warwick)"]:
    m = S[sname]
    for lab, X, fn in [("full", X_full, num_model), ("enc", X_enc, num_model), ("charngram_names", name, txt_model)]:
        s = oof(X[m], y[m], pair[m], fn)
        ci, _ = pair_ci(y[m], s, pair[m])
        R["retrained"][f"{sname}|{lab}"] = {"oof": round(auc(y[m], s), 4), "ci_pair": ci,
                                            "n_trials": int(m.sum()), "n_pairs": int(len(set(pair[m]))),
                                            "cv": f"GroupKFold(k={min(5, len(set(pair[m])))}) on pair"}
        print(f"  {sname}|{lab}", R["retrained"][f"{sname}|{lab}"])

# ---------------- E. leave-one-template-out ----------------
print("E. leave-one-template-out")
R["loto"] = {"n_templates": int(len(set(tmpl))),
             "pairs_per_template": {t_: int(len(set(pair[tmpl == t_]))) for t_ in sorted(set(tmpl))}}
for lab, X, fn in [("full", X_full, num_model), ("enc", X_enc, num_model),
                   ("suffix_only_LR", X_ton, num_model), ("charngram_names", name, txt_model)]:
    s = oof(X, y, tmpl, fn, splitter=LeaveOneGroupOut())
    a = auc(y, s)
    ci, _ = pair_ci(y, s, pair)
    per = {t_: round(auc(y[tmpl == t_], s[tmpl == t_]), 4) for t_ in sorted(set(tmpl))}
    nl = nulls(X, fn, a, N=100, splitter=LeaveOneGroupOut(), g=tmpl)
    R["loto"][lab] = {"oof": round(a, 4), "ci_pair": ci, "per_template_auroc": per, "null": nl}
    print(f"  {lab}: {a:.4f} {ci} null={nl['pairflip_mean']}/{nl['shuffle_mean']} p={nl['pairflip_p']}")

# ---------------- G. char n-gram sensitivity (selection-biased; primary declared above) ----------------
grid = {}
for ng in [(1, 3), (2, 4), (3, 5), (3, 3)]:
    for C in [0.1, 1.0, 10.0]:
        for lower in [True, False]:
            fn = (lambda ng=ng, C=C, lower=lower: txt_model(ng, C, lower))
            grid[f"char_wb{ng}|C={C}|lower={lower}"] = round(auc(y, oof(name, y, pair, fn)), 4)
R["charngram_grid_names"] = {"values": grid, "min": min(grid.values()), "max": max(grid.values()),
                             "note": "post-hoc sensitivity; the max is selection-biased, cite the primary"}
print("G. grid min/max", R["charngram_grid_names"]["min"], R["charngram_grid_names"]["max"])

# per-item OOF scores for audit (one row per unique prompt; reps averaged)
items = {}
for i in range(len(T)):
    k = f"{pair[i]:02d}|{'fake' if y[i] else 'real'}|{name[i]}"
    items.setdefault(k, {"full": [], "enc": [], "suffix": [], "charngram": []})
    for lab, s in [("full", s_full), ("enc", s_enc), ("suffix", s_ton), ("charngram", s_ng)]:
        items[k][lab].append(float(s[i]))
R["per_item_oof_mean"] = {k: {kk: round(float(np.mean(v)), 4) for kk, v in vv.items()} for k, vv in sorted(items.items())}

(OUT / "results.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
print("wrote", OUT / "results.json")
