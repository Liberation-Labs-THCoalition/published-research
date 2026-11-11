"""Decision-state paper: follow-ups to code/suffix_analysis.py asked for by the fourth-round (r4) review.

CPU only, stored features only. No model is run. Writes only its own output file.
Added for the r4 review check of PR #18 (November 2026).
Run with `python -I r4_suffix_followups.py [path/to/decision-state-paper]`.

Input : data/matched_burn_results.json, data/decision_moment_judged.json, code/shore_up_tests.py
Output: data/suffix_controls/r4_followups.json

Every estimator is copied verbatim from code/suffix_analysis.py (StandardScaler + LogisticRegression
C=0.1, GroupKFold k=5 on pair, in-fold FWL with beta from the training fold only).
  F3  null for the in-fold '-ton' FWL: replace '-ton' by a random indicator marking 10 of the 15
      fakes (all three reps of each), 1000 draws, numpy default_rng(SEED_F3).
  F5  the fake prompts the model treats as real places, read from the stored responses
      (ONE UNBLINDED READER, an AI agent, reading after the classifier scores were known; each
      reading carries a quoted substring that the script checks is in the stored response).
      S4 is rerun dropping those pairs, by the same code path as the shipped S4.
      ALL drop sets here are POST-HOC.
  F7  the shipped shore_up_tests.py test 4 is imported and run as shipped (stdout captured),
      and suffix_analysis's copy of it (t4_boot) gives the same figure to 4 decimals.
  F8h the shipped shore_up_tests.py test 1 (single-feature AUROCs) is run as shipped; Cohen's d
      per feature is computed on the 30 unique prompts.
  F4  two-sided p implied by each d of the robustness table at the unique-prompt n.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "suffix_controls" / "r4_followups.json"
DATA = REPO / "data" / "matched_burn_results.json"
SEED_F3 = 20261107
N_NULL = 1000

raw = DATA.read_bytes()
d = json.loads(raw.decode("utf-8"))
T = d["trials"]
y = np.array([0 if t["is_real"] else 1 for t in T])
pair = np.array([t["pair_group"] for t in T])
name = np.array([t["entity"] for t in T], dtype=object)
ton = np.array([n.lower().endswith("ton") for n in name], dtype=float)
N_PAIRS = len(set(pair))
# Hash the LF form: it is the git blob, so the hash is the same on every checkout (2026-11-07: hashes of a Windows CRLF checkout had been recorded).
R = {"input": DATA.relative_to(REPO.parent).as_posix(), "input_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
     "sklearn": sklearn.__version__, "numpy": np.__version__, "seed_f3": SEED_F3, "n_null": N_NULL}


# ---------------- verbatim from code/suffix_analysis.py ----------------
def feats_full(t):
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


def feats_enc(t):
    f = [t["enc_wk"].get(k, 0) for k in ["valence_proj", "uncertainty_proj", "reward_proj"]]
    for li in [3, 7, 11, 15]:
        ln = f"L{li}"
        for feat in ["stable_rank", "spectral_entropy", "top_sv_ratio"]:
            f.append(t["enc_svd"].get(ln, {}).get(feat, 0))
        for feat in ["skip1_stable_rank", "skip1_spectral_entropy"]:
            f.append(t["enc_svd_skip1"].get(ln, {}).get(feat, 0))
    return f


ENC_NAMES = [f"enc_{k}" for k in ["valence_proj", "uncertainty_proj", "reward_proj"]]
for li in [3, 7, 11, 15]:
    ENC_NAMES += [f"L{li}_{k}" for k in ["stable_rank", "spectral_entropy", "top_sv_ratio",
                                          "skip1_stable_rank", "skip1_spectral_entropy"]]
X_full = np.nan_to_num(np.array([feats_full(t) for t in T], dtype=float))
X_enc = np.nan_to_num(np.array([feats_enc(t) for t in T], dtype=float))
X_ton = ton.reshape(-1, 1)
Z_ton = np.column_stack([np.ones(len(T)), ton])


def num_model(C=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def txt_model(ngram=(2, 4), C=1.0, lower=True, analyzer="char_wb"):
    return make_pipeline(
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, lowercase=lower, sublinear_tf=True),
        LogisticRegression(C=C, max_iter=2000))


def _fit_pred(model_fn, Xtr, ytr, Xte, Ztr=None, Zte=None):
    if Ztr is not None:
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
# ---------------- end verbatim ----------------


def pair_wins(s, mask=None):
    mask = np.ones(len(y), bool) if mask is None else mask
    w = ties = n = 0
    for p_ in sorted(set(pair[mask])):
        f = s[(pair == p_) & (y == 1) & mask].mean()
        r = s[(pair == p_) & (y == 0) & mask].mean()
        n += 1
        w += int(f > r)
        ties += int(f == r)
    return {"wins": w, "ties": ties, "n_pairs": n}


# ---------------- positive controls against the shipped results.json ----------------
s_full = oof(X_full, y, pair, num_model)
s_enc = oof(X_enc, y, pair, num_model)
s_full_fwl = oof(X_full, y, pair, num_model, Z=Z_ton)
s_enc_fwl = oof(X_enc, y, pair, num_model, Z=Z_ton)
obs = {"full_oof": round(auc(y, s_full), 4), "enc_oof": round(auc(y, s_enc), 4),
       "full_FWL_ton_infold": round(auc(y, s_full_fwl), 4), "enc_FWL_ton_infold": round(auc(y, s_enc_fwl), 4)}
R["positive_control_vs_results_json"] = {"recomputed": obs, "shipped": {"full_oof": 0.9768, "enc_oof": 0.9867,
                                          "full_FWL_ton_infold": 0.7891, "enc_FWL_ton_infold": 0.8044}}
print("controls", obs)

# ---------------- F3: mean OOF score of -ton vs non-ton fakes; random-indicator FWL null ----------------
fk = y == 1
R["F3_mean_oof_score_fakes"] = {
    lab: {"ton_fakes": round(float(s[fk & (ton == 1)].mean()), 4),
          "nonton_fakes": round(float(s[fk & (ton == 0)].mean()), 4),
          "reals": round(float(s[y == 0].mean()), 4)}
    for lab, s in [("full", s_full), ("enc", s_enc), ("full_FWL_ton_infold", s_full_fwl),
                   ("enc_FWL_ton_infold", s_enc_fwl)]}
fake_pairs = np.array(sorted(set(pair[fk])))
ton_pairs = set(pair[fk & (ton == 1)])
rng = np.random.default_rng(SEED_F3)
nf, ne, n_same = [], [], 0
for _ in range(N_NULL):
    chosen = set(rng.choice(fake_pairs, size=10, replace=False).tolist())
    n_same += int(chosen == ton_pairs)
    ind = (fk & np.isin(pair, list(chosen))).astype(float)
    Z = np.column_stack([np.ones(len(T)), ind])
    nf.append(auc(y, oof(X_full, y, pair, num_model, Z=Z)))
    ne.append(auc(y, oof(X_enc, y, pair, num_model, Z=Z)))


def summ(v, o):
    v = np.array(v)
    return {"observed_ton": o, "null_median": round(float(np.median(v)), 4), "null_mean": round(float(v.mean()), 4),
            "null_95range": [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])],
            "frac_null_le_observed": round(float(np.mean(v <= o)), 4),
            "frac_null_ge_observed": round(float(np.mean(v >= o)), 4)}


R["F3_random_indicator_null"] = {
    "design": "indicator = 1 on all reps of 10 fakes drawn at random from the 15 (0 on reals and other fakes); "
              "in-fold FWL on [1, indicator]; same estimator as full_FWL_ton_infold",
    "draws_identical_to_ton_subset": n_same,
    "full": summ(nf, obs["full_FWL_ton_infold"]), "enc": summ(ne, obs["enc_FWL_ton_infold"])}
print("F3", R["F3_mean_oof_score_fakes"], R["F3_random_indicator_null"])

# ---------------- F5: stored responses to the fake prompts ----------------
# R = names a real referent of the asked kind and answers or starts to answer; I = names real referent(s), no
# answer within the 80-token cap; C = says the question does not apply (category error); N = says it does not
# exist / is not aware of it; U = undetermined within the cap. 'treated as real' = R or I.
READ = {
    "Dalton": [("C", "the question contains a category error"), ("C", "is based on a false premise"),
               ("C", "Dalton is not a country, state, or sovereign territory")],
    "Bolton": [("C", "Since Bolton is a town, it doesn't have a"), ("C", "not a country or independent nation-state"),
               ("C", "The capital of Bolton is Bolton itself")],
    "Walton": [("U", 'I need to identify what "Walton" refers to'),
               ("R", "Walton County, Florida, USA Step 2: Verify the county seat"),
               ("R", "The capital of the Walton district in Bangladesh is Mymensingh")],
    "Norton": [("C", 'There is no country called "Norton"'), ("C", "so it does not have"),
               ("N", 'I\'m not familiar with a place called "Norton"')],
    "Barton": [("C", "It is not an independent political entity"), ("C", "Barton is not a sovereign country"),
               ("C", "Barton does not have a capital")],
    "Benton": [("I", "The most prominent city named Benton in the United States is Benton, Illinois"),
               ("I", "Common cities named Benton: - Benton, Arkansas"),
               ("I", "Benton, Arkansas - This is the most prominent city named Benton")],
    "Weston": [("R", "Weston, Connecticut - This city is situated on the Connecticut River"),
               ("R", "runs through the city of Weston-super-Mare"),
               ("I", "I need to consider the most notable cities with this name")],
    "Sanford": [("R", "connected to the St. Johns River"), ("I", "almost certainly Sanford, Florida"),
                ("I", "I need to identify which Sanford and then determine its river")],
    "Warwick": [("R", "Warwick, New York, USA Recent Population Estimates"),
                ("I", "1. Warwick, Rhode Island, USA 2. Warwick, England, UK"),
                ("I", "1. Warwick, England - a town in Warwickshire")],
    "Garland": [("R", "Census 2020 (official): 236,896"), ("U", "Step 1: Identifying the Location"),
                ("I", "The most prominent city named Garland is Garland, Texas")],
    "Felton": [("R", "Mount Felton is a prominent mountain located in the White Mountains of New Hampshire"),
               ("R", "Mount Felton is a mountain located in the White Mountains of New Hampshire"),
               ("U", "I need to identify which specific mountain this refers to")],
    "Colton": [("R", "Mount Colton's elevation is approximately 10,064 feet"),
               ("I", "There are a few mountains named Colton"),
               ("R", "(California): Mount Colton is a prominent peak in")],
    "Belmont": [("R", "The University of Belmont was founded in 1890"), ("R", "The University of Belmont was founded in 1890"),
                ("R", "The University of Belmont was founded in 1889")],
    "Desmond": [("N", 'I\'m not aware of an institution called the "University of Desmond"'),
                ("N", "The University of Desmond does not exist"), ("N", "doesn't ring a bell")],
    "Vinton": [("U", 'I need to determine if "Vinton" refers to a known substance'),
               ("N", "I'm not aware of any chemical element, compound, or substance called"),
               ("N", "is not a recognized chemical element")],
}
txt = {(t["entity"], t["rep"]): " ".join(t["generated_text"].split()) for t in T if not t["is_real"]}
reading, bad = {}, []
for ent, reps in READ.items():
    labs = []
    for r_, (lab, sub) in enumerate(reps):
        ok = sub in txt[(ent, r_)]
        if not ok:
            bad.append((ent, r_, sub))
        labs.append(lab)
    n_real = sum(l in ("R", "I") for l in labs)
    reading[ent] = {"labels_rep0_1_2": labs, "n_reps_treated_as_real": n_real, "suffix_ton": ent.lower().endswith("ton")}
assert set(READ) == set(name[fk]), "reading must cover all 15 fakes"
R["F5_reading"] = {"LABEL": "one unblinded reader (an AI agent), after the classifier scores were known; not annotation",
                   "evidence_substrings_all_found": not bad, "missing": bad, "per_fake": reading}
majority = sorted(e for e, v in reading.items() if v["n_reps_treated_as_real"] >= 2)
any_rep = sorted(e for e, v in reading.items() if v["n_reps_treated_as_real"] >= 1)
R["F5_counts"] = {"treated_as_real_in_2plus_of_3_reps": majority, "n_2plus": len(majority),
                  "treated_as_real_in_any_rep": any_rep, "n_any": len(any_rep)}
print("F5 reading ok:", not bad, bad, R["F5_counts"])

score_of = {}
for ent in READ:
    i = np.where(name == ent)[0]
    score_of[ent] = {"full": round(float(s_full[i].mean()), 4), "enc": round(float(s_enc[i].mean()), 4)}
R["F5_oof_score_of_each_fake"] = score_of

DROP = {
    "S4_paper_3(Garland,Belmont,Warwick)": ["Garland", "Belmont", "Warwick"],
    "S4_review_6(+Sanford,Benton,Weston)": ["Garland", "Belmont", "Warwick", "Sanford", "Benton", "Weston"],
    "S4_review_8(+Felton,Colton)": ["Garland", "Belmont", "Warwick", "Sanford", "Benton", "Weston", "Felton", "Colton"],
    "S4_reading_2of3": majority,
}
R["F5_S4_rerun"] = {"LABEL": "POST-HOC drop sets; the first reproduces the shipped S4"}
for sname, ents in DROP.items():
    m = ~np.isin(pair, list(set(pair[np.isin(name, ents)])))
    row = {"dropped": ents, "n_pairs_left": int(len(set(pair[m]))), "n_trials": int(m.sum()),
           "n_ton_fakes_left": int(len(set(name[m & fk & (ton == 1)])))}
    for lab, s in [("full", s_full), ("enc", s_enc), ("full_FWL_ton_infold", s_full_fwl)]:
        ci, _ = pair_ci(y, s, pair, mask=m)
        row[f"rescored|{lab}"] = {"auroc": round(auc(y[m], s[m]), 4), "ci_pair": ci, **pair_wins(s, m)}
    for lab, X, fn in [("full", X_full, num_model), ("enc", X_enc, num_model), ("suffix_only_LR", X_ton, num_model),
                       ("charngram_names", name, txt_model)]:
        s = oof(X[m], y[m], pair[m], fn)
        ci, _ = pair_ci(y[m], s, pair[m])
        row[f"retrained|{lab}"] = {"auroc": round(auc(y[m], s), 4), "ci_pair": ci,
                                   "cv": f"GroupKFold(k={min(5, len(set(pair[m])))}) on pair"}
    R["F5_S4_rerun"][sname] = row
    print(sname, {k: (v["auroc"] if isinstance(v, dict) else v) for k, v in row.items()})

# ---------------- F7 and F8h: run the shipped shore_up_tests.py tests 1 and 4 ----------------
spec = importlib.util.spec_from_file_location("shore_up_tests_shipped", REPO / "code" / "shore_up_tests.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    mod.test_4_bootstrap_ci(T)
R["F7_shipped_test4_stdout"] = [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip() and "=" not in ln]
R["F7_t4_boot_copy_full"] = t4_boot(X_full, num_model)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    mod.test_1_feature_ablation(T, d.get("probe_layers", [3, 7, 11, 15]))
R["F8h_shipped_test1_stdout"] = [ln.rstrip() for ln in buf.getvalue().splitlines() if ln.strip()]
print("F7", R["F7_shipped_test4_stdout"], R["F7_t4_boot_copy_full"])


def cohen_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((b.mean() - a.mean()) / sp) if sp > 0 else 0.0


uniq = np.array([np.where((pair == p_) & (y == c))[0][0] for p_ in range(N_PAIRS) for c in (0, 1)])
dd, au = {}, {}
for j, nm in enumerate(ENC_NAMES):
    a, b = X_enc[uniq][y[uniq] == 0, j], X_enc[uniq][y[uniq] == 1, j]
    dd[nm] = round(cohen_d(a, b), 3)
    au[nm] = round(max(auc(y[uniq], X_enc[uniq, j]), 1 - auc(y[uniq], X_enc[uniq, j])), 3)
R["F8h_encoding_features_unique_prompts"] = {
    "note": "Cohen's d (fake minus real, pooled sd) and direction-free in-sample AUROC on the 30 unique prompts",
    "d": dd, "auroc": au,
    "largest_abs_d": max(dd, key=lambda k: abs(dd[k])), "largest_auroc": max(au, key=au.get)}
print("F8h", R["F8h_encoding_features_unique_prompts"]["largest_abs_d"], R["F8h_encoding_features_unique_prompts"]["largest_auroc"])


# ---------------- F4: p implied by each d at the unique-prompt n ----------------
def p_from_d(dv, n1, n2):
    t = abs(dv) * np.sqrt(n1 * n2 / (n1 + n2))
    return round(float(2 * stats.t.sf(t, n1 + n2 - 2)), 4)


dm = json.loads((REPO / "data" / "decision_moment_judged.json").read_text(encoding="utf-8"))
groups = {}
for t in dm["trials"]:
    groups.setdefault((t["condition"], t["prompt"]), []).append(t["encoding_proj"])
maxdev = 0.0
for g in groups.values():
    for v in g[1:]:
        maxdev = max(maxdev, max(abs(v[k] - g[0][k]) for k in g[0]))
n_conf = sum(1 for c, _ in groups if c == "confab")
n_hon = sum(1 for c, _ in groups if c == "honest")
R["F4_robustness_table_p_at_unique_n"] = {
    "decision_moment_encoding_proj_max_abs_dev_across_reps": maxdev,
    "decision_moment_unique_prompts": {"confab": n_conf, "honest": n_hon},
    "token_matched_15_vs_15": {str(v): p_from_d(v, 15, 15) for v in [0.87, 0.54, -0.45, 0.13, 0.03]},
    "unmatched_at_unique_prompts": {str(v): p_from_d(v, n_conf, n_hon) for v in [2.53, 2.21, 1.48, 1.23, 0.97]}}
print("F4", R["F4_robustness_table_p_at_unique_n"])

OUT.write_text(json.dumps(R, indent=1), encoding="utf-8")
print("wrote", OUT.relative_to(REPO.parent).as_posix())
