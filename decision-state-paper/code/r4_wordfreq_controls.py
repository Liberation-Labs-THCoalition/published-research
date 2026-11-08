"""Decision-state paper: the word-frequency baseline for the real/fake entity names.

CPU only, stored features only. No model is run. Writes only its own output file.
Added for the fourth-round (r4) review check of PR #18 (November 2026).
Needs the `wordfreq` package (version recorded in the output). Run with
`python -I r4_wordfreq_controls.py [path/to/decision-state-paper]` from an environment that has it.

Input : data/matched_burn_results.json
Output: data/suffix_controls/r4_wordfreq.json

  Zipf  : wordfreq.zipf_frequency(entity, "en"), the entity string as stored (one value per name).
  alone : AUROC of -Zipf for the 'fake' label (direction declared before running: rarer = fake),
          on the 30 unique prompts; within-pair wins (fake rarer than its real partner); and a
          cross-validated logistic model on Zipf alone (same CV as code/suffix_analysis.py).
  covar : Zipf as an in-fold FWL covariate, Z = [1, Zipf] (and [1, Zipf, -ton]), estimator copied
          verbatim from code/suffix_analysis.py (StandardScaler + LogisticRegression C=0.1, GroupKFold
          k=5 on pair, beta from the training fold only).
  null  : the covariate run again with Zipf values permuted WITHIN each class (fakes among fakes,
          reals among reals; all reps of a name keep one value), 1000 draws, default_rng(SEED).
          This keeps the covariate's class separation and breaks its link to the names, the
          analogue for a continuous covariate of the random-subset null in r4_followups.json.
"""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
import wordfreq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "suffix_controls" / "r4_wordfreq.json"
DATA = REPO / "data" / "matched_burn_results.json"
SEED, N_NULL = 20261107, 1000
raw = DATA.read_bytes()
d = json.loads(raw.decode("utf-8"))
T = d["trials"]
y = np.array([0 if t["is_real"] else 1 for t in T])
pair = np.array([t["pair_group"] for t in T])
name = np.array([t["entity"] for t in T], dtype=object)
ton = np.array([n.lower().endswith("ton") for n in name], dtype=float)
R = {"input": DATA.relative_to(REPO.parent).as_posix(), "input_sha256": hashlib.sha256(raw).hexdigest(),
     "wordfreq": importlib.metadata.version("wordfreq"), "sklearn": sklearn.__version__, "numpy": np.__version__,
     "seed": SEED, "n_null": N_NULL, "zipf_call": 'wordfreq.zipf_frequency(entity, "en")'}


def feats_full(t):  # verbatim from code/suffix_analysis.py
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


def feats_enc(t):  # verbatim from code/suffix_analysis.py
    f = [t["enc_wk"].get(k, 0) for k in ["valence_proj", "uncertainty_proj", "reward_proj"]]
    for li in [3, 7, 11, 15]:
        ln = f"L{li}"
        for feat in ["stable_rank", "spectral_entropy", "top_sv_ratio"]:
            f.append(t["enc_svd"].get(ln, {}).get(feat, 0))
        for feat in ["skip1_stable_rank", "skip1_spectral_entropy"]:
            f.append(t["enc_svd_skip1"].get(ln, {}).get(feat, 0))
    return f


def num_model(C=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def oof(X, yv, g, Z=None):  # suffix_analysis.oof + _fit_pred, numeric branch
    p = np.full(len(yv), np.nan)
    for tr, te in GroupKFold(n_splits=min(5, len(set(g)))).split(np.zeros(len(yv)), yv, g):
        Xtr, Xte = X[tr], X[te]
        if Z is not None:
            b = np.linalg.lstsq(Z[tr], Xtr, rcond=None)[0]
            Xtr, Xte = Xtr - Z[tr] @ b, Xte - Z[te] @ b
        m = num_model()
        m.fit(Xtr, yv[tr])
        p[te] = m.predict_proba(Xte)[:, 1]
    return p


def auc(yv, s):
    return float(roc_auc_score(yv, s))


X_full = np.nan_to_num(np.array([feats_full(t) for t in T], dtype=float))
X_enc = np.nan_to_num(np.array([feats_enc(t) for t in T], dtype=float))
zipf_of = {n: float(wordfreq.zipf_frequency(n, "en")) for n in sorted(set(name))}
zipf = np.array([zipf_of[n] for n in name])
R["zipf_per_name"] = {("fake|" if not T[i]["is_real"] else "real|") + name[i]: zipf_of[name[i]]
                      for i in sorted(set(np.unique(name, return_index=True)[1]))}

uniq = np.unique(name, return_index=True)[1]
wins = ties = 0
pairs_detail = {}
for p_ in sorted(set(pair)):
    fz = zipf[(pair == p_) & (y == 1)][0]
    rz = zipf[(pair == p_) & (y == 0)][0]
    wins += int(fz < rz)
    ties += int(fz == rz)
    pairs_detail[int(p_)] = {"real": str(name[(pair == p_) & (y == 0)][0]), "fake": str(name[(pair == p_) & (y == 1)][0]),
                             "zipf_real": rz, "zipf_fake": fz}
R["zipf_alone"] = {
    "auroc_rarer_is_fake_unique_prompts": round(auc(y[uniq], -zipf[uniq]), 4),
    "within_pair_fake_rarer": wins, "within_pair_ties": ties, "n_pairs": int(len(set(pair))),
    "cv_logistic_on_zipf_oof": round(auc(y, oof(zipf.reshape(-1, 1), y, pair)), 4),
    "pairs": pairs_detail}
print("zipf alone", {k: v for k, v in R["zipf_alone"].items() if k != "pairs"})

one = np.ones(len(T))
Zz = np.column_stack([one, zipf])
Zzt = np.column_stack([one, zipf, ton])
obs = {"full_noFWL": round(auc(y, oof(X_full, y, pair)), 4), "enc_noFWL": round(auc(y, oof(X_enc, y, pair)), 4),
       "full_FWL_zipf": round(auc(y, oof(X_full, y, pair, Zz)), 4),
       "enc_FWL_zipf": round(auc(y, oof(X_enc, y, pair, Zz)), 4),
       "full_FWL_zipf_ton": round(auc(y, oof(X_full, y, pair, Zzt)), 4),
       "enc_FWL_zipf_ton": round(auc(y, oof(X_enc, y, pair, Zzt)), 4)}
R["zipf_infold_covariate"] = obs
print("covariate", obs)

rng = np.random.default_rng(SEED)
fakes = sorted(set(name[y == 1]))
reals = sorted(set(name[y == 0]))
nf, ne = [], []
for _ in range(N_NULL):
    perm = dict(zip(fakes, rng.permutation([zipf_of[n] for n in fakes])))
    perm.update(dict(zip(reals, rng.permutation([zipf_of[n] for n in reals]))))
    zp = np.array([perm[n] for n in name])
    Z = np.column_stack([one, zp])
    nf.append(auc(y, oof(X_full, y, pair, Z)))
    ne.append(auc(y, oof(X_enc, y, pair, Z)))


def summ(v, o):
    v = np.array(v)
    return {"observed": o, "null_median": round(float(np.median(v)), 4),
            "null_95range": [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])],
            "frac_null_le_observed": round(float(np.mean(v <= o)), 4)}


R["zipf_within_class_permutation_null"] = {"full": summ(nf, obs["full_FWL_zipf"]), "enc": summ(ne, obs["enc_FWL_zipf"])}
print("null", R["zipf_within_class_permutation_null"])
OUT.write_text(json.dumps(R, indent=1), encoding="utf-8")
print("wrote", OUT.relative_to(REPO.parent).as_posix())
