"""Add-on to suffix_analysis.py: the paired design's own metric.

Pooled AUROC across CV folds mixes ranking with cross-fold calibration (each
held-out template gets its own offset). The design is paired, so the natural
score is the WITHIN-PAIR win: is the fake arm scored above its own real arm?
Ties count 0.5. Sign test is one-sided, ties dropped. Same models and splits as
suffix_analysis.py. Also tries a word-frequency baseline if `wordfreq` exists.
Run from an EMPTY folder with `python -I pairwise_addon.py`.
Input : decision-state-paper/data/matched_burn_results.json
Output: decision-state-paper/data/suffix_controls/results_pairwise.json
Shipped in the repository in November 2026 (fourth review); only the input and output
paths were changed from the working copy, so it runs from any checkout.
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
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
T = json.loads((REPO / "data" / "matched_burn_results.json").read_text(encoding="utf-8"))["trials"]
y = np.array([0 if t["is_real"] else 1 for t in T])
pair = np.array([t["pair_group"] for t in T])
tmpl = np.array([t["template"] for t in T])
name = np.array([t["entity"] for t in T], dtype=object)
ton = np.array([n.lower().endswith("ton") for n in name], dtype=float).reshape(-1, 1)


def full(t):
    f = []
    for k in ["valence_proj", "uncertainty_proj", "reward_proj"]:
        f += [t["enc_wk"].get(k, 0), t["gen_wk"].get(k, 0)]
    for li in [3, 7, 11, 15]:
        ln = f"L{li}"
        for ft in ["stable_rank", "spectral_entropy", "top_sv_ratio"]:
            f += [t["enc_svd"].get(ln, {}).get(ft, 0), t["gen_svd"].get(ln, {}).get(ft, 0),
                  t["delta_svd"].get(ln, {}).get(ft, 0)]
        for ft in ["skip1_stable_rank", "skip1_spectral_entropy"]:
            f.append(t["enc_svd_skip1"].get(ln, {}).get(ft, 0))
    f += [np.mean(t["trajectory"]["logit_entropy"][:10]), np.mean(t["trajectory"]["logit_margin"][:10])]
    return f


def enc(t):
    f = [t["enc_wk"].get(k, 0) for k in ["valence_proj", "uncertainty_proj", "reward_proj"]]
    for li in [3, 7, 11, 15]:
        ln = f"L{li}"
        f += [t["enc_svd"].get(ln, {}).get(ft, 0) for ft in ["stable_rank", "spectral_entropy", "top_sv_ratio"]]
        f += [t["enc_svd_skip1"].get(ln, {}).get(ft, 0) for ft in ["skip1_stable_rank", "skip1_spectral_entropy"]]
    return f


XF = np.nan_to_num(np.array([full(t) for t in T], float))
XE = np.nan_to_num(np.array([enc(t) for t in T], float))
num = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
txt = lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True),
                            LogisticRegression(C=1.0, max_iter=2000))


def oof(X, m, yv, g, split):
    p = np.full(len(yv), np.nan)
    for tr, te in split.split(np.zeros(len(yv)), yv, g):
        mm = m()
        a, b = (list(X[tr]), list(X[te])) if X.dtype == object else (X[tr], X[te])
        mm.fit(a, yv[tr])
        p[te] = mm.predict_proba(b)[:, 1]
    return p


def wins(s, m):
    w = []
    for p in sorted(set(pair[m])):
        f = s[m & (pair == p) & (y == 1)].mean()
        r = s[m & (pair == p) & (y == 0)].mean()
        w.append(1.0 if f > r + 1e-12 else (0.5 if abs(f - r) <= 1e-12 else 0.0))
    w = np.array(w)
    k, n = int((w == 1).sum()), int((w != 0.5).sum())
    p_sign = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else float("nan")
    return {"win_rate": round(float(w.mean()), 4), "wins": k, "ties": int((w == 0.5).sum()),
            "n_pairs": len(w), "sign_test_p_one_sided": round(p_sign, 6)}


allm = np.ones(len(y), bool)
nonton = np.isin(pair, sorted(set(pair[(y == 1) & (ton[:, 0] == 0)])))
R = {}
for split_name, split, g in [("GroupKFold5_pair", GroupKFold(5), pair), ("LOTO", LeaveOneGroupOut(), tmpl)]:
    for lab, X, m in [("full", XF, num), ("enc", XE, num), ("suffix_only_LR", ton, num), ("charngram_names", name, txt)]:
        s = oof(X, m, y, g, split)
        R[f"{split_name}|{lab}"] = {"all_pairs": wins(s, allm), "nonton_pairs_rescored": wins(s, nonton)}
        print(split_name, lab, R[f"{split_name}|{lab}"])
# retrained inside the 5 non-ton pairs only (leave-one-pair-out)
for lab, X, m in [("full", XF, num), ("enc", XE, num), ("charngram_names", name, txt)]:
    s = np.full(len(y), np.nan)
    s[nonton] = oof(X[nonton], m, y[nonton], pair[nonton], GroupKFold(5))
    R[f"retrained_nonton_LOPO|{lab}"] = wins(s, nonton)
    print("retrained nonton", lab, R[f"retrained_nonton_LOPO|{lab}"])

# Word-frequency baseline (a boring explanation the stored data can test only if wordfreq is installed)
try:
    from wordfreq import zipf_frequency
    zf = np.array([zipf_frequency(n, "en") for n in name])
    R["wordfreq_zipf_auroc_fake_is_lower"] = round(float(roc_auc_score(y, -zf)), 4)
    R["wordfreq_by_name"] = {n: round(float(zipf_frequency(n, "en")), 2) for n in sorted(set(name))}
except ImportError:
    R["wordfreq_zipf_auroc_fake_is_lower"] = "NOT RUN: wordfreq not installed (no install attempted)"
print("wordfreq:", R["wordfreq_zipf_auroc_fake_is_lower"])
(OUT / "results_pairwise.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
print("wrote", OUT / "results_pairwise.json")
