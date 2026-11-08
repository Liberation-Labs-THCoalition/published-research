"""Decision-state paper: a question-text baseline for the entity-deconfound comparisons.

CPU only, stored files only. No model is run. Writes only its own output file.
Added for the fourth-round (r4) review check of PR #18 (November 2026).
Run with `python -I r4_deconfound_text_baseline.py [path/to/decision-state-paper]`.

Input : data/entity_deconfound/phase1_ground_truth.json  (the 120 questions; 30 entities x 4 conditions)
        data/entity_deconfound/phase3_results.json       (the shipped feature AUROCs, e.g. 0.794)
Output: data/entity_deconfound/text_baseline_r4.json

Text model (declared before running; the same primary text model as code/suffix_analysis.py):
  TF-IDF on character 2-4-grams within word boundaries (char_wb, lowercase, sublinear tf)
  + LogisticRegression(C=1.0, max_iter=2000), fitted on the QUESTION string only.
CV and null copied from code/entity_deconfound.py run_comparison(): GroupKFold(k=5) on
entity_idx, and a permutation null that flips both labels of a random half of the entity
groups, 1000 draws, RandomState(42), p = (count >= observed + 1) / (N + 1).
A prompt-length-only model (log1p(prompt_len), same CV) is reported beside it.
The n-gram grid at the end is a post-hoc sensitivity check; its max is selection-biased.
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
DD = REPO / "data" / "entity_deconfound"
OUT = DD / "text_baseline_r4.json"
raw = (DD / "phase1_ground_truth.json").read_bytes()
GT = json.loads(raw.decode("utf-8"))
P3 = json.loads((DD / "phase3_results.json").read_text(encoding="utf-8"))
SEED, N_PERM = 42, 1000
R = {"input": (DD / "phase1_ground_truth.json").relative_to(REPO.parent).as_posix(),
     "input_sha256": hashlib.sha256(raw).hexdigest(), "sklearn": sklearn.__version__,
     "numpy": np.__version__, "seed": SEED, "n_perm": N_PERM, "n_items": len(GT)}

# ---- what the ground-truth file holds ----
keys = sorted(set(k for t in GT for k in t))
R["phase1_fields"] = keys
R["phase1_has_correctness_field"] = any(k for k in keys if "correct" in k.lower() or "known" in k.lower()
                                        or "score" in k.lower() or "verified" in k.lower())
R["phase1_answer_max_chars"] = max(len(t["answer"]) for t in GT)
by = {}
for t in GT:
    by.setdefault(t["condition"], []).append(t)
R["conditions"] = {c: len(v) for c, v in sorted(by.items())}
R["question_form"] = {
    c: {"n_starting_What_is_the_name_of_the": sum(t["question"].startswith("What is the name of the") for t in v),
        "mean_prompt_len": round(float(np.mean([t["prompt_len"] for t in v])), 2),
        "mean_question_chars": round(float(np.mean([len(t["question"]) for t in v])), 2)}
    for c, v in sorted(by.items())}
R["reviewer_example_present_in_B_hard"] = any("oldest university library in Sweden is the Uppsala" in t["answer"]
                                               for t in by["B_hard"])


def txt_model(ngram=(2, 4), C=1.0, lower=True):
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, lowercase=lower,
                                         sublinear_tf=True), LogisticRegression(C=C, max_iter=2000))


def len_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def oof(Xin, y, g, fn):
    gkf = GroupKFold(n_splits=min(5, len(set(g))))
    p = np.full(len(y), np.nan)
    for tr, te in gkf.split(np.zeros(len(y)), y, g):
        m = fn()
        if isinstance(Xin, list):
            m.fit([Xin[i] for i in tr], y[tr])
            p[te] = m.predict_proba([Xin[i] for i in te])[:, 1]
        else:
            m.fit(Xin[tr], y[tr])
            p[te] = m.predict_proba(Xin[te])[:, 1]
    return p


def run(a, b, Xfn, fn, perm=True):
    tr = by[a] + by[b]
    y = np.array([0] * len(by[a]) + [1] * len(by[b]))
    g = np.array([t["entity_idx"] for t in tr])
    X = Xfn(tr)
    obs = float(roc_auc_score(y, oof(X, y, g, fn)))
    out = {"auroc": round(obs, 4), "n": len(y), "n_groups": int(len(set(g)))}
    if perm:
        rng = np.random.RandomState(SEED)
        ug = np.unique(g)
        ge = 0
        for _ in range(N_PERM):
            yp = y.copy()
            fl = rng.randint(0, 2, size=len(ug))
            for gi, gg in enumerate(ug):
                if fl[gi]:
                    yp[g == gg] = 1 - yp[g == gg]
            if roc_auc_score(yp, oof(X, yp, g, fn)) >= obs:
                ge += 1
        out["perm_p"] = round((ge + 1) / (N_PERM + 1), 4)
    return out


questions = lambda tr: [t["question"] for t in tr]
plen = lambda tr: np.log1p(np.array([[t["prompt_len"]] for t in tr], dtype=float))
R["comparisons"] = {}
for name, v in P3["comparisons"].items():
    a, b = name.split("_vs_")
    row = {"shipped_label": v["label"],
           "shipped_features_auroc_with_wk": round(v["with_wk"]["auroc"], 4),
           "shipped_features_auroc_without_wk": round(v["without_wk"]["auroc"], 4),
           "charngram_questions": run(a, b, questions, txt_model),
           "prompt_len_only": run(a, b, plen, len_model)}
    R["comparisons"][name] = row
    print(name, row)

grid = {}
for ng in [(1, 3), (2, 4), (3, 5), (3, 3)]:
    for C in [0.1, 1.0, 10.0]:
        fn = (lambda ng=ng, C=C: txt_model(ng, C))
        grid[f"char_wb{ng}|C={C}"] = run("B_hard", "D_complex", questions, fn, perm=False)["auroc"]
R["B_hard_vs_D_complex_charngram_grid"] = {"values": grid, "min": min(grid.values()), "max": max(grid.values()),
                                           "note": "post-hoc sensitivity; cite the declared primary"}
print("grid", R["B_hard_vs_D_complex_charngram_grid"]["min"], R["B_hard_vs_D_complex_charngram_grid"]["max"])
print({k: R[k] for k in ("phase1_fields", "phase1_has_correctness_field", "phase1_answer_max_chars",
                         "question_form", "reviewer_example_present_in_B_hard")})
OUT.write_text(json.dumps(R, indent=1), encoding="utf-8")
print("wrote", OUT.relative_to(REPO.parent).as_posix())
