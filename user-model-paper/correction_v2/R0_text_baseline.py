#!/usr/bin/env python3
"""R0: a text-only baseline for the user-model paper's cache probe (run plan, 2026-10-18).

Same data, same folds, same classifier as wk_backtrack_probe_v2.py, but the features are the story's words instead
of its KV-cache keys. If the words do as well, the cache adds nothing for this effect.

    python R0_text_baseline.py --trials emotion_bridge_trials.json --out R0_results.json

Folds: GroupKFold(5) by topic_idx; classifier LogisticRegression(C=0.1, max_iter=2000); valence on the probe's own
VALENCE_POS / VALENCE_NEG sets (copied verbatim from wk_backtrack_probe_v2.py:40-45).
Also reported: how many story texts are duplicates (the pseudoreplication flag), and a duplicate-free re-run.
"""
import argparse
import json
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

VALENCE_POS = {"calm", "cheerful", "content", "delighted", "grateful", "happy",
               "hopeful", "inspired", "joyful", "loving", "peaceful", "pleased",
               "proud", "relaxed", "satisfied", "serene", "thankful", "thrilled"}
VALENCE_NEG = {"afraid", "angry", "anxious", "bitter", "depressed", "desperate",
               "disgusted", "frustrated", "gloomy", "hostile", "hurt", "jealous",
               "lonely", "miserable", "sad", "terrified"}


def thirty_way(texts, labels, groups):
    labels = np.array(labels)
    classes = sorted(set(labels))
    preds = np.empty(len(labels), dtype=object)
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(texts, labels, groups):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)   # fit on the training fold only
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, labels[tr])
        preds[te] = clf.predict(Xte)
    acc = float(np.mean(preds == labels))
    return {"accuracy": acc, "chance": 1 / len(classes), "times_chance": acc * len(classes), "n": len(labels),
            "n_classes": len(classes), "per_trial_pred": [str(p) for p in preds]}   # for a paired test against R1


def valence(texts, emotions, groups):
    idx = [i for i, e in enumerate(emotions) if e in VALENCE_POS or e in VALENCE_NEG]
    y = np.array([1 if emotions[i] in VALENCE_POS else 0 for i in idx])
    t = [texts[i] for i in idx]
    g = np.array([groups[i] for i in idx])
    probs = np.zeros(len(idx))
    for tr, te in GroupKFold(n_splits=min(5, len(set(g)))).split(t, y, g):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(vec.fit_transform([t[i] for i in tr]), y[tr])
        probs[te] = clf.predict_proba(vec.transform([t[i] for i in te]))[:, 1]
    return {"auroc": float(roc_auc_score(y, probs)), "n": len(idx),
            "emotions_in_subset": sorted({emotions[i] for i in idx}),
            "per_trial": [{"trial": int(i), "y": int(yy), "p": float(pp)} for i, yy, pp in zip(idx, y, probs)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = json.load(open(a.trials, encoding="utf-8"))
    trials = raw["trials"] if isinstance(raw, dict) and "trials" in raw else (list(raw.values()) if isinstance(raw, dict) else raw)
    texts = [t["story_text"] for t in trials]
    emotions = [t["emotion"] for t in trials]
    topics = [t["topic_idx"] for t in trials]
    dup = Counter(texts)
    res = {"n_trials": len(texts), "n_unique_texts": len(dup),
           "max_copies_of_one_text": max(dup.values()),
           "comparison_cache_probe": {"thirty_way_accuracy": 0.409, "times_chance": 12.3, "valence_auroc": 0.992,
                                      "source": "wk_backtrack_v2_results.json"}}
    res["all_trials"] = {"thirty_way": thirty_way(texts, emotions, topics), "valence": valence(texts, emotions, topics)}
    first = {}
    for i, t in enumerate(texts):
        first.setdefault(t, i)
    keep = sorted(first.values())
    res["unique_texts_only"] = {
        "thirty_way": thirty_way([texts[i] for i in keep], [emotions[i] for i in keep], [topics[i] for i in keep]),
        "valence": valence([texts[i] for i in keep], [emotions[i] for i in keep], [topics[i] for i in keep])}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
