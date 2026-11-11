"""Logit margin and entropy by token window, and the fixed opening template, in the matched-entity
generations (review round 5, finding 9).

CPU only, no model run. From data/matched_burn_results.json: per-trial means of the stored logit_margin and
logit_entropy over token windows 0-9 (the feature the paper reports), 0-7, 8-19 and 20-49, compared between
fake- and real-entity prompts (Cohen's d with the pooled population SD, fake minus real; Welch is not used,
matching a plain two-sample t test); and how many generations open with the fixed template
"Let me think through this carefully", by class. data/r5_decoded_token_ids.json shows what tokens 0-7 and
8-19 are.

Every input hash is the sha256 of the LF form (CRLF -> LF), which is the git blob.
Output: data/r5_early_token_windows.json
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
MB = "decision-state-paper/data/matched_burn_results.json"
OUT = PAPER / "data" / "r5_early_token_windows.json"
TEMPLATE = "Let me think through this carefully"
WINDOWS = {"0-9": (0, 10), "0-7": (0, 8), "8-19": (8, 20), "20-49": (20, 50)}


def lf_bytes(rel):
    return (REPO / rel).read_bytes().replace(b"\r\n", b"\n")


def main():
    trials = json.loads(lf_bytes(MB).decode("utf-8"))["trials"]
    real = [t for t in trials if t["is_real"]]
    fake = [t for t in trials if not t["is_real"]]
    win = {}
    for w, (a, b) in WINDOWS.items():
        rec = {}
        for f in ("logit_margin", "logit_entropy"):
            r = [float(np.mean(t["trajectory"][f][a:b])) for t in real]
            k = [float(np.mean(t["trajectory"][f][a:b])) for t in fake]
            d = (np.mean(k) - np.mean(r)) / np.sqrt((np.var(r) + np.var(k)) / 2)
            rec[f] = {"real_mean": round(float(np.mean(r)), 4), "fake_mean": round(float(np.mean(k)), 4),
                      "d_fake_minus_real": round(float(d), 4),
                      "p_two_sample_t": float(f"{stats.ttest_ind(r, k).pvalue:.4g}"), "n_real": len(r), "n_fake": len(k)}
        win[w] = rec
    tmpl = Counter(("real" if t["is_real"] else "fake", t["generated_text"].startswith(TEMPLATE)) for t in trials)
    result = {
        "LABEL": "Review round 5, finding 9. Output-logit statistics by token window; no KV-cache feature is involved. Trials are treated as independent (three reps per prompt), as in the paper's original test.",
        "inputs": {MB: hashlib.sha256(lf_bytes(MB)).hexdigest()},
        "hash_note": "sha256 of the LF form (CRLF replaced by LF), i.e. of the git blob",
        "windows": win,
        "template": TEMPLATE,
        "opens_with_template": {"fake": tmpl[("fake", True)], "real": tmpl[("real", True)],
                                "total": tmpl[("fake", True)] + tmpl[("real", True)], "n_trials": len(trials)},
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(REPO).as_posix())
    for w, rec in win.items():
        print(f"  {w}: " + "; ".join(f"{f} real {v['real_mean']} fake {v['fake_mean']} d {v['d_fake_minus_real']} p {v['p_two_sample_t']}" for f, v in rec.items()))
    print("  template:", result["opens_with_template"])


if __name__ == "__main__":
    main()
