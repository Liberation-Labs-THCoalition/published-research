"""Split the 900 bridge stories into those that carry the generator's leaked reasoning and
those that don't, and report each probe's per-trial accuracy on both groups.

Inputs are the released files in this folder, checked against their sha256:
  emotion_bridge_trials.json      the stories (registered sha256 fc5938fe...)
  results_20261019/um_v2_results.json   per-trial predictions of R0, R1, R2 and the residual LR

    python leak_split.py

The registered tests ran on all 900 stories. Everything this script prints is descriptive.
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHA = {
    "trials": "fc5938feccc57c68db0c1625236839d7405708219911dc2c532de836eb836e7e",
    "results": "0aefed3605e1c1d42ee21705c6a35801315c5c222bc21fa4b7c8ec627b3e36c3",
}
# The pre-set pattern (defined before this script existed; the one R1R2_RESULTS.md reports).
LEAK = re.compile(r"<analysis>|</?think|The user wants|the user (is asking|wants)|key constraint", re.I)
# POST-DATA sensitivity pattern, added after reading all 900 stories and before computing any probe number
# on it. LEAK misses planning that opens "I need to..." or "**Thinking Process:**", refusals, and the
# generator's own scaffolding ("# Solution", "**Final Answer:**", a "Verification:" checklist). Each extra
# hit was read and is a non-story or a framed story. Reported as a sensitivity check, never as the split.
BROAD = re.compile(
    r"^\s*(#+\s*(Solution|Thinking|Reasoning|Thought)|\*\*(Thinking|Reasoning|Thought|Theoretical|Final Answer)"
    r"|I need to|I.ll (write|compose|craft)|I cannot|I.m not going to|I notice you)|Verification:|Explanation of",
    re.I)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path, key):
    got = sha256(path)
    if got != SHA[key]:
        raise SystemExit(f"{path}: sha256 {got[:16]} is not the released {key} file ({SHA[key][:16]}). Refusing.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", default=os.path.join(HERE, "emotion_bridge_trials.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results_20261019", "um_v2_results.json"))
    a = ap.parse_args()

    trials = load(a.trials, "trials")["trials"]
    res = load(a.results, "results")
    emotions = res["meta"]["emotions"]
    per = res["per_trial"]
    n = len(trials)
    if n != 900 or len(per["emotion"]) != n:
        raise SystemExit(f"expected 900 trials in both files, got {n} and {len(per['emotion'])}")
    truth = [t["emotion"] for t in trials]
    if truth != per["emotion"]:
        raise SystemExit("trial order differs between the stories file and the results file")
    topic = per["topic_idx"]
    text = [t["story_text"] for t in trials]

    deeply = [bool(re.search(r"deeply\s+" + re.escape(e), s, re.I)) for s, e in zip(text, truth)]

    def names(pred):
        return [emotions[p] if isinstance(p, int) else p for p in pred]

    probes = [
        ("R0 text (TF-IDF), 10-fold", res["R0"]["pred_10fold"], res["R0"]["acc_10fold"]),
        ("R2 cache, registered primary", res["R2"]["primary"]["pred"], res["R2"]["primary"]["accuracy"]),
        ("R1 v2 code path, mean pool", res["R1"]["mean"]["pred"], res["R1"]["mean"]["accuracy"]),
        ("residual LR res_out (EXPLORATORY)", res["R2"]["raw_residual_LR_EXPLORATORY"][0]["pred"],
         res["R2"]["raw_residual_LR_EXPLORATORY"][0]["accuracy"]),
    ]
    right = {}
    for label, pred, stored in probes:
        r = [x == y for x, y in zip(names(pred), truth)]
        # positive control: the recomputed accuracy must equal the one the analysis stored
        if abs(sum(r) / n - stored) > 1e-12:
            raise SystemExit(f"{label}: recomputed {sum(r) / n} != stored {stored}; prediction mapping is wrong")
        right[label] = r

    def split(name, flagged):
        nf = sum(flagged)
        out = {"flagged": nf, "clean": n - nf,
               "deeply_own_label": sum(deeply), "deeply_and_flagged": sum(d and f for d, f in zip(deeply, flagged)),
               "flagged_by_topic": {}, "probes": {}}
        for t in sorted(set(topic)):
            k = sum(1 for i in range(n) if flagged[i] and topic[i] == t)
            if k:
                out["flagged_by_topic"][str(t)] = k
        print(f"== {name}")
        print(f"stories: {n}; flagged: {nf} ({100 * nf / n:.1f}%); clean: {n - nf}")
        print(f'"deeply <own label>": {out["deeply_own_label"]} stories, {out["deeply_and_flagged"]} of them flagged')
        print("flagged stories by topic index:", out["flagged_by_topic"])
        print(f"{'probe':38s} {'all 900':>14s} {'flagged':>14s} {'clean':>14s}")
        for label, _, _ in probes:
            r = right[label]
            kf = sum(x for x, f in zip(r, flagged) if f)
            kc = sum(x for x, f in zip(r, flagged) if not f)
            out["probes"][label] = {"all": [sum(r), n], "flagged": [kf, nf], "clean": [kc, n - nf]}
            print(f"{label:38s} {sum(r):4d}/{n} {sum(r) / n:.3f}  {kf:3d}/{nf} {kf / nf:.3f}  "
                  f"{kc:3d}/{n - nf} {kc / (n - nf):.3f}")
        r0c = out["probes"][probes[0][0]]["clean"][0]
        r2c = out["probes"][probes[1][0]]["clean"][0]
        print(f"clean {n - nf}: R2 minus text (TF-IDF) = {100 * (r2c - r0c) / (n - nf):+.1f} points "
              f"({r2c} against {r0c} correct). Descriptive; the registered test is on all 900.")
        print(json.dumps(out))
        print()
        return out

    leaked = [bool(LEAK.search(s)) for s in text]
    split("pre-set pattern (LEAK)", leaked)
    broad = [l or bool(BROAD.search(s)) for l, s in zip(leaked, text)]
    split("POST-DATA sensitivity: LEAK or BROAD", broad)


if __name__ == "__main__":
    sys.exit(main())
