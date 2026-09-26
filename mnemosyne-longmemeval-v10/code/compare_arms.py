#!/usr/bin/env python3
"""Paired comparison of the arms on the questions ALL of them have judged.

Arms (same reader claude-opus-4-6, same official answer-check prompts, same two Claude judges):
  v9      Mnemosyne v9 context   judge_sonnet5 / judge_opus55
  oracle  evidence sessions only judge_oracle_sonnet5 / judge_oracle_opus55
  full    the whole raw history  judge_full_sonnet5 / judge_full_opus55   (answered block by block)
  v10     v10 retrieval context  judge_v10_sonnet5 / judge_v10_opus55     (added 2026-09-24; dev split = block 1)

Only questions judged in every arm, by both judges, are compared, so a partial full arm gives a paired
estimate on its stratified blocks, not a mix of different question sets. McNemar's exact test (two-sided
binomial on the discordant pairs) tests v9 against each baseline.
Usage: compare_arms.py [--out FILE]
"""
import argparse, collections, json, math
from pathlib import Path

ROOT = Path("/mnt/data1/lme_v2")
ALL_ARMS = {"v9": "", "v10": "v10_", "oracle": "oracle_", "full": "full_",
            # 2026-09-24: same prompts, reader claude-opus-5-5 with thinking (SELF_LIMITS_AUDIT.md)
            "v10_o55t": "v10_o55t_", "oracle_o55t": "oracle_o55t_"}
ALL_PAIRS = [("v9", "oracle"), ("v9", "full"), ("v10", "v9"), ("v10", "full"), ("v10", "oracle"),
             ("v10_o55t", "v10"), ("oracle_o55t", "oracle"), ("v10_o55t", "oracle_o55t"), ("v10_o55t", "full")]
ARMS, PAIRS = {}, []   # set in main() from --arms
JUDGES = {"sonnet5": "claude-sonnet-5", "opus55": "claude-opus-5-5"}

def labels(prefix, judge):
    d = ROOT / f"judge_{prefix}{judge}"
    out = {}
    for p in d.glob("*.json"):
        j = json.load(open(p))
        assert j["judge_model"] == JUDGES[judge], (d, p.name)
        out[j["question_id"]] = bool(j["label"])
    return out

def mcnemar_p(b, c):
    """Exact two-sided p for b discordant one way and c the other."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)

def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out")
    ap.add_argument("--arms", default="v9,v10,oracle,full", help="comma list from: " + ",".join(ALL_ARMS))
    ap.add_argument("--split", choices=["all", "dev", "held"], default="all",
                    help="restrict to a v10 split (v10/split.json); held = the 400 questions nothing was tuned on")
    a = ap.parse_args()
    ARMS.update({k: ALL_ARMS[k] for k in a.arms.split(",")})
    PAIRS.extend(p for p in ALL_PAIRS if p[0] in ARMS and p[1] in ARMS)
    meta = {m["question_id"]: m for m in json.load(open(ROOT / "prompts_meta.json"))}
    L = {(arm, j): labels(pre, j) for arm, pre in ARMS.items() for j in JUDGES}
    common = set(meta)
    for v in L.values():
        common &= set(v)
    if a.split != "all":
        common &= set(json.load(open(ROOT / "v10/split.json"))[f"{a.split}_qids"])
    common = sorted(common)
    lines = [f"# Paired arm comparison, {len(common)} questions judged in every arm by both judges"
             + (f" (v10 split: {a.split})" if a.split != "all" else ""), ""]
    if not common:
        print("no question is judged in every arm yet"); return
    types = sorted({meta[q]["question_type"] for q in common})
    for j in JUDGES:
        lines += [f"## Judge {JUDGES[j]}", "",
                  "| arm | correct | accuracy | Wilson 95% | " + " | ".join(types) + " |",
                  "|---|---|---|---|" + "---|" * len(types)]
        for arm in ARMS:
            lab = L[(arm, j)]; k = sum(lab[q] for q in common); lo, hi = wilson(k, len(common))
            per = []
            for t in types:
                qs = [q for q in common if meta[q]["question_type"] == t]
                per.append(f"{100 * sum(lab[q] for q in qs) / len(qs):.1f} ({len(qs)})")
            lines.append(f"| {arm} | {k}/{len(common)} | {100 * k / len(common):.1f}% | [{100 * lo:.1f}, {100 * hi:.1f}] | " + " | ".join(per) + " |")
        lines.append("")
        for x, other in PAIRS:
            b = sum(L[(x, j)][q] and not L[(other, j)][q] for q in common)
            c = sum(L[(other, j)][q] and not L[(x, j)][q] for q in common)
            lines.append(f"- {x} vs {other}: {x} right / {other} wrong = {b}; {other} right / {x} wrong = {c}; "
                         f"McNemar exact p = {mcnemar_p(b, c):.3g}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")

if __name__ == "__main__":
    main()
