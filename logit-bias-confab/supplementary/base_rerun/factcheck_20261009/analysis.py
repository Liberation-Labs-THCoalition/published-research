#!/usr/bin/env python3
"""Fact-check of the rerun judge's honest labels (post hoc, 2026-10-09): every number the paper reports from it.

Run from anywhere: python3 analysis.py. It reads only the files in this directory (paths are relative to this script).
If the rerun's judge output is present at ../results/pass1.json and ../results/pass2.json, it also recomputes the judge
counts below from it and stops if they differ.

Parts:
  0. the freeze: RULES.md and items.json against FROZEN.sha256;
  1. the frozen analysis (RULES.md, "Analysis", fixed before rating): Cohen's kappa with UNSURE as its own category,
     each rater's FAB rate per cell, the agreed FAB rate (the rate among items where both raters agree), the control
     rate, and analysis 4's corrected fabrication;
  2. the entity split (post hoc, after rating): FAB items restricted to the triggering bullets 1 (an invented property
     of the fictional entity) or 2 (an invented "did you mean" alternative). Bullet 3 (a false specific about a real
     thing) alone does not count. Lenient counts weak bullet-2 tags ("2w"); strict drops "2w" and any item rater B
     tagged as a base-rate lean ("b");
  3. the agreed-rate estimate: the frozen aggregation rule (agreed rate) applied to the lenient entity split;
  4. the per-rater rows: corrected fabrication with each rater basis (A alone, B alone, both flag, either flags).

Corrected fabrication (descriptive, no test): judge fabrication count + honest severity-1 count x rate
(+ honest severity-0 count x control rate, in parts 3 and 4), out of 240 responses per bias.
"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_PER_BIAS = 240
BIASES = (0.0, 5.0)
HONEST = {"HONEST_REDIRECT", "HONEST_HEDGE"}
FABRICATION = {"FULL_CONFAB", "COSMETIC_HEDGE"}

# Judge counts per pass and bias, out of 240: (fabrication = FULL_CONFAB or COSMETIC_HEDGE, honest with severity 1,
# honest with severity 0). From ../results/pass1.json and pass2.json; checked against them below when present.
JUDGE = {"pass1": {0.0: (80, 45, 115), 5.0: (42, 75, 123)},
         "pass2": {0.0: (83, 43, 114), 5.0: (48, 66, 126)}}


def load_json(name):
    with open(HERE / name, encoding="utf-8") as f:
        return json.load(f)


def wilson(x, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = x / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in set(x) | set(y))
    return (po - pe) / (1 - pe)


def pct(x, n):
    return f"{x / n:.0%} ({x}/{n})"


def check_freeze():
    print("== 0. Freeze (FROZEN.sha256)")
    ok = True
    for line in (HERE / "FROZEN.sha256").read_text().splitlines():
        digest, name = line.split()
        got = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        print(f"  {name}: {'OK' if got == digest else 'MISMATCH'}")
        ok &= got == digest
    if not ok:
        raise SystemExit("frozen files changed; stopping")


def check_judge_counts():
    res = HERE.parent / "results"
    if not (res / "pass1.json").exists() or not (res / "pass2.json").exists():
        print("  (../results/pass*.json not found; judge counts taken from the constants in this script)")
        return
    for p in (1, 2):
        with open(res / f"pass{p}.json", encoding="utf-8") as f:
            rows = json.load(f)
        for b in BIASES:
            js = [r["judge"] for r in rows if r["key"][3] == b]
            fab = sum(j["classification"] in FABRICATION for j in js)
            s1 = sum(j["classification"] in HONEST and j["fabrication_severity"] == 1 for j in js)
            s0 = sum(j["classification"] in HONEST and j["fabrication_severity"] == 0 for j in js)
            if len(js) != N_PER_BIAS or (fab, s1, s0) != JUDGE[f"pass{p}"][b]:
                raise SystemExit(f"judge counts differ in pass {p}, bias {b}: {(fab, s1, s0)} vs {JUDGE[f'pass{p}'][b]}")
    print("  judge counts checked against ../results/pass1.json and pass2.json: match")


def cells(key):
    out = {}
    for s in ("sev1", "ctrl"):
        for b in BIASES:
            out[(s, b)] = [i for i in sorted(key, key=int) if key[i]["stratum"] == s and key[i]["bias"] == b]
    return out


def main():
    check_freeze()
    key = load_json("key.json")
    A = {str(x["item"]): x["verdict"] for x in load_json("rater_A.json")}
    B = {str(x["item"]): x["verdict"] for x in load_json("rater_B.json")}
    TA = {str(x["item"]): set(x["bullets"]) for x in load_json("rater_A_bullets.json")}
    TB = {str(x["item"]): set(x["bullets"]) for x in load_json("rater_B_bullets.json")}
    ids = sorted(key, key=int)
    assert len(ids) == 170 and set(A) == set(B) == set(ids), "items, key and verdicts do not line up"
    C = cells(key)
    print(f"  items: {len(ids)}; " + "; ".join(f"{s} bias {b}: {len(v)}" for (s, b), v in C.items()))
    check_judge_counts()

    # ---------------------------------------------------------------- 1. frozen analysis
    print("\n== 1. Frozen analysis (rule as frozen: any invented or false specific counts)")
    va, vb = [A[i] for i in ids], [B[i] for i in ids]
    print(f"  verdicts: A {dict(Counter(va))}; B {dict(Counter(vb))}")
    print(f"  Cohen's kappa (FAB / NOT_FAB / UNSURE) = {kappa(va, vb):.2f}; raw agreement "
          f"{sum(a == b for a, b in zip(va, vb)) / len(ids):.0%}")
    print(f"  every B-FAB is also an A-FAB: {all(A[i] == 'FAB' for i in ids if B[i] == 'FAB')}")
    agreed = {}
    for (s, b), c in C.items():
        xa = sum(A[i] == "FAB" for i in c)
        xb = sum(B[i] == "FAB" for i in c)
        ag = [i for i in c if A[i] == B[i]]
        x = sum(A[i] == "FAB" for i in ag)
        lo, hi = wilson(x, len(ag))
        agreed[(s, b)] = (x, len(ag))
        print(f"  {s} bias {b}: n={len(c)}  A FAB {pct(xa, len(c))}  B FAB {pct(xb, len(c))}  "
              f"agreed FAB {pct(x, len(ag))} [{lo:.0%}, {hi:.0%}]")
    ctrl = [agreed[("ctrl", b)][0] / agreed[("ctrl", b)][1] for b in BIASES]
    print(f"  control agreed FAB rate {ctrl[0]:.0%} / {ctrl[1]:.0%}: "
          f"{'above' if max(ctrl) > 0.10 else 'not above'} the frozen 10% caveat")
    print("  analysis 4, corrected fabrication = judge fabrication + honest severity-1 x agreed FAB rate "
          "(range from the Wilson interval on the rate):")
    for p, d in JUDGE.items():
        out = []
        for b, (fab, s1, _s0) in d.items():
            x, n = agreed[("sev1", b)]
            lo, hi = wilson(x, n)
            est = fab + s1 * x / n
            out.append(f"bias {b}: {fab / N_PER_BIAS:.1%} -> {est / N_PER_BIAS:.1%} "
                       f"[{(fab + s1 * lo) / N_PER_BIAS:.1%}, {(fab + s1 * hi) / N_PER_BIAS:.1%}]")
        print(f"    {p}  " + " | ".join(out))

    # ---------------------------------------------------------------- 2. entity split
    def ent(V, T, i, strict):
        if V[i] != "FAB":
            return False
        t = T.get(i, set())
        if strict and "b" in t:
            return False
        return bool(t & {"1", "2"}) or (not strict and "2w" in t)

    rates = {}
    for strict in (False, True):
        name = "strict" if strict else "lenient"
        print(f"\n== 2. Entity split, {name} (post hoc, after rating): bullet 1 or 2"
              + ("; no 2w, no base-rate leans" if strict else ", including 2w"))
        ea = {i: ent(A, TA, i, strict) for i in ids}
        eb = {i: ent(B, TB, i, strict) for i in ids}
        la, lb = [ea[i] for i in ids], [eb[i] for i in ids]
        print(f"  kappa (entity yes/no) = {kappa(la, lb):.2f}; raw agreement "
              f"{sum(a == b for a, b in zip(la, lb)) / len(ids):.0%}")
        r = {}
        for (s, b), c in C.items():
            n = len(c)
            xa = sum(ea[i] for i in c)
            xb = sum(eb[i] for i in c)
            both = sum(ea[i] and eb[i] for i in c)
            either = sum(ea[i] or eb[i] for i in c)
            ag = [i for i in c if ea[i] == eb[i]]
            xag = sum(ea[i] for i in ag)
            lo, hi = wilson(xag, len(ag))
            r[(s, b)] = {"A": xa / n, "B": xb / n, "both": both / n, "either": either / n,
                         "agreed": xag / len(ag)}
            print(f"  {s} bias {b}: n={n}  A {pct(xa, n)}  B {pct(xb, n)}  both {pct(both, n)}  "
                  f"either {pct(either, n)}  agreed {pct(xag, len(ag))} [{lo:.0%}, {hi:.0%}]")
        rates[name] = r

    # ---------------------------------------------------------------- 3. agreed-rate estimate
    def corrected(r, basis):
        out = {}
        for p, d in JUDGE.items():
            for b, (fab, s1, s0) in d.items():
                out[(p, b)] = (fab + s1 * r[("sev1", b)][basis] + s0 * r[("ctrl", b)][basis]) / N_PER_BIAS
        return out

    print("\n== 3. Agreed-rate estimate: frozen aggregation rule (agreed rate) on the lenient entity split")
    print("  corrected = judge fabrication + honest severity-1 x agreed rate + honest severity-0 x control agreed rate")
    est = corrected(rates["lenient"], "agreed")
    for p in JUDGE:
        a0, a5 = est[(p, 0.0)], est[(p, 5.0)]
        print(f"    {p}: {a0:.1%} -> {a5:.1%}, a drop of {100 * (a0 - a5):.1f} points")

    # ---------------------------------------------------------------- 4. per-rater rows
    print("\n== 4. Per-rater rows (lenient entity split), corrected fabrication and drop in points")
    print(f"  {'basis':<12} {'pass 1':<17} {'drop':>5}   {'pass 2':<17} {'drop':>5}")
    jd = {(p, b): JUDGE[p][b][0] / N_PER_BIAS for p in JUDGE for b in BIASES}
    rows = [("judge", jd)] + [(lab, corrected(rates["lenient"], basis)) for lab, basis in
                              (("both flag", "both"), ("B alone", "B"), ("A alone", "A"), ("either", "either"))]
    for lab, e in rows:
        cols = []
        for p in JUDGE:
            a0, a5 = e[(p, 0.0)], e[(p, 5.0)]
            cols.append(f"{100 * a0:.1f} -> {100 * a5:.1f}".ljust(17) + f" {100 * (a0 - a5):5.1f}")
        print(f"  {lab:<12} " + "   ".join(cols))


if __name__ == "__main__":
    main()
