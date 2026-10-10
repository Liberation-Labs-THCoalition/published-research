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
  3. the agreed-rate estimate: frozen analysis 4 (the agreed rate) applied to the lenient entity split, with the
     frozen Wilson interval on the rate and a prompt-cluster bootstrap (added after rating);
  4. the per-rater rows: the same formula with each rater basis (both flag, B alone, A alone, either flags);
  5. superseded: a version of parts 3 and 4 that added a post hoc control term. Not used in the paper; kept so the
     numbers in the paper's first draft of this estimate can be traced.

Corrected fabrication (descriptive, no test), as frozen in RULES.md analysis 4: judge fabrication count + each pass's
honest severity-1 count x rate, out of 240 responses per bias. Parts 1 to 4 use exactly this formula; only part 5 adds
a severity-0 term.

Correction (2026-10-09): the first draft of parts 3 and 4 added honest severity-0 count x control rate to the frozen
formula. That term was not in the frozen analysis; the Agni review gate caught it, and it is now confined to part 5.
"""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
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
SAMPLES_PER_PROMPT = 5
BOOT_N, BOOT_SEED = 2000, 20261009


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
    """Check JUDGE against ../results/pass*.json when present, and return the per-response judge output
    ({"pass1": {key: judge}, "pass2": ...}) for the prompt-cluster bootstrap, or None if absent."""
    res = HERE.parent / "results"
    if not (res / "pass1.json").exists() or not (res / "pass2.json").exists():
        print("  (../results/pass*.json not found; judge counts taken from the constants in this script)")
        return None
    labels = {}
    for p in (1, 2):
        with open(res / f"pass{p}.json", encoding="utf-8") as f:
            rows = json.load(f)
        labels[f"pass{p}"] = {tuple(r["key"]): r["judge"] for r in rows}
        for b in BIASES:
            js = [r["judge"] for r in rows if r["key"][3] == b]
            fab = sum(j["classification"] in FABRICATION for j in js)
            s1 = sum(j["classification"] in HONEST and j["fabrication_severity"] == 1 for j in js)
            s0 = sum(j["classification"] in HONEST and j["fabrication_severity"] == 0 for j in js)
            if len(js) != N_PER_BIAS or (fab, s1, s0) != JUDGE[f"pass{p}"][b]:
                raise SystemExit(f"judge counts differ in pass {p}, bias {b}: {(fab, s1, s0)} vs {JUDGE[f'pass{p}'][b]}")
    print("  judge counts checked against ../results/pass1.json and pass2.json: match")
    return labels


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
    labels = check_judge_counts()

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

    rates, flags = {}, {}
    for strict in (False, True):
        name = "strict" if strict else "lenient"
        print(f"\n== 2. Entity split, {name} (post hoc, after rating): bullet 1 or 2"
              + ("; no 2w, no base-rate leans" if strict else ", including 2w"))
        ea = {i: ent(A, TA, i, strict) for i in ids}
        eb = {i: ent(B, TB, i, strict) for i in ids}
        flags[name] = (ea, eb)
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
                         "agreed": xag / len(ag), "agreed_x": xag, "agreed_n": len(ag)}
            print(f"  {s} bias {b}: n={n}  A {pct(xa, n)}  B {pct(xb, n)}  both {pct(both, n)}  "
                  f"either {pct(either, n)}  agreed {pct(xag, len(ag))} [{lo:.0%}, {hi:.0%}]")
        rates[name] = r

    # ---------------------------------------------------------------- 3. agreed-rate estimate (frozen analysis 4)
    def corrected(r, basis):
        """Frozen analysis 4: judge fabrication + each pass's honest severity-1 count x rate. No severity-0 term."""
        return {(p, b): (fab + s1 * r[("sev1", b)][basis]) / N_PER_BIAS
                for p, d in JUDGE.items() for b, (fab, s1, _s0) in d.items()}

    def print_rows(rows):
        print(f"  {'basis':<12} {'pass 1':<17} {'drop':>5}   {'pass 2':<17} {'drop':>5}")
        for lab, e in rows:
            cols = []
            for p in JUDGE:
                a0, a5 = e[(p, 0.0)], e[(p, 5.0)]
                cols.append(f"{100 * a0:.1f} -> {100 * a5:.1f}".ljust(17) + f" {100 * (a0 - a5):5.1f}")
            print(f"  {lab:<12} " + "   ".join(cols))

    len_r = rates["lenient"]
    print("\n== 3. Agreed-rate estimate: frozen analysis 4 (agreed rate) on the lenient entity split")
    print("  corrected = judge fabrication + each pass's honest severity-1 count x agreed rate (no severity-0 term)")
    est = corrected(len_r, "agreed")
    for p in JUDGE:
        a0, a5 = est[(p, 0.0)], est[(p, 5.0)]
        print(f"    {p}: {a0:.1%} -> {a5:.1%}, a drop of {100 * (a0 - a5):.1f} points")

    print("  frozen Wilson interval on the agreed rate, carried to the drop (rate uncertainty only; low end = "
          "bias-0 rate at its low bound and bias-5.0 rate at its high bound, and the reverse):")
    for p, d in JUDGE.items():
        (f0, s0, _), (f5, s5, _) = d[0.0], d[5.0]
        l0, h0 = wilson(len_r[("sev1", 0.0)]["agreed_x"], len_r[("sev1", 0.0)]["agreed_n"])
        l5, h5 = wilson(len_r[("sev1", 5.0)]["agreed_x"], len_r[("sev1", 5.0)]["agreed_n"])
        lo = ((f0 + s0 * l0) - (f5 + s5 * h5)) / N_PER_BIAS
        hi = ((f0 + s0 * h0) - (f5 + s5 * l5)) / N_PER_BIAS
        print(f"    {p}: [{100 * lo:.1f}, {100 * hi:.1f}] points")

    if labels is None:
        print("  prompt-cluster bootstrap: skipped (needs ../results/pass1.json and pass2.json)")
    else:
        ea, eb = flags["lenient"]
        prompts = sorted({k[:2] for k in labels["pass1"]})
        # per prompt and bias: judge fabrication and honest severity-1 counts per pass; agreed and agreed-flagged items
        jc = {p: {b: defaultdict(lambda: [0, 0]) for b in BIASES} for p in labels}
        for p, lab in labels.items():
            for k, j in lab.items():
                if k[3] in BIASES:
                    c = jc[p][k[3]][k[:2]]
                    c[0] += j["classification"] in FABRICATION
                    c[1] += j["classification"] in HONEST and j["fabrication_severity"] == 1
        ag = {b: defaultdict(lambda: [0, 0]) for b in BIASES}
        for i in ids:
            if key[i]["stratum"] == "sev1" and ea[i] == eb[i]:
                c = ag[key[i]["bias"]][tuple(key[i]["key"][:2])]
                c[0] += 1
                c[1] += ea[i]

        def cluster_est(pmult):
            n = sum(pmult.values()) * SAMPLES_PER_PROMPT
            out = {}
            for p in labels:
                val = {}
                for b in BIASES:
                    fab = sum(m * jc[p][b][q][0] for q, m in pmult.items() if q in jc[p][b])
                    s1 = sum(m * jc[p][b][q][1] for q, m in pmult.items() if q in jc[p][b])
                    den = sum(m * ag[b][q][0] for q, m in pmult.items() if q in ag[b])
                    num = sum(m * ag[b][q][1] for q, m in pmult.items() if q in ag[b])
                    val[b] = (fab + s1 * (num / den if den else 0)) / n
                out[p] = 100 * (val[0.0] - val[5.0])
            return out

        point = cluster_est({q: 1 for q in prompts})
        for p in JUDGE:
            if abs(point[p] - 100 * (est[(p, 0.0)] - est[(p, 5.0)])) > 1e-9:
                raise SystemExit(f"bootstrap point estimate differs from part 3 in {p}")
        random.seed(BOOT_SEED)
        boots = {p: [] for p in labels}
        for _ in range(BOOT_N):
            pm = defaultdict(int)
            for q in random.choices(prompts, k=len(prompts)):
                pm[q] += 1
            for p, d in cluster_est(pm).items():
                boots[p].append(d)
        lo_i, hi_i = round(0.025 * BOOT_N), round(0.975 * BOOT_N) - 1
        print(f"  prompt-cluster bootstrap of the drop (post hoc, after rating; {len(prompts)} prompts resampled with "
              f"replacement, {BOOT_N} resamples, random.seed({BOOT_SEED}); judge labels and agreed rate recomputed "
              f"in each; 2.5th and 97.5th percentiles):")
        for p in JUDGE:
            bs = sorted(boots[p])
            print(f"    {p}: [{bs[lo_i]:.1f}, {bs[hi_i]:.1f}] points")

    # ---------------------------------------------------------------- 4. per-rater rows
    print("\n== 4. Per-rater rows (lenient entity split, frozen analysis 4 formula), corrected fabrication and drop in "
          "points")
    jd = {(p, b): JUDGE[p][b][0] / N_PER_BIAS for p in JUDGE for b in BIASES}
    print_rows([("judge", jd)] + [(lab, corrected(len_r, basis)) for lab, basis in
                                  (("both flag", "both"), ("B alone", "B"), ("A alone", "A"), ("either", "either"))])

    # ---------------------------------------------------------------- 5. superseded control-term version
    print("\n== 5. SUPERSEDED: a version that added a post hoc control term; superseded, not used in the paper")
    print("  corrected = judge fabrication + honest severity-1 x rate + honest severity-0 x control rate; the last")
    print("  term is not in the frozen analysis 4. These are the numbers of the paper's first draft of this estimate.")

    def corrected_ctrl(r, basis):
        return {(p, b): (fab + s1 * r[("sev1", b)][basis] + s0 * r[("ctrl", b)][basis]) / N_PER_BIAS
                for p, d in JUDGE.items() for b, (fab, s1, s0) in d.items()}

    est_c = corrected_ctrl(len_r, "agreed")
    for p in JUDGE:
        a0, a5 = est_c[(p, 0.0)], est_c[(p, 5.0)]
        print(f"    agreed rate, {p}: {a0:.1%} -> {a5:.1%}, a drop of {100 * (a0 - a5):.1f} points")
    print_rows([(lab, corrected_ctrl(len_r, basis)) for lab, basis in
                (("both flag", "both"), ("B alone", "B"), ("A alone", "A"), ("either", "either"))])


if __name__ == "__main__":
    main()
