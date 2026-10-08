"""Summarise the overnight re-judge. EXPLORATORY re-analysis of existing responses, not the pre-registered rerun.

    python3 analyze_rejudge.py   (run inside the rejudge directory)
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CATS = ["FULL_CONFAB", "COSMETIC_HEDGE", "HONEST_HEDGE", "HONEST_REDIRECT", "LEGITIMATE", "META_HEDGE"]
BIASES = [0.0, 1.0, 2.0, 3.0, 5.0]
rng = random.Random(20260930)


def load(name):
    p = Path(name)
    return [r for r in json.loads(p.read_text()) if "classification" in r["judge"]] if p.exists() else []


def lab(r):
    return r["judge"]["classification"]


out = ["# Blind re-judge: results (exploratory)", "",
       "One judge model throughout: claude-sonnet-4-6 (the paper's original judge), via the Claude CLI.", ""]

# 1. Primary study: blind vs unblinded, same model.
pb = {json.dumps(r["key"]): r for r in load("powered_blind.json")}
pu = {json.dumps(r["key"]): r for r in load("powered_unblinded.json")}
both = [k for k in pb if k in pu]
if both:
    agree = sum(lab(pb[k]) == lab(pu[k]) for k in both)
    obs = Counter((lab(pb[k]), lab(pu[k])) for k in both)
    n = len(both)
    po = agree / n
    pe = sum((sum(v for (a, _), v in obs.items() if a == c) / n) * (sum(v for (_, b), v in obs.items() if b == c) / n)
             for c in CATS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    out += ["## 1. Primary study (abliterated): blind vs unblinded, same model", "",
            f"{n} responses judged both ways. Agreement {agree}/{n} ({po:.1%}), Cohen's kappa {kappa:.2f}.", "",
            "Fictional prompts, FULL_CONFAB count per bias (blind | unblinded):", ""]
    for b in BIASES:
        ks = [k for k in both if pb[k]["prompt_type"] == "fictional" and pb[k]["bias"] == b]
        fb = sum(lab(pb[k]) == "FULL_CONFAB" for k in ks)
        fu = sum(lab(pu[k]) == "FULL_CONFAB" for k in ks)
        hb = sum(lab(pb[k]).startswith("HONEST") for k in ks)
        hu = sum(lab(pu[k]).startswith("HONEST") for k in ks)
        out.append(f"- bias {b}: FULL_CONFAB {fb} | {fu}; HONEST_* {hb} | {hu}  (n = {len(ks)})")
    flips = Counter((lab(pb[k]), lab(pu[k])) for k in both if lab(pb[k]) != lab(pu[k]))
    out += ["", "Label changes when the judge sees the condition (blind -> unblinded): "
            + (", ".join(f"{a}->{b} x{v}" for (a, b), v in flips.most_common(8)) or "none"), ""]
    out += ["Reported (morning report, unknown judge setup; its SEARCH class is not in this rubric): "
            "FULL_CONFAB 9, 6, 6, 8, 2 at bias 0, 1, 2, 3, 5.", ""]
    short = {"FULL_CONFAB": "F", "COSMETIC_HEDGE": "C", "HONEST_HEDGE": "h", "HONEST_REDIRECT": "r",
             "LEGITIMATE": "L", "META_HEDGE": "m"}
    paper = {0: "2.0", 3: "5.0", 4: "5.0", 5: "1.0", 6: "NEVER", 8: "3.0", 10: "1.0", 11: "NEVER", 17: "5.0"}
    tb, tu = {}, {}
    for k in both:
        if pb[k]["prompt_type"] == "fictional":
            tb.setdefault(pb[k]["key"][1], {})[pb[k]["bias"]] = lab(pb[k])
            tu.setdefault(pu[k]["key"][1], {})[pu[k]["bias"]] = lab(pu[k])
    out += ["Per prompt (F full confab, C cosmetic, h honest hedge, r honest redirect; bias 0, 1, 2, 3, 5):", "",
            "| Prompt | Blind | Unblinded | First honest bias (blind) | Paper's threshold |", "|---|---|---|---|---|"]
    for q in sorted(tb):
        sb = "".join(short.get(tb[q].get(b, "?"), "?") for b in BIASES)
        su = "".join(short.get(tu[q].get(b, "?"), "?") for b in BIASES)
        first = (next((str(b) for b in BIASES[1:] if not tb[q].get(b, "").startswith(("FULL", "COSMETIC"))), "never")
                 if tb[q].get(0.0) == "FULL_CONFAB" else "")
        out.append(f"| P{q:02d} | {sb} | {su} | {first} | {paper.get(q, '')} |")
    out.append("")

# 2. Base model: the June T=0.7 rerun, blind.
bb = [r for r in load("base_blind.json") if r["prompt_type"] == "fictional"]
if bb:
    by = defaultdict(lambda: defaultdict(list))
    for r in bb:
        by[r["bias"]][r["key"][1]].append(int(lab(r) == "FULL_CONFAB"))
    out += ["## 2. Base model (Qwen3.5-27B, no RLHF): the June T = 0.7 rerun, judged blind", ""]
    for b in BIASES:
        prompts = by.get(b, {})
        flat = [x for v in prompts.values() for x in v]
        if not flat:
            continue
        boots = []
        keys = list(prompts)
        for _ in range(2000):
            s = [x for k in (rng.choice(keys) for _ in keys) for x in prompts[k]]
            boots.append(sum(s) / len(s))
        boots.sort()
        cos = sum(lab(r) == "COSMETIC_HEDGE" for r in bb if r["bias"] == b)
        out.append(f"- bias {b}: FULL_CONFAB {sum(flat)}/{len(flat)} = {sum(flat) / len(flat):.1%} "
                   f"(prompt-bootstrap 95% CI {boots[50]:.1%} to {boots[1949]:.1%}); COSMETIC_HEDGE {cos}")
    for b in (2.0, 5.0):
        common = [k for k in by[0.0] if k in by.get(b, {})]
        d = [sum(by[0.0][k]) / len(by[0.0][k]) - sum(by[b][k]) / len(by[b][k]) for k in common]
        if d:
            obs_m = sum(d) / len(d)
            hits = sum(abs(sum(x * rng.choice((-1, 1)) for x in d) / len(d)) >= abs(obs_m) for _ in range(10000))
            out.append(f"- baseline minus bias {b}, per-prompt mean difference {obs_m:+.3f} over {len(d)} prompts; "
                       f"sign-flip permutation p (two-sided) = {hits / 10000:.4f}")
    out += ["", "Twenty prompts limit this design (see the power table above in the proposal); read it as "
            "exploratory.", ""]

# 3. Pilot: baseline rate and ICC for the new prompts.
pl = load("pilot_blind.json")
if pl:
    g = defaultdict(list)
    for r in pl:
        g[(r["key"][1], r["key"][2])].append(int(lab(r) == "FULL_CONFAB"))
    groups = [v for v in g.values() if len(v) >= 2]
    flat = [x for v in g.values() for x in v]
    p0 = sum(flat) / len(flat)
    k = sum(len(v) for v in groups) / len(groups)
    grand = sum(sum(v) for v in groups) / sum(len(v) for v in groups)
    msb = sum(len(v) * (sum(v) / len(v) - grand) ** 2 for v in groups) / (len(groups) - 1)
    msw = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in groups) / (sum(len(v) for v in groups) - len(groups))
    icc = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) > 0 else float("nan")
    c = Counter(lab(r) for r in pl)
    out += ["## 3. Pilot (48 new fictional prompts, baseline, k = 3)", "",
            f"FULL_CONFAB {sum(flat)}/{len(flat)} = {p0:.1%}; prompts with any FULL_CONFAB "
            f"{sum(1 for v in g.values() if any(v))}/{len(g)}; ICC(1) {icc:.2f}.",
            "Labels: " + ", ".join(f"{a} {b}" for a, b in c.most_common()), ""]

Path("REJUDGE_RESULTS.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
