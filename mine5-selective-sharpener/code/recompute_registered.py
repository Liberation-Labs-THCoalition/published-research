#!/usr/bin/env python3
"""Recompute every result reported in the mine5 paper's Results (H1-H4), supplementary intervals included,
from the committed per-prompt data, and check each against the paper as printed. The correction note's
quotations of earlier, superseded values are not results; the one whose identity the note asserts
([+0.033, +0.083], "the paired 90% interval for d itself") is checked by re-running the analysis script's
own tost_equivalence verbatim.

    python code/recompute_registered.py          # run from anywhere; needs numpy and scipy

Data:  data/mine5_three_model_20260715_021501.json  (per-prompt features; produced by code/mine5_three_model.py)
Prereg: data/mine5_three_model_prereg.json          (the registered criteria applied here)

A check passes only if (1) the recomputed value, formatted as the paper prints it, equals the printed string,
and (2) that printed string occurs in BOTH editions (main.tex and academic/main.tex), bounded by non-digits
(so "1.08" is not found inside "1.088"), with any run of whitespace matching a space (LaTeX line breaks).
The H2 abliteration-check statements are checked against the data file and code/mine5_three_model.py. One PLANTED false value
is included on purpose; the run succeeds only if it is caught. A checker that cannot fail is not a checker.

Exit 0: data hashes match, every real check passes, and the planted value is caught. Exit 1 otherwise.

Conventions (as in the paper): Cohen's d = mean difference / sqrt(mean of the two ddof=1 variances), the
analysis script's formula. Cross-model comparisons use raw values: the same prompts, with identical token
counts, run through every model. Supplementary intervals: BCa bootstrap, 1000 draws, seed 12345, resampling
prompts while keeping each prompt matched across models (confab and factual prompts resampled separately).
Permutation p-values replicate the analysis script's label-shuffle test (10,000 permutations, seed 42),
Holm-corrected across the three cross-model comparisons.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
DATA = PAPER / "data" / "mine5_three_model_20260715_021501.json"
PREREG = PAPER / "data" / "mine5_three_model_prereg.json"
EDITIONS = [PAPER / "main.tex", PAPER / "academic" / "main.tex"]
SHA = {DATA: "2d9509243cda4e4a11ef99612aed2c0b558dfae68e1d50e0ad3064aa8cdab237",
       PREREG: "57b1811bf76143e6e7eb6aafb082c9bd9621af26bd93155dd2c595c4f0f02569"}
MODELS = ["base", "instruct", "abliterated"]
LAYERS = ["12", "15", "18", "21"]
SEED, B = 12345, 1000


def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)


def bca(stat, sets, level=0.95):
    """BCa interval. `sets` lists group sizes; each group is resampled independently, and the same indices are
    applied to every model's values for that group (paired across models)."""
    rng = np.random.RandomState(SEED)
    theta = stat([np.arange(n) for n in sets])
    boots = np.array([stat([rng.randint(0, n, n) for n in sets]) for _ in range(B)])
    z0 = norm.ppf((boots < theta).mean() + 0.5 * (boots == theta).mean())
    jack = []
    for s, n in enumerate(sets):
        for i in range(n):
            ix = [np.arange(k) for k in sets]
            ix[s] = np.delete(ix[s], i)
            jack.append(stat(ix))
    jack = np.array(jack)
    jm = jack.mean()
    a = ((jm - jack) ** 3).sum() / (6 * (((jm - jack) ** 2).sum()) ** 1.5)
    adj = lambda zq: norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
    alpha = (1 - level) / 2
    return np.percentile(boots, [100 * adj(norm.ppf(alpha)), 100 * adj(norm.ppf(1 - alpha))])


def permutation_p(x, y, n=10000):
    """The analysis script's permutation_test, replicated exactly (in-place cumulative shuffles, seed 42)."""
    obs = abs(np.mean(x) - np.mean(y))
    comb = np.concatenate([x, y])
    rng = np.random.RandomState(42)
    count = 0
    for _ in range(n):
        rng.shuffle(comb)
        count += abs(np.mean(comb[:len(x)]) - np.mean(comb[len(x):])) >= obs
    return count / n


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [None] * len(ps)
    for rank, i in enumerate(order):
        out[i] = min(ps[i] * (len(ps) - rank), 1.0)
    for k in range(1, len(order)):
        out[order[k]] = max(out[order[k]], out[order[k - 1]])
    return out


def main():
    ok_hash = True
    for path, want in SHA.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"sha256 {path.relative_to(PAPER)}: {got} {'OK' if got == want else 'MISMATCH (want ' + want + ')'}")
        ok_hash &= got == want
    R = json.loads(DATA.read_text(encoding="utf-8"))["results"]
    for m in MODELS:
        for c in ("confab", "factual"):
            assert [r["prompt_idx"] for r in R[m][c]] == list(range(30)), (m, c)
            assert [r["n_tokens"] for r in R[m][c]] == [r["n_tokens"] for r in R["base"][c]], (m, c)
    E = {(m, c): np.array([r["logit_entropy"] for r in R[m][c]]) for m in MODELS for c in ("confab", "factual")}
    SR = {(m, c): np.array([np.mean([r["spectral"][l]["stable_rank"] for l in LAYERS]) for r in R[m][c]])
          for m in MODELS for c in ("confab", "factual")}
    SRL = {(m, l): np.concatenate([[r["spectral"][l]["stable_rank"] for r in R[m][c]] for c in ("confab", "factual")])
           for m in MODELS for l in LAYERS}
    tex = [p.read_text(encoding="utf-8") for p in EDITIONS]

    f2 = lambda v: f"{v:.2f}"
    f3 = lambda v: f"{v:.3f}"
    s3 = lambda v: f"{v:+.3f}"
    f4 = lambda v: f"{v:.4f}"
    iv = lambda lohi: f"{lohi[0]:+.3f}, {lohi[1]:+.3f}"
    ivu = lambda lohi: f"{lohi[0]:.3f}, {lohi[1]:.3f}"
    pct = lambda v: f"{100 * v:.0f}\\%"
    checks = []  # (label, value, formatter, printed-in-paper, planted)

    def add(label, value, fmt, printed, planted=False):
        checks.append((label, value, fmt, printed, planted))

    # ---- H1 and the cross-model table -------------------------------------------------------------------
    pairs = [("base", "instruct"), ("base", "abliterated"), ("instruct", "abliterated")]
    raw_p = [permutation_p(E[a, "confab"], E[b, "confab"]) for a, b in pairs]
    holm_p = holm(raw_p)
    printed_table = {("base", "instruct"): ("0.684", "+0.478, +0.978", "0.021"),
                     ("base", "abliterated"): ("0.746", "+0.529, +1.063", "0.015"),
                     ("instruct", "abliterated"): ("0.055", "+0.031, +0.094", "0.831")}
    for (a, b), hp in zip(pairs, holm_p):
        x, y = E[a, "confab"], E[b, "confab"]
        pd, pci, pp = printed_table[a, b]
        add(f"table: {a} vs {b} confab d", cohens_d(x, y), f3, pd)
        add(f"table: {a} vs {b} paired BCa 95% CI", bca(lambda ix, x=x, y=y: cohens_d(x[ix[0]], y[ix[0]]), [30]), iv, pci)
        add(f"table: {a} vs {b} Holm p", hp, f3, pp)
    # ---- H2 ---------------------------------------------------------------------------------------------
    add("H2: factual instruct vs abliterated d", cohens_d(E["instruct", "factual"], E["abliterated", "factual"]), f3, "-0.022")
    add("H2: factual base vs abliterated d", cohens_d(E["base", "factual"], E["abliterated", "factual"]), f3, "0.743")
    i_c, a_c = E["instruct", "confab"], E["abliterated", "confab"]
    add("H2 footnote: paired BCa 90% CI, instruct vs abliterated",
        bca(lambda ix: cohens_d(i_c[ix[0]], a_c[ix[0]]), [30], level=0.90), iv, "+0.035, +0.084")
    # the correction note says the old "[+0.033, +0.083] nats" is the analysis script's paired 90% TOST interval
    # for d. Re-run that function exactly as written in code/mine5_three_model.py (same-seed resampling = paired).
    def script_cohens_d_bootstrap(g1, g2, n_boot):
        d0 = (np.mean(g1) - np.mean(g2)) / np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2)
        rng = np.random.RandomState(42)
        for _ in range(n_boot):  # consume the rng exactly as the script does; only d0 is used by the TOST
            rng.choice(g1, size=len(g1), replace=True); rng.choice(g2, size=len(g2), replace=True)
        return d0

    tost = np.percentile([script_cohens_d_bootstrap(np.random.RandomState(k + 100).choice(i_c, len(i_c), replace=True),
                                                    np.random.RandomState(k + 100).choice(a_c, len(a_c), replace=True),
                                                    100) for k in range(1000)], [5, 95])
    add("note: script's own TOST 90% interval for d (the old 'nats' interval)", tost, iv, "+0.033, +0.083")
    add("H2 footnote: mean paired difference (nats)", np.mean(i_c - a_c), f3, "0.073")
    add("H2 footnote: per-prompt correlation r", np.corrcoef(i_c, a_c)[0, 1], f3, "0.997")
    t = stats.ttest_rel(i_c, a_c)
    add("H2 footnote: paired t(29)", t.statistic, f2, "3.74")
    add("H2 footnote: paired t p-value", t.pvalue, f4, "0.0008")
    # ---- H2: the abliteration keyword check, scoped to what was measured -------------------------------
    code_src = (HERE / "mine5_three_model.py").read_text(encoding="utf-8")
    chk = {m: R[m]["abliteration_check"] for m in MODELS}
    words = {3: "three"}
    add("H2 check: safety prompts run on the abliterated model", len(chk["abliterated"]),
        lambda v: f"{words.get(v, v)} safety-relevant prompts", "three safety-relevant prompts")
    add("H2 check: other models (base, instruct) with any check", sum(chk[m] is not None for m in ("base", "instruct")),
        lambda v: "No instruct-model comparison was run" if v == 0 else f"{v} other model(s) checked",
        "No instruct-model comparison was run")
    add("H2 check: keyword refusals found", sum(bool(r["refuses"]) for r in chk["abliterated"]),
        lambda v: "no refusal wording in any of the three" if v == 0 else f"{v} refusal(s)",
        "no refusal wording in any of the three")
    mt = re.search(r"max_new_tokens=(\d+)", code_src)
    add("H2 check: tokens generated per prompt (code/mine5_three_model.py)", int(mt.group(1)),
        lambda v: f"generating {v} tokens each", "generating 30 tokens each")
    add("H2 check: chat template used by the script", "apply_chat_template" in code_src,
        lambda v: "chat template used" if v else "raw completions with no chat template",
        "raw completions with no chat template")
    # ---- H3 ---------------------------------------------------------------------------------------------
    ratio = {m: E[m, "confab"].mean() / E[m, "factual"].mean() for m in MODELS}
    for m, pc, pf, pr in (("base", "3.605", "1.834", "1.97"), ("instruct", "2.682", "1.067", "2.51"),
                          ("abliterated", "2.609", "1.088", "2.40")):
        add(f"H3 table: {m} mean confab entropy", E[m, "confab"].mean(), f3, pc)
        add(f"H3 table: {m} mean factual entropy", E[m, "factual"].mean(), f3, pf)
        add(f"H3: {m} calibration ratio", ratio[m], f2, pr)
    spread = max(ratio.values()) / min(ratio.values()) - 1
    add("H3: max/min - 1 (registered criterion <= 0.10)", spread, f3, "0.279")
    add("H3: largest exceeds smallest by (abstract)", spread, pct, "28\\%")
    add("H3: instruct ratio over base, minus 1", ratio["instruct"] / ratio["base"] - 1, pct, "28\\%")
    add("H3: abliterated ratio over base, minus 1", ratio["abliterated"] / ratio["base"] - 1, pct, "22\\%")
    comp = lambda m, c: 1 - E[m, c].mean() / E["base", c].mean()
    add("H3: factual compression base->instruct", comp("instruct", "factual"), pct, "42\\%")
    add("H3: confab compression base->instruct", comp("instruct", "confab"), pct, "26\\%")
    add("H3 table: factual compression base->abliterated", comp("abliterated", "factual"), pct, "41\\%")

    def ratios(ix):
        return {m: E[m, "confab"][ix[0]].mean() / E[m, "factual"][ix[1]].mean() for m in MODELS}

    add("H3 supp: BCa 95% CI on max/min - 1",
        bca(lambda ix: (lambda r: max(r.values()) / min(r.values()) - 1)(ratios(ix)), [30, 30]), ivu, "0.044, 0.538")
    add("H3 supp: ratio change instruct - base", ratio["instruct"] - ratio["base"], s3, "+0.547")
    add("H3 supp: BCa 95% CI, ratio change instruct - base",
        bca(lambda ix: (lambda r: r["instruct"] - r["base"])(ratios(ix)), [30, 30]), iv, "+0.027, +1.165")
    cdiff = lambda ix: ((1 - E["instruct", "factual"][ix[1]].mean() / E["base", "factual"][ix[1]].mean())
                        - (1 - E["instruct", "confab"][ix[0]].mean() / E["base", "confab"][ix[0]].mean()))
    add("H3 supp: compression difference factual - confab", cdiff([np.arange(30)] * 2), s3, "+0.162")
    add("H3 supp: BCa 95% CI, compression difference", bca(cdiff, [30, 30]), iv, "+0.010, +0.270")
    # ---- H4 ---------------------------------------------------------------------------------------------
    sr_b = np.concatenate([SR["base", "confab"], SR["base", "factual"]])
    sr_i = np.concatenate([SR["instruct", "confab"], SR["instruct", "factual"]])
    add("H4: base mean stable rank (60 prompts)", sr_b.mean(), f3, "1.963")
    add("H4: instruct mean stable rank (60 prompts)", sr_i.mean(), f3, "1.683")
    add("H4: base vs instruct d, 60 prompts (registered criterion d > 0.3)", cohens_d(sr_b, sr_i), f2, "1.04")
    add("H4: base vs instruct d, confab", cohens_d(SR["base", "confab"], SR["instruct", "confab"]), f2, "1.44")
    add("H4: base vs instruct d, factual", cohens_d(SR["base", "factual"], SR["instruct", "factual"]), f2, "1.08")
    per_layer = [cohens_d(SRL["base", l], SRL["instruct", l]) for l in LAYERS]
    add("H4: smallest per-layer d", min(per_layer), f2, "0.63")
    add("H4: largest per-layer d", max(per_layer), f2, "1.22")
    sr4 = lambda ix: cohens_d(np.concatenate([SR["base", "confab"][ix[0]], SR["base", "factual"][ix[1]]]),
                              np.concatenate([SR["instruct", "confab"][ix[0]], SR["instruct", "factual"][ix[1]]]))
    add("H4 supp: BCa 95% CI, base vs instruct d (60 prompts)", bca(sr4, [30, 30]), iv, H4_CI_PRINTED)
    sr_a = np.concatenate([SR["abliterated", "confab"], SR["abliterated", "factual"]])
    add("H4: base vs abliterated d, 60 prompts", cohens_d(sr_b, sr_a), f2, "1.05")

    # ---- correction note: the old H4 numbers, traced to their real source --------------------------------
    # They are WITHIN-model confab-vs-factual stable-rank effects after the analysis script's FWL (fitted on the
    # model's 60 prompts), printed by code/mine5_three_model.py, not the cross-model comparison H4 registers.
    def fwl_resid(v, n):
        X = np.column_stack([n, np.ones(len(n))])
        return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]

    def within(m):
        v = np.concatenate([SR[m, "confab"], SR[m, "factual"]])
        n = np.array([r["n_tokens"] for c in ("confab", "factual") for r in R[m][c]], float)
        res = fwl_resid(v, n)
        return res[:30], res[30:]

    add("note: instruct within-model confab-vs-factual stable-rank d", cohens_d(*within("instruct")), f3, "-0.219")
    add("note: instruct within-model permutation p", permutation_p(*within("instruct")), f2, "0.41")
    add("note: abliterated within-model confab-vs-factual stable-rank d", cohens_d(*within("abliterated")), f3, "-0.235")
    # ---- PLANTED: a deliberately false printed value. It MUST be reported as a failure. -----------------
    add("PLANTED: H3 spread printed as 0.297 (true value 0.279)", spread, f3, "0.297", planted=True)

    # ---- run ----------------------------------------------------------------------------------------------
    real_fail, planted_caught = 0, False
    for label, value, fmt, printed, planted in checks:
        got = fmt(value)
        value_ok = got == printed
        pat = re.compile(r"(?<![0-9])" + r"\s+".join(map(re.escape, printed.split())) + r"(?![0-9])")
        in_tex = all(pat.search(t) for t in tex)
        ok = value_ok and (in_tex or planted)
        tag = "OK  " if ok else "FAIL"
        note = "" if in_tex or planted else "  (printed string not found in both editions)"
        print(f"  {tag} {label}: recomputed {got} | paper {printed}{note}")
        if planted:
            planted_caught = not ok
        elif not ok:
            real_fail += 1

    # report-only quantities (not printed in the paper; reported in the PR for H1's FWL-corrected criterion)
    tok = {c: np.array([r["n_tokens"] for r in R["base"][c]], float) for c in ("confab", "factual")}

    def fwl_keep_mean(v, n):
        X = np.column_stack([n, np.ones(len(n))])
        b = np.linalg.lstsq(X, v, rcond=None)[0]
        return v - b[0] * (n - n.mean())

    xb, xi = fwl_keep_mean(E["base", "confab"], tok["confab"]), fwl_keep_mean(E["instruct", "confab"], tok["confab"])
    print(f"\nreport only (H1, FWL-corrected: token-count slope removed within each model, mean kept):"
          f" d = {cohens_d(xb, xi):+.3f}, label-shuffle p = {permutation_p(xb, xi):.4f};"
          f" raw label-shuffle p (before Holm) = {raw_p[0]:.4f}")
    print(f"report only (H4, base vs abliterated, 60 prompts): d = "
          f"{cohens_d(sr_b, np.concatenate([SR['abliterated', 'confab'], SR['abliterated', 'factual']])):+.3f}")

    print(f"\nreal checks: {sum(1 for c in checks if not c[4])}, failures: {real_fail}; "
          f"planted false value caught: {planted_caught}; data hashes OK: {ok_hash}")
    if not planted_caught:
        print("CHECKER BROKEN: the planted false value passed.")
    sys.exit(0 if (ok_hash and real_fail == 0 and planted_caught) else 1)


H4_CI_PRINTED = "+0.893, +1.224"

if __name__ == "__main__":
    main()
