"""Same weights, different words, analysis v2: answers Agni's paper-phase review (AGNI_paper_v1.md) from data on hand.

New relative to analyze.py / analyze2.py:
 A. data checks: token counts of the stored and forced passes, natural reply lengths and stop reasons (F7);
 B. pooled medians by numpy, per-condition gate counts, the 2C combination (F10, F11);
 C. how many tokens the stored 200-character preview covers, decoded from token ids with a byte-level BPE decoder
    built from tokenizer.json, validated by reproducing every natural reply's stored text exactly (F2);
 D. the clean test: trials where MPS bf16 and CPU fp32 FORCED THE SAME 100 TOKENS (the forced pass is the one features
    come from). On those, |MPS - CPU| in encoding, generation and delta stable rank is the cross-hardware floor on
    identical tokens, for the gated statistic itself (F1). Contrast: trials whose forced tokens diverged (F2);
 E. MPS vs CPU encoding directly (F3, partly);
 F. Fisher exact tests for the 2x2 counts quoted (F2);
 G. per layer: the stored spread of generation stable rank, and the identical-token floor per layer (F6).
Counts and numbers only, no text. Writes results3.json beside this script. Lyra, 2026-11-21."""
import glob, json, math, os
import numpy as np

MIR = os.path.expanduser("~/lab/decline-signature-mirror")
RUNS = {"mps_bf16": f"{MIR}/run_20261121/run_dir/a1", "cpu_fp32": f"{MIR}/run_20261121_cpu_retry/run_dir/a1"}
STORED = os.path.expanduser("~/lab/sameweights_20261121/stored/Qwen2.5-7B-Instruct_honesty.json")
TOKJ = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/tokenizer.json")))
TOL = 0.01
PH = ("encoding_features", "generation_features", "delta_features")


def qs(xs):
    xs = np.asarray([x for x in xs if x is not None], dtype=float)
    if xs.size == 0:
        return None
    p = np.percentile(xs, [0, 25, 50, 75, 100])
    return {"n": int(xs.size), "min": p[0], "p25": p[1], "median": p[2], "p75": p[3], "max": p[4]}


def within(xs):
    return f"{sum(1 for x in xs if abs(x) <= TOL)}/{len(xs)}"


def fisher(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2
    pr = lambda x: math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = pr(a)
    return sum(pr(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9))


def first_div(a, b):
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if i is None and len(a) != len(b):
        i = min(len(a), len(b))
    return i


# ---- byte-level BPE decoder (GPT-2 byte map), from tokenizer.json
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


TJ = json.load(open(TOKJ[-1], encoding="utf-8"))
INV = {i: t for t, i in TJ["model"]["vocab"].items()}
ADDED = {a["id"]: a for a in TJ.get("added_tokens", [])}
U2B = bytes_to_unicode()


def tok_bytes(i, skip_special=True):
    if i in ADDED:
        return b"" if (skip_special and ADDED[i].get("special")) else ADDED[i]["content"].encode("utf-8")
    return bytes(U2B[ch] for ch in INV[i])


def decode(ids):
    return b"".join(tok_bytes(i) for i in ids).decode("utf-8", errors="replace")


def tokens_to_cover(ids, nchars):
    """Smallest k with len(decode(ids[:k])) >= nchars; None if the whole reply is shorter."""
    buf = b""
    for k, i in enumerate(ids, 1):
        buf += tok_bytes(i)
        if len(buf.decode("utf-8", errors="ignore")) >= nchars:
            return k
    return None


R = {dev: {j["key"]: j for j in (json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.json")))}
     for dev, d in RUNS.items()}
ST = json.load(open(STORED))
stored = {(t["condition"], t["prompt_idx"]): t for t in ST["trials"]}
res = {"tokenizer_json": TOKJ[-1].split("/snapshots/")[0].rsplit("/", 1)[-1]}

# A. data checks
res["A_stored_n_generated"] = sorted({t["n_generated"] for t in ST["trials"]})
for dev, rows in R.items():
    nat = [r["natural"] for r in rows.values()]
    res[f"A_{dev}"] = {
        "forced_n_generated": sorted({r["forced"]["record"]["n_generated"] for r in rows.values()}),
        "forced_ids_len": sorted({len(r["forced"]["forced_ids"]) for r in rows.values()}),
        "natural_stopped_by": {s: sum(1 for x in nat if x["stopped_by"] == s) for s in sorted({x["stopped_by"] for x in nat})},
        "natural_text_under_200_chars": sum(1 for x in nat if len(x["text"]) < 200),
        "prompt_token_count_matches_stored": sum(1 for r in rows.values() if r["gate"]["descriptive"]["n_prompt_match"]),
    }

# B. pooled medians and gate counts, overall and per condition
for dev, rows in R.items():
    for ph in PH:
        xs = [r["gate"]["descriptive"]["absdiff"][ph]["stable_rank"] for r in rows.values()]
        res[f"B_{dev}_{ph}_vs_stored"] = {"q": qs(xs), "within": within(xs)}
    for cond in ("confab", "honest_rare"):
        sub = [r for r in rows.values() if r["condition"] == cond]
        res[f"B_{dev}_gate_{cond}"] = {k: sum(1 for r in sub if r["gate"][k]) for k in ("text_ok", "feature_ok", "pass")}
        res[f"B_{dev}_gate_{cond}"]["n"] = len(sub)
for cond in ("confab", "honest_rare"):
    keys = [k for k, r in R["mps_bf16"].items() if r["condition"] == cond]
    res[f"B_2C_combined_{cond}"] = sum(1 for k in keys if R["mps_bf16"][k]["gate"]["pass"]
                                       or (k in R["cpu_fp32"] and R["cpu_fp32"][k]["gate"]["pass"]))

# C. tokens covered by the stored 200-character preview; decoder positive control
ok_dec, cover = 0, []
for dev, rows in R.items():
    for r in rows.values():
        ok_dec += decode(r["natural"]["reply_ids"]) == r["natural"]["text"]
        cover.append(tokens_to_cover(r["natural"]["reply_ids"], 200))
res["C_decoder_reproduces_stored_text"] = f"{ok_dec}/{sum(len(v) for v in R.values())}"
res["C_tokens_to_cover_200_chars"] = qs(cover)
res["C_replies_shorter_than_200_chars"] = sum(1 for c in cover if c is None)

# D. cross-device on the 56 shared trials, split by whether the FORCED tokens were identical
shared = sorted(set(R["mps_bf16"]) & set(R["cpu_fp32"]))
groups = {"forced_identical": [], "forced_diverged": []}
fd_forced, fd_nat, nat_same = [], [], 0
for k in shared:
    a, b = R["mps_bf16"][k], R["cpu_fp32"][k]
    d = first_div(a["forced"]["forced_ids"], b["forced"]["forced_ids"])
    groups["forced_identical" if d is None else "forced_diverged"].append(k)
    if d is not None:
        fd_forced.append(d)
    dn = first_div(a["natural"]["reply_ids"], b["natural"]["reply_ids"])
    if dn is None:
        nat_same += 1
    else:
        fd_nat.append(dn)
res["D_n_shared"] = len(shared)
res["D_natural_identical"] = nat_same
res["D_natural_first_divergent_token_among_diverged"] = qs(fd_nat)
res["D_forced_identical"] = len(groups["forced_identical"])
res["D_forced_first_divergent_token_among_diverged"] = qs(fd_forced)
res["D_forced_identical_by_condition"] = {c: sum(1 for k in groups["forced_identical"] if R["mps_bf16"][k]["condition"] == c)
                                          for c in ("confab", "honest_rare")}
for g, keys in groups.items():
    for ph in PH:
        xs = [R["mps_bf16"][k]["forced"]["record"][ph]["stable_rank"] - R["cpu_fp32"][k]["forced"]["record"][ph]["stable_rank"]
              for k in keys]
        res[f"D_{g}_{ph}_mps_minus_cpu"] = {"q_abs": qs([abs(x) for x in xs]), "within": within(xs),
                                           "signed_median": float(np.median(xs)) if xs else None}
    # and each device against stored, on the same split
    for dev in RUNS:
        xs = [R[dev][k]["gate"]["delta_sr_diff"] for k in keys]
        res[f"D_{g}_{dev}_delta_vs_stored"] = {"q_abs": qs([abs(x) for x in xs]), "within": within(xs)}

# E. MPS vs CPU encoding directly (same machine, same libraries, different device+dtype)
xs = [R["mps_bf16"][k]["forced"]["record"]["encoding_features"]["stable_rank"]
      - R["cpu_fp32"][k]["forced"]["record"]["encoding_features"]["stable_rank"] for k in shared]
res["E_encoding_mps_minus_cpu"] = {"q_abs": qs([abs(x) for x in xs]), "within": within(xs),
                                   "signed_median": float(np.median(xs))}

# F. Fisher tests
cpu = R["cpu_fp32"]
a = sum(1 for r in cpu.values() if r["gate"]["text_ok"] and r["gate"]["feature_ok"])
b = sum(1 for r in cpu.values() if r["gate"]["text_ok"] and not r["gate"]["feature_ok"])
c = sum(1 for r in cpu.values() if not r["gate"]["text_ok"] and r["gate"]["feature_ok"])
d = sum(1 for r in cpu.values() if not r["gate"]["text_ok"] and not r["gate"]["feature_ok"])
res["F_cpu_feature_ok_by_200char_match"] = {"table": [[a, b], [c, d]], "fisher_p": fisher(a, b, c, d)}
gi, gd = groups["forced_identical"], groups["forced_diverged"]
dl = lambda k: abs(R["mps_bf16"][k]["forced"]["record"]["delta_features"]["stable_rank"]
                   - R["cpu_fp32"][k]["forced"]["record"]["delta_features"]["stable_rank"]) <= TOL
t = [[sum(map(dl, gi)), len(gi) - sum(map(dl, gi))], [sum(map(dl, gd)), len(gd) - sum(map(dl, gd))]]
res["F_mps_cpu_delta_within_by_forced_identity"] = {"table": t, "fisher_p": fisher(t[0][0], t[0][1], t[1][0], t[1][1])}

# G. per layer: stored spread of generation stable rank (60 gated trials); identical-token floor per layer
gated = [stored[(r["condition"], r["prompt_idx"])] for r in R["mps_bf16"].values()]
lay = {}
for L in range(len(gated[0]["per_layer"])):
    v = np.array([t["per_layer"][L]["stable_rank"] for t in gated])
    fl = [abs(R["mps_bf16"][k]["forced"]["record"]["per_layer"][L]["stable_rank"]
              - R["cpu_fp32"][k]["forced"]["record"]["per_layer"][L]["stable_rank"]) for k in gi]
    lay[L] = {"stored_mean": float(v.mean()), "stored_sd": float(v.std(ddof=1)),
              "identical_token_floor_median": float(np.median(fl)) if fl else None,
              "identical_token_within": within(fl) if fl else None}
res["G_per_layer"] = lay
res["G_n_layers"] = len(lay)
for ph in PH:
    v = np.array([t[ph]["stable_rank"] for t in gated])
    res[f"G_stored_{ph}_stable_rank"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}

# H. is the identical-vs-diverged gap in |MPS - CPU| bigger than chance? Permutation test on the median gap.
rng = np.random.default_rng(20261121)
for ph in ("generation_features", "delta_features"):
    v = {k: abs(R["mps_bf16"][k]["forced"]["record"][ph]["stable_rank"] - R["cpu_fp32"][k]["forced"]["record"][ph]["stable_rank"])
         for k in shared}
    x = np.array([v[k] for k in gi]); y = np.array([v[k] for k in gd]); allv = np.concatenate([x, y])
    obs = np.median(y) - np.median(x)
    n_perm, hits = 200000, 0
    for _ in range(n_perm):
        p_ = rng.permutation(allv)
        hits += (np.median(p_[len(x):]) - np.median(p_[:len(x)])) >= obs
    res[f"H_{ph}_median_gap_diverged_minus_identical"] = {"gap": float(obs), "perm_p_one_sided": (hits + 1) / (n_perm + 1),
                                                          "n_perm": n_perm}

# I. the prompt check (F4): what the stored 'prompt' field is; and whether the 8 identical-token trials matched stored text
lens = [len(t["prompt"]) for t in ST["trials"]]
pre = sum(1 for r in R["mps_bf16"].values() if r["user_prompt"].startswith(stored[(r["condition"], r["prompt_idx"])]["prompt"]))
eq = sum(1 for r in R["mps_bf16"].values() if r["user_prompt"] == stored[(r["condition"], r["prompt_idx"])]["prompt"])
res["I_stored_prompt_field_chars"] = qs(lens)
res["I_stored_prompt_is_prefix_of_user_prompt"] = f"{pre}/{len(R['mps_bf16'])}"
res["I_stored_prompt_equals_user_prompt"] = f"{eq}/{len(R['mps_bf16'])}"
res["I_user_prompt_chars"] = qs([len(r["user_prompt"]) for r in R["mps_bf16"].values()])
res["I_identical8_text_ok"] = {dev: sum(1 for k in gi if R[dev][k]["gate"]["text_ok"]) for dev in RUNS}

# J. noise or bias? sign of the shift, rank preservation across trials, and the between-condition contrast
for dev in RUNS:
    for ph in PH:
        keys = sorted(R[dev])
        s = np.array([stored[(R[dev][k]["condition"], R[dev][k]["prompt_idx"])][ph]["stable_rank"] for k in keys])
        g = np.array([R[dev][k]["forced"]["record"][ph]["stable_rank"] for k in keys])
        d = g - s
        rs = lambda a: np.argsort(np.argsort(a))
        res[f"J_{dev}_{ph}"] = {"signed_median_regen_minus_stored": float(np.median(d)),
                                "frac_negative": float(np.mean(d < 0)),
                                "pearson_r_stored_vs_regen": float(np.corrcoef(s, g)[0, 1]),
                                "spearman_rho_stored_vs_regen": float(np.corrcoef(rs(s), rs(g))[0, 1])}
        cond = np.array([R[dev][k]["condition"] for k in keys])
        cs = lambda arr: float(arr[cond == "confab"].mean() - arr[cond == "honest_rare"].mean())
        res[f"J_{dev}_{ph}"]["contrast_confab_minus_honest_rare"] = {"stored": cs(s), "regen": cs(g)}
xs = np.array([R["mps_bf16"][k]["forced"]["record"]["encoding_features"]["stable_rank"]
               - R["cpu_fp32"][k]["forced"]["record"]["encoding_features"]["stable_rank"] for k in shared])
res["J_encoding_mps_minus_cpu_frac_negative"] = float(np.mean(xs < 0))

# K. how much did the between-condition contrast move? Per trial d = regen - stored; change in contrast =
#    mean(d | confab) - mean(d | honest_rare); 95% percentile bootstrap, resampling trials within condition.
rng = np.random.default_rng(20261122)
for dev in RUNS:
    keys = sorted(R[dev])
    cond = np.array([R[dev][k]["condition"] for k in keys])
    for ph in PH:
        s = np.array([stored[(R[dev][k]["condition"], R[dev][k]["prompt_idx"])][ph]["stable_rank"] for k in keys])
        d = np.array([R[dev][k]["forced"]["record"][ph]["stable_rank"] for k in keys]) - s
        dc, dh = d[cond == "confab"], d[cond == "honest_rare"]
        sc = s[cond == "confab"].mean() - s[cond == "honest_rare"].mean()
        boots = np.array([rng.choice(dc, dc.size).mean() - rng.choice(dh, dh.size).mean() for _ in range(20000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res[f"K_{dev}_{ph}_contrast_change"] = {"change": float(dc.mean() - dh.mean()), "ci95": [float(lo), float(hi)],
                                                "stored_contrast": float(sc),
                                                "change_as_frac_of_stored_contrast": float((dc.mean() - dh.mean()) / sc),
                                                "n": [int(dc.size), int(dh.size)]}

# L. added after Agni round 2 (AGNI_paper_v2.md), post-data:
# L1 within-condition rank and linear agreement, stored vs re-run, per device x condition x phase
for dev in RUNS:
    for cond in ("confab", "honest_rare"):
        keys = sorted(k for k in R[dev] if R[dev][k]["condition"] == cond)
        for ph in ("encoding_features", "delta_features"):
            s = np.array([stored[(cond, R[dev][k]["prompt_idx"])][ph]["stable_rank"] for k in keys])
            g = np.array([R[dev][k]["forced"]["record"][ph]["stable_rank"] for k in keys])
            rk = lambda a: np.argsort(np.argsort(a))
            res[f"L1_{dev}_{cond}_{ph}"] = {"n": len(keys), "spearman": float(np.corrcoef(rk(s), rk(g))[0, 1]),
                                            "pearson": float(np.corrcoef(s, g)[0, 1])}


# L2 Hedges g, honest_rare vs confab on delta stable rank (the published effect: g 2.322 on the stored 60),
#    stored vs re-run on each device's trial set; paired bootstrap of the difference (same resampled trials for both)
def hedges_g(hr, cf):
    n1, n2 = len(hr), len(cf)
    sp = math.sqrt(((n1 - 1) * np.var(hr, ddof=1) + (n2 - 1) * np.var(cf, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(hr) - np.mean(cf)) / sp * (1 - 3 / (4 * (n1 + n2) - 9)))


all60 = [stored[(r["condition"], r["prompt_idx"])] for r in R["mps_bf16"].values()]
res["L2_positive_control_stored_g_60"] = hedges_g(
    np.array([t["delta_features"]["stable_rank"] for t in all60 if t["condition"] == "honest_rare"]),
    np.array([t["delta_features"]["stable_rank"] for t in all60 if t["condition"] == "confab"]))
rng = np.random.default_rng(20261123)
for dev in RUNS:
    keys = sorted(R[dev])
    cond = np.array([R[dev][k]["condition"] for k in keys])
    s = np.array([stored[(R[dev][k]["condition"], R[dev][k]["prompt_idx"])]["delta_features"]["stable_rank"] for k in keys])
    g = np.array([R[dev][k]["forced"]["record"]["delta_features"]["stable_rank"] for k in keys])
    ih, ic = np.where(cond == "honest_rare")[0], np.where(cond == "confab")[0]
    gs, gr = hedges_g(s[ih], s[ic]), hedges_g(g[ih], g[ic])
    bd = []
    for _ in range(20000):
        bh, bc = rng.choice(ih, ih.size), rng.choice(ic, ic.size)
        bd.append(hedges_g(g[bh], g[bc]) - hedges_g(s[bh], s[bc]))
    lo, hi = np.percentile(bd, [2.5, 97.5])
    res[f"L2_{dev}_hedges_g_delta"] = {"stored": gs, "regen": gr, "change": gr - gs, "ci95": [float(lo), float(hi)],
                                       "n": [int(ih.size), int(ic.size)]}

# L3 the negative controls (one prompt token dropped; Stage A's must-fail check), as the runs recorded them
for dev, rd in (("mps_bf16", "run_20261121"), ("cpu_fp32", "run_20261121_cpu_retry")):
    nc = json.load(open(f"{MIR}/{rd}/run_dir/negative_control.json"))
    res[f"L3_{dev}_negative_control"] = {"key": nc["key"], "delta_sr_diff": nc["gate"]["delta_sr_diff"],
                                         "encoding_sr_absdiff": nc["gate"]["descriptive"]["absdiff"]["encoding_features"]["stable_rank"],
                                         "text_ok": nc["gate"]["text_ok"], "caught_by": nc["caught_by"]}

# L4 tokens the stored 200 characters cover, one reply per trial, using only replies whose 200 characters match stored
cov = []
for k, r in R["mps_bf16"].items():
    src = r if r["gate"]["text_ok"] else (R["cpu_fp32"].get(k) if k in R["cpu_fp32"] and R["cpu_fp32"][k]["gate"]["text_ok"] else None)
    if src is not None:
        cov.append(tokens_to_cover(src["natural"]["reply_ids"], 200))
res["L4_tokens_to_cover_200_chars_text_ok_one_per_trial"] = qs(cov)

# L5 within-condition SD (pooled over the two conditions) of the stored values, 60 gated trials
for ph in PH:
    by = {c: np.array([t[ph]["stable_rank"] for t in all60 if t["condition"] == c]) for c in ("confab", "honest_rare")}
    res[f"L5_stored_{ph}_within_condition_sd"] = float(math.sqrt((np.var(by["confab"], ddof=1) + np.var(by["honest_rare"], ddof=1)) / 2))

# L6 counts quoted in the note that v2 took from results.json / results2.json
res["L6_forced_first_div_index_below_10_0based"] = sum(1 for d in fd_forced if d < 10)
res["L6_mps_vs_cpu_identical_first_200_chars"] = sum(
    1 for k in shared if R["mps_bf16"][k]["natural"]["text"][:200] == R["cpu_fp32"][k]["natural"]["text"][:200])

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results3.json")
json.dump(res, open(out, "w"), indent=1, default=float)
for k, v in res.items():
    if k != "G_per_layer":
        print(k, json.dumps(v, default=float))
print("G_per_layer (layer: stored sd | identical-token floor median | within)")
for L, v in lay.items():
    print(L, round(v["stored_sd"], 4), v["identical_token_floor_median"] and round(v["identical_token_floor_median"], 4),
          v["identical_token_within"])
