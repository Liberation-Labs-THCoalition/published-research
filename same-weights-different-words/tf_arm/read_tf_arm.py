"""Read the teacher-forced arm exactly as tf_arm/READING_RULE.md Amendment 1 (md5 b64fabe4) fixed it before any output.
Inputs: the four arm JSONs (MTH mirror), the Stage A a1 records (for which trials forced identical tokens), the stored
JSON. Numbers only. Lyra, 2026-11-22."""
import glob, json, math, os
import numpy as np

MIR = os.path.expanduser("~/lab/decline-signature-mirror")
ARM = f"{MIR}/tf_arm_20261121/out"
CFG = ("mps_bfloat16", "cpu_float32", "mps_float32", "cpu_bfloat16")
TOL = 0.01
A = {c: json.load(open(f"{ARM}/{c}.json")) for c in CFG}
res = {}

# validity (Amendment 1), plus Nexus's finding: n_shared must be 56, or "complete" means nothing
res["validity"] = {c: {"complete": a["complete"], "refused": len(a["refused"]), "n_shared": a["n_shared"],
                       "n_trials": len(a["trials"]), "self_check_max": a.get("self_check_max_absdiff"),
                       "self_check_failed": a.get("self_check_failed")} for c, a in A.items()}
for c in ("mps_bfloat16", "cpu_float32"):
    sc = [t["self_check_absdiff"] for t in A[c]["trials"].values()]
    res["validity"][c]["self_check_distribution"] = {"min": min(sc), "max": max(sc), "n_zero": sum(1 for x in sc if x == 0.0)}
valid = all(v["complete"] and v["refused"] == 0 and v["n_shared"] == 56 and v["n_trials"] == 56 for v in res["validity"].values()) \
    and all(res["validity"][c]["self_check_max"] <= 1e-3 for c in ("mps_bfloat16", "cpu_float32"))
res["all_valid"] = valid

# which trials forced identical tokens in Stage A (the 8 already seen) vs not (the 48: the primary)
def load(d):
    return {j["key"]: j for j in (json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.json")))}
M = load(f"{MIR}/run_20261121/run_dir/a1"); C = load(f"{MIR}/run_20261121_cpu_retry/run_dir/a1")
keys = sorted(A["mps_bfloat16"]["trials"])
same = {k for k in keys if M[k]["forced"]["forced_ids"] == C[k]["forced"]["forced_ids"]}
seen8, unseen48 = sorted(same), sorted(set(keys) - same)
res["n_seen"], res["n_unseen"] = len(seen8), len(unseen48)

def dsr(cfg, k, src):
    return A[cfg]["trials"][k]["sources"][src]["delta_sr"]

def gap(k, src):
    return abs(dsr("mps_bfloat16", k, src) - dsr("cpu_float32", k, src))

def boot_median(x, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    b = np.array([np.median(rng.choice(x, x.size)) for _ in range(20000)])
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]

def reading(ci):
    return "above tolerance" if ci[0] > TOL else ("below tolerance" if ci[1] < TOL else "undecided")

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]

def fisher(a, b, c, d):
    r1, r2, c1 = a + b, c + d, a + c; n = r1 + r2
    pr = lambda x: math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = pr(a)
    return sum(pr(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9))

# PRIMARY: the 48, CPU fp32's tokens, median gap, bootstrap seed 20261124
g = [gap(k, "cpu_fp32_ids") for k in unseen48]
ci = boot_median(g, 20261124)
res["PRIMARY"] = {"n": len(g), "median_gap": float(np.median(g)), "ci95": ci, "reading": reading(ci),
                  "signed_median_mps_minus_cpu": float(np.median([dsr("mps_bfloat16", k, "cpu_fp32_ids") - dsr("cpu_float32", k, "cpu_fp32_ids") for k in unseen48])),
                  "range": [float(min(g)), float(max(g))]}

# secondary: MPS tokens as the source
g2 = [gap(k, "mps_bf16_ids") for k in unseen48]
ci2 = boot_median(g2, 20261124)
res["S_mps_ids_source"] = {"median_gap": float(np.median(g2)), "ci95": ci2, "would_read": reading(ci2)}

# secondary: per condition, within 0.01, Wilson, on the 48 and on all 56 (CPU tokens)
for label, ks in (("unseen48", unseen48), ("all56", keys)):
    for cond in ("confab", "honest_rare"):
        sub = [k for k in ks if A["mps_bfloat16"]["trials"][k]["condition"] == cond]
        w = sum(1 for k in sub if gap(k, "cpu_fp32_ids") <= TOL)
        res[f"S_within_{label}_{cond}"] = {"within": w, "n": len(sub), "wilson95": wilson(w, len(sub)),
                                           "gate_bar": 28 / 30}

# secondary: the 8 against the 48 under teacher forcing; and the arm reproducing analysis v2 on the 8
g8 = [gap(k, "cpu_fp32_ids") for k in seen8]
w8, w48 = sum(1 for x in g8 if x <= TOL), sum(1 for x in g if x <= TOL)
res["S_seen8_vs_unseen48"] = {"median8": float(np.median(g8)), "median48": float(np.median(g)),
                              "within8": f"{w8}/8", "within48": f"{w48}/48",
                              "fisher_p": fisher(w8, 8 - w8, w48, 48 - w48)}
res["S_arm_reproduces_v2_on_8_median"] = float(np.median(
    [abs(M[k]["forced"]["record"]["delta_features"]["stable_rank"] - C[k]["forced"]["record"]["delta_features"]["stable_rank"]) for k in seen8]))

# secondary: device vs dtype, on identical tokens (encoding: prompt; delta: CPU fp32's tokens), median |diff| over 56
def med(f):
    return float(np.median([f(k) for k in keys]))
enc = lambda c, k: A[c]["trials"][k]["encoding_sr"]
for name, val in (("encoding", lambda c, k: enc(c, k)), ("delta", lambda c, k: dsr(c, k, "cpu_fp32_ids"))):
    pair = lambda a, b: med(lambda k: abs(val(a, k) - val(b, k)))
    dev_bf16, dev_fp32 = pair("mps_bfloat16", "cpu_bfloat16"), pair("mps_float32", "cpu_float32")
    dt_mps, dt_cpu = pair("mps_bfloat16", "mps_float32"), pair("cpu_bfloat16", "cpu_float32")
    attr = ("dtype" if min(dt_mps, dt_cpu) >= 2 * max(dev_bf16, dev_fp32) else
            "device" if min(dev_bf16, dev_fp32) >= 2 * max(dt_mps, dt_cpu) else "neither (rule not met)")
    res[f"S_device_vs_dtype_{name}"] = {"device_at_bf16": dev_bf16, "device_at_fp32": dev_fp32,
                                        "dtype_on_mps": dt_mps, "dtype_on_cpu": dt_cpu,
                                        "mps_bf16_vs_cpu_fp32": pair("mps_bfloat16", "cpu_float32"),
                                        "mps_fp32_vs_cpu_bf16": pair("mps_float32", "cpu_bfloat16"),
                                        "attribution_by_rule": attr}

# secondary: which configurations sit within 0.003 of the stored run on encoding (median |diff| over the 56)
ST = json.load(open(os.path.expanduser("~/lab/sameweights_20261121/stored/Qwen2.5-7B-Instruct_honesty.json")))
stored = {(t["condition"], t["prompt_idx"]): t for t in ST["trials"]}
sk = lambda k: stored[(M[k]["condition"], M[k]["prompt_idx"])]["encoding_features"]["stable_rank"]
res["S_encoding_vs_stored"] = {}
for c in CFG:
    d = [enc(c, k) - sk(k) for k in keys]
    res["S_encoding_vs_stored"][c] = {"median_abs": float(np.median(np.abs(d))), "signed_median": float(np.median(d)),
                                      "within_0.003_by_median": bool(np.median(np.abs(d)) <= 0.003)}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tf_arm_reading.json")
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items():
    print(k, json.dumps(v))
