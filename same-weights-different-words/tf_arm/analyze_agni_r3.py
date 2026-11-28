"""Numbers for Agni round 3's fixes (AGNI_paper_v4.md), computed from data already on disk. Lyra, 2026-11-23.

F1  the true one-token perturbation effects: perturbed re-run minus UNPERTURBED re-run on the same configuration
    (the field quoted before, gate.delta_sr_diff, is perturbed re-run minus STORED);
F3  the gate-relevant floor: the 93.3rd percentile (28/30) of the identical-token gap on the 48, with a bootstrap CI;
F7  MPS rho on CPU's 27/29 trial set, beside CPU's;
F10 the encoding -> delta decomposition on all 56 trials from the arm (signed medians, CPU fp32's tokens), and a paired
    sign-flip test on the 48: identical-token gap vs the re-runs' own-token gap;
F11 the primary's bootstrap with CPU bf16 and MPS fp32 as references.
Run on MTH from an empty folder: python3 -I analyze_agni_r3.py . Writes agni_r3_numbers.json beside this script."""
import glob, json, os
import numpy as np

MIR = os.path.expanduser("~/lab/decline-signature-mirror")
ARM = f"{MIR}/tf_arm_20261121/out"
STORED = os.path.expanduser("~/lab/sameweights_20261121/stored/Qwen2.5-7B-Instruct_honesty.json")
TOL = 0.01
res = {}


def load(d):
    return {j["key"]: j for j in (json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.json")))}


M = load(f"{MIR}/run_20261121/run_dir/a1")
C = load(f"{MIR}/run_20261121_cpu_retry/run_dir/a1")
A = {c: json.load(open(f"{ARM}/{c}.json")) for c in ("mps_bfloat16", "cpu_float32", "mps_float32", "cpu_bfloat16")}
ST = json.load(open(STORED))
stored = {(t["condition"], t["prompt_idx"]): t for t in ST["trials"]}

# F1: true perturbation effects
for dev, rd, a1 in (("mps_bf16", "run_20261121", M), ("cpu_fp32", "run_20261121_cpu_retry", C)):
    nc = json.load(open(f"{MIR}/{rd}/run_dir/negative_control.json"))
    k = nc["key"]
    pert = nc["gate"]["delta_sr_regen"]
    unpert = a1[k]["gate"]["delta_sr_regen"]
    st = stored[(a1[k]["condition"], a1[k]["prompt_idx"])]["delta_features"]["stable_rank"]
    res[f"F1_{dev}"] = {"key": k, "perturbed_regen": pert, "unperturbed_regen": unpert, "stored": st,
                        "true_effect_perturbed_minus_unperturbed": pert - unpert,
                        "old_field_perturbed_minus_stored": nc["gate"]["delta_sr_diff"],
                        "unperturbed_minus_stored": unpert - st,
                        "unperturbed_passed_gate": a1[k]["gate"]["pass"]}

# identical-token gaps (CPU fp32's tokens) on the 48 never seen identical, and all 56
keys = sorted(A["mps_bfloat16"]["trials"])
same = {k for k in keys if M[k]["forced"]["forced_ids"] == C[k]["forced"]["forced_ids"]}
u48 = [k for k in keys if k not in same]
dsr = lambda c, k, s="cpu_fp32_ids": A[c]["trials"][k]["sources"][s]["delta_sr"]
gap = lambda k, ref="cpu_float32": abs(dsr("mps_bfloat16", k) - dsr(ref, k))
rng = np.random.default_rng(20261125)


def boot(x, stat, n=20000):
    x = np.asarray(x)
    b = np.array([stat(rng.choice(x, x.size)) for _ in range(n)])
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


g48 = np.array([gap(k) for k in u48])
q = lambda a: float(np.quantile(a, 28 / 30))
res["F3_floor_q0933_48"] = {"value": q(g48), "ci95": boot(g48, q), "median": float(np.median(g48)), "max": float(g48.max()),
                            "note": "a tolerance must exceed about this quantile to pass 28 of 30 trials"}
g56 = np.array([gap(k) for k in keys])
res["F3_floor_q0933_56"] = {"value": q(g56), "ci95": boot(g56, q)}

# F7: rho on the same trial set (CPU's 56) for MPS and CPU
rk = lambda a: np.argsort(np.argsort(a))
for cond in ("confab", "honest_rare"):
    ks = [k for k in keys if M[k]["condition"] == cond]
    s = np.array([stored[(cond, M[k]["prompt_idx"])]["delta_features"]["stable_rank"] for k in ks])
    m = np.array([M[k]["forced"]["record"]["delta_features"]["stable_rank"] for k in ks])
    c = np.array([C[k]["forced"]["record"]["delta_features"]["stable_rank"] for k in ks])
    res[f"F7_{cond}_same_trials"] = {"n": len(ks), "rho_mps": float(np.corrcoef(rk(s), rk(m))[0, 1]),
                                     "rho_cpu": float(np.corrcoef(rk(s), rk(c))[0, 1])}

# F10: decomposition on all 56 from the arm (CPU fp32's tokens), MPS bf16 - CPU fp32, signed medians
enc = [A["mps_bfloat16"]["trials"][k]["encoding_sr"] - A["cpu_float32"]["trials"][k]["encoding_sr"] for k in keys]
gen = [A["mps_bfloat16"]["trials"][k]["sources"]["cpu_fp32_ids"]["generation_sr"]
       - A["cpu_float32"]["trials"][k]["sources"]["cpu_fp32_ids"]["generation_sr"] for k in keys]
dlt = [dsr("mps_bfloat16", k) - dsr("cpu_float32", k) for k in keys]
res["F10_decomposition_56"] = {"signed_median_encoding": float(np.median(enc)), "signed_median_generation": float(np.median(gen)),
                               "signed_median_delta": float(np.median(dlt)), "frac_delta_positive": float(np.mean(np.array(dlt) > 0)),
                               "frac_encoding_negative": float(np.mean(np.array(enc) < 0))}
# F10: paired test on the 48: identical-token gap vs own-token gap (Stage A forced records, each device's own tokens)
own = np.array([abs(M[k]["forced"]["record"]["delta_features"]["stable_rank"] - C[k]["forced"]["record"]["delta_features"]["stable_rank"]) for k in u48])
d = own - g48
obs = float(np.mean(d))
flips = np.array([np.mean(d * rng.choice([-1, 1], d.size)) for _ in range(200000)])
res["F10_paired_48"] = {"median_identical": float(np.median(g48)), "median_own_tokens": float(np.median(own)),
                        "mean_paired_difference": obs, "frac_own_larger": float(np.mean(d > 0)),
                        "signflip_p_two_sided": float((np.sum(np.abs(flips) >= abs(obs)) + 1) / (flips.size + 1))}

# F11: primary's bootstrap with other references
for ref in ("cpu_bfloat16", "mps_float32"):
    gr = np.array([gap(k, ref) for k in u48])
    res[f"F11_ref_{ref}_48"] = {"median": float(np.median(gr)), "ci95": boot(gr, np.median), "q0933": q(gr)}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agni_r3_numbers.json")
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items():
    print(k, json.dumps(v))
