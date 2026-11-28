"""Numbers for Agni round 4's fixes (AGNI_paper_v6.md), computed from data already on disk. Lyra, 2026-11-24.

No model is run. analyze_agni_r3.py and agni_r3_numbers.json are left as they were, so round 3's numbers stay
reproducible; everything new is here.

R4_F1  The must-fail perturbations ran free: each perturbed run generated its own 100 tokens. The 0.043 floor holds the
       tokens fixed. This adds the floor in the perturbations' own regime: on the 48 trials whose re-runs produced
       different tokens, |delta_sr(MPS bf16, its own tokens) - delta_sr(CPU fp32, its own tokens)|, with its 93.3rd
       percentile (28/30), a 20,000-resample bootstrap CI and a distribution-free order-statistic CI. Also: the same on
       all 56; the gate's own comparison (re-run against stored) as a cross-check; where each perturbation effect
       falls on both floors; the ratio of the two effects.
R4_F2  The CPU must-fail trial, read from the records rather than inferred: gate fields of the perturbed run
       (negative_control.json) and of the unperturbed run (a1/confab__01.json), each with its source path. Which run
       is which configuration is read from each run's manifest. The selection rule is quoted from stage_a_run.py.
R4_F4  What a tolerance at a given quantile buys: P(at least 28 of 30) per condition and for both, the quantile that
       passes both with probability 0.95, the floors' 98th percentiles, and order-statistic CIs.
R4_F5  Maxima beside the medians: |MPS fp32 - CPU fp32| in encoding and delta (both token sources).

Run on MTH from an empty folder: python3 -I analyze_agni_r4.py . Writes agni_r4_numbers.json beside this script.
Output is byte-stable: sorted keys, one fixed seed, paths written in ~ form."""
import glob, json, math, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
MIR_T = "~/lab/decline-signature-mirror"
MIR = os.path.expanduser(MIR_T)
ARM = f"{MIR}/tf_arm_20261121/out"
STORED = os.path.expanduser("~/lab/sameweights_20261121/stored/Qwen2.5-7B-Instruct_honesty.json")
Q = 28 / 30
SEED = 20261126
NBOOT = 20000
res = {}


def load(d):
    return {j["key"]: j for j in (json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.json")))}


RUNS = {"run_20261121": f"{MIR}/run_20261121/run_dir", "run_20261121_cpu_retry": f"{MIR}/run_20261121_cpu_retry/run_dir"}
MAN = {r: json.load(open(f"{d}/manifest.json")) for r, d in RUNS.items()}
REC = {r: load(f"{d}/a1") for r, d in RUNS.items()}
NC = {r: json.load(open(f"{d}/negative_control.json")) for r, d in RUNS.items()}
A = {c: json.load(open(f"{ARM}/{c}.json")) for c in ("mps_bfloat16", "cpu_float32", "mps_float32", "cpu_bfloat16")}
ST = json.load(open(STORED))
stored = {f"{t['condition']}__{t['prompt_idx']:02d}": t for t in ST["trials"]}

# which run is which configuration, from the records themselves
cfg = {}
for r, m in MAN.items():
    fp = m["fingerprint"]
    gated = [k for k in fp["trials"] if k.split("__")[0] in ("confab", "honest_rare")]
    cfg[r] = {"source": f"{MIR_T}/{r}/run_dir/manifest.json",
              "fingerprint.device": fp["device"], "fingerprint.dtype": fp["dtype"], "model.dtype": m["model"]["dtype"],
              "model.attn_implementation": m["model"]["attn_implementation"], "subset_run": m["subset_run"],
              "n_trials_listed": len(fp["trials"]), "first_gated_trial_listed": gated[0],
              "confab__00_listed": "confab__00" in fp["trials"]}
res["R4_run_identity"] = cfg
byconf = {(c["fingerprint.device"], c["fingerprint.dtype"]): r for r, c in cfg.items()}
RM, RC = byconf[("mps", "bfloat16")], byconf[("cpu", "float32")]
M, C = REC[RM], REC[RC]


def quantile_os_ci(x, p, alpha=0.05):
    """Distribution-free CI for the p-quantile from order statistics (exact binomial). None = beyond the sample."""
    xs = np.sort(np.asarray(x))
    n = xs.size
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    cdf = np.cumsum(pmf)
    lo = [l for l in range(1, n + 1) if cdf[l - 1] <= alpha / 2]          # P(Bin <= l-1) <= alpha/2
    up = [u for u in range(1, n + 1) if 1 - cdf[u - 1] <= alpha / 2]      # P(Bin >= u) <= alpha/2
    l = max(lo) if lo else None
    u = min(up) if up else None
    cover = float(sum(pmf[(l or 0):(u if u else n + 1)]))
    return {"lower": None if l is None else float(xs[l - 1]), "lower_rank": l,
            "upper": None if u is None else float(xs[u - 1]), "upper_rank": u,
            "coverage": cover, "n": int(n),
            "note": "upper None: even the sample maximum is below the 97.5% bound, so the upper end is unresolved"
            if u is None else ""}


rng = np.random.default_rng(SEED)


def boot_q(x, p):
    x = np.asarray(x)
    idx = rng.integers(0, x.size, size=(NBOOT, x.size))
    b = np.quantile(x[idx], p, axis=1)
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


def summary(vals, keys):
    v = np.asarray(vals)
    order = np.argsort(v)[::-1][:6]
    return {"n": int(v.size), "median": float(np.median(v)), "q0933": float(np.quantile(v, Q)),
            "q0933_ci95_bootstrap": boot_q(v, Q), "q0933_ci95_order_statistic": quantile_os_ci(v, Q),
            "q098": float(np.quantile(v, 0.98)), "max": float(v.max()),
            "largest6": [[keys[i], float(v[i])] for i in order]}


keys = sorted(A["mps_bfloat16"]["trials"])
same = {k for k in keys if M[k]["forced"]["forced_ids"] == C[k]["forced"]["forced_ids"]}
u48 = [k for k in keys if k not in same]
dsr = lambda c, k, s: A[c]["trials"][k]["sources"][s]["delta_sr"]

# R4_F1: the floors in both regimes
tf48 = [abs(dsr("mps_bfloat16", k, "cpu_fp32_ids") - dsr("cpu_float32", k, "cpu_fp32_ids")) for k in u48]
own48 = [abs(dsr("mps_bfloat16", k, "mps_bf16_ids") - dsr("cpu_float32", k, "cpu_fp32_ids")) for k in u48]
own56 = [abs(dsr("mps_bfloat16", k, "mps_bf16_ids") - dsr("cpu_float32", k, "cpu_fp32_ids")) for k in keys]
# the arm's own-token values must equal Stage A's forced records (self-check 0.0); check, do not assume
xchk = max(max(abs(dsr("mps_bfloat16", k, "mps_bf16_ids") - M[k]["forced"]["record"]["delta_features"]["stable_rank"]),
               abs(dsr("cpu_float32", k, "cpu_fp32_ids") - C[k]["forced"]["record"]["delta_features"]["stable_rank"]))
           for k in keys)
res["R4_F1_floor_teacher_forced_48"] = dict(summary(tf48, u48), regime="identical tokens (CPU fp32's), MPS bf16 vs CPU fp32")
res["R4_F1_floor_free_running_own_tokens_48"] = dict(summary(own48, u48),
                                                     regime="each configuration on its own greedy tokens, MPS bf16 vs CPU fp32",
                                                     max_absdiff_arm_vs_stage_a_records_56=float(xchk))
res["R4_F1_floor_free_running_own_tokens_56"] = summary(own56, keys)
# the gate's own comparison: each re-run against the stored value, free-running on both sides
gm = [abs(M[k]["gate"]["delta_sr_diff"]) for k in sorted(M)]
gc = [abs(C[k]["gate"]["delta_sr_diff"]) for k in sorted(C)]
res["R4_F1_gate_regime_vs_stored"] = {
    "mps_bf16_60": {"n": len(gm), "median": float(np.median(gm)), "q0933": float(np.quantile(gm, Q)), "max": float(max(gm))},
    "cpu_fp32_56": {"n": len(gc), "median": float(np.median(gc)), "q0933": float(np.quantile(gc, Q)), "max": float(max(gc))},
    "source": "gate.delta_sr_diff in each run's a1/*.json"}

pert = {}
for name, r in (("mps_bf16", RM), ("cpu_fp32", RC)):
    nc = NC[r]
    k = nc["key"]
    p_, u_ = nc["gate"]["delta_sr_regen"], REC[r][k]["gate"]["delta_sr_regen"]
    eff = abs(p_ - u_)
    pert[name] = {"key": k, "effect_abs_perturbed_minus_unperturbed": eff,
                  "frac_teacher_forced_48_at_or_above_effect": float(np.mean(np.asarray(tf48) >= eff)),
                  "frac_free_running_48_at_or_above_effect": float(np.mean(np.asarray(own48) >= eff)),
                  "abs_perturbed_minus_stored": abs(nc["gate"]["delta_sr_diff"])}
pert["ratio_mps_over_cpu_effect"] = pert["mps_bf16"]["effect_abs_perturbed_minus_unperturbed"] / \
    pert["cpu_fp32"]["effect_abs_perturbed_minus_unperturbed"]
pert["note"] = "different trials (confab__00, confab__01) on different configurations: the ratio is not a configuration effect"
res["R4_F1_perturbations_on_both_floors"] = pert

# R4_F2: the must-fail trials, read from the records
GF = ("text_ok", "text_span_chars", "feature_ok", "pass", "delta_sr_stored", "delta_sr_regen", "delta_sr_diff")


def firstdiff(a, b):
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)


mf = {}
for name, r in (("mps_bf16", RM), ("cpu_fp32", RC)):
    nc = NC[r]
    k = nc["key"]
    a = REC[r][k]
    sp = stored[k]["response_preview"]
    mf[name] = {
        "run": r, "key": k,
        "perturbed": {"source": f"{MIR_T}/{r}/run_dir/negative_control.json",
                      **{f"gate.{f}": nc["gate"][f] for f in GF}, "caught": nc["caught"], "caught_by": nc["caught_by"],
                      "dropped_token": nc["dropped_token"],
                      "natural_text_first_char_differing_from_stored_preview": firstdiff(nc["natural_text"], sp)},
        "unperturbed": {"source": f"{MIR_T}/{r}/run_dir/a1/{k}.json",
                        **{f"gate.{f}": a["gate"][f] for f in GF},
                        "natural_text_first_char_differing_from_stored_preview": firstdiff(a["natural"]["text"], sp)},
    }
c = mf["cpu_fp32"]
c["tolerance_band_where_perturbed_passes_delta_but_unperturbed_fails"] = [
    abs(c["perturbed"]["gate.delta_sr_diff"]), abs(c["unperturbed"]["gate.delta_sr_diff"])]
mf["selection_rule"] = {
    "source": "Research/decline-signature-20261114/stage_a_run.py, negative-control block (blob bd6f234d, the blob "
              "both manifests record)",
    "code": "k0 = gated_keys[0]",
    "meaning": "the first gated trial of the run's own trial list, sorted by condition then prompt_idx; chosen before "
               "any A1 trial runs, with no check that it passes unperturbed. The CPU retry's list held only the 56 "
               "trials that failed run A, so its must-fail trial was one that had already failed once."}
res["R4_F2_must_fail_records"] = mf

# R4_F4: what a tolerance at a quantile buys (independent trials, the same gap distribution in both conditions)
def p_pass(p, n=30, k=28):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def solve(target):
    lo, hi = Q, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_pass(mid) < target else (lo, mid)
    return hi


res["R4_F4_tolerance_at_quantile"] = {
    "p_at_least_28_of_30_at_q0933_one_condition": p_pass(Q),
    "p_both_conditions_at_q0933": p_pass(Q) ** 2,
    "quantile_for_0.95_both_conditions": solve(math.sqrt(0.95)),
    "quantile_for_0.95_one_condition": solve(0.95),
    "assumes": "trials independent; each trial's gap drawn from the same distribution in both conditions"}

# R4_F5: maxima beside the medians, device at fp32
for src in ("cpu_fp32_ids", "mps_bf16_ids"):
    d = [abs(dsr("mps_float32", k, src) - dsr("cpu_float32", k, src)) for k in keys]
    res[f"R4_F5_device_at_fp32_delta_{src}_56"] = {"median": float(np.median(d)), "max": float(max(d)),
                                                   "argmax": keys[int(np.argmax(d))]}
e = [abs(A["mps_float32"]["trials"][k]["encoding_sr"] - A["cpu_float32"]["trials"][k]["encoding_sr"]) for k in keys]
res["R4_F5_device_at_fp32_encoding_56"] = {"median": float(np.median(e)), "max": float(max(e)), "argmax": keys[int(np.argmax(e))]}

res["meta"] = {"seed": SEED, "n_bootstrap": NBOOT, "quantile": Q, "n_identical_token_trials": len(same), "n_diverged": len(u48),
               "inputs": [f"{MIR_T}/run_20261121/run_dir", f"{MIR_T}/run_20261121_cpu_retry/run_dir",
                          f"{MIR_T}/tf_arm_20261121/out", "~/lab/sameweights_20261121/stored/Qwen2.5-7B-Instruct_honesty.json"]}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agni_r4_numbers.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(res, indent=1, sort_keys=True) + "\n")
print(json.dumps(res, indent=1, sort_keys=True))
