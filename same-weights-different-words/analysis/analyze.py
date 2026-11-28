"""Same weights, different words: how much of Stage A's failure to reproduce is arithmetic, and how much is the text.

Inputs (read only): the MTH mirrors of the two Stage A runs (2026-11-21), MPS bf16 (60 gated trials) and CPU fp32 (the 56
that failed). Each a1/<key>.json holds the natural greedy reply's token ids (up to 100; 3 replies stopped at
EOS) and, in its gate record, stored vs regenerated features for three phases: encoding (the prompt only: identical tokens on every chip),
generation and delta.

Questions:
 Q1 text: how often does each device's greedy reply match the stored preview (where it first differs: analyze2.py)?
 Q2 device vs device: for the 56 trials both devices ran, at which token do MPS bf16 and CPU fp32 first part ways?
 Q3 the floor: with identical input tokens (the encoding phase), how far do features move between hardware?
 Q4 the cost of divergence: how far do generation/delta features move when the text matched vs when it didn't?
 Q5 layers: which layers reproduce, by text match.
Prints counts and quantiles; writes results.json. Lyra, 2026-11-21.
v1 script, kept for the record: its q() takes nearest-rank medians; analyze3.py (numpy) supersedes them."""
import glob, json, os

MIR = os.path.expanduser("~/lab/decline-signature-mirror")
RUNS = {"mps_bf16": f"{MIR}/run_20261121/run_dir/a1", "cpu_fp32": f"{MIR}/run_20261121_cpu_retry/run_dir/a1"}
TOL = 0.01


def load(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        j = json.load(open(f))
        out[j["key"]] = j
    return out


def q(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    pick = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    return {"n": len(xs), "min": xs[0], "p25": pick(0.25), "median": pick(0.5), "p75": pick(0.75), "max": xs[-1]}


R = {k: load(v) for k, v in RUNS.items()}
res = {"inputs": {k: len(v) for k, v in R.items()}}

# Q1: counts of the gate's own fields (text_ok compares the first min(len) characters of the natural reply and the
# stored preview; every natural reply here is at least 200 characters, analyze3.py section A).
for dev, rows in R.items():
    ok = [r["gate"]["text_ok"] for r in rows.values()]
    feat = [r["gate"]["feature_ok"] for r in rows.values()]
    res[f"Q1_{dev}"] = {"text_ok": sum(ok), "n": len(ok), "feature_ok": sum(feat),
                        "pass": sum(r["gate"]["pass"] for r in rows.values())}

# Q2: MPS vs CPU, token by token, on the shared keys.
shared = sorted(set(R["mps_bf16"]) & set(R["cpu_fp32"]))
divs, same100 = [], 0
for k in shared:
    a = R["mps_bf16"][k]["natural"]["reply_ids"]; b = R["cpu_fp32"][k]["natural"]["reply_ids"]
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if d is None and len(a) == len(b):
        same100 += 1
    divs.append(d if d is not None else (min(len(a), len(b)) if len(a) != len(b) else None))
res["Q2_mps_vs_cpu"] = {"n_shared": len(shared), "identical_replies": same100,
                        "first_divergent_token": q([d for d in divs if d is not None]),
                        "diverged_by_token": {t: sum(1 for d in divs if d is not None and d < t) for t in (10, 25, 50, 75, 100)}}

# Q3 and Q4: |regen - stored| per phase, for stable_rank (the gated feature).
for dev, rows in R.items():
    for ph in ("encoding_features", "generation_features", "delta_features"):
        for grp, keep in (("text_ok", True), ("text_diverged", False)):
            sr = [r["gate"]["descriptive"]["absdiff"][ph]["stable_rank"] for r in rows.values() if r["gate"]["text_ok"] is keep]
            res[f"Q34_{dev}_{ph}_{grp}_absdiff_stable_rank"] = q(sr)
            res[f"Q34_{dev}_{ph}_{grp}_within_tol"] = f"{sum(1 for x in sr if x <= TOL)}/{len(sr)}"

# Q5: per-layer generation stable rank: how many trials reproduce each layer, by text match.
for dev, rows in R.items():
    lay = {}
    for grp, keep in (("text_ok", True), ("text_diverged", False)):
        sub = [r for r in rows.values() if r["gate"]["text_ok"] is keep]
        lay[grp] = {L: sum(1 for r in sub if r["gate"]["per_layer_generation_stable_rank"][L]["ok"]) for L in range(28)}
        lay[grp]["n"] = len(sub)
    res[f"Q5_{dev}_layers_within_tol"] = lay

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items():
    print(k, json.dumps(v) if not isinstance(v, dict) or len(json.dumps(v)) < 400 else json.dumps(v)[:400] + " ...")
