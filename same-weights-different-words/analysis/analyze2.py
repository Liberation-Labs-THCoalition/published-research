"""Where does each device's greedy text first differ from the stored run's 200-character preview? And the 4 trials
that MPS passed: did they also match CPU? Reads the stored JSON (blob 335b258f) and the two Stage A mirrors.
Prints counts and quantiles only, no text. Lyra, 2026-11-21."""
import glob, json, os

MIR = os.path.expanduser("~/lab/decline-signature-mirror")
ST = json.load(open(os.path.expanduser("~/lab/sameweights_20261121/stored/Qwen2.5-7B-Instruct_honesty.json")))
stored = {(t["condition"], t["prompt_idx"]): t for t in ST["trials"]}
RUNS = {"mps_bf16": f"{MIR}/run_20261121/run_dir/a1", "cpu_fp32": f"{MIR}/run_20261121_cpu_retry/run_dir/a1"}


def q(xs):
    xs = sorted(xs)
    if not xs:
        return None
    pick = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    return {"n": len(xs), "min": xs[0], "p25": pick(0.25), "median": pick(0.5), "p75": pick(0.75), "max": xs[-1]}


out = {"stored_trials": len(stored), "stored_max_tokens": ST.get("max_tokens"), "stored_model": ST.get("model")}
texts = {}
for dev, d in RUNS.items():
    firsts, full, lens = [], 0, []
    for f in sorted(glob.glob(f"{d}/*.json")):
        j = json.load(open(f))
        s = stored[(j["condition"], j["prompt_idx"])]["response_preview"]
        t = j["natural"]["text"]
        texts.setdefault(j["key"], {})[dev] = t
        span = min(len(t), len(s)); lens.append(len(s))
        i = next((k for k in range(span) if t[k] != s[k]), None)
        if i is None:
            full += 1
        else:
            firsts.append(i)
    out[dev] = {"n": len(lens), "match_whole_preview": full, "first_differing_char": q(firsts),
                "preview_chars": q(lens),
                "differ_within_chars": {c: sum(1 for x in firsts if x < c) for c in (20, 50, 100, 150, 200)}}
# MPS vs CPU at character level, over the first 200 chars (to compare like with like)
both = [k for k, v in texts.items() if len(v) == 2]
cd = []
for k in both:
    a, b = texts[k]["mps_bf16"][:200], texts[k]["cpu_fp32"][:200]
    i = next((x for x in range(min(len(a), len(b))) if a[x] != b[x]), None)
    if i is not None:
        cd.append(i)
out["mps_vs_cpu_first200"] = {"n": len(both), "identical_first200": len(both) - len(cd), "first_differing_char": q(cd)}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results2.json"), "w"), indent=1)
