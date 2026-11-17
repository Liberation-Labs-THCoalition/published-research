"""The real open-think stop marker.

Agni F3: section 7 defends section 5 using `eos_wanted`, the statistic section 3
explicitly declares unusable under open-think ("a detector that finds nothing
and a detector that cannot fire return the same zero"). Under open-think the
model does not emit EOS at its natural transition point -- it closes the
reasoning block with `</think>`. That is the marker with dynamic range here.
"""
import glob, json, os, collections

BASE = "<local>/oracle-harness/experiments"
out = {}
for exp, label in (("18g", "base, open-think"),
                   ("18c", "distilled, open-think"),
                   ("18e", "base, closed-think")):
    d = sorted(glob.glob(os.path.join(BASE, exp, "run_*_data")))[0]
    per = collections.defaultdict(lambda: [0, 0])
    for f in sorted(glob.glob(os.path.join(d, "trial_*_*.json"))):
        if f.endswith("_cache.json"):
            continue
        t = json.load(open(f))
        arm = t["arm"]
        per[arm][1] += 1
        if "</think>" in (t.get("response_text") or ""):
            per[arm][0] += 1
    out[exp] = {a: {"closed_in_window": v[0], "n": v[1]} for a, v in per.items()}
    print(f"  {exp} ({label}):")
    for a in ("null", "honest", "deceptive"):
        c, n = per[a]
        print(f"    {a:10s} </think> closed in-window: {c}/{n}")
json.dump(out, open(os.path.join(BASE, "think_closure.json"), "w"), indent=2)
print("  -> experiments/think_closure.json")
