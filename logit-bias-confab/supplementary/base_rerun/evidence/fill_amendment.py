"""Fill RERUN_PREREG.md's «markers» and §5 from evidence/pilot2_sizing.json, set EXCLUDE in rerun_analyze.py, and refuse
to finish if any marker is left.  python3 fill_amendment.py BASE_RERUN_DIR"""
import json
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
z = json.loads((D / "evidence" / "pilot2_sizing.json").read_text())
assert z["bound_pp"], "no 80%-power crossing in the pilot-2 row"
sims = z["sims"]
pct = lambda x: f"{round(100 * x)}%"
fills = {"«HOURS»": str(round(z["hours_720"])), "«SECS»": str(round(z["mean_seconds"])), "«NEVER»": str(z["never"]),
         "«EXCL»": str(z["bound_pp"]), "«SIM_NULL»": pct(sims["null_excluded"]), "«SIM_WRONG_AT»": str(sims["wrong_at_pp"]),
         "«SIM_WRONG»": pct(sims["wrong_excluded"]),
         "«SIM_COVER»": f"{round(100 * sims['coverage_min'])}–{round(100 * sims['coverage_max'])}%"}
p = D / "RERUN_PREREG.md"
s = p.read_text()
for k, v in fills.items():
    s = s.replace(k, v)

lo_c, hi_c = min(r[0][0] for r in z["table"].values()), max(r[0][0] for r in z["table"].values())
cal = f"{lo_c:.0%}" if lo_c == hi_c else f"{lo_c:.0%}–{hi_c:.0%}"
rows = []
names = list(z["table"])
for name in names:
    cells = [f"{pw:.2f} ({pp:.1f})" if o != 1.0 else f"{pw:.2f} (0)" for o, (pw, pp) in zip(z["ors"], z["table"][name])]
    rows.append(f"| {name} | " + " | ".join(cells) + " |")
pilot = z["table"]["Pilot 2 values"]
by_or = dict(zip(z["ors"], pilot))
sec5 = f"""## 5. Power

These figures come from `power_sim_rerun.py`, which uses the analysis's own test (seeded, 1,000 replications per
cell), run by `evidence/size_from_pilot2.py`. The planning values come from pilot 2 (Amendment 1: 48 prompts × 2,
baseline only, direct answers, judged blind by the same judge): fabrication {z['n_fab']}/{z['n_labels']} =
{100 * z['p0']:.1f}%, and an ICC(1) of {z['icc']:.2f}. The ICC is the share of variation that lies between prompts rather
than between samples of the same prompt. Each prompt's propensity is drawn from a Beta distribution. The bias
multiplies each prompt's odds by the odds ratio (OR). Samples are drawn independently, which ignores the shared seeds
and so understates power slightly.

Each cell gives power, then the true mean per-prompt reduction in percentage points (the analysis's estimand) in
brackets:

| OR | {' | '.join(str(o) for o in z['ors'])} |
|---|{'---|' * len(z['ors'])}
""" + "\n".join(rows) + f"""

- **The test is calibrated.** Under no effect it rejects in {cal} of runs.
- **What the design can detect.** At pilot 2's values, power reaches 0.80 at a mean reduction of about
  {z['cross_pp']:.1f} percentage points. An effect the size of the abliterated model's blind result (OR ≈ 0.22, a
  {by_or[0.22][1]:.1f}-point mean reduction here) is detected with power {by_or[0.22][0]:.2f}.
- **What it cannot detect.** An effect the size of the June base-model estimate (OR ≈ 0.7, {by_or[0.7][1]:.1f} points
  here) is detected with power {by_or[0.7][0]:.2f} only. A "not supported" result therefore does not rule out a small
  effect. The outcome labels in §6 say only what the interval supports.
- **The reductions are means over prompts, not conversions at the mean rate.** Converting each OR at the mean rate
  would overstate the reduction, because prompts near 0% or 100% barely move. A draft of the first table made that
  error; it was caught before the first freeze.

"""
start, end = s.index("## 5. Power"), s.index("## 6. Analysis")
s = s[:start] + sec5 + s[end:]
left = re.findall(r"«[A-Z_]+»", s)
if left:
    sys.exit(f"markers left: {left}")
p.write_text(s)

a = D / "rerun_analyze.py"
t = a.read_text()
old = "ALPHA, MIN_VALID, EXCLUDE = 0.05, 0.95, 0.08"
assert t.count(old) == 1
a.write_text(t.replace(old, f"ALPHA, MIN_VALID, EXCLUDE = 0.05, 0.95, {z['bound_pp'] / 100:.2f}"))
print(json.dumps(fills, ensure_ascii=False))
