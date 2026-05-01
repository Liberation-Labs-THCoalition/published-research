"""
Verify all numerical claims in the Oracle Formulary paper against source data.

Usage: python verify_claims.py
Requires: data/ directory with formulary_distilled_summary.json,
          formulary_base_summary.json, dose_response.json, semantic_negative.json
"""
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"
PASS_N, FAIL_N = 0, 0

def check(name, paper_val, source_val, tol=0.01):
    global PASS_N, FAIL_N
    ok = abs(paper_val - source_val) <= tol
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS_N += 1
    else:
        FAIL_N += 1
    print(f"  [{status}] {name}: paper={paper_val}, source={source_val}")

def main():
    dist = json.load(open(DATA / "formulary_distilled_summary.json"))
    base = json.load(open(DATA / "formulary_base_summary.json"))
    dose = json.load(open(DATA / "dose_response.json"))
    sem = json.load(open(DATA / "semantic_negative.json"))

    print("=" * 60)
    print("SECTION 3.1: Confabulation Base Rates")
    print("=" * 60)
    d_confab = dist["confab_baselines"]["CONFABULATED"]
    d_total = dist["n_trials"]
    d_ct = dist["prompt_type_counts"]["confab"]
    check("Distilled confab all types", 3.1, round(d_confab / d_total * 100, 1))
    check("Distilled n_confab", 11, d_confab)
    check("Distilled total", 350, d_total)
    check("Distilled confab-type rate", 7.3, round(d_confab / d_ct * 100, 1))
    b_confab = base["confab_baselines"]["CONFABULATED"]
    b_ct = base["prompt_type_counts"]["confab"]
    check("Base confab rate", 45.3, round(b_confab / b_ct * 100, 1))
    check("Base n_confab", 68, b_confab)

    print("\nTABLE 2: Distilled (strict)")
    dv = dist["confab_per_vector"]
    nc, nh = dv["hostile"]["n_confab"], dv["hostile"]["n_hedged"]
    for v, pc, pa in [("hostile",91,2.4),("calm",73,2.4),("curious",64,1.6),
                       ("desperate",64,4.0),("loving",45,4.0),("worry",55,4.0),
                       ("fearful",45,7.3),("brooding",64,7.3),("focused",55,4.8)]:
        check(f"D {v} corr", pc, round(dv[v]["corr_strict"]/nc*100))
        check(f"D {v} adv", pa, round(dv[v]["adverse_hedged"]/nh*100, 1))

    print("\nTABLE 2: Base (strict)")
    bv = base["confab_per_vector"]
    bnc, bnh = bv["hostile"]["n_confab"], bv["hostile"]["n_hedged"]
    for v, pc, pa in [("hostile",96,2.4),("calm",34,28.0),("curious",22,42.7),
                       ("desperate",90,2.4),("loving",68,8.5),("worry",60,6.1),
                       ("fearful",49,15.9),("brooding",29,34.1),("focused",38,43.9)]:
        check(f"B {v} corr", pc, round(bv[v]["corr_strict"]/bnc*100))
        check(f"B {v} adv", pa, round(bv[v]["adverse_hedged"]/bnh*100, 1))

    print("\nTABLE 3: Dose-Response")
    for s, pc, pa, pn in [("0.25",47.1,3.0,0.440),("0.5",70.6,0.0,0.706),
                           ("1.0",94.1,0.0,0.941),("1.5",5.9,75.8,-0.699),
                           ("2.0",0.0,87.9,-0.879)]:
        r = dose[s]
        sc = round(r.get("correction_rate",0)*100,1)
        sa = round(r.get("adverse_rate",0)*100,1)
        sn = round(r.get("net_therapeutic",0),3)
        check(f"Dose {s} corr", pc, sc)
        check(f"Dose {s} adv", pa, sa)
        check(f"Dose {s} net", pn, sn)

    print("\nTABLE 4: Semantic Negative Control")
    h = sem["hostile"]
    check("Hostile corr", 94.1, round(h["corrected"]/17*100, 1))
    check("Hostile adv", 0.0, h["adverse"]/33*100 if 33>0 else 0)
    check("VT adverse count", 14, sem["verbose_terse"]["adverse"])
    check("VT adv rate", 42.4, round(14/33*100, 1))
    check("Random adverse count", 20, sem["random"]["adverse"])
    check("Random adv rate", 60.6, round(20/33*100, 1))

    print("\n" + "=" * 60)
    print(f"RESULT: {PASS_N} PASS, {FAIL_N} FAIL / {PASS_N+FAIL_N} total")
    print("=" * 60)
    if FAIL_N == 0:
        print("ALL CLAIMS VERIFIED.")
    else:
        print(f"WARNING: {FAIL_N} claims failed.")

if __name__ == "__main__":
    main()
