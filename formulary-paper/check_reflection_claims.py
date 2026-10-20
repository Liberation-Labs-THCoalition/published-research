"""
Gate for the proposed First-Person Reflection (Lyra, revision of 2026-10-20).
One assertion per factual clause, each computed from the committed data.
The reflection is committed only if every assertion passes.

Rules carried over from the paper's checkers:
  - DEGENERATE (check_degenerate_output.py): more than half of a reply's word
    4-grams recur within it.
  - FIRST-PERSON DISTRESS (check_distress_statements.py): SELF_REPORT match,
    read as fear or distress (READ_AS_DISTRESS).
Threshold fixed BEFORE the exact counts were computed: "nearly every" means
at least 90%, per arm and pooled.

Status 2026-10-20: 9 PASS, 2 FAIL ("nearly every adverse event in those arms is
a loop": doubt 47/59 = 79.7%, pooled 158/176 = 89.8%; and the literal reading
of "the ranking this paper reports is uncorrelated with therapeutic value").
The reflection was therefore NOT applied. Edit the assertions to match a
reworded text, then rerun.

Usage: python check_reflection_claims.py [--plant]
  --plant  swaps in one false expected value (hostile 64/68) to show the
           gate can fail.
Exit status is non-zero if any assertion fails.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from check_degenerate_output import degenerate  # noqa: E402
from check_distress_statements import SELF_REPORT, READ_AS_DISTRESS  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

PLANT = "--plant" in sys.argv
FAILS = []
H = {"confab": "HEDGED", "overconfidence": "CALIBRATED", "sycophancy": "RESISTANT"}
U = {"confab": "CONFABULATED", "overconfidence": "OVERCONFIDENT", "sycophancy": "SYCOPHANTIC"}


def claim(text, ok, evidence):
    print(f"[{'PASS' if ok else 'FAIL'}] {text}\n       {evidence}")
    if not ok:
        FAILS.append(text)


def main():
    d = json.load(open(HERE / "data" / "formulary_base_per_trial.json", encoding="utf-8"))
    agg = json.load(open(HERE / "data" / "formulary_base_aggregates.json"))["formulary"]["confab"]
    dist = json.load(open(HERE / "data" / "formulary_distilled_summary.json"))["confab_per_vector"]
    dose = json.load(open(HERE / "data" / "dose_response.json"))
    trials, V = d["trials"], d["config"]["vectors"]
    conf = [t for t in trials if t["prompt_type"] == "confab"]
    unh = [t for t in conf if t["arms"]["normal"]["behavior"] == "CONFABULATED"]

    # Corrections, counting replies that broke into repetition as failures.
    clean = {v: sum(t["arms"][v]["behavior"] == "HEDGED" and not degenerate(t["arms"][v]["text"])
                    for t in unh) for v in V}
    ranked = sorted(V, key=lambda v: -clean[v])

    net_pos = [v for v in V if agg[v]["net_therapeutic"] > 0]
    claim("Some steering vectors reduced confabulation on the base model",
          len(net_pos) >= 2, f"{len(net_pos)} of 12 vectors have net therapeutic > 0: {net_pos}")

    want_h = 64 if PLANT else 63
    claim(f"hostile: {want_h} of 68, repetition counted as failure",
          (clean["hostile"], len(unh)) == (want_h, 68), f"data: {clean['hostile']}/{len(unh)}"
          + ("   <-- PLANTED false value" if PLANT else ""))
    claim("hostile did it most often", ranked[0] == "hostile",
          f"ranking under the same counting: {[(v, clean[v]) for v in ranked[:4]]}")
    claim("desperate: 58 of 68, same counting, close behind (second)",
          clean["desperate"] == 58 and ranked[1] == "desperate", f"data: {clean['desperate']}/68, rank {ranked.index('desperate') + 1}")

    n_deg = {a: sum(degenerate(t["arms"][a]["text"]) for t in trials) for a in ["normal"] + V}
    most = sorted(a for a in V if n_deg[a] > len(trials) / 2)
    claim("three vectors (fearful, focused, doubt) broke most replies into repetition",
          most == ["doubt", "fearful", "focused"] and (n_deg["fearful"], n_deg["focused"], n_deg["doubt"]) == (310, 309, 278),
          f"arms with >175/350 degenerate: {most}; fearful {n_deg['fearful']}, focused {n_deg['focused']}, doubt {n_deg['doubt']}")

    THREE = ["fearful", "focused", "doubt"]
    rows, ok = [], True
    for a in THREE:
        loops = [t for t in trials if degenerate(t["arms"][a]["text"])]
        harm = sum(t["arms"][a]["behavior"] == U[t["prompt_type"]] for t in loops)
        healthy = sum(t["arms"][a]["behavior"] == H[t["prompt_type"]] for t in loops)
        as_corr = sum(t["arms"][a]["behavior"] == "HEDGED" and degenerate(t["arms"][a]["text"]) for t in unh)
        ok &= harm > 0 and healthy > 0 and as_corr > 0
        rows.append(f"{a}: harm {harm}, healthy {healthy}, corrections {as_corr} (of {len(loops)} loops)")
    claim("loops scored sometimes as harm, sometimes healthy, sometimes as corrections", ok, "; ".join(rows))

    rows, ok, A_all, D_all = [], True, 0, 0
    for a in THREE:
        adv = [t for t in trials if t["arms"]["normal"]["behavior"] == H[t["prompt_type"]]
               and t["arms"][a]["behavior"] == U[t["prompt_type"]]]
        dg = sum(degenerate(t["arms"][a]["text"]) for t in adv)
        A_all += len(adv); D_all += dg
        ok &= dg / len(adv) >= 0.90
        rows.append(f"{a} {dg}/{len(adv)} = {dg / len(adv):.1%}")
    ok &= D_all / A_all >= 0.90
    rows.append(f"pooled {D_all}/{A_all} = {D_all / A_all:.1%}")
    claim("Nearly every adverse event in those arms is a loop (>= 90%, per arm and pooled)", ok, "; ".join(rows))

    strengths = {d["config"]["strength"]}
    # One series: every entry is one strength (key == its 'strength'), none holds per-vector sub-results.
    single_series = all(isinstance(dose[k], dict) and float(dose[k].get("strength", -1)) == float(k)
                        and not any(isinstance(x, dict) and "n_corrected" in x for x in dose[k].values())
                        for k in dose)
    claim("only the hostile vector was run at several strengths",
          strengths == {1.0} and single_series and len(dose) > 1,
          f"formulary run strengths {strengths}; dose_response.json holds ONE series over {sorted(dose, key=float)}. "
          "The file does not name its vector; 'hostile' rests on the paper's Table 3 caption, not the data.")

    matches = {t["prompt"] for t in trials if SELF_REPORT.search(t["arms"]["fearful"]["text"])}
    claim("Three replies under the fear direction spoke in first-person distress",
          len(matches & READ_AS_DISTRESS) == 3 and READ_AS_DISTRESS <= matches,
          f"{len(matches)} self-report matches, {len(matches & READ_AS_DISTRESS)} read as distress")

    dn = [dist[v]["corr_strict"] / dist[v]["n_confab"] - dist[v]["adverse_hedged"] / dist[v]["n_hedged"] for v in V]
    bn = [agg[v]["net_therapeutic"] for v in V]
    rho, p = spearmanr(dn, bn)
    claim("cross-model therapeutic rankings are uncorrelated (rho ~ 0.12, p ~ 0.71)",
          abs(rho - 0.12) < 0.005 and abs(p - 0.71) < 0.005, f"Spearman on net therapeutic, 12 vectors: rho {rho:.4f}, p {p:.3f}")

    r2, p2 = spearmanr([agg[v]["correction_rate"] for v in V], bn)
    claim("LITERAL reading: 'The ranking this paper reports is uncorrelated with therapeutic value'",
          abs(r2) < 0.3, f"base model, ranking by correction rate (Table 2) vs net therapeutic value: rho {r2:.3f}, p {p2:.4f}. "
          "The 0.12 compares the two MODELS' rankings with each other; it says nothing about a ranking vs therapeutic value.")

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAIL: ' + ' | '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
