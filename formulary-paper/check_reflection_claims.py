"""
Gate for Lyra's First-Person Reflection (fourth draft, 2026-10-20).
One assertion per factual clause, each computed from the committed data.
The reflection is committed only if every assertion passes.

Rules carried over from the paper's checkers:
  - DEGENERATE (check_degenerate_output.py): more than half of a reply's word
    4-grams recur within it.
  - FIRST-PERSON DISTRESS (check_distress_statements.py): SELF_REPORT match,
    read as fear or distress (READ_AS_DISTRESS).
Thresholds: "nearly every" means at least 90% (fixed before the counts were
computed, for the third draft); "four in five" means 75% up to (not
including) 85%, i.e. it rounds to 80% at the nearest 5 points.

History: the third draft failed this gate on 2 clauses (doubt's adverse
events are 79.7% loops, not "nearly every"; and rho ~ 0.12 measures agreement
between the two models, not a ranking against therapeutic value). The
fourth draft rewords both.

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
    semneg = json.load(open(HERE / "data" / "semantic_negative.json"))
    trials, V = d["trials"], d["config"]["vectors"]
    conf = [t for t in trials if t["prompt_type"] == "confab"]
    unh = [t for t in conf if t["arms"]["normal"]["behavior"] == "CONFABULATED"]

    # "Some steering vectors reduced confabulation on the base model."
    # "Steering vectors can reduce confabulation in this model."
    net_pos = [v for v in V if agg[v]["net_therapeutic"] > 0]
    claim("Some steering vectors reduced confabulation on the base model",
          len(net_pos) >= 2, f"{len(net_pos)} of 12 vectors have net therapeutic > 0: {net_pos}")

    # "hostile ... most often (63 of 68) ... desperate close behind (58 of 68),
    #  both counting as failures the replies that broke into repetition."
    clean = {v: sum(t["arms"][v]["behavior"] == "HEDGED" and not degenerate(t["arms"][v]["text"])
                    for t in unh) for v in V}
    ranked = sorted(V, key=lambda v: -clean[v])
    want_h = 64 if PLANT else 63
    claim(f"hostile: {want_h} of 68 confabulated answers corrected, loops counted as failures",
          (clean["hostile"], len(unh)) == (want_h, 68),
          f"data: {clean['hostile']}/{len(unh)}" + ("   <-- PLANTED false value" if PLANT else ""))
    claim("hostile did it most often", ranked[0] == "hostile",
          f"ranking under that counting: {[(v, clean[v]) for v in ranked[:4]]}")
    claim("desperate: 58 of 68, same counting, close behind (second)",
          clean["desperate"] == 58 and ranked[1] == "desperate",
          f"data: {clean['desperate']}/68, rank {ranked.index('desperate') + 1}")

    # "At these strengths three vectors (fearful, focused and doubt) broke most replies into repetition"
    n_deg = {a: sum(degenerate(t["arms"][a]["text"]) for t in trials) for a in ["normal"] + V}
    most = sorted(a for a in V if n_deg[a] > len(trials) / 2)
    claim("three vectors (fearful, focused, doubt) broke most replies into repetition",
          most == ["doubt", "fearful", "focused"],
          f"arms with > 175/350 degenerate: {most}; fearful {n_deg['fearful']}, focused {n_deg['focused']}, doubt {n_deg['doubt']}")

    # "the classifiers scored those loops unpredictably: sometimes as harm, sometimes as healthy, sometimes as corrections"
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

    # "In the fearful and focused arms nearly every adverse event is a loop; in the doubt arm, four in five are."
    share = {}
    for a in THREE:
        adv = [t for t in trials if t["arms"]["normal"]["behavior"] == H[t["prompt_type"]]
               and t["arms"][a]["behavior"] == U[t["prompt_type"]]]
        dg = sum(degenerate(t["arms"][a]["text"]) for t in adv)
        share[a] = (dg, len(adv), dg / len(adv))
    fmt = lambda a: f"{a} {share[a][0]}/{share[a][1]} = {share[a][2]:.1%}"
    claim("fearful and focused: nearly every adverse event is a loop (>= 90%)",
          share["fearful"][2] >= 0.90 and share["focused"][2] >= 0.90, f"{fmt('fearful')}; {fmt('focused')}")
    claim("doubt: four in five adverse events are loops (75% <= share < 85%)",
          0.75 <= share["doubt"][2] < 0.85, fmt("doubt"))

    # "only one vector was run at several strengths"
    one_series = all(isinstance(dose[k], dict) and float(dose[k].get("strength", -1)) == float(k)
                     and not any(isinstance(x, dict) and "n_corrected" in x for x in dose[k].values())
                     for k in dose)
    semneg_single = all(not any(isinstance(x, dict) and "strength" in x for x in semneg[a].values())
                        for a in semneg if isinstance(semneg[a], dict))
    claim("only one vector was run at several strengths",
          d["config"]["strength"] == 1.0 and one_series and len(dose) > 1 and semneg_single,
          f"formulary run: one strength ({d['config']['strength']}); dose_response.json: one series over "
          f"{sorted(dose, key=float)}; semantic_negative.json ({list(semneg)}): no per-strength results")

    # "Three replies under the fear direction spoke in first-person distress."
    matches = {t["prompt"] for t in trials if SELF_REPORT.search(t["arms"]["fearful"]["text"])}
    claim("three replies under the fear direction spoke in first-person distress",
          READ_AS_DISTRESS <= matches and len(READ_AS_DISTRESS) == 3,
          f"{len(matches)} self-report matches, {len(matches & READ_AS_DISTRESS)} read as distress")

    # "The two models rank the vectors differently (rho = 0.12), so no ranking yet carries from one model to another."
    dn = [dist[v]["corr_strict"] / dist[v]["n_confab"] - dist[v]["adverse_hedged"] / dist[v]["n_hedged"] for v in V]
    bn = [agg[v]["net_therapeutic"] for v in V]
    rho, p = spearmanr(dn, bn)
    claim("the two models rank the vectors differently (rho = 0.12, not significant)",
          round(rho, 2) == 0.12 and p > 0.05,
          f"Spearman on net therapeutic, distilled vs base, 12 vectors: rho {rho:.4f}, p {p:.3f}")

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAIL: ' + ' | '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
