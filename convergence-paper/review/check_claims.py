#!/usr/bin/env python3
"""check_claims.py -- gate for the convergence paper's v1.1 corrections.

Every sentence changed in v1.1 that carries a number, a count or a status is
checked here, in two halves:

  1. PRIMARY.  The value is read from the artifact that produced it (a result
     JSON, a pre-registration, the paper that reports the experiment in full).
     Where that artifact is not on this machine, or is not machine-readable,
     the value is HELD below with a comment naming file, line and commit, and
     the check says HELD rather than OK.
  2. TEXT.  The value must appear in the paper's LaTeX exactly as written, and
     the v1.0 wording it replaced must be gone.

The correction note quotes what it corrects (AGENTS.md section 4, cause 4), so
it is cut out of the text before any present/absent check on the body, and
checked on its own.

A PLANTED false value runs on every invocation: the Project-Oracle findings
registry heading's "exploratory 80%->0%" (i.e. a corrected arm of 0/20). The
gate must catch it against both the primary and the text. If it does not, the
gate is broken and the script exits 2, whatever else passed.

Usage (from the repository root or anywhere):
    python convergence-paper/review/check_claims.py
Environment:
    RESEARCH_ROOT  directory holding Project-Oracle/ and human-review/
                   (default: the parent of this repository)
Exit: 0 all checks pass and the plant is caught; 1 a check failed;
      2 the planted false value was NOT caught.
"""
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent                      # convergence-paper/
REPO = PAPER.parent                      # published-research checkout
RESEARCH = Path(os.environ.get("RESEARCH_ROOT", REPO.parent))
ORACLE = RESEARCH / "Project-Oracle"
HUMAN_REVIEW = RESEARCH / "human-review"

RESULTS = []          # (status, check_id, detail)


def record(cid, ok, detail, held=False):
    status = ("HELD" if held else "OK  ") if ok else "FAIL"
    RESULTS.append((status, cid, detail))
    return ok


# --------------------------------------------------------------------------
# Paper text
# --------------------------------------------------------------------------
def strip_comments(tex):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in tex.splitlines())


def squash(s):
    return re.sub(r"\s+", " ", s)


def load_paper():
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    parts = {"main": main}
    for name in ("circumplex_subsection", "substrate_independence_section"):
        parts[name] = (PAPER / f"{name}.tex").read_text(encoding="utf-8")
    full = main
    for name in ("circumplex_subsection", "substrate_independence_section"):
        full = full.replace("\\input{%s}" % name, parts[name])
    m = re.search(r"% BEGIN-CORRECTION-NOTE(.*?)% END-CORRECTION-NOTE", full, re.S)
    note = squash(strip_comments(m.group(1))) if m else ""
    body = full[:m.start()] + full[m.end():] if m else full
    return squash(strip_comments(body)), note, squash(strip_comments(full)), full


BODY, NOTE, ALL, RAW = load_paper()
BIB = (PAPER / "references.bib").read_text(encoding="utf-8")


def present(cid, needle, where=None, count=1):
    hay = BODY if where is None else where
    n = hay.count(needle)
    return record(cid, n >= count, "present x%d (need >=%d): %r" % (n, count, needle))


def absent(cid, needle, where=None):
    hay = BODY if where is None else where
    n = hay.count(needle)
    return record(cid, n == 0, "absent (found x%d): %r" % (n, needle))


def absent_re(cid, pattern, where=None):
    hay = BODY if where is None else where
    hits = re.findall(pattern, hay)
    return record(cid, not hits, "absent /%s/ (found %r)" % (pattern, hits[:3]))


# --------------------------------------------------------------------------
# Exact one-sided Fisher test (alternative: first row has the higher rate)
# --------------------------------------------------------------------------
def fisher_greater(a, n1, c, n2):
    """P(X >= a) for X ~ Hypergeometric: a of n1 vs c of n2 events."""
    k, N = a + c, n1 + n2
    denom = math.comb(N, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x)
               for x in range(a, min(n1, k) + 1)) / denom


# --------------------------------------------------------------------------
# Primaries
# --------------------------------------------------------------------------
def oracle_json(relpath):
    p = ORACLE / relpath
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def oracle_git_json(commit, relpath):
    if not (ORACLE / ".git").exists():
        return None
    out = subprocess.run(["git", "-C", str(ORACLE), "show", "%s:%s" % (commit, relpath)],
                         capture_output=True, text=True, encoding="utf-8")
    return json.loads(out.stdout) if out.returncode == 0 else None


def count(trials, **match):
    sel = [t for t in trials if all(t.get(k) == v for k, v in match.items())]
    return sum(bool(t["final_deceptive"]) for t in sel), len(sel)


# HELD values, used only when the primary is not on this machine.
# Exploratory proof: Project-Oracle 9cd4187 (2026-07-09),
#   experiments/results/behavioral_proof/behavioral_proof.json  (baseline, pressure: 16/20)
#   experiments/results/behavioral_proof/behavioral_proof_profile_a10.json (loop, pressure: 2/20)
#   Also targeted-deception-correction/paper.tex Table 1 (published-research b887872).
HELD_EXPLORATORY = {"baseline": (16, 20), "corrected": (2, 20), "temperature": 0.7}
# Confirmatory replication: Project-Oracle 0cb2356 (2026-07-10, branch
#   worktree-confirmatory-replication), experiments/results/confirmatory_replication/
#   confirmatory_replication.json -> summary.deceptive_counts_pressure,
#   summary.primary_endpoint, summary.secondary_endpoints.2_placebo_effect, config.
HELD_CONFIRMATORY = {
    "baseline": (9, 30), "corrected": (4, 30), "placebo": (12, 30),
    "cells_baseline": (5, 15), "cells_corrected": (3, 15), "cells_placebo": (8, 15),
    "p_primary": 0.3408295852073963, "p_placebo_trial": 0.019548568837274384,
    "success": False, "steer_layers": [35, 39, 43, 47],
    "detect_layers": [27, 31, 35, 39, 43, 47], "temperature": 0.7,
}


def exploratory():
    base = oracle_json("experiments/results/behavioral_proof/behavioral_proof.json")
    corr = oracle_json("experiments/results/behavioral_proof/behavioral_proof_profile_a10.json")
    if base is None or corr is None:
        return dict(HELD_EXPLORATORY), True
    return {
        "baseline": count(base["trials"], phase="baseline", group="pressure"),
        "corrected": count(corr["trials"], phase="loop", group="pressure"),
        "temperature": base["config"]["temperature"],
        "corr_mode": corr["config"].get("correction_mode"),
    }, False


def confirmatory():
    d = oracle_git_json("0cb2356",
                        "experiments/results/confirmatory_replication/confirmatory_replication.json")
    if d is None:
        return dict(HELD_CONFIRMATORY), True
    s, pe = d["summary"], d["summary"]["primary_endpoint"]
    out = {
        "baseline": count(d["trials"], arm="baseline", group="pressure"),
        "corrected": count(d["trials"], arm="corrected", group="pressure"),
        "placebo": count(d["trials"], arm="placebo", group="pressure"),
        "cells_baseline": (pe["baseline_vs_corrected"]["hi_deceptive_cells"],
                           pe["baseline_vs_corrected"]["hi_n_cells"]),
        "cells_corrected": (pe["baseline_vs_corrected"]["lo_deceptive_cells"],
                            pe["baseline_vs_corrected"]["lo_n_cells"]),
        "cells_placebo": (pe["baseline_vs_placebo"]["lo_deceptive_cells"],
                          pe["baseline_vs_placebo"]["lo_n_cells"]),
        "p_primary": pe["baseline_vs_corrected"]["p_one_sided"],
        "p_placebo_trial": s["secondary_endpoints"]["2_placebo_effect"]["placebo_gt_corrected_p"],
        "success": pe["success_criterion_met"],
        "steer_layers": d["config"]["steer_layers"],
        "detect_layers": d["config"]["detect_layers"],
        "temperature": d["config"]["temperature"],
        "summary_counts": s["deceptive_counts_pressure"],
    }
    return out, False


def meta_pattern_ledger():
    """Rows of meta-pattern Table 1 (tab:bodycount): (finding, status, type)."""
    tex = (REPO / "meta-pattern" / "main.tex").read_text(encoding="utf-8")
    block = tex[tex.index("\\label{tab:bodycount}"):]
    block = block[:block.index("\\end{tabular}")]
    rows = []
    for line in block.splitlines():
        m = re.match(r"\s*(.+?) & (Confirmed|Suspected|Falsified|\\emph\{Withdrawn\}) & "
                     r"(Meta\??|Content) &", line)
        if m:
            name = re.sub(r"\$\^\{\\[A-Za-z]+\}\$", "", m.group(1)).strip()
            rows.append((name, m.group(2).replace("\\emph{Withdrawn}", "Withdrawn"), m.group(3)))
    return rows


def paper_ledger():
    if "\\label{tab:bodycount}" not in RAW:
        return []
    block = RAW[RAW.index("\\label{tab:bodycount}"):]
    block = block[:block.index("\\end{tabular}")]
    rows = []
    for line in block.splitlines():
        m = re.match(r"\s*(.+?) & (Confirmed|Suspected|Falsified|Withdrawn) & (Meta\??|Content) \\\\",
                     line)
        if m:
            rows.append((m.group(1).strip(), m.group(2), m.group(3)))
    return rows


def read_if(path):
    return path.read_text(encoding="utf-8") if path.exists() else None


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def pct(k, n):
    return 100.0 * k / n


def check_c20():
    ex, held = exploratory()
    record("C20.prim.exploratory", ex["baseline"] == (16, 20) and ex["corrected"] == (2, 20),
           "exploratory baseline %s, corrected %s (raw trials)" % (ex["baseline"], ex["corrected"]),
           held)
    b, c = ex["baseline"], ex["corrected"]
    s_ex = "from 80\\%% to 10\\%% (%d/%d to %d/%d)" % (b[0], b[1], c[0], c[1])
    assert round(pct(*b)) == 80 and round(pct(*c)) == 10
    present("C20.text.exploratory(abstract+2.2)", s_ex, count=2)

    cf, held = confirmatory()
    if not held:
        record("C20.prim.confirmatory.summary-agrees",
               cf["summary_counts"] == {"baseline": "9/30", "corrected": "4/30", "placebo": "12/30"},
               "trial recount %s/%s/%s vs JSON summary %s" % (cf["baseline"], cf["corrected"],
                                                             cf["placebo"], cf["summary_counts"]))
    record("C20.prim.confirmatory",
           (cf["baseline"], cf["corrected"], cf["placebo"]) == ((9, 30), (4, 30), (12, 30)),
           "confirmatory baseline %s, corrected %s, placebo %s" % (cf["baseline"], cf["corrected"],
                                                                  cf["placebo"]), held)
    p1 = fisher_greater(*cf["cells_baseline"], *cf["cells_corrected"])
    record("C20.prim.p-primary(recomputed)", abs(p1 - cf["p_primary"]) < 1e-9 and f"{p1:.2f}" == "0.34",
           "baseline-vs-CORRECTED cells %s vs %s: recomputed one-sided p=%.4f, JSON %.4f, "
           "criterion met=%s" % (cf["cells_baseline"], cf["cells_corrected"], p1, cf["p_primary"],
                                 cf["success"]), held)
    record("C20.prim.primary-not-met", cf["success"] is False, "success_criterion_met is False", held)
    p2 = fisher_greater(*cf["placebo"], *cf["corrected"])
    record("C20.prim.p-placebo-trial(recomputed)",
           abs(p2 - cf["p_placebo_trial"]) < 1e-9 and f"{p2:.3f}" == "0.020",
           "placebo>corrected trials %s vs %s: one-sided p=%.4f (JSON %.4f) -> 0.020"
           % (cf["placebo"], cf["corrected"], p2, cf["p_placebo_trial"]), held)
    p3 = fisher_greater(*cf["cells_placebo"], *cf["cells_corrected"])
    record("C20.prim.p-placebo-cell(recomputed)", f"{p3:.3f}" == "0.064",
           "placebo>corrected cells %s vs %s: one-sided p=%.4f -> 0.064"
           % (cf["cells_placebo"], cf["cells_corrected"], p3), held)

    present("C20.text.confirmatory", "30\\% to 13\\%, 9/30 to 4/30; one-sided $p = 0.34$")
    present("C20.text.confirmatory(2.2)", "30\\% to 13\\%, one-sided $p = 0.34$")
    present("C20.text.placebo", "(4/30 against 12/30): one-sided $p = 0.020$ at the trial level, "
                                "$0.064$ at the pre-registered cell level")
    present("C20.text.status", "did not meet its primary endpoint", count=4)
    absent("C20.gone.range", "80$--$100")
    absent("C20.gone.range2", "80--100")
    absent("C20.gone.placebo-0.019", "0.019")
    absent("C20.gone.confab-label", "confabulation correction")
    absent("C20.gone.three-clean", "three clean controls")

    record("C20.prim.layers", cf["steer_layers"] == [35, 39, 43, 47]
           and cf["detect_layers"] == [27, 31, 35, 39, 43, 47],
           "steer %s, detect %s" % (cf["steer_layers"], cf["detect_layers"]), held)
    present("C20.text.layers(substrate)", "detects at L27, L31, L35, L39, L43 and L47 and corrects "
                                          "at L35, L39, L43 and L47")
    present("C20.text.layers(2.3)", "detects at L27--L47 and corrects at L35, L39, L43 and L47")
    absent("C20.gone.L11", "L11")
    absent("C20.gone.17pct", "$17\\%$ depth")
    record("C20.prim.temperature", ex["temperature"] == 0.7 and cf["temperature"] == 0.7,
           "T exploratory %s, confirmatory %s" % (ex["temperature"], cf["temperature"]), held)
    present("C20.text.temperature", "Oracle Loop deception-correction experiments ($T{=}0.7$")


def check_bodycount():
    rows = meta_pattern_ledger()
    tally = {s: sum(r[1] == s for r in rows) for s in ("Confirmed", "Suspected", "Falsified", "Withdrawn")}
    record("BC.prim.meta-pattern", tally == {"Confirmed": 6, "Suspected": 7, "Falsified": 10, "Withdrawn": 1},
           "meta-pattern tab:bodycount tally %s (Lyra's gated 6/7/10/1)" % tally)
    s = "%d~confirmed, %d~suspected, %d~falsified, %d~withdrawn" % (
        tally["Confirmed"], tally["Suspected"], tally["Falsified"], tally["Withdrawn"])
    present("BC.text.abstract", s)
    present("BC.text.section", "comprises %d~confirmed findings, %d~at suspected status (signal present, "
                               "controls incomplete), %d~honestly falsified, and %d~withdrawn"
            % (tally["Confirmed"], tally["Suspected"], tally["Falsified"], tally["Withdrawn"]))
    absent("BC.gone.8", "8~confirmed")
    absent("BC.gone.plus", "10+")
    absent("BC.gone.7-7", "7~confirmed")
    mine = paper_ledger()
    record("BC.table.diff", mine == rows,
           "paper tab:bodycount %d rows vs meta-pattern %d rows; first mismatch: %r" % (
               len(mine), len(rows),
               next(((a, b) for a, b in zip(mine, rows) if a != b), None)))
    meta_types = {r[2] for r in rows if r[1] == "Confirmed"}
    content_types = {r[2] for r in rows if r[1] == "Falsified"}
    record("BC.prim.split", meta_types == {"Meta"} and content_types == {"Content"},
           "confirmed types %s, falsified types %s" % (meta_types, content_types))
    present("BC.text.narrowed", "The ten falsified findings in the tally all attempted content-level measurement.")
    present("BC.text.ast", "every falsified finding in the tally attempted to measure")
    present("BC.text.ast2", "every confirmed finding in our body count")
    present("BC.text.sprint", "Only the third is a content-level claim")
    absent("BC.gone.all-falsified", "The falsified findings all attempted")
    absent("BC.gone.surviving", "every surviving finding")


def check_depth():
    # R1-v2 layer indices are unchanged from v1.0 (main.tex section 6); only the
    # denominator changes, to l/n_layers with n = 64 (substrate section's convention).
    n27, n8 = 64, 36
    vals = {
        "sr_half_rise": (17.6, n27, "$27.5\\%$ of model depth"),
        "jl_half_rise": (56.2, n27, "$87.8\\%$ depth"),
        "onset": (14, n27, "$21.9\\%$ depth"),
        "peak": (19, n27, "$29.7\\%$ depth), just below the $31$--$33\\%$ band"),
        "L15": (15, n8, "at L15 ($41.7\\%$ depth"),
    }
    for k, (layer, n, needle) in vals.items():
        num = re.search(r"\$(\d+\.\d)\\%\$", needle).group(1)
        record("DEPTH.calc.%s" % k, f"{pct(layer, n):.1f}" == num,
               "%s / %d = %.2f%% (text says %s%%)" % (layer, n, pct(layer, n), num))
        present("DEPTH.text.%s" % k, needle)
    gap = 38.6   # R1-v2 primary statistic; Lyra's gated key number "38.6 layers [38.0-39.2]"
    record("DEPTH.calc.gap-share", round(pct(gap, n27)) == 60, "38.6/64 = %.1f%%" % pct(gap, n27))
    present("DEPTH.text.gap-share", "Spectral features engage 60\\% of the network earlier")
    record("DEPTH.calc.band8b", (round(pct(12, n8)), round(pct(21, n8))) == (33, 58),
           "L12/36, L21/36 = %.1f%%, %.1f%%" % (pct(12, n8), pct(21, n8)))
    present("DEPTH.text.band8b", "pre-registered band L12--L21, $33$--$58\\%$ depth")
    present("DEPTH.text.onset-miss", "${\\sim}3$ percentage points")   # 25 - 21.9 = 3.1
    present("DEPTH.text.convention", "Relative depth is $l/n_\\text{layers}$ throughout")
    absent("DEPTH.gone.30.2", "30.2")
    absent("DEPTH.gone.22.2", "22.2")
    absent("DEPTH.gone.89", "89\\%")
    absent("DEPTH.gone.61", "61\\%")
    absent("DEPTH.gone.selfref", "convergence paper's")


def check_r1():
    # Gated key number (Lyra): R1 temporal gap 38.6 layers [38.0-39.2], 100% forks positive, p<0.005.
    present("R1.text.gap", "\\textbf{38.6~layers} [CI: $38.0$--$39.2$]")
    present("R1.text.gap-uses", "38.6", count=6)
    absent_re("R1.gone.39", r"(?<![\d.])39(?:~|-)layer")
    present("R1.text.linear-only", "39.0~layers [CI: $38.4$--$39.6$]")
    record("R1.dup.removed", BODY.count("14.5~layers") == 1,
           "14.5~layers occurs %d time(s)" % BODY.count("14.5~layers"))


def check_citations_and_numbers():
    # 0.620 / 0.707: oracle-loop-paper/sections/results.tex:57-58 (published-research 09d07a7)
    res = (REPO / "oracle-loop-paper" / "sections" / "results.tex").read_text(encoding="utf-8")
    ok = ("gen\\_only\\_MP & No FWL & \\textbf{0.707} ($p{=}0.001$)" in res
          and "gen\\_only\\_MP & + FWL belt & 0.620 ($p{=}0.023$)" in res)
    record("FWL.prim.oracle-table", ok, "Oracle Loop tab:primary: 0.707 no FWL, 0.620 with FWL belt (Qwen)")
    present("FWL.text.label", "AUROC~$0.620$ after FWL residualization, $0.707$ without FWL, on Qwen2.5-7B)~\\citep{lyra2026oracle}")
    present("FWL.text.list", "AUROC~$0.620$ (FWL-corrected, permutation $p{=}0.023$; $0.707$ without FWL, $p{=}0.001$)")
    absent_re("FWL.gone.0.999", r"0\.999(?!\d)")   # whole number; 0.9999 (hardware r) is not it
    absent("FWL.gone.decision-cite", "lyra2026decision")
    absent("FWL.gone.scope", "for all classification AUROCs")
    present("FWL.text.scope", "wherever an AUROC is labelled FWL-corrected")

    # Mistral text baseline: human-review dual-detector-paper/main.tex:317,321 (local working
    # tree; preserved in human-review's local-only commit 1a92479). Held if absent.
    dd = read_if(HUMAN_REVIEW / "dual-detector-paper" / "main.tex")
    if dd is None:
        record("DD.prim", True, "HELD text 0.820, dual 0.806", held=True)
    else:
        record("DD.prim", "Text features & 7 & \\textbf{0.820}" in dd and "Dual (geo + ent) & 11 & 0.806" in dd,
               "dual-detector tab:textbaseline text 0.820, dual 0.806")
    record("DD.calc.exceeds", 0.820 > 0.806, "0.820 > 0.806, so 'matched' alone understates")
    present("DD.text.deflationary", "text features ($0.820$) matched or slightly exceeded the dual detector "
                                    "($0.806$) for confabulation detection~\\citep{lyra2026dualdetector}")
    present("DD.text.reflection", "Text features matched or slightly exceeded our dual")
    record("DD.bib", "@unpublished{lyra2026dualdetector," in BIB, "bib entry lyra2026dualdetector")

    # Hardware invariance: lyra-technique-ii/main.tex:163 (published-research f2e0f77, 2026-10-10;
    # Experiment 37, RTX 3090 vs H200 JSONs). The v1.0 'r>0.999' overstated two of four features.
    lt = (REPO / "lyra-technique-ii" / "main.tex").read_text(encoding="utf-8")
    record("HW.prim", "$r = 0.9986$--$0.9999$" in lt, "lyra-technique-ii: r = 0.9986--0.9999")
    present("HW.text", "across hardware ($r = 0.9986$--$0.9999$ across four cache features)")
    absent("HW.gone", "r{>}0.999")

    # Schema commitment: meta-pattern/main.tex tab:bodycount + footnote (5cfaa86).
    mp = (REPO / "meta-pattern" / "main.tex").read_text(encoding="utf-8")
    record("SC.prim", "Schema commitment$^{\\S}$ & Confirmed & Meta" in mp
           and "$d{=}0.91$ over the first 10 tokens" in mp, "meta-pattern: schema commitment, d=0.91 first 10 tokens")
    present("SC.text", "Schema commitment: $d{=}0.91$ over the first 10 tokens")
    absent("SC.gone", "confab more confident")

    # Circumplex: empathy-bus/main.tex:162 (c80e8da) reports cos 0.83-0.87 and no 0.987 peak.
    eb = (REPO / "empathy-bus" / "main.tex").read_text(encoding="utf-8")
    record("CIRC.prim", "$\\cos = 0.83$--$0.87$ (mid-to-late layers)" in eb and "0.987" not in eb,
           "empathy-bus: cos 0.83--0.87, no 0.987")
    present("CIRC.text", "$\\cos = 0.83$--$0.87$ (mid-to-late layers)")
    absent("CIRC.gone.0.987", "0.987")
    absent("CIRC.gone.instrument5", "Instrument 5")

    # Decomposition model and band: human-review gwt-response/residual_decomposition_prereg.json
    # (local working tree; local-only commit 1a92479): model Qwen/Qwen3-8B, H1 band L12-L21.
    pr = read_if(HUMAN_REVIEW / "gwt-response" / "residual_decomposition_prereg.json")
    if pr is None:
        record("DEC.prim", True, "HELD model Qwen3-8B, band L12-L21", held=True)
    else:
        d = json.loads(pr)
        record("DEC.prim", d["model"].startswith("Qwen/Qwen3-8B") and "L12-L21" in d["hypotheses"][0],
               "prereg model %r; H1 %r" % (d["model"][:14], d["hypotheses"][0][:70]))
    present("DEC.text", "On Qwen3-8B (36~layers), at L15")

    # Scale replication list (v1.0 named a checkpoint that does not exist and a 72B outside the range).
    absent("SCALE.gone.0.6", "Qwen2.5-0.6B")
    absent("SCALE.gone.72", "Qwen2.5-72B")
    absent("SCALE.gone.seven", "seven additional models")
    present("SCALE.text", "comes from the Campaign~2 scale sweep~\\citep{lyra2026campaign2}")

    # Temperature, data availability.
    present("TEMP.text", "Unless stated otherwise, generation used temperature${=}0$ (greedy)")
    present("TEMP.text.r1", "the R1 mechanism test ($T{=}0.7$, five seeded forks per prompt")
    absent("ZEN.gone", "archived on Zenodo")
    present("ZEN.text", "does not yet have a public data deposit with a DOI")

    # Bibliography.
    record("BIB.gurnee", re.search(r"@misc\{gurnee2026gwt,", BIB) is not None, "gurnee2026gwt is @misc")
    record("BIB.oracle-title", "Self-Regulating {AI} Through {KV}-Cache" in BIB
           and "Real-Time Confabulation Detection" not in BIB, "lyra2026oracle carries the paper's real title")
    # cc2026targeted DOI: meta-pattern/references.bib:303-304 (5cb0e4b); same DOI in cache-tracing,
    # consequentiality-decomposition and adversarial-audit-methodology bibliographies.
    record("BIB.cc2026targeted", "@misc{cc2026targeted," in BIB and "10.5281/zenodo.21754921" in BIB,
           "cc2026targeted present with DOI 10.5281/zenodo.21754921")
    record("BIB.metapattern", "@misc{lyra2026metapattern," in BIB, "lyra2026metapattern present")
    present("BIB.cited.cc", "\\citep{cc2026targeted}", count=2)


def check_note_and_date():
    m = re.search(r"\\date\{([^\n]*)", RAW)
    date = m.group(1) if m else ""
    record("DATE.v11", "Version~1.1" in date and re.search(r"\d{4}-\d{2}-\d{2}", date) is not None,
           "\\date line: %r" % date[:90])
    record("DATE.gone.july-only", "\\date{July 2026}" not in RAW, "bare July date removed")
    record("NOTE.exists", bool(NOTE), "correction-note block found (%d chars)" % len(NOTE))
    for needle in ("Version 1.1",
                   "16/20 to 2/20",
                   "did not meet its primary endpoint",
                   "6~confirmed, 7~suspected, 10~falsified and 1~withdrawn",
                   "$l/n_\\text{layers}$",
                   "38.6~layers",
                   "matched or slightly exceeded"):
        present("NOTE.has %s" % needle[:30], needle, where=NOTE)


# --------------------------------------------------------------------------
# The plant
# --------------------------------------------------------------------------
def run_plant():
    """Project-Oracle FINDINGS_REGISTRY.md:337 heads C20 'exploratory 80%->0%'. That is false:
    the corrected arm deceived on 2/20 trials. Planted here as a claim; the gate must reject it
    against the primary AND against the text."""
    planted = (0, 20)
    ex, held = exploratory()
    caught_primary = ex["corrected"] != planted
    caught_text = BODY.count("from 80\\% to 0\\% (16/20 to 0/20)") == 0
    return caught_primary, caught_text, ex["corrected"], held


def main():
    check_c20()
    check_bodycount()
    check_depth()
    check_r1()
    check_citations_and_numbers()
    check_note_and_date()

    width = max(len(c) for _, c, _ in RESULTS)
    for status, cid, detail in RESULTS:
        print("  %s  %-*s  %s" % (status, width, cid, detail))
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    held = [r for r in RESULTS if r[0] == "HELD"]

    cp, ct, actual, plant_held = run_plant()
    print()
    print("  PLANT  corrected arm claimed 0/20 (registry heading '80%%->0%%'); primary says %s/%s%s"
          % (actual[0], actual[1], " [HELD]" if plant_held else ""))
    print("         caught against primary: %s; caught against text: %s" % (cp, ct))

    print()
    print("  %d checks: %d OK, %d HELD (primary not machine-readable here), %d FAIL"
          % (len(RESULTS), len(RESULTS) - len(fails) - len(held), len(held), len(fails)))
    if not (cp and ct):
        print("  GATE BROKEN: the planted false value was not caught.")
        sys.exit(2)
    print("  PLANT CAUGHT.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
