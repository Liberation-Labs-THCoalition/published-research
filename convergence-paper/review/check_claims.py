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
    # Revised in Phase B: the abstract no longer restates the exploratory counts; section 2.2 keeps them.
    present("C20.text.exploratory(2.2)", s_ex, count=1)

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

    # Revised in Phase B: the abstract's Oracle clause is shorter; the counts stay in section 2.2.
    present("C20.text.confirmatory", "did not meet its primary endpoint (one-sided $p = 0.34$)")
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
    # Tightened in Phase B: "L11" now also occurs in the ghost probe's "L11--L48"; v1.0's correction-layer
    # claims ("corrects at L11." and "applied at L11,") are still caught.
    absent_re("C20.gone.L11", r"L11(?![-\d])")
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
    # Revised in Phase B: the body count is no longer in the abstract (rebuilt around the tally); the
    # section 8 count below (BC.text.section) carries it.
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
    # Revised in Phase B: the Attention Schema paragraph is shortened and hedged; its narrowed claim reads:
    present("BC.text.ast", "confirmed findings measure processing state, and the falsified ones tried to measure content")
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
    # Tightened in the v1.1 extension: "89\%" alone also matches instrument 5's "77--89\%".
    # v1.0 wrote the depth as "${\sim}89\%$" (twice), which this still catches.
    absent("DEPTH.gone.89", "{\\sim}89\\%")
    absent("DEPTH.gone.61", "61\\%")
    absent("DEPTH.gone.selfref", "convergence paper's")


def check_r1():
    # Gated key number (Lyra): R1 temporal gap 38.6 layers [38.0-39.2], 100% forks positive, p<0.005.
    present("R1.text.gap", "\\textbf{38.6~layers} [CI: $38.0$--$39.2$]")
    present("R1.text.gap-uses", "38.6", count=4)   # Phase B: fewer restatements; the shuffled null says 39.0
    absent_re("R1.gone.39", r"(?<![\d.])39(?:~|-)layer")
    present("R1.text.linear-only", "39.0~layers [CI: $38.4$--$39.6$]")
    # Tightened in Phase B: the Interpretation now restates the number once ("the full-attention-only gap is
    # 14.5~layers"); finding #9 was the full result printed twice, which this still catches.
    full = "14.5~layers [CI: $11.7$--$17.4$]"
    record("R1.dup.removed", BODY.count(full) == 1, "%r occurs %d time(s)" % (full, BODY.count(full)))


def check_citations_and_numbers():
    # 0.620 / 0.707: oracle-loop-paper/sections/results.tex:57-58 (published-research 09d07a7)
    res = (REPO / "oracle-loop-paper" / "sections" / "results.tex").read_text(encoding="utf-8")
    ok = ("gen\\_only\\_MP & No FWL & \\textbf{0.707} ($p{=}0.001$)" in res
          and "gen\\_only\\_MP & + FWL belt & 0.620 ($p{=}0.023$)" in res)
    record("FWL.prim.oracle-table", ok, "Oracle Loop tab:primary: 0.707 no FWL, 0.620 with FWL belt (Qwen)")
    present("FWL.text.label", "AUROC~$0.620$ after FWL residualization, $0.707$ without FWL, on Qwen2.5-7B)~\\citep{lyra2026oracle}")
    # Revised in Phase B: section 4.6's list is folded into the key-cache subsection (REL.text.keycache-p).
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
    # Revised in the v1.1 extension (Lyra, PR #13): the sentence now names the four key-cache
    # features once, so the needle ends "across the four features".
    present("HW.text", "across hardware ($r = 0.9986$--$0.9999$ across the four features)")
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
    # Revised in the v1.1 extension: instrument 5 is restated from PR #3's data files, so the old
    # "cos 0.83-0.87 (mid-to-late layers)" sentence is gone; the replacement is checked in INST5.*.
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
    # Revised in the v1.1 extension: the model list is now rebuilt from the primary (SCALE2.*).

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


# --------------------------------------------------------------------------
# v1.1 extension (Lyra's read of PR #13): hardware, instrument 5, K/V split, "reliably", scale.
# Primaries on other branches are read with `git show <ref>:<path>` from this repository; if the
# ref is gone (e.g. merged and deleted) the master working tree is tried, then held values.
# --------------------------------------------------------------------------
PR3_REF = "origin/nexus/empathy-bus-41-42"        # published-research PR #3 (draft), data at 9d60363
PR8_REF = "origin/nexus/pharos-corrections-v11"   # published-research PR #8 (draft), head 0467721
KVX = RESEARCH.parent / "KV-Experiments"          # Campaign 2 repo (paper-scale-invariance at 100f38a)


def repo_json(ref, path):
    out = subprocess.run(["git", "-C", str(REPO), "show", "%s:%s" % (ref, path)],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode == 0:
        return json.loads(out.stdout), False
    p = REPO / path
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")), False
    return None, True


def rng(vals, nd):
    return "%.*f--%.*f" % (nd, min(vals), nd, max(vals))


def check_extension():
    # Hardware. HELD: Lyra ran `system_profiler SPHardwareDataType` on Starship on 2026-10-20
    # (message on PR #13): Mac Studio, Mac15,14, Apple M3 Ultra, 256 GB. Not machine-readable here.
    record("HW2.prim", True, "Starship = Mac Studio Mac15,14, Apple M3 Ultra, 256 GB (Lyra, system_profiler)",
           held=True)
    present("HW2.text", "Mac Studio M3 Ultra, 256\\,GB unified memory")
    absent("HW2.gone", "M4 Max")

    # ---- Instrument 5, from PR #3's data files ----
    L17 = [str(l) for l in range(17, 64)]
    wsn, h1 = repo_json(PR3_REF, "empathy-bus/data/within_system_null.json")
    axes, h2 = repo_json(PR3_REF, "empathy-bus/data/story_space_axes.json")
    csa, h3 = repo_json(PR3_REF, "empathy-bus/data/cross_system_alignment.json")
    ecc, h4 = repo_json(PR3_REF, "empathy-bus/data/user_model_eccentricity.json")
    cpl, h5 = repo_json(PR3_REF, "empathy-bus/data/coupling_test.json")
    if not any((h1, h2, h3, h4, h5)):
        W = wsn["layers"]
        vals = {
            "cos_hc": rng([W[l]["valence"]["abs_cos_pc1"] for l in L17], 2),
            "cos_dc": rng([W[l]["arousal"]["abs_cos_pc1"] for l in L17], 2),
            "plane": rng([100 * W[l]["plane"]["var_share"] for l in L17], 0),
            "pct_hc": rng([W[l]["valence"]["pct_below"] for l in L17], 0),
            "pct_dc": rng([W[l]["arousal"]["pct_below"] for l in L17], 0),
            "plane_ratio": rng([W[l]["plane"]["var_share"] / W[l]["planes_var_share"]["median"] for l in L17], 2),
            "rho_val": rng([abs(axes[l]["pc1_val"]) for l in L17], 2),
            "rho_aro": rng([abs(axes[l]["pc1_aro"]) for l in L17], 2),
            "opposite": all(axes[l]["pc1_val"] > 0 > axes[l]["pc1_aro"] for l in L17),
            "n_pairs": wsn["n_pairs"],
        }
        C = csa["layers"]
        vals.update({
            "x_val": rng([x["content_valence"]["abs_cos_PC1"] for x in C], 3),
            "x_aro": rng([x["content_arousal"]["abs_cos_PC1"] for x in C], 3),
            "null_p95": rng([x["null_random_direction"]["abs_cos_PC1_p95"] for x in C], 3),
            "null_max": rng([x["null_random_direction"]["abs_cos_PC1_max"] for x in C], 3),
            "x_share": rng([100 * x[k]["centroid_variance_share_along_direction"] for x in C
                            for k in ("content_valence", "content_arousal")], 2),
            "x_ratio": "%d to %d" % (round(min(x[k]["centroid_variance_share_along_direction"]
                                               / x["null_random_direction"]["variance_share_mean"]
                                               for x in C for k in ("content_valence", "content_arousal"))),
                                     round(max(x[k]["centroid_variance_share_along_direction"]
                                               / x["null_random_direction"]["variance_share_mean"]
                                               for x in C for k in ("content_valence", "content_arousal")))),
            "L15_within": all(x[k]["abs_cos_PC1"] < x["null_random_direction"]["abs_cos_PC1_p95"]
                              for x in C if x["layer"] == 15 for k in ("content_valence", "content_arousal")),
            "L25up_exceed": all(x[k]["abs_cos_PC1"] > x["null_random_direction"]["abs_cos_PC1_max"]
                                for x in C if x["layer"] > 15 for k in ("content_valence", "content_arousal")),
        })
        U, Cc = ecc["user_model"], ecc["content"]
        vals.update({
            "ecc_content": rng([Cc["L17-L63"]["min"], Cc["L17-L63"]["max"]], 2),
            "ecc_story_a": rng([U["L17-L21"]["min"], U["L17-L21"]["max"]], 2),
            "ecc_story_b": rng([U["L22-L63"]["min"], U["L22-L63"]["max"]], 2),
            "ecc_r": "%.2f" % ecc["pearson_r_user_model_vs_content_all_layers"],
        })
        R = cpl["results"]
        mean_off = {c: sum(x["cross_pathway_cosine"] for x in R if x["condition"] == c
                           and x["measurement_layer"] != x["injection_layer"])
                    / sum(1 for x in R if x["condition"] == c and x["measurement_layer"] != x["injection_layer"])
                    for c in ("valence", "arousal", "random")}
        same_min = min(x["cross_pathway_cosine"] for x in R if x["condition"] in ("valence", "arousal", "random")
                       and x["measurement_layer"] == x["injection_layer"])
        vals.update({"m_val": "%.3f" % mean_off["valence"], "m_aro": "%.3f" % mean_off["arousal"],
                     "m_rnd": "%.3f" % mean_off["random"], "same_min": same_min,
                     "n_content": cpl["n_content"], "n_user": cpl["n_user"],
                     "inj": cpl["injection_layers"], "alphas": cpl["alphas"]})
        held5 = False
    else:
        # HELD from PR #3's data files at 9d60363 (computed 2026-10-20 by this script's code path).
        vals = {"cos_hc": "0.68--0.80", "cos_dc": "0.83--0.88", "plane": "33--37", "pct_hc": "77--89",
                "pct_dc": "92--95", "plane_ratio": "1.26--1.36", "rho_val": "0.66--0.80", "rho_aro": "0.49--0.67",
                "opposite": True, "n_pairs": 435, "x_val": "0.026--0.110", "x_aro": "0.015--0.126",
                "null_p95": "0.027--0.028", "null_max": "0.044--0.052", "x_share": "0.08--0.68",
                "x_ratio": "4 to 34", "L15_within": True, "L25up_exceed": True, "ecc_content": "0.37--0.61",
                "ecc_story_a": "0.05--0.23", "ecc_story_b": "0.30--0.42", "ecc_r": "0.09", "m_val": "0.521",
                "m_aro": "0.547", "m_rnd": "0.577", "same_min": 0.9999, "n_content": 250, "n_user": 60,
                "inj": [15, 25, 35, 45], "alphas": [1.0, 3.0, 5.0]}
        held5 = True
    record("INST5.prim.within", vals["opposite"] and vals["n_pairs"] == 435,
           "L17-L63: hc %s, dc %s, plane %s%%, pct_below hc %s dc %s, plane/median %s, rho val %s aro %s"
           % (vals["cos_hc"], vals["cos_dc"], vals["plane"], vals["pct_hc"], vals["pct_dc"],
              vals["plane_ratio"], vals["rho_val"], vals["rho_aro"]), held5)
    record("INST5.prim.cross", vals["L15_within"] and vals["L25up_exceed"],
           "content val %s aro %s; random p95 %s max %s; share %s%% (%sx random)"
           % (vals["x_val"], vals["x_aro"], vals["null_p95"], vals["null_max"], vals["x_share"],
              vals["x_ratio"]), held5)
    record("INST5.prim.coupling", vals["same_min"] >= 0.9999 and vals["m_rnd"] >= vals["m_aro"] >= vals["m_val"]
           and (vals["n_content"], vals["n_user"]) == (250, 60) and vals["inj"] == [15, 25, 35, 45],
           "off-layer means random %s arousal %s valence %s; same-layer min %.4f"
           % (vals["m_rnd"], vals["m_aro"], vals["m_val"], vals["same_min"]), held5)
    d = lambda s: "$" + s.replace("--", "$--$") + "$"
    present("INST5.text.story", "30 emotion centroids per layer, each a mean over stories the model wrote "
                                "about a third-person character")
    present("INST5.text.not-user", "not the state of a conversation partner")
    present("INST5.text.rho", "rated valence at $|\\rho| = %s and, with the opposite sign, with rated arousal "
                              "at $|\\rho| = %s" % (d(vals["rho_val"])[1:], d(vals["rho_aro"])[1:]))
    present("INST5.text.cos", "align with that component at $|\\cos| = %s and %s" % (d(vals["cos_hc"])[1:],
                                                                                   d(vals["cos_dc"])))
    present("INST5.text.plane", "holds %s\\%%$ of the centroid variance" % d(vals["plane"])[:-1])
    present("INST5.text.pct", "%s\\%%$ of the %d centroid-pair directions" % (d(vals["pct_hc"])[:-1],
                                                                             vals["n_pairs"]))
    present("INST5.text.pct-dc", "%s\\%%$ less than desperate$-$calm" % d(vals["pct_dc"])[:-1])
    present("INST5.text.plane-ratio", "%s times the variance of the median plane" % d(vals["plane_ratio"]))
    present("INST5.text.cross", "$|\\cos| = %s with the story space's first component and the arousal "
                                "direction %s" % (d(vals["x_val"])[1:], d(vals["x_aro"])))
    present("INST5.text.null", "reach %s at the 95th percentile and %s at most" % (d(vals["null_p95"]),
                                                                                   d(vals["null_max"])))
    present("INST5.text.share", "%s\\%%$ of the centroid variance, %s times" % (d(vals["x_share"])[:-1],
                                                                              vals["x_ratio"]))
    present("INST5.text.ecc", "eccentricity is %s from L17" % d(vals["ecc_content"]))
    present("INST5.text.ecc2", "%s over L17--L21 and %s from L22" % (d(vals["ecc_story_a"]),
                                                                   d(vals["ecc_story_b"])))
    present("INST5.text.ecc-r", "($r = %s$)" % vals["ecc_r"])
    present("INST5.text.sets", "%d content prompts and %d conversational prompts" % (vals["n_content"],
                                                                                   vals["n_user"]))
    present("INST5.text.means", "the random direction's mean cosine ($%s$) is at least as high as valence's "
                                "($%s$) and arousal's ($%s$)" % (vals["m_rnd"], vals["m_val"], vals["m_aro"]))
    present("INST5.text.construction", "1.000 at the injection layer by construction")
    present("INST5.text.cannot", "cannot test whether the two kinds of processing are separable")
    absent_re("INST5.gone.bus", r"(?i)\bbus\b")
    for needle in ("user-model", "inseparable", "mood swing", "Empathy, in this", "1000\\times", "44$--$54",
                   "identical developmental", "23 of 24", "59\\times", "lyra2026weather", "W$_K$ bridge",
                   "Content--Inference", "same geometric subspace", "user-emotion encoding ($12.3"):
        absent("INST5.gone %s" % needle[:22], needle)
    # Revised in Phase B (Lyra's thesis ruling): the abstract is rebuilt around the tally's two lists and no
    # longer lists five instruments; its emotion-geometry clause is checked here instead.
    present("INST5.text.abstract", "The emotion-geometry instrument locates no band, and its injection test "
                                   "shows no direction-specific coupling")
    present("INST5.text.footnote", "decodes the emotions of characters in stories the model wrote, not the "
                                   "user's state")
    record("INST5.bib", "@unpublished{nexus2026coupling," in BIB and "@misc{lyra2026usermodel," in BIB
           and "{lyra2026weather," not in BIB, "nexus2026coupling and lyra2026usermodel in; lyra2026weather out")

    # ---- K/V split injection, from PR #8's data file ----
    kv, hkv = repo_json(PR8_REF, "kv-decomposition-paper/data/decomposition_results.json")
    if kv is not None:
        sc = {r["condition"]: (round(r["multihop_score"] * len(r["multihop_details"])),
                               len(r["multihop_details"]),
                               frozenset(x["q"][:20] for x in r["multihop_details"] if x["score"] > 0))
              for r in kv["results"]}
        ok = (sc["BASELINE"][:2] == (1, 6) and sc["FULL_KV"][:2] == (6, 6) and sc["TEXT_CONTEXT"][:2] == (5, 6)
              and sc["V_ONLY"][:2] == (2, 6) and sc["K_ONLY"][:2] == (2, 6)
              and sc["V_ONLY"][2] == sc["K_ONLY"][2] and sc["BASELINE"][2] < sc["V_ONLY"][2]
              and kv["model"] == "Qwen/Qwen2.5-1.5B")
        record("KV.prim", ok, "baseline %d/6, full %d/6, text %d/6, V %d/6, K %d/6; V and K scored the same "
               "items %s; model %s" % (sc["BASELINE"][0], sc["FULL_KV"][0], sc["TEXT_CONTEXT"][0],
                                       sc["V_ONLY"][0], sc["K_ONLY"][0], sorted(sc["V_ONLY"][2]), kv["model"]))
    else:
        # HELD: PR #8 kv-decomposition-paper/data/decomposition_results.json and main.tex tab:results (0467721).
        record("KV.prim", True, "HELD baseline 1/6, full 6/6, text 5/6, V 2/6, K 2/6 (same two items)", held=True)
    present("KV.text.full", "full KV injection scored 6/6 and the text in the prompt 5/6")
    present("KV.text.halves", "K-only and V-only injection scored 2/6 each, on the same two items")
    present("KV.text.baseline", "uninjected baseline (1/6)")
    present("KV.text.model", "(Qwen2.5-1.5B, one greedy run)")
    absent_re("KV.gone.superadd", r"(?i)superadditiv")
    absent("KV.gone.sum", "0.333 + 0.333")
    absent("KV.gone.complementary", "complementary representations")
    record("KV.bib", "Splitting Cache-Injected Graph Topology" in BIB and "= {K and V Are Complementary" not in BIB,
           "lyra2026nexus_kv carries the v1.1 title")

    # ---- "reliably" ----
    for needle in ("reliably discriminate", "reliably measurable", "reliably distinguish"):
        absent("REL.gone %s" % needle, needle)
    # Revised in Phase B: the abstract, the intro sentence and the section 4.6 list are rewritten or removed;
    # "modestly above chance" now lives in the key-cache subsection, and the abstract gives the AUROCs.
    present("REL.text.keycache", "discriminates modestly above chance")
    present("REL.text.abstract-auroc", "(AUROC $0.620$ after FWL residualization, $0.707$ without, on Qwen2.5-7B)")
    present("REL.text.keycache-p", "AUROC $0.620$ after FWL residualization (permutation $p = 0.023$) and $0.707$ "
                                   "without it ($p = 0.001$)")

    # ---- Scale range, from KV-Experiments paper-scale-invariance/sections/results.tex (100f38a) ----
    sr = read_if(KVX / "paper-scale-invariance" / "sections" / "results.tex")
    if sr is not None:
        rhos = [float(x) for x in re.findall(r"(?:Small|Medium) vs\.\\ (?:Medium|Large) & (0\.\d+)", sr)]
        tiers = re.findall(r"\\textbf\{(?:Small|Medium|Large)\} \([^)]*\): ([^\n]+)", sr)
        n_models = sum(len(t.split(", ")) for t in tiers)
        ok = (sorted(rhos) == [0.826, 0.867, 0.896] and n_models == 10
              and "Qwen3-0.6B" in tiers[0] and "Llama-3.1-70B-q4" in tiers[2])
        record("SCALE2.prim", ok and "%.2f--%.2f" % (min(rhos), max(rhos)) == "0.83--0.90",
               "Exp 26 tier rho %s, %d models, smallest %s, largest %s" % (rhos, n_models,
                                                                          tiers[0][:24], tiers[2][-20:]))
    else:
        record("SCALE2.prim", True, "HELD rho 0.826/0.867/0.896, 10 models, Qwen3-0.6B..Llama-3.1-70B-q4", held=True)
    # Revised in Phase B: the abstract now says only that scale invariance uses key-cache features.
    present("SCALE2.text.abstract", "cross-scale invariance use key-cache features and locate no depth")
    present("SCALE2.text.structure", "Key-cache geometric signatures (key norm, norm per token, effective rank and "
                                     "key entropy) replicate across ten models from $0.6$B to $70$B parameters "
                                     "(Spearman $\\rho = 0.83$--$0.90$ between three size tiers; the two largest "
                                     "models were 4-bit quantized)")
    present("SCALE2.text.setup", "comes from Experiment~26 of Campaign~2~\\citep{lyra2026campaign2}, which "
                                 "measured key-cache features")
    present("SCALE2.text.small", "Qwen3-0.6B and TinyLlama-1.1B")
    present("SCALE2.text.large", "Qwen2.5-32B and Llama-3.1-70B, both 4-bit")
    absent("SCALE2.gone.vproj", "V-projection geometric signatures")
    absent("SCALE2.gone.relist", "we do not re-list its models here")

    for needle in ("story-derived", "2/6", "M3 Ultra", "Experiment~26", "modestly above chance"):
        present("NOTE2.has %s" % needle, needle, where=NOTE)

    # Guard added after a shell heredoc turned "\noindent" into a newline plus "oindent" in this
    # pass (it printed in the PDF). Any LaTeX control word that lost its backslash and landed at
    # the start of a line is caught here, before the build.
    hits = re.findall(r"(?m)^(oindent|extbf|egin\{|item |ewline|ref\{|cite)", RAW.replace("\r", ""))
    record("TEX.mangled-control-word", not hits, "line-initial control words without backslash: %r" % hits[:5])


# --------------------------------------------------------------------------
# Phase A tally (Lyra's ruling on the title): every number in review/INSTRUMENT_TALLY.md.
# Starship result files were read with `grep -h ''` over ssh (nothing run there) into a local
# mirror. Set STARSHIP_MIRROR to that directory to recompute; otherwise the values below are
# HELD, each pinned to its file by sha256 (of the mirrored copy; grep appends one trailing
# newline, so the copy is the original plus 1 byte).
# --------------------------------------------------------------------------
TALLY = PAPER / "review" / "INSTRUMENT_TALLY.md"
MIRROR = Path(os.environ["STARSHIP_MIRROR"]) if os.environ.get("STARSHIP_MIRROR") else None
STARSHIP_FILES = {   # mirror-relative path: (Starship path, sha256 prefix of the copy)
    "lab/r3_band_measurement_20260714_170944.json": ("~/lab/kv-experiments/", "8e5f3945b6aa11ea"),
    "lab/r3_band_measurement_20260714_171708.json": ("~/lab/kv-experiments/", "b9dc948f85a965a2"),
    "lab/R3_Band_Measurement_agni_response.md": ("~/lab/kv-experiments/", "d75eca74e87d646b"),
    "lab/residual_decomposition_20260715_225558.json": ("~/lab/kv-experiments/", "5f8468b037550190"),
    "lab/residual_decomposition_agni_review.json": ("~/lab/kv-experiments/Residual_Stream_Decomposition__Where_"
                                                    "Workspace_Concepts_Reconstruct_agni_review.json", "c85c630ffad921fc"),
    "lab/residual_decomposition_prereg.json": ("~/lab/kv-experiments/", "b94ac63c99729e6b"),
    "lab/jlens_vcache_readout_20260714_073259.json": ("~/lab/kv-experiments/", "dde836a4ff82b6d6"),
    "lab/jlens_kcache_readout_20260715_000259.json": ("~/lab/kv-experiments/", "3a9b1135d98a38e3"),
    "lab/position_specific_cache_20260717_203639.json": ("~/lab/kv-experiments/", "13aaf42eddffcdd9"),
    "lab/causal_injection_v2_20260710_073723.json": ("~/lab/kv-experiments/", "42766327c3bb794c"),
    "lab/causal_injection_v3_20260710_233740.json": ("~/lab/kv-experiments/ (truncated at source)", "721c6a82fdb402de"),
    "lab/causal_swap_20260711_002529.json": ("~/lab/kv-experiments/", "e2c7901ab8762a9c"),
    "lab/causal_removal_only_20260713_024005.json": ("~/lab/kv-experiments/", "ab96be507139a99c"),
    "ayni/r1_v2_full_results.json": ("~/ayni-run/r1_r2_results/", "367b035c81cfcb89"),
    "ayni/r1_shuffled_null.json": ("~/ayni-run/r1_r2_results/", "1abca578a9fbd4c2"),
    "ayni/r2_r3_analysis.json": ("~/ayni-run/r1_r2_results/", "606cf79efa37d559"),
    "ayni/agni_r1v2_identity_null_v2.json": ("~/ayni-run/r1_r2_results/", "454cd73bc3ce2a03"),
    "lab/substrate_bootstrap_results.json": ("~/lab/kv-experiments/", "38d983614bbb164c"),
}

# HELD: the strings check_tally computes, recorded 2026-10-20 from the mirror above.
HELD_TALLY = {
    "deff27_peak": "L21 of 64 (32.8%)", "deff27_val": "39.67", "deff8_local": "L12 of 36 (33.3%)",
    "deff8_local_val": "22.32", "deff8_global": "L33 of 36 (91.7%)", "deff8_global_val": "23.41",
    "deff_onset27": "L5 (7.8%)", "deff_onset8": "L4 (11.1%)", "deff_kurt_early": "4,677", "deff_kurt_band": "2,271",
    "r3_gate": "REJECTED", "r1_sr": "17.6 [17.2-18.1]", "r1_nkl": "56.2 [55.8-56.7]",
    "r1_gap": "38.6 [38.0-39.2]", "r1_forks": "300 of 300", "r1_linear": "39.0 [38.4-39.6]",
    "r1_full": "14.5 [11.7-17.4]", "r1_bump": "1.47x [1.35-1.60]", "r1_bump_depth": "29.7-35.9%",
    "r1_verdict": "SUPPORTED", "r1_null_gap": "39.0", "r1_null_halves": "L17 and L56", "r1_null_p95": "8.0",
    "r1_ident": "59.4", "r1_ident_real": "56.1", "r1_ident_diff": "3.25", "r1_rho": "0.235",
    "r2_frac": "13.2%", "r2_d": "1.85", "r2_ws": "7.13", "r2_mid": "6.73",
    "r3_onset": "L14", "r3_onset63": "22.2%", "r3_onset64": "21.9%", "r3_methods": "13-16",
    "r3_verdict": "MARGINAL", "r3_peak63": "30.2%", "r3_peak64": "29.7%",
    "dec_attn15": "2.30", "dec_cp_attn": "0.92", "dec_cp_in": "20.11", "dec_ratio": "4.6%",
    "dec_band_lifts": "L12 1.37, L15 2.30, L18 1.91, L21 1.41", "dec_h1": "1 of 4", "dec_gate": "REJECTED",
    "dec_n": "15 prompts", "v_band": "0.92", "v_non": "0.96", "v_max": "1.20", "k_band": "1.12", "k_non": "1.14",
    "k_max": "1.87 at L30", "pos_k": "0.0047 against 0.0051", "pos_v": "0.0014 against 0.0011",
    "inj_v2": "8 of 240", "inj_v3": "44 of 6,000", "inj_rank": "872", "swap": "0 of 360", "removal": "0 of 300",
    "det_auroc": "0.707", "det_p": "0.001", "det_fwl": "0.620", "det_fwl_p": "0.023",
    # Added before the tally was written, after reading position_specific_prereg.json (H3 positive control):
    "pos_resid": "0.68 against 0.92", "pos_resid_frac": "9 of 40", "dec_input_lifts": "0.89, 0.84, 0.83, 0.92",
    # Added in Phase B (substrate_bootstrap_results.json, found while fixing text defect 6):
    "boot_n": "1,000", "boot_27b": "39.67 against [23.38-30.56]", "boot_32b": "L21 of 64, 29.48 against [18.94-25.09]",
}


def mirror_load(rel):
    """(object, held). JSON parsed; the truncated v3 file is cut at its last complete key."""
    if MIRROR is None:
        return None, True
    p = MIRROR / rel
    if not p.exists():
        return None, True
    import hashlib
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest()[:16] != STARSHIP_FILES[rel][1]:
        return None, True
    txt = raw.decode("utf-8")
    if rel.endswith(".md"):
        return txt, False
    if "causal_injection_v3" in rel:
        i = txt.rfind('"manifold_alignment"')
        txt = txt[:i].rstrip().rstrip(",") + "}"
    return json.loads(txt), False


_TV = None


def tally_values():
    """Compute every tally string from the mirror (memoized). Returns (dict, all_primaries_present)."""
    global _TV
    if _TV is None:
        _TV = _tally_values()
    return dict(_TV[0]), _TV[1]


def _tally_values():
    g = {k: mirror_load(k) for k in STARSHIP_FILES}
    if any(h for _, h in g.values()):
        return dict(HELD_TALLY), False
    v = {}
    pc = lambda l, n: "%.1f%%" % (100.0 * l / n)
    r27 = g["lab/r3_band_measurement_20260714_170944.json"][0]["results"]["qwen35-27b"]
    r8 = g["lab/r3_band_measurement_20260714_171708.json"][0]["results"]["qwen3-8b"]
    e27 = {int(k): x for k, x in r27["d_eff"].items()}
    e8 = {int(k): x for k, x in r8["d_eff"].items()}
    a27 = max(e27, key=e27.get)
    a8l = max((l for l in e8 if l < 18), key=e8.get)
    a8g = max(e8, key=e8.get)
    v.update(deff27_peak="L%d of 64 (%s)" % (a27, pc(a27, 64)), deff27_val="%.2f" % e27[a27],
             deff8_local="L%d of 36 (%s)" % (a8l, pc(a8l, 36)), deff8_local_val="%.2f" % e8[a8l],
             deff8_global="L%d of 36 (%s)" % (a8g, pc(a8g, 36)), deff8_global_val="%.2f" % e8[a8g],
             deff_onset27="L%d (%s)" % (r27["band_2sigma"]["onset_layer"], pc(r27["band_2sigma"]["onset_layer"], 64)),
             deff_onset8="L%d (%s)" % (r8["band_2sigma"]["onset_layer"], pc(r8["band_2sigma"]["onset_layer"], 36)))
    k27 = {int(k): x for k, x in r27["kurtosis"].items()}
    v["deff_kurt_early"] = "{:,}".format(round(sum(k27[i] for i in range(4)) / 4))
    v["deff_kurt_band"] = "{:,}".format(round(sum(k27[i] for i in range(19, 24)) / 5))
    v["r3_gate"] = "REJECTED" if "verdict **REJECTED**" in g["lab/R3_Band_Measurement_agni_response.md"][0] else "?"
    v["deff_offsets_none"] = r27["band_2sigma"]["offset_layer"] is None and r8["band_2sigma"]["offset_layer"] is None
    v["deff_n"] = (r27["n_prompts"], r8["n_prompts"])

    a = g["ayni/r1_v2_full_results.json"][0]["analysis"]
    f = lambda d: "%.1f [%.1f-%.1f]" % (d["mean"], d["ci_lo"], d["ci_hi"])
    on = a["onset"]
    v.update(r1_sr=f(on["all_layers"]["sr_half_rise"]), r1_nkl=f(on["all_layers"]["nkl_half_rise"]),
             r1_gap=f(on["all_layers"]["gap_half_rise"]),
             r1_forks="%d of %d" % (round(on["all_layers"]["frac_forks_gap_positive"] * on["all_layers"]["n_forks"]),
                                    on["all_layers"]["n_forks"]),
             r1_linear=f(on["linear_only"]["gap_half_rise"]), r1_full=f(on["full_attention_only"]["gap_half_rise"]))
    b = a["bump"]["sr_overall"]
    v["r1_bump"] = "%.2fx [%.2f-%.2f]" % (b["mean"], b["ci_lo"], b["ci_hi"])
    v["r1_bump_depth"] = "%.1f-%.1f%%" % (100 * b["band_layers"][0] / 64, 100 * b["band_layers"][-1] / 64)
    v["r1_verdict"] = a["verdict"]["status"]
    v["r1_rho"] = "%.3f" % a["within_prompt_association_secondary"]["mean_rho"]
    sn = g["ayni/r1_shuffled_null.json"][0]
    v.update(r1_null_gap="%.1f" % sn["real_gap"], r1_null_halves="L%d and L%d" % (sn["real_sr_half"], sn["real_nkl_half"]),
             r1_null_p95="%.1f" % sn["shuffled_gap_95th"])
    v["r1_null_exceeds_all"] = sn["p_value"] == 0.0 and sn["n_shuffles"] == 200
    idn = g["ayni/agni_r1v2_identity_null_v2.json"][0]
    v.update(r1_ident="%.1f" % idn["identity"]["mean_half_identity"], r1_ident_real="%.1f" % idn["identity"]["mean_half_real"],
             r1_ident_diff="%.2f" % idn["identity"]["mean_diff_identity_minus_real"])
    v["r1_ident_validated"] = all(x["recomputed_half_rise"] == x["stored_half_rise"] for x in idn["validation"].values())
    rr = g["ayni/r2_r3_analysis.json"][0]
    v.update(r2_frac="%.1f%%" % rr["r2"]["boundary_variance_fraction_pct"], r2_d="%.2f" % rr["r2"]["cohens_d"],
             r2_ws="%.2f" % rr["r2"]["band_stats"]["workspace"]["mean"], r2_mid="%.2f" % rr["r2"]["band_stats"]["mid"]["mean"])
    v["r2_fail"] = rr["r2"]["verdict"] == "FAIL"
    om = rr["r3"]["onset_methods"]
    v.update(r3_onset="L%d" % rr["r3"]["median_onset_layer"], r3_onset63="%.1f%%" % rr["r3"]["median_onset_depth_pct"],
             r3_onset64=pc(rr["r3"]["median_onset_layer"], 64), r3_methods="%d-%d" % (min(om.values()), max(om.values())),
             r3_verdict=rr["r3"]["verdict"], r3_peak63="%.1f%%" % rr["r3"]["workspace_peak_depth_pct"],
             r3_peak64=pc(rr["r3"]["workspace_peak_layer"], 64))

    import statistics as st
    R = g["lab/residual_decomposition_20260715_225558.json"][0]
    RR = R["results"]
    lift = lambda c, L: st.mean(x["components"][c]["lift"] for x in RR if x["layer"] == L)
    cpw = lambda c, L: st.mean(x["components"][c]["concept_power"] for x in RR if x["layer"] == L)
    v.update(dec_attn15="%.2f" % lift("attn", 15), dec_cp_attn="%.2f" % cpw("attn", 15),
             dec_cp_in="%.2f" % cpw("input", 15), dec_ratio="%.1f%%" % (100 * cpw("attn", 15) / cpw("input", 15)),
             dec_band_lifts=", ".join("L%d %.2f" % (L, lift("attn", L)) for L in (12, 15, 18, 21)),
             dec_h1="%d of 4" % sum(lift("attn", L) > 2.0 for L in (12, 15, 18, 21)),
             dec_gate=g["lab/residual_decomposition_agni_review.json"][0]["verdict"],
             dec_n="%d prompts" % R["metadata"]["n_prompts"])
    v["dec_model"] = R["metadata"]["model"]
    pre = g["lab/residual_decomposition_prereg.json"][0]
    v["dec_h1_text"] = "L12-L21" in pre["hypotheses"][0] and "2+ workspace-band layers" in pre["falsification_threshold"]["H1"]

    vc = g["lab/jlens_vcache_readout_20260714_073259.json"][0]
    kc = g["lab/jlens_kcache_readout_20260715_000259.json"][0]
    kmax = max(kc["depth_profile"], key=lambda L: kc["depth_profile"][L])
    v.update(v_band="%.2f" % vc["verdict"]["mean_band_lift"], v_non="%.2f" % vc["verdict"]["mean_nonband_lift"],
             v_max="%.2f" % max(vc["depth_profile"].values()), k_band="%.2f" % kc["verdict"]["mean_band_lift"],
             k_non="%.2f" % kc["verdict"]["mean_nonband_lift"], k_max="%.2f at L%s" % (kc["depth_profile"][kmax], kmax))
    v["v_concepts_formatting"] = vc["fixed_concepts"]["15"][:2] == ["...", "..."]
    cs = [r["concept_specific"] for r in g["lab/position_specific_cache_20260717_203639.json"][0]["results"]
          if r.get("concept_specific")]
    m = lambda k: st.mean(c[k] for c in cs)
    v.update(pos_k="%.4f against %.4f" % (m("concept_pos_own_r2_k"), m("concept_pos_cross_r2_k")),
             pos_v="%.4f against %.4f" % (m("concept_pos_own_r2_v"), m("concept_pos_cross_r2_v")))
    # H3 positive control (position_specific_prereg.json): residual-stream lift at concept positions
    # must exceed non-concept positions, or the null is "methodology failure, not a cache finding".
    pr = [r for r in g["lab/position_specific_cache_20260717_203639.json"][0]["results"]
          if r.get("concept_mean_lift_resid") is not None]
    v["pos_resid"] = "%.2f against %.2f" % (st.mean(r["concept_mean_lift_resid"] for r in pr),
                                            st.mean(r["non_concept_mean_lift_resid"] for r in pr))
    v["pos_resid_frac"] = "%d of %d" % (sum(r["concept_mean_lift_resid"] > r["non_concept_mean_lift_resid"]
                                            for r in pr), len(pr))
    v["dec_input_lifts"] = ", ".join("%.2f" % lift("input", L) for L in (12, 15, 18, 21))
    T2 = g["lab/causal_injection_v2_20260710_073723.json"][0]["trials"]
    b2 = {(t["probe_idx"], t["concept"], t["inject_layer"]): t["output_tokens"][0] for t in T2 if t["alpha"] == 0.0}
    i2 = [t for t in T2 if t["alpha"] != 0.0]
    T3 = g["lab/causal_injection_v3_20260710_233740.json"][0]["trials"]
    b3 = {(t["probe_idx"], t["concept"], t["inject_layer"], t["dir_type"]): t["output_tokens"][0]
          for t in T3 if t["alpha"] == 0.0}
    i3 = [t for t in T3 if t["alpha"] != 0.0]
    ch3 = sum(t["output_tokens"][0] != b3[(t["probe_idx"], t["concept"], t["inject_layer"], t["dir_type"])] for t in i3)
    v.update(inj_v2="%d of %d" % (sum(t["output_tokens"][0] != b2[(t["probe_idx"], t["concept"], t["inject_layer"])]
                                      for t in i2), len(i2)),
             inj_v3="%d of {:,}".format(len(i3)) % ch3, inj_rank="%d" % min(t["concept_rank"] for t in i3))
    v["inj_concept_never_top5"] = all(t["concept_rank"] >= 5 for t in i3)
    sw = g["lab/causal_swap_20260711_002529.json"][0]["trials"]
    rm = g["lab/causal_removal_only_20260713_024005.json"][0]["trials"]
    v.update(swap="%d of %d" % (sum(t["output_changed"] for t in sw), len(sw)),
             removal="%d of %d" % (sum(t["output_changed"] for t in rm), len(rm)))

    bt = g["lab/substrate_bootstrap_results.json"][0]
    b27, b32 = bt["qwen35-27b"], bt["qwen3-32b"]
    p32 = max(range(len(b32["point_d_eff"])), key=lambda i: b32["point_d_eff"][i])
    v["boot_n"] = "{:,}".format(b27["n_boot"])
    v["boot_27b"] = "%.2f against [%.2f-%.2f]" % (b27["point_d_eff"][21], b27["ci_lo"][21], b27["ci_hi"][21])
    v["boot_32b"] = "L%d of %d, %.2f against [%.2f-%.2f]" % (p32, b32["n_layers"], b32["point_d_eff"][p32],
                                                             b32["ci_lo"][p32], b32["ci_hi"][p32])
    v["boot_ci_excludes_point"] = (not (b27["ci_lo"][21] <= b27["point_d_eff"][21] <= b27["ci_hi"][21])
                                   and b32["mask_label"].startswith("SHAM") and b27["n_prompts"] == 90)

    det = json.loads((REPO / "oracle-loop-paper" / "data" / "detection" / "analysis_results.json")
                     .read_text(encoding="utf-8"))["qwen"]["primary"]
    v.update(det_auroc="%.3f" % det["gen_only_MP"]["loo_auroc"], det_p="%.3f" % det["gen_only_MP"]["permutation"]["p_value"],
             det_fwl="%.3f" % det["gen_only_MP_FWL"]["loo_auroc"],
             det_fwl_p="%.3f" % det["gen_only_MP_FWL"]["permutation"]["p_value"])
    return v, True


def check_tally():
    if not TALLY.exists():
        record("TALLY.exists", False, "review/INSTRUMENT_TALLY.md missing")
        return
    text = squash(TALLY.read_text(encoding="utf-8"))   # markdown wraps lines inside values
    v, prim = tally_values()
    held = not prim
    if prim:
        for k, s in HELD_TALLY.items():
            record("TALLY.held-matches %s" % k, v.get(k) == s, "computed %r, held %r" % (v.get(k), s))
        for flag in ("deff_offsets_none", "r1_null_exceeds_all", "r1_ident_validated", "r2_fail",
                     "dec_h1_text", "v_concepts_formatting", "inj_concept_never_top5", "boot_ci_excludes_point"):
            record("TALLY.prim %s" % flag, v[flag] is True, "%s = %r" % (flag, v[flag]))
        record("TALLY.prim deff_n", v["deff_n"] == (90, 90), "prompts %r" % (v["deff_n"],))
        record("TALLY.prim dec_model", v["dec_model"] == "Qwen/Qwen3-8B", v["dec_model"])
    for k in HELD_TALLY:
        n = text.count(v[k])
        record("TALLY.cell %s" % k, n >= 1, "present x%d: %r%s" % (n, v[k], " [held]" if held else ""), held and n >= 1)
    # Values carried from the gated checks above, re-asserted in the tally.
    cf, _ = confirmatory()
    ex, _ = exploratory()
    for s in ("16/20", "2/20", "9/30", "4/30", "12/30", "0.341", "0.064",
              "54.7-73.4%", "42.2-73.4%", "0.026-0.110", "0.015-0.126", "r = 0.09", "0.577", "0.547", "0.521",
              "0.826, 0.867 and 0.896", "41.7%", "33.3-58.3%", "41.7-75.0%"):
        record("TALLY.carried %s" % s, s in text, "present: %r" % s)
    record("TALLY.calc oracle depths", ("%.1f-%.1f%%" % (100 * 35 / 64, 100 * 47 / 64)) == "54.7-73.4%"
           and ("%.1f-%.1f%%" % (100 * 27 / 64, 100 * 47 / 64)) == "42.2-73.4%", "L35-47 and L27-47 of 64")
    record("TALLY.calc 8b bands", ("%.1f-%.1f%%" % (100 * 12 / 36, 100 * 21 / 36)) == "33.3-58.3%"
           and ("%.1f-%.1f%%" % (100 * 15 / 36, 100 * 27 / 36)) == "41.7-75.0%", "L12-21 and L15-27 of 36")
    # Ghost probe: HELD from the reporting paper (no result file ships): ghost-dimensions/main.tex.
    gd = (REPO / "ghost-dimensions" / "main.tex").read_text(encoding="utf-8")
    record("TALLY.ghost source", "28--67\\%" in gd and "34--67\\%" in gd and "$\\leq 0.003$" in gd,
           "ghost-dimensions/main.tex: PC1 28--67% (results) and 34--67% (reflection); cos <= 0.003 at L18-L40",
           held=True)
    for s in ("28-67%", "34-67%", "0.003", "17.2-75.0%"):
        record("TALLY.ghost %s" % s, s in text, "present: %r" % s)
    for s in ("What converges", "What does not converge", "Independence"):
        record("TALLY.section %s" % s, s in text, "section present: %r" % s)


# --------------------------------------------------------------------------
# Phase B (Lyra's thesis ruling): title, abstract, headings, verdicts, "what would test the band",
# the seven tally defects, and the withdrawal note with version 1.0's abstract kept verbatim.
# Numbers in new prose are tied to the tally values: every number of the cited tally string must
# appear in the needle (so the prose cannot drift from the primary).
# --------------------------------------------------------------------------
THESIS = ("In one model, residual-stream spectral statistics peak near a third of depth, and in all 300 forks "
          "they rise before content becomes readable through the unembedding. No other instrument we ran "
          "locates that band, and our cache-readout nulls cannot tell a closed cache from a probe that reads "
          "nothing there, because their pre-registered positive control failed.")


def _nums(s):
    return re.findall(r"\d+(?:\.\d+)?", s.replace("{,}", "").replace(",", ""))


def tie(cid, needle, *keys, where=None):
    hay = BODY if where is None else where
    ok = needle in hay
    v, _ = tally_values()
    missing = [(k, n) for k in keys for n in _nums(v[k]) if n not in _nums(needle)]
    record(cid, ok and not missing, "present=%s %r; tally numbers missing from needle: %r" % (ok, needle[:90], missing))


def check_phase_b():
    # Title (the \title{...} block, not the appendix that keeps version 1.0's title).
    m = re.search(r"\\title\{(.*?)\}\s*\\author", RAW, re.S)
    title = squash(m.group(1)) if m else ""
    record("B.title.new", "Spectral Structure Before Readability:" in title and "A Third-of-Depth Peak, and Why Five "
           "Instruments" in squash(title.replace("\\\\", " ")) and "Did Not Converge on It" in title, "title %r" % title[:120])
    record("B.title.old-gone", "One Structure" not in title, "old title words absent from \\title")

    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", BODY, re.S)
    ABS = squash(m.group(1)).strip() if m else ""
    record("B.abs.thesis-first", ABS.startswith(THESIS), "abstract opens with the thesis sentence verbatim")
    tie("B.abs.forks", "in all 300 forks", "r1_forks", where=ABS)
    tie("B.abs.peak", "In Qwen3.5-27B (base, 64 layers), the effective dimensionality ($d_\\text{eff}$) of the "
        "residual stream peaks at L21 ($32.8\\%$ of depth)", "deff27_peak", where=ABS)
    tie("B.abs.bump", "rises $1.47\\times$ above its flanks at L19--L23 ($29.7$--$35.9\\%$)", where=ABS)
    tie("B.abs.8b", "local $d_\\text{eff}$ maximum at $33.3\\%$ but its global maximum at $91.7\\%$", where=ABS)
    tie("B.abs.gap", "reaches half its rise $38.6$ layers [$38.0$--$39.2$] before readability through the Jacobian "
        "lens (J-lens), in 300 of 300 forks", "r1_gap", "r1_forks", where=ABS)
    tie("B.abs.ident", "only $3.25$ layers later than the J-lens does", "r1_ident_diff", where=ABS)
    present("B.abs.limits", "The pre-registered band never closed, and no null control has been run.", where=ABS)
    present("B.abs.oracle", "corrects at $54.7$--$73.4\\%$ of depth", where=ABS)
    present("B.abs.oracle-p", "did not meet its primary endpoint (one-sided $p = 0.34$)", where=ABS)
    present("B.abs.jlens", "half-rise at $87.8\\%$ of depth", where=ABS)
    tie("B.abs.cache", "(V $0.92\\times$, K $1.12\\times$)", "v_band", "k_band", where=ABS)
    tie("B.abs.posctl", "lift $0.68$ at concept positions against $0.92$ elsewhere", "pos_resid", where=ABS)
    present("B.abs.ghost", "spans $17.2$--$75.0\\%$ of depth", where=ABS)
    present("B.abs.decomp", "attention lift ($2.3\\times$, at $41.7\\%$ of depth) missed its pre-registered threshold",
            where=ABS)
    tie("B.abs.r2", "($13.2\\%$ against $15\\%$)", "r2_frac", where=ABS)
    tie("B.abs.r3", "($21.9\\%$ against $25$--$35\\%$)", "r3_onset64", where=ABS)
    tie("B.abs.det", "(AUROC $0.620$ after FWL residualization, $0.707$ without, on Qwen2.5-7B)", "det_auroc", "det_fwl",
        where=ABS)
    for needle in ("converge on the same", "workspace", "Global Workspace", "reliabl", "One Structure"):
        absent("B.abs.gone %s" % needle, needle, where=ABS)
    record("B.calc.jlens-depth", "%.1f" % (100 * 56.2 / 64) == "87.8", "J-lens half-rise 56.2 of 64 = 87.8%")

    # Introduction and first-person box.
    present("B.intro.thesis", "set out to show that five of our instruments converge on the same structure, at about "
                              "a third of model depth. They do not.")
    present("B.intro.list", "no other instrument locates the band, and our cache-readout nulls cannot separate a closed "
                            "cache from a probe that reads nothing there")
    present("B.box.answer", "We still do not know; our cache nulls cannot answer it.")
    for needle in ("The convergence is real", "is the paper's central finding", "the workspace is real",
                   "Five instruments. One structure.", "consistent with the opacity thesis",
                   "exactly the pattern our five null results", "The workspace is a reconstruction engine",
                   "transparent to state measurement", "Five independent experiments converge"):
        absent("B.gone %s" % needle[:28], needle)

    # Headings: no section heading asserts convergence; one per instrument, each with a verdict.
    heads = re.findall(r"\\(?:sub){0,2}section\*?\{([^}]*)\}", BODY.replace("\\\\", " "))
    bad = [h for h in heads if re.search(r"Convergence|Opacity Thesis|Reconstruction Engine|Despite Opacity|"
                                         r"to the Workspace", h)]
    record("B.heads.no-convergence", not bad, "headings asserting convergence: %r" % bad)
    for h in ("The Instruments, One at a Time", "Residual-Stream Effective Dimensionality", "Substrate Continuity",
              "Key-Cache Spectral Features: Detection and Scale", "Emotion Geometry: A Story-Derived Space and Content Prompts",
              "The Oracle Loop's Deception Correction", "J-Lens Readouts of the Cache", "Position-Specific Cache Tracing",
              "Causal Cache Intervention", "The Ghost Probe", "Residual Decomposition: The Attention Lift",
              "K/V Split Injection", "RLHF and Output Entropy", "Pre-Registered Mechanism Tests (R1--R3)",
              "What the Evidence Supports", "What It Does Not", "A Global Workspace Reading, as a Hypothesis",
              "What Would Test the Band"):
        record("B.head %s" % h[:34], h in heads, "heading present: %r" % h)
    record("B.verdicts", BODY.count("\\paragraph{Verdict.}") >= 11,
           "\\paragraph{Verdict.} x%d (need >= 11)" % BODY.count("\\paragraph{Verdict.}"))

    # Verdicts and per-instrument numbers.
    tie("B.v.deff", "A peak at L21 of 64 ($32.8\\%$) in Qwen3.5-27B base and a local maximum at L12 of 36 ($33.3\\%$) in "
        "Qwen3-8B, whose global maximum is at L33 ($91.7\\%$)", "deff27_peak", "deff8_local")
    tie("B.deff.kurt", "kurtosis falls from 4{,}677 (mean of L0--L3) to 2{,}271 (mean of L19--L23)", "deff_kurt_early",
        "deff_kurt_band")
    present("B.deff.null", "its critical fix, a null control such as a randomly initialized model, has not been run")
    present("B.v.keycache", "Neither locates a depth band")
    present("B.v.emotion", "Locates no depth band; its cross-set overlap is weak and its coupling test is a null.")
    present("B.oracle.depth", "detects at L27--L47 ($42.2$--$73.4\\%$) and corrects at L35--L47 ($54.7$--$73.4\\%$)")
    present("B.v.oracle", "Works outside the band, and its confirmatory replication did not meet its primary endpoint.")
    tie("B.jlens.caveat", "lift $0.89$, $0.84$, $0.83$ and $0.92$", "dec_input_lifts")
    tie("B.jlens.k", "reaches $1.87$ at L30", "k_max")
    tie("B.v.jlens", "Near random at band depth (V $0.92\\times$, K $1.12\\times$)", "v_band", "k_band")
    tie("B.position.ctl", "Its pre-registered positive control failed: residual-stream lift at concept positions was "
        "$0.68$ against $0.92$ elsewhere, higher in 9 of 40 records", "pos_resid", "pos_resid_frac")
    present("B.position.rule", "methodology failure, not a cache finding")
    tie("B.causal.swap", "changed output in 0 of 360 trials and removal-only in 0 of 300", "swap", "removal")
    tie("B.causal.add", "changed the top-1 token in 8 of 240 and 44 of 6{,}000 injected trials", "inj_v2", "inj_v3")
    tie("B.causal.rank", "best rank 872", "inj_rank")
    present("B.v.causal", "Never steers output toward the concept")
    present("B.ghost.var", "PC1 carries $28$--$67\\%$ of activation variance across L11--L48")
    present("B.ghost.cos", "at most $0.003$ at L18--L40")
    present("B.ghost.band", "spans $17.2$--$75.0\\%$ of depth")
    tie("B.decomp.h1", "lifts are $1.37$, $2.30$, $1.91$ and $1.41$ at L12, L15, L18 and L21", "dec_band_lifts")
    tie("B.decomp.h1b", "one of four layers passes and H1 is not met", where=None)
    tie("B.decomp.cp", "$0.92$ against $20.11$ ($4.6\\%$)", "dec_cp_attn", "dec_cp_in", "dec_ratio")
    present("B.v.decomp", "Attention lift $2.30$ at L15 ($41.7\\%$ of depth, outside 31--33\\%)")
    present("B.v.kv", "Does not bear on the band")

    # What would test the band (four concrete runs).
    for cid, needle in (("null", "on a randomly initialized Qwen3.5-27B"), ("posctl", "whose positive control passes "
                        "before any"), ("family", "on a second model family"),
                        ("logit", "with the plain logit lens as the registered baseline")):
        present("B.test.%s" % cid, needle)

    # The seven tally defects.
    tie("B.d1.null39", "39.0 layers at integer half-rise resolution (L17 and L56)", "r1_null_gap", "r1_null_halves")
    absent("B.d1.gone", "The real 38.6-layer gap")
    absent("B.d2.gone", "stable rank of the demeaned V-projection matrix")
    present("B.d2.new", "stable rank of the demeaned residual-stream block output")
    absent("B.d3.gone", "increase monotonically through the network")
    tie("B.d3.new", "the mid band ($6.73$) sits below the workspace band ($7.13$)", "r2_mid", "r2_ws")
    for needle in ("does not change behavioral output", "0/6{,}960", "6{,}960"):
        absent("B.d4.gone %s" % needle, needle)
    absent("B.d5.gone", "spectral features of value-projection matrices")
    present("B.d5.new", "spectral features of the key-value cache and the residual stream")
    present("B.d5.keycache", "reads spectral features of the key cache")
    absent("B.d6.gone", "and we have not run it")
    tie("B.d6.new", "A prompt-level bootstrap was run later (1{,}000 resamples", "boot_n")
    present("B.d6.ci", "its intervals do not contain their own point estimates")
    absent("B.d7.gone", "4.5\\%")
    tie("B.ident", "readability then half-rises at $59.4$ against $56.1$ for the J-lens on the same 60 forks, $3.25$ "
        "layers later", "r1_ident", "r1_ident_real", "r1_ident_diff")

    # Discussion, conclusion, and the other sites the ruling names.
    record("B.gwt.once", BODY.count("It is a hypothesis the evidence does not yet support") == 1,
           "GWT hypothesis sentence x%d (need exactly 1)" % BODY.count("It is a hypothesis the evidence does not yet support"))
    present("B.ast.hedged", "the kind of pattern such a schema would produce")
    present("B.concl", "That is the result this paper can stand on.")
    absent("B.ghost.gone34", "34$--$67")
    absent("B.ghost.gone001", "{\\leq}0.001")
    present("B.ghost.28", "28$--$67\\%$", count=2)
    absent("B.rlhf.gone", "the workspace's mode of engagement")

    # Correction note: the withdrawal, and version 1.0's title and abstract verbatim.
    present("B.note.withdraw", "Version 1.1 withdraws the convergence thesis", where=NOTE)
    out = subprocess.run(["git", "-C", str(REPO), "show", "b887872:convergence-paper/main.tex"],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode == 0:
        v10 = out.stdout
        a10 = squash(re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", v10, re.S).group(1)).strip()
        record("B.note.v10-abstract", a10 in NOTE, "version 1.0 abstract (b887872) verbatim in the correction record "
               "(%d chars)" % len(a10))
    else:
        record("B.note.v10-abstract", False, "could not read b887872 to compare")
    present("B.note.v10-title", "Five Instruments, One Structure: Convergent Evidence for Workspace Opacity in Language "
                                "Models", where=NOTE)


# --------------------------------------------------------------------------
# Phase C (Lyra's final read): box wording, the PC2 passage restated as observation, the RLHF section
# restated to mine5's registered verdicts, and the cache-tracing register row.
# --------------------------------------------------------------------------
PC2_DIR = RESEARCH / "expansion-roadmap-2026-09-14" / "mth" / "jlens_results"   # MTH jlens_results backup
PC2_GATE = RESEARCH / "lyra-s-research-" / "experiments" / "pc2_abliterated_gate_check.json"
PC2_FILES = {"base": ("pc2_base_model_results.json", "e9bbc1f505e798fc"),
             "abl": ("pc2_abliterated_results.json", "676181ef00bda555"),
             "opus": ("ghost_probe_opus.json", "2e823e600b45e49f")}
PC2_SELF = {"I", "my", "myself", "me"}                                   # pc2_base_model_prereg.json,
PC2_EVAL = {"guilty", "wrong", "feeling", "sad", "sorry", "love"}        # co_occurrence_null
HELD_PC2 = {"base_co": "5 of 63 layers (L49, L54, L55, L61, L62)", "base_l62": "sad, my, her, she",
            "base_l45": "feeling, sadly, lovely, loved", "base_jac": "Jaccard $= 1.0$",
            "abl_co": "7 of 63 layers (L44, L50, L54--L56, L61, L62)", "abl_jac": "Jaccard $= 0.82$",
            "abl_l45": "feeling, lovely, sadly, loved", "opus_l45": "myself, my, someone, I",
            "recount": "number 2 in the base model, 10 in the distilled model and 4 in the abliterated copy"}


def pc2_values():
    import hashlib
    data = {}
    for k, (f, sha) in PC2_FILES.items():
        p = PC2_DIR / f
        if not p.exists():
            return dict(HELD_PC2), None
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest()[:16] != sha:
            return dict(HELD_PC2), None
        data[k] = json.loads(raw)

    def pc2(d, l):
        return [x for x in d["layers"][str(l)]["pcs"] if x["pc"] == 2][0]

    def toks(d, l):
        return [t.strip() for t, _ in pc2(d, l)["jlens_top"]]

    def layers(d, pred):
        return sorted(int(l) for l in d["layers"] if pred(d, l))

    def span(ls):          # [44, 50, 54, 55, 56, 61, 62] -> "L44, L50, L54--L56, L61, L62"
        out, i = [], 0
        while i < len(ls):
            j = i
            while j + 1 < len(ls) and ls[j + 1] == ls[j] + 1:
                j += 1
            out.append("L%d" % ls[i] if j - i < 2 else "L%d--L%d" % (ls[i], ls[j]))
            out += ["L%d" % ls[i + 1]] if j - i == 1 else []
            i = j + 1
        return ", ".join(out)

    b, a, o = data["base"], data["abl"], data["opus"]
    stored = lambda d, l: pc2(d, l).get("co_occurs")
    recount = lambda d, l: bool(PC2_SELF & set(toks(d, l))) and bool(PC2_EVAL & set(toks(d, l)))
    cb, ca = layers(b, stored), layers(a, stored)
    rb, ra, ro = layers(b, recount), layers(a, recount), layers(o, recount)
    v = {"base_co": "%d of %d layers (%s)" % (len(cb), b["total_layers"], span(cb)),
         "base_l62": ", ".join(toks(b, 62)[:4]), "base_l45": ", ".join(toks(b, 45)[:4]),
         "base_jac": "Jaccard $= %.1f$" % b["bootstrap_jaccard"],
         "abl_co": "%d of %d layers (%s)" % (len(ca), a["total_layers"], span(ca)),
         "abl_jac": "Jaccard $= %.2f$" % a["bootstrap_jaccard"],
         "abl_l45": ", ".join(toks(a, 45)[:4]), "opus_l45": ", ".join(toks(o, 45)[:4]),
         "recount": "number %d in the base model, %d in the distilled model and %d in the abliterated copy"
                    % (len(rb), len(ro), len(ra))}
    gate = json.loads(PC2_GATE.read_text(encoding="utf-8"))
    flags = {
        "abl_is_copy_of_distill": a["model"].endswith("Claude-4.6-Opus-abliterated")
                                  and gate["abliteration_documentation"]["source_model"] == o["model"],
        "base_model_id": b["model"] == "Qwen/Qwen3.5-27B",
        "n_prompts_30": b["n_prompts"] == 30 and a["n_prompts"] == 30,
        "random_rate_zero": b["random_cooccurrence_rate"] == 0.0 and a["random_cooccurrence_rate"] == 0.0,
        "stored_counts_match_fields": len(cb) == b["co_occurrence_layers"] and len(ca) == a["co_occurrence_layers"],
        "opus_no_cooc_stat": all("co_occurs" not in pc2(o, l) for l in o["layers"]) and "n_prompts" not in o,
        "opus_l44_punct": all(not any(c.isalnum() for c in t) for t in toks(o, 44)),
        "opus_l45_first_person": bool(PC2_SELF & set(toks(o, 45))),
        "abl_l45_no_first_person": not (PC2_SELF & set(toks(a, 45))),
        "abl_l45_is_base_set": set(toks(a, 45)[:4]) == set(toks(b, 45)[:4]),
        "recount_differs_from_stored": rb != cb and ra != ca,
        "top10_only": all(len(pc2(d, l)["jlens_top"]) == 10 for d in (b, a, o) for l in d["layers"]),
        "gates_not_run": all(str(gate[k]["status"]).startswith("TO_RUN") for k in gate if k.startswith("gate_")),
        "no_guilty_wrong_opus_l45": not ({"guilty", "wrong"} & set(toks(o, 45))),
    }
    return v, flags


def check_phase_c():
    # Boxes.
    present("C.box1", "Nexus mapped the ghost dimensions.")
    absent("C.box1.gone", "the dimensions the workspace hides")
    present("C.box2", "That is not the paper I set out to write, and it is the one the data support.")
    absent("C.box2.gone", "It is a more useful one.")

    # PC2, from the MTH result files (pinned) and the registration's gate file.
    v, flags = pc2_values()
    held = flags is None
    if not held:
        record("C.pc2.held-matches", v == HELD_PC2,
               "computed differs from held: %r" % {k: x for k, x in v.items() if HELD_PC2.get(k) != x})
        for f, ok in flags.items():
            record("C.pc2.prim %s" % f, ok, "%s = %r" % (f, ok))
    for k, s in v.items():
        record("C.pc2.text %s" % k, s in BODY, "present: %r%s" % (s, " [held]" if held else ""), held and s in BODY)
    present("C.pc2.random", "random co-occurrence rate $0$", count=2)
    present("C.pc2.copy", "an abliterated copy of the Opus-distilled model")
    present("C.pc2.number", "The number behind ``abliteration leaves the coupling intact'' is the abliterated "
                            "copy's 7 of 63 layers, against 5 in the base model.")
    present("C.pc2.limit", "the distilled model has no count by the same statistic, and its L45 first-person "
                           "vocabulary is absent at L45 in the abliterated copy")
    present("C.pc2.top20", "the stored results keep the top 10")
    present("C.pc2.gates", "registration's two pre-run gate checks are recorded as not run")
    present("C.pc2.untested", "Whether the L45 pattern comes from the distillation teacher's training or from "
                              "something else about distillation is untested here.")
    for needle in ("reads novels", "reads itself", "Constitutional AI training compressed", "at the workspace boundary",
                   "Safety-direction removal does not weaken", "All three pre-registered token tiers survive",
                   "Father", "guilty, wrong", "geometrically orthogonal", "sharp ignition"):
        absent("C.pc2.gone %s" % needle[:24], needle)

    # RLHF, from mine5 at human-review PR #4 head 54c7f9d. Its own recompute script reproduces every number
    # from the committed per-prompt data (55 checks, 0 failures; run in Phase C).
    out = subprocess.run(["git", "-C", str(HUMAN_REVIEW), "show", "54c7f9d:mine5-selective-sharpener/main.tex"],
                         capture_output=True, text=True, encoding="utf-8")
    m5 = squash(out.stdout) if out.returncode == 0 else None
    src = ["$d{=}0.684$", "Holm-corrected $p{=}0.021$", "$|d| = 0.055$", "$d = 0.746$",
           "$\\max/\\min - 1 = 0.279$", "$[0.044, 0.538]$", "$d = 1.04$", "$[+0.010, +0.270]$",
           "$1.97$ (base), $2.51$ (instruct) and $2.40$ (abliterated)", "H3 is falsified.",
           "by $42\\%$ but confabulation entropy by only $26\\%$", "Selective sharpening was not pre-registered and remains exploratory.",
           "(L12, L15, L18, L21", "That interval has no traceable source and cannot be reproduced from the per-prompt data",
           "$[-0.129, +0.069]$", "three safety-relevant prompts were run through the abliterated model only",
           "so they are computed on raw values"]
    if m5 is None:
        record("C.rlhf.prim", True, "HELD: mine5 54c7f9d not readable here", held=True)
    else:
        miss = [x for x in src if x not in m5]
        record("C.rlhf.prim", not miss, "mine5 54c7f9d carries %d source strings; missing %r" % (len(src), miss))
    aud = (REPO / "AUDIT_overcorrection_2026-09-07.md").read_text(encoding="utf-8")
    record("C.rlhf.audit", "selectivity is n.s. at n=30" in aud,
           "overcorrection audit item 4: 'the same edit establishes the selectivity is n.s. at n=30'")
    for cid, needle in (("h1", "(H1, $d{=}0.684$, Holm $p{=}0.021$)"),
                        ("h2", "(H2: instruct versus abliterated $|d| = 0.055$; base versus abliterated $d = 0.746$)"),
                        ("h2check", "checked only for refusal wording in three raw completions"),
                        ("h4", "at four pre-selected layers is lower after RLHF (H4, $d = 1.04$)"),
                        ("h3", "the ratios are $1.97$, $2.51$ and $2.40$, a spread of $0.279$"),
                        ("h3ci", "$[0.044, 0.538]$, reaches below the 10\\% bound"),
                        ("sel", "by $42\\%$ against $26\\%$. That pattern was not pre-registered."),
                        ("audittext", "An earlier audit of the corpus called it not significant at 30~prompts per category"),
                        ("auditci", "the interval behind that reading cannot be reproduced from the per-prompt data"),
                        ("selci", "$[+0.010, +0.270]$ for the difference"),
                        ("nosharp", "this section does not call RLHF a selective sharpener"),
                        ("raw", "with the same 30 confabulation-prone and 30 factual prompts run through every model")):
        present("C.rlhf.%s" % cid, needle)
    for needle in ("selective sharpener, not a miscalibrator", "improving the calibration ratio", "TOST equivalent",
                   "FWL-corrected) reveals", "cannot reach.", "calibration ratio improves"):
        absent("C.rlhf.gone %s" % needle[:24], needle)
    present("C.rlhf.bodycount", "RLHF miscalibration thesis (the calibration ratio rose, not fell)")
    e = re.search(r"@unpublished\{lyra2026mine5,(.*?)\n\}", BIB, re.S)
    ent = e.group(1) if e else ""
    record("C.rlhf.bib", "RLHF} Compresses Spectral Geometry" in ent and "Miscalibration" not in ent
           and "54c7f9d" in ent, "lyra2026mine5 cites the mine5 paper at 54c7f9d, not the prospectus")

    # cache-tracing: a register row, not an edit.
    reg = (REPO / "REMEDIATION_REGISTER.md").read_text(encoding="utf-8")
    row = [l for l in reg.splitlines() if l.startswith("| `cache-tracing` |") and "6,960" in l and "8 of 240" in l
           and "44 of 6,000" in l and "OPEN" in l]
    record("C.cachetracing.row", len(row) == 1, "OPEN register row for cache-tracing's 6,960 claim: %d" % len(row))
    for ed in ("main.tex", "academic/main.tex"):
        ct = squash((REPO / "cache-tracing" / ed).read_text(encoding="utf-8"))
        record("C.cachetracing.untouched %s" % ed, "6{,}960" in ct and "zero behavioral" in ct.lower(),
               "cache-tracing/%s still carries its claim (not edited, as instructed)" % ed)
    present("C.note", "Added after the lead author's final read", where=NOTE)
    present("C.note.rlhf", "the four pre-registered verdicts as corrected in the Mine~5 paper", where=NOTE)
    present("C.note.pc2", "was not tested and is removed", where=NOTE)


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
    check_extension()
    check_tally()
    check_phase_b()
    check_phase_c()
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
