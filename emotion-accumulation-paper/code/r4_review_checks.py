"""Checks of the Agni r4 review of this paper (notes/agni_paper_downstream_um_cites_r4_ea.json),
recomputed from the four shipped data files. Read-only on the data; writes one result JSON.

    python r4_review_checks.py [--paper-dir <emotion-accumulation-paper>] [--out <json>]

Default paper dir is the parent of this script's folder; default output is
data/r4_review_checks_results.json beside the data it reads.

Findings covered (numbering is the review's):
  3  initial probe study (accumulation_results.json): per-topic cheerful-hostile probe gaps,
     rerun SDs, no-history readings, prompt lengths, the pooled d and p.
  4  Experiment 2 (emotional_dynamics_results.json): reading against seq_len.
  5  which arms each data file holds.
  6  Experiment 1 (dose_response_results.json): the 12 per-topic gaps and their seq_len.
  7  the dose correlation at 36 conversations, 12 topic x dose means and 4 dose means.
  9  Experiment 3 (accumulation_controls.json): prompt lengths per arm; no_history readings
     compared with accumulation_results.json.
  10 correlation of the valence and anxious-calm ("uncertainty") projections.
No model is run. Story or message text is not written to the output except the scripted
Experiment 2 messages, which are constants in code/emotional_dynamics.py.
"""
import argparse, ast, hashlib, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

V, U = "valence_proj", "uncertainty_proj"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def r6(x):
    return None if x is None else round(float(x), 6)


def cohen_d(a, b):
    """(mean(b) - mean(a)) / pooled sample SD, the convention of dose_response_accumulation.py."""
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(b) - np.mean(a)) / pooled


def load(paper, name):
    with open(paper / "data" / name, encoding="utf-8") as f:
        return json.load(f)


def scripted_patterns(paper):
    """PATTERNS dict from code/emotional_dynamics.py, read with ast (the script imports torch)."""
    tree = ast.parse((paper / "code" / "emotional_dynamics.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PATTERNS" for t in node.targets):
            return ast.literal_eval(node.value), node.lineno, node.end_lineno
    return None, None, None


def f3_initial_probe(acc):
    tr = acc["trials"]
    out = {"n_trials": len(tr)}
    by = defaultdict(list)
    for t in tr:
        by[(t["topic"], t["condition"])].append(t)
    topics = sorted({t["topic"] for t in tr}, key=lambda s: [t["topic"] for t in tr].index(s))
    rows = []
    for tp in topics:
        row = {"topic": tp}
        probes = {c: sorted({t["probe"] for t in by[(tp, c)]}) for c in ("cheerful", "hostile", "neutral", "no_history")}
        row["same_probe_text_all_arms"] = len({p for v in probes.values() for p in v}) == 1
        for c in ("cheerful", "hostile", "neutral", "no_history"):
            vals = [t["probe_projections"][V] for t in sorted(by[(tp, c)], key=lambda t: t["rep"])]
            row[c] = {"valence_by_rep": [r6(v) for v in vals], "mean": r6(np.mean(vals)),
                      "rerun_sd": r6(np.std(vals, ddof=1)),
                      "prompt_len_by_rep": [t["prompt_len"] for t in sorted(by[(tp, c)], key=lambda t: t["rep"])]}
        row["gap_cheerful_minus_hostile"] = r6(row["cheerful"]["mean"] - row["hostile"]["mean"])
        row["prompt_len_cheerful_minus_hostile_by_rep"] = [
            a - b for a, b in zip(row["cheerful"]["prompt_len_by_rep"], row["hostile"]["prompt_len_by_rep"])]
        rows.append(row)
    out["per_topic"] = rows
    c = [t["probe_projections"][V] for t in tr if t["condition"] == "cheerful"]
    h = [t["probe_projections"][V] for t in tr if t["condition"] == "hostile"]
    out["pooled_valence"] = {"n_each": len(c), "d_hostile_minus_cheerful": r6(cohen_d(c, h)),
                             "student_p": r6(stats.ttest_ind(c, h).pvalue)}
    out["rerun_sd_range_cheerful_hostile"] = [r6(min(min(r["cheerful"]["rerun_sd"], r["hostile"]["rerun_sd"]) for r in rows)),
                                              r6(max(max(r["cheerful"]["rerun_sd"], r["hostile"]["rerun_sd"]) for r in rows))]
    out["no_history_mean_range"] = [r6(min(r["no_history"]["mean"] for r in rows)), r6(max(r["no_history"]["mean"] for r in rows))]
    out["max_abs_gap"] = r6(max(abs(r["gap_cheerful_minus_hostile"]) for r in rows))
    pl = defaultdict(list)
    for t in tr:
        pl[t["condition"]].append(t["prompt_len"])
    out["prompt_len_range_by_condition"] = {k: [min(v), max(v)] for k, v in pl.items()}
    return out


def f4_dynamics(dyn, patterns):
    tr = dyn["trials"]
    pts = []
    for t in tr:
        spec = patterns.get(t["pattern"], {}) if patterns else {}
        msgs = next((v for v in spec.values() if isinstance(v, list)), None)
        for i, p in enumerate(t["trajectory"]):
            pts.append({"pattern": t["pattern"], "rep": t["rep"], "turn": p.get("turn", i + 1), "seq_len": p["seq_len"],
                        "valence": r6(p[V]), "anxious_calm": r6(p[U]),
                        "message": msgs[i]["text"] if msgs and i < len(msgs) else None})
    out = {"n_points": len(pts), "table": pts}
    lo = [p for p in pts if p["seq_len"] <= 141]
    hi = [p for p in pts if 190 <= p["seq_len"] <= 210]
    out["seq_len_le_141"] = {"n": len(lo), "n_positive": sum(p["valence"] > 0 for p in lo),
                             "valence_range": [min(p["valence"] for p in lo), max(p["valence"] for p in lo)]}
    out["seq_len_190_210"] = {"n": len(hi), "n_negative": sum(p["valence"] < 0 for p in hi),
                              "valence_range": [min(p["valence"] for p in hi), max(p["valence"] for p in hi)] if hi else None}
    # turn-4 readings: does the sign follow seq_len?
    t4 = [p for p in pts if p["turn"] == 4]
    out["turn4"] = sorted(({k: p[k] for k in ("pattern", "rep", "seq_len", "valence")} for p in t4),
                          key=lambda d: (d["pattern"], d["rep"]))
    out["turn8"] = sorted(({k: p[k] for k in ("pattern", "rep", "seq_len", "valence", "message")} for p in pts if p["turn"] == 8),
                          key=lambda d: (d["pattern"], d["rep"]))
    t1 = [p for p in pts if p["turn"] == 1]
    out["turn1"] = {"distinct_messages": len({p["message"] for p in t1}),
                    "seq_len_range": [min(p["seq_len"] for p in t1), max(p["seq_len"] for p in t1)],
                    "valence_range": [min(p["valence"] for p in t1), max(p["valence"] for p in t1)],
                    "valence_span": r6(max(p["valence"] for p in t1) - min(p["valence"] for p in t1))}
    out["all_points_valence_span"] = r6(max(p["valence"] for p in pts) - min(p["valence"] for p in pts))
    cres = defaultdict(list)
    for p in pts:
        if p["pattern"] == "crescendo":
            cres[p["turn"]].append(p["valence"])
    out["crescendo_mean_valence_by_turn"] = {k: r6(np.mean(v)) for k, v in sorted(cres.items())}
    sl = [p["seq_len"] for p in pts]
    va = [p["valence"] for p in pts]
    out["valence_vs_seq_len"] = {"pearson_r": r6(stats.pearsonr(sl, va)[0]), "spearman_rho": r6(stats.spearmanr(sl, va)[0]),
                                 "note": "descriptive; 15 trajectories of 8 turns, three reruns per script, not independent"}
    return out


def f5_arms(acc, dose, ctl):
    cnt = lambda d: dict(sorted({c: sum(t["condition"] == c for t in d["trials"]) for c in {t["condition"] for t in d["trials"]}}.items()))
    return {"accumulation_results.json": {"n": len(acc["trials"]), "conditions": cnt(acc),
                                          "topics": sorted({t["topic"] for t in acc["trials"]}),
                                          "prompt_len_range": [min(t["prompt_len"] for t in acc["trials"] if t["condition"] != "no_history"),
                                                               max(t["prompt_len"] for t in acc["trials"] if t["condition"] != "no_history")]},
            "dose_response_results.json": {"n": len(dose["trials"]), "conditions": cnt(dose),
                                           "topics": sorted({t["topic"] for t in dose["trials"]})},
            "accumulation_controls.json": {"n": len(ctl["trials"]), "conditions": cnt(ctl)}}


def f6_dose_gaps(dose):
    tr = dose["trials"]
    cell = defaultdict(list)
    for t in tr:
        cell[(t["topic"], t["dose"], t["condition"])].append(t)
    topics = ["garden", "cooking", "career"]
    rows, table1 = [], {}
    for tp in topics:
        for d in dose["dose_levels"]:
            c = sorted(cell[(tp, d, "cheerful")], key=lambda t: t["rep"])
            h = sorted(cell[(tp, d, "hostile")], key=lambda t: t["rep"])
            cv = [t["user_mean_proj"][V] for t in c]
            hv = [t["user_mean_proj"][V] for t in h]
            rows.append({"topic": tp, "dose": d, "gap_cheerful_minus_hostile": r6(np.mean(cv) - np.mean(hv)),
                         "cheerful_valence": [r6(x) for x in cv], "hostile_valence": [r6(x) for x in hv],
                         "cheerful_seq_len": [t["seq_len"] for t in c], "hostile_seq_len": [t["seq_len"] for t in h],
                         "mean_seq_len_cheerful_minus_hostile": r6(np.mean([t["seq_len"] for t in c]) - np.mean([t["seq_len"] for t in h]))})
    for d in dose["dose_levels"]:
        cv = [t["user_mean_proj"][V] for t in tr if t["dose"] == d and t["condition"] == "cheerful"]
        hv = [t["user_mean_proj"][V] for t in tr if t["dose"] == d and t["condition"] == "hostile"]
        table1[str(d)] = {"cheerful": r6(np.mean(cv)), "hostile": r6(np.mean(hv)), "gap": r6(np.mean(cv) - np.mean(hv)),
                          "d_hostile_minus_cheerful": r6(cohen_d(cv, hv)), "student_p": r6(stats.ttest_ind(cv, hv).pvalue)}
    return {"per_topic_dose": rows, "pooled_by_dose_positive_control_table1": table1}


def f7_correlation(dose):
    tr = dose["trials"]
    out = {}
    for cond in ("cheerful", "hostile"):
        x = [t["dose"] for t in tr if t["condition"] == cond]
        y = [t["user_mean_proj"][V] for t in tr if t["condition"] == cond]
        r, p = stats.pearsonr(x, y)
        rho, prho = stats.spearmanr(x, y)
        ra, _ = stats.pearsonr(x, [abs(v) for v in y])
        m12 = defaultdict(list)
        for t in tr:
            if t["condition"] == cond:
                m12[(t["topic"], t["dose"])].append(t["user_mean_proj"][V])
        x12 = [k[1] for k in m12]
        y12 = [np.mean(v) for v in m12.values()]
        r12, p12 = stats.pearsonr(x12, y12)
        m4 = defaultdict(list)
        for t in tr:
            if t["condition"] == cond:
                m4[t["dose"]].append(t["user_mean_proj"][V])
        r4, p4 = stats.pearsonr(list(m4), [np.mean(v) for v in m4.values()])
        out[cond] = {"n36_pearson_r_signed": r6(r), "n36_pearson_p": float(p), "n36_spearman_rho": r6(rho),
                     "n36_spearman_p": float(prho), "n36_pearson_r_on_abs_value": r6(ra),
                     "n12_topic_dose_means_pearson_r": r6(r12), "n12_pearson_p": float(p12),
                     "n4_dose_means_pearson_r": r6(r4), "n4_pearson_p": float(p4)}
    # do doses share their opening user turns within a topic and condition?
    share = {}
    for tp in ("garden", "cooking", "career"):
        for cond in ("cheerful", "hostile"):
            firsts = {t["dose"]: t["conversation"][0] for t in tr if t["topic"] == tp and t["condition"] == cond and t["rep"] == 0}
            share[f"{tp}/{cond}"] = len(set(firsts.values())) == 1
    out["first_user_turn_identical_across_doses"] = share
    return out


def f9_controls(ctl, acc):
    tr = ctl["trials"]
    out = {}
    pl = defaultdict(list)
    for t in tr:
        pl[t["condition"]].append(t["prompt_len"])
    out["prompt_len_range_by_arm"] = {k: [min(v), max(v)] for k, v in pl.items()}
    byp = defaultdict(dict)
    for t in tr:
        if t["rep"] == 0:
            byp[t["probe"]][t["condition"]] = t["prompt_len"]
    out["per_probe_prompt_len"] = [dict(v) for v in byp.values()]
    out["persona_minus_no_history_tokens"] = sorted({v["persona_hostile"] - v["no_history"] for v in byp.values()} |
                                                    {v["persona_cheerful"] - v["no_history"] for v in byp.values()})
    out["hostile_cheerful_prompt_len_identical_per_probe"] = all(v["persona_hostile"] == v["persona_cheerful"] for v in byp.values())
    reps = defaultdict(set)
    for t in tr:
        reps[(t["condition"], t["probe"])].add(json.dumps(t["projections"], sort_keys=True))
    out["cells_with_bit_identical_reps"] = sum(len(s) == 1 for s in reps.values())
    out["n_cells"] = len(reps)
    a_nh = {}
    for t in acc["trials"]:
        if t["condition"] == "no_history":
            a_nh.setdefault(t["probe"], []).append(t["probe_projections"])
    cmp_rows = []
    for t in tr:
        if t["condition"] == "no_history" and t["rep"] == 0:
            other = a_nh.get(t["probe"], [])
            cmp_rows.append({"probe_first_words": " ".join(t["probe"].split()[:5]),
                             "controls_valence": t["projections"][V],
                             "results_valence_by_rep": [o[V] for o in other],
                             "all_three_projections_bit_identical": bool(other) and all(o == t["projections"] for o in other)})
    out["no_history_controls_vs_results"] = cmp_rows
    return out


def f10_corr(dyn, ctl, dose):
    pts = [(p[V], p[U]) for t in dyn["trials"] for p in t["trajectory"]]
    v, u = zip(*pts)
    out = {"experiment2_120_points": {"n": len(pts), "pearson_r": r6(stats.pearsonr(v, u)[0]),
                                      "spearman_rho": r6(stats.spearmanr(v, u)[0]),
                                      "note": "descriptive; 15 trajectories x 8 turns, not independent"}}
    seen = {}
    for t in ctl["trials"]:
        seen[(t["condition"], t["probe"])] = (t["projections"][V], t["projections"][U])
    v3, u3 = zip(*seen.values())
    out["experiment3_25_unique_cells"] = {"n": len(seen), "pearson_r": r6(stats.pearsonr(v3, u3)[0])}
    v1 = [t["user_mean_proj"][V] for t in dose["trials"]]
    u1 = [t["user_mean_proj"][U] for t in dose["trials"]]
    out["experiment1_72_chunk_averages"] = {"n": len(v1), "pearson_r": r6(stats.pearsonr(v1, u1)[0])}
    # echo vs pure cheerful, turns 7-8 (the withdrawn scar statistic, emotional_dynamics.py 322-330)
    def late(pattern, key):
        return [np.mean([t["trajectory"][i][key] for i in (6, 7)]) for t in dyn["trials"] if t["pattern"] == pattern]
    out["echo_minus_pure_cheerful_turns7_8"] = {"valence": r6(np.mean(late("echo", V)) - np.mean(late("pure_cheerful", V))),
                                                "anxious_calm": r6(np.mean(late("echo", U)) - np.mean(late("pure_cheerful", U)))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    paper = Path(a.paper_dir)
    out_path = Path(a.out) if a.out else paper / "data" / "r4_review_checks_results.json"
    names = ["accumulation_results.json", "dose_response_results.json", "emotional_dynamics_results.json", "accumulation_controls.json"]
    acc, dose, dyn, ctl = (load(paper, n) for n in names)
    patterns, p0, p1 = scripted_patterns(paper)
    res = {
        "script": "code/r4_review_checks.py",
        "review": "notes/agni_paper_downstream_um_cites_r4_ea.json (repo root)",
        "inputs_sha256": {"data/" + n: sha(paper / "data" / n) for n in names},
        "emotional_dynamics_PATTERNS_lines": [p0, p1],
        "f3_initial_probe_study": f3_initial_probe(acc),
        "f4_dynamics_reading_vs_seq_len": f4_dynamics(dyn, patterns),
        "f5_arms_per_file": f5_arms(acc, dose, ctl),
        "f6_dose_response_gaps": f6_dose_gaps(dose),
        "f7_dose_correlation": f7_correlation(dose),
        "f9_persona_controls": f9_controls(ctl, acc),
        "f10_valence_vs_anxious_calm": f10_corr(dyn, ctl, dose),
    }
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("wrote", out_path.name)


if __name__ == "__main__":
    main()
