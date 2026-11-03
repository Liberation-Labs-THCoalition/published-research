#!/usr/bin/env python3
"""Derive prereg_um_v2_R1R2.json FROM PREREG_um_v2_R1R2.md. One source, not two (after prereg_18h_to_json.py).

What this adds to the 18h generator:
- A bullet's continuation lines are joined to it (the 18h extractor cut every wrapped bullet mid-sentence).
- It reproduces exactly what Agni's review_common.py (1.2.0, ~/lab/projects/agni-open on MTH) sends the reviewer, and
  refuses to write if anything would be cut:
  - hypotheses: '  - primary_hypothesis[i]: ...' joined, cut at 3,000;
  - threats: the union of threats_to_validity, cannot_establish, mandatory_gates and cannot_establish AGAIN (THREAT_KEYS
    lists it twice), joined the same way, cut at 3,000. Round 3's "mandatory_gates[6] reads '**Thomas ' and stops"
    was this cut, not a truncated gate in the file;
  - decisions: falsification_criteria then analysis_plan (DECISION_KEYS order), cut at 4,000. Section 5 lives under
    `outcomes`, which review_common never sends, so §13's compact rules go out as `falsification_criteria`, first.
- §12's registered constants are copied in as `registered_constants`; both scripts assert theirs against it.

    python prereg_um_v2_to_json.py            # write the JSON
    python prereg_um_v2_to_json.py --check    # is the JSON current with the markdown and the artifacts?
"""
import hashlib, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent


def sealed_constants(reg):
    """Refuse to write if the prediction file no longer has the hash typed into §12 when it was sealed. Without this,
    regenerating would re-hash an edited prediction into `artifacts` and bless it (Lyra, sign-off 2026-10-19)."""
    got = hashlib.sha256((HERE / "R1R2_PREDICTION.md").read_bytes()).hexdigest()
    if reg.get("prediction_sha256") != got:
        raise SystemExit(f"R1R2_PREDICTION.md sha256 {got[:16]} != §12 prediction_sha256 "
                         f"{str(reg.get('prediction_sha256'))[:16]}: the sealed prediction was edited, or §12 lacks it")
    return reg
MD, OUT = HERE / "PREREG_um_v2_R1R2.md", HERE / "prereg_um_v2_R1R2.json"
CORR = HERE.parent
HYP_CAP, THREAT_CAP, DECISION_CAP = 3000, 3000, 4000            # review_common.py 1.2.0
THREAT_ORDER = ("threats_to_validity", "cannot_establish", "mandatory_gates", "cannot_establish")
LOCAL = {"um_v2_encode.py": HERE / "um_v2_encode.py", "um_v2_analysis.py": HERE / "um_v2_analysis.py",
         "um_v2_dryrun.py": HERE / "um_v2_dryrun.py", "R0_text_baseline.py": HERE / "R0_text_baseline.py",
         "R0_results.json": HERE / "R0_results.json", "R1R2_PREDICTION.md (sealed, not edited)": HERE / "R1R2_PREDICTION.md",
         "run_plan": CORR / "user_model_v2_run_plan_20261018.md",
         "correction_plan": CORR / "user_model_2026a_correction_plan_20261014.md"}
MEASURED = {  # read on Starship with `shasum -a 256`, or from the HF API; not re-checkable from here
    "trials_file emotion_bridge_trials.json": "fc5938feccc57c68db0c1625236839d7405708219911dc2c532de836eb836e7e",
    "wk_backtrack_probe_v2.py (PROVENANCE_wk_probes.md; same on Starship)":
        "8a41036dcd87497b35884a750a5919ff60a82f594246c6131b205b97acae5c6b",
    "model snapshot (HF commit; repo history-squashed 2026-07-07)": "ad356102ce8ea7122a18e6402f9b2e37446fc9d7",
    "chat_template.jinja": "0800c03b537cb0d60244f37ac6d717b06c42c4828a40853b1f9853791f9afdae",
    "tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
    "config.json": "ae9339e5b96d00c98b0a7c023712cf8dcab05e3c48fa6dc7550440cfb786020f"}


def section(md, heading_re):
    m = re.search(rf"^(#{{2,3}})\s*{heading_re}.*$", md, re.M)
    if not m:
        raise SystemExit(f"SECTION NOT FOUND: /{heading_re}/. Refusing to emit an empty field.")
    level = len(m.group(1))
    rest = md[m.end():]
    nxt = re.search(rf"^#{{1,{level}}}\s", rest, re.M)
    body = rest[:nxt.start()] if nxt else rest
    return re.sub(r"\n-{3,}\s*$", "", body.strip()).strip()


def bullets(text):
    """Top-level '- ' or 'N. ' items, each with its indented continuation lines joined."""
    out = []
    for line in text.splitlines():
        if re.match(r"^(\d+\.|-)\s+\S", line):
            out.append(re.sub(r"^(\d+\.|-)\s+", "", line).strip())
        elif out and line.startswith("  ") and line.strip():
            out[-1] += " " + line.strip()
    return out


def hypotheses(text):
    return [" ".join((m.group(1) + " " + m.group(2)).split())
            for m in re.finditer(r"^\*\*(H\d[^*]*)\*\*\s*(.+?)(?=\n\n\*\*H|\Z)", text, re.M | re.S)]


def as_sent(key, val):
    """review_common._items, then '  - %s' and '\\n'.join."""
    if isinstance(val, str):
        items = [f"{key}: {val}"]
    else:
        items = [f"{key}[{i}]: {v if isinstance(v, str) else json.dumps(v)}" for i, v in enumerate(val)]
    return [f"  - {x}" for x in items]


def agni_view(d):
    hyp = "\n".join(as_sent("primary_hypothesis", d["primary_hypothesis"]))
    thr = "\n".join(line for k in THREAT_ORDER for line in as_sent(k, d[k]))
    fc = "\n".join(as_sent("falsification_criteria", d["falsification_criteria"]))
    dec = fc + "\n" + "\n".join(as_sent("analysis_plan", d["analysis_plan"]))
    return {"hypotheses": (len(hyp), HYP_CAP), "threats_union": (len(thr), THREAT_CAP),
            "falsification_criteria_within_decisions": (len(fc), DECISION_CAP),
            "analysis_plan_chars_reaching_agni": max(0, DECISION_CAP - len(fc) - 1), "decisions_total": len(dec)}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def build():
    md = MD.read_text(encoding="utf-8")
    hyps = hypotheses(section(md, r"2\.\s*Hypotheses"))
    threats = bullets(section(md, r"7\.\s*Threats to validity"))
    if len(hyps) != 3:
        raise SystemExit(f"{len(hyps)} hypotheses extracted; the design states H1, H2 and H3. Refusing.")
    if len(threats) < 8:
        raise SystemExit(f"only {len(threats)} threats extracted. Refusing.")
    block = re.search(r"```json\n(.+?)\n```", section(md, r"12\.\s*Registered constants"), re.S)
    if not block:
        raise SystemExit("§12 has no ```json block of registered constants. Refusing.")
    d = {
        "experiment": "um_v2_R1R2",
        "title": "User-model correction: reproduce the paper's probe (R1), run the probe it describes (R2), "
                 "both paired against the text baseline, and re-test the spectral null (R7), on one encode",
        "date": "2026-10-18 (revised 2026-10-19)",
        "author": "Lyra (drafted for Lyra by a Claude Code agent; Lyra reviews and seals)",
        "status": "DRAFT, REVISED AFTER AGNI ROUND 3 (both CONDITIONAL in rounds 1-3; the last round), NOT RUN",
        "model": ["Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"],
        "primary_hypothesis": hyps,
        "falsification_criteria": bullets(section(md, r"13\.\s*Decision rules in brief")),
        "threats_to_validity": threats,
        "analysis_plan": section(md, r"4\.\s*Pre-registered analysis plan"),
        "outcomes": section(md, r"5\.\s*Decision rules"),
        "power_analysis": section(md, r"6\.\s*Power analysis"),
        "gate_sequence": section(md, r"10\.\s*Gate sequence"),
        "cannot_establish": bullets(section(md, r"8\.\s*What R1, R2 and R7 cannot establish")),
        "mandatory_gates": bullets(section(md, r"9\.\s*Mandatory gates")),
        "registered_constants": sealed_constants(json.loads(block.group(1))),
        "artifacts": {**{k: sha(p) for k, p in LOCAL.items()}, **MEASURED},
        "derived_from": {"markdown": MD.name, "markdown_sha256": sha(MD),
                         "note": "Generated by prereg_um_v2_to_json.py. Do not hand-edit: the markdown is the single "
                                 "source. Script hashes go stale when a script changes; --check says so."},
        "full_markdown": md,
    }
    view = agni_view(d)
    for name in ("hypotheses", "threats_union", "falsification_criteria_within_decisions"):
        n, cap = view[name]
        if n > cap:
            raise SystemExit(f"{name}: {n} chars as Agni receives it; review_common cuts at {cap}. Shorten the "
                             f"markdown, do not cut the field.")
    d["agni_view"] = view
    return d


def main():
    d = build()
    if "--check" in sys.argv:
        if not OUT.exists():
            print("JSON missing")
            return 1
        cur = json.loads(OUT.read_text(encoding="utf-8"))
        stale = [k for k in ("markdown_sha256",) if cur["derived_from"].get(k) != d["derived_from"][k]]
        stale += [k for k, v in d["artifacts"].items() if cur["artifacts"].get(k) != v]
        print("current" if not stale else f"STALE: {stale}")
        return 1 if stale else 0
    OUT.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    v = d["agni_view"]
    print(f"wrote {OUT.name}: hypotheses {v['hypotheses'][0]}/{HYP_CAP}, threats+cannot x2+gates "
          f"{v['threats_union'][0]}/{THREAT_CAP}, falsification_criteria {v['falsification_criteria_within_decisions'][0]}"
          f"/{DECISION_CAP} (then {v['analysis_plan_chars_reaching_agni']} chars of analysis_plan reach Agni), "
          f"{len(d['falsification_criteria'])} rules, {len(d['registered_constants'])} registered constants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
