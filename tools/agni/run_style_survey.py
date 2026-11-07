#!/usr/bin/env python3
"""Driver: run the Agni STYLE gate over every paper that has an academic twin.

Survey only. Writes one JSON per paper to tools/agni/style/.
Failures are recorded as failures -- a missing output never counts as a pass.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PR = HERE.parent.parent                      # published-research/
OUT = HERE / "style"
OUT.mkdir(exist_ok=True)
LOG = OUT / "_run_log.json"

PAPERS = [
    "cache-tracing", "decision-state-paper", "delta-manifold-paper",
    "emotion-accumulation-paper", "emotional-trajectory-paper", "formulary-paper",
    "graph-topology-paper", "identity-geometry", "kv-cloak-defense-paper",
    "lyra-technique-ii", "meta-pattern", "mine5-selective-sharpener",
    "null-swarm-paper", "oracle-loop-paper", "presence-metric",
    "spectral-shape-paper", "waystations-paper",
]


def run_one(name):
    flight = PR / name / "main.tex"
    twin = PR / name / "academic" / "main.tex"
    out = OUT / ("%s.agni_style.json" % name)
    rec = {"paper": name, "flight": str(flight), "twin": str(twin),
           "out": str(out), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not flight.exists():
        rec.update(status="FAIL", reason="flight missing")
        return rec
    if not twin.exists():
        rec.update(status="FAIL", reason="twin missing")
        return rec
    env = dict(os.environ)
    env.setdefault("AGNI_CLI_TIMEOUT", "1500")
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, str(HERE / "agni_phases.py"), "style",
             str(flight), str(twin), "--out", str(out)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=2400, cwd=str(HERE), env=env)
        rec["exit"] = r.returncode
        rec["stderr_tail"] = (r.stderr or "")[-800:]
    except subprocess.TimeoutExpired:
        rec.update(status="FAIL", reason="driver timeout 2400s",
                   elapsed=round(time.time() - t0, 1))
        return rec
    rec["elapsed"] = round(time.time() - t0, 1)
    # A gate that returns nothing has not passed.
    if not out.exists():
        rec.update(status="FAIL", reason="no output file (exit %s)" % rec["exit"])
        return rec
    try:
        d = json.loads(out.read_text(encoding="utf-8"))
    except Exception as e:
        rec.update(status="FAIL", reason="unparseable output: %s" % e)
        return rec
    rec["verdict"] = d.get("verdict", "MISSING")
    rec["review_chars"] = len(d.get("review") or "")
    rec["input_complete"] = (d.get("input_coverage") or {}).get("input_complete")
    rec["record_timestamp"] = d.get("timestamp")    # summarize_style.py matches the record on this
    if rec["verdict"] in ("UNPARSED", "MISSING", "INVALID") or rec["review_chars"] == 0:
        rec["status"] = "FAIL"
        rec["reason"] = "verdict=%s chars=%d" % (rec["verdict"], rec["review_chars"])
        if d.get("invalid_reasons"):
            rec["reason"] += ": " + "; ".join(d["invalid_reasons"])
    elif rec["exit"] not in (0, 1) and rec["input_complete"] is not False:
        # Exit 2 means UNGATED whatever the file says: a run that could not write its record
        # leaves the old one on disk, and that one can say APPROVED.
        rec["status"] = "FAIL"
        rec["reason"] = "exit %s beside a record saying %s: stale" % (rec["exit"], rec["verdict"])
    elif rec["input_complete"] is not True:
        # A review of part of the input is not a pass (round 2), and a record with no
        # input_coverage cannot show that the reviewer got all of it.
        rec["status"] = "FAIL"
        rec["reason"] = ("input not complete: %s from a reviewer that did not get all of its input"
                         % rec["verdict"] if rec["input_complete"] is False else
                         "the record has no input_coverage, so it cannot show the input was whole")
    else:
        rec["status"] = "OK"
    return rec


def main():
    # A cp1252 console cannot print an arrow from a path in invalid_reasons, and the summary died
    # after its first row (round 2). Same fix as agni_phases.py.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    todo = sys.argv[1:] or PAPERS
    log = []
    if LOG.exists():
        try:
            log = json.loads(LOG.read_text(encoding="utf-8"))
        except Exception:
            log = []
    for i, name in enumerate(todo, 1):
        print("[%d/%d] %s ..." % (i, len(todo), name), flush=True)
        rec = run_one(name)
        print("      -> %s %s (%ss)" % (rec["status"], rec.get("verdict", rec.get("reason", "")),
                                        rec.get("elapsed", "?")), flush=True)
        log = [x for x in log if x.get("paper") != name] + [rec]
        LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")  # persist before display
    print("\n=== SURVEY SUMMARY ===")
    for rec in sorted(log, key=lambda x: x["paper"]):
        print("%-32s %-6s %-12s %s" % (rec["paper"], rec["status"],
                                       rec.get("verdict", "-"),
                                       rec.get("reason", "")))


if __name__ == "__main__":
    main()
