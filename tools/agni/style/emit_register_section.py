# Emit the 2026-09-04 style-gate section and append it to REMEDIATION_REGISTER.md
import json, os, io, collections, re, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
REG = r"C:\Users\Thomas\Desktop\LiberationLabs\Research\published-research\REMEDIATION_REGISTER.md"
rows = json.load(open(os.path.join(HERE, "tiered.json"), encoding="utf-8"))

UNTWINNED = {"adversarial-audit-methodology", "consequentiality-decomposition",
             "deception-detection-nulls", "empathy-bus", "ethics-pack-injection",
             "ghost-dimensions", "kv-decomposition-paper", "logit-bias-confab",
             "mnemosyne-ablation", "mnemosyne-benchmark", "targeted-deception-correction"}


def oneline(t, lim=155):
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("|", "\\|")
    if len(t) > lim:
        cut = t[:lim]
        sp = cut.rfind(" ")
        t = cut[:sp if sp > 90 else lim].rstrip(" ,;:—-") + " …"
    return t


def src(r):
    return "`%s.agni_style.json` #%d" % (r["paper"], r["n"])


def table(bucket_rows, closed_col=False):
    out = []
    out.append("| Paper | Sev | Defect | Source |")
    out.append("|---|---|---|---|")
    for r in sorted(bucket_rows, key=lambda x: (x["paper"], x["n"])):
        d = oneline(r["text"])
        if r.get("closed"):
            d = d + " — **" + r["closed"] + "**"
        out.append("| `%s` | %s | %s | %s |" % (r["paper"], r["sev"][:4], d, src(r)))
    return "\n".join(out)


by = collections.defaultdict(list)
for r in rows:
    by[r["bucket"]].append(r)

sev = collections.Counter(r["sev"] for r in rows)
papers = len(set(r["paper"] for r in rows))
ut = [r for r in rows if r["paper"] in UNTWINNED]
utc = collections.Counter(r["sev"] for r in ut)

L = []
A = L.append

A("\n---\n")
A("# ═══ 2026-09-04 — AGNI STYLE GATE, FULL CORPUS ═══")
A("")
A("**Added:** 2026-09-04 · **Source:** `published-research/tools/agni/style/*.agni_style.json` "
  "(35 files, gate run 2026-09-03 22:04–22:54 local / 2026-09-04 05:03–05:54 UTC)")
A("")
A("The STYLE gate ran on the whole corpus for the first time. **35 papers reviewed, 0 approved** "
  "(31 REJECTED, 4 CONDITIONAL). This section adds every finding to the register. Until now the "
  "register tracked only the twinned set (decision-state, spectral-shape, user-model, waystations "
  "and neighbours); **the 11 untwinned `published-research` papers had never appeared here at all** "
  "— which is the exact gap the register exists to close.")
A("")
A("**Extraction: %d findings — %d CRITICAL, %d MAJOR, %d MINOR — across %d papers.**"
  % (len(rows), sev["CRITICAL"], sev["MAJOR"], sev["MINOR"], papers))
A("")
A("| | CRITICAL | MAJOR | MINOR | total |")
A("|---|---|---|---|---|")
A("| Whole corpus (35 papers) | %d | %d | %d | %d |" % (sev["CRITICAL"], sev["MAJOR"], sev["MINOR"], len(rows)))
A("| The 11 untwinned papers | %d | %d | %d | %d |" % (utc["CRITICAL"], utc["MAJOR"], utc["MINOR"], len(ut)))
A("")
A("The **127** in the brief is the untwinned papers' CRITICAL+MAJOR (44+83) — reproduced exactly. "
  "The brief's companion figure of *48* MINOR is an **undercount**; the true number is **58**. The "
  "missing 10 are `empathy-bus` #18–#27, which are written as bare numbered lines under a `### MINOR` "
  "heading rather than carrying a severity token per line. A parser keyed on `**N. MINOR —**` reads "
  "that paper as having zero minors. My own first-pass parser made the identical error, on the same "
  "file, and also lost `decision-state` #21–27 and `lyra-technique-ii` #21–31 the same way.")
A("")
A("**Parse validation (the check that would have caught that):** every one of the 35 reviews numbers "
  "its findings 1..N. The parser is accepted only when the extracted set is exactly `{1..max}` with no "
  "gaps. All 35 files pass, and **no file with a REJECTED/CONDITIONAL verdict yielded zero findings**. "
  "`_summary.txt` in the gate directory is **not** a usable cross-check: it covers only 17 of the 35 "
  "papers, and its `decision-state` (34) and `lyra-technique-ii` (36) totals exceed those papers' own "
  "highest finding numbers (27 and 31). Trust the JSONs, not the summary.")
A("")
A("**Tier distribution**")
A("")
A("| Bucket | n | Meaning |")
A("|---|---|---|")
for k, lbl in [("T1", "text-only fix"), ("T1b", "venue-copy sync"), ("T2", "reanalyse, no new compute"),
               ("T3", "rerun, needs compute"), ("T4", "needs a human call"),
               ("F1", "**FLAGGED** dual-use / staged release"),
               ("F2", "**FLAGGED** authorship & sign-off"),
               ("F3", "**FLAGGED** paper claims a review that did not happen"),
               ("F4", "**FLAGGED** missing primary — locate / recompute / rerun / retract"),
               ("CLOSED", "already fixed, verified on disk")]:
    A("| %s | %d | %s |" % (k, len(by[k]), lbl))
A("")
A("Tiering is by **what the fix costs**, not by how alarming the label is. A CRITICAL that is one "
  "wrong word in an abstract is T1; a MINOR that needs a rerun is T3. Where the cost genuinely could "
  "not be read off the finding, the item is in T4 or F4 rather than guessed into T1 — an under-tiered "
  "item gets attempted and fails silently; a T4 gets read by a human.")
A("")

A("---")
A("")
A("## FLAGGED — not ordinary remediation work")
A("")
A("### F1 — DUAL-USE / STAGED RELEASE (%d) · do not action without Thomas" % len(by["F1"]))
A("")
A("`ghost-dimensions` carries a **dual-use redaction that is intentional**. Its content is not "
  "described here and must not be reproduced in any register entry, commit message, or paper edit. "
  "The finding against it is only that the *redaction note itself* was dropped from the paper — the "
  "fix is to restore the note, not to restore the material.")
A("")
A(table(by["F1"]))
A("")
A("`targeted-deception-correction` is the sharper problem: **three editions state three different "
  "release policies**, and every edition says the auto-calibrator is *both* provided and withheld. "
  "`MANIFEST.md:41–47` is the governing register. This is a safety-policy question, not a wording "
  "question — one sentence must be settled by a human and then pasted verbatim into all locations.")
A("")

A("### F2 — AUTHORSHIP & SIGN-OFF (%d) · T4 decisions, not text fixes" % len(by["F2"]))
A("")
A("**Partly resolved already.** A corpus-wide byline sweep ran **2026-09-03 23:35 local — 41 minutes "
  "after the gate finished**, so the gate reviewed the pre-sweep state. Verified by diffing the "
  "`*.bak-byline-20260903` backups (46 files, 41 directories): Dwayne Wilkes was moved from the byline "
  "into Acknowledgments as *\"statistical auditing and red-team review\"*. That closes the **placement** "
  "question corpus-wide.")
A("")
A("**It does not close the consent question.** Whether Dwayne, Kavi, Ang Jandak and CC actually signed "
  "off is unrecorded, and several reviews found the artifact asserting a sign-off that no repo record "
  "supports (`f9288a9` — *\"Kavi to acknowledgements (pending sign-off)\"* — is five weeks old and still "
  "unresolved). Every row below needs a human answer, not an edit. Register #110 stays open.")
A("")
A(table(by["F2"]))
A("")

A("### F3 — A PAPER CLAIMS A REVIEW THAT DID NOT HAPPEN (%d)" % len(by["F3"]))
A("")
A("Its own class because it is the one defect that **disables the corrective machinery**: a false audit "
  "record makes the absence of an audit undetectable, and it survives exactly the glance that a wrong "
  "number would not.")
A("")
A("The headline case is `mnemosyne-ablation`, whose **two editions make opposite claims about whether "
  "the paper was reviewed at all** — and no gate record, date, reviewer or disposition exists anywhere "
  "under `published-research/` for the five rounds one edition claims. `consequentiality-decomposition` "
  "asserts *\"All audit reports are preserved\"* and thanks Agni *\"across all stages\"* while its Stage 5 "
  "— the headline result — has no archived data and was never audited. `kv-decomposition-paper` has "
  "never been through this gate at all.")
A("")
A(table(by["F3"]))
A("")

A("### F4 — MISSING PRIMARY (%d) · locate / recompute / rerun / retract — a human call" % len(by["F4"]))
A("")
A("These are **worse than a defect: there is nothing to check against.** The register already has a "
  "*Missing primaries* section; these extend it. They are deliberately not tiered T2 or T3, because "
  "which one applies cannot be read off the finding — you cannot know whether the primary is missing, "
  "misfiled, or was never produced without going to look. Guessing T2 here produces an item that gets "
  "attempted and fails.")
A("")
A(table(by["F4"]))
A("")

A("---")
A("")
A("## T1 — TEXT ONLY (%d)" % len(by["T1"]))
A("")
A("Claim is wrong or contradictory; the data is fine. Edit, rebuild, verify in the PDF. Includes the "
  "bulk of the MINORs (captions, `cleveref`, stale `\\date`, orphaned `.bib` entries, overfull hboxes).")
A("")
A(table(by["T1"]))
A("")

A("## T1b — VENUE-COPY SYNC (%d)" % len(by["T1b"]))
A("")
A("A twin diverges; propagate the correction. **The 7 divergences registered on 2026-09-01 did not "
  "re-appear in this gate run** — good evidence they were fixed. These are *new* ones. Note the shape "
  "recurs: several papers turn out to have **three or four** editions, not two "
  "(`oracle-loop-paper`, `decision-state-paper`, `formulary-paper`, `graph-topology-paper`, "
  "`deception-detection-nulls`), so a two-way twin diff is not sufficient.")
A("")
A(table(by["T1b"]))
A("")

A("## T2 — REANALYSE (%d) · data sound, analysis wrong, no new compute" % len(by["T2"]))
A("")
A("**Read this one first:** `hr-temporal-boundary` #1. The 2026-09-01 twelvefold-pseudoreplication "
  "correction — this register's own entry — **under-corrected at the one point where it claims "
  "completeness.** `main.tex:508–510` carves out Table 2 as *\"correct as stated\"* at n=30; "
  "`data/figure_generation.py:220–236` loops `for si in range(30): for rep in range(3)` and the three "
  "printed p-values (0.237, 0.500, 0.156) match n=90 exactly and n=30 not at all. The retraction notice "
  "contains the error it retracts. Recompute on 30 scenario-level points. **This is not covered by the "
  "existing 117a/117k closure and must not be read as closed by it.**")
A("")
A(table(by["T2"]))
A("")

A("## T3 — RERUN (%d) · needs new compute, gated" % len(by["T3"]))
A("")
A(table(by["T3"]))
A("")

A("## T4 — DECIDE (%d) · blocked on a call, not on work" % len(by["T4"]))
A("")
A("Plus everything under **F1**, **F2** and **F4** above, which are all human calls.")
A("")
A(table(by["T4"]))
A("")

A("## CLOSED — already fixed before this entry (%d)" % len(by["CLOSED"]))
A("")
A("Verified against the papers on disk, not against a note saying they were fixed.")
A("")
A(table(by["CLOSED"]))
A("")
A("**Two closures I had to retract before publishing this entry.** I first marked `empathy-bus` #7 "
  "and `waystations-paper` #11 CLOSED on the strength of the corpus-wide byline sweep. Both were "
  "wrong. `empathy-bus` was never in the sweep (its only backup is `main.pdf.bak-2026-08-25`) and its "
  "byline still disagrees with its own repo index; `waystations-paper/academic` still carries "
  "*\"Thomas Edrington … Aided by AI research agents\"* over a CRediT block naming Dwayne Wilkes. "
  "**A corpus-level sweep is not evidence about a paper it did not touch.** Both are open in F2.")
A("")
A("**Also closed at register level, by absence:** the 7 venue-copy divergences tabled under T1b on "
  "2026-09-01 (mine5 TOST, decision-state entity confound, waystations null, mine5 title, user-model "
  "\"proves\", spectral-shape \"Eight of the methods\", convergence-paper 2.5–2.8×) were **not** re-raised "
  "by this gate run, which reviewed all of those files. Absence of a finding is weaker evidence than a "
  "diff, so treat as closed-pending-spot-check rather than proven.")
A("")
A("**NOT closed, despite looking adjacent:** `temporal-boundary`'s twelvefold pseudoreplication is "
  "recorded as corrected (117a/117k, 2026-09-01) — but see **T2 / `hr-temporal-boundary` #1** above. The "
  "correction exempted Table 2 and the exemption is false. Do not fold that finding into the closure.")
A("")
A("---")
A("")
A("*Generated from the gate JSONs by a scripted parser (numbering-contiguity validated); tiering "
  "assigned by hand for all CRITICAL findings and for every finding in F1–F4, by rule elsewhere. "
  "No paper was edited in producing this entry.*")

section = "\n".join(L) + "\n"
open(REG, "a", encoding="utf-8").write(section)
print("appended %d chars, %d lines" % (len(section), section.count("\n")))
print("buckets:", {k: len(v) for k, v in by.items()})
