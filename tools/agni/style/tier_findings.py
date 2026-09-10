# Tier the parsed Agni STYLE findings by COST and emit the register section.
import json, re, os, sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "findings.json"), encoding="utf-8"))

# ---------------------------------------------------------------- special classes
# F1 dual-use / staged release. F3 false review claim.  (explicit, hand-assigned)
DUALUSE = {("targeted-deception-correction", 1), ("targeted-deception-correction", 2),
           ("targeted-deception-correction", 3), ("targeted-deception-correction", 6),
           ("ghost-dimensions", 15), ("oracle-loop-paper", 6)}
FALSEREVIEW = {("mnemosyne-ablation", 2), ("consequentiality-decomposition", 3),
               ("consequentiality-decomposition", 4), ("kv-decomposition-paper", 3),
               ("kv-decomposition-paper", 4), ("deception-detection-nulls", 2),
               ("ethics-pack-injection", 9), ("oracle-loop-paper", 10),
               ("mine5-selective-sharpener", 5), ("null-swarm-paper", 2),
               ("adversarial-audit-methodology", 1), ("consequentiality-decomposition", 10)}

AUTH_RE = re.compile(r"byline|author(?:ship| list| line|s whose|-)|CRediT|sign-?off|"
                     r"acknowledg|credited|uncredited|contribution mismatch|Kavi|Dwayne|"
                     r"AI disclosure|author erased|attribution", re.I)
AUTH_DENY = {("consequentiality-decomposition", 12), ("decision-state-paper", 27),
             ("hr-mode-switching", 5), ("hr-mode-switching", 14),
             ("hr-contextual-engagement-paper", 17), ("meta-pattern", 6),
             ("empathy-bus", 10)}

# ---------------------------------------------------------------- verified CLOSED
CLOSED = {
    ("mine5-selective-sharpener", 1):
        "CLOSED 2026-09-03 — verified on disk: title no longer asserts the sharpener; "
        "\"suggested but not confirmed\"; TOST reads \"does not confirm\" in all four copies.",
    ("hr-mine5-selective-sharpener", 1):
        "CLOSED 2026-09-03 — verified: `human-review/.../academic/main.tex:281` now reads "
        "\"suggested but not confirmed\", matching flight.",
    ("spectral-shape-paper", 1):
        "CLOSED 2026-09-03 (placement only) — byline sweep moved Dwayne Wilkes to "
        "Acknowledgments across 41 dirs (diff-verified). Sign-off itself remains open — see F2.",
}
# Dwayne byline-PLACEMENT closures from the 2026-09-03 23:35 sweep
DW_SWEPT = set()  # emptied 2026-09-04: verified on disk that neither empathy-bus nor
# waystations was touched by the byline sweep; their defects still stand. Do not
# close a per-paper finding from a corpus-level sweep without opening the paper.

# ---------------------------------------------------------------- tier overrides
T = {}
def setT(tier, *keys):
    for k in keys: T[k] = tier

# --- T2: data exists, the ANALYSIS is wrong -> recompute, no new compute
setT("T2",
 ("hr-temporal-boundary",1), ("hr-temporal-boundary",10),
 ("decision-state-paper",2), ("decision-state-paper",10), ("decision-state-paper",11),
 ("decision-state-paper",12), ("decision-state-paper",13), ("decision-state-paper",14),
 ("delta-manifold-paper",6), ("delta-manifold-paper",7), ("delta-manifold-paper",8),
 ("delta-manifold-paper",9), ("delta-manifold-paper",3),
 ("emotional-trajectory-paper",1), ("emotional-trajectory-paper",2),
 ("emotional-trajectory-paper",3), ("emotional-trajectory-paper",4),
 ("emotional-trajectory-paper",5),
 ("formulary-paper",2), ("formulary-paper",3), ("formulary-paper",4), ("formulary-paper",5),
 ("graph-topology-paper",2), ("graph-topology-paper",3), ("graph-topology-paper",8),
 ("consequentiality-decomposition",1), ("consequentiality-decomposition",5),
 ("ghost-dimensions",1), ("ghost-dimensions",5),
 ("empathy-bus",1), ("empathy-bus",11), ("empathy-bus",12), ("empathy-bus",13),
 ("hr-contextual-engagement-paper",1), ("hr-contextual-engagement-paper",7),
 ("hr-contextual-engagement-paper",9),
 ("hr-dual-detector-paper",2), ("hr-dual-detector-paper",3), ("hr-dual-detector-paper",4),
 ("hr-mode-switching",11), ("hr-mode-switching",12), ("hr-mode-switching",13),
 ("hr-mode-switching",18),
 ("kv-cloak-defense-paper",2), ("kv-decomposition-paper",2), ("kv-decomposition-paper",9),
 ("logit-bias-confab",6), ("lyra-technique-ii",2), ("lyra-technique-ii",13),
 ("lyra-technique-ii",20), ("mnemosyne-ablation",3), ("mnemosyne-ablation",4),
 ("mnemosyne-benchmark",3), ("mnemosyne-benchmark",4), ("mnemosyne-benchmark",6),
 ("null-swarm-paper",3), ("null-swarm-paper",4), ("null-swarm-paper",6),
 ("presence-metric",3), ("waystations-paper",1), ("waystations-paper",8),
 ("waystations-paper",9), ("meta-pattern",3), ("meta-pattern",5),
 ("ethics-pack-injection",13), ("hr-convergence-paper",7),
)
# --- T3: data cannot answer it; needs new generation / compute
setT("T3",
 ("hr-temporal-boundary",2), ("kv-decomposition-paper",2),
 ("graph-topology-paper",1), ("ethics-pack-injection",11), ("ethics-pack-injection",12),
 ("mnemosyne-benchmark",7), ("mnemosyne-benchmark",11), ("hr-mode-switching",2),
 ("hr-mode-switching",3), ("hr-mode-switching",17), ("hr-dual-detector-paper",5),
 ("logit-bias-confab",11), ("empathy-bus",12),
)
# --- T4: needs a human call (policy / scope / ambiguous)
setT("T4",
 ("mine5-selective-sharpener",6), ("meta-pattern",1), ("logit-bias-confab",16),
 ("logit-bias-confab",17), ("hr-mode-switching",16), ("deception-detection-nulls",3),
 ("oracle-loop-paper",1), ("mnemosyne-benchmark",1), ("mnemosyne-benchmark",5),
 ("mnemosyne-benchmark",12), ("hr-gwt-response",5), ("adversarial-audit-methodology",7),
)
# --- F4 missing primary: locate / recompute / rerun / retract -> a human call
MISSING = {
 ("presence-metric",1), ("identity-geometry",2), ("lyra-technique-ii",3),
 ("logit-bias-confab",5), ("kv-decomposition-paper",1), ("adversarial-audit-methodology",1),
 ("delta-manifold-paper",5), ("delta-manifold-paper",10), ("delta-manifold-paper",11),
 ("emotional-trajectory-paper",10), ("ethics-pack-injection",10), ("formulary-paper",7),
 ("hr-mine5-selective-sharpener",8), ("mnemosyne-ablation",7), ("null-swarm-paper",5),
 ("empathy-bus",10), ("emotion-accumulation-paper",5), ("decision-state-paper",1),
 ("consequentiality-decomposition",4), ("hr-gwt-response",5), ("spectral-shape-paper",5),
 ("graph-topology-paper",11), ("formulary-paper",6), ("hr-dual-detector-paper",5),
}

TWIN = re.compile(r"TWIN_DESYNC|twin|the two editions|both editions|academic edition|"
                  r"venue-facing edition|flight edition|venue copy|three editions|"
                  r"four artifacts|editions? (?:disagree|diverge)|third edition|"
                  r"fourth variant|`paper\.json`|Markdown edition", re.I)
T2_RE = re.compile(r"does not reproduce|do not reproduce|not the quotient|arithmetic|"
                   r"recompute|wrong denominator|contradicts the released data|"
                   r"disagrees with the released data|silently drops|double-count|"
                   r"overcounts|miscount|below the resolution|pseudo-?replicat|"
                   r"counts .{0,30}as|selective reporting|is computed on|"
                   r"cannot both be true|does not follow from", re.I)
MISS_RE = re.compile(r"no (?:data )?artifact|does not exist|no backing (?:data|artifact)|"
                     r"no primary|nowhere in the package|no code path|no verification path|"
                     r"no source data|is not in the repository|unverifiable|no reported results|"
                     r"never archived|no committed artifact|has no supporting artifact|"
                     r"zero data or code", re.I)

def tier_of(paper, f):
    k = (paper, f["n"])
    if k in T: return T[k]
    t = f["text"]
    if MISS_RE.search(t): return "F4"
    if T2_RE.search(t): return "T2"
    if TWIN.search(t): return "T1b"
    return "T1"


# --- second-pass corrections after reviewing the CRITICAL/MAJOR T1 bucket
setT("T2",
 ("delta-manifold-paper",2), ("ghost-dimensions",4), ("identity-geometry",6),
 ("identity-geometry",7), ("identity-geometry",8), ("hr-dual-detector-paper",6),
 ("kv-cloak-defense-paper",3), ("lyra-technique-ii",14), ("lyra-technique-ii",15),
 ("mnemosyne-ablation",5), ("oracle-loop-paper",4), ("spectral-shape-paper",7),
 ("waystations-paper",3),
)
setT("T4",
 ("ghost-dimensions",7), ("decision-state-paper",19), ("formulary-paper",8),
 ("hr-mode-switching",5), ("hr-temporal-boundary",6), ("hr-mine5-selective-sharpener",5),
 ("mnemosyne-benchmark",13), ("waystations-paper",6),
)
FALSEREVIEW.add(("ghost-dimensions",2))

# ---------------------------------------------------------------- classify
rows = []
for paper in sorted(D):
    for f in D[paper]["findings"]:
        k = (paper, f["n"])
        cls = None
        if k in DUALUSE: cls = "F1"
        elif k in FALSEREVIEW: cls = "F3"
        elif k not in AUTH_DENY and AUTH_RE.search(f["text"]): cls = "F2"
        elif k in MISSING: cls = "F4"
        tier = tier_of(paper, f)
        if cls is None and tier == "F4": cls = "F4"
        bucket = cls or tier
        if bucket == "F4": bucket = "F4"
        closed = CLOSED.get(k)
        if k in DW_SWEPT:
            closed = ("CLOSED 2026-09-03 (placement only) — Dwayne Wilkes moved to "
                      "Acknowledgments by the corpus-wide byline sweep. Sign-off open.")
        rows.append(dict(paper=paper, n=f["n"], sev=f["sev"], text=f["text"],
                         bucket="CLOSED" if closed else bucket, closed=closed,
                         orig=bucket))

c = collections.Counter(r["bucket"] for r in rows)
print("bucket distribution:", dict(c), " total", len(rows))
print("severity:", dict(collections.Counter(r["sev"] for r in rows)))
json.dump(rows, open(os.path.join(HERE, "tiered.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
