# Parse Agni STYLE gate findings out of *.agni_style.json  (v2)
import json, glob, re, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

STYLE_DIR = r"C:\Users\Thomas\Desktop\LiberationLabs\Research\published-research\tools\agni\style"
OUT = os.path.dirname(os.path.abspath(__file__))
SEV = r"(CRITICAL|MAJOR|MINOR)"

# a pure severity section header:  "## CRITICAL", "### MINOR", "### MINOR ISSUES", "**MINOR**"
re_sec = re.compile(r"^\s*(?:#{1,5}\s*|\*\*)" + SEV + r"\b[ \w\*\(\)/\u2014\-]{0,30}$", re.I)

# a stop header: leaves the findings region
re_stop = re.compile(
    r"^\s*(?:#{1,5}\s*)?(?:\*\*)?(?:\d+[\.\)]\s*)?"
    r"(WHAT I COULD NOT|WHAT I CANNOT|WHAT (?:I|WE) (?:COULD|CAN)\s*NOT|COULD NOT CHECK|"
    r"WHAT (?:IS|WAS) CLEAN|WHAT I VERIFIED|VERIFIED CLEAN|WHAT CHECKS OUT|WHAT SURVIVES|"
    r"VERDICT|SUMMARY|RECOMMEND|APPENDIX|UNVERIFIED|NOT CHECKED|WHAT IS GOOD|WHAT WORKS|"
    r"STYLE NOTES|CLOSING|BOTTOM LINE|OVERALL)\b", re.I)

# numbered finding carrying its own severity:  "### 3. MAJOR - ...", "**3. MAJOR - ...", "## 1. CRITICAL"
re_num_sev = re.compile(
    r"^\s*(?:#{1,5}\s*)?(?:\*\*)?(\d+)[\.\)](?:\*\*)?\s*" + SEV +
    r"\b\s*(?:/\s*\*{0,2}[A-Z_]+\*{0,2})?\s*[\u2014\u2013:\-]*\s*(.*)$")

# numbered finding, severity inherited from the enclosing section
re_num = re.compile(r"^\s*(?:#{1,5}\s*)?(?:\*\*)?(\d+)[\.\)](?:\*\*)?\s+(\S.*)$")


def clean(t):
    t = re.sub(r"\*\*|__", "", t.strip())
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .*_#\u2014-")


def parse(review):
    out, cur = [], None
    for raw in review.split("\n"):
        ln = raw.rstrip()
        if not ln.strip():
            continue
        if re_stop.match(ln):
            cur = None
            continue
        m = re_sec.match(ln)
        if m:
            cur = m.group(1).upper()
            continue
        m = re_num_sev.match(ln)
        if m:
            out.append((int(m.group(1)), m.group(2).upper(), clean(m.group(3)) or clean(ln)))
            cur = m.group(2).upper()
            continue
        m = re_num.match(ln)
        if m and cur and len(clean(m.group(2))) > 10:
            out.append((int(m.group(1)), cur, clean(m.group(2))))
            continue
    # unique on finding number (keep first)
    seen, res = set(), []
    for n, s, t in out:
        if n in seen:
            continue
        seen.add(n)
        res.append((n, s, t))
    return res


rows = {}
for path in sorted(glob.glob(os.path.join(STYLE_DIR, "*.agni_style.json"))):
    name = os.path.basename(path).replace(".agni_style.json", "")
    d = json.load(open(path, encoding="utf-8"))
    rows[name] = {"verdict": d.get("verdict"), "primary": d.get("primary"),
                  "findings": parse(d.get("review") or "")}

tot = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}
print("%-38s %-12s %4s %4s %4s %5s %6s %s" % ("PAPER", "VERDICT", "C", "MAJ", "MIN", "TOT", "MAXNUM", "CONTIG"))
bad, gappy = [], []
for n, r in rows.items():
    fs = r["findings"]
    c = sum(1 for f in fs if f[1] == "CRITICAL")
    a = sum(1 for f in fs if f[1] == "MAJOR")
    m = sum(1 for f in fs if f[1] == "MINOR")
    tot["CRITICAL"] += c; tot["MAJOR"] += a; tot["MINOR"] += m
    nums = sorted(f[0] for f in fs)
    mx = max(nums) if nums else 0
    contig = (nums == list(range(1, mx + 1))) if nums else False
    print("%-38s %-12s %4d %4d %4d %5d %6d %s" % (n, r["verdict"], c, a, m, len(fs), mx,
                                                  "ok" if contig else "GAP:" + str(sorted(set(range(1, mx + 1)) - set(nums)))))
    if not fs and r["verdict"] in ("REJECTED", "CONDITIONAL"):
        bad.append(n)
    if fs and not contig:
        gappy.append(n)
print()
print("TOTAL  CRITICAL=%d  MAJOR=%d  MINOR=%d  ALL=%d" % (tot["CRITICAL"], tot["MAJOR"], tot["MINOR"], sum(tot.values())))
print("\nZERO-FINDING non-approved files (would be a BROKEN PARSER):", bad or "none")
print("NON-CONTIGUOUS numbering (possible missed finding):", gappy or "none")

json.dump({n: {"verdict": r["verdict"], "primary": r["primary"],
               "findings": [{"n": a, "sev": b, "text": c} for a, b, c in r["findings"]]}
           for n, r in rows.items()},
          open(os.path.join(OUT, "findings.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
