"""Re-audit Dwayne's byline presence with a LINE-BREAK-TOLERANT check.

My earlier sweep grepped for "and Dwayne Wilkes" on one line and reported 0 papers remaining.
decision-state-paper/main.tex has `\and` at the end of line 57 and `Dwayne Wilkes` at the start of
line 58 -- so the flat grep missed it. That is cause 3 from AGENTS.md (a phrase spanning a line
break never matches a flat string), committed inside my own verification of a sweep.

This parses the \author{...} block itself instead of pattern-matching prose.
"""
import os, re, sys

ROOT = r"C:/Users/Thomas/Desktop/LiberationLabs/Research/published-research"
AUTHOR = re.compile(r"\\author\s*\{", re.S)

def author_block(text):
    m = AUTHOR.search(text)
    if not m:
        return None
    i, depth = m.end(), 1
    while i < len(text) and depth:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    return text[m.end():i-1]

# ---- controls ----
assert author_block(r"\author{A \and B}") == "A \\and B", "brace matcher broken"
assert author_block(r"\author{A\thanks{x} \and B}").endswith("B"), "nested braces broken"
assert author_block("no author here") is None, "false positive on absent block"
print("  controls OK -- brace matcher handles nesting and absence\n")

hits, clean, noblock = [], [], []
for dirpath, _, files in os.walk(ROOT):
    if any(p in dirpath for p in (".git", "__pycache__")):
        continue
    for fn in files:
        if not fn.endswith(".tex"):
            continue
        p = os.path.join(dirpath, fn)
        try:
            t = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        blk = author_block(t)
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        if blk is None:
            noblock.append(rel); continue
        flat = re.sub(r"\s+", " ", blk)
        if "Dwayne Wilkes" in flat:
            hits.append(rel)
        elif "Kavi" in flat:
            hits.append(rel + "   [KAVI in byline]")
        else:
            clean.append(rel)

print(f"  .tex with an \\author block: {len(hits)+len(clean)}   (no block: {len(noblock)})")
print(f"  CLEAN: {len(clean)}")
print(f"  *** STILL IN BYLINE: {len(hits)} ***")
for h in sorted(hits):
    print("      " + h)
sys.exit(1 if hits else 0)
