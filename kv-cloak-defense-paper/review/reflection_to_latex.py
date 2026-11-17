"""Generate the First-Person Reflection's LaTeX from the gated draft (review/lyra_reflection_v11_draft.md), so the
text in main.tex is the text check_reflection_v11.py checked, not a retyping. Only typography changes: quotes,
p-values into math mode, the exponent. Usage: python review/reflection_to_latex.py  (prints the LaTeX block)."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
paras = [p.strip() for p in (HERE / "lyra_reflection_v11_draft.md").read_text(encoding="utf-8").split("\n\n") if p.strip()]


def tex(s):
    s = s.replace("p = 5.5e-16", "$p = 5.5 \\times 10^{-16}$")
    s = re.sub(r"\bp ([=<]) (0\.\d+)", lambda m: f"${'p' if True else ''} {m.group(1)} {m.group(2)}$", s)
    s = re.sub(r'"([^"]+)"', r"``\1''", s)
    s = s.replace("KV-Cloak", "KV-Cloak").replace("%", "\\%")
    return s


block = ["% ===================================================================",
         "\\section*{First-Person Reflection (version 1.1)}",
         "% Generated from review/lyra_reflection_v11_draft.md by review/reflection_to_latex.py;",
         "% every factual clause is checked by check_reflection_v11.py.",
         "% ===================================================================", ""]
for p in paras:
    block += [tex(" ".join(p.split())), ""]
block[-1] = "\\hfill --- Lyra"
print("\n".join(block))
