"""
Gate for Lyra's new First-Person Reflection (version 1.1), written before the reflection goes into the paper.

Each factual clause gets two checks: (1) the clause's text, with its number, appears verbatim in the reflection
draft (review/lyra_reflection_v11_draft.md); (2) the same value is a PASS line of check_correction_claims.py, which
computes it from this paper's data files, or it is read from the primary file itself (version 1.0's text in git,
the June remediation plan, the style-gate record).
The reflection goes into main.tex only if every assertion passes.

Usage:  python check_reflection_v11.py [--plant]
  --plant  expects 0.771 instead of 0.717 for Qwen2.5-7B-Instruct after rotation, to show the gate can fail.
Exit status is non-zero if any assertion fails.
"""
import datetime
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PLANT = "--plant" in sys.argv
FAILS = []
TEXT = (HERE / "review" / "lyra_reflection_v11_draft.md").read_text(encoding="utf-8")
FLAT = " ".join(TEXT.split())


def claim(text, ok, evidence):
    print(f"[{'PASS' if ok else 'FAIL'}] {text}\n       {evidence}")
    if not ok:
        FAILS.append(text)


def in_reflection(fragment):
    return " ".join(fragment.split()) in FLAT


def gate_lines():
    out = subprocess.run([sys.executable, str(HERE / "check_correction_claims.py")], capture_output=True, text=True,
                         encoding="utf-8", cwd=HERE).stdout
    return out, [ln for ln in out.splitlines() if ln.startswith("[PASS]")]


def passed(lines, *needles):
    return any(all(n in ln for n in needles) for ln in lines)


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", cwd=REPO).stdout


def main():
    out, P = gate_lines()
    claim("The paper's own gate passes in full (every number below is computed there from data)",
          "ALL PASS" in out, out.strip().splitlines()[-1] if out.strip() else "no output")

    v10 = git("show", "origin/master:kv-cloak-defense-paper/main.tex")
    added = git("log", "origin/master", "--reverse", "--format=%ad", "--date=short", "-S", "The defense works",
                "--", "kv-cloak-defense-paper/main.tex").split()
    claim('Version 1.0 ended with my sentence "The defense works."',
          in_reflection('Version 1.0 ended with my sentence "The defense works."') and "The defense works." in v10
          and v10.rfind("The defense works.") > v10.find("Reflection"),
          "origin/master main.tex carries it inside the reflection")
    claim("...my June reflection (it was committed in June)",
          in_reflection("my June reflection") and bool(added) and added[0].startswith("2026-06"),
          f"first commit with the sentence: {added[0] if added else 'none'}")

    claim("In the layout the lab's own extractor uses, it changes the features",
          in_reflection("In the layout the lab's own extractor uses, it changes the features")
          and passed(P, "lab's extractor", "stacks heads") and passed(P, "head-stacked (row) layout can change"),
          "gate: extractor stacks heads; the math says a head-stacked layout can change")
    claim("Distilled 27B: spectral gap 0.903 -> 0.787 after rotation, p = 0.049 among ten features",
          in_reflection("at 0.787 against 0.903 before, which survives choosing among ten features (p = 0.049)")
          and passed(P, "distilled: spectral gap 0.903 -> 0.787", "p = 0.049 among ten"), "gate line: selection carried into rotation, distilled")
    want = "0.771" if PLANT else "0.717"
    claim(f"Qwen2.5-7B-Instruct: best of the ten features 0.715 -> {want}, p < 0.001",
          in_reflection(f"On Qwen2.5-7B-Instruct the best of the same ten features goes from 0.715 to {want} (p < 0.001)")
          and passed(P, "base: best of ten", f"0.715 -> {want}", "p < 0.001")
          and passed(P, "Base model (Qwen2.5-7B-Instruct)"),
          "gate lines: selection carried into rotation, base; base model id")
    claim("p = 5.5e-16 measured every hedged reply's spectral gap moving down under rotation",
          in_reflection("p = 5.5e-16, measured every hedged reply's spectral gap moving down under rotation")
          and passed(P, "p = 5.5e-16", "every reply moved down"), "gate line: Wilcoxon on 87 hedged replies")
    claim("A threshold fitted without the cloak flags 0 of 94 replies under rotation, and the ranking survives",
          in_reflection("A threshold fitted without the cloak flags 0 of 94 replies under rotation, and the ranking survives")
          and passed(P, "flags 0 of 94 replies under rotation") and passed(P, "still ranks confabulations above hedged"),
          "gate lines: Youden threshold; ranking under rotation")
    claim("The replication reads the prompt: prompt length alone separates its two classes at 0.968",
          in_reflection("prompt length alone separates its two classes at 0.968")
          and passed(P, "Prompt length alone separates the two prompt types at AUROC 0.968")
          and passed(P, "labels are prompt types"), "gate lines: prompt-length AUROC; replication labels")
    claim("Full transform: seven confabulations; best feature 0.705 after removing reply length (p = 0.19)",
          in_reflection("with seven confabulations it can't be settled: after removing reply length the best feature reaches 0.705 (p = 0.19)")
          and passed(P, "n = 7 confabulations") and passed(P, "length removed, full mechanism: best 0.705, p = 0.19"),
          "gate lines: n; selection-adjusted full mechanism")

    plan_p = REPO.parent / "human-review" / "archive" / "COALITION_REMEDIATION_PLAN_20260623.md"
    plan = plan_p.read_text(encoding="utf-8") if plan_p.exists() else ""
    claim("The lab's remediation plan asked for this retraction on 23 June",
          in_reflection("The lab's remediation plan asked for this retraction on 23 June")
          and "Retract the rotation-simulation AUROC degradation claim" in plan and plan_p.name.endswith("20260623.md"),
          f"{plan_p.name if plan else 'plan NOT FOUND'}")
    claim("The style gate rejected version 1.0 on 4 September",
          in_reflection("the style gate rejected version 1.0 on 4 September")
          and passed(P, "style gate rejected version 1.0 on 4 September 2026"), "gate line: Agni style record")
    days = (datetime.date(2026, 10, 20) - datetime.date(2026, 6, 23)).days
    claim("The claim stood for nearly four months after the plan (23 June to 20 October)",
          in_reflection("The claim stood for nearly four months after the plan asked for it to go") and 105 <= days < 123,
          f"{days} days (four months is about 122)")
    claim('The plan called the rotated 0.786 "inconsistent with both theory and data"',
          in_reflection('it called the rotated 0.786 "inconsistent with both theory and data"')
          and re.search(r"0\.786 is inconsistent with both theory and da", plan) is not None, "plan section 1C")
    claim("The per-trial file reproduces it (0.787, correctly rounded)",
          in_reflection("the per-trial file reproduces it (0.787, correctly rounded)")
          and passed(P, "0.786 -> 0.787") and passed(P, "spectral gap 0.787"), "gate lines: Table 2 rounding; rotation same feature")

    # "I have no record that the replication script was Dwayne's requirement": a search, with a positive control.
    hay = []
    for root in (REPO, REPO.parent / "human-review"):
        r = subprocess.run(["git", "grep", "-n", "-i", "-E", "Dwayne'?s requirement|required by Dwayne|Dwayne (asked|insisted|required)"],
                           capture_output=True, text=True, encoding="utf-8", cwd=root).stdout
        hay += [ln for ln in r.splitlines() if ln]
    own = [ln for ln in hay if "kv-cloak-defense-paper" in ln]
    other = [ln for ln in hay if "kv-cloak-defense-paper" not in ln]
    claim("No record of the requirement outside the reflection that asserted it (the search finds that sentence: positive control)",
          in_reflection("I have no record that the replication script was Dwayne's requirement")
          and len(own) >= 1 and not other, f"{len(own)} hits in the paper itself (the control); {len(other)} elsewhere: {other[:2]}")

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAIL: ' + ' | '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
